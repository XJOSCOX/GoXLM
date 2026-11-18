"""Loopback API and a single cancellable inference slot."""
from dataclasses import dataclass, field
import json
from threading import Event, Lock
import time
from typing import Annotated
from uuid import UUID

from anyio import CancelScope
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, UUID4
from starlette.concurrency import run_in_threadpool
from starlette.middleware.trustedhost import TrustedHostMiddleware
import torch

from .engine import (CheckpointError, Completion, MAX_NEW_TOKENS, ModelId, ModelRegistry,
                     model_info, prompt_tokens, resolve_device)

ORIGINS = ("http://localhost:13000", "http://127.0.0.1:13000")


class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    generation_id: UUID4
    model_id: ModelId
    prompt: Annotated[str, Field(strict=True, min_length=1, max_length=256)]
    temperature: Annotated[float, Field(strict=True, ge=0.05, le=2.0)] = 0.8
    top_k: Annotated[int, Field(strict=True, ge=1, le=2048)] = 40
    max_new_tokens: Annotated[int, Field(strict=True, ge=1, le=MAX_NEW_TOKENS)] = 64


@dataclass
class Run:
    id: UUID
    cancelled: Event = field(default_factory=Event)


class Slot:
    def __init__(self):
        self.lock = Lock()
        self.active = None

    def reserve(self, generation_id):
        with self.lock:
            if self.active is None:
                self.active = Run(generation_id)
                return self.active
            return None

    def cancel(self, generation_id):
        with self.lock:
            if self.active is None or self.active.id != generation_id:
                return False
            self.active.cancelled.set()
            return True

    def release(self, run):
        with self.lock:
            if self.active is run:
                run.cancelled.set()
                self.active = None


class GenerationResponse(StreamingResponse):
    def __init__(self, body, slot, run):
        super().__init__(body, media_type="application/x-ndjson",
                         headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})
        self.slot = slot
        self.run = run

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            # StreamingResponse cancels its iterator on disconnect. AnyIO's worker
            # waits for the in-flight model operation before the slot is released.
            try:
                with CancelScope(shield=True):
                    await self.body_iterator.aclose()
            finally:
                self.slot.release(self.run)


def event(kind, **data):
    return json.dumps({"type": kind, **data}, ensure_ascii=False) + "\n"


def error(status, message):
    return JSONResponse(status_code=status, content={"detail": message},
                        headers={"Cache-Control": "no-store"})


def create_app(settings, *, registry_factory=ModelRegistry):
    device = resolve_device(settings.device)
    registry = registry_factory(settings, device)
    slot = Slot()
    app = FastAPI(title="GoXLM Local", docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1"])
    app.add_middleware(CORSMiddleware, allow_origins=list(ORIGINS), allow_methods=["GET", "POST"],
                       allow_headers=["Content-Type"])

    @app.middleware("http")
    async def local_request(request, call_next):
        origin = request.headers.get("origin")
        if origin is not None and origin not in ORIGINS:
            return error(403, "This browser origin is outside the local workspace.")
        if request.method == "POST" and request.url.path == "/api/generate":
            if request.headers.get("content-type", "").split(";", 1)[0].strip() != "application/json":
                return error(415, "Send an application/json request.")
            length = request.headers.get("content-length", "")
            if not length.isascii() or not length.isdecimal():
                return error(411, "A request content length is required.")
            if len(length) > 4 or int(length) > 4096:
                return error(413, "The generation request is too large.")
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(RequestValidationError)
    async def invalid_request(_request, _exc):
        return error(422, "Invalid request. Check the model ID, prompt, and generation settings.")

    @app.get("/api/models")
    async def models():
        return {"models": [model_info(entry) for entry in registry.entries.values()],
                "device": {"type": device.torch_device.type, "label": device.label,
                           "fallback": device.fallback}}

    @app.post("/api/generations/{generation_id}/cancel")
    async def cancel(generation_id: UUID4):
        if not slot.cancel(generation_id):
            return error(404, "No active generation matches this ID.")
        return {"cancelled": True}

    @app.post("/api/generate", response_model=None)
    async def generate(body: GenerateRequest):
        entry = registry.entries.get(body.model_id)
        if entry is None:
            return error(404, "The requested model ID is not in the allowlist.")
        try:
            tokens = prompt_tokens(entry, body.prompt)
        except ValueError as exc:
            return error(422, str(exc))
        if not body.prompt.strip():
            return error(422, "Enter text to complete.")
        if body.top_k > model_info(entry)["vocab_size"]:
            return error(422, "Top-k exceeds the model vocabulary.")
        run = slot.reserve(body.generation_id)
        if run is None:
            return error(409, "Another generation is active. Wait for it to finish or stop it.")

        async def stream():
            yield event("start", generation_id=str(run.id), prompt_tokens=len(tokens))
            count = 0
            try:
                if run.cancelled.is_set():
                    yield event("done", generated_tokens=0, tokens_per_second=0, cancelled=True)
                    return
                model = await run_in_threadpool(registry.get, entry)
                completion = Completion(model, tokens, device.torch_device, body.temperature, body.top_k)
                started = time.perf_counter()
                for count in range(1, body.max_new_tokens + 1):
                    if run.cancelled.is_set():
                        count -= 1
                        break
                    text = await run_in_threadpool(completion.step)
                    rate = count / max(time.perf_counter() - started, 1e-9)
                    yield event("delta", text=text, generated_tokens=count, tokens_per_second=rate)
                rate = count / max(time.perf_counter() - started, 1e-9)
                tail = completion.finish()
                if tail:
                    yield event("delta", text=tail, generated_tokens=count, tokens_per_second=rate)
                yield event("done", generated_tokens=count, tokens_per_second=rate,
                            cancelled=run.cancelled.is_set())
            except CheckpointError as exc:
                yield event("error", message=str(exc))
            except torch.OutOfMemoryError:
                yield event("error", message="The GPU ran out of memory during inference.")
            except Exception:
                yield event("error", message="Local inference failed. Check the service and model settings.")

        return GenerationResponse(stream(), slot, run)

    return app
