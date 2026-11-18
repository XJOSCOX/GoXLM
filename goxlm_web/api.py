"""Loopback API and a single cancellable inference slot."""
from dataclasses import dataclass, field
import json
import math
from threading import Event, Lock
import time
from typing import Annotated, Literal
from uuid import UUID

from anyio import CancelScope, create_task_group
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, UUID4
from starlette.concurrency import run_in_threadpool
from starlette.middleware.trustedhost import TrustedHostMiddleware
import torch

from .engine import (CheckpointError, Completion, GenerationCancelled, MAX_NEW_TOKENS, ModelId, ModelRegistry,
                     model_info, prompt_tokens, resolve_device)

ORIGINS = ("http://localhost:13000", "http://127.0.0.1:13000")


class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    generation_id: UUID4
    model_id: ModelId
    prompt: Annotated[str, Field(strict=True, min_length=1, max_length=256)]
    temperature: Annotated[float, Field(strict=True, ge=0.05, le=2.0)] = 0.8
    top_k: Annotated[int, Field(strict=True, ge=1, le=2048)] = 40
    strategy: Literal["full", "cached"] = "full"
    seed: Annotated[int, Field(strict=True, ge=0, le=2**32 - 1)] | None = None
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
        disconnected = False

        async def monitor():
            nonlocal disconnected
            while True:
                if (await receive())["type"] == "http.disconnect":
                    disconnected = True
                    self.run.cancelled.set()
                    return

        async def output(message):
            if not disconnected:
                await send(message)

        try:
            async with create_task_group() as tasks:
                tasks.start_soon(monitor)
                try:
                    await self.stream_response(output)
                except OSError:
                    self.run.cancelled.set()
                finally:
                    tasks.cancel_scope.cancel()
        finally:
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
        return {"models": [registry.info(entry) for entry in registry.entries.values()],
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

        acquired = time.perf_counter()

        async def stream():
            completion = None
            load_ms = None
            stamps = []
            count = 0
            rate = 0
            yield event("start", generation_id=str(run.id), prompt_tokens=len(tokens),
                        strategy=body.strategy)
            try:
                if run.cancelled.is_set():
                    raise GenerationCancelled
                load_start = time.perf_counter()
                with CancelScope(shield=True):
                    model = await run_in_threadpool(registry.get, entry, run.cancelled)
                load_ms = (time.perf_counter() - load_start) * 1000
                if run.cancelled.is_set():
                    raise GenerationCancelled
                completion = Completion(model, tokens, device.torch_device, body.temperature,
                                        body.top_k, body.strategy, body.seed)
                for _ in range(body.max_new_tokens):
                    if run.cancelled.is_set():
                        break
                    with CancelScope(shield=True):
                        text = await run_in_threadpool(completion.step)
                    stamp = time.perf_counter()
                    count += 1
                    stamps.append(stamp)
                    rate = (count - 1) / max(stamp - stamps[0], 1e-9) if count > 1 else 0
                    yield event("delta", text=text, generated_tokens=count, tokens_per_second=rate)
                tail = completion.finish()
                if tail:
                    yield event("delta", text=tail, generated_tokens=count, tokens_per_second=rate)
            except GenerationCancelled:
                pass
            except CheckpointError as exc:
                yield event("error", message=str(exc))
                return
            except torch.OutOfMemoryError:
                yield event("error", message="The GPU ran out of memory during inference.")
                return
            except Exception:
                yield event("error", message="Local inference failed. Check the service and model settings.")
                return
            finally:
                rebuilds = 0
                if completion is not None:
                    rebuilds = completion.cache.rebuilds if completion.cache is not None else 0
                    completion.close()
            intervals = sorted((b - a) * 1000 for a, b in zip(stamps, stamps[1:]))
            yield event("done", generated_tokens=count, cancelled=run.cancelled.is_set(),
                        tokens_per_second=(count - 1) / max(stamps[-1] - stamps[0], 1e-9)
                        if intervals else 0,
                        first_token_ms=(stamps[0] - acquired) * 1000 if stamps else None,
                        load_ms=load_ms, p95_token_ms=intervals[math.ceil(.95 * len(intervals)) - 1]
                        if intervals else None, cache_rebuilds=rebuilds)

        return GenerationResponse(stream(), slot, run)

    return app
