import assert from "node:assert/strict";
import { test } from "node:test";
import { readEvents, requireSuccess } from "../app/client.ts";

test("reads split JSON lines and UTF-8 characters from the stream", async () => {
  const input = new TextEncoder().encode(
    '{"type":"delta","text":"µ","generated_tokens":2,"tokens_per_second":3}\n' +
    '{"type":"done","generated_tokens":2,"tokens_per_second":3,"cancelled":false}'
  );
  const stream = new ReadableStream({
    start(controller) {
      for (const byte of input) controller.enqueue(Uint8Array.of(byte));
      controller.close();
    },
  });
  const events = [];
  for await (const event of readEvents(new Response(stream))) events.push(event);
  assert.deepEqual(events.map((event) => event.type), ["delta", "done"]);
  assert.equal(events[0].text, "µ");
  assert.equal(events[1].generated_tokens, 2);
});

test("uses the service's safe error detail", async () => {
  await assert.rejects(requireSuccess(Response.json({ detail: "Model unavailable" }, { status: 404 })),
    /Model unavailable/);
});
