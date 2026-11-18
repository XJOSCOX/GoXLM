export const API = "http://127.0.0.1:18000/api";

export type Model = {
  id: string;
  label: string;
  parameter_count: number;
  context_length: number;
  prefix_tokens: number;
  vocab_size: number;
  max_new_tokens: number;
};
export type ModelList = {
  models: Model[];
  device: { type: "cpu" | "cuda"; label: string; fallback: boolean };
};
export type GenerationEvent =
  | { type: "start"; generation_id: string; prompt_tokens: number }
  | { type: "delta"; text: string; generated_tokens: number; tokens_per_second: number }
  | { type: "done"; generated_tokens: number; tokens_per_second: number; cancelled: boolean }
  | { type: "error"; message: string };

export async function requireSuccess(response: Response) {
  if (response.ok) return response;
  const body = await response.json().catch(() => null);
  throw new Error(body?.detail || `Local service returned HTTP ${response.status}.`);
}

export async function* readEvents(response: Response): AsyncGenerator<GenerationEvent> {
  const reader = response.body?.getReader();
  if (!reader) throw new Error("The inference service returned no stream.");
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    for (;;) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value, { stream: !done });
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";
      for (const line of lines) {
        if (line.trim()) yield JSON.parse(line) as GenerationEvent;
      }
      if (done) {
        if (buffer.trim()) yield JSON.parse(buffer) as GenerationEvent;
        return;
      }
    }
  } finally {
    await reader.cancel().catch(() => {});
    reader.releaseLock();
  }
}
