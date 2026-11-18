"use client";

import { useEffect, useRef, useState, type FormEvent, type KeyboardEvent } from "react";
import { AlertCircle, ArrowUp, ChevronDown, Cpu, Menu, SlidersHorizontal, Square } from "lucide-react";
import { API, readEvents, requireSuccess, type ModelList } from "./client";
import { ConversationRail, EmptyWorkspace, MessageList, type Conversation } from "./workspace-ui";

type Theme = "dark" | "light";
type Run = { id: string; controller: AbortController };

const STORAGE = "goxlm.workspace.v1";
const integer = (value: number, max: number) => Number.isInteger(value) && value >= 1 && value <= max;

export default function Home() {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [theme, setTheme] = useState<Theme>("dark");
  const [loaded, setLoaded] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);
  const [service, setService] = useState<ModelList | null>(null);
  const [connecting, setConnecting] = useState(true);
  const [modelId, setModelId] = useState("");
  const [draft, setDraft] = useState("");
  const [temperature, setTemperature] = useState("0.8");
  const [topK, setTopK] = useState("40");
  const [maxTokens, setMaxTokens] = useState("64");
  const [showSettings, setShowSettings] = useState(false);
  const [busy, setBusy] = useState(false);
  const [phase, setPhase] = useState("Idle");
  const [generated, setGenerated] = useState(0);
  const [rate, setRate] = useState(0);
  const [error, setError] = useState("");
  const current = useRef<Run | null>(null);
  const composer = useRef<HTMLTextAreaElement>(null);
  const scrollEnd = useRef<HTMLDivElement>(null);

  async function loadModels(signal?: AbortSignal) {
    setConnecting(true);
    try {
      const response = await requireSuccess(await fetch(`${API}/models`, { cache: "no-store", signal }));
      const data: ModelList = await response.json();
      if (signal?.aborted) return;
      setService(data);
      setModelId((id) => data.models.some((m) => m.id === id) ? id : (data.models[0]?.id ?? ""));
      setError("");
    } catch {
      if (!signal?.aborted) {
        setService(null);
        setError("Cannot reach the local inference service. Start the WSL backend, then retry.");
      }
    } finally {
      if (!signal?.aborted) setConnecting(false);
    }
  }

  useEffect(() => {
    try {
      const raw = window.localStorage.getItem(STORAGE);
      if (raw) {
        const saved = JSON.parse(raw);
        if (Array.isArray(saved.conversations)) setConversations(saved.conversations);
        if (typeof saved.selected === "string") setSelected(saved.selected);
        if (saved.theme === "light") setTheme("light");
      }
    } catch {
      // The workspace still works if this browser blocks storage.
    }
    setLoaded(true);
    const controller = new AbortController();
    void loadModels(controller.signal);
    return () => {
      controller.abort();
      current.current?.controller.abort();
    };
  }, []);

  useEffect(() => {
    if (!loaded) return;
    try {
      window.localStorage.setItem(STORAGE, JSON.stringify({ conversations, selected, theme }));
    } catch {
      // Session state remains available in memory.
    }
  }, [loaded, conversations, selected, theme]);

  const active = conversations.find((item) => item.id === selected);
  const model = service?.models.find((item) => item.id === modelId);
  const promptTokens = draft.length ? new TextEncoder().encode(draft).length + (model?.prefix_tokens ?? 0) : 0;
  const context = model?.context_length ?? 128;
  const fits = promptTokens <= context;
  const temp = Number(temperature);
  const k = Number(topK);
  const max = Number(maxTokens);
  const settingsValid = temp >= 0.05 && temp <= 2 && integer(k, model?.vocab_size ?? 256)
    && integer(max, model?.max_new_tokens ?? 256);
  const canSubmit = Boolean(model && !busy && draft.trim() && fits && settingsValid);

  useEffect(() => {
    scrollEnd.current?.scrollIntoView({ block: "nearest" });
  }, [active?.messages, busy]);

  function newConversation() {
    if (current.current) return;
    const id = crypto.randomUUID();
    setConversations((items) => [{ id, title: "New conversation", messages: [] }, ...items]);
    selectConversation(id);
  }

  function selectConversation(id: string) {
    setSelected(id);
    setDraft("");
    setGenerated(0);
    setRate(0);
    setPhase("Idle");
    setMobileOpen(false);
    if (service) setError("");
    composer.current?.focus();
  }

  function usePrompt(prompt: string) {
    setDraft(prompt);
    composer.current?.focus();
  }

  async function send(e?: FormEvent) {
    e?.preventDefault();
    if (!canSubmit || !model || current.current) return;
    const conversationId = active?.id ?? crypto.randomUUID();
    const assistantId = crypto.randomUUID();
    const input = draft;
    const run = { id: crypto.randomUUID(), controller: new AbortController() };
    current.current = run;

    setConversations((items) => {
      const previous = items.find((item) => item.id === conversationId);
      const firstMessage = !previous?.messages.length;
      const trimmed = input.trim().replace(/\s+/g, " ");
      const title = firstMessage ? trimmed.slice(0, 38) + (trimmed.length > 38 ? "…" : "") : previous.title;
      return [{ id: conversationId, title, messages: [
        ...(previous?.messages ?? []),
        { id: crypto.randomUUID(), role: "user", text: input },
        { id: assistantId, role: "assistant", text: "", model: model.label },
      ] }, ...items.filter((item) => item.id !== conversationId)];
    });
    setSelected(conversationId);
    setBusy(true);
    setGenerated(0);
    setRate(0);
    setPhase("Starting");
    setDraft("");
    setError("");

    try {
      const stream = await requireSuccess(await fetch(`${API}/generate`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ generation_id: run.id, model_id: model.id, prompt: input,
          temperature: temp, top_k: k, max_new_tokens: max }),
        signal: run.controller.signal,
      }));
      let finished = false;
      for await (const event of readEvents(stream)) {
        if (event.type === "error") throw new Error(event.message);
        if (event.type === "start") setPhase("Generating");
        if (event.type === "delta" || event.type === "done") {
          setGenerated(event.generated_tokens);
          setRate(event.tokens_per_second);
        }
        if (event.type === "delta" && event.text) {
          setConversations((items) => items.map((item) => item.id !== conversationId ? item : ({
            ...item, messages: item.messages.map((message) => message.id !== assistantId ? message :
              { ...message, text: message.text + event.text }),
          })));
        }
        if (event.type === "done") {
          setPhase(event.cancelled ? "Stopped" : "Complete");
          finished = true;
        }
      }
      if (!finished) throw new Error("The generation stream ended early. Retry the prompt.");
    } catch (err) {
      if (run.controller.signal.aborted) setPhase("Stopped");
      else {
        setPhase("Failed");
        setError(err instanceof Error && err.message !== "Failed to fetch"
          ? err.message : "The local service disconnected. Check the WSL backend and retry.");
      }
    } finally {
      if (current.current === run) {
        current.current = null;
        setBusy(false);
      }
      composer.current?.focus();
    }
  }

  async function stop() {
    const run = current.current;
    if (!run) return;
    setPhase("Stopping");
    try {
      await requireSuccess(await fetch(`${API}/generations/${run.id}/cancel`, { method: "POST" }));
    } catch {
      run.controller.abort();
    }
  }

  function onComposerKey(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      void send();
    }
  }

  return (
    <main data-theme={theme} className="workspace flex h-dvh min-h-[370px] overflow-hidden">
      <ConversationRail conversations={conversations} selected={selected} busy={busy} open={mobileOpen}
        connected={!!service} device={service?.device.label} theme={theme}
        onClose={() => setMobileOpen(false)} onNew={newConversation} onSelect={selectConversation}
        onTheme={() => setTheme(theme === "dark" ? "light" : "dark")} />
      <section className="flex min-w-0 flex-1 flex-col">
        <header className="flex min-h-[70px] shrink-0 flex-wrap items-center justify-between gap-3 border-b border-[var(--line)] px-4 py-3 sm:px-7">
          <div className="flex items-center gap-3">
            <button type="button" aria-label="Open conversations" onClick={() => setMobileOpen(true)}
              className="rounded-lg border border-[var(--line)] p-2.5 text-[var(--muted)] md:hidden"><Menu size={16} /></button>
            <div>
              <div className="text-sm font-semibold tracking-tight">Text completion</div>
              <div className="mt-1 flex items-center gap-1.5 text-xs text-[var(--muted)]">
                <span className="h-1.5 w-1.5 rounded-full bg-[var(--accent)]" /> Experimental
              </div>
            </div>
          </div>
          <div className="flex items-center gap-2 sm:gap-3">
            {model && <span className="hidden items-center gap-1.5 text-xs text-[var(--muted)] lg:flex">
              <Cpu size={14} /> {(model.parameter_count / 1_000_000).toFixed(2)}M params
              <span className="px-1 text-[var(--quiet)]">·</span> {model.context_length} context
            </span>}
            <label className="relative">
              <span className="sr-only">Model</span>
              <select value={modelId} onChange={(e) => setModelId(e.target.value)} disabled={busy || !service}
                className="max-w-[190px] appearance-none rounded-lg border border-[var(--line)] bg-[var(--panel)] py-2 pl-3 pr-8 text-sm font-medium">
                {!service && <option value="">{connecting ? "Connecting…" : "Model offline"}</option>}
                {service?.models.map((m) => <option key={m.id} value={m.id}>{m.label}</option>)}
              </select>
              <ChevronDown size={14} className="pointer-events-none absolute right-2.5 top-2.5 text-[var(--quiet)]" />
              {model && <span className="mt-0.5 block text-right text-[10px] text-[var(--muted)] lg:hidden">
                {(model.parameter_count / 1_000_000).toFixed(2)}M params
              </span>}
            </label>
          </div>
        </header>

        <div className="flex min-h-0 flex-1 flex-col overflow-y-auto overscroll-contain">
          {active?.messages.length
            ? <MessageList messages={active.messages} busy={busy} />
            : <EmptyWorkspace onPrompt={usePrompt} />}
          <div ref={scrollEnd} />
        </div>

        <footer className="mx-auto w-full max-w-[860px] shrink-0 px-4 pb-4 pt-3 sm:px-8 sm:pb-6">
          {error && <div role="alert" className="mb-3 flex items-center gap-2.5 rounded-xl bg-[var(--error-bg)] px-3 py-2.5 text-sm text-[var(--error)]">
            <AlertCircle size={16} className="shrink-0" /><span className="flex-1">{error}</span>
            {!service && <button type="button" onClick={() => void loadModels()} className="shrink-0 font-semibold underline">Retry</button>}
          </div>}
          {showSettings && <fieldset className="mb-3 grid grid-cols-3 gap-2 rounded-xl border border-[var(--line)] bg-[var(--panel)] px-3 py-2.5 sm:gap-4">
            <legend className="sr-only">Generation settings</legend>
            {[
              { label: "Temperature", value: temperature, set: setTemperature, min: 0.05, max: 2, step: 0.05 },
              { label: "Top-k", value: topK, set: setTopK, min: 1, max: model?.vocab_size ?? 256, step: 1 },
              { label: "Max new tokens", value: maxTokens, set: setMaxTokens, min: 1, max: model?.max_new_tokens ?? 256, step: 1 },
            ].map(({ label, value, set, min, max, step }) => <label key={label} className="text-xs text-[var(--muted)]">
              {label}<input type="number" value={value} onChange={(e) => set(e.target.value)} min={min} max={max} step={step}
                disabled={busy} className="mt-1 block w-full rounded-md border border-[var(--line)] bg-[var(--canvas)] px-2 py-1.5 text-[var(--text)]" />
            </label>)}
            {!settingsValid && <p className="col-span-3 text-xs text-[var(--error)]">Enter values within the displayed ranges.</p>}
          </fieldset>}
          <form onSubmit={(e) => void send(e)} className="composer rounded-2xl border border-[var(--line)] bg-[var(--panel)] p-3 shadow-[0_12px_32px_#05090d22] sm:p-3.5">
            <label htmlFor="prompt" className="sr-only">Text to complete</label>
            <textarea id="prompt" ref={composer} value={draft} onChange={(e) => setDraft(e.target.value)}
              onKeyDown={onComposerKey} maxLength={256} rows={2} disabled={busy}
              placeholder="Enter text for GoXLM to continue…"
              className="block min-h-[52px] w-full resize-none bg-transparent px-1 py-1 text-[15px] leading-6 placeholder:text-[var(--quiet)] disabled:opacity-60" />
            <div className="mt-1 flex items-center gap-2">
              <button type="button" aria-expanded={showSettings} title="Generation settings" aria-label="Generation settings"
                onClick={() => setShowSettings(!showSettings)}
                className="flex h-8 items-center gap-1.5 rounded-lg px-2 text-xs font-medium text-[var(--muted)] hover:bg-[var(--panel-hover)]">
                <SlidersHorizontal size={15} /><span className="hidden sm:inline">Settings</span>
              </button>
              <span className={`ml-auto text-xs tabular-nums ${fits ? "text-[var(--muted)]" : "text-[var(--error)]"}`}
                title="The selected model counts UTF-8 bytes as tokens">{promptTokens} / {context} prompt tokens</span>
              {busy
                ? <button type="button" onClick={() => void stop()} disabled={phase === "Stopping"}
                    className="flex h-9 items-center gap-1.5 rounded-lg bg-[var(--accent)] px-3 text-xs font-semibold text-[var(--accent-ink)] disabled:opacity-50">
                    <Square size={13} fill="currentColor" /> Stop
                  </button>
                : <button type="submit" aria-label="Generate completion" title="Generate completion" disabled={!canSubmit}
                    className="grid h-9 w-9 place-items-center rounded-lg bg-[var(--accent)] text-[var(--accent-ink)] disabled:opacity-45">
                    <ArrowUp size={19} strokeWidth={2.3} />
                  </button>}
            </div>
          </form>
          {!fits && <p role="alert" className="mt-2 text-xs text-[var(--error)]">Shorten the prompt to stay within the model context.</p>}
          <div className="mt-3 flex flex-wrap items-center justify-between gap-2 px-1 text-xs text-[var(--muted)]">
            <span>Only the current input is sent. History stays in this browser.</span>
            <span aria-live="polite" className="tabular-nums">{phase}{generated > 0 && ` · ${generated} tokens · ${rate.toFixed(1)} tok/s`}</span>
          </div>
          {service?.device.fallback && <p className="mt-1.5 px-1 text-xs text-[var(--muted)]">CUDA is unavailable. The service is using CPU.</p>}
        </footer>
      </section>
    </main>
  );
}
