import { ArrowUpRight, Atom, CodeXml, MessageSquare, Moon, Orbit, Plus, Sigma, Sun, X } from "lucide-react";

export type Message = { id: string; role: "user" | "assistant"; text: string; model?: string; group?: string;
  metrics?: { first: number | null; p95: number | null; rate: number; rebuilds: number };
  stopped?: boolean };
export type Conversation = { id: string; title: string; messages: Message[] };

type RailProps = {
  conversations: Conversation[];
  selected: string | null;
  busy: boolean;
  open: boolean;
  connected: boolean;
  device?: string;
  theme: "dark" | "black";
  onClose: () => void;
  onNew: () => void;
  onSelect: (id: string) => void;
  onTheme: () => void;
};

export function ConversationRail(props: RailProps) {
  const { conversations, selected, busy, open, connected, device, theme, onClose, onNew, onSelect, onTheme } = props;
  const label = device?.replace(/^NVIDIA GeForce /, "") ?? "Service offline";
  return <>
    {open && <button type="button" aria-label="Close conversation sidebar" onClick={onClose}
      className="fixed inset-0 z-10 bg-black/50 md:hidden" />}
    <aside aria-label="Conversations" className={`z-20 flex h-full w-[248px] shrink-0 flex-col border-r border-[var(--line)] bg-[var(--rail)] px-4 py-5
      ${open ? "fixed inset-y-0 left-0" : "hidden"} md:static md:flex`}>
      <div className="mb-7 flex items-center gap-3 px-1">
        <div className="grid h-10 w-10 shrink-0 place-items-center rounded-lg border border-[var(--line)] bg-[var(--accent-surface)] text-[var(--text)]">
          <Orbit size={23} strokeWidth={1.9} />
        </div>
        <div className="min-w-0 flex-1">
          <div className="text-xl font-bold leading-6 tracking-tight">GoXLM</div>
          <div className="mt-0.5 text-xs text-[var(--muted)]">Local workspace</div>
        </div>
        <button type="button" aria-label="Close conversations" onClick={onClose}
          className="grid h-8 w-8 place-items-center rounded-md text-[var(--quiet)] hover:bg-[var(--panel-hover)] md:hidden"><X size={17} /></button>
      </div>

      <button type="button" onClick={onNew} disabled={busy}
        className="flex w-full items-center justify-start gap-2.5 rounded-lg border border-[var(--line)] bg-[var(--accent-surface)] px-3 py-2.5 text-sm font-semibold text-[var(--accent)] transition-colors hover:bg-[var(--panel-hover)] disabled:opacity-50">
        <Plus size={17} /> New conversation
      </button>

      <nav aria-label="Recent conversations" className="mt-7 min-h-0 flex-1 overflow-y-auto">
        <h2 className="mb-2.5 px-2 text-[11px] font-semibold tracking-[0.14em] text-[var(--quiet)] uppercase">Conversations</h2>
        {conversations.length === 0 && <p className="px-2 py-2 text-sm leading-6 text-[var(--muted)]">No conversations yet.</p>}
        {conversations.map(({ id, title }) => (
          <button key={id} type="button" onClick={() => onSelect(id)} disabled={busy}
            aria-current={selected === id ? "true" : undefined}
            className={`mb-1 flex w-full items-center gap-2.5 rounded-lg px-2.5 py-2.5 text-left text-sm
              ${selected === id ? "bg-[var(--panel-hover)] text-[var(--text)]" : "text-[var(--muted)] hover:bg-[var(--panel)]"}`}>
            <MessageSquare size={15} className="shrink-0 text-[var(--quiet)]" />
            <span className="truncate">{title}</span>
          </button>
        ))}
      </nav>

      <div className="space-y-2 border-t border-[var(--line)] pt-3">
        <div className="flex items-center gap-2.5 rounded-lg px-2.5 py-2 text-sm text-[var(--muted)]">
          <span className={`h-2 w-2 shrink-0 rounded-full ${connected ? "bg-[var(--accent)]" : "bg-[var(--quiet)]"}`} />
          <span className="truncate" title={device}>{label}</span>
          {connected && <span className="ml-auto text-xs text-[var(--quiet)]">Local</span>}
        </div>
        <button type="button" onClick={onTheme}
          className="flex w-full items-center gap-2.5 rounded-lg px-2.5 py-2.5 text-left text-sm text-[var(--muted)] hover:bg-[var(--panel-hover)]">
          {theme === "dark" ? <Moon size={16} /> : <Sun size={16} />} {theme === "dark" ? "True black" : "Graphite mode"}
        </button>
      </div>
    </aside>
  </>;
}

const ideas = [
  { label: "Science", text: "A semiconductor conducts electricity when", icon: Atom },
  { label: "Programming", text: "def fibonacci(n):\n    ", icon: CodeXml },
  { label: "Mathematics", text: "The derivative of a quadratic function is", icon: Sigma },
];

export function EmptyWorkspace({ onPrompt }: { onPrompt: (prompt: string) => void }) {
  return <div className="flex min-h-[350px] flex-1 items-center justify-center px-4 py-8 sm:px-8">
    <div className="w-full max-w-[796px]">
      <div className="mb-5 grid size-11 place-items-center rounded-lg border border-[var(--line)] bg-[var(--panel)] text-[var(--text)]">
        <Orbit size={23} strokeWidth={1.65} />
      </div>
      <h1 className="max-w-[680px] text-[36px] leading-[1.08] font-semibold tracking-[-0.06em] sm:text-[52px]">What should GoXLM continue?</h1>
      <p className="mt-3 max-w-[510px] text-sm leading-6 text-[var(--muted)] sm:text-[15px]">
        Give your local model a starting point. Experimental completions may be inaccurate or incoherent.
      </p>
      <div className="mt-7 grid gap-2 sm:mt-8 sm:grid-cols-3 sm:gap-3">
        {ideas.map(({ label, text, icon: Icon }) => <button key={label} type="button" onClick={() => onPrompt(text)}
          className="group flex min-h-[70px] flex-col rounded-lg border border-[var(--line)] bg-[var(--panel)] p-3.5 text-left transition-colors hover:bg-[var(--panel-hover)] sm:min-h-[102px]">
          <span className="mb-1.5 flex w-full items-center gap-2 text-xs font-semibold text-[var(--text)]">
            <Icon size={16} className="text-[var(--accent)]" /> {label}
            <ArrowUpRight size={14} className="ml-auto text-[var(--quiet)] group-hover:text-[var(--accent)]" />
          </span>
          <span className="text-sm leading-[1.45] text-[var(--muted)]">{text.trimEnd()}</span>
        </button>)}
      </div>
    </div>
  </div>;
}

export function MessageList({ messages, busy }: { messages: Message[]; busy: boolean }) {
  return <div className="mx-auto w-full max-w-[860px] px-4 py-8 sm:px-8" aria-label="Messages">
    {messages.map((message, i) => {
      const generating = busy && i === messages.length - 1;
      if (message.role === "user") return <div key={message.id} className="mb-8 flex justify-end">
        <div className="max-w-[90%] rounded-xl rounded-br-sm border border-[var(--line)] bg-[var(--panel-hover)] px-4 py-3 sm:max-w-[78%]">
          <div className="mb-1 text-xs font-semibold text-[var(--muted)]">You</div>
          <p className="text-[15px] leading-6 wrap-anywhere whitespace-pre-wrap">{message.text}</p>
        </div>
      </div>;
      if (message.group) {
        if (messages[i - 1]?.group === message.group) return null;
        const pair = messages.slice(i, i + 2).filter((item) => item.group === message.group);
        return <div key={message.id} aria-label="Model comparison" className="mb-8 grid gap-3 sm:grid-cols-2">
          {pair.map((item) => <article key={item.id}
            className="min-w-0 rounded-lg border border-[var(--line)] bg-[var(--panel)] p-4">
            <Assistant message={item} waiting={busy} />
          </article>)}
        </div>;
      }
      return <div key={message.id} className="mb-8"><Assistant message={message} waiting={generating} /></div>;
    })}
  </div>;
}

function Assistant({ message, waiting }: { message: Message; waiting: boolean }) {
  const metrics = message.metrics;
  return <div className="flex items-start gap-3">
    <div className="grid size-8 shrink-0 place-items-center rounded-lg bg-[var(--accent-surface)] text-[var(--accent)]">
      <Orbit size={17} />
    </div>
    <div className="min-w-0 pt-1">
      <div className="mb-2 text-sm font-semibold">{message.model ?? "GoXLM"}</div>
      <p className="text-[15px] leading-[1.75] wrap-anywhere whitespace-pre-wrap">
        {message.text || (message.stopped ? "Stopped." : waiting && !metrics ? "Waiting for text…" : "No completion returned.")}
      </p>
      {metrics && <p className="mt-3 text-xs text-[var(--muted)]">
        {metrics.first !== null && <>First token {metrics.first.toFixed(1)} ms · </>}
        {metrics.rate.toFixed(1)} tok/s
        {metrics.p95 !== null && <> · p95 {metrics.p95.toFixed(1)} ms</>}
        {metrics.rebuilds > 0 && <> · {metrics.rebuilds} window rebuilds</>}
        {message.stopped && <> · Stopped</>}
      </p>}
    </div>
  </div>;
}
