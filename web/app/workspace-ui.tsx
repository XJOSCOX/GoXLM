import { ArrowUpRight, Atom, CodeXml, MessageSquare, Moon, Orbit, Plus, Sigma, Sun, X } from "lucide-react";

export type Message = { id: string; role: "user" | "assistant"; text: string; model?: string };
export type Conversation = { id: string; title: string; messages: Message[] };

type RailProps = {
  conversations: Conversation[];
  selected: string | null;
  busy: boolean;
  open: boolean;
  connected: boolean;
  device?: string;
  theme: "dark" | "light";
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
    <aside aria-label="Conversations" className={`z-20 flex h-full w-[258px] shrink-0 flex-col border-r border-[var(--line)] bg-[var(--rail)] px-4 py-5
      ${open ? "fixed inset-y-0 left-0" : "hidden"} md:static md:flex`}>
      <div className="mb-7 flex items-center gap-3 px-1">
        <div className="grid h-10 w-10 shrink-0 place-items-center rounded-xl bg-[var(--accent)] text-[var(--accent-ink)]">
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
        className="flex w-full items-center justify-center gap-2.5 rounded-xl border border-[var(--line)] bg-[var(--accent-surface)] py-3 text-sm font-semibold text-[var(--accent)] transition-colors hover:bg-[var(--panel-hover)] disabled:opacity-50">
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
          {theme === "dark" ? <Sun size={16} /> : <Moon size={16} />} {theme === "dark" ? "Light mode" : "Dark mode"}
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
  return <div className="relative isolate flex min-h-[370px] flex-1 items-center justify-center px-4 py-7 sm:px-8">
    <div className="hero-halo pointer-events-none absolute top-1/2 left-1/2 -z-10 h-[360px] w-full max-w-[730px] -translate-x-1/2 -translate-y-1/2 opacity-50" />
    <div className="w-full max-w-[702px]">
      <div className="mx-auto mb-5 grid h-[60px] w-[60px] place-items-center rounded-[18px] border border-[var(--line)] bg-[var(--accent-surface)] text-[var(--accent)]">
        <Orbit size={31} strokeWidth={1.65} />
      </div>
      <h1 className="text-center text-[33px] font-semibold tracking-[-0.045em] sm:text-[46px]">Start with a prompt.</h1>
      <p className="mx-auto mt-2.5 max-w-[425px] text-center text-sm leading-6 text-[var(--muted)] sm:text-[15px]">
        GoXLM continues the text you provide. The local model is experimental and may produce incoherent results.
      </p>
      <div className="mt-7 grid gap-2 sm:mt-9 sm:grid-cols-3 sm:gap-3">
        {ideas.map(({ label, text, icon: Icon }) => <button key={label} type="button" onClick={() => onPrompt(text)}
          className="group flex min-h-[76px] flex-col rounded-xl border border-[var(--line)] bg-[var(--panel)] p-3.5 text-left transition-colors hover:bg-[var(--panel-hover)] sm:min-h-[110px]">
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
        <div className="max-w-[90%] rounded-2xl rounded-br-md border border-[var(--line)] bg-[var(--panel-hover)] px-4 py-3 sm:max-w-[78%]">
          <div className="mb-1 text-xs font-semibold text-[var(--muted)]">You</div>
          <p className="text-[15px] leading-6 wrap-anywhere whitespace-pre-wrap">{message.text}</p>
        </div>
      </div>;
      return <div key={message.id} className="mb-8 flex items-start gap-3 sm:gap-4">
        <div className="grid h-8 w-8 shrink-0 place-items-center rounded-lg bg-[var(--accent-surface)] text-[var(--accent)]"><Orbit size={17} /></div>
        <div className="min-w-0 pt-1">
          <div className="mb-2 text-sm font-semibold">{message.model ?? "GoXLM"}</div>
          <p className="text-[15px] leading-7 wrap-anywhere whitespace-pre-wrap">
            {message.text || (generating ? "Waiting for text…" : "No completion returned.")}
            {generating && message.text && <span className="ml-1 inline-block h-3 w-[2px] animate-pulse bg-[var(--accent)]" aria-hidden />}
          </p>
        </div>
      </div>;
    })}
  </div>;
}
