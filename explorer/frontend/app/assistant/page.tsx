"use client";

import { useRef, useState } from "react";
import { streamInvestigation } from "@/lib/api";
import type { SSEEvent, OrchestratorResponse } from "@/lib/types";
import ReactMarkdown from "react-markdown";
import { Bot, Send, Loader2, User, RefreshCw } from "lucide-react";

interface Message {
  role:    "user" | "assistant" | "system";
  content: string;
  result?: OrchestratorResponse;
}

export default function AssistantPage() {
  const [messages, setMessages]   = useState<Message[]>([
    {
      role: "system",
      content:
        "👋 **Welcome to the AMR AI Assistant.**\n\nDescribe an incident or ask me to investigate something. I'll use real-time FAISS similarity search + LLM analysis to diagnose issues, rank causes, and suggest solutions.",
    },
  ]);
  const [input,   setInput]   = useState("");
  const [busy,    setBusy]    = useState(false);
  const [steps,   setSteps]   = useState<string[]>([]);
  const bottomRef             = useRef<HTMLDivElement>(null);
  const unsubRef              = useRef<(() => void) | null>(null);

  function scrollDown() {
    setTimeout(() => bottomRef.current?.scrollIntoView({ behavior: "smooth" }), 50);
  }

  function cancel() {
    unsubRef.current?.();
    setBusy(false);
    setSteps([]);
  }

  function reset() {
    cancel();
    setMessages([
      {
        role: "system",
        content:
          "👋 **Welcome to the AMR AI Assistant.**\n\nDescribe an incident or ask me to investigate something.",
      },
    ]);
    setInput("");
  }

  async function send() {
    const text = input.trim();
    if (!text || busy) return;
    setInput("");
    setBusy(true);
    setSteps([]);

    const userMsg: Message = { role: "user", content: text };
    setMessages((p) => [...p, userMsg]);
    scrollDown();

    // Push empty assistant bubble
    const placeholderIdx = messages.length + 1;
    setMessages((p) => [...p, { role: "assistant", content: "" }]);

    let accumulated = "";

    const unsub = streamInvestigation(
      { title: text, description: text },
      (ev: SSEEvent) => {
        setSteps((p) => {
          const label =
            ev.step === "start"             ? "Initialising…"              :
            ev.step === "bag_analysis"      ? "Analysing bag logs…"         :
            ev.step === "similarity_search" ? "Searching past incidents…"   :
            ev.step === "llm_analysis"      ? "Running AI analysis…"        :
            ev.step === "complete"          ? "Done"                        :
            ev.message;
          return [...new Set([...p, label])];
        });

        if (ev.message && ev.step !== "complete" && ev.step !== "error") {
          accumulated += `\n_${ev.message}_`;
          setMessages((p) =>
            p.map((m, i) => (i === placeholderIdx ? { ...m, content: accumulated } : m))
          );
          scrollDown();
        }

        if (ev.step === "complete" && ev.data) {
          const r = ev.data;
          const summary = [
            `**Confidence:** ${(r.confidence_score * 100).toFixed(0)}%` +
              (r.human_intervention_required ? " ⚠️ Human intervention required" : " ✅"),
            "",
            "**Top Causes:**",
            ...r.ranked_causes.slice(0, 3).map(
              (c, i) => `${i + 1}. ${c.description} *(${(c.confidence * 100).toFixed(0)}%)*`
            ),
            "",
            "**Top Solutions:**",
            ...r.ranked_solutions.slice(0, 3).map(
              (s, i) => `${i + 1}. ${s.description}`
            ),
            "",
            r.raw_analysis.slice(0, 600) + (r.raw_analysis.length > 600 ? "…" : ""),
          ].join("\n");

          setMessages((p) =>
            p.map((m, i) =>
              i === placeholderIdx ? { ...m, content: summary, result: r } : m
            )
          );
          setBusy(false);
          setSteps([]);
          scrollDown();
          unsub();
        }

        if (ev.step === "error") {
          setMessages((p) =>
            p.map((m, i) =>
              i === placeholderIdx
                ? { ...m, content: `❌ Error: ${ev.error ?? "Unknown server error."}` }
                : m
            )
          );
          setBusy(false);
          setSteps([]);
          unsub();
        }
      },
      () => {
        setBusy(false);
        setSteps([]);
      }
    );

    unsubRef.current = unsub;
  }

  function onKey(e: React.KeyboardEvent) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  }

  return (
    <div className="flex flex-col h-screen max-h-screen">
      {/* Header */}
      <div className="flex items-center justify-between px-6 py-4 border-b border-slate-800 shrink-0">
        <div className="flex items-center gap-2.5">
          <Bot size={20} className="text-blue-400" />
          <div>
            <h1 className="text-base font-semibold text-slate-100">AI Assistant</h1>
            <p className="text-xs text-slate-500">Streaming incident analysis</p>
          </div>
        </div>
        <button onClick={reset} className="btn btn-ghost gap-1.5 text-xs">
          <RefreshCw size={13} /> New chat
        </button>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-4 py-4 space-y-4">
        {messages.map((msg, i) => (
          <div
            key={i}
            className={`flex gap-3 ${msg.role === "user" ? "justify-end" : "justify-start"} animate-fade-in`}
          >
            {msg.role !== "user" && (
              <div className="w-7 h-7 rounded-full bg-blue-600/20 border border-blue-600/30 flex items-center justify-center shrink-0 mt-0.5">
                <Bot size={13} className="text-blue-400" />
              </div>
            )}
            <div
              className={`max-w-[75%] rounded-2xl px-4 py-2.5 text-sm ${
                msg.role === "user"
                  ? "bg-blue-600 text-white rounded-tr-sm"
                  : msg.role === "system"
                  ? "bg-slate-800/50 text-slate-400 border border-slate-700/40 rounded-tl-sm"
                  : "bg-slate-800 text-slate-200 border border-slate-700/40 rounded-tl-sm"
              }`}
            >
              {msg.content ? (
                <div className="prose-dark">
                  <ReactMarkdown>{msg.content}</ReactMarkdown>
                </div>
              ) : busy && i === messages.length - 1 ? (
                <div className="flex items-center gap-2 text-slate-500">
                  <Loader2 size={13} className="animate-spin" />
                  <span className="text-xs">{steps[steps.length - 1] ?? "Thinking…"}</span>
                </div>
              ) : null}
            </div>
            {msg.role === "user" && (
              <div className="w-7 h-7 rounded-full bg-slate-700 border border-slate-600 flex items-center justify-center shrink-0 mt-0.5">
                <User size={13} className="text-slate-400" />
              </div>
            )}
          </div>
        ))}
        <div ref={bottomRef} />
      </div>

      {/* Input bar */}
      <div className="px-4 pb-4 pt-2 border-t border-slate-800 shrink-0">
        {busy && (
          <div className="flex items-center gap-2 text-xs text-blue-300 mb-2">
            <Loader2 size={12} className="animate-spin" />
            <span>{steps[steps.length - 1] ?? "Processing…"}</span>
            <button onClick={cancel} className="ml-auto text-slate-500 hover:text-slate-300">Cancel</button>
          </div>
        )}
        <div className="flex gap-2">
          <textarea
            rows={2}
            className="input resize-none text-sm flex-1 py-2"
            placeholder="Describe an AMR incident… (Enter to send, Shift+Enter for new line)"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={onKey}
            disabled={busy}
          />
          <button
            className="btn btn-primary px-4 self-end gap-1.5"
            onClick={send}
            disabled={busy || !input.trim()}
          >
            <Send size={15} />
          </button>
        </div>
      </div>
    </div>
  );
}
