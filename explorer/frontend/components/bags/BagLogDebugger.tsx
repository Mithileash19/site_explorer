"use client";

import { useState, useMemo } from "react";
import type { BagLogAnalysisResponse, LogEntry } from "@/lib/types";
import {
  Filter,
  ChevronDown,
  ChevronUp,
} from "lucide-react";

const LEVEL_COLOUR: Record<string, string> = {
  DEBUG: "text-slate-500",
  INFO:  "text-blue-400",
  WARN:  "text-amber-400",
  WARNING: "text-amber-400",
  ERROR: "text-red-400",
  FATAL: "text-red-300 font-semibold",
};

const LEVEL_BAR: Record<string, string> = {
  DEBUG: "bg-slate-600",
  INFO:  "bg-blue-500",
  WARN:  "bg-amber-500",
  WARNING: "bg-amber-500",
  ERROR: "bg-red-500",
  FATAL: "bg-red-400",
};

function LogRow({ entry }: { entry: LogEntry }) {
  const [open, setOpen] = useState(false);
  const colour = LEVEL_COLOUR[entry.level.toUpperCase()] ?? "text-slate-300";
  const barCol = LEVEL_BAR[entry.level.toUpperCase()] ?? "bg-slate-600";
  const ts = entry.datetime || `${entry.timestamp.toFixed(3)}s`;

  return (
    <div
      className="font-mono text-xs border-b border-slate-800 hover:bg-slate-700/20 cursor-pointer flex"
      onClick={() => setOpen((p) => !p)}
    >
      {/* Level colour bar */}
      <div className={`w-1 shrink-0 ${barCol}`} />
      <div className="flex-1 min-w-0">
        <div className="flex items-start gap-2 px-3 py-1.5">
          <span className="text-slate-500 shrink-0 w-44 truncate">{ts}</span>
          <span className={`shrink-0 w-14 ${colour}`}>{entry.level}</span>
          <span className="text-slate-500 shrink-0 w-28 truncate">{entry.node}</span>
          <span className="text-slate-300 flex-1 truncate">{entry.message}</span>
          {open ? (
            <ChevronUp size={12} className="text-slate-600 shrink-0 mt-0.5" />
          ) : (
            <ChevronDown size={12} className="text-slate-600 shrink-0 mt-0.5" />
          )}
        </div>
        {open && (
          <div className="px-3 pb-2 text-slate-400 whitespace-pre-wrap break-all leading-relaxed pl-[12.5rem]">
            {entry.message}
          </div>
        )}
      </div>
    </div>
  );
}

interface Props {
  analysis: BagLogAnalysisResponse;
}

const LEVELS = ["ALL", "DEBUG", "INFO", "WARN", "ERROR", "FATAL"] as const;

export default function BagLogDebugger({ analysis }: Props) {
  const [levelFilter, setLevelFilter] = useState<string>("ALL");
  const [nodeFilter,  setNodeFilter]  = useState("");
  const [search,      setSearch]      = useState("");
  const [maxRows,     setMaxRows]     = useState(200);

  const filtered: LogEntry[] = useMemo(() => {
    return analysis.log_entries.filter((e) => {
      if (levelFilter !== "ALL" && e.level.toUpperCase() !== levelFilter) return false;
      if (nodeFilter && !e.node.toLowerCase().includes(nodeFilter.toLowerCase())) return false;
      if (search && !e.message.toLowerCase().includes(search.toLowerCase())) return false;
      return true;
    });
  }, [analysis.log_entries, levelFilter, nodeFilter, search]);

  return (
    <div className="space-y-3 animate-fade-in">
      {/* ── Search & Filter Bar ──────────────────────────── */}
      <div className="flex flex-wrap items-center gap-2">
        {/* Level buttons */}
        <div className="flex gap-1">
          {LEVELS.map((l) => (
            <button
              key={l}
              onClick={() => setLevelFilter(l)}
              className={`px-2 py-1 rounded text-xs font-medium transition-colors ${
                levelFilter === l
                  ? "bg-blue-600 text-white"
                  : "bg-slate-800 text-slate-400 hover:bg-slate-700"
              }`}
            >
              {l}
            </button>
          ))}
        </div>
        <div className="flex-1 flex gap-2 min-w-0">
          <input
            className="input flex-1 min-w-[100px] py-1 text-xs"
            placeholder="Filter by node / topic…"
            value={nodeFilter}
            onChange={(e) => setNodeFilter(e.target.value)}
          />
          <input
            className="input flex-1 min-w-[140px] py-1 text-xs"
            placeholder="Search messages…"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
        </div>
        <span className="badge badge-blue text-xs flex items-center gap-1">
          <Filter size={11} />
          {filtered.length} logs
        </span>
      </div>

      {/* ── Logs Viewer ──────────────────────────────────── */}
      <div className="border border-slate-700/60 rounded-lg overflow-hidden bg-[#0b1120] max-h-[520px] overflow-y-auto">
        {/* Header */}
        <div className="flex items-center gap-2 px-3 py-1.5 text-xs text-slate-500 border-b border-slate-700/40 font-medium bg-slate-800/40 sticky top-0 z-10">
          <span className="w-1 shrink-0" />
          <span className="w-44">Timestamp</span>
          <span className="w-14">Level</span>
          <span className="w-28">Node</span>
          <span className="flex-1">Message</span>
        </div>
        {filtered.length === 0 ? (
          <p className="py-8 text-center text-xs text-slate-600">
            No log entries match this filter.
          </p>
        ) : (
          <>
            {filtered.slice(0, maxRows).map((entry, i) => (
              <LogRow key={i} entry={entry} />
            ))}
            {filtered.length > maxRows && (
              <button
                onClick={() => setMaxRows((n) => n + 200)}
                className="w-full py-2 text-xs text-slate-500 hover:text-slate-300 hover:bg-slate-700/20 transition-colors"
              >
                Show more ({filtered.length - maxRows} remaining)
              </button>
            )}
          </>
        )}
      </div>
    </div>
  );
}
