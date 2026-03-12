"use client";

import { useState } from "react";
import { analyzeBag, fetchTimeline } from "@/lib/api";
import type { BagLogAnalysisResponse, BagTimeline } from "@/lib/types";
import BagUpload from "@/components/bags/BagUpload";
import LogVolumeChart from "@/components/bags/LogVolumeChart";
import BagLogDebugger from "@/components/bags/BagLogDebugger";
import MapDiffPanel from "@/components/bags/MapDiffPanel";
import ReactMarkdown from "react-markdown";
import {
  PackageSearch,
  Loader2,
  GitCompare,
  ChevronDown,
  ChevronRight,
  Info,
  Bot,
} from "lucide-react";

type Tab = "logs" | "mapdiff";

export default function BagsPage() {
  const [bagPath, setBagPath] = useState<string | null>(null);
  const [timeline, setTimeline] = useState<BagTimeline | null>(null);
  const [analysis, setAnalysis] = useState<BagLogAnalysisResponse | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [error, setError] = useState("");
  const [tab, setTab] = useState<Tab>("logs");
  const [timelineOpen, setTimelineOpen] = useState(true);
  const [showRawLLM, setShowRawLLM] = useState(false);

  async function onUploaded(path: string) {
    setBagPath(path);
    setAnalysis(null);
    setTimeline(null);
    setError("");
    try {
      const tl = await fetchTimeline(path);
      setTimeline(tl);
    } catch {
      /* timeline is optional */
    }
  }

  async function runAnalysis(windowStart?: number, windowEnd?: number) {
    if (!bagPath) return;
    setAnalyzing(true);
    setError("");
    try {
      const res = await analyzeBag(bagPath, windowStart, windowEnd);
      setAnalysis(res);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Analysis failed");
    } finally {
      setAnalyzing(false);
    }
  }

  function onRangeSelect(start: number, end: number) {
    runAnalysis(start, end);
  }

  return (
    <div className="p-6 max-w-6xl mx-auto animate-fade-in space-y-5">
      {/* ── Header ────────────────────────────────────────── */}
      <div className="flex items-center gap-3">
        <PackageSearch size={20} className="text-blue-400" />
        <div>
          <h1 className="text-xl font-bold text-slate-100">Bag Analyzer</h1>
          <p className="text-xs text-slate-400">
            Upload a ROS bag, explore logs with timeline, run AI analysis
          </p>
        </div>
      </div>

      {/* ── Upload ────────────────────────────────────────── */}
      <section className="card">
        <h2 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">
          Upload Bag File
        </h2>
        <BagUpload onUploaded={onUploaded} />
      </section>

      {/* ── Collapsible Timeline ──────────────────────────── */}
      {timeline && (
        <section className="card p-0 overflow-hidden">
          <button
            onClick={() => setTimelineOpen((p) => !p)}
            className="flex items-center gap-2 w-full px-4 py-3 text-xs font-semibold text-slate-400 uppercase tracking-wider hover:bg-white/5 transition-colors"
          >
            {timelineOpen ? (
              <ChevronDown size={14} />
            ) : (
              <ChevronRight size={14} />
            )}
            Log Volume Timeline
          </button>
          {timelineOpen && (
            <div className="px-4 pb-4">
              <LogVolumeChart
                buckets={timeline.buckets}
                onRangeSelect={onRangeSelect}
              />
              <p className="text-xs text-slate-500 mt-2">
                Drag on the timeline to select a time range and analyze logs
              </p>
            </div>
          )}
        </section>
      )}

      {/* ── Analyze button ────────────────────────────────── */}
      {bagPath && !analyzing && (
        <button
          className="btn btn-primary w-full py-3 text-sm gap-2"
          onClick={() => runAnalysis()}
        >
          <PackageSearch size={16} />
          Analyze Full Bag
        </button>
      )}

      {/* ── Loading ───────────────────────────────────────── */}
      {analyzing && (
        <div className="flex items-center justify-center gap-3 text-blue-300 py-10">
          <Loader2 className="animate-spin" size={22} />
          <span className="text-sm">Running analysis… this may take a moment</span>
        </div>
      )}

      {/* ── Error ─────────────────────────────────────────── */}
      {error && (
        <div className="card border-red-800/30 bg-red-900/10 text-red-300 text-sm">
          {error}
        </div>
      )}

      {/* ── Analysis Results ──────────────────────────────── */}
      {analysis && (
        <>
          {/* Stats bar — 4 cards, no Critical / Anomalies */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            {(
              [
                { label: "Duration", value: `${analysis.duration_secs.toFixed(1)}s` },
                { label: "Messages", value: analysis.total_messages },
                { label: "Errors", value: analysis.error_count, cls: "text-red-400" },
                { label: "Warnings", value: analysis.warning_count, cls: "text-amber-400" },
              ] as { label: string; value: string | number; cls?: string }[]
            ).map(({ label, value, cls }) => (
              <div key={label} className="card text-center py-3">
                <p className={`text-xl font-bold ${cls ?? "text-slate-100"}`}>
                  {value}
                </p>
                <p className="text-xs text-slate-500 mt-0.5">{label}</p>
              </div>
            ))}
          </div>

          {/* Engine hypothesis */}
          {analysis.engine_hypothesis && (
            <div className="card border-l-2 border-blue-500 rounded-l-none bg-blue-900/10">
              <div className="flex items-center gap-1.5 mb-1 text-xs font-semibold text-blue-300 uppercase tracking-wide">
                <Info size={13} /> Rule-Engine Hypothesis
              </div>
              <p className="text-xs text-slate-400">
                {analysis.engine_hypothesis}
              </p>
            </div>
          )}

          {/* Tab bar */}
          <div className="flex gap-2 border-b border-slate-700/50 pb-2">
            {(
              [
                { id: "logs", label: "Log Analysis", icon: PackageSearch },
                { id: "mapdiff", label: "Map Diff", icon: GitCompare },
              ] as { id: Tab; label: string; icon: React.ElementType }[]
            ).map(({ id, label, icon: Icon }) => (
              <button
                key={id}
                onClick={() => setTab(id)}
                className={`flex items-center gap-1.5 px-3 py-1.5 text-sm rounded-md transition-colors ${
                  tab === id
                    ? "bg-blue-600/20 text-blue-400 border border-blue-600/30"
                    : "text-slate-400 hover:text-slate-200 hover:bg-white/5"
                }`}
              >
                <Icon size={14} />
                {label}
              </button>
            ))}
          </div>

          {/* Tab content */}
          {tab === "logs" && <BagLogDebugger analysis={analysis} />}
          {tab === "mapdiff" && bagPath && (
            <div className="card">
              <MapDiffPanel bagPath={bagPath} />
            </div>
          )}

          {/* ── LLM Analysis (bottom) ─────────────────────── */}
          <section className="card">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider flex items-center gap-1.5">
                <Bot size={13} className="text-blue-400" /> AI Analysis
              </h3>
              {analysis.llm_summary && (
                <button
                  onClick={() => setShowRawLLM((p) => !p)}
                  className="text-xs text-slate-500 hover:text-slate-300 transition-colors"
                >
                  {showRawLLM ? "Formatted" : "Raw"}
                </button>
              )}
            </div>
            {analysis.llm_summary ? (
              showRawLLM ? (
                <pre className="text-xs text-slate-400 whitespace-pre-wrap font-mono overflow-x-auto">
                  {analysis.llm_summary}
                </pre>
              ) : (
                <div className="prose-dark">
                  <ReactMarkdown>{analysis.llm_summary}</ReactMarkdown>
                </div>
              )
            ) : (
              <p className="text-xs text-slate-500">
                AI analysis will appear here after running analysis.
              </p>
            )}
          </section>
        </>
      )}
    </div>
  );
}
