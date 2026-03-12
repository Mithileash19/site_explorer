"use client";

import { useState } from "react";
import { runMapDiff } from "@/lib/api";
import type { MapDiffResponse } from "@/lib/types";
import { GitCompare, Loader2, AlertTriangle } from "lucide-react";

interface Props {
  bagPath: string;
}

export default function MapDiffPanel({ bagPath }: Props) {
  const [result,  setResult]  = useState<MapDiffResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error,   setError]   = useState("");
  const [topic,   setTopic]   = useState("");

  async function run() {
    setLoading(true);
    setError("");
    setResult(null);
    try {
      const r = await runMapDiff(bagPath, topic || undefined);
      setResult(r);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Map diff failed");
    } finally {
      setLoading(false);
    }
  }

  const iouColour =
    result && result.iou_score > 0.85 ? "text-green-400" :
    result && result.iou_score > 0.6  ? "text-amber-400" :
    "text-red-400";

  return (
    <div className="space-y-4">
      <div className="flex gap-2">
        <input
          className="input text-sm flex-1"
          placeholder="/map topic (leave blank = auto)"
          value={topic}
          onChange={(e) => setTopic(e.target.value)}
        />
        <button
          className="btn btn-primary gap-1.5 shrink-0"
          onClick={run}
          disabled={loading || !bagPath}
        >
          {loading ? <Loader2 size={14} className="animate-spin" /> : <GitCompare size={14} />}
          {loading ? "Running…" : "Run Diff"}
        </button>
      </div>

      {error && (
        <div className="flex items-center gap-2 text-xs text-red-300 bg-red-900/20 rounded-lg p-3 border border-red-800/30">
          <AlertTriangle size={14} /> {error}
        </div>
      )}

      {result && (
        <div className="space-y-3 animate-fade-in">
          <div className="flex items-center gap-3">
            <span className="text-xs text-slate-400">Map similarity (IoU)</span>
            <span className={`text-xl font-bold ${iouColour}`}>
              {(result.iou_score * 100).toFixed(1)}%
            </span>
            <span className={`badge ${result.iou_score > 0.85 ? "badge-green" : result.iou_score > 0.6 ? "badge-yellow" : "badge-red"}`}>
              {result.iou_score > 0.85 ? "stable" : result.iou_score > 0.6 ? "shifted" : "major change"}
            </span>
          </div>
          {result.message && (
            <p className="text-xs text-slate-500">{result.message}</p>
          )}
          {result.diff_image_b64 && (
            <div className="rounded-lg overflow-hidden border border-slate-700/40">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={`data:image/png;base64,${result.diff_image_b64}`}
                alt="Map diff overlay"
                className="w-full h-auto"
              />
            </div>
          )}
        </div>
      )}
    </div>
  );
}
