"use client";

import type { SimilarCase } from "@/lib/types";
import { Clock, CheckCircle } from "lucide-react";
import { useState } from "react";

interface Props {
  cases: SimilarCase[];
}

export default function SimilarCasesTable({ cases }: Props) {
  const [expanded, setExpanded] = useState<string | null>(null);

  if (cases.length === 0)
    return <p className="text-xs text-slate-600 py-3">No similar past incidents found.</p>;

  return (
    <div className="space-y-2">
      {cases.map((c) => {
        const pct = (c.similarity * 100).toFixed(0);
        const open = expanded === c.id;
        return (
          <div
            key={c.id}
            className="border border-slate-700/60 rounded-lg overflow-hidden cursor-pointer hover:border-slate-600 transition-colors"
            onClick={() => setExpanded(open ? null : c.id)}
          >
            <div className="flex items-center gap-3 p-3">
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium text-slate-200 truncate">{c.title}</p>
                <p className="text-xs text-slate-500 truncate mt-0.5">{c.description}</p>
              </div>
              <div className="flex items-center gap-1.5 shrink-0">
                <span className="badge badge-blue">{pct}% match</span>
              </div>
            </div>
            {open && (
              <div className="px-4 pb-3 border-t border-slate-700/40 pt-2 space-y-1.5">
                <div className="flex items-center gap-1.5 text-xs text-slate-400">
                  <Clock size={11} className="text-slate-600" />
                  <span className="text-slate-500">Case ID:</span> {c.id}
                </div>
                {c.resolution && (
                  <div className="flex items-start gap-1.5 text-xs text-green-300 bg-green-900/20 rounded-md p-2.5 border border-green-800/30">
                    <CheckCircle size={12} className="shrink-0 mt-0.5" />
                    <div>
                      <p className="font-semibold mb-0.5">Resolution</p>
                      <p className="text-green-400/80">{c.resolution}</p>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
