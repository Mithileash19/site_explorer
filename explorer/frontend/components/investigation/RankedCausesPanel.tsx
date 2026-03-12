"use client";

import type { RankedItem } from "@/lib/types";
import { ChevronDown, ChevronUp } from "lucide-react";
import { useState } from "react";

interface Props {
  items: RankedItem[];
  title: string;
  accentColour?: string;
}

function Item({ item, rank, colour }: { item: RankedItem; rank: number; colour: string }) {
  const [open, setOpen] = useState(false);
  const pct = (item.confidence * 100).toFixed(0);

  return (
    <div
      className="border border-slate-700/60 rounded-lg overflow-hidden cursor-pointer hover:border-slate-600 transition-colors"
      onClick={() => setOpen((p) => !p)}
    >
      <div className="flex items-center gap-3 p-3">
        <span className="text-xs font-bold text-slate-600 w-4 shrink-0">#{rank}</span>
        <div className="flex-1 min-w-0">
          <p className="text-sm text-slate-200 truncate">{item.description}</p>
          {/* bar */}
          <div className="mt-1.5 h-1 bg-slate-800 rounded-full overflow-hidden">
            <div
              className="h-full rounded-full transition-all"
              style={{ width: `${pct}%`, background: colour }}
            />
          </div>
        </div>
        <span className="text-xs font-semibold shrink-0" style={{ color: colour }}>{pct}%</span>
        {open ? <ChevronUp size={13} className="text-slate-600 shrink-0" /> : <ChevronDown size={13} className="text-slate-600 shrink-0" />}
      </div>

      {open && item.evidence.length > 0 && (
        <div className="px-4 pb-3 border-t border-slate-700/40 pt-2 space-y-1">
          {item.evidence.map((ev, i) => (
            <p key={i} className="text-xs text-slate-500 flex items-start gap-1.5">
              <span className="text-slate-600 shrink-0 mt-0.5">›</span>{ev}
            </p>
          ))}
        </div>
      )}
    </div>
  );
}

export default function RankedCausesPanel({ items, title, accentColour = "#3b82f6" }: Props) {
  return (
    <div>
      <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">{title}</h3>
      {items.length === 0 ? (
        <p className="text-xs text-slate-600 py-3">No items returned.</p>
      ) : (
        <div className="space-y-2">
          {items.map((item, i) => (
            <Item key={i} item={item} rank={i + 1} colour={accentColour} />
          ))}
        </div>
      )}
    </div>
  );
}
