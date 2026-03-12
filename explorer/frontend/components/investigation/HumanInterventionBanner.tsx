"use client";

import { AlertTriangle, Phone } from "lucide-react";

interface Props {
  required: boolean;
  confidence: number;
}

export default function HumanInterventionBanner({ required, confidence }: Props) {
  if (!required) return null;

  return (
    <div className="flex items-start gap-3 p-4 rounded-xl bg-red-900/20 border border-red-700/40 animate-fade-in">
      <div className="p-2 bg-red-600/20 rounded-lg shrink-0">
        <AlertTriangle size={18} className="text-red-400" />
      </div>
      <div className="flex-1">
        <p className="text-sm font-semibold text-red-300 flex items-center gap-2">
          Human Intervention Required
          <Phone size={13} className="text-red-400" />
        </p>
        <p className="text-xs text-red-400/70 mt-0.5">
          AI confidence is {(confidence * 100).toFixed(0)}% — below the automatic resolution threshold.
          A field engineer should review this incident directly.
        </p>
      </div>
    </div>
  );
}
