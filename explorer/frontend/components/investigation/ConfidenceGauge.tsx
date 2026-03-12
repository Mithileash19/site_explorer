"use client";

interface Props {
  score: number; // 0-1
}

export default function ConfidenceGauge({ score }: Props) {
  const pct = Math.round(score * 100);
  const colour =
    pct >= 70 ? "#22c55e" :
    pct >= 40 ? "#f59e0b" :
    "#ef4444";

  // SVG arc parameters
  const R    = 52;
  const CX   = 60;
  const CY   = 60;
  const FULL = 2 * Math.PI * R;
  // Draw only the top 75% of the circle (270°)
  const ARC  = FULL * 0.75;
  const dash = `${(pct / 100) * ARC} ${FULL}`;

  return (
    <div className="flex flex-col items-center gap-1">
      <svg width="120" height="80" viewBox="0 0 120 80" className="overflow-visible">
        {/* Track */}
        <circle
          cx={CX} cy={CY} r={R}
          fill="none"
          stroke="#1e293b"
          strokeWidth="10"
          strokeDasharray={`${ARC} ${FULL}`}
          strokeDashoffset={FULL * 0.125}
          strokeLinecap="round"
          transform={`rotate(-135 ${CX} ${CY})`}
        />
        {/* Arc */}
        <circle
          cx={CX} cy={CY} r={R}
          fill="none"
          stroke={colour}
          strokeWidth="10"
          strokeDasharray={dash}
          strokeDashoffset={FULL * 0.125}
          strokeLinecap="round"
          transform={`rotate(-135 ${CX} ${CY})`}
          style={{ transition: "stroke-dasharray 0.6s ease" }}
        />
        <text x={CX} y={CY + 6} textAnchor="middle" fill={colour} fontSize="18" fontWeight="700">
          {pct}%
        </text>
      </svg>
      <p className="text-xs text-slate-500 -mt-1">AI Confidence</p>
    </div>
  );
}
