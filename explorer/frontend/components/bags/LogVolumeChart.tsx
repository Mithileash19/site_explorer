"use client";

import { useEffect, useRef, useCallback, useState } from "react";
import type { TimelineBucket } from "@/lib/types";

interface Props {
  buckets: TimelineBucket[];
  onRangeSelect?: (start: number, end: number) => void;
}

const PAD       = { top: 8, bottom: 34, left: 52, right: 8 };
const CANVAS_H  = 240;
const HANDLE_HIT = 7;

const C_BG      = "#111217";
const C_GRID    = "#1c2128";
const C_AXIS    = "#9ba7b2";
const C_BASELINE = "#2f3540";
const C_GREEN   = "#73BF69";
const C_WARN    = "#F2CC0C";
const C_ERROR   = "#E02F44";

type DragMode = "select" | "resize-left" | "resize-right" | "pan";

interface DragState {
  active:          boolean;
  mode:            DragMode;
  startX:          number;
  currentX:        number;
  sel:             [number, number] | null;
  selBeforeResize: [number, number] | null;
  hoverIdx:        number | null;
}

function fmtDatetime(unixSec: number): string {
  const d  = new Date(unixSec * 1000);
  const mo = String(d.getMonth() + 1).padStart(2, "0");
  const dy = String(d.getDate()).padStart(2, "0");
  const hh = String(d.getHours()).padStart(2, "0");
  const mm = String(d.getMinutes()).padStart(2, "0");
  return `${mo}/${dy} ${hh}:${mm}`;
}

function fmtDuration(sec: number): string {
  if (!isFinite(sec)) return "--";
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  const s = Math.floor(sec % 60);
  if (h > 0) return `${h}:${p2(m)}:${p2(s)}`;
  return `${p2(m)}:${p2(s)}`;
}
function p2(n: number) { return String(n).padStart(2, "0"); }

export default function LogVolumeChart({ buckets, onRangeSelect }: Props) {
  const wrapRef   = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [canvasW, setCanvasW] = useState(800);

  const [tooltip, setTooltip] = useState<{
    x: number; y: number; lines: string[];
  } | null>(null);
  const [selInfo, setSelInfo] = useState<{
    label: string; secondary: string;
  } | null>(null);
  const [cursor, setCursor] = useState("crosshair");

  const ds = useRef<DragState>({
    active: false, mode: "select",
    startX: 0, currentX: 0,
    sel: null, selBeforeResize: null, hoverIdx: null,
  });

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver((e) => {
      const w = e[0].contentRect.width;
      if (w > 0) setCanvasW(Math.floor(w));
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const chartW   = canvasW - PAD.left - PAD.right;
  const chartH   = CANVAS_H - PAD.top - PAD.bottom;
  const barW     = buckets.length > 0 ? chartW / buckets.length : 1;
  const maxCount = Math.max(...buckets.map((b) => b.count), 1);
  const logMax   = Math.log2(maxCount + 1);

  const bucketSecs  = buckets.length > 0 ? buckets[0].t_end - buckets[0].t_start : 60;
  const bucketLabel =
    bucketSecs >= 120 ? `per ${Math.round(bucketSecs / 60)}m`
    : bucketSecs >= 60 ? "per 1m"
    : `per ${Math.round(bucketSecs)}s`;

  const xToBucket = useCallback((x: number) => {
    const i = Math.floor((x - PAD.left) / barW);
    return Math.max(0, Math.min(buckets.length - 1, i));
  }, [barW, buckets.length]);

  const bCentreX = useCallback((i: number) =>
    PAD.left + i * barW + barW / 2, [barW]);

  const selPx = useCallback((sel: [number, number] | null) => {
    if (!sel) return null;
    return { x1: PAD.left + sel[0] * barW, x2: PAD.left + (sel[1] + 1) * barW };
  }, [barW]);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || buckets.length === 0) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const st = ds.current;

    const dpr = window.devicePixelRatio || 1;
    canvas.width  = canvasW * dpr;
    canvas.height = CANVAS_H * dpr;
    canvas.style.width  = `${canvasW}px`;
    canvas.style.height = `${CANVAS_H}px`;
    ctx.scale(dpr, dpr);

    ctx.fillStyle = C_BG;
    ctx.fillRect(0, 0, canvasW, CANVAS_H);

    /* Y-axis ticks: 0.500, 1, 2, 4, 8 ... */
    const yTicks: number[] = [];
    let tv = 0.5;
    while (tv <= maxCount * 2) {
      yTicks.push(tv);
      tv = tv < 1 ? 1 : tv * 2;
    }

    ctx.font = "10px 'Inter', sans-serif";
    ctx.textAlign = "right";

    yTicks.forEach((v) => {
      const y = PAD.top + chartH - (Math.log2(v + 1) / logMax) * chartH;
      if (y < PAD.top - 2 || y > PAD.top + chartH + 2) return;
      ctx.strokeStyle = C_GRID;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(PAD.left, y);
      ctx.lineTo(canvasW - PAD.right, y);
      ctx.stroke();
      ctx.fillStyle = C_AXIS;
      const lbl = v < 1 ? v.toFixed(3) : v >= 1000 ? `${Math.round(v / 1000)}k` : String(Math.round(v));
      ctx.fillText(lbl, PAD.left - 5, y + 3.5);
    });

    ctx.strokeStyle = C_BASELINE;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(PAD.left, PAD.top + chartH);
    ctx.lineTo(canvasW - PAD.right, PAD.top + chartH);
    ctx.stroke();

    /* Bars */
    const gap  = Math.max(0.5, barW * 0.08);
    const netW = Math.max(1, barW - gap);

    buckets.forEach((b, i) => {
      if (b.count === 0) return;
      const x     = PAD.left + i * barW + gap / 2;
      const total = Math.max(1.5, (Math.log2(b.count + 1) / logMax) * chartH);
      const yBase = PAD.top + chartH - total;
      const errR  = b.error_count / b.count;
      const warnR = b.warn_count  / b.count;
      const errH  = errR  * total;
      const warnH = warnR * total;
      const normH = total - errH - warnH;

      if (normH > 0) { ctx.fillStyle = C_GREEN; ctx.fillRect(x, yBase + errH + warnH, netW, normH); }
      if (warnH > 0) { ctx.fillStyle = C_WARN;  ctx.fillRect(x, yBase + errH,          netW, warnH); }
      if (errH  > 0) { ctx.fillStyle = C_ERROR; ctx.fillRect(x, yBase,                 netW, errH);  }

      if (st.hoverIdx === i) {
        ctx.fillStyle = "rgba(255,255,255,0.05)";
        ctx.fillRect(PAD.left + i * barW, PAD.top, barW, chartH);
      }
    });

    /* X-axis datetime labels */
    ctx.fillStyle = C_AXIS;
    ctx.textAlign = "center";
    ctx.font = "10px 'Inter', sans-serif";
    const step = Math.max(1, Math.ceil(buckets.length / 10));
    buckets.forEach((b, i) => {
      if (i % step !== 0) return;
      ctx.fillText(fmtDatetime(b.t_start), bCentreX(i), CANVAS_H - 6);
    });

    /* Selection overlay */
    let sx1 = 0, sx2 = 0, hasOverlay = false;
    if (st.active && st.mode === "select") {
      sx1 = Math.min(st.startX, st.currentX);
      sx2 = Math.max(st.startX, st.currentX);
      hasOverlay = sx2 - sx1 > 3;
    } else if (st.sel) {
      const px = selPx(st.sel)!;
      sx1 = px.x1; sx2 = px.x2;
      hasOverlay = true;
    }

    if (hasOverlay) {
      ctx.fillStyle = "rgba(0,0,0,0.50)";
      ctx.fillRect(PAD.left, PAD.top, sx1 - PAD.left,            chartH);
      ctx.fillRect(sx2,      PAD.top, canvasW - PAD.right - sx2, chartH);

      ctx.fillStyle = "rgba(59,130,246,0.10)";
      ctx.fillRect(sx1, PAD.top, sx2 - sx1, chartH);

      ctx.strokeStyle = "rgba(99,155,255,0.75)";
      ctx.lineWidth = 1.5;
      ctx.strokeRect(sx1 + 0.75, PAD.top, sx2 - sx1 - 1.5, chartH);

      [sx1, sx2].forEach((hx) => {
        const g = ctx.createLinearGradient(hx - 3, 0, hx + 3, 0);
        g.addColorStop(0,   "transparent");
        g.addColorStop(0.5, "rgba(99,155,255,0.95)");
        g.addColorStop(1,   "transparent");
        ctx.fillStyle = g;
        ctx.fillRect(hx - 3, PAD.top, 6, chartH);
      });

      if (st.active && st.mode === "select") {
        const lB = xToBucket(sx1), rB = xToBucket(sx2);
        ctx.font      = "bold 11px 'Inter', sans-serif";
        ctx.textAlign = "center";
        ctx.fillStyle = "rgba(99,155,255,1)";
        ctx.fillText(fmtDatetime(buckets[lB].t_start), Math.max(sx1, PAD.left + 44),            PAD.top + 14);
        ctx.fillText(fmtDatetime(buckets[rB].t_end),   Math.min(sx2, canvasW - PAD.right - 44), PAD.top + 14);
      }
    }

    if (!hasOverlay && st.hoverIdx !== null && !st.active) {
      const cx = bCentreX(st.hoverIdx);
      ctx.strokeStyle = "rgba(99,155,255,0.30)";
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(cx, PAD.top);
      ctx.lineTo(cx, PAD.top + chartH);
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }, [buckets, canvasW, chartH, logMax, maxCount, barW, bCentreX, selPx, xToBucket]);

  useEffect(() => { draw(); }, [draw]);

  const computeCursor = useCallback((x: number): string => {
    const px = selPx(ds.current.sel);
    if (px) {
      if (Math.abs(x - px.x1) < HANDLE_HIT || Math.abs(x - px.x2) < HANDLE_HIT) return "ew-resize";
      if (x > px.x1 && x < px.x2) return "grab";
    }
    return "crosshair";
  }, [selPx]);

  const showTooltip = useCallback((mx: number, my: number) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const idx = xToBucket(mx);
    if (idx < 0 || idx >= buckets.length) { setTooltip(null); return; }
    const b = buckets[idx];
    if (b.count === 0) { setTooltip(null); return; }
    const rect = canvas.getBoundingClientRect();
    setTooltip({
      x: rect.left + mx,
      y: rect.top  + my,
      lines: [
        `${fmtDatetime(b.t_start)} – ${fmtDatetime(b.t_end)}`,
        `${b.count.toLocaleString()} msgs`,
        ...(b.error_count ? [`${b.error_count} errors`]  : []),
        ...(b.warn_count  ? [`${b.warn_count} warnings`] : []),
      ],
    });
  }, [buckets, xToBucket]);

  function handleMouseDown(e: React.MouseEvent<HTMLCanvasElement>) {
    const rect = canvasRef.current!.getBoundingClientRect();
    const x = e.clientX - rect.left;
    if (x < PAD.left || x > canvasW - PAD.right) return;

    const st = ds.current;
    const px = selPx(st.sel);
    let mode: DragMode = "select";
    if (px) {
      if (Math.abs(x - px.x1) < HANDLE_HIT) mode = "resize-left";
      else if (Math.abs(x - px.x2) < HANDLE_HIT) mode = "resize-right";
      else if (x > px.x1 && x < px.x2) mode = "pan";
    }

    st.active = true; st.mode = mode; st.startX = x; st.currentX = x;
    st.selBeforeResize = st.sel ? ([...st.sel] as [number, number]) : null;
    if (mode === "select") { st.sel = null; setSelInfo(null); }
    setTooltip(null);
    setCursor(mode === "pan" ? "grabbing" : "ew-resize");
    draw();
  }

  function handleMouseMove(e: React.MouseEvent<HTMLCanvasElement>) {
    const rect = canvasRef.current!.getBoundingClientRect();
    const x    = e.clientX - rect.left;
    const st   = ds.current;

    if (!st.active) {
      const inChart = x >= PAD.left && x <= canvasW - PAD.right;
      st.hoverIdx = inChart ? xToBucket(x) : null;
      setCursor(computeCursor(x));
      if (inChart) showTooltip(x, e.clientY - rect.top);
      else setTooltip(null);
      draw();
      return;
    }

    setTooltip(null);
    const clamped  = Math.max(PAD.left, Math.min(x, canvasW - PAD.right));
    const dBuckets = Math.round((clamped - st.startX) / barW);

    if (st.mode === "resize-left" && st.selBeforeResize) {
      st.sel = [Math.max(0, Math.min(st.selBeforeResize[0] + dBuckets, st.selBeforeResize[1])), st.selBeforeResize[1]];
    } else if (st.mode === "resize-right" && st.selBeforeResize) {
      st.sel = [st.selBeforeResize[0], Math.max(st.selBeforeResize[0], Math.min(st.selBeforeResize[1] + dBuckets, buckets.length - 1))];
    } else if (st.mode === "pan" && st.selBeforeResize) {
      const w  = st.selBeforeResize[1] - st.selBeforeResize[0];
      const nl = Math.max(0, Math.min(st.selBeforeResize[0] + dBuckets, buckets.length - 1 - w));
      st.sel = [nl, nl + w];
    } else {
      st.currentX = clamped;
    }
    draw();
  }

  function commitSelection() {
    const st = ds.current;
    if (!st.active) return;
    st.active = false;
    if (st.mode === "select") {
      const x1 = Math.min(st.startX, st.currentX);
      const x2 = Math.max(st.startX, st.currentX);
      if (x2 - x1 > 5) {
        const i1 = xToBucket(x1), i2 = xToBucket(x2);
        st.sel = [i1, i2];
        fireSelection(i1, i2);
      }
    } else if (st.sel) {
      fireSelection(st.sel[0], st.sel[1]);
    }
    setCursor("crosshair");
    draw();
  }

  function fireSelection(i1: number, i2: number) {
    const dur = buckets[i2].t_end - buckets[i1].t_start;
    const cnt = buckets.slice(i1, i2 + 1).reduce((s, b) => s + b.count, 0);
    setSelInfo({
      label:     `${fmtDatetime(buckets[i1].t_start)} → ${fmtDatetime(buckets[i2].t_end)}`,
      secondary: `${fmtDuration(dur)} · ${cnt.toLocaleString()} msgs`,
    });
    onRangeSelect?.(buckets[i1].t_start, buckets[i2].t_end);
  }

  function clearSelection() {
    ds.current.sel = null;
    setSelInfo(null);
    setTooltip(null);
    draw();
  }

  if (buckets.length === 0) return null;

  const totalCount  = buckets.reduce((s, b) => s + b.count, 0);
  const totalErrors = buckets.reduce((s, b) => s + b.error_count, 0);

  return (
    <div ref={wrapRef} className="w-full rounded-lg overflow-visible"
         style={{ background: C_BG, border: "1px solid #2a2f3a" }}>

      {/* Header */}
      <div className="flex items-start justify-between px-4 pt-3 pb-2">
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium" style={{ color: "#d9dde3" }}>
            Log volume {bucketLabel}
          </span>
          <span className="relative flex h-2 w-2">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-60" />
            <span className="relative inline-flex rounded-full h-2 w-2 bg-green-400" />
          </span>
        </div>
        <div className="flex flex-col items-end leading-none">
          <span className="text-xs font-medium mb-1" style={{ color: "#8e9bb2" }}>Lines</span>
          <span className="font-bold tabular-nums" style={{ color: "#F2911A", fontSize: "2rem", lineHeight: 1 }}>
            {totalCount.toLocaleString()}
          </span>
          {totalErrors > 0 && (
            <span className="text-xs mt-1" style={{ color: C_ERROR }}>
              {totalErrors.toLocaleString()} errors
            </span>
          )}
        </div>
      </div>

      {/* Canvas */}
      <div className="relative overflow-visible">
        <canvas
          ref={canvasRef}
          style={{ cursor, display: "block", width: "100%" }}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={commitSelection}
          onMouseLeave={() => {
            if (ds.current.active) commitSelection();
            ds.current.hoverIdx = null;
            setTooltip(null);
            draw();
          }}
          onDoubleClick={clearSelection}
          className="select-none"
        />

        {tooltip && (
          <div
            className="fixed z-50 pointer-events-none px-3 py-2 rounded-md shadow-2xl text-xs"
            style={{
              left: tooltip.x + 16,
              top:  tooltip.y - 56,
              background: "#1c2128",
              border: "1px solid #3a4150",
              minWidth: 168,
            }}
          >
            {tooltip.lines.map((l, i) => (
              <div key={i} style={{ color: i === 0 ? "#d9dde3" : "#8e9bb2" }}
                   className={i === 0 ? "font-medium mb-1" : ""}>
                {l}
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Selection info */}
      {selInfo && (
        <div className="flex items-center gap-3 px-4 py-2 text-xs"
             style={{ background: "rgba(59,130,246,0.08)", borderTop: "1px solid rgba(99,155,255,0.22)" }}>
          <span className="font-semibold" style={{ color: "rgba(99,155,255,1)" }}>{selInfo.label}</span>
          <span style={{ color: "#8e9bb2" }}>{selInfo.secondary}</span>
          <button onClick={clearSelection} className="ml-auto transition-opacity hover:opacity-100 opacity-70"
                  style={{ color: "rgba(99,155,255,1)" }}>
            ✕ Clear
          </button>
        </div>
      )}

      {/* Legend */}
      <div className="flex items-center gap-5 px-4 py-2 text-xs"
           style={{ borderTop: "1px solid #1c2128", color: "#8e9bb2" }}>
        {[
          { color: C_GREEN, label: "Normal"   },
          { color: C_WARN,  label: "Warnings" },
          { color: C_ERROR, label: "Errors"   },
        ].map(({ color, label }) => (
          <span key={label} className="flex items-center gap-1.5">
            <svg width="16" height="4" viewBox="0 0 16 4">
              <line x1="0" y1="2" x2="16" y2="2" stroke={color} strokeWidth="2.5" />
            </svg>
            {label}
          </span>
        ))}
        <span className="ml-auto" style={{ color: "#454d5a", fontSize: "0.7rem" }}>
          Drag · resize edges · double-click to reset
        </span>
      </div>
    </div>
  );
}
