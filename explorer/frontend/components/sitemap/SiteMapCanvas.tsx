"use client";

import React, {
  useRef,
  useEffect,
  useCallback,
  useMemo,
  useState,
  forwardRef,
  useImperativeHandle,
} from "react";
import type {
  SiteMapMeta,
  SiteMapData,
  SiteMapSpot,
  SiteMapRack,
  SiteMapRegion,
  SiteMapMarker,
  SiteMapNode,
} from "@/lib/types";

// ── Types ──────────────────────────────────────────────────────────────────────

export interface Layers {
  spots: boolean;
  racks: boolean;
  regions: boolean;
  markers: boolean;
  nodes: boolean;
}

interface Transform {
  scale: number;
  panX: number;
  panY: number;
}

interface Props {
  meta: SiteMapMeta;
  data: SiteMapData;
  markers?: SiteMapMarker[];
  searchQuery?: string;
  layers: Layers;
  hiddenSpotTypes?: ReadonlySet<string>;
  hiddenRegionTypes?: ReadonlySet<string>;
  onSpotSelect?: (spot: SiteMapSpot | null) => void;
}

// ── Public handle ─────────────────────────────────────────────────────────────

export interface SiteMapCanvasHandle {
  /** Pan+zoom the map so (pixX, pixY) image-pixel point is centred in view. */
  panTo(pixX: number, pixY: number): void;
  /** Smoothly reset the map to its default full-map fit view. */
  fitMap(): void;
}

// ── Coordinate helpers ─────────────────────────────────────────────────────────

export function worldToPixel(
  wx: number,
  wy: number,
  oX: number,
  oY: number,
  res: number,
  imgH: number
): [number, number] {
  return [(wx - oX) / res, imgH - (wy - oY) / res];
}

// ── Main component ─────────────────────────────────────────────────────────────

const SiteMapCanvas = forwardRef<SiteMapCanvasHandle, Props>(function SiteMapCanvas({
  meta,
  data,
  markers = [],
  searchQuery = "",
  layers,
  hiddenSpotTypes,
  hiddenRegionTypes,
  onSpotSelect,
}: Props, ref) {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef    = useRef<HTMLCanvasElement>(null);
  const mapImgRef    = useRef<HTMLImageElement | null>(null);
  const imgLoadedRef = useRef(false);
  const transformRef = useRef<Transform>({ scale: 0.5, panX: 0, panY: 0 });
  const dragRef      = useRef<{ sx: number; sy: number; px: number; py: number } | null>(null);
  const hoveredRef         = useRef<SiteMapSpot | null>(null);
  const hoveredMarkerRef   = useRef<SiteMapMarker | null>(null);
  const hoveredRackRef     = useRef<{ rack: SiteMapRack; px: [number, number] } | null>(null);
  const hoveredRegionRef   = useRef<SiteMapRegion | null>(null);
  const hoveredNodeRef     = useRef<SiteMapNode | null>(null);
  const rafRef             = useRef<number>(0);
  const focusTargetRef     = useRef<{ px: number; py: number; startMs: number } | null>(null);
  const panAnimRef         = useRef<number>(0);

  // re-render trigger
  const [, forceUpdate] = useState(0);

  // ── Load map image ──────────────────────────────────────────────────────────

  useEffect(() => {
    if (!meta.b64) return;
    imgLoadedRef.current = false;
    const img = new Image();
    img.onload = () => {
      mapImgRef.current = img;
      imgLoadedRef.current = true;
      // fit-to-view on first load
      if (containerRef.current) {
        const cW = containerRef.current.clientWidth;
        const cH = containerRef.current.clientHeight;
        const s  = Math.min((cW / meta.width) * 0.95, (cH / meta.height) * 0.95, 2);
        transformRef.current = {
          scale: s,
          panX:  (cW - meta.width  * s) / 2,
          panY:  (cH - meta.height * s) / 2,
        };
      }
      forceUpdate(n => n + 1);
    };
    img.src = meta.b64;
  }, [meta.b64, meta.width, meta.height]);

  // ── Coordinate conversion ───────────────────────────────────────────────────

  const w2p = useCallback(
    (wx: number, wy: number): [number, number] =>
      worldToPixel(wx, wy, meta.origin[0], meta.origin[1], meta.resolution, meta.height),
    [meta]
  );

  // ── Re-fit view when container is resized (minimized / maximized) ───────────

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    let lastW = container.clientWidth;
    let lastH = container.clientHeight;

    const observer = new ResizeObserver(entries => {
      for (const entry of entries) {
        const { width: cW, height: cH } = entry.contentRect;
        // Ignore sub-10px jitter (e.g. scrollbar appearance)
        if (Math.abs(cW - lastW) < 10 && Math.abs(cH - lastH) < 10) continue;
        lastW = cW;
        lastH = cH;
        if (cW <= 0 || cH <= 0 || !imgLoadedRef.current) return;
        const s = Math.min((cW / meta.width) * 0.95, (cH / meta.height) * 0.95, 2);
        transformRef.current = {
          scale: s,
          panX:  (cW - meta.width  * s) / 2,
          panY:  (cH - meta.height * s) / 2,
        };
      }
    });

    observer.observe(container);
    return () => observer.disconnect();
  }, [meta.width, meta.height]);

  // ── Pre-compute spot pixel coords ───────────────────────────────────────────

  const spotPixels = useMemo(
    () => data.spots.map(s => ({ spot: s, px: w2p(s.x, s.y) })),
    [data.spots, w2p]
  );

  const rackPixels = useMemo(
    () => data.racks.map(r => ({ rack: r, px: w2p(r.x, r.y) as [number, number] })),
    [data.racks, w2p]
  );

  const regionPolygonPixels = useMemo(
    () => data.regions.map(region => ({
      region,
      polygon: region.polygon.map(([wx, wy]: [number, number]) => w2p(wx, wy) as [number, number]),
    })),
    [data.regions, w2p]
  );

  const nodePixels = useMemo(
    () => data.nodes.map(node => ({ node, px: w2p(node.x, node.y) as [number, number] })),
    [data.nodes, w2p]
  );

  const edgePixels = useMemo(() => {
    const byId = new Map<number, [number, number]>();
    nodePixels.forEach(({ node, px }) => byId.set(node.id, px));
    return data.edges
      .map((edge) => {
        const p1 = byId.get(edge.node1);
        const p2 = byId.get(edge.node2);
        if (!p1 || !p2) return null;
        return { edge, p1, p2 };
      })
      .filter((v): v is { edge: (typeof data.edges)[number]; p1: [number, number]; p2: [number, number] } => v !== null);
  }, [data, nodePixels]);

  // ── Draw function ───────────────────────────────────────────────────────────

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const container = containerRef.current;
    if (container) {
      const dpr = window.devicePixelRatio ?? 1;
      const cW  = container.clientWidth;
      const cH  = container.clientHeight;
      if (canvas.width !== cW * dpr || canvas.height !== cH * dpr) {
        canvas.width  = cW * dpr;
        canvas.height = cH * dpr;
        canvas.style.width  = `${cW}px`;
        canvas.style.height = `${cH}px`;
        ctx.scale(dpr, dpr);
      }
    }

    const { scale, panX, panY } = transformRef.current;
    const cW = canvas.width  / (window.devicePixelRatio ?? 1);
    const cH = canvas.height / (window.devicePixelRatio ?? 1);

    ctx.clearRect(0, 0, cW, cH);

    ctx.save();
    ctx.translate(panX, panY);
    ctx.scale(scale, scale);

    // ── Map image ────────────────────────────────────────────────────────
    if (mapImgRef.current && imgLoadedRef.current) {
      ctx.drawImage(mapImgRef.current, 0, 0, meta.width, meta.height);
    } else {
      ctx.fillStyle = "#f1f5f9";
      ctx.fillRect(0, 0, meta.width, meta.height);
    }

    // Shared search term used by all layers
    const csq = searchQuery.toLowerCase().trim();

    // Deferred tooltip queue — tooltips are collected here during element passes
    // and flushed after all element bodies are drawn so they always render on top.
    const tooltipQueue: Array<() => void> = [];

    // ── Regions ──────────────────────────────────────────────────────────
    if (layers.regions) {
      // Two-pass: fill first, then stroke+labels — so borders sit on top of fills
      // Pass 1: fills
      data.regions.forEach(region => {
        if (region.polygon.length < 3) return;
        if (hiddenRegionTypes?.has(region.type)) return;
        ctx.beginPath();
        region.polygon.forEach(([wx, wy]: [number, number], i: number) => {
          const [px, py] = w2p(wx, wy);
          if (i === 0) { ctx.moveTo(px, py); } else { ctx.lineTo(px, py); }
        });
        ctx.closePath();
        // Boost fill alpha ×2 for visibility on the light map
        ctx.fillStyle = region.color.replace(
          /rgba\((.+),\s*([\d.]+)\)/,
          (_: string, rgb: string, a: string) =>
            `rgba(${rgb},${Math.min(0.45, parseFloat(a) * 2)})`
        );
        ctx.fill();
      });

      // Pass 2: borders only (region text labels intentionally disabled)

      data.regions.forEach(region => {
        if (region.polygon.length < 3) return;
        if (hiddenRegionTypes?.has(region.type)) return;

        // Compute world bbox
        const xs = region.polygon.map(([x]: [number, number]) => x);
        const ys = region.polygon.map(([, y]: [number, number]) => y);

        const regionMatched = csq !== "" &&
          (region.name || region.type).toLowerCase() === csq;
        const isHoveredRegion = hoveredRegionRef.current === region;

        // Visible border
        ctx.beginPath();
        region.polygon.forEach(([wx, wy]: [number, number], i: number) => {
          const [px, py] = w2p(wx, wy);
          if (i === 0) { ctx.moveTo(px, py); } else { ctx.lineTo(px, py); }
        });
        ctx.closePath();
        if (regionMatched) {
          ctx.strokeStyle = "#facc15";
          ctx.lineWidth   = Math.max(2, 4 / scale);
        } else if (isHoveredRegion) {
          ctx.strokeStyle = "#ffffff";
          ctx.lineWidth   = Math.max(1.5, 2.5 / scale);
        } else {
          ctx.strokeStyle = region.color.replace(
            /rgba\((.+),\s*[\d.]+\)/,
            (_: string, rgb: string) => `rgba(${rgb},0.80)`
          );
          ctx.lineWidth = Math.max(0.8, 1.5 / scale);
        }
        ctx.stroke();

        // Centroid (used for hover tooltip)
        const cxW = xs.reduce((a: number, v: number) => a + v, 0) / xs.length;
        const cyW = ys.reduce((a: number, v: number) => a + v, 0) / ys.length;
        const [lx, ly] = w2p(cxW, cyW);

        // Hover tooltip — deferred so it renders above all elements
        if (isHoveredRegion) {
          const _lx = lx, _ly = ly;
          tooltipQueue.push(() => {
            type TooltipLine = [string, "header" | "data"];
            const headerText = region.name || region.type;
            const lines: TooltipLine[] = [
              [headerText, "header"],
              ...(region.name && region.name !== region.type
                ? [[region.type, "data"] as TooltipLine]
                : []),
            ];
            const fs = Math.max(8, 10 / scale);

            ctx.font = `600 ${fs}px ui-sans-serif,system-ui,sans-serif`;
            const headerW = ctx.measureText(lines[0][0]).width;
            ctx.font = `${fs}px ui-sans-serif,system-ui,sans-serif`;
            const dataW = lines.length > 1
              ? Math.max(...lines.slice(1).map(([l]) => ctx.measureText(l).width))
              : 0;
            const maxTw = Math.max(headerW, dataW);

            const padX  = 6 / scale;
            const padY  = 5 / scale;
            const lineH = fs + 3;
            const bx    = _lx + 8 / scale;
            const by    = _ly - (lines.length * lineH) / 2;
            const bw    = maxTw + padX * 2;
            const bh    = lines.length * lineH + padY * 2;

            ctx.fillStyle   = "rgba(10,15,28,0.92)";
            ctx.strokeStyle = "rgba(192,132,252,0.35)";
            ctx.lineWidth   = 1 / scale;
            ctx.beginPath();
            ctx.roundRect(bx - padX, by - padY, bw, bh, 4 / scale);
            ctx.fill();
            ctx.stroke();

            ctx.textBaseline = "top";
            lines.forEach(([text, type], li) => {
              ctx.font      = type === "header"
                ? `600 ${fs}px ui-sans-serif,system-ui,sans-serif`
                : `${fs}px ui-sans-serif,system-ui,sans-serif`;
              ctx.fillStyle = type === "header" ? "#d8b4fe" : "#94a3b8";
              ctx.fillText(text, bx, by + li * lineH);
            });
            ctx.textBaseline = "alphabetic";
          });
        }
      });
    }

    // ── Graph nodes/edges ────────────────────────────────────────────────
    if (layers.nodes) {
      const hoveredNode = hoveredNodeRef.current;

      // Draw edges first so nodes are always visible above lines.
      edgePixels.forEach(({ edge, p1, p2 }) => {
        const touchesHovered = !!hoveredNode && (edge.node1 === hoveredNode.id || edge.node2 === hoveredNode.id);
        ctx.beginPath();
        ctx.moveTo(p1[0], p1[1]);
        ctx.lineTo(p2[0], p2[1]);
        ctx.strokeStyle = touchesHovered ? "rgba(56,189,248,0.75)" : "rgba(148,163,184,0.28)";
        ctx.lineWidth = touchesHovered ? Math.max(1.5, 2.5 / scale) : Math.max(0.8, 1.2 / scale);
        ctx.stroke();
      });

      nodePixels.forEach(({ node, px: [px, py] }) => {
        const isHovered = hoveredNode?.id === node.id;
        const matched = csq !== "" && (String(node.id) === csq || `node ${node.id}` === csq);
        const r = (matched ? 7 : isHovered ? 6 : 4.5) / scale;

        ctx.beginPath();
        ctx.arc(px, py, r, 0, Math.PI * 2);
        if (matched) {
          ctx.fillStyle = "#facc15";
        } else if (node.parkable) {
          ctx.fillStyle = isHovered ? "#6ee7b7" : "#34d399";
        } else {
          ctx.fillStyle = isHovered ? "#93c5fd" : "#60a5fa";
        }
        ctx.fill();

        ctx.strokeStyle = matched
          ? "#f59e0b"
          : isHovered
          ? "#ffffff"
          : "rgba(255,255,255,0.55)";
        ctx.lineWidth = (matched || isHovered ? 2 : 1.3) / scale;
        ctx.stroke();

        if (isHovered || scale > 0.8) {
          const fs = Math.max(7, 9 / scale);
          const label = String(node.id);
          ctx.font = `600 ${fs}px ui-sans-serif,system-ui,sans-serif`;
          ctx.textAlign = "center";
          ctx.textBaseline = "middle";
          ctx.fillStyle = isHovered ? "#ffffff" : "rgba(15,23,42,0.92)";
          ctx.fillText(label, px, py);
          ctx.textAlign = "left";
          ctx.textBaseline = "alphabetic";
        }

        if (isHovered) {
          const _px = px, _py = py, _node = node, _r = r;
          tooltipQueue.push(() => {
            type TooltipLine = [string, "header" | "data"];
            const lines: TooltipLine[] = [
              [`Node ${_node.id}`, "header"],
              [`${_node.x.toFixed(3)} m, ${_node.y.toFixed(3)} m`, "data"],
              [`parkable: ${_node.parkable ? "yes" : "no"}`, "data"],
              [`radius: ${_node.radius.toFixed(3)} m`, "data"],
            ];
            const fs = Math.max(8, 10 / scale);

            ctx.font = `600 ${fs}px ui-sans-serif,system-ui,sans-serif`;
            const headerW = ctx.measureText(lines[0][0]).width;
            ctx.font = `${fs}px ui-sans-serif,system-ui,sans-serif`;
            const dataW = Math.max(...lines.slice(1).map(([l]) => ctx.measureText(l).width));
            const maxTw = Math.max(headerW, dataW);

            const padX = 6 / scale;
            const padY = 5 / scale;
            const lineH = fs + 3;
            const bx = _px + _r + 9 / scale;
            const by = _py - (lines.length * lineH) / 2;
            const bw = maxTw + padX * 2;
            const bh = lines.length * lineH + padY * 2;

            ctx.fillStyle = "rgba(10,15,28,0.92)";
            ctx.strokeStyle = "rgba(56,189,248,0.35)";
            ctx.lineWidth = 1 / scale;
            ctx.beginPath();
            ctx.roundRect(bx - padX, by - padY, bw, bh, 4 / scale);
            ctx.fill();
            ctx.stroke();

            ctx.textBaseline = "top";
            lines.forEach(([text, type], li) => {
              ctx.font = type === "header"
                ? `600 ${fs}px ui-sans-serif,system-ui,sans-serif`
                : `${fs}px ui-sans-serif,system-ui,sans-serif`;
              ctx.fillStyle = type === "header" ? "#7dd3fc" : "#94a3b8";
              ctx.fillText(text, bx, by + li * lineH);
            });
            ctx.textBaseline = "alphabetic";
          });
        }
      });
    }

    // ── Racks ─────────────────────────────────────────────────────────────
    if (layers.racks) {
      // Package icon SVG paths (Lucide Package, 24×24 viewport)
      const PKG_BODY = "M11 21.73a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73z";
      const PKG_VLINE = "M12 22V12";
      const PKG_SEAM  = "m3.3 7 7.703 4.734a2 2 0 0 0 1.994 0L20.7 7";
      const PKG_LID   = "m7.5 4.27 9 5.15";

      rackPixels.forEach(({ rack, px: [px, py] }) => {
        const rackMatched = csq !== "" &&
          (rack.label || `${rack.section}-${rack.row}`).toLowerCase() === csq;
        const isHovered = hoveredRackRef.current?.rack === rack;

        // Progressive detail: at very low zoom (minimized map), draw a small amber
        // dot instead of the full icon to prevent clutter.
        if (scale < 0.18 && !rackMatched) {
          const dr = 2.2 / scale;
          ctx.beginPath();
          ctx.arc(px, py, dr, 0, Math.PI * 2);
          ctx.fillStyle = "rgba(245,158,11,0.75)";
          ctx.fill();
          return;
        }

        // Target screen size: 11px normal, 13px hovered/matched → capped max
        const screenPx = rackMatched ? 13 : isHovered ? 12 : 11;
        // Transform from SVG (24×24) coords → canvas image-pixel coords
        const is = screenPx / (24 * scale);

        ctx.save();
        ctx.translate(px, py);
        ctx.scale(is, is);
        ctx.translate(-12, -12);  // centre the 24×24 icon

        // Body (hexagon)
        const bodyPath = new Path2D(PKG_BODY);
        ctx.fillStyle = rackMatched
          ? "#fef08a"
          : isHovered
          ? "#fcd34d"
          : "rgba(245,158,11,0.88)";
        ctx.fill(bodyPath);

        ctx.strokeStyle = rackMatched
          ? "#f59e0b"
          : isHovered
          ? "#ffffff"
          : "rgba(251,191,36,0.85)";
        ctx.lineWidth = rackMatched || isHovered ? 1.4 : 1.1;
        ctx.stroke(bodyPath);

        // Interior detail lines
        ctx.strokeStyle = rackMatched
          ? "rgba(120,60,0,0.70)"
          : isHovered
          ? "rgba(120,60,0,0.60)"
          : "rgba(120,60,0,0.55)";
        ctx.lineWidth = 1.1;
        ctx.lineCap  = "round";

        ctx.stroke(new Path2D(PKG_VLINE));
        ctx.stroke(new Path2D(PKG_SEAM));
        ctx.stroke(new Path2D(PKG_LID));

        ctx.restore();

        // Matched: glow ring (in image-pixel space, outside the save/restore block)
        if (rackMatched) {
          const glowR = (screenPx * 0.9) / scale;
          ctx.beginPath();
          ctx.arc(px, py, glowR, 0, Math.PI * 2);
          ctx.strokeStyle = "rgba(251,191,36,0.30)";
          ctx.lineWidth   = 3 / scale;
          ctx.stroke();
        }

        // Hover tooltip — deferred so it renders above all elements
        if (isHovered) {
          const _px = px, _py = py, _rack = rack, _screenPx = screenPx;
          tooltipQueue.push(() => {
            const label = _rack.label || `${_rack.section}-${_rack.row}`;
            type TooltipLine = [string, "header" | "data"];
            const lines: TooltipLine[] = [
              [label, "header"],
              [`${_rack.section} / row ${_rack.row}`, "data"],
              [`${_rack.x.toFixed(2)} m, ${_rack.y.toFixed(2)} m`, "data"],
            ];
            const fs = Math.max(8, 10 / scale);

            ctx.font = `600 ${fs}px ui-sans-serif,system-ui,sans-serif`;
            const headerW = ctx.measureText(lines[0][0]).width;
            ctx.font = `${fs}px ui-sans-serif,system-ui,sans-serif`;
            const dataW = Math.max(...lines.slice(1).map(([l]) => ctx.measureText(l).width));
            const maxTw = Math.max(headerW, dataW);

            const padX  = 6 / scale;
            const padY  = 5 / scale;
            const lineH = fs + 3;
            const ttX   = _px + (_screenPx * 0.6) / scale + 9 / scale;
            const ttY   = _py - (lines.length * lineH) / 2;
            const bw    = maxTw + padX * 2;
            const bh    = lines.length * lineH + padY * 2;

            ctx.fillStyle   = "rgba(10,15,28,0.92)";
            ctx.strokeStyle = "rgba(251,191,36,0.35)";
            ctx.lineWidth   = 1 / scale;
            ctx.beginPath();
            ctx.roundRect(ttX - padX, ttY - padY, bw, bh, 4 / scale);
            ctx.fill();
            ctx.stroke();

            ctx.textBaseline = "top";
            lines.forEach(([text, type], li) => {
              ctx.font      = type === "header"
                ? `600 ${fs}px ui-sans-serif,system-ui,sans-serif`
                : `${fs}px ui-sans-serif,system-ui,sans-serif`;
              ctx.fillStyle = type === "header" ? "#fcd34d" : "#94a3b8";
              ctx.fillText(text, ttX, ttY + li * lineH);
            });
            ctx.textBaseline = "alphabetic";
          });
        }
      });
    }

    // ── Spots ─────────────────────────────────────────────────────────────
    if (layers.spots) {
      spotPixels.forEach(({ spot, px: [px, py] }) => {
        if (hiddenSpotTypes?.has(spot.type)) return;
        const matched = csq !== "" &&
          (spot.name.toLowerCase() === csq || String(spot._idx) === csq);
        const isHovered = hoveredRef.current?._idx === spot._idx;
        // Base radius: always ≥5px on screen; larger for selected/matched
        const r = (matched ? 8 : isHovered ? 7 : 5) / scale;

        ctx.beginPath();
        ctx.arc(px, py, r, 0, Math.PI * 2);
        ctx.fillStyle = matched ? "#facc15" : spot.color;
        ctx.fill();

        // Always draw outline — makes spots pop on dark map background
        ctx.strokeStyle = matched
          ? "#fbbf24"
          : isHovered
          ? "#ffffff"
          : "rgba(255,255,255,0.45)";
        ctx.lineWidth = (matched || isHovered ? 2 : 1.5) / scale;
        ctx.stroke();

        // Direction yaw indicator (finger) when zoomed in
        if (scale > 0.7) {
          ctx.beginPath();
          ctx.moveTo(px, py);
          ctx.lineTo(
            px + Math.cos(spot.yaw) * r * 2.2,
            py - Math.sin(spot.yaw) * r * 2.2
          );
          ctx.strokeStyle = matched || isHovered ? "#ffffff" : "rgba(255,255,255,0.5)";
          ctx.lineWidth   = 1.5 / scale;
          ctx.stroke();
        }

        // Tooltip label for hovered / matched and zoomed in enough
        if (isHovered) {
          const _px = px, _py = py, _spot = spot, _r = r;
          tooltipQueue.push(() => {
            type TooltipLine = [string, "header" | "data"];
            const lines: TooltipLine[] = [
              [_spot.name, "header"],
              [_spot.type, "data"],
              [`${_spot.x.toFixed(2)} m, ${_spot.y.toFixed(2)} m`, "data"],
            ];
            const fs = Math.max(8, 10 / scale);

            ctx.font = `600 ${fs}px ui-sans-serif,system-ui,sans-serif`;
            const headerW = ctx.measureText(lines[0][0]).width;
            ctx.font = `${fs}px ui-sans-serif,system-ui,sans-serif`;
            const dataW = Math.max(...lines.slice(1).map(([l]) => ctx.measureText(l).width));
            const maxTw = Math.max(headerW, dataW);

            const padX  = 6 / scale;
            const padY  = 5 / scale;
            const lineH = fs + 3;
            const bx    = _px + _r + 9 / scale;
            const by    = _py - (lines.length * lineH) / 2;
            const bw    = maxTw + padX * 2;
            const bh    = lines.length * lineH + padY * 2;

            ctx.fillStyle   = "rgba(10,15,28,0.92)";
            ctx.strokeStyle = "rgba(96,165,250,0.35)";
            ctx.lineWidth   = 1 / scale;
            ctx.beginPath();
            ctx.roundRect(bx - padX, by - padY, bw, bh, 4 / scale);
            ctx.fill();
            ctx.stroke();

            ctx.textBaseline = "top";
            lines.forEach(([text, type], li) => {
              ctx.font      = type === "header"
                ? `600 ${fs}px ui-sans-serif,system-ui,sans-serif`
                : `${fs}px ui-sans-serif,system-ui,sans-serif`;
              ctx.fillStyle = type === "header" ? "#93c5fd" : "#94a3b8";
              ctx.fillText(text, bx, by + li * lineH);
            });
            ctx.textBaseline = "alphabetic";
          });
        } else if (matched && scale > 0.4) {
          const fs   = Math.max(8, 12 / scale);
          const text = spot.name;
          ctx.font      = `bold ${fs}px sans-serif`;
          const tw      = ctx.measureText(text).width;
          const bx      = px + r + 4 / scale;
          const by      = py - fs * 0.4;
          ctx.fillStyle = "rgba(15,23,42,0.88)";
          ctx.fillRect(bx - 2 / scale, by - fs, tw + 4 / scale, fs + 4 / scale);
          ctx.fillStyle = "#f8fafc";
          ctx.textBaseline = "middle";
          ctx.fillText(text, bx, by - fs * 0.2);
          ctx.textBaseline = "alphabetic";
        }
      });
    }
    // ── AR Markers ─────────────────────────────────────────────────────────
    if (layers.markers && markers.length > 0) {
      const r = Math.max(3, 6 / scale);  // body radius in image-pixel space

      markers.forEach(m => {
        const [px, py] = w2p(m.x, m.y);
        const isHovered = hoveredMarkerRef.current?.id === m.id;

        // Soft glow halo on hover
        if (isHovered) {
          ctx.beginPath();
          ctx.arc(px, py, r * 3, 0, Math.PI * 2);
          ctx.fillStyle = "rgba(239,68,68,0.10)";
          ctx.fill();
        }

        // Direction arrow (drawn behind body, rotated by yaw)
        ctx.save();
        ctx.translate(px, py);
        ctx.rotate(-m.yaw);  // yaw CCW from east; canvas Y is flipped

        const arrowLen = r * 2.4;
        const ahHW     = r * 0.50;  // arrowhead half-width
        const ahL      = r * 0.60;  // arrowhead length

        ctx.strokeStyle = isHovered ? "#f87171" : "rgba(239,68,68,0.75)";
        ctx.fillStyle   = isHovered ? "#f87171" : "rgba(239,68,68,0.75)";
        ctx.lineWidth   = 1.2 / scale;
        ctx.lineCap     = "round";

        // Shaft from body edge to arrowhead base
        ctx.beginPath();
        ctx.moveTo(r + 0.5 / scale, 0);
        ctx.lineTo(arrowLen - ahL, 0);
        ctx.stroke();

        // Arrowhead triangle
        ctx.beginPath();
        ctx.moveTo(arrowLen, 0);
        ctx.lineTo(arrowLen - ahL,  ahHW);
        ctx.lineTo(arrowLen - ahL, -ahHW);
        ctx.closePath();
        ctx.fill();

        ctx.restore();

        // Body circle
        ctx.beginPath();
        ctx.arc(px, py, r, 0, Math.PI * 2);
        ctx.fillStyle = isHovered ? "rgba(239,68,68,0.22)" : "rgba(239,68,68,0.10)";
        ctx.fill();

        ctx.strokeStyle = isHovered ? "#f87171" : "rgba(239,68,68,0.80)";
        ctx.lineWidth   = (isHovered ? 1.6 : 1.1) / scale;
        ctx.stroke();

        // Centre dot — anchors the marker position precisely
        ctx.beginPath();
        ctx.arc(px, py, Math.max(1, r * 0.30), 0, Math.PI * 2);
        ctx.fillStyle = isHovered ? "#f87171" : "rgba(239,68,68,0.85)";
        ctx.fill();

        // ID label — only when zoomed in or hovered
        if (scale > 0.45 || isHovered) {
          const fs  = Math.max(6, Math.min(10, 9 / scale));
          const lbl = String(m.id);
          ctx.font      = `600 ${fs}px ui-sans-serif,system-ui,sans-serif`;
          ctx.textAlign = "center";

          const tw   = ctx.measureText(lbl).width;
          const padX = 2.5 / scale;
          const padY = 1.5 / scale;
          const lx   = px - tw / 2 - padX;
          const ly   = py - r - 5 / scale - fs - padY * 2;
          ctx.fillStyle = "rgba(10,15,28,0.72)";
          ctx.beginPath();
          ctx.roundRect(lx, ly, tw + padX * 2, fs + padY * 2, 2 / scale);
          ctx.fill();

          ctx.textBaseline = "top";
          ctx.fillStyle    = isHovered ? "#fca5a5" : "#fda4af";
          ctx.fillText(lbl, px, ly + padY);
          ctx.textBaseline = "alphabetic";
          ctx.textAlign    = "left";
        }

        // Hover tooltip — deferred so it renders above all elements
        if (isHovered) {
          const _px = px, _py = py, _m = m;
          tooltipQueue.push(() => {
            const yawDeg = (_m.yaw * 180 / Math.PI).toFixed(1);
            type TooltipLine = [string, "header" | "data"];
            const lines: TooltipLine[] = [
              [`Marker ${_m.id}`, "header"],
              [`${_m.x.toFixed(2)} m, ${_m.y.toFixed(2)} m`, "data"],
              [`yaw ${yawDeg}°`, "data"],
            ];
            const fs = Math.max(8, 10 / scale);

            ctx.font = `600 ${fs}px ui-sans-serif,system-ui,sans-serif`;
            const headerW = ctx.measureText(lines[0][0]).width;
            ctx.font = `${fs}px ui-sans-serif,system-ui,sans-serif`;
            const dataW  = Math.max(...lines.slice(1).map(([l]) => ctx.measureText(l).width));
            const maxTw  = Math.max(headerW, dataW);

            const padX  = 6 / scale;
            const padY  = 5 / scale;
            const lineH = fs + 3;
            const bx    = _px + r + 9 / scale;
            const by    = _py - (lines.length * lineH) / 2;
            const bw    = maxTw + padX * 2;
            const bh    = lines.length * lineH + padY * 2;

            ctx.fillStyle   = "rgba(10,15,28,0.92)";
            ctx.strokeStyle = "rgba(239,68,68,0.35)";
            ctx.lineWidth   = 1 / scale;
            ctx.beginPath();
            ctx.roundRect(bx - padX, by - padY, bw, bh, 4 / scale);
            ctx.fill();
            ctx.stroke();

            ctx.textBaseline = "top";
            lines.forEach(([text, type], li) => {
              ctx.font      = type === "header"
                ? `600 ${fs}px ui-sans-serif,system-ui,sans-serif`
                : `${fs}px ui-sans-serif,system-ui,sans-serif`;
              ctx.fillStyle = type === "header" ? "#fca5a5" : "#94a3b8";
              ctx.fillText(text, bx, by + li * lineH);
            });
            ctx.textBaseline = "alphabetic";
          });
        }
      });
    }

    // ── Flush deferred tooltip queue (always on top of all map elements) ──
    tooltipQueue.forEach(fn => fn());

    ctx.restore();

    // ── Focus pulse (shown when panTo is called from search) ──────────────
    {
      const ft = focusTargetRef.current;
      if (ft) {
        const elapsed  = (performance.now() - ft.startMs) / 1000;
        const lifetime = 2.8;
        if (elapsed < lifetime) {
          const { scale, panX, panY } = transformRef.current;
          const sx = ft.px * scale + panX;
          const sy = ft.py * scale + panY;
          const t     = elapsed / lifetime;
          const alpha = Math.pow(1 - t, 1.5); // ease-out fade

          ctx.save();

          // Outer expanding ring
          ctx.beginPath();
          ctx.arc(sx, sy, 20 + t * 60, 0, Math.PI * 2);
          ctx.strokeStyle = `rgba(250, 204, 21, ${alpha * 0.55})`;
          ctx.lineWidth   = 2;
          ctx.stroke();

          // Inner expanding ring (brighter)
          ctx.beginPath();
          ctx.arc(sx, sy, 10 + t * 40, 0, Math.PI * 2);
          ctx.strokeStyle = `rgba(250, 204, 21, ${alpha * 0.9})`;
          ctx.lineWidth   = 3;
          ctx.stroke();

          // Centre filled dot
          ctx.beginPath();
          ctx.arc(sx, sy, 5, 0, Math.PI * 2);
          ctx.fillStyle = `rgba(250, 204, 21, ${Math.max(0.3, alpha)})`;
          ctx.shadowColor = "#facc15";
          ctx.shadowBlur  = 12;
          ctx.fill();
          ctx.shadowBlur  = 0;

          // Crosshair lines
          ctx.strokeStyle = `rgba(250, 204, 21, ${alpha * 0.75})`;
          ctx.lineWidth   = 1.5;
          ctx.beginPath();
          ctx.moveTo(sx - 22, sy);
          ctx.lineTo(sx + 22, sy);
          ctx.moveTo(sx, sy - 22);
          ctx.lineTo(sx, sy + 22);
          ctx.stroke();

          ctx.restore();
        } else {
          focusTargetRef.current = null;
        }
      }
    }

    // ── Minimap compass ───────────────────────────────────────────────────
    {
      const margin = 12;
      const size   = 32;
      ctx.save();
      ctx.globalAlpha = 0.7;
      ctx.fillStyle   = "#1e293b";
      ctx.beginPath();
      ctx.roundRect(cW - size - margin, margin, size, size, 6);
      ctx.fill();
      ctx.fillStyle = "#94a3b8";
      ctx.font      = "10px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("N", cW - size / 2 - margin, margin + 12);
      ctx.restore();
    }

    // ── Scale bar ────────────────────────────────────────────────────────
    {
      const scaleMeters = 10;
      const scalePixels = (scaleMeters / meta.resolution) * scale;
      const bx = 16, by = canvas.height / (window.devicePixelRatio ?? 1) - 18;
      ctx.save();
      ctx.strokeStyle = "#94a3b8";
      ctx.lineWidth   = 2;
      ctx.beginPath();
      ctx.moveTo(bx, by);
      ctx.lineTo(bx + scalePixels, by);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(bx, by - 4);
      ctx.lineTo(bx, by + 4);
      ctx.moveTo(bx + scalePixels, by - 4);
      ctx.lineTo(bx + scalePixels, by + 4);
      ctx.stroke();
      ctx.fillStyle = "#94a3b8";
      ctx.font      = "10px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(`${scaleMeters}m`, bx + scalePixels / 2, by - 6);
      ctx.restore();
    }
  }, [
    meta,
    data,
    markers,
    searchQuery,
    layers,
    spotPixels,
    rackPixels,
    nodePixels,
    edgePixels,
    w2p,
    hiddenSpotTypes,
    hiddenRegionTypes,
  ]);

  // ── Animation loop ──────────────────────────────────────────────────────────

  useEffect(() => {
    const loop = () => {
      draw();
      rafRef.current = requestAnimationFrame(loop);
    };
    rafRef.current = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(rafRef.current);
  }, [draw]);

  // ── Mouse / wheel interactions ──────────────────────────────────────────────

  const onWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault();
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect   = canvas.getBoundingClientRect();
    const mx     = e.clientX - rect.left;
    const my     = e.clientY - rect.top;
    const factor = e.deltaY < 0 ? 1.12 : 1 / 1.12;
    transformRef.current = (() => {
      const { scale, panX, panY } = transformRef.current;
      const ns = Math.min(15, Math.max(0.04, scale * factor));
      const sf = ns / scale;
      return { scale: ns, panX: mx - sf * (mx - panX), panY: my - sf * (my - panY) };
    })();
    forceUpdate(n => n + 1);
  }, []);

  const onMouseDown = useCallback((e: React.MouseEvent) => {
    dragRef.current = {
      sx: e.clientX,
      sy: e.clientY,
      px: transformRef.current.panX,
      py: transformRef.current.panY,
    };
  }, []);

  const onMouseMove = useCallback(
    (e: React.MouseEvent) => {
      const canvas = canvasRef.current;
      if (!canvas) return;

      // Pan
      if (dragRef.current) {
        const dx = e.clientX - dragRef.current.sx;
        const dy = e.clientY - dragRef.current.sy;
        transformRef.current = {
          ...transformRef.current,
          panX: dragRef.current.px + dx,
          panY: dragRef.current.py + dy,
        };
      }

      // Hover hit-test (all element types)
      {
        const rect   = canvas.getBoundingClientRect();
        const { scale, panX, panY } = transformRef.current;
        const mx = (e.clientX - rect.left - panX) / scale;
        const my = (e.clientY - rect.top  - panY) / scale;

        // Spots
        if (layers.spots) {
          const hitR = Math.max(6, 8 / scale);
          const hit  = spotPixels.find(({ px: [px, py] }) => Math.hypot(mx - px, my - py) < hitR);
          hoveredRef.current = hit ? hit.spot : null;
        } else {
          hoveredRef.current = null;
        }

        // Racks
        if (layers.racks) {
          const hitR = Math.max(5, 8 / scale);
          const hit  = rackPixels.find(({ px: [px, py] }) => Math.hypot(mx - px, my - py) < hitR);
          hoveredRackRef.current = hit ?? null;
        } else {
          hoveredRackRef.current = null;
        }

        // Regions (point-in-polygon ray-casting)
        if (layers.regions) {
          let hitRegion: SiteMapRegion | null = null;
          for (const { region, polygon } of regionPolygonPixels) {
            if (polygon.length < 3) continue;
            // Ray-casting algorithm
            let inside = false;
            for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
              const [xi, yi] = polygon[i];
              const [xj, yj] = polygon[j];
              const intersect =
                yi > my !== yj > my &&
                mx < ((xj - xi) * (my - yi)) / (yj - yi) + xi;
              if (intersect) inside = !inside;
            }
            if (inside) { hitRegion = region; break; }
          }
          hoveredRegionRef.current = hitRegion;
        } else {
          hoveredRegionRef.current = null;
        }

        // Markers
        if (layers.markers && markers.length > 0) {
          const hitR = Math.max(6, 14 / scale);
          const hit  = markers.find(m => {
            const [px, py] = w2p(m.x, m.y);
            return Math.hypot(mx - px, my - py) < hitR;
          });
          hoveredMarkerRef.current = hit ?? null;
        } else {
          hoveredMarkerRef.current = null;
        }

        // Nodes
        if (layers.nodes) {
          const hitR = Math.max(6, 10 / scale);
          const hit = nodePixels.find(({ px: [px, py] }) => Math.hypot(mx - px, my - py) < hitR);
          hoveredNodeRef.current = hit ? hit.node : null;
        } else {
          hoveredNodeRef.current = null;
        }

        const anyHit =
          hoveredRef.current     ||
          hoveredRackRef.current ||
          hoveredRegionRef.current ||
          hoveredMarkerRef.current ||
          hoveredNodeRef.current;
        canvas.style.cursor = anyHit ? "pointer" : dragRef.current ? "grabbing" : "grab";
      }
    },
    [
      layers.spots, layers.racks, layers.regions, layers.markers, layers.nodes,
      spotPixels, rackPixels, regionPolygonPixels, markers, nodePixels, w2p,
    ]
  );

  const onMouseUp = useCallback(
    (e: React.MouseEvent) => {
      const wasDrag =
        dragRef.current &&
        (Math.abs(e.clientX - dragRef.current.sx) > 4 ||
          Math.abs(e.clientY - dragRef.current.sy) > 4);
      dragRef.current = null;
      if (!wasDrag && hoveredRef.current) {
        onSpotSelect?.(hoveredRef.current);
      }
    },
    [onSpotSelect]
  );

  const onMouseLeave = useCallback(() => {
    dragRef.current          = null;
    hoveredRef.current       = null;
    hoveredMarkerRef.current = null;
    hoveredRackRef.current   = null;
    hoveredRegionRef.current = null;
    hoveredNodeRef.current   = null;
  }, []);

  // ── Zoom controls ──────────────────────────────────────────────────────────

  const zoom = useCallback((direction: "in" | "out" | "fit") => {
    if (direction === "fit") {
      if (containerRef.current) {
        const cW = containerRef.current.clientWidth;
        const cH = containerRef.current.clientHeight;
        const s  = Math.min((cW / meta.width) * 0.95, (cH / meta.height) * 0.95, 2);
        transformRef.current = {
          scale: s,
          panX:  (cW - meta.width  * s) / 2,
          panY:  (cH - meta.height * s) / 2,
        };
        forceUpdate(n => n + 1);
      }
    } else {
      const f = direction === "in" ? 1.25 : 1 / 1.25;
      const { scale, panX, panY } = transformRef.current;
      const ns = Math.min(15, Math.max(0.04, scale * f));
      const canvas = canvasRef.current;
      if (canvas) {
        const cx = canvas.clientWidth  / 2;
        const cy = canvas.clientHeight / 2;
        const sf = ns / scale;
        transformRef.current = { scale: ns, panX: cx - sf * (cx - panX), panY: cy - sf * (cy - panY) };
      }
      forceUpdate(n => n + 1);
    }
  }, [meta.width, meta.height]);

  // ── Imperative panTo ────────────────────────────────────────────────────

  const panTo = useCallback((pixX: number, pixY: number) => {
    const container = containerRef.current;
    if (!container) return;
    cancelAnimationFrame(panAnimRef.current);

    const cW = container.clientWidth;
    const cH = container.clientHeight;

    // Slight zoom-in: bring scale up to at least 2.5 if currently below;
    // never zoom out from where the user already is.
    const { scale: s0, panX: px0, panY: py0 } = transformRef.current;
    const ts = Math.max(s0, 2.5);

    const targetPanX = cW / 2 - pixX * ts;
    const targetPanY = cH / 2 - pixY * ts;

    const duration = 450; // ms
    const startTime = performance.now();

    const animate = (now: number) => {
      const raw  = Math.min(1, (now - startTime) / duration);
      const ease = 1 - Math.pow(1 - raw, 3); // ease-out cubic
      transformRef.current = {
        scale: s0 + (ts - s0) * ease,
        panX:  px0 + (targetPanX - px0) * ease,
        panY:  py0 + (targetPanY - py0) * ease,
      };
      forceUpdate(n => n + 1);
      if (raw < 1) panAnimRef.current = requestAnimationFrame(animate);
    };
    panAnimRef.current = requestAnimationFrame(animate);

    // Start pulse immediately so it's visible as the map arrives
    focusTargetRef.current = { px: pixX, py: pixY, startMs: performance.now() };
  }, []);

  // ── Imperative fitMap (animated reset to full-map view) ──────────────────

  const fitMap = useCallback(() => {
    const container = containerRef.current;
    if (!container) return;
    cancelAnimationFrame(panAnimRef.current);
    focusTargetRef.current = null;

    const cW = container.clientWidth;
    const cH = container.clientHeight;
    const targetScale = Math.min((cW / meta.width) * 0.95, (cH / meta.height) * 0.95, 2);
    const targetPanX  = (cW - meta.width  * targetScale) / 2;
    const targetPanY  = (cH - meta.height * targetScale) / 2;

    const { scale: s0, panX: px0, panY: py0 } = transformRef.current;
    const duration = 450;
    const startTime = performance.now();

    const animate = (now: number) => {
      const raw  = Math.min(1, (now - startTime) / duration);
      const ease = 1 - Math.pow(1 - raw, 3); // ease-out cubic
      transformRef.current = {
        scale: s0 + (targetScale - s0) * ease,
        panX:  px0 + (targetPanX  - px0) * ease,
        panY:  py0 + (targetPanY  - py0) * ease,
      };
      forceUpdate(n => n + 1);
      if (raw < 1) panAnimRef.current = requestAnimationFrame(animate);
    };
    panAnimRef.current = requestAnimationFrame(animate);
  }, [meta.width, meta.height]);

  useImperativeHandle(ref, () => ({ panTo, fitMap }), [panTo, fitMap]);

  return (
    <div ref={containerRef} className="relative w-full h-full bg-[#0a0f1a] overflow-hidden select-none">
      <canvas
        ref={canvasRef}
        className="absolute inset-0"
        style={{ cursor: "grab" }}
        onWheel={onWheel}
        onMouseDown={onMouseDown}
        onMouseMove={onMouseMove}
        onMouseUp={onMouseUp}
        onMouseLeave={onMouseLeave}
      />

      {/* Zoom controls */}
      <div className="absolute bottom-8 right-3 flex flex-col gap-1 z-10">
        {[
          { label: "+",  action: "in"  as const, title: "Zoom in"  },
          { label: "−",  action: "out" as const, title: "Zoom out" },
          { label: "⊡",  action: "fit" as const, title: "Fit map"  },
        ].map(({ label, action, title }) => (
          <button
            key={action}
            title={title}
            onClick={() => zoom(action)}
            className="w-7 h-7 rounded bg-[#1e293b]/90 border border-white/10 text-slate-300 hover:text-white hover:bg-slate-700 text-sm font-bold flex items-center justify-center transition-colors"
          >
            {label}
          </button>
        ))}
      </div>

      {/* Map not loaded placeholder */}
      {!imgLoadedRef.current && (
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
          <span className="text-slate-500 text-sm">Loading map…</span>
        </div>
      )}
    </div>
  );
});

export default SiteMapCanvas;
