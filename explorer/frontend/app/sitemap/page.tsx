"use client";

import { useEffect, useState, useCallback, useRef, useMemo } from "react";
import {
  listSiteMapSites,
  getSiteMapMeta,
  getSiteMapData,
  getSiteMarkers,
  getSiteBranchInfo,
  setSiteBranch,
  clearSiteBranch,
  syncSiteRepo,
  getBranchCleanupPlan,
  runBranchCleanup,
} from "@/lib/api";
import type {
  SiteMapMeta,
  SiteMapData,
  SiteMapSpot,
  SiteMapMarker,
  BranchInfo,
  BranchCleanupPlan,
  BranchCleanupResult,
} from "@/lib/types";
import SiteMapCanvas, { type Layers, type SiteMapCanvasHandle, worldToPixel } from "@/components/sitemap/SiteMapCanvas";
import {
  Map,
  Search,
  Layers as LayersIcon,
  ChevronDown,
  Loader2,
  Package,
  MapPin,
  Zap,
  Navigation,
  X,
  RefreshCw,
  Bot,
  GitBranch,
  Check,
  Trash2,
  AlertTriangle,
} from "lucide-react";
import clsx from "clsx";

// ── Legend config ──────────────────────────────────────────────────────────────

// Spot types that actually appear in the data
const SPOT_LEGEND = [
  { type: "action_spot",    color: "#3b82f6", label: "Action Spot"  },
  { type: "waiting_spot",   color: "#a855f7", label: "Waiting Spot" },
  { type: "charging_spot",  color: "#22c55e", label: "Charging Spot"},
  { type: "loading_spot",   color: "#eab308", label: "Loading Spot" },
  { type: "unloading_spot", color: "#f97316", label: "Unloading"    },
  { type: "exception_spot", color: "#ef4444", label: "Exception"    },
  { type: "transport_spot", color: "#06b6d4", label: "Transport"    },
  { type: "idle_spot",      color: "#64748b", label: "Idle Spot"    },
];

const REGION_LEGEND = [
  { type: "aisle",               color: "rgba(59,130,246,0.10)",  label: "Aisle"         },
  { type: "loading",             color: "rgba(234,179,8,0.22)",   label: "Loading"       },
  { type: "unloading",           color: "rgba(34,197,94,0.22)",   label: "Unloading"     },
  { type: "exception_unloading", color: "rgba(239,68,68,0.22)",   label: "Exception"     },
  { type: "idle",                color: "rgba(100,116,139,0.28)", label: "Idle Zone"     },
  { type: "charging",            color: "rgba(34,197,94,0.28)",   label: "Charging Zone" },
  { type: "replenishment",       color: "rgba(251,146,60,0.22)",  label: "Replenishment" },
];

// ── Page ───────────────────────────────────────────────────────────────────────

export default function SiteMapPage() {
  // Site state
  const [sites,    setSites]    = useState<{ id: string; name: string }[]>([]);
  const [siteId,   setSiteId]   = useState("");
  const [meta,     setMeta]     = useState<SiteMapMeta | null>(null);
  const [mapData,  setMapData]  = useState<SiteMapData | null>(null);
  const [markers,  setMarkers]  = useState<SiteMapMarker[]>([]);
  const [loading,  setLoading]  = useState(false);
  const [mapErr,   setMapErr]   = useState("");

  // UI
  const [inputText,     setInputText]     = useState("");
  const [searchQuery,   setSearchQuery]   = useState(""); // committed on click – drives canvas
  const [layers,        setLayers]        = useState<Layers>({
    spots:      true,
    racks:      true,
    regions:    true,
    markers:    true,
    nodes:      true,
  });
  const [hiddenSpotTypes,   setHiddenSpotTypes]   = useState<Set<string>>(new Set());
  const [hiddenRegionTypes, setHiddenRegionTypes] = useState<Set<string>>(new Set());
  const [selectedSpot,      setSelectedSpot]      = useState<SiteMapSpot | null>(null);
  const [showSearchDropdown, setShowSearchDropdown] = useState(false);
  const searchRef       = useRef<HTMLDivElement>(null);
  const canvasRef       = useRef<SiteMapCanvasHandle>(null);

  // Site selector dropdown
  const [showSiteDropdown, setShowSiteDropdown] = useState(false);
  const siteDropdownRef = useRef<HTMLDivElement>(null);

  // Branch state
  const [branchInfo,         setBranchInfo]         = useState<BranchInfo | null>(null);
  const [showBranchDropdown, setShowBranchDropdown] = useState(false);
  const [syncing,            setSyncing]            = useState(false);
  const branchDropdownRef = useRef<HTMLDivElement>(null);

  // Sidebar width state (resizable)
  const [sidebarWidth, setSidebarWidth] = useState(256);
  const sidebarResizeRef = useRef<{ startX: number; origW: number } | null>(null);

  const onSidebarResizeStart = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    sidebarResizeRef.current = { startX: e.clientX, origW: sidebarWidth };
    function onMove(ev: MouseEvent) {
      if (!sidebarResizeRef.current) return;
      const newW = Math.min(480, Math.max(180, sidebarResizeRef.current.origW + (ev.clientX - sidebarResizeRef.current.startX)));
      setSidebarWidth(newW);
    }
    function onUp() {
      sidebarResizeRef.current = null;
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    }
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
  }, [sidebarWidth]);

  // Branch cleanup modal state
  const [showCleanupModal,  setShowCleanupModal]  = useState(false);
  const [cleanupPlan,       setCleanupPlan]       = useState<BranchCleanupPlan | null>(null);
  const [cleanupResult,     setCleanupResult]     = useState<BranchCleanupResult | null>(null);
  const [cleanupLoading,    setCleanupLoading]    = useState(false);
  const [cleanupPlanLoading,setCleanupPlanLoading]= useState(false);
  const [cleanupErr,        setCleanupErr]        = useState("");

  // ── Close site dropdown on outside click ───────────────────────────────
  useEffect(() => {
    function handleOutside(e: MouseEvent) {
      if (siteDropdownRef.current && !siteDropdownRef.current.contains(e.target as Node)) {
        setShowSiteDropdown(false);
      }
    }
    if (showSiteDropdown) {
      document.addEventListener("mousedown", handleOutside);
      return () => document.removeEventListener("mousedown", handleOutside);
    }
  }, [showSiteDropdown]);

  // ── Close branch dropdown on outside click ────────────────────────────────
  useEffect(() => {
    function handleOutside(e: MouseEvent) {
      if (branchDropdownRef.current && !branchDropdownRef.current.contains(e.target as Node)) {
        setShowBranchDropdown(false);
      }
    }
    if (showBranchDropdown) {
      document.addEventListener("mousedown", handleOutside);
      return () => document.removeEventListener("mousedown", handleOutside);
    }
  }, [showBranchDropdown]);

  // ── Close search dropdown on outside click ───────────────────────────────
  useEffect(() => {
    function handleOutside(e: MouseEvent) {
      if (searchRef.current && !searchRef.current.contains(e.target as Node)) {
        setShowSearchDropdown(false);
      }
    }
    if (showSearchDropdown) {
      document.addEventListener("mousedown", handleOutside);
      return () => document.removeEventListener("mousedown", handleOutside);
    }
  }, [showSearchDropdown]);

  // ── Load sites on mount ───────────────────────────────────────────────────

  useEffect(() => {
    listSiteMapSites().then(list => {
      setSites(list);
      if (list.length > 0) setSiteId(list[0].id);
    });
  }, []);

  // ── Load site map when siteId changes ─────────────────────────────────────

  const loadSite = useCallback(async (id: string) => {
    if (!id) return;
    setLoading(true);
    setMeta(null);
    setMapData(null);
    setMarkers([]);
    setMapErr("");
    setSelectedSpot(null);
    setBranchInfo(null);
    try {
      const [metaRes, dataRes, markersRes] = await Promise.all([
        getSiteMapMeta(id, false),          // light map — matches Rapyuta fleet UI
        getSiteMapData(id),
        getSiteMarkers(id).catch(() => ({ markers: [] })),
      ]);
      setMeta(metaRes);
      setMapData(dataRes);
      setMarkers(markersRes.markers);
      // Load branch info non-blocking (best-effort)
      getSiteBranchInfo(id).then(setBranchInfo).catch(() => setBranchInfo(null));
    } catch (e: unknown) {
      setMapErr(e instanceof Error ? e.message : "Failed to load site map");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (siteId) loadSite(siteId);
  }, [siteId, loadSite]);

  const openCleanupModal = useCallback(async () => {
    setCleanupPlan(null);
    setCleanupResult(null);
    setCleanupErr("");
    setShowCleanupModal(true);
    setCleanupPlanLoading(true);
    try {
      const plan = await getBranchCleanupPlan();
      setCleanupPlan(plan);
    } catch (e: unknown) {
      setCleanupErr(e instanceof Error ? e.message : "Failed to load cleanup plan.");
    } finally {
      setCleanupPlanLoading(false);
    }
  }, []);

  // ── Legend type toggle helpers ─────────────────────────────────────────────

  const toggleSpotType = useCallback((type: string) => {
    setHiddenSpotTypes(prev => {
      const next = new Set(prev);
      if (next.has(type)) { next.delete(type); } else { next.add(type); }
      return next;
    });
  }, []);

  const toggleRegionType = useCallback((type: string) => {
    setHiddenRegionTypes(prev => {
      const next = new Set(prev);
      if (next.has(type)) { next.delete(type); } else { next.add(type); }
      return next;
    });
  }, []);

  // ── Derived data ──────────────────────────────────────────────────────────

  const spotTypeGroups = mapData
    ? mapData.spots.reduce<Record<string, number>>((acc, s) => {
        acc[s.type] = (acc[s.type] ?? 0) + 1;
        return acc;
      }, {})
    : {};

  const regionTypeGroups = mapData
    ? mapData.regions.reduce<Record<string, number>>((acc, r) => {
        acc[r.type] = (acc[r.type] ?? 0) + 1;
        return acc;
      }, {})
    : {};

  // ── Search suggestions ────────────────────────────────────────────────────

  const searchSuggestions = useMemo(() => {
    const q = inputText.toLowerCase().trim();
    if (!q || !mapData) return [];
    const results: {
      label: string;
      sub: string;
      category: "spot" | "rack" | "region" | "node";
      pixX: number;
      pixY: number;
      score: number;
    }[] = [];

    const score = (text: string) =>
      text === q ? 3 : text.startsWith(q) ? 2 : 1;

    if (mapData && meta) {
      const w2p = (wx: number, wy: number) =>
        worldToPixel(wx, wy, meta.origin[0], meta.origin[1], meta.resolution, meta.height);

      // Spots — match by name or index
      mapData.spots.forEach(s => {
        const sl = s.name.toLowerCase();
        const idl = String(s._idx);
        if (sl.includes(q) || idl.includes(q)) {
          const [px, py] = w2p(s.x, s.y);
          results.push({ label: s.name, sub: `#${s._idx} · ${s.type.replace(/_/g, " ")}`, category: "spot", pixX: px, pixY: py, score: score(sl) });
        }
      });

      // Racks — match by section / row / label
      const seenRack = new Set<string>();
      mapData.racks.forEach(r => {
        const key = r.label || `${r.section}-${r.row}`;
        if (!seenRack.has(key) && (
          r.section.toLowerCase().includes(q) ||
          r.row.toLowerCase().includes(q) ||
          r.label.toLowerCase().includes(q)
        )) {
          seenRack.add(key);
          const [px, py] = w2p(r.x, r.y);
          results.push({ label: key, sub: `Section ${r.section} · Row ${r.row}`, category: "rack", pixX: px, pixY: py, score: score(key.toLowerCase()) });
        }
      });

      // Regions — match by id / name / type, use polygon centroid
      const seenRegion = new Set<string>();
      mapData.regions.forEach(r => {
        const key = r.name || String(r.id);
        if (!seenRegion.has(key) && (
          String(r.id).toLowerCase().includes(q) ||
          r.name.toLowerCase().includes(q) ||
          r.type.toLowerCase().includes(q)
        )) {
          seenRegion.add(key);
          const pts = r.polygon.map(([wx, wy]) => w2p(wx, wy));
          const cx  = pts.reduce((a, [px]) => a + px, 0) / pts.length;
          const cy  = pts.reduce((a, [, py]) => a + py, 0) / pts.length;
          results.push({ label: r.name || r.type, sub: `#${r.id} · ${r.type.replace(/_/g, " ")}`, category: "region", pixX: cx, pixY: cy, score: score(r.name.toLowerCase()) });
        }
      });

      // Nodes — match by id or "node <id>"
      mapData.nodes.forEach(n => {
        const idStr = String(n.id);
        const nodeToken = `node ${n.id}`;
        if (idStr.includes(q) || nodeToken.includes(q)) {
          const [px, py] = w2p(n.x, n.y);
          results.push({
            label: idStr,
            sub: `Node ${n.id} · ${n.parkable ? "parkable" : "transit"}`,
            category: "node",
            pixX: px,
            pixY: py,
            score: score(idStr),
          });
        }
      });
    }

    // Deduplicate by label, sort by score (exact first), no hard cap
    const seen = new Set<string>();
    return results
      .filter(r => {
        const key = `${r.category}:${r.label}`;
        if (seen.has(key)) return false;
        seen.add(key);
        return true;
      })
      .sort((a, b) => b.score - a.score);
  }, [inputText, mapData, meta]);

  // ── Render ─────────────────────────────────────────────────────────────────

  return (
    <div className="flex flex-col h-screen overflow-hidden bg-[#0a0f1a]">

      {/* ── Top bar ─────────────────────────────────────────────────────── */}
      <header className="shrink-0 flex items-center gap-3 px-4 h-12 border-b border-white/[0.07] bg-[#0d1321]/95 backdrop-blur-sm relative z-[60]">
        <div className="flex items-center gap-2 shrink-0">
          <div className="w-7 h-7 rounded-lg bg-blue-600/20 border border-blue-500/30 flex items-center justify-center">
            <Map size={14} className="text-blue-400" />
          </div>
          <span className="text-sm font-semibold text-slate-100">Site Map</span>
        </div>

        <div className="h-4 w-px bg-white/10 shrink-0" />

        <div className="flex items-center gap-2 ml-auto">
          {/* ── Search bar with suggestions ─────────────────────────────── */}
          <div className="relative" ref={searchRef}>
            <div className="relative">
              <Search size={12} className="absolute left-2.5 top-2 text-slate-500 pointer-events-none" />
              <input
                type="text"
                placeholder="Search spots, racks, regions, nodes..."
                value={inputText}
                onChange={e => {
                  const val = e.target.value;
                  setInputText(val);
                  setSearchQuery("");
                  setShowSearchDropdown(true);
                  if (val === "") canvasRef.current?.fitMap();
                }}
                onFocus={() => setShowSearchDropdown(true)}
                onKeyDown={e => {
                  if (e.key === "Enter") {
                    const first = searchSuggestions[0];
                    if (first) {
                      setInputText(first.label);
                      setSearchQuery(first.label);
                      setShowSearchDropdown(false);
                      canvasRef.current?.panTo(first.pixX, first.pixY);
                    }
                  } else if (e.key === "Escape") {
                    setShowSearchDropdown(false);
                  }
                }}
                className="h-7 w-64 pl-7 pr-6 rounded-lg bg-white/[0.05] border border-white/[0.08] text-slate-200 placeholder-slate-600 text-xs focus:outline-none focus:border-blue-500/60 focus:bg-white/[0.07] transition-all"
              />
              {inputText && (
                <button
                  onClick={() => { setInputText(""); setSearchQuery(""); setShowSearchDropdown(false); canvasRef.current?.fitMap(); }}
                  className="absolute right-1.5 top-1.5 text-slate-500 hover:text-slate-300 transition-colors"
                >
                  <X size={11} />
                </button>
              )}
            </div>
            {showSearchDropdown && searchSuggestions.length > 0 && (
              <div className="absolute top-full mt-1.5 left-0 z-50 w-80 bg-[#0f172a] border border-white/[0.1] rounded-xl shadow-2xl shadow-black/50 py-1">
                <div className="px-3 py-1.5 text-[10px] text-slate-500 uppercase tracking-wider border-b border-white/[0.06] mb-1 flex items-center justify-between">
                  <span>Results</span>
                  <span className="text-slate-600">{searchSuggestions.length}</span>
                </div>
                <div className="max-h-72 overflow-y-auto overscroll-contain">
                  {searchSuggestions.map((s, i) => (
                    <button
                      key={i}
                      onMouseDown={() => {
                        setInputText(s.label);
                        setSearchQuery(s.label);
                        setShowSearchDropdown(false);
                        canvasRef.current?.panTo(s.pixX, s.pixY);
                      }}
                      className="w-full text-left px-3 py-2 flex items-center gap-2.5 transition-colors hover:bg-white/[0.05]"
                    >
                      <span className={clsx(
                        "shrink-0 text-[9px] font-bold uppercase px-1.5 py-0.5 rounded-md tracking-wide w-12 text-center",
                        s.category === "spot"   ? "bg-blue-500/15 text-blue-400"
                          : s.category === "rack"   ? "bg-slate-500/15 text-slate-400"
                          : s.category === "region" ? "bg-purple-500/15 text-purple-400"
                          : s.category === "node"   ? "bg-orange-500/15 text-orange-400"
                          : "bg-amber-500/15 text-amber-400",
                      )}>
                        {s.category}
                      </span>
                      <div className="min-w-0 flex-1">
                        <p className="text-xs truncate font-medium text-slate-200">{s.label}</p>
                        <p className="text-[10px] truncate text-slate-500">{s.sub}</p>
                      </div>
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Site selector */}
          <div className="relative" ref={siteDropdownRef}>
            <button
              onClick={() => setShowSiteDropdown(v => !v)}
              className={clsx(
                "h-7 flex items-center gap-1.5 pl-2.5 pr-2 rounded-lg border text-xs font-medium transition-all",
                showSiteDropdown
                  ? "bg-blue-500/15 border-blue-500/40 text-blue-300"
                  : "bg-white/[0.05] border-white/[0.08] text-slate-300 hover:bg-white/[0.08] hover:border-white/[0.14] hover:text-slate-100"
              )}
            >
              <MapPin size={11} className={showSiteDropdown ? "text-blue-400" : "text-slate-500"} />
              <span className="max-w-[110px] truncate">{siteId || "Select site"}</span>
              <ChevronDown size={10} className={clsx("transition-transform", showSiteDropdown && "rotate-180")} />
            </button>

            {showSiteDropdown && (
              <div className="absolute top-full mt-1.5 left-0 z-50 min-w-[180px] bg-[#0f172a] border border-white/[0.1] rounded-xl shadow-2xl shadow-black/50 py-1">
                <div className="px-3 py-1.5 text-[10px] text-slate-500 uppercase tracking-wider border-b border-white/[0.06] mb-1">
                  Sites
                </div>
                <div className="max-h-64 overflow-y-auto overscroll-contain">
                  {sites.length === 0 ? (
                    <p className="px-3 py-2 text-xs text-slate-500">No sites available</p>
                  ) : (
                    sites.map(s => (
                      <button
                        key={s.id}
                        onMouseDown={() => {
                          setSiteId(s.id);
                          setShowSiteDropdown(false);
                          setInputText("");
                          setSearchQuery("");
                          setShowSearchDropdown(false);
                        }}
                        className={clsx(
                          "w-full text-left px-3 py-1.5 text-xs flex items-center gap-2 hover:bg-white/[0.05] transition-colors",
                          s.id === siteId ? "text-blue-400" : "text-slate-300"
                        )}
                      >
                        <span className="w-3 shrink-0">
                          {s.id === siteId && <Check size={11} />}
                        </span>
                        <MapPin size={10} className={s.id === siteId ? "text-blue-400" : "text-slate-600"} />
                        <span className="truncate">{s.id}</span>
                      </button>
                    ))
                  )}
                </div>
              </div>
            )}
          </div>

          {/* Reload */}
          <button
            onClick={() => loadSite(siteId)}
            disabled={loading || !siteId}
            title="Reload site map"
            className="h-7 w-7 flex items-center justify-center rounded-lg bg-white/[0.05] border border-white/[0.08] text-slate-400 hover:text-slate-200 hover:bg-white/[0.08] transition-all disabled:opacity-30"
          >
            <RefreshCw size={12} className={loading ? "animate-spin" : ""} />
          </button>

          {/* ── Branch badge + override dropdown ──────────────────────────── */}
          {branchInfo && (
            <div className="relative" ref={branchDropdownRef}>
              <button
                onClick={() => setShowBranchDropdown(v => !v)}
                title={branchInfo.last_commit ? `${branchInfo.last_commit.hash.slice(0, 7)}: ${branchInfo.last_commit.message}` : branchInfo.ref}
                className={clsx(
                  "h-7 flex items-center gap-1.5 px-2.5 rounded-lg text-xs font-mono transition-all",
                  branchInfo.is_site_specific
                    ? "bg-orange-500/10 text-orange-400 border border-orange-500/20 hover:bg-orange-500/15"
                    : "bg-white/[0.05] text-slate-400 border border-white/[0.08] hover:bg-white/[0.08]"
                )}
              >
                <GitBranch size={10} />
                <span>{branchInfo.branch}</span>
                {branchInfo.is_override && <span className="text-amber-400 text-[10px]">*</span>}
                <ChevronDown size={10} />
              </button>

              {showBranchDropdown && (
                <div className="absolute top-full mt-1.5 right-0 z-50 bg-[#0f172a] border border-white/[0.1] rounded-xl shadow-2xl shadow-black/50 min-w-52 py-1">
                  <div className="px-3 py-1.5 text-[10px] text-slate-500 uppercase tracking-wider border-b border-white/[0.06] mb-1">
                    Branch — {siteId}
                  </div>
                  <div className="max-h-64 overflow-y-auto overscroll-contain">
                    {branchInfo.available_branches.map(b => (
                      <button
                        key={b}
                        onClick={async () => {
                          setShowBranchDropdown(false);
                          const updated = await setSiteBranch(siteId, b);
                          setBranchInfo(updated);
                          loadSite(siteId);
                        }}
                        className={clsx(
                          "w-full text-left px-3 py-1.5 text-xs font-mono flex items-center gap-2 hover:bg-white/[0.05] transition-colors",
                          b === branchInfo.branch ? "text-emerald-400" : "text-slate-300"
                        )}
                      >
                        <span className="w-3 shrink-0">
                          {b === branchInfo.branch && <Check size={11} />}
                        </span>
                        {b}
                      </button>
                    ))}
                  </div>
                  {branchInfo.is_override && (
                    <div className="border-t border-white/[0.06] mt-1 pt-1">
                      <button
                        onClick={async () => {
                          setShowBranchDropdown(false);
                          const updated = await clearSiteBranch(siteId);
                          setBranchInfo(updated);
                          loadSite(siteId);
                        }}
                        className="w-full text-left px-3 py-1.5 text-xs text-amber-400 hover:bg-white/[0.05] transition-colors flex items-center gap-2"
                      >
                        ↩ Reset to auto-detect
                      </button>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          {/* ── Sync button ───────────────────────────────────────────── */}
          <button
            onClick={async () => {
              setSyncing(true);
              try {
                await syncSiteRepo();
                const info = await getSiteBranchInfo(siteId);
                setBranchInfo(info);
                loadSite(siteId);
              } catch { /* ignore */ }
              finally { setSyncing(false); }
            }}
            disabled={syncing || !siteId}
            title="Sync from remote (git fetch)"
            className="h-7 w-7 flex items-center justify-center rounded-lg bg-white/[0.05] border border-white/[0.08] text-slate-400 hover:text-slate-200 hover:bg-white/[0.08] transition-all disabled:opacity-30"
          >
            <RefreshCw size={12} className={syncing ? "animate-spin" : ""} />
          </button>

          {/* Branch cleanup */}
          <button
            onClick={openCleanupModal}
            title="Branch cleanup"
            className="h-7 w-7 flex items-center justify-center rounded-lg bg-white/[0.05] border border-white/[0.08] text-slate-400 hover:text-amber-300 hover:bg-white/[0.08] transition-all"
          >
            <GitBranch size={12} />
          </button>

        </div>
      </header>

      {/* ── Branch Cleanup Modal ─────────────────────────────────────────── */}
      {showCleanupModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm" onClick={() => setShowCleanupModal(false)}>
          <div
            className="bg-[#0f172a] border border-white/[0.1] rounded-2xl shadow-2xl shadow-black/60 w-full max-w-lg mx-4 p-5"
            onClick={e => e.stopPropagation()}
          >
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2 text-slate-100 font-semibold">
                <GitBranch size={15} className="text-amber-400" />
                Branch Cleanup
              </div>
              <button onClick={() => setShowCleanupModal(false)} className="text-slate-600 hover:text-slate-300 transition-colors">
                <X size={15} />
              </button>
            </div>

            <p className="text-xs text-slate-400 mb-4">
              Keep only <code className="bg-white/10 px-1 rounded">main</code> and site-specific branches.
              Removes local remote-tracking refs for all other branches — safe, no data is modified.
            </p>

            {cleanupPlanLoading && (
              <div className="flex items-center gap-2 text-slate-400 text-sm py-4">
                <Loader2 size={14} className="animate-spin" /> Loading plan…
              </div>
            )}

            {cleanupErr && (
              <div className="flex items-center gap-2 text-red-400 text-xs bg-red-900/20 rounded p-3 mb-3">
                <AlertTriangle size={13} />{cleanupErr}
              </div>
            )}

            {cleanupResult && (
              <div className="mb-4 space-y-2">
                <div className="text-xs font-semibold text-green-400 uppercase tracking-wider">Cleanup complete</div>
                <div className="grid grid-cols-2 gap-2">
                  <div className="bg-red-900/20 border border-red-800/30 rounded p-2">
                    <div className="text-[10px] text-red-400 uppercase tracking-wider mb-1">Removed ({cleanupResult.removed.length})</div>
                    {cleanupResult.removed.length === 0
                      ? <div className="text-xs text-slate-500">None</div>
                      : cleanupResult.removed.map(b => <div key={b} className="text-xs font-mono text-red-300">{b}</div>)
                    }
                  </div>
                  <div className="bg-green-900/20 border border-green-800/30 rounded p-2">
                    <div className="text-[10px] text-green-400 uppercase tracking-wider mb-1">Kept ({cleanupResult.kept.length})</div>
                    <div className="max-h-28 overflow-y-auto space-y-px">
                      {cleanupResult.kept.map(b => <div key={b} className="text-xs font-mono text-green-300">{b}</div>)}
                    </div>
                  </div>
                </div>
                {cleanupResult.errors.length > 0 && (
                  <div className="text-xs text-amber-400">
                    Failed to remove: {cleanupResult.errors.join(", ")}
                  </div>
                )}
              </div>
            )}

            {!cleanupResult && cleanupPlan && !cleanupPlanLoading && (
              <div className="mb-4 space-y-3">
                <div className="grid grid-cols-3 gap-2 text-center">
                  <div className="bg-white/5 rounded p-2">
                    <div className="text-lg font-bold text-slate-200">{cleanupPlan.total_branches}</div>
                    <div className="text-[10px] text-slate-500">Total</div>
                  </div>
                  <div className="bg-green-900/20 border border-green-800/30 rounded p-2">
                    <div className="text-lg font-bold text-green-400">{cleanupPlan.valid_branches.length}</div>
                    <div className="text-[10px] text-green-600">Keep</div>
                  </div>
                  <div className="bg-red-900/20 border border-red-800/30 rounded p-2">
                    <div className="text-lg font-bold text-red-400">{cleanupPlan.invalid_branches.length}</div>
                    <div className="text-[10px] text-red-600">Remove</div>
                  </div>
                </div>

                {cleanupPlan.invalid_branches.length > 0 && (
                  <div>
                    <div className="text-[10px] text-red-400 uppercase tracking-wider mb-1">Will be removed</div>
                    <div className="max-h-28 overflow-y-auto bg-red-900/10 border border-red-900/30 rounded p-2 space-y-px">
                      {cleanupPlan.invalid_branches.map(b => (
                        <div key={b} className="text-xs font-mono text-red-300 flex items-center gap-1">
                          <X size={10} className="text-red-500 shrink-0" />{b}
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {cleanupPlan.sites_without_own_branch.length > 0 && (
                  <div className="text-xs text-slate-500">
                    <span className="text-amber-400">{cleanupPlan.sites_without_own_branch.length}</span> site(s) have no dedicated branch and will use <code className="bg-white/10 px-1 rounded">main</code>.
                  </div>
                )}
              </div>
            )}

            <div className="flex justify-end gap-2">
              <button
                onClick={() => setShowCleanupModal(false)}
                className="px-3 py-1.5 text-xs text-slate-400 hover:text-white transition-colors"
              >
                {cleanupResult ? "Close" : "Cancel"}
              </button>
              {!cleanupResult && cleanupPlan && cleanupPlan.invalid_branches.length > 0 && (
                <button
                  disabled={cleanupLoading}
                  onClick={async () => {
                    setCleanupLoading(true);
                    setCleanupErr("");
                    try {
                      const result = await runBranchCleanup();
                      setCleanupResult(result);
                      // Refresh branch info for current site
                      if (siteId) {
                        getSiteBranchInfo(siteId).then(setBranchInfo).catch(() => {});
                      }
                    } catch (e: unknown) {
                      setCleanupErr(e instanceof Error ? e.message : "Cleanup failed");
                    } finally {
                      setCleanupLoading(false);
                    }
                  }}
                  className="px-4 py-1.5 text-xs font-medium rounded-md bg-red-700 hover:bg-red-600 disabled:opacity-50 text-white transition-colors flex items-center gap-1.5"
                >
                  {cleanupLoading ? <Loader2 size={12} className="animate-spin" /> : <Trash2 size={12} />}
                  Remove {cleanupPlan.invalid_branches.length} branch{cleanupPlan.invalid_branches.length !== 1 ? "es" : ""}
                </button>
              )}
              {!cleanupResult && cleanupPlan && cleanupPlan.invalid_branches.length === 0 && (
                <span className="px-4 py-1.5 text-xs text-green-400 flex items-center gap-1.5">
                  <Check size={12} /> Already clean
                </span>
              )}
            </div>
          </div>
        </div>
      )}

      {/* ── Main layout ─────────────────────────────────────────────────── */}
      <div className="flex flex-1 overflow-hidden">

        {/* ── Left panel (resizable) ───────────────────────────────── */}
        <aside
          className="relative shrink-0 flex flex-col border-r border-white/[0.06] bg-[#080e1a] overflow-y-auto overflow-x-hidden"
          style={{ width: sidebarWidth, minWidth: sidebarWidth }}
        >
          {/* Inner scroll container */}
          <div className="flex flex-col flex-1 overflow-y-auto overflow-x-hidden" style={{ width: sidebarWidth }}>

          {/* Layers */}
          <div className="p-4 border-b border-white/[0.06]">
            <p className="text-[11px] font-semibold text-slate-400 uppercase tracking-widest mb-2.5 flex items-center gap-1.5">
              <LayersIcon size={11} /> Layers
            </p>
            <div className="grid grid-cols-2 gap-1.5">
              {([
                { key: "spots",      label: "Spots",      dot: "#60a5fa" },
                { key: "racks",      label: "Racks",      dot: "#fbbf24" },
                { key: "regions",    label: "Regions",    dot: "#a78bfa" },
                { key: "markers",    label: "AR Markers", dot: "#f87171" },
                { key: "nodes",      label: "Nodes",      dot: "#f97316" },
              ] as const).map(({ key, label, dot }) => (
                <button
                  key={key}
                  onClick={() => setLayers(l => ({ ...l, [key]: !l[key] }))}
                  className={clsx(
                    "flex items-center gap-2 px-2.5 py-2 rounded-lg text-xs font-medium transition-all",
                    layers[key]
                      ? "bg-white/[0.07] text-slate-200 border border-white/[0.08]"
                      : "bg-transparent text-slate-600 border border-transparent hover:text-slate-400"
                  )}
                >
                  <span
                    className="w-2 h-2 rounded-full shrink-0"
                    style={{ backgroundColor: dot, opacity: layers[key] ? 1 : 0.3 }}
                  />
                  {label}
                </button>
              ))}
            </div>
          </div>

          {/* Site stats */}
          {mapData && (
            <div className="p-4 border-b border-white/[0.06]">
              <p className="text-[11px] font-semibold text-slate-400 uppercase tracking-widest mb-2.5">Site Info</p>
              <div className="grid grid-cols-2 gap-2">
                {[
                  { icon: <MapPin size={11} />,  label: "Spots",     value: mapData.spots.length,   color: "text-blue-400"  },
                  { icon: <Package size={11} />, label: "Racks",     value: mapData.racks.length,   color: "text-amber-400" },
                  { icon: <Zap size={11} />,     label: "Regions",   value: mapData.regions.length, color: "text-purple-400"},
                  { icon: <GitBranch size={11} />, label: "Nodes",   value: mapData.nodes.length,   color: "text-orange-400"},
                  { icon: <Bot size={11} />,     label: "Robots",    value: mapData.robots.length,  color: "text-emerald-400"},
                  { icon: <Navigation size={11} />, label: "Markers", value: markers.length,        color: "text-red-400"   },
                ].map(({ icon, label, value, color }) => (
                  <div key={label} className="bg-white/[0.04] rounded-lg p-2.5 border border-white/[0.05]">
                    <span className={clsx("mb-1.5 block", color)}>{icon}</span>
                    <p className="text-base font-bold text-slate-100 leading-none">{value}</p>
                    <p className="text-[11px] text-slate-500 mt-1">{label}</p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Legend */}
          <div className="p-4 border-b border-white/[0.06]">
            <div className="flex items-center justify-between mb-3">
              <p className="text-[11px] font-semibold text-slate-400 uppercase tracking-widest">Legend</p>
              <span className="text-[10px] text-slate-600">click to toggle</span>
            </div>

            {/* ── Spot types ── */}
            <p className="text-[10px] text-slate-500 uppercase tracking-widest mb-2">Spots</p>
            {hiddenSpotTypes.size > 0 && (
              <button
                onClick={() => setHiddenSpotTypes(new Set())}
                className="text-[10px] text-blue-400 hover:text-blue-300 transition-colors mb-1"
              >
                show all
              </button>
            )}
            <div className="space-y-0.5 mb-3.5">
              {SPOT_LEGEND.filter(({ type }) => spotTypeGroups[type] !== undefined).map(({ color, label, type }) => {
                const hidden = hiddenSpotTypes.has(type);
                return (
                  <button
                    key={type}
                    onClick={() => toggleSpotType(type)}
                    title={hidden ? `Show ${label}` : `Hide ${label}`}
                    className={clsx(
                      "w-full flex items-center gap-2 px-2 py-1.5 rounded-md text-left transition-all select-none",
                      hidden ? "opacity-35 hover:opacity-60" : "hover:bg-white/[0.04]"
                    )}
                  >
                    <span
                      className={clsx("w-2.5 h-2.5 rounded-full shrink-0", hidden && "grayscale")}
                      style={{ backgroundColor: color }}
                    />
                    <span className={clsx("text-xs text-slate-400 flex-1", hidden && "line-through decoration-slate-600")}>
                      {label}
                    </span>
                    <span className="text-[11px] font-mono text-slate-600">{spotTypeGroups[type]}</span>
                  </button>
                );
              })}
              {Object.entries(spotTypeGroups)
                .filter(([t]) => !SPOT_LEGEND.find(l => l.type === t))
                .map(([type, count]) => {
                  const hidden = hiddenSpotTypes.has(type);
                  return (
                    <button
                      key={type}
                      onClick={() => toggleSpotType(type)}
                      className={clsx(
                        "w-full flex items-center gap-2 px-2 py-1.5 rounded-md text-left transition-all select-none",
                        hidden ? "opacity-35 hover:opacity-60" : "hover:bg-white/[0.04]"
                      )}
                    >
                      <span className="w-2.5 h-2.5 rounded-full bg-slate-500 shrink-0" />
                      <span className={clsx("text-xs text-slate-400 flex-1 capitalize", hidden && "line-through decoration-slate-600")}>
                        {type.replace(/_/g, " ")}
                      </span>
                      <span className="text-[11px] font-mono text-slate-600">{count}</span>
                    </button>
                  );
                })}
            </div>

            {/* ── Region types ── */}
            <div className="flex items-center justify-between mb-2 border-t border-white/[0.04] pt-3">
              <p className="text-[10px] text-slate-500 uppercase tracking-widest">Regions</p>
              {hiddenRegionTypes.size > 0 && (
                <button
                  onClick={() => setHiddenRegionTypes(new Set())}
                  className="text-[10px] text-blue-400 hover:text-blue-300 transition-colors"
                >
                  show all
                </button>
              )}
            </div>
            <div className="space-y-0.5">
              {REGION_LEGEND.filter(({ type }) => regionTypeGroups[type] !== undefined).map(({ color, label, type }) => {
                const hidden = hiddenRegionTypes.has(type);
                return (
                  <button
                    key={type}
                    onClick={() => toggleRegionType(type)}
                    title={hidden ? `Show ${label}` : `Hide ${label}`}
                    className={clsx(
                      "w-full flex items-center gap-2 px-2 py-1.5 rounded-md text-left transition-all select-none",
                      hidden ? "opacity-35 hover:opacity-60" : "hover:bg-white/[0.04]"
                    )}
                  >
                    <span
                      className={clsx("w-2.5 h-2.5 rounded-sm shrink-0 border border-white/10", hidden && "grayscale")}
                      style={{ backgroundColor: color.replace(/[\d.]+\)$/, "0.8)") }}
                    />
                    <span className={clsx("text-xs text-slate-400 flex-1", hidden && "line-through decoration-slate-600")}>
                      {label}
                    </span>
                    <span className="text-[11px] font-mono text-slate-600">{regionTypeGroups[type]}</span>
                  </button>
                );
              })}
              {Object.entries(regionTypeGroups)
                .filter(([t]) => !REGION_LEGEND.find(l => l.type === t))
                .map(([type, count]) => {
                  const hidden = hiddenRegionTypes.has(type);
                  return (
                    <button
                      key={type}
                      onClick={() => toggleRegionType(type)}
                      className={clsx(
                        "w-full flex items-center gap-2 px-2 py-1.5 rounded-md text-left transition-all select-none",
                        hidden ? "opacity-35 hover:opacity-60" : "hover:bg-white/[0.04]"
                      )}
                    >
                      <span className="w-2.5 h-2.5 rounded-sm bg-slate-600 border border-white/10 shrink-0" />
                      <span className={clsx("text-xs text-slate-400 flex-1 capitalize", hidden && "line-through decoration-slate-600")}>
                        {type.replace(/_/g, " ")}
                      </span>
                      <span className="text-[11px] font-mono text-slate-600">{count}</span>
                    </button>
                  );
                })}
            </div>
          </div>

          </div>

          {/* Right-edge resize handle */}
          <div
            onMouseDown={onSidebarResizeStart}
            className="absolute top-0 right-0 w-1 h-full cursor-col-resize hover:bg-blue-500/40 active:bg-blue-500/60 transition-colors z-10"
            title="Drag to resize"
          />
        </aside>

        {/* ── Centre: map canvas ───────────────────────────────────────── */}
        <main className="flex-1 flex flex-col overflow-hidden bg-[#0a0f1a]">
          {/* Map area */}
          <div className="flex-1 relative overflow-hidden">
            {mapErr && (
              <div className="absolute inset-0 flex items-center justify-center z-10 pointer-events-none">
                <div className="bg-[#0f172a]/90 border border-red-500/20 rounded-xl px-5 py-3 text-red-400 text-sm backdrop-blur-sm">
                  {mapErr}
                </div>
              </div>
            )}

            {loading && (
              <div className="absolute inset-0 flex items-center justify-center z-10 bg-[#0a0f1a]/80 backdrop-blur-sm">
                <div className="flex flex-col items-center gap-3">
                  <div className="w-10 h-10 rounded-full border-2 border-blue-500/30 border-t-blue-400 animate-spin" />
                  <p className="text-slate-500 text-xs tracking-wide">Loading map…</p>
                </div>
              </div>
            )}

            {meta && mapData ? (
              <SiteMapCanvas
                ref={canvasRef}
                meta={meta}
                data={mapData}
                markers={markers}
                searchQuery={searchQuery}
                layers={layers}
                hiddenSpotTypes={hiddenSpotTypes}
                hiddenRegionTypes={hiddenRegionTypes}
                onSpotSelect={setSelectedSpot}
              />
            ) : !loading && !mapErr ? (
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="text-center">
                  <div className="w-16 h-16 rounded-2xl bg-white/[0.03] border border-white/[0.06] flex items-center justify-center mx-auto mb-4">
                    <Map size={28} className="text-slate-700" />
                  </div>
                  <p className="text-sm text-slate-600">No site selected</p>
                  <p className="text-xs text-slate-700 mt-1">Choose a site from the header</p>
                </div>
              </div>
            ) : null}

            {/* Selected spot tooltip overlay */}
            {selectedSpot && (
              <div className="absolute top-3 right-3 z-10 w-52 bg-[#0f172a]/95 border border-white/[0.1] rounded-xl p-3 shadow-2xl shadow-black/60 backdrop-blur-sm">
                <div className="flex items-center justify-between mb-2.5">
                  <span className="text-xs font-semibold text-slate-100">{selectedSpot.name}</span>
                  <button onClick={() => setSelectedSpot(null)} className="text-slate-600 hover:text-slate-300 transition-colors">
                    <X size={12} />
                  </button>
                </div>
                <div className="space-y-1.5">
                  {[
                    { label: "Type",  value: selectedSpot.type.replace(/_/g," ")  },
                    { label: "X",     value: `${selectedSpot.x.toFixed(3)} m` },
                    { label: "Y",     value: `${selectedSpot.y.toFixed(3)} m` },
                    { label: "Yaw",   value: `${(selectedSpot.yaw * 180 / Math.PI).toFixed(1)}°` },
                    { label: "Robot", value: selectedSpot.robot || "—" },
                  ].map(({ label, value }) => (
                    <div key={label} className="flex justify-between items-center">
                      <span className="text-[10px] text-slate-600 uppercase tracking-wide">{label}</span>
                      <span className="text-[11px] text-slate-300 font-mono">{value}</span>
                    </div>
                  ))}
                </div>
                <div className="mt-2.5 pt-2 border-t border-white/[0.06] flex items-center gap-1.5">
                  <span className="w-2.5 h-2.5 rounded-full shrink-0" style={{ backgroundColor: selectedSpot.color }} />
                  <span className="text-[10px] text-slate-600 capitalize">{selectedSpot.type.replace(/_/g," ")}</span>
                </div>
              </div>
            )}

          </div>
        </main>
      </div>
    </div>
  );
}
