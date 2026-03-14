"use client";

import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  ReferenceArea,
} from "recharts";
import ReactMarkdown from "react-markdown";
import {
  ChevronDown,
  Search,
  Loader2,
  Sparkles,
  RotateCcw,
  ChevronRight,
  Check,
  X,
  ClipboardList,
} from "lucide-react";

import {
  fetchLogSites,
  fetchLogHostnamesV2,
  fetchLogDeploymentsV2,
  fetchLogVolume,
  fetchLogQuery,
  analyseLogsAndSlack,
} from "@/lib/api";
import type {
  LokiLogLine,
  AnalyseResponse,
} from "@/lib/types";

/* ═══════════════════════════════════════════════════════════════════════════ */
/*  DESIGN TOKENS                                                            */
/* ═══════════════════════════════════════════════════════════════════════════ */
const PAGE_BG = "#111827";
const FILTER_BG = "#1a1d23";
const CHART_BG = "#0b0e12";
const LOG_BG = "#0b0e12";
const LOG_ALT = "#0f1318";
const LOG_HOVER = "#1a1f2e";
const BORDER = "#2d3139";

const LEVEL_BAR: Record<string, string> = {
  info: "#4ade80",
  debug: "#6b7280",
  warn: "#f59e0b",
  warning: "#f59e0b",
  error: "#ef4444",
  fatal: "#dc2626",
  unknown: "#6b7280",
};

const CHART_GREEN = "#4ade80";
const COUNT_RED = "#ef4444";

const ENVIRONMENTS = [
  "sootballs-prod-logs-loki-US-latest",
  "sootballs-prod-logs-loki",
  "sootballs-staging-logs-loki",
  "rio-loki",
  "Loki",
];

const MAX_DISPLAY_LINES = 4000;

/* ═══════════════════════════════════════════════════════════════════════════ */
/*  QUICK RANGE PRESETS                                                      */
/* ═══════════════════════════════════════════════════════════════════════════ */
const QUICK_RANGES: { label: string; ms: number }[] = [
  { label: "Last 15m", ms: 15 * 60 * 1000 },
  { label: "Last 1h", ms: 60 * 60 * 1000 },
  { label: "Last 3h", ms: 3 * 60 * 60 * 1000 },
  { label: "Last 6h", ms: 6 * 60 * 60 * 1000 },
  { label: "Last 12h", ms: 12 * 60 * 60 * 1000 },
  { label: "Last 24h", ms: 24 * 60 * 60 * 1000 },
  { label: "Last 2d", ms: 2 * 24 * 60 * 60 * 1000 },
];

/* ═══════════════════════════════════════════════════════════════════════════ */
/*  HELPERS                                                                  */
/* ═══════════════════════════════════════════════════════════════════════════ */
function detectLevel(line: string): string {
  const upper = line.toUpperCase();
  if (upper.includes("FATAL")) return "fatal";
  if (upper.includes("ERROR") || upper.includes("ERR]")) return "error";
  if (upper.includes("WARN")) return "warn";
  if (upper.includes("DEBUG")) return "debug";
  if (upper.includes("INFO")) return "info";
  return "unknown";
}

function nsToMs(ns: string): number {
  return Math.floor(parseInt(ns, 10) / 1_000_000);
}

function fmtTs(ms: number): string {
  return new Date(ms).toISOString().replace("T", " ").replace("Z", "").slice(0, 23);
}

function fmtTimeAxis(ms: number): string {
  const d = new Date(ms);
  const h = String(d.getUTCHours()).padStart(2, "0");
  const m = String(d.getUTCMinutes()).padStart(2, "0");
  return `${h}:${m}`;
}

function toDateStr(ms: number): string {
  return new Date(ms).toISOString().slice(0, 10);
}

function toTimeStr(ms: number): string {
  return new Date(ms).toISOString().slice(11, 19);
}

function parseDateTimeToMs(dateStr: string, timeStr: string): number {
  const dt = new Date(`${dateStr}T${timeStr}Z`);
  return dt.getTime();
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/*  MULTI-SELECT DROPDOWN COMPONENT                                          */
/* ═══════════════════════════════════════════════════════════════════════════ */
interface MultiSelectProps {
  label: string;
  options: string[];
  selected: string[];
  onChange: (v: string[]) => void;
  loading?: boolean;
  disabled?: boolean;
}

function MultiSelect({ label, options, selected, onChange, loading, disabled }: MultiSelectProps) {
  const [open, setOpen] = useState(false);
  const [search, setSearch] = useState("");
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    function onClickOutside(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    }
    document.addEventListener("mousedown", onClickOutside);
    return () => document.removeEventListener("mousedown", onClickOutside);
  }, []);

  const filtered = options.filter((o) =>
    o.toLowerCase().includes(search.toLowerCase())
  );

  const display =
    selected.length === 0
      ? "All"
      : selected.length === 1
        ? selected[0]
        : `Selected (${selected.length})`;

  function toggle(val: string) {
    if (selected.includes(val)) {
      onChange(selected.filter((s) => s !== val));
    } else {
      onChange([...selected, val]);
    }
  }

  return (
    <div ref={ref} className="relative">
      <button
        type="button"
        onClick={() => !disabled && setOpen(!open)}
        disabled={disabled}
        className="flex items-center gap-1.5 rounded px-2.5 py-1.5 text-xs transition-colors hover:bg-white/5 disabled:opacity-40"
        style={{ background: open ? "#2d3139" : "transparent" }}
      >
        <span className="text-slate-500">{label}</span>
        <span className="max-w-[120px] truncate text-white">{display}</span>
        <ChevronDown size={12} className="text-slate-500" />
      </button>
      {open && !disabled && (
        <div
          className="absolute left-0 top-full z-50 mt-1 w-56 overflow-hidden rounded-md border shadow-xl"
          style={{ background: FILTER_BG, borderColor: BORDER }}
        >
          <div className="p-1.5">
            <input
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search..."
              autoFocus
              className="w-full rounded bg-black/30 px-2 py-1 text-xs text-white placeholder-slate-500 outline-none"
            />
          </div>
          <div className="max-h-[280px] overflow-y-auto">
            {loading ? (
              <div className="flex items-center gap-2 px-3 py-4 text-xs text-slate-400">
                <Loader2 size={12} className="animate-spin" /> Loading…
              </div>
            ) : filtered.length === 0 ? (
              <div className="px-3 py-4 text-xs text-slate-500">No results</div>
            ) : (
              filtered.map((opt) => {
                const isSelected = selected.includes(opt);
                return (
                  <button
                    key={opt}
                    type="button"
                    onClick={() => toggle(opt)}
                    className="flex w-full items-center gap-2 px-3 py-1.5 text-left text-xs transition-colors hover:bg-white/5"
                  >
                    <span
                      className="flex h-3.5 w-3.5 shrink-0 items-center justify-center rounded-sm border"
                      style={{
                        borderColor: isSelected ? "#3b82f6" : "#4b5563",
                        background: isSelected ? "#3b82f6" : "transparent",
                      }}
                    >
                      {isSelected && <Check size={10} className="text-white" />}
                    </span>
                    <span className={isSelected ? "text-white" : "text-slate-300"}>
                      {opt}
                    </span>
                  </button>
                );
              })
            )}
          </div>
        </div>
      )}
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/*  SINGLE-SELECT DROPDOWN                                                   */
/* ═══════════════════════════════════════════════════════════════════════════ */
interface SingleSelectProps {
  label: string;
  options: string[];
  value: string;
  onChange: (v: string) => void;
  loading?: boolean;
}

function SingleSelect({ label, options, value, onChange, loading }: SingleSelectProps) {
  const [open, setOpen] = useState(false);
  const [search, setSearch] = useState("");
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    function onClickOutside(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    }
    document.addEventListener("mousedown", onClickOutside);
    return () => document.removeEventListener("mousedown", onClickOutside);
  }, []);

  const filtered = options.filter((o) =>
    o.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div ref={ref} className="relative">
      <button
        type="button"
        onClick={() => setOpen(!open)}
        className="flex items-center gap-1.5 rounded px-2.5 py-1.5 text-xs transition-colors hover:bg-white/5"
        style={{ background: open ? "#2d3139" : "transparent" }}
      >
        <span className="text-slate-500">{label}</span>
        <span className="max-w-[200px] truncate text-white">{value || "All"}</span>
        <ChevronDown size={12} className="text-slate-500" />
      </button>
      {open && (
        <div
          className="absolute left-0 top-full z-50 mt-1 w-64 overflow-hidden rounded-md border shadow-xl"
          style={{ background: FILTER_BG, borderColor: BORDER }}
        >
          <div className="p-1.5">
            <input
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search..."
              autoFocus
              className="w-full rounded bg-black/30 px-2 py-1 text-xs text-white placeholder-slate-500 outline-none"
            />
          </div>
          <div className="max-h-[280px] overflow-y-auto">
            {loading ? (
              <div className="flex items-center gap-2 px-3 py-4 text-xs text-slate-400">
                <Loader2 size={12} className="animate-spin" /> Loading…
              </div>
            ) : filtered.length === 0 ? (
              <div className="px-3 py-4 text-xs text-slate-500">No results</div>
            ) : (
              filtered.map((opt) => (
                <button
                  key={opt}
                  type="button"
                  onClick={() => {
                    onChange(opt);
                    setOpen(false);
                  }}
                  className="flex w-full items-center gap-2 px-3 py-1.5 text-left text-xs transition-colors hover:bg-white/5"
                >
                  {value === opt && <Check size={12} className="text-blue-400" />}
                  <span className={value === opt ? "text-white" : "text-slate-300"}>
                    {opt}
                  </span>
                </button>
              ))
            )}
          </div>
        </div>
      )}
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/*  MAIN PAGE                                                                */
/* ═══════════════════════════════════════════════════════════════════════════ */
export default function LogViewerPage() {
  /* ── Filter state (cascading) ──────────────────────────────────────────── */
  const [env, setEnv] = useState(ENVIRONMENTS[0]);
  const [siteOptions, setSiteOptions] = useState<string[]>([]);
  const [selectedSite, setSelectedSite] = useState("");
  const [hostnameOptions, setHostnameOptions] = useState<string[]>([]);
  const [selectedHostnames, setSelectedHostnames] = useState<string[]>([]);
  const [deploymentOptions, setDeploymentOptions] = useState<string[]>([]);
  const [selectedDeployments, setSelectedDeployments] = useState<string[]>([]);
  const [searchText, setSearchText] = useState("");
  const [excludeText, setExcludeText] = useState("");

  const [loadingSites, setLoadingSites] = useState(false);
  const [loadingHosts, setLoadingHosts] = useState(false);
  const [loadingDeps, setLoadingDeps] = useState(false);

  /* ── Time range state ──────────────────────────────────────────────────── */
  const now = Date.now();
  const [fromDate, setFromDate] = useState(toDateStr(now - 15 * 60 * 1000));
  const [fromTime, setFromTime] = useState(toTimeStr(now - 15 * 60 * 1000));
  const [toDate, setToDate] = useState(toDateStr(now));
  const [toTime, setToTime] = useState(toTimeStr(now));
  const [activeQuick, setActiveQuick] = useState("Last 15m");

  /* ── Log data ──────────────────────────────────────────────────────────── */
  const [allLines, setAllLines] = useState<LokiLogLine[]>([]);
  const [totalCount, setTotalCount] = useState(0);
  const [loading, setLoading] = useState(false);
  const [fetchError, setFetchError] = useState("");

  /* ── Volume data ───────────────────────────────────────────────────────── */
  const [volumeData, setVolumeData] = useState<{ ts: number; label: string; count: number }[]>([]);

  /* ── Chart brush ───────────────────────────────────────────────────────── */
  const [brushStart, setBrushStart] = useState<number | null>(null);
  const [brushEnd, setBrushEnd] = useState<number | null>(null);
  const [selecting, setSelecting] = useState(false);

  /* ── Section collapse ──────────────────────────────────────────────────── */
  const [logsOpen, setLogsOpen] = useState(true);

  /* ── Log Analysis ──────────────────────────────────────────────────────── */
  const [issueDesc, setIssueDesc] = useState("");
  const [analysisLines, setAnalysisLines] = useState<1000 | 2000>(2000);
  const [analysing, setAnalysing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState<AnalyseResponse | null>(null);
  const [analysisError, setAnalysisError] = useState("");

  /* ── Expanded rows ─────────────────────────────────────────────────────── */
  const [expandedIdx, setExpandedIdx] = useState<number | null>(null);

  /* ════ CASCADING FILTER EFFECTS ════════════════════════════════════════ */

  /* Fetch sites when env changes */
  useEffect(() => {
    setSelectedSite("");
    setHostnameOptions([]);
    setSelectedHostnames([]);
    setDeploymentOptions([]);
    setSelectedDeployments([]);
    setLoadingSites(true);
    fetchLogSites(env)
      .then(setSiteOptions)
      .catch(() => setSiteOptions([]))
      .finally(() => setLoadingSites(false));
  }, [env]);

  /* Fetch hostnames when site changes */
  useEffect(() => {
    setSelectedHostnames([]);
    setDeploymentOptions([]);
    setSelectedDeployments([]);
    if (!selectedSite) {
      setHostnameOptions([]);
      return;
    }
    setLoadingHosts(true);
    fetchLogHostnamesV2(env, selectedSite)
      .then(setHostnameOptions)
      .catch(() => setHostnameOptions([]))
      .finally(() => setLoadingHosts(false));
  }, [env, selectedSite]);

  /* Fetch deployments when hostname changes */
  useEffect(() => {
    setSelectedDeployments([]);
    if (!selectedSite || selectedHostnames.length !== 1) {
      setDeploymentOptions([]);
      return;
    }
    setLoadingDeps(true);
    fetchLogDeploymentsV2(env, selectedSite, selectedHostnames[0])
      .then(setDeploymentOptions)
      .catch(() => setDeploymentOptions([]))
      .finally(() => setLoadingDeps(false));
  }, [env, selectedSite, selectedHostnames]);

  /* ════ TIME HELPERS ═════════════════════════════════════════════════════ */

  function applyQuickRange(label: string, ms: number) {
    const n = Date.now();
    setFromDate(toDateStr(n - ms));
    setFromTime(toTimeStr(n - ms));
    setToDate(toDateStr(n));
    setToTime(toTimeStr(n));
    setActiveQuick(label);
  }

  function getFromMs(): number {
    return parseDateTimeToMs(fromDate, fromTime);
  }

  function getToMs(): number {
    return parseDateTimeToMs(toDate, toTime);
  }

  /* ════ FETCH LOGS + VOLUME ═════════════════════════════════════════════ */

  const doFetch = useCallback(async () => {
    if (!selectedSite) return;
    setLoading(true);
    setFetchError("");
    setExpandedIdx(null);
    setBrushStart(null);
    setBrushEnd(null);
    setAnalysisResult(null);

    const fMs = getFromMs();
    const tMs = getToMs();

    try {
      const [logResp, volResp] = await Promise.all([
        fetchLogQuery({
          env,
          site: selectedSite,
          hostname: selectedHostnames.length === 1 ? selectedHostnames[0] : undefined,
          deployment: selectedDeployments.length === 1 ? selectedDeployments[0] : undefined,
          search: searchText || undefined,
          exclude: excludeText || undefined,
          from: fMs,
          to: tMs,
          limit: MAX_DISPLAY_LINES,
        }),
        fetchLogVolume({
          env,
          site: selectedSite,
          hostname: selectedHostnames.length === 1 ? selectedHostnames[0] : undefined,
          deployment: selectedDeployments.length === 1 ? selectedDeployments[0] : undefined,
          from: fMs,
          to: tMs,
        }),
      ]);

      setAllLines(logResp.lines);
      setTotalCount(logResp.total_count);

      // Build chart-friendly volume data
      setVolumeData(
        volResp.map((b) => ({
          ts: b.ts * 1000,
          label: fmtTimeAxis(b.ts * 1000),
          count: b.count,
        }))
      );
    } catch (e) {
      setFetchError(e instanceof Error ? e.message : "Failed to fetch logs.");
      setAllLines([]);
      setVolumeData([]);
    } finally {
      setLoading(false);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [env, selectedSite, selectedHostnames, selectedDeployments, searchText, excludeText, fromDate, fromTime, toDate, toTime]);

  /* ── Brush-filtered lines ──────────────────────────────────────────────── */
  const filteredLines = useMemo(() => {
    if (brushStart === null || brushEnd === null) return allLines;
    const loMs = Math.min(brushStart, brushEnd);
    const hiMs = Math.max(brushStart, brushEnd);
    return allLines.filter((l) => {
      const ms = nsToMs(l.ts);
      return ms >= loMs && ms <= hiMs;
    });
  }, [allLines, brushStart, brushEnd]);

  /* ── Chart mouse handlers ──────────────────────────────────────────────── */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const onChartDown = (e: Record<string, any> | null) => {
    if (e?.activePayload?.[0]?.payload) {
      setSelecting(true);
      setBrushStart(e.activePayload[0].payload.ts);
      setBrushEnd(null);
    }
  };
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const onChartMove = (e: Record<string, any> | null) => {
    if (selecting && e?.activePayload?.[0]?.payload) {
      setBrushEnd(e.activePayload[0].payload.ts);
    }
  };
  const onChartUp = () => {
    setSelecting(false);
    // Update From/To inputs when brush selection completes
    if (brushStart !== null && brushEnd !== null) {
      const lo = Math.min(brushStart, brushEnd);
      const hi = Math.max(brushStart, brushEnd);
      setFromDate(toDateStr(lo));
      setFromTime(toTimeStr(lo));
      setToDate(toDateStr(hi));
      setToTime(toTimeStr(hi));
      setActiveQuick("");
    }
  };
  const clearBrush = () => {
    setBrushStart(null);
    setBrushEnd(null);
  };

  /* ── AI Log Analysis handler ───────────────────────────────────────────── */
  const handleAnalyse = useCallback(async () => {
    if (!issueDesc.trim()) return;
    setAnalysing(true);
    setAnalysisError("");
    setAnalysisResult(null);

    // Prioritize ERROR/WARN lines, then fill with others
    const lines = [...filteredLines];
    const errorWarn = lines.filter((l) => {
      const lvl = detectLevel(l.line);
      return lvl === "error" || lvl === "fatal" || lvl === "warn";
    });
    const others = lines.filter((l) => {
      const lvl = detectLevel(l.line);
      return lvl !== "error" && lvl !== "fatal" && lvl !== "warn";
    });
    const prioritized = [...errorWarn, ...others].slice(0, analysisLines);

    try {
      const logEntries = prioritized.map((l) => ({
        timestamp_ms: nsToMs(l.ts),
        level: detectLevel(l.line),
        hostname: l.labels?.hostname || "",
        deployment: l.labels?.deployment_name || "",
        message: l.line,
        labels: l.labels,
      }));
      const resp = await analyseLogsAndSlack({
        logs: logEntries,
        time_from: new Date(getFromMs()).toISOString(),
        time_to: new Date(getToMs()).toISOString(),
        site_id: selectedSite || undefined,
        env: env || undefined,
        hostname: selectedHostnames[0] || undefined,
        deployment: selectedDeployments[0] || undefined,
        issue_description: issueDesc,
      });
      setAnalysisResult(resp);
    } catch (e) {
      setAnalysisError(e instanceof Error ? e.message : "Analysis failed.");
    } finally {
      setAnalysing(false);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [filteredLines, env, selectedSite, selectedHostnames, selectedDeployments, issueDesc, analysisLines]);

  /* count label */
  const countLabel = useMemo(() => {
    const showing = filteredLines.length;
    if (totalCount > MAX_DISPLAY_LINES) {
      return `Showing ${showing.toLocaleString()} of ${totalCount.toLocaleString()} lines (capped at ${MAX_DISPLAY_LINES.toLocaleString()})`;
    }
    return `Showing ${showing.toLocaleString()} lines`;
  }, [filteredLines.length, totalCount]);

  /* ═══════════════════════════════════════════════════════════════════════ */
  /*  RENDER                                                               */
  /* ═══════════════════════════════════════════════════════════════════════ */
  return (
    <div className="min-h-screen" style={{ background: PAGE_BG }}>
      {/* ── FILTER BAR ───────────────────────────────────────────────────── */}
      <div
        className="flex flex-wrap items-center gap-0.5 border-b px-3 py-1.5"
        style={{ background: FILTER_BG, borderColor: BORDER }}
      >
        <SingleSelect
          label="environment"
          options={ENVIRONMENTS}
          value={env}
          onChange={(v) => {
            setEnv(v);
            setActiveQuick(activeQuick);
          }}
        />

        <div className="mx-0.5 h-5 w-px" style={{ background: BORDER }} />

        <SingleSelect
          label="site"
          options={siteOptions}
          value={selectedSite}
          onChange={setSelectedSite}
          loading={loadingSites}
        />

        <div className="mx-0.5 h-5 w-px" style={{ background: BORDER }} />

        <MultiSelect
          label="hostname"
          options={hostnameOptions}
          selected={selectedHostnames}
          onChange={setSelectedHostnames}
          loading={loadingHosts}
          disabled={!selectedSite}
        />

        <div className="mx-0.5 h-5 w-px" style={{ background: BORDER }} />

        <MultiSelect
          label="deployment_name"
          options={deploymentOptions}
          selected={selectedDeployments}
          onChange={setSelectedDeployments}
          loading={loadingDeps}
          disabled={selectedHostnames.length !== 1}
        />

        <div className="mx-0.5 h-5 w-px" style={{ background: BORDER }} />

        {/* Search input */}
        <div className="flex items-center gap-1 rounded px-2 py-1" style={{ background: "#12151a" }}>
          <Search size={11} className="text-slate-500" />
          <input
            value={searchText}
            onChange={(e) => setSearchText(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && doFetch()}
            placeholder="Search"
            className="w-[120px] bg-transparent text-xs text-white placeholder-slate-500 outline-none"
          />
        </div>

        <div className="mx-0.5 h-5 w-px" style={{ background: BORDER }} />

        {/* Exclude input */}
        <div className="flex items-center gap-1 rounded px-2 py-1" style={{ background: "#12151a" }}>
          <X size={11} className="text-slate-500" />
          <input
            value={excludeText}
            onChange={(e) => setExcludeText(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && doFetch()}
            placeholder="Exclude"
            className="w-[100px] bg-transparent text-xs text-white placeholder-slate-500 outline-none"
          />
        </div>

        {/* Fetch button */}
        <button
          onClick={doFetch}
          disabled={loading || !selectedSite}
          className="ml-auto flex items-center gap-1.5 rounded bg-blue-600 px-3 py-1 text-xs font-medium text-white transition-colors hover:bg-blue-500 disabled:opacity-40"
        >
          {loading ? <Loader2 size={12} className="animate-spin" /> : <Search size={12} />}
          Run query
        </button>
      </div>

      {/* ── TIME RANGE PICKER ────────────────────────────────────────────── */}
      <div
        className="flex flex-wrap items-start gap-4 border-b px-4 py-3"
        style={{ background: FILTER_BG, borderColor: BORDER }}
      >
        {/* Quick range buttons */}
        <div className="flex flex-wrap gap-1.5">
          {QUICK_RANGES.map((q) => (
            <button
              key={q.label}
              onClick={() => applyQuickRange(q.label, q.ms)}
              className="rounded px-2.5 py-1 text-xs font-medium transition-colors"
              style={{
                background: activeQuick === q.label ? "#3b82f6" : "#1e293b",
                color: activeQuick === q.label ? "#fff" : "#94a3b8",
              }}
            >
              {q.label}
            </button>
          ))}
        </div>

        {/* Custom range inputs */}
        <div className="flex items-end gap-3">
          <div className="flex items-center gap-1.5">
            <span className="text-[10px] text-slate-500">From</span>
            <input
              type="date"
              value={fromDate}
              onChange={(e) => { setFromDate(e.target.value); setActiveQuick(""); }}
              className="rounded border bg-black/30 px-2 py-1 text-xs text-white outline-none focus:border-blue-500"
              style={{ borderColor: BORDER, colorScheme: "dark" }}
            />
            <input
              type="text"
              value={fromTime}
              onChange={(e) => { setFromTime(e.target.value); setActiveQuick(""); }}
              placeholder="HH:MM:SS"
              className="w-[80px] rounded border bg-black/30 px-2 py-1 text-xs text-white outline-none focus:border-blue-500"
              style={{ borderColor: BORDER }}
            />
            <span className="text-[10px] text-slate-600">UTC</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="text-[10px] text-slate-500">To</span>
            <input
              type="date"
              value={toDate}
              onChange={(e) => { setToDate(e.target.value); setActiveQuick(""); }}
              className="rounded border bg-black/30 px-2 py-1 text-xs text-white outline-none focus:border-blue-500"
              style={{ borderColor: BORDER, colorScheme: "dark" }}
            />
            <input
              type="text"
              value={toTime}
              onChange={(e) => { setToTime(e.target.value); setActiveQuick(""); }}
              placeholder="HH:MM:SS"
              className="w-[80px] rounded border bg-black/30 px-2 py-1 text-xs text-white outline-none focus:border-blue-500"
              style={{ borderColor: BORDER }}
            />
            <span className="text-[10px] text-slate-600">UTC</span>
          </div>
          <button
            onClick={doFetch}
            disabled={loading || !selectedSite}
            className="rounded bg-blue-600 px-3 py-1 text-xs font-medium text-white transition-colors hover:bg-blue-500 disabled:opacity-40"
          >
            Apply
          </button>
        </div>
      </div>

      {/* ── CHART: Log volume ────────────────────────────────────────────── */}
      {volumeData.length > 0 && (
        <div className="border-b" style={{ background: CHART_BG, borderColor: BORDER }}>
          <div className="flex items-center justify-between px-4 pt-3">
            <div className="flex items-center gap-1.5 text-xs text-slate-400">
              <span>Log volume</span>
            </div>
            <div className="flex items-center gap-3">
              {brushStart !== null && (
                <button
                  onClick={clearBrush}
                  className="flex items-center gap-1 text-xs text-cyan-400 hover:text-cyan-300"
                >
                  <RotateCcw size={11} /> Reset
                </button>
              )}
              <div className="text-right">
                <div className="text-[10px] uppercase tracking-wider text-slate-500">Lines</div>
                <div className="text-lg font-bold" style={{ color: COUNT_RED }}>
                  {totalCount.toLocaleString()}
                </div>
              </div>
            </div>
          </div>
          <div className="h-[160px] px-2 pb-1">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={volumeData}
                onMouseDown={onChartDown}
                onMouseMove={onChartMove}
                onMouseUp={onChartUp}
                barCategoryGap={0}
                barGap={0}
              >
                <XAxis
                  dataKey="label"
                  tick={{ fontSize: 10, fill: "#64748b" }}
                  tickLine={false}
                  axisLine={{ stroke: "#334155" }}
                  interval={Math.max(Math.floor(volumeData.length / 12), 1)}
                />
                <YAxis
                  scale="log"
                  domain={[0.5, "auto"]}
                  allowDataOverflow
                  tick={{ fontSize: 10, fill: "#64748b" }}
                  tickLine={false}
                  axisLine={false}
                  width={40}
                  tickFormatter={(v: number) => (v >= 1 ? String(Math.round(v)) : String(v))}
                />
                <Tooltip
                  contentStyle={{
                    background: "#1e293b",
                    border: `1px solid ${BORDER}`,
                    borderRadius: 6,
                    fontSize: 12,
                  }}
                  labelStyle={{ color: "#94a3b8" }}
                />
                <Bar dataKey="count" fill={CHART_GREEN} radius={[1, 1, 0, 0]} />
                {brushStart !== null && brushEnd !== null && (
                  <ReferenceArea
                    x1={fmtTimeAxis(Math.min(brushStart, brushEnd))}
                    x2={fmtTimeAxis(Math.max(brushStart, brushEnd))}
                    fill="rgba(56,189,248,0.18)"
                    stroke="rgba(56,189,248,0.5)"
                    strokeDasharray="3 3"
                  />
                )}
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="flex items-center gap-1.5 px-4 pb-2 text-[11px] text-slate-500">
            <span className="inline-block h-2.5 w-2.5 rounded-sm" style={{ background: CHART_GREEN }} />
            <span>{`{site="${selectedSite || "…"}"}`}</span>
          </div>
        </div>
      )}

      {/* Error */}
      {fetchError && (
        <div className="border-b px-4 py-3 text-xs text-red-300" style={{ borderColor: BORDER, background: "#1a0505" }}>
          {fetchError}
        </div>
      )}

      {/* Loading */}
      {loading && (
        <div className="flex items-center gap-2 border-b px-4 py-6 text-xs text-slate-400" style={{ borderColor: BORDER, background: CHART_BG }}>
          <Loader2 size={14} className="animate-spin text-blue-400" />
          Querying Loki…
        </div>
      )}

      {/* ── LOGS SECTION ─────────────────────────────────────────────────── */}
      {allLines.length > 0 && (
        <div style={{ background: LOG_BG }}>
          <button
            onClick={() => setLogsOpen(!logsOpen)}
            className="flex w-full items-center gap-1.5 border-b px-4 py-2 text-left text-xs font-medium text-slate-300 transition-colors hover:bg-white/[0.02]"
            style={{ borderColor: BORDER }}
          >
            <ChevronRight
              size={14}
              className={`text-slate-500 transition-transform ${logsOpen ? "rotate-90" : ""}`}
            />
            Logs
            <span className="ml-1 text-slate-500">{countLabel}</span>
          </button>

          {logsOpen && (
            <div className="max-h-[600px] overflow-y-auto" style={{ background: LOG_BG }}>
              {filteredLines.map((log, idx) => {
                const lvl = detectLevel(log.line);
                const barColor = LEVEL_BAR[lvl] || LEVEL_BAR.unknown;
                const isExpanded = expandedIdx === idx;
                const rowBg = idx % 2 === 0 ? LOG_BG : LOG_ALT;
                const tsMs = nsToMs(log.ts);
                const ts = fmtTs(tsMs);
                const host = log.labels?.hostname || "";
                const dep = log.labels?.deployment_name || "";

                return (
                  <div key={idx}>
                    <div
                      onClick={() => setExpandedIdx(isExpanded ? null : idx)}
                      className="group flex cursor-pointer items-stretch transition-colors"
                      style={{ background: isExpanded ? LOG_HOVER : rowBg }}
                      onMouseEnter={(e) => {
                        if (!isExpanded) e.currentTarget.style.background = LOG_HOVER;
                      }}
                      onMouseLeave={(e) => {
                        if (!isExpanded) e.currentTarget.style.background = rowBg;
                      }}
                    >
                      <div
                        className="w-[3px] shrink-0"
                        style={{ background: barColor }}
                      />
                      <div
                        className="flex-1 truncate px-3 py-[3px] text-slate-200"
                        style={{
                          fontFamily: "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace",
                          fontSize: "12px",
                          lineHeight: "20px",
                        }}
                      >
                        {ts}&nbsp;&nbsp;{host}&nbsp;&nbsp;{dep}&nbsp;&nbsp;{log.line}
                      </div>
                    </div>
                    {isExpanded && (
                      <div
                        className="border-b px-6 py-3"
                        style={{
                          background: "#141922",
                          borderColor: BORDER,
                        }}
                      >
                        <pre
                          className="whitespace-pre-wrap text-xs text-slate-300"
                          style={{
                            fontFamily: "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace",
                          }}
                        >
                          {log.line}
                        </pre>
                        {Object.keys(log.labels).length > 0 && (
                          <div className="mt-2 flex flex-wrap gap-1.5">
                            {Object.entries(log.labels).map(([k, v]) => (
                              <span
                                key={k}
                                className="rounded border px-1.5 py-0.5 text-[10px]"
                                style={{
                                  borderColor: BORDER,
                                  background: "#1a1d23",
                                  color: "#94a3b8",
                                }}
                              >
                                {k}=<span className="text-white">{v}</span>
                              </span>
                            ))}
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                );
              })}
              {filteredLines.length === 0 && (
                <div className="px-4 py-8 text-center text-xs text-slate-500">
                  No logs match the current filters.
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* Empty state */}
      {allLines.length === 0 && !loading && !fetchError && (
        <div className="flex flex-col items-center justify-center gap-3 py-20 text-center">
          <Search size={28} className="text-slate-600" />
          <p className="text-sm text-slate-400">
            Select a <span className="text-white">site</span> from the filter bar and run a query.
          </p>
        </div>
      )}

      {/* ── LOG ANALYSIS PANEL ───────────────────────────────────────────── */}
      <div
        className="border-t p-5"
        style={{ borderColor: BORDER, background: "#0d1117" }}
      >
        <div className="mb-4 flex items-center gap-2">
          <ClipboardList size={18} className="text-amber-300" />
          <h2 className="text-sm font-semibold text-slate-100">
            Log Analysis
          </h2>
        </div>

        <div className="space-y-3">
          <div>
            <label className="mb-1 block text-xs text-slate-400">
              Issue description *
            </label>
            <textarea
              rows={3}
              value={issueDesc}
              onChange={(e) => setIssueDesc(e.target.value)}
              placeholder="Describe the issue you want to understand from these logs…"
              className="w-full resize-none rounded border bg-black/30 px-3 py-2 text-xs text-white placeholder-slate-600 outline-none focus:border-blue-500"
              style={{ borderColor: BORDER }}
            />
          </div>

          <div className="flex items-center gap-4">
            <span className="text-xs text-slate-400">Lines to analyse:</span>
            <label className="flex items-center gap-1.5 text-xs text-slate-300 cursor-pointer">
              <input
                type="radio"
                name="analysisLines"
                checked={analysisLines === 1000}
                onChange={() => setAnalysisLines(1000)}
                className="accent-blue-500"
              />
              1000 lines (faster)
            </label>
            <label className="flex items-center gap-1.5 text-xs text-slate-300 cursor-pointer">
              <input
                type="radio"
                name="analysisLines"
                checked={analysisLines === 2000}
                onChange={() => setAnalysisLines(2000)}
                className="accent-blue-500"
              />
              2000 lines (thorough)
            </label>
          </div>

          <div className="flex items-center gap-3">
            <button
              onClick={handleAnalyse}
              disabled={analysing || !issueDesc.trim() || allLines.length === 0}
              className="flex items-center gap-1.5 rounded bg-blue-600 px-4 py-1.5 text-xs font-medium text-white transition-colors hover:bg-blue-500 disabled:opacity-40"
            >
              {analysing ? (
                <>
                  <Loader2 size={13} className="animate-spin" /> Analysing…
                </>
              ) : (
                <>
                  <Sparkles size={13} /> Analyse Logs
                </>
              )}
            </button>
            {allLines.length === 0 && (
              <span className="text-[11px] text-slate-600">Fetch logs first</span>
            )}
          </div>
        </div>

        {analysisError && (
          <div className="mt-4 rounded border px-4 py-3 text-xs text-red-300" style={{ borderColor: "#7f1d1d", background: "#1a0505" }}>
            {analysisError}
          </div>
        )}

        {analysisResult && (
          <div className="mt-4 space-y-3">
            <div className="flex flex-wrap gap-2 text-[11px] text-slate-500">
              <span className="rounded border px-2 py-0.5" style={{ borderColor: BORDER }}>
                Model: <span className="text-slate-300">{analysisResult.model_used}</span>
              </span>
              <span className="rounded border px-2 py-0.5" style={{ borderColor: BORDER }}>
                Logs: <span className="text-slate-300">{analysisResult.log_count}</span>
              </span>
            </div>
            <div
              className="prose-dark max-h-[500px] overflow-y-auto rounded border p-4"
              style={{ borderColor: BORDER, background: "#0b0e12" }}
            >
              <ReactMarkdown>{analysisResult.summary}</ReactMarkdown>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
