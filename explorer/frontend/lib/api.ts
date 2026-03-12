import axios from "axios";
import type {
  HealthResponse,
  BagLogAnalysisResponse,
  BagTimeline,
  MapDiffResponse,
  OrchestratorResponse,
  SiteInfo,
  SiteData,
  MapConfig,
  FleetStatusResponse,
  SSEEvent,
  SiteMapMeta,
  SiteMapData,
  SiteMarkers,
  BranchInfo,
  BranchCleanupPlan,
  BranchCleanupResult,
} from "./types";

const BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

const http = axios.create({
  baseURL: `${BASE}/api/v1`,
  timeout: 120_000,
  headers: { "Content-Type": "application/json" },
});

// Rewrite the Axios error message to the FastAPI `detail` string so every
// catch block automatically receives a human-readable message instead of the
// generic "Request failed with status code NNN".
http.interceptors.response.use(
  res => res,
  err => {
    const detail = err?.response?.data?.detail;
    if (detail && typeof detail === "string") {
      err.message = detail;
    }
    return Promise.reject(err);
  },
);

// ── Health ────────────────────────────────────────────────────────────────────

export async function fetchHealth(): Promise<HealthResponse> {
  const { data } = await http.get<HealthResponse>("/health");
  return data;
}

// ── Sites ─────────────────────────────────────────────────────────────────────

export async function listSites(): Promise<SiteInfo[]> {
  const { data } = await http.get<SiteInfo[]>("/sites");
  return data;
}

export async function getSiteConfig(siteId: string): Promise<MapConfig> {
  const { data } = await http.get<MapConfig>(`/sites/${siteId}/config`);
  return data;
}

export async function getSiteMapB64(siteId: string, darkMode = true): Promise<string | null> {
  try {
    const { data } = await http.get<{ b64: string; width: number; height: number }>(
      `/sites/${siteId}/map?dark_mode=${darkMode}`
    );
    return data.b64 || null;
  } catch {
    return null;
  }
}

/** @deprecated Use getSiteMapB64 instead */
export function getSiteMapUrl(siteId: string, darkMode = true): string {
  return `${BASE}/api/v1/sites/${siteId}/map?dark_mode=${darkMode}`;
}

export async function getSiteData(siteId: string): Promise<SiteData> {
  const { data } = await http.get<SiteData>(`/sites/${siteId}/data`);
  return data;
}

export async function getFleetStatus(siteId: string): Promise<FleetStatusResponse> {
  const { data } = await http.get<FleetStatusResponse>(`/fleet/status?site_id=${siteId}`);
  return data;
}

// ── Bags ──────────────────────────────────────────────────────────────────────

export async function uploadBag(file: File): Promise<{ bag_path: string; size_mb: number }> {
  const form = new FormData();
  form.append("file", file);
  const { data } = await http.post<{ bag_path: string; size_mb: number }>(
    "/bags/upload",
    form,
    { headers: { "Content-Type": "multipart/form-data" } }
  );
  return data;
}

export async function fetchTimeline(bagPath: string, nBuckets = 60): Promise<BagTimeline> {
  const { data } = await http.get<BagTimeline>(
    `/bags/timeline?bag_path=${encodeURIComponent(bagPath)}&n_buckets=${nBuckets}`
  );
  return data;
}

export async function analyzeBag(
  bagPath: string,
  windowStart?: number,
  windowEnd?: number
): Promise<BagLogAnalysisResponse> {
  const { data } = await http.post<BagLogAnalysisResponse>("/bags/analyze", {
    bag_path: bagPath,
    window_start: windowStart,
    window_end: windowEnd,
  });
  return data;
}

export async function runMapDiff(
  bagPath: string,
  topicOverride?: string
): Promise<MapDiffResponse> {
  const { data } = await http.post<MapDiffResponse>("/bags/mapdiff", {
    bag_path: bagPath,
    topic_override: topicOverride,
  });
  return data;
}

// ── Investigation ─────────────────────────────────────────────────────────────

export async function investigate(payload: {
  title: string;
  description: string;
  bag_path?: string;
  site_id?: string;
  grafana_link?: string;
  slack_url?: string;
  sw_version?: string;
  config_changed?: boolean;
}): Promise<OrchestratorResponse> {
  const { data } = await http.post<OrchestratorResponse>("/investigate", payload);
  return data;
}

/**
 * Opens a native EventSource to the SSE investigation endpoint.
 * Returns an unsubscribe function.
 */
export function streamInvestigation(
  params: {
    title: string;
    description: string;
    bag_path?: string;
    site_id?: string;
  },
  onEvent: (event: SSEEvent) => void,
  onError?: (err: Event) => void
): () => void {
  const qs = new URLSearchParams({
    title: params.title,
    description: params.description,
    ...(params.bag_path ? { bag_path: params.bag_path } : {}),
    ...(params.site_id ? { site_id: params.site_id } : {}),
  });
  const es = new EventSource(`${BASE}/api/v1/investigate/stream?${qs}`);
  es.onmessage = (e) => {
    try {
      onEvent(JSON.parse(e.data) as SSEEvent);
    } catch {
      /* ignore malformed frames */
    }
  };
  if (onError) es.onerror = onError;
  return () => es.close();
}

// ── Site Map (sootballs_sites) ────────────────────────────────────────────────

export async function listSiteMapSites(): Promise<{ id: string; name: string }[]> {
  const { data } = await http.get("/sitemap/sites");
  return data;
}

export async function getSiteMapMeta(siteId: string, darkMode = true): Promise<SiteMapMeta> {
  const { data } = await http.get<SiteMapMeta>(`/sitemap/${siteId}/map?dark_mode=${darkMode}`);
  return data;
}

export async function getSiteMapData(siteId: string): Promise<SiteMapData> {
  const { data } = await http.get<SiteMapData>(`/sitemap/${siteId}/data`);
  data.nodes = data.nodes ?? [];
  data.edges = data.edges ?? [];
  // Stamp a unique index on every spot so duplicate names never cause key collisions
  data.spots = data.spots.map((s, i) => ({ ...s, _idx: i }));
  return data;
}

export async function getSiteMarkers(siteId: string): Promise<SiteMarkers> {
  const { data } = await http.get<SiteMarkers>(`/sitemap/${siteId}/markers`);
  return data;
}

// ── Sitemap — Git Branch ──────────────────────────────────────────────────────

export async function getSiteBranchInfo(siteId: string): Promise<BranchInfo> {
  const { data } = await http.get<BranchInfo>(`/sitemap/${siteId}/branch`);
  return data;
}

export async function setSiteBranch(siteId: string, branch: string): Promise<BranchInfo> {
  const { data } = await http.post<BranchInfo>(`/sitemap/${siteId}/branch`, { branch });
  return data;
}

export async function clearSiteBranch(siteId: string): Promise<BranchInfo> {
  const { data } = await http.delete<BranchInfo>(`/sitemap/${siteId}/branch`);
  return data;
}

export async function syncSiteRepo(): Promise<{ branches_found: number }> {
  const { data } = await http.post<{ branches_found: number }>("/sitemap/sync");
  return data;
}

export async function getBranchCleanupPlan(): Promise<BranchCleanupPlan> {
  const { data } = await http.get<BranchCleanupPlan>("/sitemap/cleanup/plan");
  return data;
}

export async function runBranchCleanup(): Promise<BranchCleanupResult> {
  const { data } = await http.post<BranchCleanupResult>("/sitemap/cleanup");
  return data;
}
