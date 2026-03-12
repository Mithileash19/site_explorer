// ──────────────────────────────────────────────
// Health
// ──────────────────────────────────────────────
export interface HealthResponse {
  status: "ok" | "degraded";
  version: string;
  model: string;
  faiss_entries: number;
  sites_loaded: number;
}

// ──────────────────────────────────────────────
// Bag Analysis
// ──────────────────────────────────────────────
export interface LogEntry {
  timestamp: number;
  datetime?: string;
  level: string;
  node: string;
  message: string;
}

export interface BagLogAnalysisResponse {
  bag_path: string;
  duration_secs: number;
  total_messages: number;
  error_count: number;
  warning_count: number;
  engine_hypothesis: string;
  llm_summary: string;
  log_entries: LogEntry[];
}

export interface TimelineBucket {
  t_start: number;
  t_end: number;
  count: number;
  error_count: number;
  warn_count: number;
}

export interface BagTimeline {
  bag_path: string;
  buckets: TimelineBucket[];
}

export interface MapDiffResponse {
  iou_score: number;
  diff_image_b64: string;
  message: string;
}

// ──────────────────────────────────────────────
// Investigation
// ──────────────────────────────────────────────
export interface RankedItem {
  description: string;
  confidence: number;
  evidence: string[];
}

export interface SimilarCase {
  id: string;
  title: string;
  description: string;
  similarity: number;
  resolution: string;
}

export interface OrchestratorResponse {
  status: string;
  confidence_score: number;
  human_intervention_required: boolean;
  issue_summary: string;
  ranked_causes: RankedItem[];
  ranked_solutions: RankedItem[];
  similar_cases: SimilarCase[];
  log_anomaly_summary: string;
  safety_assessment: string;
  raw_analysis: string;
}

// ──────────────────────────────────────────────
// Site / Fleet
// ──────────────────────────────────────────────
export interface SiteInfo {
  id: string;
  name: string;
  description?: string;
  robot_count?: number;
}

export interface NodeData {
  id: string;
  x: number;
  y: number;
  label?: string;
  type?: string;
}

export interface EdgeData {
  from: string;
  to: string;
  weight?: number;
}

export interface SiteData {
  nodes: NodeData[];
  edges: EdgeData[];
  spots?: Record<string, unknown>[];
  storage?: Record<string, unknown>[];
}

export interface MapConfig {
  resolution: number;
  origin: [number, number, number];
  width: number;
  height: number;
}

export interface FleetStatusResponse {
  site_id: string;
  online_robots: number;
  total_robots: number;
  active_missions: number;
  alerts: string[];
}

// ──────────────────────────────────────────────
// Site Map (sootballs_sites)
// ──────────────────────────────────────────────
export interface SiteMapMeta {
  resolution: number;
  origin: [number, number, number];
  width: number;
  height: number;
  b64: string;
}

export interface SiteMapSpot {
  _idx: number;   // unique per-array index, added client-side
  name: string;
  type: string;
  x: number;
  y: number;
  yaw: number;
  robot: string;
  color: string;
}

export interface SiteMapRack {
  section: string;
  row: string;
  label: string;
  x: number;
  y: number;
  orientation: number;
  direction: string;
}

export interface SiteMapRegion {
  type: string;
  id: string | number;
  name: string;
  polygon: [number, number][];
  color: string;
}

export interface SiteMapRobot {
  id: number;
  name: string;
}

export interface SiteMapNode {
  id: number;
  x: number;
  y: number;
  parkable: boolean;
  radius: number;
  meta_kind?: string;
  spin_mode?: number | null;
  spin_turn?: number | null;
}

export interface SiteMapEdge {
  id: number;
  node1: number;
  node2: number;
  directed: boolean;
  speed_scale_estimate?: string;
}

export interface SiteMapData {
  spots: SiteMapSpot[];
  racks: SiteMapRack[];
  regions: SiteMapRegion[];
  robots: SiteMapRobot[];
  nodes: SiteMapNode[];
  edges: SiteMapEdge[];
}

// ──────────────────────────────────────────────
// AR Markers
// ──────────────────────────────────────────────

/** A single AR fiducial marker in world (ROS map frame) coordinates. */
export interface SiteMapMarker {
  /** Aruco marker ID */
  id: number;
  /** World X position (metres) */
  x: number;
  /** World Y position (metres) */
  y: number;
  /** World Z position (metres, usually 0) */
  z: number;
  /** Yaw orientation (radians) */
  yaw: number;
}

export interface SiteMarkers {
  markers: SiteMapMarker[];
}

// ──────────────────────────────────────────────
// Git Branch Info
// ──────────────────────────────────────────────

export interface CommitInfo {
  hash: string;
  message: string;
  date: string;
}

export interface BranchInfo {
  /** short branch name e.g. "mncyok001" or "main" */
  branch: string;
  /** full remote ref e.g. "origin/mncyok001" */
  ref: string;
  /** true when the site has its own dedicated branch */
  is_site_specific: boolean;
  /** true when a manual override is active */
  is_override: boolean;
  last_commit: CommitInfo | null;
  available_branches: string[];
}

export interface BranchCleanupPlan {
  /** branches that will be kept (main + valid site names) */
  valid_branches: string[];
  /** local remote-tracking refs that would be removed */
  invalid_branches: string[];
  /** site IDs with no dedicated branch (will use main) */
  sites_without_own_branch: string[];
  /** total remote branches before cleanup */
  total_branches: number;
}

export interface BranchCleanupResult {
  /** branches whose local tracking ref was successfully deleted */
  removed: string[];
  /** branches that were kept */
  kept: string[];
  /** branches that failed to be removed */
  errors: string[];
}

// ──────────────────────────────────────────────
// SSE streaming
// ──────────────────────────────────────────────
export type SSEStepType =
  | "start"
  | "bag_analysis"
  | "similarity_search"
  | "llm_analysis"
  | "complete"
  | "error";

export interface SSEEvent {
  step: SSEStepType;
  message: string;
  data?: OrchestratorResponse;
  error?: string;
}

// ──────────────────────────────────────────────
// API error envelope
// ──────────────────────────────────────────────
export interface APIError {
  detail: string;
  status?: number;
}
