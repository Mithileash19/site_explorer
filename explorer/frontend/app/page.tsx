"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { fetchHealth, listSites, getFleetStatus } from "@/lib/api";
import type { HealthResponse, SiteInfo, FleetStatusResponse } from "@/lib/types";
import {
  Activity,
  Bot,
  Database,
  Map,
  PackageSearch,
  SearchCode,
  AlertTriangle,
  CheckCircle,
  Loader2,
} from "lucide-react";

function Stat({ label, value, sub }: { label: string; value: string | number; sub?: string }) {
  return (
    <div className="card flex flex-col gap-1">
      <p className="text-xs text-slate-500 font-medium uppercase tracking-wide">{label}</p>
      <p className="text-3xl font-bold text-slate-100">{value}</p>
      {sub && <p className="text-xs text-slate-500">{sub}</p>}
    </div>
  );
}

function SiteCard({ site }: { site: SiteInfo }) {
  const [fleet, setFleet] = useState<FleetStatusResponse | null>(null);

  useEffect(() => {
    getFleetStatus(site.id)
      .then(setFleet)
      .catch(() => null);
  }, [site.id]);

  const onlineRatio = fleet ? fleet.online_robots / Math.max(fleet.total_robots, 1) : null;
  const healthy = onlineRatio !== null && onlineRatio >= 0.8;

  return (
    <div className="card hover:border-blue-600/40 transition-colors">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Map size={16} className="text-blue-400" />
          <span className="font-semibold text-sm text-slate-200">{site.name}</span>
        </div>
        {fleet && (
          <span className={`badge ${healthy ? "badge-green" : "badge-yellow"}`}>
            {healthy ? "healthy" : "degraded"}
          </span>
        )}
      </div>
      <p className="text-xs text-slate-500 mb-4 leading-relaxed">{site.description ?? ""}</p>
      <div className="flex items-center justify-between text-xs text-slate-400">
        <span>{fleet ? `${fleet.online_robots} / ${fleet.total_robots} robots online` : `${site.robot_count ?? 0} robots`}</span>
        {fleet && fleet.active_missions > 0 && (
          <span className="badge badge-blue">{fleet.active_missions} active</span>
        )}
      </div>
      {fleet && fleet.alerts.length > 0 && (
        <div className="mt-3 border-t border-slate-700/50 pt-3 space-y-1">
          {fleet.alerts.slice(0, 2).map((a, i) => (
            <p key={i} className="flex items-center gap-1.5 text-xs text-amber-400">
              <AlertTriangle size={11} /> {a}
            </p>
          ))}
        </div>
      )}
      <div className="mt-4">
        <Link href={`/fleet?site=${site.id}`} className="btn btn-ghost w-full justify-center text-xs">
          Open Map
        </Link>
      </div>
    </div>
  );
}

const QUICK_LINKS = [
  { href: "/bags",        icon: PackageSearch, label: "Analyze a Bag",    desc: "Upload & extract ROS logs"  },
  { href: "/investigate", icon: SearchCode,    label: "Investigate",      desc: "AI-powered incident analysis"},
  { href: "/assistant",   icon: Bot,           label: "AI Assistant",     desc: "Chat with streaming AI"     },
];

export default function DashboardPage() {
  const [health, setHealth]   = useState<HealthResponse | null>(null);
  const [sites,  setSites]    = useState<SiteInfo[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.allSettled([fetchHealth(), listSites()]).then(([h, s]) => {
      if (h.status === "fulfilled") setHealth(h.value);
      if (s.status === "fulfilled") setSites(s.value);
      setLoading(false);
    });
  }, []);

  return (
    <div className="p-6 max-w-7xl mx-auto animate-fade-in">
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-2xl font-bold text-slate-100">Platform Dashboard</h1>
          <p className="text-sm text-slate-400 mt-0.5">AMR fleet operations &amp; AI-powered incident intelligence</p>
        </div>
        {health && (
          <span className={`badge ${health.status === "ok" ? "badge-green" : "badge-yellow"} text-sm px-3 py-1`}>
            {health.status === "ok"
              ? <><CheckCircle size={12} /> System Online</>
              : <><AlertTriangle size={12} /> Degraded</>}
          </span>
        )}
      </div>

      {/* Stats row */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        {loading ? (
          Array.from({ length: 4 }).map((_, i) => (
            <div key={i} className="card h-24 flex items-center justify-center">
              <Loader2 className="animate-spin text-slate-600" size={20} />
            </div>
          ))
        ) : (
          <>
            <Stat label="Sites"          value={sites.length}                sub="active deployments" />
            <Stat label="AI Model"        value={health?.model ?? "—"}        sub={health?.status === "ok" ? "connected" : "offline"} />
            <Stat label="Known Incidents" value={health?.faiss_entries ?? 0}  sub="in vector DB" />
            <Stat label="Sites Loaded"    value={health?.sites_loaded ?? 0}   sub="map data available" />
          </>
        )}
      </div>

      {/* Quick actions */}
      <section className="mb-8">
        <h2 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-3">Quick Actions</h2>
        <div className="grid grid-cols-2 lg:grid-cols-3 gap-3">
          {QUICK_LINKS.map(({ href, icon: Icon, label, desc }) => (
            <Link
              key={href}
              href={href}
              className="card hover:border-blue-500/40 hover:bg-blue-600/5 transition-all group cursor-pointer flex flex-col gap-2"
            >
              <div className="flex items-center gap-2">
                <div className="p-1.5 bg-blue-600/15 rounded-md group-hover:bg-blue-600/25 transition-colors">
                  <Icon size={16} className="text-blue-400" />
                </div>
                <span className="text-sm font-medium text-slate-200">{label}</span>
              </div>
              <p className="text-xs text-slate-500">{desc}</p>
            </Link>
          ))}
        </div>
      </section>

      {/* Site grid */}
      <section>
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-sm font-semibold text-slate-400 uppercase tracking-wider">Active Sites</h2>
          <Activity size={14} className="text-slate-600" />
        </div>
        {loading ? (
          <div className="flex items-center gap-2 text-slate-500 text-sm">
            <Loader2 className="animate-spin" size={16} /> Loading sites…
          </div>
        ) : sites.length === 0 ? (
          <div className="card text-center py-12">
            <Database size={32} className="text-slate-600 mx-auto mb-3" />
            <p className="text-sm text-slate-400">No sites loaded. Configure <code className="text-blue-400">SITES_ROOT</code> or enable git sync.</p>
          </div>
        ) : (
          <div className="grid sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
            {sites.map((s) => <SiteCard key={s.id} site={s} />)}
          </div>
        )}
      </section>
    </div>
  );
}
