"use client";

import { useState } from "react";
import { useForm } from "react-hook-form";
import { streamInvestigation } from "@/lib/api";
import type { OrchestratorResponse, SSEEvent } from "@/lib/types";
import ConfidenceGauge          from "@/components/investigation/ConfidenceGauge";
import RankedCausesPanel        from "@/components/investigation/RankedCausesPanel";
import SimilarCasesTable        from "@/components/investigation/SimilarCasesTable";
import HumanInterventionBanner  from "@/components/investigation/HumanInterventionBanner";
import ReactMarkdown             from "react-markdown";
import {
  SearchCode,
  Loader2,
  CheckCircle,
  XCircle,
  ChevronRight,
} from "lucide-react";

interface FormValues {
  title:          string;
  description:    string;
  bag_path?:      string;
  site_id?:       string;
  grafana_link?:  string;
  sw_version?:    string;
  config_changed: boolean;
}

type StepStatus = "pending" | "running" | "done" | "error";

interface Step {
  id:      string;
  label:   string;
  status:  StepStatus;
  message: string;
}

const STEP_LABELS: Record<string, string> = {
  start:            "Initialising",
  bag_analysis:     "Analysing bag logs",
  similarity_search:"Searching past incidents",
  llm_analysis:     "Running AI analysis",
  complete:         "Complete",
};

function StepRow({ step }: { step: Step }) {
  return (
    <div className="flex items-center gap-2.5 text-sm">
      {step.status === "running" && <Loader2 size={14} className="animate-spin text-blue-400 shrink-0" />}
      {step.status === "done"    && <CheckCircle size={14} className="text-green-400 shrink-0" />}
      {step.status === "error"   && <XCircle     size={14} className="text-red-400 shrink-0" />}
      {step.status === "pending" && <div className="w-3.5 h-3.5 rounded-full border border-slate-600 shrink-0" />}
      <span className={
        step.status === "running" ? "text-blue-300" :
        step.status === "done"    ? "text-green-300" :
        step.status === "error"   ? "text-red-300"   :
        "text-slate-500"
      }>
        {step.label}
      </span>
      {step.message && step.status === "running" && (
        <span className="text-xs text-slate-600 truncate">{step.message}</span>
      )}
    </div>
  );
}

export default function InvestigatePage() {
  const { register, handleSubmit, formState: { errors } } = useForm<FormValues>({
    defaultValues: { config_changed: false },
  });

  const [steps,    setSteps]    = useState<Step[]>([]);
  const [result,   setResult]   = useState<OrchestratorResponse | null>(null);
  const [running,  setRunning]  = useState(false);
  const [errMsg,   setErrMsg]   = useState("");
  const [tab,      setTab]      = useState<"causes" | "solutions" | "similar" | "raw">("causes");

  function onSubmit(data: FormValues) {
    setResult(null);
    setErrMsg("");
    setRunning(true);

    // Build initial step list
    const initial: Step[] = Object.keys(STEP_LABELS).map((id) => ({
      id,
      label:   STEP_LABELS[id],
      status:  "pending",
      message: "",
    }));
    setSteps(initial);

    const unsub = streamInvestigation(
      {
        title:       data.title,
        description: data.description,
        bag_path:    data.bag_path || undefined,
        site_id:     data.site_id  || undefined,
      },
      (ev: SSEEvent) => {
        setSteps((prev) =>
          prev.map((s) => {
            if (s.id === ev.step) return { ...s, status: "running", message: ev.message };
            // mark previous steps done
            const prevIdx = Object.keys(STEP_LABELS).indexOf(s.id);
            const curIdx  = Object.keys(STEP_LABELS).indexOf(ev.step);
            if (prevIdx < curIdx && s.status === "running") return { ...s, status: "done" };
            return s;
          })
        );

        if (ev.step === "complete" && ev.data) {
          setResult(ev.data);
          setSteps((prev) => prev.map((s) => ({ ...s, status: "done" })));
          setRunning(false);
          unsub();
        }
        if (ev.step === "error") {
          setErrMsg(ev.error ?? "Unknown error from server.");
          setRunning(false);
          unsub();
        }
      },
      () => {
        if (running) {
          setErrMsg("Connection to server lost. Please retry.");
          setRunning(false);
        }
      }
    );
  }

  return (
    <div className="p-6 max-w-5xl mx-auto animate-fade-in">
      <div className="flex items-center gap-3 mb-6">
        <SearchCode size={20} className="text-blue-400" />
        <div>
          <h1 className="text-xl font-bold text-slate-100">Incident Investigation</h1>
          <p className="text-xs text-slate-400">AI-powered root cause analysis with historical similarity search</p>
        </div>
      </div>

      <div className="grid lg:grid-cols-2 gap-6">
        {/* Form */}
        <div>
          <form onSubmit={handleSubmit(onSubmit)} className="card space-y-4">
            <h2 className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Incident Details</h2>

            <div>
              <label className="text-xs text-slate-400 mb-1 block">Title *</label>
              <input
                className={`input ${errors.title ? "border-red-500" : ""}`}
                placeholder="e.g. Robot stopped mid-mission at zone 3"
                {...register("title", { required: true })}
              />
            </div>

            <div>
              <label className="text-xs text-slate-400 mb-1 block">Description *</label>
              <textarea
                rows={4}
                className={`input resize-none ${errors.description ? "border-red-500" : ""}`}
                placeholder="Describe what happened, when, and any observed symptoms…"
                {...register("description", { required: true })}
              />
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="text-xs text-slate-400 mb-1 block">Bag file path</label>
                <input className="input text-sm" placeholder="/data/bags/file.bag" {...register("bag_path")} />
              </div>
              <div>
                <label className="text-xs text-slate-400 mb-1 block">Site ID</label>
                <input className="input text-sm" placeholder="site-alpha" {...register("site_id")} />
              </div>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="text-xs text-slate-400 mb-1 block">Grafana link</label>
                <input className="input text-sm" placeholder="https://…" {...register("grafana_link")} />
              </div>
              <div>
                <label className="text-xs text-slate-400 mb-1 block">SW version</label>
                <input className="input text-sm" placeholder="v1.3.2" {...register("sw_version")} />
              </div>
            </div>

            <label className="flex items-center gap-2 cursor-pointer text-sm text-slate-300 select-none">
              <input type="checkbox" className="rounded border-slate-600 bg-slate-800" {...register("config_changed")} />
              Config / firmware changed recently
            </label>

            <button
              type="submit"
              className="btn btn-primary w-full py-3 gap-2"
              disabled={running}
            >
              {running
                ? <><Loader2 size={16} className="animate-spin" /> Investigating…</>
                : <><SearchCode size={16} /> Start Investigation</>}
            </button>
          </form>

          {/* SSE progress */}
          {steps.length > 0 && (
            <div className="card mt-4 space-y-2">
              <p className="text-xs text-slate-500 uppercase tracking-wide font-medium mb-1">Progress</p>
              {steps.map((s) => <StepRow key={s.id} step={s} />)}
            </div>
          )}
        </div>

        {/* Results */}
        <div>
          {errMsg && (
            <div className="card border-red-800/30 bg-red-900/10 text-red-300 text-sm mb-4">{errMsg}</div>
          )}

          {result && (
            <div className="space-y-4 animate-fade-in">
              {/* Top row: gauge + banner */}
              <div className="card flex items-center gap-5">
                <ConfidenceGauge score={result.confidence_score} />
                <div className="flex-1">
                  <p className="text-xs text-slate-500 mb-1">Confidence score</p>
                  <p className="text-2xl font-bold text-slate-100">
                    {(result.confidence_score * 100).toFixed(0)}%
                  </p>
                  <p className="text-xs text-slate-500 mt-1">
                    {result.human_intervention_required
                      ? "Below automatic resolution threshold"
                      : "Within automatic resolution range"}
                  </p>
                </div>
              </div>

              <HumanInterventionBanner
                required={result.human_intervention_required}
                confidence={result.confidence_score}
              />

              {/* Tabs */}
              <div>
                <div className="flex gap-1 mb-3 border-b border-slate-700/50 pb-2">
                  {([
                    { id: "causes",    label: "Root Causes"  },
                    { id: "solutions", label: "Solutions"    },
                    { id: "similar",   label: "Similar Cases"},
                    { id: "raw",       label: "Raw Analysis" },
                  ] as const).map(({ id, label }) => (
                    <button
                      key={id}
                      onClick={() => setTab(id)}
                      className={`flex items-center gap-1 px-2.5 py-1.5 text-xs rounded-md transition-colors ${
                        tab === id
                          ? "bg-blue-600/20 text-blue-400 border border-blue-600/30"
                          : "text-slate-400 hover:text-slate-200"
                      }`}
                    >
                      {label}
                      {id === "similar" && result.similar_cases.length > 0 && (
                        <span className="badge badge-blue text-[10px] px-1 py-0">{result.similar_cases.length}</span>
                      )}
                    </button>
                  ))}
                </div>

                <div className="card">
                  {tab === "causes" && (
                    <RankedCausesPanel
                      items={result.ranked_causes}
                      title="Ranked Root Causes"
                      accentColour="#ef4444"
                    />
                  )}
                  {tab === "solutions" && (
                    <RankedCausesPanel
                      items={result.ranked_solutions}
                      title="Recommended Solutions"
                      accentColour="#22c55e"
                    />
                  )}
                  {tab === "similar" && (
                    <div>
                      <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">
                        Similar Past Incidents
                      </h3>
                      <SimilarCasesTable cases={result.similar_cases} />
                    </div>
                  )}
                  {tab === "raw" && (
                    <div className="prose-dark max-h-[400px] overflow-y-auto">
                      <ReactMarkdown>{result.raw_analysis}</ReactMarkdown>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}

          {!result && !running && steps.length === 0 && (
            <div className="card h-64 flex flex-col items-center justify-center text-center gap-3">
              <SearchCode size={36} className="text-slate-700" />
              <p className="text-sm text-slate-500">Fill in the incident details and start<br />the investigation to see results here.</p>
              <div className="flex items-center gap-1.5 text-xs text-slate-600">
                <ChevronRight size={12} /> Uses FAISS similarity + LLM analysis
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
