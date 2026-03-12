"""
services/ai/llm_service.py
───────────────────────────
Single LLM call returning all 5 analysis sections in one response.
Works with Ollama (local) or any OpenAI-compatible endpoint.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

from openai import OpenAI

from core.config import settings
from core.logging import get_logger

logger = get_logger(__name__)

_SECTIONS = [
    "log_timeline", "node_analysis", "error_analysis",
    "pattern_analysis", "technical_conclusion",
]
_EMPTY_RESULT: Dict[str, Any] = {k: "" for k in _SECTIONS}

_DELIMITERS = {
    "log_timeline":         "###LOG_TIMELINE###",
    "node_analysis":        "###NODE_ANALYSIS###",
    "error_analysis":       "###ERROR_ANALYSIS###",
    "pattern_analysis":     "###PATTERN_ANALYSIS###",
    "technical_conclusion": "###CONCLUSION###",
}

_MAX_LOG_LINES = 120

_STOPWORDS = {
    "the","a","an","is","it","in","on","at","of","to","and","or","was","were",
    "be","been","being","have","has","had","do","does","did","will","would",
    "could","should","not","no","with","for","from","by","are","that","this",
    "but","if","as","so","up","its","my","we","i","robot","system","issue",
    "error","problem","occurred","happened","during","after","before","when",
}


def _extract_keywords(description: str) -> List[str]:
    tokens = re.findall(r"[a-zA-Z0-9_/]+", description.lower())
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 2]


def _find_relevant_logs(logs: List[Dict], keywords: List[str]) -> List[Dict]:
    if not keywords:
        return [e for e in logs if e["log_level"] in ("ERROR", "FATAL")]
    relevant = []
    for e in logs:
        text = (e["message"] + " " + e["node_name"]).lower()
        if e["log_level"] in ("ERROR", "FATAL") or any(kw in text for kw in keywords):
            relevant.append(e)
    return relevant


def _parse_sections(raw: str) -> Dict[str, str]:
    result: Dict[str, str] = {}
    delim_order = list(_DELIMITERS.items())
    for i, (key, token) in enumerate(delim_order):
        start = raw.find(token)
        if start == -1:
            result[key] = ""
            continue
        content_start = start + len(token)
        if i + 1 < len(delim_order):
            next_token = delim_order[i + 1][1]
            end        = raw.find(next_token, content_start)
            content    = raw[content_start:end] if end != -1 else raw[content_start:]
        else:
            content = raw[content_start:]
        result[key] = content.strip()
    return result


class LLMService:
    def __init__(self) -> None:
        # If an OpenAI key is provided, point at OpenAI; otherwise use Ollama
        if settings.openai_api_key:
            self.client = OpenAI(api_key=settings.openai_api_key)
            self.model  = "gpt-4o"
        else:
            self.client = OpenAI(base_url=settings.ollama_base_url, api_key="ollama")
            self.model  = settings.ollama_model
        logger.info("LLMService: model=%s", self.model)

    def _call(self, system: str, user: str, max_tokens: int = 3500) -> str:
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                temperature=0.1,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as exc:
            logger.error("LLM call failed: %s", exc)
            return f"LLM Error: {exc}"

    def generate_log_incident_summary(
        self,
        robot_name: str,
        incident_time: str,
        filtered_logs: List[Dict[str, Any]],
        priority_logs: List[Dict[str, Any]],
        issue_description: str = "",
        engine_hypothesis: str = "",
    ) -> Dict[str, Any]:
        """Single LLM call producing 5 structured analysis sections."""
        if not filtered_logs:
            r = dict(_EMPTY_RESULT)
            r["log_timeline"] = "No logs found in the specified time window."
            return r

        desc_clean    = issue_description.strip()
        keywords      = _extract_keywords(desc_clean) if desc_clean else []
        relevant_logs = _find_relevant_logs(filtered_logs, keywords)

        def fmt(entries: List[Dict]) -> str:
            return "\n".join(
                f"[{e['log_level']:5s}] {e['datetime']}  {e['node_name'][:40]:40s}  {e['message']}"
                for e in entries
            )

        err_entries   = [e for e in filtered_logs if e["log_level"] in ("ERROR", "FATAL", "WARN")]
        other_entries = [e for e in filtered_logs if e not in err_entries]
        cap_others    = max(0, _MAX_LOG_LINES - len(err_entries))
        trimmed       = sorted(err_entries + other_entries[:cap_others], key=lambda e: e["timestamp"])

        log_block = fmt(trimmed)
        err_block = fmt([e for e in trimmed if e["log_level"] in ("ERROR", "FATAL", "WARN")]) or "(none)"

        rel_seen, rel_dedup = set(), []
        for e in sorted(relevant_logs, key=lambda x: x["timestamp"]):
            key = (e["timestamp"], e["node_name"], e["message"])
            if key not in rel_seen:
                rel_seen.add(key)
                rel_dedup.append(e)
        rel_block = fmt(rel_dedup) or "(No log entries matched the reported issue keywords)"

        total  = len(filtered_logs)
        n_err  = sum(1 for e in filtered_logs if e["log_level"] in ("ERROR", "FATAL"))
        n_warn = sum(1 for e in filtered_logs if e["log_level"] == "WARN")
        span   = f"{filtered_logs[0]['datetime']}  →  {filtered_logs[-1]['datetime']}"
        nodes  = ", ".join(sorted({e["node_name"] for e in filtered_logs}))

        hypothesis_block = (
            f"\nRULE-BASED HYPOTHESIS FROM SIGNAL ANALYSIS:\n{engine_hypothesis}\n"
            if engine_hypothesis else ""
        )

        description_block = ""
        if desc_clean:
            description_block = (
                f"\n══ ENGINEER'S REPORTED ISSUE ══\n"
                f'  "{desc_clean}"\n'
                f"  Keywords: {', '.join(keywords) or '(none)'}\n"
                f"  Matching log entries ({len(rel_dedup)}):\n{rel_block}\n"
                f"══════════════════════════════\n"
            )

        system_prompt = (
            "You are a senior ROS/AMR field engineer performing root-cause analysis. "
            "Read the reported issue and find log evidence that explains or contradicts it. "
            "Cite exact node names, timestamps, and verbatim messages as evidence. "
            "Do NOT fabricate log entries. Do NOT repeat the full log in your output.\n\n"
            f"Output EXACTLY these five sections in order, each starting with its delimiter:\n"
            + "\n".join(f"  {d}" for d in _DELIMITERS.values())
        )

        user_prompt = (
            f"Robot: {robot_name} | Incident: {incident_time} | Span: {span}\n"
            f"Counts: {total} total | {n_err} ERROR/FATAL | {n_warn} WARN\n"
            f"Active nodes: {nodes}\n"
            f"{description_block}"
            f"{hypothesis_block}\n"
            f"FULL /rosout LOG ({len(trimmed)} entries):\n{log_block}\n\n"
            f"ERROR/WARN entries:\n{err_block}"
        )

        raw    = self._call(system_prompt, user_prompt)
        result = _parse_sections(raw)
        for k in _SECTIONS:
            if k not in result:
                result[k] = ""
        result["_raw"] = raw
        return result

    def generate_investigation_summary(self, prompt: str) -> str:
        """Feed a pre-built investigation prompt (from LogAnalyzerEngine) to the LLM."""
        system = (
            "You are an expert ROS/AMR diagnostics engineer. "
            "Analyse the incident and give a structured response with: "
            "Root Cause, Confidence Level, Evidence, and Recommended Next Steps."
        )
        return self._call(system, prompt, max_tokens=1500)
