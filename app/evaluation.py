"""Retrieval and answer quality evaluation with reproducible reports.

Faithfulness: when Ragas can run with a local/mock-compatible setup we use it;
otherwise we expose a clearly labeled `lexical_support` diagnostic that is
NOT claimed to be faithfulness. Automated metrics do not guarantee factual
correctness.
"""

from __future__ import annotations

import json
import logging
import math
import platform
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from app.config import Settings, get_settings
from app.retrieval import RetrievalMode

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def recall_at_k(relevant: set[str], retrieved: list[str], k: int) -> float:
    if not relevant:
        return 0.0
    top = retrieved[:k]
    return len(relevant.intersection(top)) / len(relevant)


def precision_at_k(relevant: set[str], retrieved: list[str], k: int) -> float:
    top = retrieved[:k]
    if not top:
        return 0.0
    return len(relevant.intersection(top)) / len(top)


def mrr(relevant: set[str], retrieved: list[str]) -> float:
    for i, cid in enumerate(retrieved, start=1):
        if cid in relevant:
            return 1.0 / i
    return 0.0


def dcg_at_k(relevances: list[float], k: int) -> float:
    total = 0.0
    for i, rel in enumerate(relevances[:k], start=1):
        total += (2**rel - 1) / math.log2(i + 1)
    return total


def ndcg_at_k(relevant: set[str], retrieved: list[str], k: int) -> float:
    rels = [1.0 if cid in relevant else 0.0 for cid in retrieved[:k]]
    ideal = sorted(rels, reverse=True)
    # Ideal should include all relevant up to k
    ideal = [1.0] * min(len(relevant), k) + [0.0] * max(0, k - min(len(relevant), k))
    ideal = ideal[:k]
    idcg = dcg_at_k(ideal, k)
    if idcg == 0:
        return 0.0
    return dcg_at_k(rels, k) / idcg


def normalize_answer(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\u0600-\u06FF\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def exact_match(prediction: str, reference: str) -> float:
    return 1.0 if normalize_answer(prediction) == normalize_answer(reference) else 0.0


def token_f1(prediction: str, reference: str) -> float:
    pred_toks = normalize_answer(prediction).split()
    ref_toks = normalize_answer(reference).split()
    if not pred_toks or not ref_toks:
        return 0.0
    common: dict[str, int] = {}
    for t in pred_toks:
        common[t] = common.get(t, 0) + 1
    overlap = 0
    for t in ref_toks:
        if common.get(t, 0) > 0:
            overlap += 1
            common[t] -= 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_toks)
    recall = overlap / len(ref_toks)
    return 2 * precision * recall / (precision + recall)


def lexical_support_score(answer: str, contexts: list[str]) -> float:
    """Diagnostic overlap of answer tokens with context — NOT faithfulness."""
    if not answer or not contexts:
        return 0.0
    a = set(normalize_answer(answer).split())
    c = set(normalize_answer(" ".join(contexts)).split())
    if not a:
        return 0.0
    return len(a & c) / len(a)


# ---------------------------------------------------------------------------
# Benchmark I/O
# ---------------------------------------------------------------------------


def load_benchmark(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "items" not in data:
        raise ValueError("Benchmark JSON must contain an 'items' list")
    return data


def resolve_relevance_labels(
    benchmark: dict[str, Any],
    chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    """Fill relevant_chunk_ids from relevant_spans when IDs are missing.

    Gold spans are verified phrases from the sample corpus. Chunk IDs may
    change when chunking settings change; re-resolution keeps evaluation usable.
    """
    resolved_items = []
    for item in benchmark["items"]:
        item = dict(item)
        ids = list(item.get("relevant_chunk_ids") or [])
        spans = item.get("relevant_spans") or []
        if not ids and spans:
            matched: list[str] = []
            for span in spans:
                span_norm = span.strip().lower()
                for ch in chunks:
                    if span_norm in ch["text"].lower():
                        matched.append(ch["chunk_id"])
            # Preserve order, unique
            seen = set()
            ids = []
            for cid in matched:
                if cid not in seen:
                    seen.add(cid)
                    ids.append(cid)
        item["relevant_chunk_ids"] = ids
        resolved_items.append(item)
    out = dict(benchmark)
    out["items"] = resolved_items
    return out


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------


RetrieveFn = Callable[..., list[str]]


class EvaluationRunner:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def evaluate_retrieval(
        self,
        benchmark: dict[str, Any],
        retrieve_fn: Callable[[str, str, int, bool], list[str]],
        *,
        modes: list[tuple[str, str, bool]] | None = None,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """modes: list of (label, retrieval_mode, rerank)."""
        modes = modes or [
            ("Dense", "dense", False),
            ("BM25", "sparse", False),
            ("Hybrid", "hybrid", False),
            ("Hybrid + Rerank", "hybrid", True),
        ]
        items = benchmark["items"]
        # Skip insufficient-evidence items for pure retrieval ranking metrics
        # unless they have relevant_chunk_ids.
        ranked_items = [it for it in items if it.get("relevant_chunk_ids")]

        table_rows = []
        per_mode_details: dict[str, Any] = {}

        for label, mode, do_rerank in modes:
            recalls, precisions, mrrs, ndcgs, latencies = [], [], [], [], []
            for item in ranked_items:
                t0 = time.perf_counter()
                retrieved = retrieve_fn(item["question"], mode, top_k, do_rerank)
                latencies.append((time.perf_counter() - t0) * 1000)
                relevant = set(item["relevant_chunk_ids"])
                recalls.append(recall_at_k(relevant, retrieved, top_k))
                precisions.append(precision_at_k(relevant, retrieved, top_k))
                mrrs.append(mrr(relevant, retrieved))
                ndcgs.append(ndcg_at_k(relevant, retrieved, top_k))

            row = {
                "Method": label,
                f"Recall@{top_k}": float(np.mean(recalls)) if recalls else 0.0,
                f"Precision@{top_k}": float(np.mean(precisions)) if precisions else 0.0,
                "MRR": float(np.mean(mrrs)) if mrrs else 0.0,
                f"NDCG@{top_k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
                "Latency_ms_avg": float(np.mean(latencies)) if latencies else 0.0,
                "Latency_ms_p50": float(np.median(latencies)) if latencies else 0.0,
                "Latency_ms_p95": float(np.percentile(latencies, 95)) if latencies else 0.0,
            }
            table_rows.append(row)
            per_mode_details[label] = row

        df = pd.DataFrame(table_rows)
        return {
            "table": df,
            "per_mode": per_mode_details,
            "sample_count": len(ranked_items),
            "top_k": top_k,
        }

    def evaluate_answers(
        self,
        benchmark: dict[str, Any],
        answer_fn: Callable[[str], dict[str, Any]],
    ) -> dict[str, Any]:
        items = benchmark["items"]
        ems, f1s, latencies = [], [], []
        abstentions = 0
        insufficient_items = 0
        correct_abstentions = 0
        lexical_supports = []

        for item in items:
            t0 = time.perf_counter()
            out = answer_fn(item["question"])
            latencies.append((time.perf_counter() - t0) * 1000)
            answer = out.get("answer", "")
            abstained = bool(out.get("abstained"))
            contexts = out.get("contexts") or [
                s.get("text_preview", "") for s in out.get("sources", [])
            ]
            is_insufficient = bool(item.get("insufficient_evidence"))
            if is_insufficient:
                insufficient_items += 1
            if abstained:
                abstentions += 1
                if is_insufficient:
                    correct_abstentions += 1
                ems.append(1.0 if is_insufficient else 0.0)
                f1s.append(1.0 if is_insufficient else 0.0)
            else:
                ref = item.get("reference_answer", "")
                ems.append(exact_match(answer, ref))
                f1s.append(token_f1(answer, ref))
                lexical_supports.append(lexical_support_score(answer, contexts))

        return {
            "exact_match": float(np.mean(ems)) if ems else 0.0,
            "token_f1": float(np.mean(f1s)) if f1s else 0.0,
            "abstention_rate": abstentions / len(items) if items else 0.0,
            "correct_abstention_rate": (
                correct_abstentions / insufficient_items if insufficient_items else None
            ),
            "lexical_support_mean": (
                float(np.mean(lexical_supports)) if lexical_supports else None
            ),
            "lexical_support_note": (
                "Diagnostic token overlap with retrieved context; "
                "NOT a faithfulness metric."
            ),
            "latency_ms_avg": float(np.mean(latencies)) if latencies else 0.0,
            "latency_ms_p50": float(np.median(latencies)) if latencies else 0.0,
            "latency_ms_p95": float(np.percentile(latencies, 95)) if latencies else 0.0,
            "sample_count": len(items),
            "faithfulness": {
                "status": "unavailable_without_llm_judge",
                "note": (
                    "Ragas faithfulness typically requires an LLM judge. "
                    "This project does not call paid APIs. Use human review "
                    "or configure a local judge separately."
                ),
            },
        }

    def write_report(
        self,
        *,
        retrieval_results: dict[str, Any],
        answer_results: dict[str, Any] | None,
        benchmark: dict[str, Any],
        experiment_name: str | None = None,
        extra_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        ts = datetime.now(timezone.utc)
        report_id = ts.strftime("%Y%m%dT%H%M%SZ")
        if experiment_name:
            report_id = f"{experiment_name}_{report_id}"

        metrics_dir = self.settings.reports_dir / "metrics"
        figures_dir = self.settings.reports_dir / "figures"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        figures_dir.mkdir(parents=True, exist_ok=True)

        df: pd.DataFrame = retrieval_results["table"]
        csv_path = metrics_dir / f"{report_id}_retrieval.csv"
        json_path = metrics_dir / f"{report_id}_report.json"
        df.to_csv(csv_path, index=False)

        # Simple matplotlib-free chart via CSV + optional plotly in UI.
        # Also write a minimal SVG bar chart for Recall.
        recall_col = [c for c in df.columns if c.startswith("Recall@")]
        if recall_col:
            self._write_bar_svg(
                df,
                x="Method",
                y=recall_col[0],
                path=figures_dir / f"{report_id}_recall.svg",
                title=recall_col[0],
            )
        if "Latency_ms_avg" in df.columns:
            self._write_bar_svg(
                df,
                x="Method",
                y="Latency_ms_avg",
                path=figures_dir / f"{report_id}_latency.svg",
                title="Avg Latency (ms)",
            )

        report = {
            "report_id": report_id,
            "timestamp": ts.isoformat(),
            "benchmark_version": benchmark.get("version", "unknown"),
            "benchmark_name": benchmark.get("name", "benchmark"),
            "sample_count": retrieval_results["sample_count"],
            "models": {
                "embedding_model": self.settings.embedding_model,
                "reranker_model": self.settings.reranker_model,
                "llm_model": self.settings.llm_model,
                "mock_models": self.settings.mock_models,
            },
            "retrieval_settings": {
                "top_k": retrieval_results["top_k"],
                "rrf_k": self.settings.rrf_k,
                "enable_reranking": self.settings.enable_reranking,
            },
            "chunking_settings": {
                "chunk_size": self.settings.chunk_size,
                "chunk_overlap": self.settings.chunk_overlap,
                "chunk_strategy": self.settings.chunk_strategy,
            },
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
            },
            "retrieval_metrics": df.to_dict(orient="records"),
            "answer_metrics": answer_results,
            "extra_config": extra_config or {},
            "artifacts": {
                "csv": str(csv_path),
                "json": str(json_path),
            },
            "limitations": [
                "Automated metrics do not guarantee factual correctness.",
                "Default cross-encoder is English-oriented.",
                "Relevance labels may need regeneration if chunking changes.",
            ],
        }
        json_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return report

    def _write_bar_svg(
        self, df: pd.DataFrame, x: str, y: str, path: Path, title: str
    ) -> None:
        width, height = 640, 360
        margin = 60
        values = df[y].tolist()
        labels = df[x].tolist()
        max_v = max(values) if values else 1.0
        max_v = max_v if max_v > 0 else 1.0
        bar_w = (width - 2 * margin) / max(1, len(values))
        parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
            f'<rect width="100%" height="100%" fill="#0b1c2c"/>',
            f'<text x="{margin}" y="30" fill="#7dd3fc" font-size="16">{title}</text>',
        ]
        for i, (lab, val) in enumerate(zip(labels, values)):
            h = ((height - 2 * margin) * (val / max_v))
            x0 = margin + i * bar_w + 10
            y0 = height - margin - h
            parts.append(
                f'<rect x="{x0}" y="{y0}" width="{bar_w - 20}" height="{h}" fill="#22d3ee" rx="6"/>'
            )
            parts.append(
                f'<text x="{x0}" y="{height - margin + 18}" fill="#cbd5e1" font-size="11">{lab}</text>'
            )
            parts.append(
                f'<text x="{x0}" y="{y0 - 6}" fill="#e2e8f0" font-size="11">{val:.3f}</text>'
            )
        parts.append("</svg>")
        path.write_text("\n".join(parts), encoding="utf-8")

    def list_reports(self) -> list[dict[str, Any]]:
        metrics_dir = self.settings.reports_dir / "metrics"
        reports = []
        if not metrics_dir.exists():
            return reports
        for path in sorted(metrics_dir.glob("*_report.json"), reverse=True):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                reports.append(
                    {
                        "report_id": data.get("report_id", path.stem),
                        "path": str(path),
                        "timestamp": data.get("timestamp"),
                        "benchmark_version": data.get("benchmark_version"),
                        "sample_count": data.get("sample_count"),
                    }
                )
            except Exception:
                continue
        return reports


def run_pipeline_evaluation(
    pipeline,
    *,
    benchmark_path: Path | None = None,
    modes: list[str] | None = None,
    top_k: int = 5,
    include_answer_eval: bool = False,
    experiment_name: str | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    settings = settings or get_settings()
    runner = EvaluationRunner(settings)
    path = benchmark_path or (settings.evaluation_dir / "benchmark.json")
    benchmark = load_benchmark(path)
    chunks = pipeline.db.list_chunks()
    benchmark = resolve_relevance_labels(benchmark, chunks)

    mode_map = {
        "dense": ("Dense", "dense", False),
        "sparse": ("BM25", "sparse", False),
        "bm25": ("BM25", "sparse", False),
        "hybrid": ("Hybrid", "hybrid", False),
        "hybrid_rerank": ("Hybrid + Rerank", "hybrid", True),
    }
    selected_modes = None
    if modes:
        selected_modes = [mode_map[m] for m in modes if m in mode_map]

    def retrieve_fn(question: str, mode: str, k: int, do_rerank: bool) -> list[str]:
        out = pipeline.search(
            question, retrieval_mode=mode, top_k=k, rerank=do_rerank  # type: ignore[arg-type]
        )
        return [r.chunk_id for r in out["results"]]

    retrieval_results = runner.evaluate_retrieval(
        benchmark, retrieve_fn, modes=selected_modes, top_k=top_k
    )

    answer_results = None
    if include_answer_eval:

        def answer_fn(question: str) -> dict[str, Any]:
            # Ephemeral conversation each time
            return pipeline.chat(question)

        answer_results = runner.evaluate_answers(benchmark, answer_fn)

    report = runner.write_report(
        retrieval_results=retrieval_results,
        answer_results=answer_results,
        benchmark=benchmark,
        experiment_name=experiment_name,
    )
    return report
