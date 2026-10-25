"""Streamlit dashboard for Advanced RAG Evaluation."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import httpx
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="Advanced RAG Evaluation",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --bg: #0a1628;
  --panel: #12233a;
  --card: #16304f;
  --cyan: #22d3ee;
  --blue: #3b82f6;
  --text: #e2e8f0;
  --muted: #94a3b8;
}

.stApp {
  background: radial-gradient(1200px 600px at 10% -10%, #123055 0%, var(--bg) 45%),
              linear-gradient(180deg, #07101d 0%, var(--bg) 100%);
  color: var(--text);
  font-family: 'DM Sans', sans-serif;
}

h1, h2, h3 { color: var(--text) !important; letter-spacing: -0.02em; }
.block-container { padding-top: 1.5rem; max-width: 1200px; }

div[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0d1b2e 0%, #0a1422 100%);
  border-right: 1px solid #1e3a5f;
}

.hero-brand {
  font-size: 1.75rem; font-weight: 700; color: var(--cyan);
  margin-bottom: 0.25rem;
}
.hero-sub { color: var(--muted); margin-bottom: 1.25rem; }

.card {
  background: linear-gradient(145deg, var(--card), var(--panel));
  border: 1px solid #23466d;
  border-radius: 16px;
  padding: 1rem 1.15rem;
  margin-bottom: 0.75rem;
  box-shadow: 0 8px 24px rgba(0,0,0,0.25);
}
.card-title { color: var(--cyan); font-weight: 600; margin-bottom: 0.35rem; }
.muted { color: var(--muted); font-size: 0.9rem; }
.mono { font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; }
.score { color: var(--cyan); font-weight: 600; }
.source-chip {
  display: inline-block; background: #1d4ed8; color: white;
  border-radius: 999px; padding: 0.15rem 0.6rem; font-size: 0.75rem;
  margin-right: 0.35rem; margin-top: 0.25rem;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def api_client() -> httpx.Client:
    return httpx.Client(base_url=API_URL, timeout=120.0)


def safe_get(path: str):
    try:
        with api_client() as client:
            r = client.get(path)
            r.raise_for_status()
            return r.json()
    except Exception as exc:
        st.error(f"API error ({path}): {exc}")
        return None


def safe_post(path: str, **kwargs):
    try:
        with api_client() as client:
            r = client.post(path, **kwargs)
            if r.status_code >= 400:
                st.error(r.json().get("detail", r.text))
                return None
            return r.json()
    except Exception as exc:
        st.error(f"API error ({path}): {exc}")
        return None


def safe_delete(path: str):
    try:
        with api_client() as client:
            r = client.delete(path)
            if r.status_code >= 400:
                st.error(r.json().get("detail", r.text))
                return None
            return r.json()
    except Exception as exc:
        st.error(f"API error ({path}): {exc}")
        return None


def page_header():
    st.markdown('<div class="hero-brand">Advanced RAG Evaluation</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-sub">Hybrid retrieval · Cross-encoder reranking · Grounded generation · Measurable quality</div>',
        unsafe_allow_html=True,
    )


def chat_page():
    page_header()
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = None
    if "messages" not in st.session_state:
        st.session_state.messages = []

    col_cfg, col_chat = st.columns([1, 2.4])
    with col_cfg:
        st.markdown("#### Retrieval settings")
        mode = st.selectbox("Mode", ["hybrid", "dense", "sparse"], index=0)
        top_k = st.slider("Top-K", 1, 20, 5)
        rerank = st.toggle("Rerank", value=True)
        if st.button("Clear conversation", use_container_width=True):
            if st.session_state.conversation_id:
                safe_delete(f"/conversations/{st.session_state.conversation_id}")
            st.session_state.conversation_id = None
            st.session_state.messages = []
            st.rerun()

    with col_chat:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("sources"):
                    chips = []
                    for s in msg["sources"]:
                        page = s.get("page")
                        page_bit = f" p.{page}" if page is not None else ""
                        chips.append(
                            f'<span class="source-chip">{s["document"]}{page_bit}</span>'
                        )
                    st.markdown(" ".join(chips), unsafe_allow_html=True)
                if msg.get("latency_ms") is not None:
                    st.caption(f"Latency: {msg['latency_ms']:.0f} ms")

        prompt = st.chat_input("Ask a question about your documents…")
        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.spinner("Retrieving and generating…"):
                out = safe_post(
                    "/chat",
                    json={
                        "question": prompt,
                        "conversation_id": st.session_state.conversation_id,
                        "retrieval_mode": mode,
                        "top_k": top_k,
                        "rerank": rerank,
                    },
                )
            if out:
                st.session_state.conversation_id = out["conversation_id"]
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": out["answer"],
                        "sources": out.get("sources", []),
                        "latency_ms": out.get("latency_ms"),
                    }
                )
            st.rerun()


def documents_page():
    page_header()
    st.markdown("#### Upload documents")
    uploaded = st.file_uploader(
        "PDF, DOCX, or TXT",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
    )
    if uploaded and st.button("Ingest selected", type="primary"):
        for f in uploaded:
            with st.spinner(f"Ingesting {f.name}…"):
                try:
                    with api_client() as client:
                        r = client.post(
                            "/documents/upload",
                            files={"file": (f.name, f.getvalue(), f.type or "application/octet-stream")},
                        )
                    if r.status_code >= 400:
                        st.error(f"{f.name}: {r.json().get('detail', r.text)}")
                    else:
                        data = r.json()
                        if data.get("duplicate"):
                            st.warning(f"{f.name}: duplicate — {data.get('message')}")
                        else:
                            st.success(
                                f"{f.name}: indexed ({data['chunk_count']} chunks)"
                            )
                except Exception as exc:
                    st.error(str(exc))

    st.markdown("#### Indexed documents")
    data = safe_get("/documents")
    if not data:
        return
    if not data["documents"]:
        st.info("No documents indexed yet.")
        return
    for doc in data["documents"]:
        c1, c2 = st.columns([4, 1])
        with c1:
            st.markdown(
                f'<div class="card"><div class="card-title">{doc["filename"]}</div>'
                f'<div class="muted mono">{doc["document_id"]} · {doc["file_type"]} · '
                f'{doc["chunk_count"]} chunks · status: {doc["processing_status"]}</div></div>',
                unsafe_allow_html=True,
            )
        with c2:
            if st.button("Delete", key=f"del_{doc['document_id']}"):
                safe_delete(f"/documents/{doc['document_id']}")
                st.rerun()


def explorer_page():
    page_header()
    st.markdown("#### Retrieval explorer")
    query = st.text_input("Query", placeholder="What is the refund period?")
    c1, c2, c3 = st.columns(3)
    with c1:
        mode = st.selectbox("Mode", ["hybrid", "dense", "sparse"], key="ex_mode")
    with c2:
        top_k = st.slider("Top-K", 1, 20, 5, key="ex_k")
    with c3:
        rerank = st.toggle("Rerank", value=True, key="ex_rerank")

    if st.button("Search", type="primary") and query.strip():
        out = safe_post(
            "/search",
            json={
                "query": query,
                "retrieval_mode": mode,
                "top_k": top_k,
                "rerank": rerank,
            },
        )
        if not out:
            return
        st.caption(
            f"Mode={out['retrieval_mode']} · reranked={out['reranked']} · "
            f"latency={out['latency_ms']:.1f} ms"
        )
        for i, r in enumerate(out["results"], start=1):
            st.markdown(
                f'<div class="card"><div class="card-title">#{i} · '
                f'<span class="score">{r["score"]:.4f}</span></div>'
                f'<div class="muted">{r["document"]} · page={r["page"]} · '
                f'{r["chunk_id"]}</div>'
                f'<div style="margin-top:0.5rem">{r["text"]}</div>'
                f'<div class="muted mono" style="margin-top:0.5rem">'
                f'dense={r.get("dense_score")} · bm25={r.get("bm25_score")} · '
                f'rerank={r.get("rerank_score")}</div></div>',
                unsafe_allow_html=True,
            )


def evaluation_page():
    page_header()
    st.markdown("#### Evaluation dashboard")
    c1, c2 = st.columns(2)
    with c1:
        top_k = st.slider("Eval Top-K", 1, 20, 5, key="eval_k")
        include_answers = st.toggle("Include answer metrics", value=False)
        experiment = st.text_input("Experiment name", value="retrieval_compare")
    with c2:
        modes = st.multiselect(
            "Modes",
            ["dense", "sparse", "hybrid", "hybrid_rerank"],
            default=["dense", "sparse", "hybrid", "hybrid_rerank"],
        )

    if st.button("Run evaluation", type="primary"):
        with st.spinner("Running benchmark…"):
            out = safe_post(
                "/evaluation/run",
                json={
                    "top_k": top_k,
                    "modes": modes,
                    "include_answer_eval": include_answers,
                    "experiment_name": experiment,
                },
            )
        if out:
            st.success(f"Report saved: {out['report_id']}")
            df = pd.DataFrame(out["metrics"]["retrieval"])
            st.dataframe(df, use_container_width=True)
            recall_cols = [c for c in df.columns if c.startswith("Recall@")]
            if recall_cols:
                fig = px.bar(
                    df,
                    x="Method",
                    y=recall_cols[0],
                    title=recall_cols[0],
                    color_discrete_sequence=["#22d3ee"],
                )
                fig.update_layout(
                    paper_bgcolor="#0a1628",
                    plot_bgcolor="#12233a",
                    font_color="#e2e8f0",
                )
                st.plotly_chart(fig, use_container_width=True)
            if "Latency_ms_avg" in df.columns:
                fig2 = px.bar(
                    df,
                    x="Method",
                    y="Latency_ms_avg",
                    title="Average latency (ms)",
                    color_discrete_sequence=["#3b82f6"],
                )
                fig2.update_layout(
                    paper_bgcolor="#0a1628",
                    plot_bgcolor="#12233a",
                    font_color="#e2e8f0",
                )
                st.plotly_chart(fig2, use_container_width=True)

    st.markdown("#### Previous reports")
    reports = safe_get("/evaluation/results")
    if reports and reports.get("reports"):
        for r in reports["reports"][:20]:
            st.markdown(
                f'<div class="card"><div class="card-title">{r["report_id"]}</div>'
                f'<div class="muted mono">{r["path"]}</div></div>',
                unsafe_allow_html=True,
            )
            if st.button("Load metrics", key=f"load_{r['report_id']}"):
                try:
                    data = json.loads(Path(r["path"]).read_text(encoding="utf-8"))
                    st.json(data.get("retrieval_metrics"))
                except Exception as exc:
                    st.error(str(exc))
    else:
        st.info("No evaluation reports yet.")


def main():
    with st.sidebar:
        st.markdown("### Navigation")
        page = st.radio(
            "Section",
            ["Chat", "Documents", "Retrieval Explorer", "Evaluation Dashboard"],
            label_visibility="collapsed",
        )
        health = safe_get("/health")
        if health:
            st.markdown("---")
            st.caption(f"API: {health['status']}")
            st.caption(f"Index: {'ready' if health['index_ready'] else 'not ready'}")
            st.caption(f"Mock models: {health['mock_models']}")
            st.caption(f"Embed: {health['embedding_model']}")

    if page == "Chat":
        chat_page()
    elif page == "Documents":
        documents_page()
    elif page == "Retrieval Explorer":
        explorer_page()
    else:
        evaluation_page()


if __name__ == "__main__":
    main()
