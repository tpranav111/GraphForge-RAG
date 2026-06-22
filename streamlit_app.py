from __future__ import annotations

import json
import io
import os
import hashlib
import sys
import time
from datetime import datetime, timezone
from typing import Any, Iterable
from urllib.parse import urlparse

import streamlit as st

ROOT = os.path.dirname(__file__)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from advanced_graphrag import (  # noqa: E402
    DEFAULT_HF_MODEL_PROFILE,
    GraphRAGConfig,
    GraphRAGEngine,
    HashingEmbeddingModel,
    HeuristicLLM,
    InMemoryArtifactStore,
    InMemoryGraphStore,
    InMemoryVectorStore,
)
from advanced_graphrag.domain import Answer, Evidence, RetrievalBundle  # noqa: E402
from advanced_graphrag.ingestion.extraction import LLMJsonGraphExtractor, RuleBasedExtractor  # noqa: E402
from advanced_graphrag.models.base import ChatMessage  # noqa: E402
from advanced_graphrag.models.openai_compatible import (  # noqa: E402
    OpenAICompatibleEmbeddingModel,
    OpenAICompatibleLLM,
)
from advanced_graphrag.models.huggingface import (  # noqa: E402
    SentenceTransformerEmbeddingModel,
    TransformersCausalLLM,
)


APP_TITLE = "GraphRAG Workbench"
FAST_GENERATION_REPO_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_GENERATION_REPO_ID = "Qwen/Qwen2.5-1.5B-Instruct"
BALANCED_GENERATION_REPO_ID = DEFAULT_GENERATION_REPO_ID
QUALITY_GENERATION_REPO_ID = "Qwen/Qwen3-8B"
LIGHT_GENERATION_REPO_ID = FAST_GENERATION_REPO_ID
CUSTOM_GENERATION_REPO_ID = "tpranav/EmpPolRAG_Qwen4-4B-Instruct"
QUALITY_EMBEDDING_REPO_ID = "BAAI/bge-m3"
QUALITY_EMBEDDING_DIMENSION = 1024
DEFAULT_EMBEDDING_REPO_ID = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_EMBEDDING_DIMENSION = 384
FALLBACK_EMBEDDING_REPO_ID = DEFAULT_EMBEDDING_REPO_ID
FALLBACK_EMBEDDING_DIMENSION = DEFAULT_EMBEDDING_DIMENSION
DEFAULT_HF_DEVICE_MAP = ""
DEFAULT_HF_MAX_NEW_TOKENS = 384
DEFAULT_HF_ROUTER_PROVIDER = "hf-inference"
TRACE_TEXT_LIMIT = 6000
TRACE_VECTOR_SAMPLE = 8
SUPPORTED_UPLOAD_TYPES = ["txt", "md", "pdf"]

SYNTHETIC_CORPUS = {
    "atlas_ops": (
        "Atlas Robotics builds sensor fusion systems for Harbor Drones. "
        "Harbor Drones integrates Raven Battery packs for long bridge flights. "
        "Sentinel Analytics monitors Harbor Drones failures and reports risk alerts to Atlas Robotics."
    ),
    "raven_supply": (
        "Raven Battery supplies modular cells to Atlas Robotics. "
        "Marina City deployed Harbor Drones for bridge inspection. "
        "Atlas Robotics coordinates the deployment with Dockside Control."
    ),
    "graphrag_research": (
        "Microsoft GraphRAG builds entity graphs and community summaries for global sensemaking. "
        "HippoRAG uses fact embeddings and Personalized PageRank over graph memory. "
        "FalkorDB GraphRAG combines vector search, full text search, traversal, and symbolic queries."
    ),
}

QUERY_PRESETS = (
    "Who supplies modular cells to Atlas Robotics?",
    "How is Atlas Robotics connected to Raven Battery?",
    "Give an overall overview of the robotics deployment themes",
    "cypher: entity:Harbor Drones",
    "How do graph walks and embeddings work together in GraphRAG?",
)


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide", initial_sidebar_state="expanded")
    _inject_css()
    _ensure_session()
    _render_sidebar()
    _render_header()

    corpus_tab, ask_tab, graph_tab, evidence_tab, logs_tab, settings_tab = st.tabs(
        ["Corpus", "Ask", "Graph", "Evidence", "Logs", "Settings"]
    )
    with corpus_tab:
        _render_corpus_tab()
    with ask_tab:
        _render_ask_tab()
    with graph_tab:
        _render_graph_tab()
    with evidence_tab:
        _render_evidence_tab()
    with logs_tab:
        _render_logs_tab()
    with settings_tab:
        _render_settings_tab()


def _ensure_session() -> None:
    defaults = {
        "runtime": "Local deterministic",
        "hash_dimension": 256,
        "chat_base_url": "https://router.huggingface.co/v1",
        "embedding_base_url": "http://localhost:8080/v1",
        "hf_router_provider": DEFAULT_HF_ROUTER_PROVIDER,
        "hf_hub_token": os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "",
        "hf_device_map": DEFAULT_HF_DEVICE_MAP,
        "hf_torch_dtype": "auto",
        "hf_max_new_tokens": DEFAULT_HF_MAX_NEW_TOKENS,
        "hf_enable_thinking": False,
        "hf_trust_remote_code": False,
        "hf_local_files_only": True,
        "hf_embedding_device": "",
        "hf_embedding_batch_size": 16,
        "hf_cache_folder": "",
        "chat_api_token": os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "",
        "embedding_api_token": os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "",
        "reuse_chat_token_for_embeddings": True,
        "generation_model": DEFAULT_GENERATION_REPO_ID,
        "embedding_model": DEFAULT_EMBEDDING_REPO_ID,
        "embedding_dimension": DEFAULT_EMBEDDING_DIMENSION,
        "chunk_size": 700,
        "chunk_overlap": 90,
        "max_context_tokens": 6000,
        "max_chunks": 16,
        "max_entities": 32,
        "max_relationships": 48,
        "max_communities": 6,
        "ppr_alpha": 0.65,
        "min_relationship_confidence": 0.30,
        "enable_symbolic_queries": True,
        "enable_community_summaries": True,
        "graph_extraction_mode": "LLM JSON",
        "llm_extraction_max_entities": 24,
        "llm_extraction_fallback": True,
        "query_text": QUERY_PRESETS[0],
        "last_retrieval": None,
        "last_answer": None,
        "events": [],
        "trace_events": [],
        "trace_max_events": 1000,
        "ingestion_runs": [],
        "engine_signature": None,
        "last_runtime": "Local deterministic",
        "hosted_preflight_signature": None,
        "hosted_preflight_ok": False,
        "hosted_preflight_report": {},
        "local_preflight_report": {},
        "local_auto_repair_notice": "",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)
    _repair_empty_endpoint_defaults()
    if "engine" not in st.session_state:
        _rebuild_engine()


def _build_config() -> GraphRAGConfig:
    chunk_size = int(st.session_state.chunk_size)
    chunk_overlap = _effective_chunk_overlap()
    return GraphRAGConfig(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        max_context_tokens=int(st.session_state.max_context_tokens),
        max_entities=int(st.session_state.max_entities),
        max_relationships=int(st.session_state.max_relationships),
        max_chunks=int(st.session_state.max_chunks),
        max_communities=int(st.session_state.max_communities),
        ppr_alpha=float(st.session_state.ppr_alpha),
        min_relationship_confidence=float(st.session_state.min_relationship_confidence),
        enable_symbolic_queries=bool(st.session_state.enable_symbolic_queries),
        enable_community_summaries=bool(st.session_state.enable_community_summaries),
    )


def _build_graph_extractor(llm: Any, *, runtime: str) -> Any:
    mode = str(st.session_state.get("graph_extraction_mode", "LLM JSON"))
    if mode == "Rule-based":
        _trace(
            category="app_event",
            operation="graph_extractor_selected",
            status="ok",
            component="engine",
            detail={"mode": mode, "runtime": runtime},
            summary="Using rule-based graph extractor",
        )
        return RuleBasedExtractor()

    fallback = RuleBasedExtractor() if bool(st.session_state.get("llm_extraction_fallback", True)) else None
    extractor = LLMJsonGraphExtractor(
        llm,
        max_entities_per_chunk=int(st.session_state.get("llm_extraction_max_entities", 24)),
        fallback_extractor=fallback,
    )
    _trace(
        category="app_event",
        operation="graph_extractor_selected",
        status="ok",
        component="engine",
        detail={
            "mode": mode,
            "runtime": runtime,
            "max_entities_per_chunk": int(st.session_state.get("llm_extraction_max_entities", 24)),
            "fallback_enabled": fallback is not None,
        },
        summary="Using LLM JSON graph extractor",
    )
    return extractor


def _rebuild_engine() -> bool:
    errors = _runtime_config_errors()
    if errors:
        for error in errors:
            st.error(error)
        return False
    if st.session_state.get("runtime") == "Hosted HuggingFace-compatible" and not _hosted_preflight_ready():
        st.error("Validate the hosted chat and embedding backends before rebuilding.")
        return False

    runtime = st.session_state.get("runtime", "Local deterministic")
    graph_store = InMemoryGraphStore()
    vector_store = InMemoryVectorStore()
    artifact_store = InMemoryArtifactStore()
    try:
        if runtime == "Hosted HuggingFace-compatible":
            chat_token = _optional_token("chat_api_token")
            embedding_token = chat_token if st.session_state.reuse_chat_token_for_embeddings else _optional_token(
                "embedding_api_token"
            )
            embedder = OpenAICompatibleEmbeddingModel(
                base_url=st.session_state.embedding_base_url,
                model=st.session_state.embedding_model,
                dimension=int(st.session_state.embedding_dimension),
                api_key=embedding_token,
                event_sink=_api_trace_sink,
            )
            llm = OpenAICompatibleLLM(
                base_url=st.session_state.chat_base_url,
                model=_hosted_generation_model_id(),
                api_key=chat_token,
                event_sink=_api_trace_sink,
            )
            embedder = _ObservedEmbeddingModel(
                embedder,
                component="hosted_embeddings",
                model_name=str(st.session_state.embedding_model),
                runtime=runtime,
            )
            llm = _ObservedLLM(
                llm,
                component="hosted_chat",
                model_name=_hosted_generation_model_id(),
                runtime=runtime,
            )
        elif runtime == "HuggingFace Hub local":
            hf_token = _optional_token("hf_hub_token")
            cache_folder = _optional_secret("hf_cache_folder")
            local_files_only = bool(st.session_state.hf_local_files_only)
            embedder = SentenceTransformerEmbeddingModel(
                st.session_state.embedding_model,
                dimension=int(st.session_state.embedding_dimension),
                device=_optional_secret("hf_embedding_device"),
                batch_size=int(st.session_state.hf_embedding_batch_size),
                trust_remote_code=bool(st.session_state.hf_trust_remote_code),
                token=hf_token,
                cache_folder=cache_folder,
                local_files_only=local_files_only,
            )
            llm = TransformersCausalLLM(
                st.session_state.generation_model,
                device_map=st.session_state.hf_device_map,
                torch_dtype=st.session_state.hf_torch_dtype,
                max_new_tokens=int(st.session_state.hf_max_new_tokens),
                trust_remote_code=bool(st.session_state.hf_trust_remote_code),
                token=hf_token,
                cache_dir=cache_folder,
                local_files_only=local_files_only,
                chat_template_kwargs={"enable_thinking": bool(st.session_state.hf_enable_thinking)},
            )
            embedder = _ObservedEmbeddingModel(
                embedder,
                component="local_hf_embeddings",
                model_name=str(st.session_state.embedding_model),
                runtime=runtime,
            )
            llm = _ObservedLLM(
                llm,
                component="local_hf_generation",
                model_name=str(st.session_state.generation_model),
                runtime=runtime,
            )
        else:
            embedder = HashingEmbeddingModel(dimension=int(st.session_state.hash_dimension))
            llm = HeuristicLLM()
            embedder = _ObservedEmbeddingModel(
                embedder,
                component="deterministic_embeddings",
                model_name=f"hashing-{st.session_state.hash_dimension}",
                runtime=runtime,
            )
            llm = _ObservedLLM(
                llm,
                component="deterministic_llm",
                model_name="heuristic",
                runtime=runtime,
            )

        extractor = _build_graph_extractor(llm, runtime=runtime)
        engine = GraphRAGEngine(
            config=_build_config(),
            graph_store=graph_store,
            vector_store=vector_store,
            artifact_store=artifact_store,
            embedder=embedder,
            llm=llm,
            extractor=extractor,
        )
    except Exception as exc:
        _event("Engine rebuild failed", {"runtime": runtime, "error": str(exc)})
        _render_runtime_exception(f"Failed to rebuild {runtime} engine", exc, runtime=runtime)
        return False

    st.session_state.graph_store = graph_store
    st.session_state.vector_store = vector_store
    st.session_state.artifact_store = artifact_store
    st.session_state.engine = engine
    st.session_state.last_retrieval = None
    st.session_state.last_answer = None
    st.session_state.engine_signature = _current_engine_signature()
    _event("Engine rebuilt", {"runtime": runtime, "graph_extraction_mode": st.session_state.graph_extraction_mode})
    return True


def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown("### Runtime")
        st.radio(
            "Model backend",
            ["Local deterministic", "HuggingFace Hub local", "Hosted HuggingFace-compatible"],
            key="runtime",
            on_change=_on_runtime_change,
            horizontal=False,
        )

        if st.session_state.runtime == "Hosted HuggingFace-compatible":
            st.text_input(
                "Chat base URL",
                key="chat_base_url",
                placeholder="https://router.huggingface.co/v1",
                help="OpenAI-compatible chat endpoint. For HF Router, use https://router.huggingface.co/v1.",
            )
            st.text_input(
                "Embedding base URL",
                key="embedding_base_url",
                placeholder="http://127.0.0.1:8080/v1",
                help="OpenAI-compatible embeddings endpoint, for example local vLLM/TEI at http://127.0.0.1:8080/v1.",
            )
            st.text_input(
                "Chat HF/API token",
                key="chat_api_token",
                type="password",
                placeholder="hf_...",
                help="Bearer token sent to the chat endpoint. For HuggingFace, use an HF token with access to the model/endpoint.",
            )
            st.toggle("Reuse chat token for embeddings", key="reuse_chat_token_for_embeddings")
            if not st.session_state.reuse_chat_token_for_embeddings:
                st.text_input(
                    "Embedding HF/API token",
                    key="embedding_api_token",
                    type="password",
                    placeholder="hf_...",
                    help="Bearer token sent to the embedding endpoint.",
                )
            st.text_input(
                "Generation repo id",
                key="generation_model",
                placeholder=DEFAULT_GENERATION_REPO_ID,
                help="HF model repo for chat generation. This workspace defaults to your EmpPolRAG Qwen repo.",
            )
            if _uses_hf_router(st.session_state.chat_base_url):
                if str(st.session_state.generation_model).strip() == DEFAULT_GENERATION_REPO_ID:
                    st.warning(
                        "HF Router can only serve models supported by a selected Inference Provider. "
                        "If validation reports `model_not_supported`, use `HuggingFace Hub local` for this repo "
                        "or deploy it behind your own OpenAI-compatible endpoint."
                    )
                st.text_input(
                    "HF router provider suffix",
                    key="hf_router_provider",
                    placeholder=DEFAULT_HF_ROUTER_PROVIDER,
                    help="For router.huggingface.co, this is appended to the model as repo:provider. Use hf-inference unless you know another provider serves the model.",
                )
                st.caption(f"Resolved hosted model id: `{_hosted_generation_model_id()}`")
            st.text_input(
                "Embedding repo id",
                key="embedding_model",
                placeholder=DEFAULT_EMBEDDING_REPO_ID,
                help="Embedding model repo served by the embedding endpoint. all-MiniLM-L6-v2 uses 384 dimensions; BGE-M3 uses 1024.",
            )
            st.number_input(
                "Embedding dimension",
                min_value=16,
                max_value=8192,
                step=16,
                key="embedding_dimension",
                help="Use 384 for all-MiniLM-L6-v2 or 1024 for BAAI/bge-m3.",
            )
            with st.expander("Hosted backend requirements", expanded=False):
                st.markdown(
                    """
                    - `Chat base URL` must point to a service that exposes `/chat/completions`.
                    - HF Router only serves models supported by a provider; a model repo alone is not enough.
                    - `Embedding base URL` must point to a running service that exposes `/embeddings`.
                    - For local BGE-M3 embeddings, start vLLM/TEI on port `8080` before validating.
                    """
                )
                st.code(
                    "vllm serve BAAI/bge-m3 --task embed --host 127.0.0.1 --port 8080",
                    language="powershell",
                )
            test_col_a, test_col_b = st.columns(2)
            with test_col_a:
                if st.button("Test Chat", use_container_width=True):
                    _test_hosted_chat()
            with test_col_b:
                if st.button("Test Embeddings", use_container_width=True):
                    _test_hosted_embeddings()
            if st.button("Validate Hosted Backend", use_container_width=True, type="primary"):
                _run_hosted_preflight(show_success=True)
            _render_hosted_preflight_status()
        elif st.session_state.runtime == "HuggingFace Hub local":
            st.markdown("#### Local Model Preset")
            notice = str(st.session_state.get("local_auto_repair_notice", "")).strip()
            if notice:
                st.warning(notice)
            preset_a, preset_b = st.columns(2)
            with preset_a:
                if st.button("Cached Qwen 1.5B + MiniLM", use_container_width=True, type="primary"):
                    _apply_local_model_preset(DEFAULT_GENERATION_REPO_ID, DEFAULT_EMBEDDING_REPO_ID, DEFAULT_EMBEDDING_DIMENSION)
                    _rerun()
            with preset_b:
                if st.button("Fast Qwen 0.5B + MiniLM", use_container_width=True):
                    _apply_local_model_preset(FAST_GENERATION_REPO_ID, DEFAULT_EMBEDDING_REPO_ID, DEFAULT_EMBEDDING_DIMENSION)
                    _rerun()
            if st.button("Quality Qwen3-8B + BGE-M3", use_container_width=True):
                _apply_local_model_preset(
                    QUALITY_GENERATION_REPO_ID,
                    QUALITY_EMBEDDING_REPO_ID,
                    QUALITY_EMBEDDING_DIMENSION,
                )
                _rerun()
            if st.button("Your EmpPolRAG repo + MiniLM", use_container_width=True):
                _apply_local_model_preset(CUSTOM_GENERATION_REPO_ID, DEFAULT_EMBEDDING_REPO_ID, DEFAULT_EMBEDDING_DIMENSION)
                _rerun()
            if st.button("Reset local defaults", use_container_width=True):
                _apply_local_model_preset(DEFAULT_GENERATION_REPO_ID, DEFAULT_EMBEDDING_REPO_ID, DEFAULT_EMBEDDING_DIMENSION)
                _rerun()
            st.text_input(
                "HF token",
                key="hf_hub_token",
                type="password",
                help="Token used by Transformers/SentenceTransformers to download private or gated repos from Hugging Face.",
            )
            st.text_input(
                "Generation repo id",
                key="generation_model",
                placeholder=DEFAULT_GENERATION_REPO_ID,
                help="Local Transformers causal LM repo. Default uses the cached Qwen/Qwen2.5-1.5B-Instruct model.",
            )
            st.text_input(
                "Embedding repo id",
                key="embedding_model",
                placeholder=DEFAULT_EMBEDDING_REPO_ID,
                help="Local SentenceTransformers-compatible embedding repo. Default uses cached all-MiniLM-L6-v2. BGE-M3 requires a complete cache including tokenizer files.",
            )
            st.number_input(
                "Embedding dimension",
                min_value=16,
                max_value=8192,
                step=16,
                key="embedding_dimension",
                help="all-MiniLM-L6-v2 uses 384 dimensions. BAAI/bge-m3 uses 1024 dimensions.",
            )
            st.text_input("HF cache folder", key="hf_cache_folder", help="Optional local cache directory for downloaded weights.")
            st.text_input(
                "LLM device map",
                key="hf_device_map",
                placeholder="blank = default CPU load",
                help="Leave blank for reliable local CPU loading. `auto` requires accelerate and may fail if accelerate is not installed in this Python environment.",
            )
            st.selectbox("LLM torch dtype", ["auto", "float16", "bfloat16", "float32"], key="hf_torch_dtype")
            st.number_input("Max new tokens", min_value=64, max_value=8192, step=64, key="hf_max_new_tokens")
            st.toggle(
                "Enable Qwen thinking mode",
                key="hf_enable_thinking",
                help="Keep off for GraphRAG ingestion and concise answers. Turn on only when you want reasoning traces.",
            )
            st.text_input("Embedding device", key="hf_embedding_device", help="Optional: cuda, cpu, mps, or leave blank.")
            st.number_input("Embedding batch size", min_value=1, max_value=256, step=1, key="hf_embedding_batch_size")
            st.toggle("Trust remote code", key="hf_trust_remote_code")
            st.toggle("Local files only", key="hf_local_files_only")
            if st.button("Check Local Setup", use_container_width=True):
                _run_local_preflight(show_success=True)
            _render_local_preflight_status()
            st.info(
                "Click `Rebuild` after choosing a preset. The default preset loads only complete cached models and should not call HuggingFace."
            )
        else:
            st.number_input("Hash dimension", min_value=32, max_value=4096, step=32, key="hash_dimension")

        st.markdown("### Retrieval")
        st.slider("Chunk size", 120, 2000, key="chunk_size", step=20)
        st.slider("Chunk overlap", 0, 400, key="chunk_overlap", step=10)
        if int(st.session_state.chunk_overlap) >= int(st.session_state.chunk_size):
            st.warning(
                f"Chunk overlap must be smaller than chunk size. "
                f"The active config will use {_effective_chunk_overlap()}."
            )
        st.slider("Context tokens", 1000, 32000, key="max_context_tokens", step=500)
        st.slider("Chunks", 2, 64, key="max_chunks")
        st.slider("Entities", 4, 128, key="max_entities")
        st.slider("Relationships", 4, 160, key="max_relationships")
        st.slider("Communities", 1, 32, key="max_communities")
        st.slider("PPR alpha", 0.10, 0.95, key="ppr_alpha", step=0.05)
        st.slider("Min relationship confidence", 0.0, 1.0, key="min_relationship_confidence", step=0.05)
        st.toggle("Symbolic queries", key="enable_symbolic_queries")
        st.toggle("Community summaries", key="enable_community_summaries")

        _render_graph_construction_controls()

        errors = _runtime_config_errors()
        for error in errors:
            st.error(error)

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Rebuild", use_container_width=True):
                if _rebuild_engine():
                    _rerun()
        with col_b:
            if st.button("Clear", use_container_width=True):
                _clear_workspace()
                _rerun()
        if not _engine_matches_sidebar():
            st.warning("Settings changed. Click Rebuild before ingesting or querying.")

def _render_graph_construction_controls() -> None:
    st.markdown("### Graph Construction")
    st.selectbox("Extractor", ["LLM JSON", "Rule-based"], key="graph_extraction_mode")
    if st.session_state.graph_extraction_mode == "LLM JSON":
        st.number_input(
            "Max entities per chunk",
            min_value=4,
            max_value=128,
            step=1,
            key="llm_extraction_max_entities",
        )
        st.toggle(
            "Fallback to rule-based on malformed JSON",
            key="llm_extraction_fallback",
            help="Keeps ingestion from producing an empty graph if the local model returns invalid JSON for a chunk.",
        )
        if st.session_state.runtime == "Local deterministic":
            st.warning("LLM JSON extraction needs a real model backend. Local deterministic will fall back if enabled.")
    else:
        st.caption("Rule-based extraction uses capitalized phrase matching and sequential RELATED_TO links.")


def _render_header() -> None:
    stats = _stats()
    active_runtime = (st.session_state.engine_signature or {}).get("runtime", "not built")
    st.markdown(
        f"""
        <div class="topbar">
            <div>
                <div class="eyebrow">Production GraphRAG Console</div>
                <h1>{APP_TITLE}</h1>
            </div>
            <div class="runtime-pill">Selected: {st.session_state.runtime}<br/>Active: {active_runtime}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Documents", stats["documents"])
    c2.metric("Chunks", stats["chunks"])
    c3.metric("Entities", stats["entities"])
    c4.metric("Relationships", stats["relationships"])
    c5.metric("Communities", stats["communities"])
    c6.metric("Vectors", stats["vectors"])


def _render_corpus_tab() -> None:
    left, right = st.columns([1.25, 1], gap="large")
    with left:
        st.subheader("Ingestion")
        doc_id = st.text_input("Document ID", value=_next_doc_id())
        metadata_raw = st.text_area("Metadata JSON", value='{"source": "manual"}', height=90)
        text = st.text_area("Document text", value="", height=260)
        uploaded = st.file_uploader("Files", type=SUPPORTED_UPLOAD_TYPES, accept_multiple_files=True)
        st.caption("Supported uploads: `.txt`, `.md`, `.pdf`. PDFs must contain an extractable text layer.")

        b1, b2, b3 = st.columns([1, 1, 1])
        with b1:
            if st.button("Ingest Text", use_container_width=True, type="primary"):
                _ingest_manual(doc_id, text, metadata_raw)
        with b2:
            if st.button("Ingest Files", use_container_width=True):
                _ingest_files(uploaded, metadata_raw)
        with b3:
            if st.button("Load Synthetic", use_container_width=True):
                _load_synthetic_corpus()

        _render_ingestion_status()

    with right:
        st.subheader("Corpus")
        rows = _document_rows()
        if rows:
            st.dataframe(rows, use_container_width=True, hide_index=True)
        else:
            st.info("No documents indexed.")
        st.subheader("Recent Runs")
        events = list(reversed(st.session_state.events[-8:]))
        if events:
            st.dataframe(events, use_container_width=True, hide_index=True)
        else:
            st.info("No runs recorded.")


def _render_ask_tab() -> None:
    st.subheader("Question")
    if not _engine_matches_sidebar():
        st.warning(
            "Model settings changed after the engine was built. Click `Rebuild` in the sidebar before querying."
        )
    engine_runtime = (st.session_state.engine_signature or {}).get("runtime", "unknown")
    st.caption(f"Active engine runtime: {engine_runtime}")
    preset_col, apply_col = st.columns([4, 1])
    with preset_col:
        preset = st.selectbox("Preset", list(QUERY_PRESETS))
    with apply_col:
        if st.button("Use Preset", use_container_width=True):
            st.session_state.query_text = preset
            _rerun()
    query = st.text_input("Query", key="query_text")
    col_a, col_b, col_c = st.columns([1, 1, 4])
    with col_a:
        run_answer = st.button("Answer", use_container_width=True, type="primary")
    with col_b:
        run_retrieve = st.button("Retrieve", use_container_width=True)

    if run_answer or run_retrieve:
        if not query.strip():
            st.warning("Query is empty.")
        elif not _engine_matches_sidebar():
            st.error("Rebuild required before running this query.")
        else:
            try:
                with st.spinner("Running GraphRAG pipeline"):
                    if run_answer:
                        started = time.perf_counter()
                        answer = st.session_state.engine.answer(query)
                        st.session_state.last_answer = answer
                        st.session_state.last_retrieval = answer.retrieval
                        _event("Answered query", {"query": query, "confidence": answer.confidence})
                        _trace(
                            category="pipeline",
                            operation="answer",
                            status="ok",
                            component="ask_tab",
                            duration_ms=_duration_ms(started),
                            detail={"query": query, "runtime": engine_runtime},
                            output=_answer_trace(answer),
                            summary=f"Answered query with confidence {answer.confidence:.3f}",
                        )
                    else:
                        started = time.perf_counter()
                        retrieval = st.session_state.engine.retrieve(query)
                        st.session_state.last_retrieval = retrieval
                        st.session_state.last_answer = None
                        _event("Retrieved query", {"query": query, "evidence": len(retrieval.evidence)})
                        _trace(
                            category="pipeline",
                            operation="retrieve",
                            status="ok",
                            component="ask_tab",
                            duration_ms=_duration_ms(started),
                            detail={"query": query, "runtime": engine_runtime},
                            output=_retrieval_trace(retrieval),
                            summary=f"Retrieved {len(retrieval.evidence)} evidence items",
                        )
            except Exception as exc:
                _event("Query failed", {"query": query, "runtime": engine_runtime, "error": str(exc)})
                _trace(
                    category="pipeline",
                    operation="answer" if run_answer else "retrieve",
                    status="error",
                    component="ask_tab",
                    detail={"query": query, "runtime": engine_runtime},
                    error=str(exc),
                    summary="Query failed",
                )
                _render_runtime_exception("GraphRAG query failed", exc, runtime=engine_runtime)

    answer = st.session_state.last_answer
    retrieval = st.session_state.last_retrieval
    if answer:
        _render_answer(answer)
    elif retrieval:
        _render_retrieval(retrieval)
    else:
        st.info("Load or ingest a corpus, then run a query.")


def _render_answer(answer: Answer) -> None:
    st.markdown('<div class="answer-panel">', unsafe_allow_html=True)
    st.markdown("#### Answer")
    st.write(answer.answer)
    st.progress(min(max(answer.confidence, 0.0), 1.0), text=f"Confidence {answer.confidence:.3f}")
    st.markdown("</div>", unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1], gap="large")
    with c1:
        st.subheader("Citations")
        st.dataframe(list(answer.citations), use_container_width=True, hide_index=True)
    with c2:
        st.subheader("Graph Evidence")
        st.dataframe(list(answer.graph_evidence), use_container_width=True, hide_index=True)
    _render_retrieval(answer.retrieval)


def _render_retrieval(retrieval: RetrievalBundle) -> None:
    plan = retrieval.query_plan
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Intent", plan.intent.value)
    c2.metric("Evidence", len(retrieval.evidence))
    c3.metric("Keywords", len(plan.keywords))
    c4.metric("Entities", len(plan.entities))

    with st.expander("Query Plan", expanded=False):
        st.json(
            {
                "query": plan.query,
                "intent": plan.intent.value,
                "keywords": list(plan.keywords),
                "entities": list(plan.entities),
                "subqueries": list(plan.subqueries),
                "symbolic_query": plan.symbolic_query,
            }
        )

    st.subheader("Top Evidence")
    rows = _evidence_rows(retrieval.evidence)
    if rows:
        st.dataframe(rows[:20], use_container_width=True, hide_index=True)
    else:
        st.info("No evidence returned.")

    with st.expander("Assembled Context", expanded=False):
        st.code(retrieval.context, language="text")


def _render_graph_tab() -> None:
    graph_store = st.session_state.graph_store
    st.subheader("Graph")
    col_a, col_b = st.columns([1.2, 1], gap="large")
    with col_a:
        edge_limit = st.slider("Rendered relationships", 1, 120, min(40, max(1, len(graph_store.relationships()))))
        dot = _graphviz_dot(graph_store.entities(), graph_store.relationships()[:edge_limit])
        if graph_store.entities():
            st.graphviz_chart(dot, use_container_width=True)
        else:
            st.info("No graph entities available.")
    with col_b:
        st.subheader("Entities")
        st.dataframe(_entity_rows(graph_store.entities()), use_container_width=True, hide_index=True)

    rel_col, community_col = st.columns([1.35, 1], gap="large")
    with rel_col:
        st.subheader("Relationships")
        st.dataframe(_relationship_rows(graph_store), use_container_width=True, hide_index=True)
    with community_col:
        st.subheader("Community Summaries")
        st.dataframe(_community_rows(graph_store), use_container_width=True, hide_index=True)


def _render_evidence_tab() -> None:
    st.subheader("Evidence Inspector")
    retrieval = st.session_state.last_retrieval
    if not retrieval:
        st.info("No retrieval run available.")
        return

    kinds = sorted({item.kind for item in retrieval.evidence})
    selected = st.multiselect("Kinds", kinds, default=kinds)
    filtered = [item for item in retrieval.evidence if item.kind in selected]
    st.dataframe(_evidence_rows(filtered), use_container_width=True, hide_index=True)

    selected_id = st.selectbox("Evidence item", [item.id for item in filtered] if filtered else [])
    item = next((candidate for candidate in filtered if candidate.id == selected_id), None)
    if item:
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("#### Text")
            st.write(item.text)
        with c2:
            st.markdown("#### Metadata")
            st.json({"score": item.score, "source_ids": item.source_ids, "metadata": item.metadata})


def _render_settings_tab() -> None:
    st.subheader("Runtime State")
    config = st.session_state.engine.config
    model_profile = {
        "default_profile": DEFAULT_HF_MODEL_PROFILE.name,
        "generation_repo_id": DEFAULT_HF_MODEL_PROFILE.generation_repo_id,
        "embedding_repo_id": DEFAULT_HF_MODEL_PROFILE.embedding_repo_id,
        "embedding_dimension": DEFAULT_HF_MODEL_PROFILE.embedding_dimension,
        "reranker_repo_id": DEFAULT_HF_MODEL_PROFILE.reranker_repo_id,
    }
    c1, c2 = st.columns([1, 1], gap="large")
    with c1:
        st.json(
            {
                "runtime": st.session_state.runtime,
                "config": config.__dict__,
                "model_profile": model_profile,
                "recommended_local_models": {
                    "generation_repo_id": DEFAULT_GENERATION_REPO_ID,
                    "fast_generation_repo_id": FAST_GENERATION_REPO_ID,
                    "balanced_generation_repo_id": BALANCED_GENERATION_REPO_ID,
                    "quality_generation_repo_id": QUALITY_GENERATION_REPO_ID,
                    "light_generation_repo_id": LIGHT_GENERATION_REPO_ID,
                    "custom_generation_repo_id": CUSTOM_GENERATION_REPO_ID,
                    "embedding_repo_id": DEFAULT_EMBEDDING_REPO_ID,
                    "embedding_dimension": DEFAULT_EMBEDDING_DIMENSION,
                    "quality_embedding_repo_id": QUALITY_EMBEDDING_REPO_ID,
                    "quality_embedding_dimension": QUALITY_EMBEDDING_DIMENSION,
                    "fallback_embedding_repo_id": FALLBACK_EMBEDDING_REPO_ID,
                    "fallback_embedding_dimension": FALLBACK_EMBEDDING_DIMENSION,
                },
                "configured_generation": {
                    "repo_id": st.session_state.generation_model,
                    "hosted_model_id": _hosted_generation_model_id()
                    if st.session_state.runtime == "Hosted HuggingFace-compatible"
                    else st.session_state.generation_model,
                    "default_repo_id": DEFAULT_GENERATION_REPO_ID,
                },
                "graph_extraction": {
                    "mode": st.session_state.graph_extraction_mode,
                    "llm_max_entities_per_chunk": int(st.session_state.llm_extraction_max_entities),
                    "fallback_to_rule_based": bool(st.session_state.llm_extraction_fallback),
                },
                "auth": {
                    "hf_hub_token_configured": bool(_optional_token("hf_hub_token")),
                    "chat_token_configured": bool(_optional_token("chat_api_token")),
                    "embedding_token_configured": bool(
                        _optional_token("chat_api_token")
                        if st.session_state.reuse_chat_token_for_embeddings
                        else _optional_token("embedding_api_token")
                    ),
                    "reuse_chat_token_for_embeddings": st.session_state.reuse_chat_token_for_embeddings,
                    "tokens_hidden": True,
                },
            }
        )
    with c2:
        snapshot = _export_snapshot()
        st.download_button(
            "Download Snapshot",
            data=json.dumps(snapshot, indent=2),
            file_name="graphrag_workspace_snapshot.json",
            mime="application/json",
            use_container_width=True,
        )
        st.subheader("Vector Namespaces")
        st.dataframe(_vector_rows(), use_container_width=True, hide_index=True)


def _render_logs_tab() -> None:
    st.subheader("Operational Logs")
    traces = list(st.session_state.get("trace_events", []))
    total = len(traces)
    errors = sum(1 for row in traces if row.get("status") == "error")
    api_calls = sum(1 for row in traces if row.get("category") == "api_call")
    model_calls = sum(1 for row in traces if row.get("category") == "model_call")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Trace Events", total)
    c2.metric("Errors", errors)
    c3.metric("API Calls", api_calls)
    c4.metric("Model Calls", model_calls)

    controls = st.columns([1.2, 1.2, 1.2, 2, 1])
    categories = sorted({str(row.get("category", "")) for row in traces if row.get("category")})
    statuses = sorted({str(row.get("status", "")) for row in traces if row.get("status")})
    with controls[0]:
        selected_categories = st.multiselect("Category", categories, default=categories)
    with controls[1]:
        selected_statuses = st.multiselect("Status", statuses, default=statuses)
    with controls[2]:
        max_rows = st.number_input("Rows", min_value=25, max_value=5000, value=250, step=25)
    with controls[3]:
        search = st.text_input("Search logs", placeholder="query, model, endpoint, document id, error")
    with controls[4]:
        if st.button("Clear Logs", use_container_width=True):
            st.session_state.trace_events = []
            _trace(
                category="app_event",
                operation="logs_cleared",
                status="ok",
                component="ui",
                detail={"cleared_by": "user"},
            )
            _rerun()

    filtered = _filter_trace_events(
        traces,
        categories=set(selected_categories),
        statuses=set(selected_statuses),
        search=search,
    )
    rows = [_trace_table_row(row) for row in reversed(filtered[-int(max_rows) :])]
    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.info("No log events match the current filters.")

    export_col_a, export_col_b = st.columns([1, 1])
    with export_col_a:
        st.download_button(
            "Download Filtered JSON",
            data=json.dumps(filtered, indent=2, ensure_ascii=True),
            file_name="graphrag_trace_events.json",
            mime="application/json",
            use_container_width=True,
        )
    with export_col_b:
        st.download_button(
            "Download Filtered JSONL",
            data="\n".join(json.dumps(row, ensure_ascii=True) for row in filtered),
            file_name="graphrag_trace_events.jsonl",
            mime="application/x-ndjson",
            use_container_width=True,
        )

    if filtered:
        st.subheader("Event Detail")
        event_ids = [row["id"] for row in reversed(filtered[-int(max_rows) :])]
        selected_id = st.selectbox("Trace event", event_ids)
        selected = next((row for row in filtered if row.get("id") == selected_id), None)
        if selected:
            st.json(selected)


def _filter_trace_events(
    traces: list[dict[str, Any]],
    *,
    categories: set[str],
    statuses: set[str],
    search: str,
) -> list[dict[str, Any]]:
    needle = search.strip().lower()
    filtered = []
    for row in traces:
        if categories and str(row.get("category", "")) not in categories:
            continue
        if statuses and str(row.get("status", "")) not in statuses:
            continue
        if needle and needle not in json.dumps(row, ensure_ascii=True, sort_keys=True).lower():
            continue
        filtered.append(row)
    return filtered


def _trace_table_row(row: dict[str, Any]) -> dict[str, Any]:
    detail = row.get("detail") if isinstance(row.get("detail"), dict) else {}
    return {
        "time": row.get("time_local"),
        "category": row.get("category"),
        "status": row.get("status"),
        "operation": row.get("operation"),
        "component": row.get("component"),
        "duration_ms": row.get("duration_ms"),
        "summary": row.get("summary") or detail.get("summary", ""),
        "id": row.get("id"),
    }


def _ingest_manual(document_id: str, text: str, metadata_raw: str) -> None:
    if not _engine_matches_sidebar():
        st.error("Rebuild required before ingestion.")
        return
    if not text.strip():
        st.warning("Document text is empty.")
        return
    metadata = _parse_metadata(metadata_raw)
    doc_id = document_id.strip() or _next_doc_id()
    progress_area = st.empty()
    _prepare_ingestion_run([(doc_id, len(text))])
    _update_ingestion_status(doc_id, "Queued", 0.05, "Waiting for ingestion")
    _render_ingestion_progress(progress_area)
    if _ingest_one_document(doc_id, text, metadata, progress_area):
        st.success("Document ingested.")


def _ingest_files(uploaded: Iterable[Any] | None, metadata_raw: str) -> None:
    if not _engine_matches_sidebar():
        st.error("Rebuild required before ingestion.")
        return
    files = list(uploaded or [])
    if not files:
        st.warning("No files selected.")
        return
    metadata = _parse_metadata(metadata_raw)
    progress_area = st.empty()
    pending: list[tuple[str, str, str, dict[str, Any]]] = []
    for file in files:
        filename = getattr(file, "name", "unknown")
        try:
            text, file_info = _read_uploaded_file(file)
        except Exception as exc:
            st.error(f"Could not read uploaded file `{filename}`: {exc}")
            _event("File decode failed", {"filename": filename, "error": str(exc)})
            _trace(
                category="file_io",
                operation="decode_upload",
                status="error",
                component="corpus_tab",
                detail={"filename": filename},
                error=str(exc),
                summary=f"Failed to decode {filename}",
            )
            continue
        doc_id = os.path.splitext(filename)[0]
        pending.append((doc_id, filename, text, file_info))
        _trace(
            category="file_io",
            operation="decode_upload",
            status="ok",
            component="corpus_tab",
            detail={"filename": filename, **file_info},
            output={"characters": len(text), "preview": text[:1000]},
            summary=f"Decoded {filename}",
        )
    if not pending:
        st.warning("No readable files selected.")
        return
    _prepare_ingestion_run([(doc_id, len(text)) for doc_id, _, text, _ in pending])
    _render_ingestion_progress(progress_area)
    for doc_id, filename, text, file_info in pending:
        _update_ingestion_status(doc_id, "Decoded", 0.18, f"Read {filename}")
        _render_ingestion_progress(progress_area)
        file_metadata = dict(metadata)
        file_metadata.update({"filename": filename, **file_info})
        _ingest_one_document(doc_id, text, file_metadata, progress_area)
    _render_batch_result("file", len(files))


def _read_uploaded_file(file: Any) -> tuple[str, dict[str, Any]]:
    filename = getattr(file, "name", "uploaded")
    suffix = os.path.splitext(filename)[1].lower()
    data = file.getvalue()
    if suffix == ".pdf":
        return _extract_pdf_text(data, filename)
    if suffix in {".txt", ".md", ""}:
        text = data.decode("utf-8", errors="replace")
        if not text.strip():
            raise ValueError("Text file is empty.")
        return text, {
            "filename": filename,
            "content_type": getattr(file, "type", "") or "text/plain",
            "parser": "utf-8",
            "source_format": suffix.lstrip(".") or "text",
        }
    raise ValueError(f"Unsupported file type `{suffix}`. Supported types: {', '.join(SUPPORTED_UPLOAD_TYPES)}.")


def _extract_pdf_text(data: bytes, filename: str) -> tuple[str, dict[str, Any]]:
    try:
        from pypdf import PdfReader
    except Exception as exc:
        raise RuntimeError("Install `pypdf` to ingest PDF files: `python -m pip install pypdf`.") from exc

    try:
        reader = PdfReader(io.BytesIO(data))
    except Exception as exc:
        raise ValueError(f"Could not parse PDF: {exc}") from exc

    if getattr(reader, "is_encrypted", False):
        try:
            decrypt_result = reader.decrypt("")
        except Exception as exc:
            raise ValueError("PDF is encrypted and could not be opened with an empty password.") from exc
        if decrypt_result == 0:
            raise ValueError("PDF is encrypted. Remove the password before uploading.")

    pages = []
    empty_pages = 0
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            page_text = page.extract_text() or ""
        except Exception as exc:
            raise ValueError(f"Could not extract text from page {page_number}: {exc}") from exc
        page_text = page_text.strip()
        if page_text:
            pages.append(f"[Page {page_number}]\n{page_text}")
        else:
            empty_pages += 1

    text = "\n\n".join(pages).strip()
    if not text:
        raise ValueError(
            "No extractable text found in this PDF. It may be scanned/image-only; run OCR first, then upload the OCR text or searchable PDF."
        )

    return text, {
        "filename": filename,
        "content_type": "application/pdf",
        "parser": "pypdf",
        "source_format": "pdf",
        "pdf_pages": len(reader.pages),
        "pdf_pages_with_text": len(pages),
        "pdf_empty_pages": empty_pages,
    }


def _load_synthetic_corpus() -> None:
    if not _engine_matches_sidebar():
        st.error("Rebuild required before loading synthetic data.")
        return
    progress_area = st.empty()
    _prepare_ingestion_run([(document_id, len(text)) for document_id, text in SYNTHETIC_CORPUS.items()])
    _render_ingestion_progress(progress_area)
    for document_id, text in SYNTHETIC_CORPUS.items():
        _ingest_one_document(document_id, text, {"source": "synthetic"}, progress_area)
    _render_batch_result("synthetic document", len(SYNTHETIC_CORPUS))


def _prepare_ingestion_run(items: list[tuple[str, int]]) -> None:
    st.session_state.ingestion_runs = [
        {
            "document_id": document_id,
            "status": "Queued",
            "progress": 0.0,
            "characters": character_count,
            "chunks": 0,
            "entities": 0,
            "relationships": 0,
            "detail": "Waiting",
        }
        for document_id, character_count in items
    ]


def _ingest_one_document(
    document_id: str,
    text: str,
    metadata: dict[str, Any],
    progress_area: Any,
) -> bool:
    before = _stats()
    started = time.perf_counter()
    try:
        _update_ingestion_status(document_id, "Ingesting", 0.35, "Chunking, extracting graph facts, and embedding")
        _render_ingestion_progress(progress_area)
        st.session_state.engine.ingest_text(text, document_id=document_id, metadata=metadata)
        after = _stats()
        _update_ingestion_status(
            document_id,
            "Indexed",
            0.82,
            "Graph, communities, and vectors written",
            chunks=max(0, after["chunks"] - before["chunks"]),
            entities=max(0, after["entities"] - before["entities"]),
            relationships=max(0, after["relationships"] - before["relationships"]),
        )
        _render_ingestion_progress(progress_area)
        _update_ingestion_status(document_id, "Done", 1.0, "Ready for retrieval")
        _render_ingestion_progress(progress_area)
        _event("Ingested document", {"document_id": document_id, "characters": len(text)})
        _trace(
            category="pipeline",
            operation="ingest_document",
            status="ok",
            component="corpus_tab",
            duration_ms=_duration_ms(started),
            detail={
                "document_id": document_id,
                "metadata": metadata,
                "characters": len(text),
                "graph_extraction_mode": st.session_state.graph_extraction_mode,
            },
            output={"before": before, "after": after, "delta": {key: after[key] - before[key] for key in before}},
            summary=f"Ingested {document_id}",
        )
        return True
    except Exception as exc:
        _update_ingestion_status(document_id, "Failed", 1.0, str(exc))
        _render_ingestion_progress(progress_area)
        _event("Ingestion failed", {"document_id": document_id, "error": str(exc)})
        _trace(
            category="pipeline",
            operation="ingest_document",
            status="error",
            component="corpus_tab",
            duration_ms=_duration_ms(started),
            detail={
                "document_id": document_id,
                "metadata": metadata,
                "characters": len(text),
                "graph_extraction_mode": st.session_state.graph_extraction_mode,
            },
            error=str(exc),
            summary=f"Ingestion failed for {document_id}",
        )
        st.error(f"Failed to ingest {document_id}: {exc}")
        return False


def _update_ingestion_status(
    document_id: str,
    status: str,
    progress: float,
    detail: str,
    *,
    chunks: int | None = None,
    entities: int | None = None,
    relationships: int | None = None,
) -> None:
    rows = st.session_state.setdefault("ingestion_runs", [])
    for row in rows:
        if row["document_id"] != document_id:
            continue
        row["status"] = status
        row["progress"] = round(min(max(progress, 0.0), 1.0), 3)
        row["detail"] = detail
        if chunks is not None:
            row["chunks"] = chunks
        if entities is not None:
            row["entities"] = entities
        if relationships is not None:
            row["relationships"] = relationships
        return


def _render_ingestion_status() -> None:
    rows = st.session_state.get("ingestion_runs", [])
    if not rows:
        return
    st.subheader("Ingestion Status")
    _render_ingestion_progress(st.container())


def _render_ingestion_progress(container: Any) -> None:
    rows = st.session_state.get("ingestion_runs", [])
    if not rows:
        return
    completed = sum(1 for row in rows if row["status"] == "Done")
    failed = sum(1 for row in rows if row["status"] == "Failed")
    total = len(rows)
    aggregate = sum(float(row["progress"]) for row in rows) / total
    label = f"{completed}/{total} documents complete"
    if failed:
        label += f" - {failed} failed"
    with container.container():
        st.progress(aggregate, text=label)
        st.dataframe(_ingestion_rows(), use_container_width=True, hide_index=True)


def _ingestion_rows() -> list[dict[str, Any]]:
    rows = []
    for row in st.session_state.get("ingestion_runs", []):
        rows.append(
            {
                "document_id": row["document_id"],
                "status": row["status"],
                "progress": f"{int(float(row['progress']) * 100)}%",
                "characters": row["characters"],
                "chunks": row["chunks"],
                "entities": row["entities"],
                "relationships": row["relationships"],
                "detail": row["detail"],
            }
        )
    return rows


def _render_batch_result(label: str, total: int) -> None:
    failed = sum(1 for row in st.session_state.get("ingestion_runs", []) if row["status"] == "Failed")
    completed = total - failed
    if failed:
        st.warning(f"Ingested {completed}/{total} {label}s. {failed} failed.")
    else:
        st.success(f"Ingested {total} {label}s.")


class _ObservedLLM:
    def __init__(self, inner: Any, *, component: str, model_name: str, runtime: str) -> None:
        self.inner = inner
        self.component = component
        self.model_name = model_name
        self.runtime = runtime

    def complete(self, messages: list[ChatMessage] | tuple[ChatMessage, ...], *, temperature: float = 0.0) -> str:
        started = time.perf_counter()
        detail = {
            "runtime": self.runtime,
            "model": self.model_name,
            "temperature": temperature,
            "message_count": len(messages),
            "messages": [{"role": message.role, "content": message.content} for message in messages],
        }
        try:
            result = self.inner.complete(messages, temperature=temperature)
            _trace(
                category="model_call",
                operation="llm.complete",
                status="ok",
                component=self.component,
                duration_ms=_duration_ms(started),
                detail=detail,
                output={
                    "text": result,
                    "characters": len(result),
                },
                summary=f"{self.model_name} returned {len(result)} chars",
            )
            return result
        except Exception as exc:
            _trace(
                category="model_call",
                operation="llm.complete",
                status="error",
                component=self.component,
                duration_ms=_duration_ms(started),
                detail=detail,
                error=str(exc),
                summary=f"{self.model_name} failed",
            )
            raise


class _ObservedEmbeddingModel:
    def __init__(self, inner: Any, *, component: str, model_name: str, runtime: str) -> None:
        self.inner = inner
        self.component = component
        self.model_name = model_name
        self.runtime = runtime
        self.dimension = int(getattr(inner, "dimension", 0) or 0)

    def embed(self, texts: list[str] | tuple[str, ...]) -> list[list[float]]:
        started = time.perf_counter()
        text_list = list(texts)
        detail = {
            "runtime": self.runtime,
            "model": self.model_name,
            "input_count": len(text_list),
            "input_characters": [len(text) for text in text_list],
            "texts": text_list,
        }
        try:
            vectors = self.inner.embed(text_list)
            vector_lengths = [len(vector) for vector in vectors]
            _trace(
                category="model_call",
                operation="embedding.embed",
                status="ok",
                component=self.component,
                duration_ms=_duration_ms(started),
                detail=detail,
                output={
                    "vector_count": len(vectors),
                    "vector_dimensions": vector_lengths,
                    "first_vector_sample": vectors[0][:TRACE_VECTOR_SAMPLE] if vectors else [],
                },
                summary=f"{self.model_name} embedded {len(text_list)} texts",
            )
            return vectors
        except Exception as exc:
            _trace(
                category="model_call",
                operation="embedding.embed",
                status="error",
                component=self.component,
                duration_ms=_duration_ms(started),
                detail=detail,
                error=str(exc),
                summary=f"{self.model_name} embedding failed",
            )
            raise


def _apply_local_model_preset(generation_repo_id: str, embedding_repo_id: str, embedding_dimension: int) -> None:
    st.session_state.runtime = "HuggingFace Hub local"
    st.session_state.generation_model = generation_repo_id
    st.session_state.embedding_model = embedding_repo_id
    st.session_state.embedding_dimension = embedding_dimension
    st.session_state.hf_device_map = DEFAULT_HF_DEVICE_MAP
    st.session_state.hf_torch_dtype = "auto"
    st.session_state.hf_max_new_tokens = DEFAULT_HF_MAX_NEW_TOKENS
    st.session_state.hf_enable_thinking = False
    st.session_state.hf_embedding_batch_size = 16
    st.session_state.hf_embedding_device = ""
    st.session_state.hf_trust_remote_code = False
    st.session_state.hf_local_files_only = _hf_cache_ready(generation_repo_id, role="generation") and _hf_cache_ready(
        embedding_repo_id,
        role="embedding",
    )
    st.session_state.hosted_preflight_ok = False
    st.session_state.hosted_preflight_signature = None
    _event(
        "Applied local model preset",
        {
            "generation_model": generation_repo_id,
            "embedding_model": embedding_repo_id,
            "embedding_dimension": embedding_dimension,
        },
    )


def _on_runtime_change() -> None:
    runtime = st.session_state.get("runtime", "Local deterministic")
    previous = st.session_state.get("last_runtime")
    if runtime == "HuggingFace Hub local" and previous != runtime:
        _apply_local_model_preset(DEFAULT_GENERATION_REPO_ID, DEFAULT_EMBEDDING_REPO_ID, DEFAULT_EMBEDDING_DIMENSION)
    elif runtime == "Hosted HuggingFace-compatible" and previous != runtime:
        st.session_state.chat_base_url = "https://router.huggingface.co/v1"
        st.session_state.embedding_base_url = "http://localhost:8080/v1"
        st.session_state.hf_router_provider = DEFAULT_HF_ROUTER_PROVIDER
        st.session_state.generation_model = DEFAULT_GENERATION_REPO_ID
        st.session_state.embedding_model = DEFAULT_EMBEDDING_REPO_ID
        st.session_state.embedding_dimension = DEFAULT_EMBEDDING_DIMENSION
        st.session_state.hosted_preflight_ok = False
        st.session_state.hosted_preflight_signature = None
        st.session_state.hosted_preflight_report = {}
    st.session_state.last_runtime = runtime


def _clear_workspace() -> None:
    st.session_state.events = []
    st.session_state.ingestion_runs = []
    _rebuild_engine()


def _test_hosted_chat() -> None:
    errors = _runtime_config_errors()
    if errors:
        for error in errors:
            st.error(error)
        return
    ok, response = _check_hosted_chat()
    if ok:
        st.success("Hosted chat endpoint responded.")
        st.code(response, language="text")
    else:
        st.error(f"Hosted chat test failed: {response}")


def _test_hosted_embeddings() -> None:
    errors = _runtime_config_errors()
    if errors:
        for error in errors:
            st.error(error)
        return
    token = _optional_token("chat_api_token") if st.session_state.reuse_chat_token_for_embeddings else _optional_token(
        "embedding_api_token"
    )
    ok, message = _check_hosted_embeddings(token)
    if ok:
        vector_length = int(message)
        st.success(f"Hosted embedding endpoint responded with {vector_length} dimensions.")
    else:
        st.error(f"Hosted embedding test failed: {message}")


def _run_hosted_preflight(*, show_success: bool = False) -> bool:
    errors = _runtime_config_errors()
    if errors:
        st.session_state.hosted_preflight_ok = False
        st.session_state.hosted_preflight_signature = None
        st.session_state.hosted_preflight_report = {"errors": errors}
        for error in errors:
            st.error(error)
        _trace(
            category="preflight",
            operation="hosted_backend_validation",
            status="error",
            component="hosted_backend",
            detail=st.session_state.hosted_preflight_report,
            summary="Hosted backend validation failed before API checks",
        )
        return False

    chat_ok, chat_message = _check_hosted_chat()
    embedding_token = _hosted_embedding_token()
    embedding_ok, embedding_message = _check_hosted_embeddings(embedding_token)
    ok = chat_ok and embedding_ok
    st.session_state.hosted_preflight_ok = ok
    st.session_state.hosted_preflight_signature = _hosted_preflight_signature() if ok else None
    st.session_state.hosted_preflight_report = {
        "chat": {"ok": chat_ok, "message": chat_message},
        "embeddings": {"ok": embedding_ok, "message": embedding_message},
    }
    _trace(
        category="preflight",
        operation="hosted_backend_validation",
        status="ok" if ok else "error",
        component="hosted_backend",
        detail=st.session_state.hosted_preflight_report,
        summary="Hosted backend validated" if ok else "Hosted backend validation failed",
    )
    if ok and show_success:
        st.success("Hosted backend validated.")
    elif not ok:
        if not chat_ok:
            st.error(f"Hosted chat validation failed: {chat_message}")
            for hint in _hosted_failure_hints(chat_message, surface="chat"):
                st.info(hint)
        if not embedding_ok:
            st.error(f"Hosted embedding validation failed: {embedding_message}")
            for hint in _hosted_failure_hints(embedding_message, surface="embeddings"):
                st.info(hint)
    return ok


def _run_local_preflight(*, show_success: bool = False) -> bool:
    errors = _runtime_config_errors()
    if errors:
        st.session_state.local_preflight_report = {"config": {"status": "error", "messages": errors}}
        for error in errors:
            st.error(error)
        _trace(
            category="preflight",
            operation="local_setup_check",
            status="error",
            component="local_hf_backend",
            detail=st.session_state.local_preflight_report,
            summary="Local setup check failed before dependency checks",
        )
        return False

    report = {
        "dependencies": _check_local_hf_dependencies(),
        "generation_repo": _check_local_hf_repo(
            str(st.session_state.generation_model).strip(),
            role="generation",
        ),
        "embedding_repo": _check_local_hf_repo(
            str(st.session_state.embedding_model).strip(),
            role="embedding",
        ),
        "settings": {
            "local_files_only": bool(st.session_state.hf_local_files_only),
            "cache_folder": _optional_secret("hf_cache_folder") or "default HuggingFace cache",
            "embedding_dimension": int(st.session_state.embedding_dimension),
        },
    }
    st.session_state.local_preflight_report = report
    ok = all(
        section.get("status") != "error"
        for section in report.values()
        if isinstance(section, dict) and "status" in section
    )
    if ok and show_success:
        st.success("Local setup check passed or produced only non-blocking warnings.")
    elif not ok:
        st.error("Local setup check found blocking issues.")
    _trace(
        category="preflight",
        operation="local_setup_check",
        status="ok" if ok else "error",
        component="local_hf_backend",
        detail=report,
        summary="Local setup check passed" if ok else "Local setup check failed",
    )
    return ok


def _check_local_hf_dependencies() -> dict[str, Any]:
    try:
        from advanced_graphrag.models.huggingface import _ensure_workspace_vendor

        _ensure_workspace_vendor()
        import accelerate
        import huggingface_hub
        import sentence_transformers
        import torch
        import transformers
    except Exception as exc:
        return {
            "status": "error",
            "message": f"Missing or incompatible local HuggingFace dependency: {exc}",
        }

    return {
        "status": "ok",
        "torch": getattr(torch, "__version__", "unknown"),
        "transformers": getattr(transformers, "__version__", "unknown"),
        "sentence_transformers": getattr(sentence_transformers, "__version__", "unknown"),
        "accelerate": getattr(accelerate, "__version__", "unknown"),
        "huggingface_hub": getattr(huggingface_hub, "__version__", "unknown"),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
    }


def _check_local_hf_repo(repo_id_or_path: str, *, role: str) -> dict[str, Any]:
    if not repo_id_or_path:
        return {"status": "error", "message": f"{role.title()} repo id is empty."}

    if os.path.isdir(repo_id_or_path):
        return _check_local_model_directory(repo_id_or_path, role=role)

    local_files_only = bool(st.session_state.hf_local_files_only)
    cached = _cached_hf_files(repo_id_or_path)
    if local_files_only:
        if _hf_cache_ready(repo_id_or_path, role=role):
            return {
                "status": "ok",
                "message": f"Found a complete cached {role} snapshot for `{repo_id_or_path}`.",
                "cached_probe_files": cached,
            }
        if cached:
            return {
                "status": "error",
                "message": (
                    f"`{repo_id_or_path}` is only partially cached for {role} use: {', '.join(cached)}. "
                    "Use a complete cached model or disable Local files only to download missing files."
                ),
                "cached_probe_files": cached,
            }
        return {
            "status": "error",
            "message": (
                f"`{repo_id_or_path}` is not visible in the configured HF cache, but Local files only is enabled. "
                "Disable Local files only for the first download or set HF cache folder to a complete snapshot."
            ),
        }

    try:
        from advanced_graphrag.models.huggingface import _ensure_workspace_vendor

        _ensure_workspace_vendor()
        from huggingface_hub import model_info

        info = model_info(repo_id_or_path, token=_optional_token("hf_hub_token"))
        siblings = [sibling.rfilename for sibling in getattr(info, "siblings", [])]
        has_weights = any(_looks_like_weight_file(name) for name in siblings)
        has_config = any(name.endswith("config.json") for name in siblings)
        if not has_weights:
            return {
                "status": "warning",
                "message": (
                    f"`{repo_id_or_path}` is reachable, but the metadata check did not see standard model weights. "
                    "Rebuild may still work for custom repos, but verify the repo contains compatible weights."
                ),
            }
        return {
            "status": "ok",
            "message": f"`{repo_id_or_path}` is reachable on HuggingFace.",
            "files": {"has_config": has_config, "has_weights": has_weights, "count": len(siblings)},
            "cached_probe_files": cached,
        }
    except Exception as exc:
        status = "warning" if _hf_cache_ready(repo_id_or_path, role=role) else "error"
        return {
            "status": status,
            "message": (
                f"Could not verify `{repo_id_or_path}` through HuggingFace Hub metadata: {exc}. "
                "If this machine is offline or blocked by firewall, Rebuild works only when the full model snapshot is already cached."
            ),
            "cached_probe_files": cached,
        }


def _check_local_model_directory(path: str, *, role: str) -> dict[str, Any]:
    names = set(os.listdir(path))
    has_config = "config.json" in names or "sentence_bert_config.json" in names
    has_weights = any(_looks_like_weight_file(name) for name in names)
    if role == "embedding" and "modules.json" in names:
        has_config = True
    if not has_config:
        return {"status": "error", "message": f"Local model path `{path}` is missing a config file."}
    if not has_weights and role == "generation":
        return {"status": "error", "message": f"Local generation path `{path}` is missing model weight files."}
    if not has_weights and role == "embedding":
        return {
            "status": "warning",
            "message": f"Local embedding path `{path}` has config files but no obvious top-level weight file.",
        }
    return {"status": "ok", "message": f"Local {role} model path `{path}` looks usable."}


def _cached_hf_files(repo_id: str) -> list[str]:
    probe_files = [
        "config.json",
        "config_sentence_transformers.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "vocab.txt",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
        "sentence_bert_config.json",
        "modules.json",
        "model.safetensors",
        "pytorch_model.bin",
    ]
    try:
        from advanced_graphrag.models.huggingface import _ensure_workspace_vendor

        _ensure_workspace_vendor()
        from huggingface_hub import try_to_load_from_cache
    except Exception:
        return []

    cache_dir = _optional_secret("hf_cache_folder")
    found: list[str] = []
    for filename in probe_files:
        try:
            cached = try_to_load_from_cache(repo_id, filename, cache_dir=cache_dir)
        except Exception:
            continue
        if isinstance(cached, str) and os.path.exists(cached):
            found.append(filename)
    return found


def _hf_cache_ready(repo_id: str, *, role: str) -> bool:
    files = set(_cached_hf_files(repo_id))
    has_config = bool(files & {"config.json", "sentence_bert_config.json", "modules.json"})
    has_weights = bool(files & {"model.safetensors", "pytorch_model.bin"})
    has_tokenizer = bool(files & {"tokenizer.json", "tokenizer_config.json", "vocab.txt", "vocab.json"})
    if role == "generation":
        return has_config and has_weights and has_tokenizer
    if role == "embedding":
        return has_config and has_weights and has_tokenizer and "modules.json" in files
    return has_config and has_weights


def _looks_like_weight_file(name: str) -> bool:
    lower = name.lower()
    return lower.endswith((".safetensors", ".bin", ".ckpt", ".h5", ".msgpack"))


def _check_hosted_chat() -> tuple[bool, str]:
    try:
        llm = OpenAICompatibleLLM(
            base_url=st.session_state.chat_base_url,
            model=_hosted_generation_model_id(),
            api_key=_optional_token("chat_api_token"),
            timeout=60.0,
            event_sink=_api_trace_sink,
        )
        response = llm.complete(
            [ChatMessage("user", "Reply with exactly: GraphRAG hosted chat OK")],
            temperature=0.0,
        )
        return True, response
    except Exception as exc:
        return False, str(exc)


def _check_hosted_embeddings(token: str | None) -> tuple[bool, str]:
    try:
        embedder = OpenAICompatibleEmbeddingModel(
            base_url=st.session_state.embedding_base_url,
            model=st.session_state.embedding_model,
            dimension=int(st.session_state.embedding_dimension),
            api_key=token,
            timeout=60.0,
            event_sink=_api_trace_sink,
        )
        vector = embedder.embed(["GraphRAG embedding smoke test"])[0]
        return True, str(len(vector))
    except Exception as exc:
        return False, str(exc)


def _render_hosted_preflight_status() -> None:
    report = st.session_state.get("hosted_preflight_report") or {}
    if _hosted_preflight_ready():
        st.success("Hosted backend preflight passed.")
    elif report:
        st.warning("Hosted backend is not validated for the current settings.")
        with st.expander("Last hosted validation report", expanded=False):
            st.json(report)
            for hint in _hosted_report_hints(report):
                st.info(hint)
    else:
        st.info("Validate the hosted backend before rebuilding, ingesting, or querying.")


def _render_local_preflight_status() -> None:
    report = st.session_state.get("local_preflight_report") or {}
    if not report:
        st.caption("Use `Check Local Setup` before the first local model rebuild.")
        return
    with st.expander("Last local setup check", expanded=False):
        st.json(report)
        for hint in _local_report_hints(report):
            st.info(hint)


def _local_report_hints(report: dict[str, Any]) -> list[str]:
    hints: list[str] = []
    dependencies = report.get("dependencies", {}) if isinstance(report, dict) else {}
    if dependencies.get("status") == "error":
        hints.append("Install the project HF extra: `python -m pip install -e .[hf,ui]`.")
    for key in ("generation_repo", "embedding_repo"):
        section = report.get(key, {}) if isinstance(report, dict) else {}
        if section.get("status") == "error":
            hints.extend(_backend_failure_hints(str(section.get("message", "")), runtime="HuggingFace Hub local"))
        elif section.get("status") == "warning":
            hints.append(
                "A warning from the metadata check is not always fatal. It means the online check could not prove the repo is loadable from this machine."
            )
    return list(dict.fromkeys(hints))


def _hosted_report_hints(report: dict[str, Any]) -> list[str]:
    hints: list[str] = []
    chat = report.get("chat", {}) if isinstance(report, dict) else {}
    embeddings = report.get("embeddings", {}) if isinstance(report, dict) else {}
    if chat and not chat.get("ok"):
        hints.extend(_hosted_failure_hints(str(chat.get("message", "")), surface="chat"))
    if embeddings and not embeddings.get("ok"):
        hints.extend(_hosted_failure_hints(str(embeddings.get("message", "")), surface="embeddings"))
    return list(dict.fromkeys(hints))


def _hosted_failure_hints(message: str, *, surface: str) -> list[str]:
    lowered = message.lower()
    hints: list[str] = []
    if "model_not_supported" in lowered or "not supported by provider" in lowered:
        hints.append(
            "Your generation repo is not served by the selected HF Router provider. "
            "Use `HuggingFace Hub local`, deploy a dedicated HF Inference Endpoint, or run your own vLLM/TGI/Space endpoint for this repo."
        )
        hints.append(
            "For HF Router, replace the generation repo with a model listed in the HF Inference Providers playground for the selected provider."
        )
    if "invalid username or password" in lowered or "http 401" in lowered:
        hints.append(
            "The HF token is invalid for this endpoint. Paste a fresh raw `hf_...` token with Inference Providers permission."
        )
    if "actively refused" in lowered or "connection refused" in lowered or "winerror 10061" in lowered:
        if surface == "embeddings":
            hints.append(
                "No embedding server is running at the configured URL. Start vLLM/TEI for embeddings or switch to `HuggingFace Hub local`."
            )
            hints.append(
                "For BGE-M3 with vLLM, run: `vllm serve BAAI/bge-m3 --task embed --host 127.0.0.1 --port 8080`."
            )
        else:
            hints.append("No chat server is running at the configured chat URL.")
    if "bad request" in lowered and not hints:
        hints.append(
            "The provider rejected the request. Check that the endpoint is OpenAI-compatible and that the model id is valid for that provider."
        )
    return hints


def _render_runtime_exception(label: str, exc: BaseException, *, runtime: str) -> None:
    message = str(exc) or exc.__class__.__name__
    st.error(f"{label}: {message}")
    for hint in _backend_failure_hints(message, runtime=runtime):
        st.info(hint)


def _backend_failure_hints(message: str, *, runtime: str) -> list[str]:
    lowered = message.lower()
    hints: list[str] = []
    if runtime == "HuggingFace Hub local" or "huggingface" in lowered:
        if "baai/bge-m3" in lowered:
            hints.append(
                "BAAI/bge-m3 is a valid high-quality embedding model, but it must be fully downloaded or available in the HF cache before local loading can work."
            )
            hints.append(
                "For the reliable cached path on this machine, choose `Cached Qwen 1.5B + MiniLM`, keep embedding dimension at 384, then click Rebuild."
            )
        if "sentence-transformers/all-minilm-l6-v2" in lowered:
            hints.append("MiniLM uses embedding dimension 384. Set the embedding dimension to 384 before rebuilding.")
        if "can't load the model" in lowered or "could not load" in lowered or "missing model weight" in lowered:
            hints.append(
                "Check that the repo id is exact, the model has weight files, and no local folder with the same repo-style name is shadowing the HF repo."
            )
        if "local files only" in lowered:
            hints.append(
                "Disable `Local files only` for the first download, or point `HF cache folder` to a complete downloaded snapshot."
            )
        if "401" in lowered or "unauthorized" in lowered or "invalid username" in lowered or "private" in lowered:
            hints.append("Paste a raw `hf_...` token with read access to the model repo.")
        if "403" in lowered or "gated" in lowered:
            hints.append("Accept the model terms on HuggingFace, then use a token that has access to that gated repo.")
        if "winerror 10013" in lowered or "connection" in lowered or "timed out" in lowered:
            hints.append(
                "This machine could not reach HuggingFace. Allow Python through firewall/proxy, pre-download the model, or use a local cache folder."
            )
        if "accelerate" in lowered:
            hints.append(
                "Clear the `LLM device map` field and rebuild. `device_map=auto` requires accelerate; the cached default model does not need it."
            )
            hints.append("Only install or repair `accelerate` if you specifically want `LLM device map` set to `auto`.")
        if "huggingface-hub" in lowered or "huggingface_hub" in lowered:
            hints.append(
                "Use a transformers-compatible `huggingface_hub` version. This workspace also checks `.vendor` first if packages were vendored locally."
            )
        if "embedding dimension mismatch" in lowered:
            hints.append(
                "Update `Embedding dimension` to the value reported in the error. BGE-M3 is 1024; all-MiniLM-L6-v2 is 384."
            )
        if "out of memory" in lowered or ("cuda" in lowered and "memory" in lowered):
            hints.append(
                "The model did not fit in available memory. Use Fast Qwen 0.5B + MiniLM, reduce max new tokens, load on CPU, or serve a quantized model through a local OpenAI-compatible server."
            )
        if "qwen/qwen3-8b" in lowered:
            hints.append("Qwen3-8B is a quality preset but is not the reliable cached default. Try `Cached Qwen 1.5B + MiniLM` first on this machine.")
    if runtime == "Hosted HuggingFace-compatible":
        hints.extend(_hosted_failure_hints(message, surface="chat"))
        hints.extend(_hosted_failure_hints(message, surface="embeddings"))
    if not hints:
        hints.append("Open `Settings` and confirm the active engine matches the selected runtime before retrying.")
    return list(dict.fromkeys(hints))


def _parse_metadata(raw: str) -> dict[str, Any]:
    if not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        st.warning("Metadata JSON is invalid. Using empty metadata.")
        return {}
    if not isinstance(parsed, dict):
        st.warning("Metadata must be a JSON object. Using empty metadata.")
        return {}
    return parsed


def _optional_secret(key: str) -> str | None:
    value = str(st.session_state.get(key, "")).strip()
    return value or None


def _optional_token(key: str) -> str | None:
    value = str(st.session_state.get(key, "")).strip().strip("`").strip('"').strip("'")
    lowered = value.lower()
    for prefix in ("bearer ", "bearer:", "token "):
        if lowered.startswith(prefix):
            value = value[len(prefix) :].strip()
            lowered = value.lower()
    return value or None


def _token_fingerprint(value: str | None) -> str | None:
    if not value:
        return None
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def _hosted_embedding_token() -> str | None:
    return _optional_token("chat_api_token") if st.session_state.reuse_chat_token_for_embeddings else _optional_token(
        "embedding_api_token"
    )


def _hosted_preflight_signature() -> dict[str, Any]:
    return {
        "chat_base_url": str(st.session_state.chat_base_url).strip(),
        "embedding_base_url": str(st.session_state.embedding_base_url).strip(),
        "generation_model": str(st.session_state.generation_model).strip(),
        "hosted_generation_model_id": _hosted_generation_model_id(),
        "hf_router_provider": str(st.session_state.hf_router_provider).strip(),
        "embedding_model": str(st.session_state.embedding_model).strip(),
        "embedding_dimension": int(st.session_state.embedding_dimension),
        "chat_token_fingerprint": _token_fingerprint(_optional_token("chat_api_token")),
        "embedding_token_fingerprint": _token_fingerprint(_hosted_embedding_token()),
        "reuse_chat_token_for_embeddings": bool(st.session_state.reuse_chat_token_for_embeddings),
    }


def _hosted_preflight_ready() -> bool:
    if st.session_state.get("runtime") != "Hosted HuggingFace-compatible":
        return True
    return bool(st.session_state.get("hosted_preflight_ok")) and (
        st.session_state.get("hosted_preflight_signature") == _hosted_preflight_signature()
    )


def _repair_empty_endpoint_defaults() -> None:
    st.session_state.local_auto_repair_notice = ""
    if not str(st.session_state.get("chat_base_url", "")).strip():
        st.session_state.chat_base_url = "https://router.huggingface.co/v1"
    if not str(st.session_state.get("embedding_base_url", "")).strip():
        st.session_state.embedding_base_url = "http://localhost:8080/v1"
    if not str(st.session_state.get("hf_router_provider", "")).strip():
        st.session_state.hf_router_provider = DEFAULT_HF_ROUTER_PROVIDER
    if not str(st.session_state.get("generation_model", "")).strip():
        st.session_state.generation_model = DEFAULT_GENERATION_REPO_ID
    if not str(st.session_state.get("embedding_model", "")).strip():
        st.session_state.embedding_model = DEFAULT_EMBEDDING_REPO_ID
    if not int(st.session_state.get("embedding_dimension", 0) or 0):
        st.session_state.embedding_dimension = DEFAULT_EMBEDDING_DIMENSION
    if st.session_state.get("runtime") == "HuggingFace Hub local":
        _repair_local_model_selection()


def _repair_local_model_selection() -> None:
    notices: list[str] = []
    generation_model = str(st.session_state.get("generation_model", "")).strip()
    embedding_model = str(st.session_state.get("embedding_model", "")).strip()

    old_uncached_generation = {"Qwen/Qwen3-4B", "Qwen/Qwen3-8B"}
    if generation_model in old_uncached_generation and not _hf_cache_ready(generation_model, role="generation"):
        st.session_state.generation_model = DEFAULT_GENERATION_REPO_ID
        notices.append(
            f"Switched generation from `{generation_model}` to cached `{DEFAULT_GENERATION_REPO_ID}`."
        )

    if embedding_model == QUALITY_EMBEDDING_REPO_ID and not _hf_cache_ready(embedding_model, role="embedding"):
        st.session_state.embedding_model = DEFAULT_EMBEDDING_REPO_ID
        st.session_state.embedding_dimension = DEFAULT_EMBEDDING_DIMENSION
        notices.append(
            f"Switched embedding from incomplete cached `{QUALITY_EMBEDDING_REPO_ID}` to cached `{DEFAULT_EMBEDDING_REPO_ID}`."
        )

    if str(st.session_state.get("embedding_model", "")).strip() == DEFAULT_EMBEDDING_REPO_ID:
        st.session_state.embedding_dimension = DEFAULT_EMBEDDING_DIMENSION
    elif str(st.session_state.get("embedding_model", "")).strip() == QUALITY_EMBEDDING_REPO_ID:
        st.session_state.embedding_dimension = QUALITY_EMBEDDING_DIMENSION
    if int(st.session_state.get("hf_max_new_tokens", 0) or 0) == 1024:
        st.session_state.hf_max_new_tokens = DEFAULT_HF_MAX_NEW_TOKENS
        notices.append(f"Reduced max new tokens to `{DEFAULT_HF_MAX_NEW_TOKENS}` for faster local inference.")
    if str(st.session_state.get("hf_device_map", "")).strip().lower() == "auto":
        st.session_state.hf_device_map = DEFAULT_HF_DEVICE_MAP
        notices.append("Cleared `device_map=auto`; the cached default loads without accelerate.")
    if (
        str(st.session_state.get("generation_model", "")).strip() in {DEFAULT_GENERATION_REPO_ID, FAST_GENERATION_REPO_ID}
        and str(st.session_state.get("embedding_model", "")).strip() == DEFAULT_EMBEDDING_REPO_ID
        and not bool(st.session_state.get("hf_local_files_only"))
    ):
        st.session_state.hf_local_files_only = True
        notices.append("Enabled `Local files only` for the cached default model pair.")

    if notices:
        st.session_state.local_auto_repair_notice = " ".join(notices)


def _runtime_config_errors() -> list[str]:
    runtime = st.session_state.get("runtime", "Local deterministic")
    errors: list[str] = []
    if runtime == "Hosted HuggingFace-compatible":
        chat_base_url = str(st.session_state.get("chat_base_url", "")).strip()
        embedding_base_url = str(st.session_state.get("embedding_base_url", "")).strip()
        if not _is_http_url(chat_base_url):
            errors.append(
                "Chat base URL must be a full HTTP(S) URL, for example "
                "`https://router.huggingface.co/v1` or `http://127.0.0.1:8000/v1`."
            )
        if not _is_http_url(embedding_base_url):
            errors.append(
                "Embedding base URL must be a full HTTP(S) URL, for example "
                "`http://127.0.0.1:8080/v1`."
            )
        if _uses_hf_router(chat_base_url):
            chat_token = _optional_token("chat_api_token")
            if not chat_token:
                errors.append("Hugging Face Router requires `Chat HF/API token`.")
            elif chat_token.lower().startswith("hf_your") or chat_token == "hf_...":
                errors.append("Paste a real Hugging Face token, not the placeholder `hf_...`.")
            elif not chat_token.startswith("hf_"):
                errors.append("HF Router tokens normally start with `hf_`. Paste the raw token, not username/password.")
        if not str(st.session_state.get("generation_model", "")).strip():
            errors.append("Generation repo id is required.")
        if not str(st.session_state.get("embedding_model", "")).strip():
            errors.append("Embedding repo id is required.")
    elif runtime == "HuggingFace Hub local":
        if not str(st.session_state.get("generation_model", "")).strip():
            errors.append("Generation repo id is required.")
        if not str(st.session_state.get("embedding_model", "")).strip():
            errors.append("Embedding repo id is required.")
        hf_token = _optional_token("hf_hub_token")
        if hf_token and (hf_token.lower().startswith("hf_your") or hf_token == "hf_..."):
            errors.append("Paste a real Hugging Face token, not the placeholder `hf_...`.")
        cache_folder = _optional_secret("hf_cache_folder")
        if bool(st.session_state.get("hf_local_files_only")) and cache_folder and not os.path.isdir(cache_folder):
            errors.append("`HF cache folder` must exist when `Local files only` is enabled.")
    return errors


def _is_http_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _effective_chunk_overlap() -> int:
    return min(int(st.session_state.chunk_overlap), int(st.session_state.chunk_size) - 1)


def _current_engine_signature() -> dict[str, Any]:
    runtime = st.session_state.get("runtime", "Local deterministic")
    signature: dict[str, Any] = {
        "runtime": runtime,
        "chunk_size": int(st.session_state.chunk_size),
        "chunk_overlap": _effective_chunk_overlap(),
        "max_context_tokens": int(st.session_state.max_context_tokens),
        "max_entities": int(st.session_state.max_entities),
        "max_relationships": int(st.session_state.max_relationships),
        "max_chunks": int(st.session_state.max_chunks),
        "max_communities": int(st.session_state.max_communities),
        "ppr_alpha": float(st.session_state.ppr_alpha),
        "min_relationship_confidence": float(st.session_state.min_relationship_confidence),
        "enable_symbolic_queries": bool(st.session_state.enable_symbolic_queries),
        "enable_community_summaries": bool(st.session_state.enable_community_summaries),
        "graph_extraction_mode": st.session_state.graph_extraction_mode,
        "llm_extraction_max_entities": int(st.session_state.llm_extraction_max_entities),
        "llm_extraction_fallback": bool(st.session_state.llm_extraction_fallback),
    }
    if runtime == "Hosted HuggingFace-compatible":
        signature.update(
            {
                "chat_base_url": st.session_state.chat_base_url,
                "embedding_base_url": st.session_state.embedding_base_url,
                "generation_model": st.session_state.generation_model,
                "hosted_generation_model_id": _hosted_generation_model_id(),
                "hf_router_provider": st.session_state.hf_router_provider,
                "embedding_model": st.session_state.embedding_model,
                "embedding_dimension": int(st.session_state.embedding_dimension),
                "chat_token_configured": bool(_optional_token("chat_api_token")),
                "embedding_token_configured": bool(_hosted_embedding_token()),
                "chat_token_fingerprint": _token_fingerprint(_optional_token("chat_api_token")),
                "embedding_token_fingerprint": _token_fingerprint(_hosted_embedding_token()),
                "reuse_chat_token_for_embeddings": bool(st.session_state.reuse_chat_token_for_embeddings),
            }
        )
    elif runtime == "HuggingFace Hub local":
        signature.update(
            {
                "generation_model": st.session_state.generation_model,
                "embedding_model": st.session_state.embedding_model,
                "embedding_dimension": int(st.session_state.embedding_dimension),
                "hf_token_configured": bool(_optional_token("hf_hub_token")),
                "hf_device_map": st.session_state.hf_device_map,
                "hf_torch_dtype": st.session_state.hf_torch_dtype,
                "hf_max_new_tokens": int(st.session_state.hf_max_new_tokens),
                "hf_enable_thinking": bool(st.session_state.hf_enable_thinking),
                "hf_trust_remote_code": bool(st.session_state.hf_trust_remote_code),
                "hf_local_files_only": bool(st.session_state.hf_local_files_only),
                "hf_embedding_device": st.session_state.hf_embedding_device,
                "hf_embedding_batch_size": int(st.session_state.hf_embedding_batch_size),
                "hf_cache_folder": st.session_state.hf_cache_folder,
            }
        )
    else:
        signature["hash_dimension"] = int(st.session_state.hash_dimension)
    return signature


def _engine_matches_sidebar() -> bool:
    return (
        not _runtime_config_errors()
        and _hosted_preflight_ready()
        and st.session_state.get("engine_signature") == _current_engine_signature()
    )


def _uses_hf_router(base_url: str) -> bool:
    return "router.huggingface.co" in base_url.lower()


def _hosted_generation_model_id() -> str:
    model = str(st.session_state.generation_model).strip()
    provider = str(st.session_state.get("hf_router_provider", "")).strip().lstrip(":")
    if _uses_hf_router(st.session_state.chat_base_url) and provider and ":" not in model:
        return f"{model}:{provider}"
    return model


def _stats() -> dict[str, int]:
    graph = st.session_state.graph_store
    vectors = sum(len(namespace) for namespace in getattr(st.session_state.vector_store, "_items", {}).values())
    return {
        "documents": len(graph.documents_by_id),
        "chunks": len(graph.chunks()),
        "entities": len(graph.entities()),
        "relationships": len(graph.relationships()),
        "communities": len(graph.community_summaries()),
        "vectors": vectors,
    }


def _document_rows() -> list[dict[str, Any]]:
    rows = []
    for document in st.session_state.graph_store.documents_by_id.values():
        rows.append(
            {
                "document_id": document.id,
                "characters": len(document.text),
                "metadata": json.dumps(document.metadata, sort_keys=True),
            }
        )
    return sorted(rows, key=lambda row: row["document_id"])


def _entity_rows(entities: Iterable[Any]) -> list[dict[str, Any]]:
    return [
        {
            "name": entity.name,
            "type": entity.type,
            "id": entity.id,
            "description": entity.description,
        }
        for entity in sorted(entities, key=lambda item: item.name.lower())
    ]


def _relationship_rows(graph_store: Any) -> list[dict[str, Any]]:
    rows = []
    for rel in graph_store.relationships():
        src = graph_store.get_entity(rel.source_entity_id)
        tgt = graph_store.get_entity(rel.target_entity_id)
        rows.append(
            {
                "source": src.name if src else rel.source_entity_id,
                "type": rel.type,
                "target": tgt.name if tgt else rel.target_entity_id,
                "confidence": round(rel.confidence, 3),
                "fact": rel.fact,
            }
        )
    return rows


def _community_rows(graph_store: Any) -> list[dict[str, Any]]:
    return [
        {
            "community_id": summary.community_id,
            "entities": len(summary.entity_ids),
            "summary": summary.text,
        }
        for summary in graph_store.community_summaries()
    ]


def _evidence_rows(evidence: Iterable[Evidence]) -> list[dict[str, Any]]:
    return [
        {
            "kind": item.kind,
            "score": round(float(item.score), 4),
            "id": item.id,
            "sources": ", ".join(item.source_ids),
            "text": item.text,
        }
        for item in evidence
    ]


def _vector_rows() -> list[dict[str, Any]]:
    items = getattr(st.session_state.vector_store, "_items", {})
    rows = []
    for namespace, entries in sorted(items.items()):
        dimensions = []
        for _, (_, vector, _) in entries.items():
            dimensions.append(len(vector))
        rows.append(
            {
                "namespace": namespace,
                "items": len(entries),
                "dimension": dimensions[0] if dimensions else None,
            }
        )
    return rows


def _export_snapshot() -> dict[str, Any]:
    graph = st.session_state.graph_store
    return {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "stats": _stats(),
        "documents": _document_rows(),
        "entities": _entity_rows(graph.entities()),
        "relationships": _relationship_rows(graph),
        "communities": _community_rows(graph),
        "last_query": st.session_state.last_retrieval.query_plan.query if st.session_state.last_retrieval else None,
        "last_answer": st.session_state.last_answer.answer if st.session_state.last_answer else None,
    }


def _graphviz_dot(entities: list[Any], relationships: list[Any]) -> str:
    names_by_id = {entity.id: entity.name for entity in entities}
    lines = [
        "digraph G {",
        '  graph [rankdir=LR, bgcolor="transparent", pad="0.3", nodesep="0.45", ranksep="0.65"];',
        '  node [shape=box, style="rounded,filled", color="#B8C4CC", fillcolor="#FFFFFF", fontname="Inter", fontsize=10, margin="0.10,0.07"];',
        '  edge [color="#6B7A86", fontname="Inter", fontsize=9, arrowsize=0.7];',
    ]
    rel_entity_ids = {rel.source_entity_id for rel in relationships} | {rel.target_entity_id for rel in relationships}
    for entity in entities:
        if entity.id in rel_entity_ids:
            label = _dot_escape(entity.name)
            lines.append(f'  "{_dot_escape(entity.id)}" [label="{label}"];')
    for rel in relationships:
        src = _dot_escape(rel.source_entity_id)
        tgt = _dot_escape(rel.target_entity_id)
        label = _dot_escape(rel.type.replace("_", " "))
        tooltip = _dot_escape(rel.fact)
        if rel.source_entity_id in names_by_id and rel.target_entity_id in names_by_id:
            lines.append(f'  "{src}" -> "{tgt}" [label="{label}", tooltip="{tooltip}"];')
    lines.append("}")
    return "\n".join(lines)


def _dot_escape(value: Any) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def _next_doc_id() -> str:
    graph = st.session_state.get("graph_store")
    count = len(graph.documents_by_id) if graph else 0
    return f"doc_{count + 1}"


def _duration_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def _api_trace_sink(payload: dict[str, Any]) -> None:
    _trace(
        category=str(payload.get("category", "api_call")),
        operation=str(payload.get("operation", "api_call")),
        status=str(payload.get("status", "ok")),
        component=str(payload.get("component", "api")),
        duration_ms=payload.get("duration_ms"),
        detail=payload.get("detail", {}),
        summary=_api_trace_summary(payload),
    )


def _api_trace_summary(payload: dict[str, Any]) -> str:
    detail = payload.get("detail", {}) if isinstance(payload.get("detail"), dict) else {}
    path = detail.get("path") or detail.get("url") or "endpoint"
    status = payload.get("status", "ok")
    return f"{status} {path}"


def _trace(
    *,
    category: str,
    operation: str,
    status: str = "ok",
    component: str = "",
    duration_ms: int | None = None,
    detail: Any | None = None,
    output: Any | None = None,
    error: str | None = None,
    summary: str = "",
) -> None:
    traces = st.session_state.setdefault("trace_events", [])
    now = datetime.now(timezone.utc)
    event = {
        "id": _trace_id(now, len(traces)),
        "time_utc": now.isoformat(),
        "time_local": datetime.now().strftime("%H:%M:%S"),
        "category": category,
        "operation": operation,
        "status": status,
        "component": component,
        "duration_ms": duration_ms,
        "runtime": st.session_state.get("runtime", ""),
        "active_runtime": (st.session_state.get("engine_signature") or {}).get("runtime", ""),
        "summary": summary,
        "detail": _trace_safe(detail or {}),
    }
    if output is not None:
        event["output"] = _trace_safe(output)
    if error is not None:
        event["error"] = _truncate_text(error)
    traces.append(event)
    limit = int(st.session_state.get("trace_max_events", 1000) or 1000)
    if limit > 0:
        del traces[:-limit]


def _trace_id(now: datetime, ordinal: int) -> str:
    raw = f"{now.isoformat()}:{ordinal}:{os.getpid()}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def _trace_safe(value: Any) -> Any:
    if isinstance(value, dict):
        safe: dict[str, Any] = {}
        for key, item in value.items():
            lowered = str(key).lower()
            if any(secret in lowered for secret in ("token", "api_key", "authorization", "password", "secret")):
                safe[str(key)] = "<redacted>"
            else:
                safe[str(key)] = _trace_safe(item)
        return safe
    if isinstance(value, (list, tuple)):
        return [_trace_safe(item) for item in value[:200]]
    if isinstance(value, float):
        return round(value, 8)
    if isinstance(value, (str, bytes)):
        text = value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value
        return _truncate_text(text)
    if isinstance(value, (int, bool)) or value is None:
        return value
    return _truncate_text(str(value))


def _truncate_text(text: str) -> str:
    if len(text) <= TRACE_TEXT_LIMIT:
        return text
    return f"{text[:TRACE_TEXT_LIMIT]}... <truncated {len(text) - TRACE_TEXT_LIMIT} chars>"


def _retrieval_trace(retrieval: RetrievalBundle) -> dict[str, Any]:
    return {
        "query_plan": {
            "query": retrieval.query_plan.query,
            "intent": retrieval.query_plan.intent.value,
            "keywords": list(retrieval.query_plan.keywords),
            "entities": list(retrieval.query_plan.entities),
            "subqueries": list(retrieval.query_plan.subqueries),
            "symbolic_query": retrieval.query_plan.symbolic_query,
        },
        "evidence_count": len(retrieval.evidence),
        "evidence": [
            {
                "id": item.id,
                "kind": item.kind,
                "score": item.score,
                "source_ids": list(item.source_ids),
                "text": item.text,
                "metadata": item.metadata,
            }
            for item in retrieval.evidence[:50]
        ],
        "context": retrieval.context,
    }


def _answer_trace(answer: Answer) -> dict[str, Any]:
    return {
        "answer": answer.answer,
        "confidence": answer.confidence,
        "citations": list(answer.citations),
        "graph_evidence": list(answer.graph_evidence),
        "retrieval": _retrieval_trace(answer.retrieval),
    }


def _event(name: str, payload: dict[str, Any]) -> None:
    events = st.session_state.setdefault("events", [])
    events.append(
        {
            "time": datetime.now().strftime("%H:%M:%S"),
            "event": name,
            "details": json.dumps(payload, sort_keys=True),
        }
    )
    del events[:-50]
    _trace(
        category="app_event",
        operation=name,
        status="ok",
        component="ui",
        detail=payload,
        summary=name,
    )


def _rerun() -> None:
    if hasattr(st, "rerun"):
        st.rerun()
    else:  # pragma: no cover - old Streamlit fallback
        st.experimental_rerun()


def _inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --panel: #ffffff;
            --ink: #172026;
            --muted: #657381;
            --line: #d8dee6;
            --teal: #0f766e;
            --teal-hover: #115e59;
            --teal-active: #134e4a;
            --button-bg: #ffffff;
            --button-text: #102028;
            --button-border: #aab7c3;
            --button-hover-bg: #eef7f6;
            --amber: #b45309;
            --rose: #be123c;
        }
        html, body, [data-testid="stAppViewContainer"] {
            background: #f6f8fa !important;
            color: var(--ink) !important;
        }
        .stApp {
            background: #f6f8fa !important;
            color: var(--ink) !important;
        }
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2.5rem;
            max-width: 1520px;
        }
        .topbar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            margin-bottom: 0.75rem;
            border: 1px solid var(--line);
            background: var(--panel);
            border-radius: 8px;
            padding: 1rem 1.1rem;
        }
        .topbar h1 {
            margin: 0;
            font-size: 1.55rem;
            line-height: 1.1;
            letter-spacing: 0;
        }
        .eyebrow {
            color: var(--muted);
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0;
            margin-bottom: 0.2rem;
        }
        .runtime-pill {
            border: 1px solid #b6ded8;
            background: #eefaf7;
            color: #0f5c55;
            border-radius: 999px;
            padding: 0.42rem 0.7rem;
            font-weight: 650;
            white-space: nowrap;
        }
        [data-testid="stMetric"] {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 0.7rem 0.8rem;
        }
        [data-testid="stMetricValue"] {
            font-size: 1.35rem;
        }
        .answer-panel {
            border: 1px solid #b6ded8;
            background: #fbfffe;
            border-radius: 8px;
            padding: 0.85rem 1rem;
            margin: 0.5rem 0 1rem 0;
        }
        div[data-testid="stDataFrame"] {
            border: 1px solid var(--line);
            border-radius: 8px;
            overflow: hidden;
        }
        section[data-testid="stSidebar"] {
            border-right: 1px solid var(--line);
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.25rem;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 6px;
            padding: 0.45rem 0.8rem;
        }
        .stButton > button,
        .stDownloadButton > button,
        button[kind="secondary"],
        button[data-testid="baseButton-secondary"],
        button[data-testid="stBaseButton-secondary"] {
            background: var(--button-bg) !important;
            color: var(--button-text) !important;
            border: 1px solid var(--button-border) !important;
            border-radius: 6px !important;
            box-shadow: none !important;
            font-weight: 650 !important;
            min-height: 2.35rem;
        }
        .stButton > button:hover,
        .stDownloadButton > button:hover,
        button[kind="secondary"]:hover,
        button[data-testid="baseButton-secondary"]:hover,
        button[data-testid="stBaseButton-secondary"]:hover {
            background: var(--button-hover-bg) !important;
            color: var(--teal-active) !important;
            border-color: var(--teal) !important;
        }
        .stButton > button:focus,
        .stDownloadButton > button:focus,
        button[kind="secondary"]:focus,
        button[data-testid="baseButton-secondary"]:focus,
        button[data-testid="stBaseButton-secondary"]:focus {
            color: var(--button-text) !important;
            border-color: var(--teal) !important;
            box-shadow: 0 0 0 0.18rem rgba(15, 118, 110, 0.18) !important;
        }
        .stButton > button:active,
        .stDownloadButton > button:active,
        button[kind="secondary"]:active,
        button[data-testid="baseButton-secondary"]:active,
        button[data-testid="stBaseButton-secondary"]:active {
            background: #dff0ee !important;
            color: var(--teal-active) !important;
            border-color: var(--teal-active) !important;
        }
        button[kind="primary"],
        button[data-testid="baseButton-primary"],
        button[data-testid="stBaseButton-primary"] {
            background: var(--teal) !important;
            color: #ffffff !important;
            border: 1px solid var(--teal) !important;
            border-radius: 6px !important;
            box-shadow: none !important;
            font-weight: 700 !important;
            min-height: 2.35rem;
        }
        button[kind="primary"]:hover,
        button[data-testid="baseButton-primary"]:hover,
        button[data-testid="stBaseButton-primary"]:hover {
            background: var(--teal-hover) !important;
            color: #ffffff !important;
            border-color: var(--teal-hover) !important;
        }
        button[kind="primary"]:focus,
        button[data-testid="baseButton-primary"]:focus,
        button[data-testid="stBaseButton-primary"]:focus {
            background: var(--teal) !important;
            color: #ffffff !important;
            border-color: var(--teal) !important;
            box-shadow: 0 0 0 0.18rem rgba(15, 118, 110, 0.22) !important;
        }
        button[kind="primary"]:active,
        button[data-testid="baseButton-primary"]:active,
        button[data-testid="stBaseButton-primary"]:active {
            background: var(--teal-active) !important;
            color: #ffffff !important;
            border-color: var(--teal-active) !important;
        }
        button:disabled,
        button[disabled] {
            background: #edf1f4 !important;
            color: #7a8792 !important;
            border-color: #d4dbe2 !important;
            opacity: 1 !important;
        }
        textarea, input {
            background: #ffffff !important;
            color: var(--ink) !important;
            border-radius: 6px !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
