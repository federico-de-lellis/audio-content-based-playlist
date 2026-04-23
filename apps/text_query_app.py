from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from playlist_tools import (
    PLAYLIST_DIR,
    load_analysis_bundle,
    pretty_track_label,
    project_relative_string,
    rank_tracks_by_embedding,
    write_m3u8,
)


TEXT_PLAYLIST_PATH = PLAYLIST_DIR / "text_query.m3u8"
DEFAULT_CLAP_CHECKPOINT = "music_speech_epoch_15_esc_89.25.pt"
DEFAULT_CLAP_AMODEL = "HTSAT-base"


@st.cache_resource(show_spinner=False)
def get_bundle():
    return load_analysis_bundle(load_clap_embeddings=True)


def resolve_checkpoint_path(raw_value: str) -> Path:
    if not raw_value:
        raw_value = DEFAULT_CLAP_CHECKPOINT
    candidate = Path(raw_value).expanduser()
    if candidate.is_absolute():
        return candidate
    return (PROJECT_ROOT / candidate).resolve()


@st.cache_resource(show_spinner=False)
def load_clap_model(checkpoint_path: str, device: str):
    try:
        os.environ.setdefault("HF_HOME", str((PROJECT_ROOT / ".cache" / "huggingface").resolve()))
        import laion_clap
    except Exception as exc:  # pragma: no cover - dependency-specific
        raise RuntimeError("laion-clap is not installed. Install requirements.txt first.") from exc

    model = laion_clap.CLAP_Module(
        enable_fusion=False,
        device=device,
        amodel=DEFAULT_CLAP_AMODEL,
        tmodel="roberta",
    )
    model.load_ckpt(checkpoint_path)
    return model


def embed_text_query(model, query: str) -> np.ndarray:
    embedding = model.get_text_embedding([query], use_tensor=False)
    embedding = np.asarray(embedding, dtype=np.float32)
    if embedding.ndim == 2:
        return embedding[0]
    return embedding.reshape(-1)


def main() -> None:
    st.set_page_config(page_title="Text Query Search", layout="wide")
    bundle = get_bundle()

    st.title("Freeform Text-to-Audio Search")
    st.write("Match a natural-language query against precomputed CLAP audio embeddings.")

    if not bundle.has_clap_audio_embeddings():
        st.error(
            "Text search requires `analysis/clap_audio_embeddings.npy`. "
            "Run `scripts/analyze_collection.py` first."
        )
        return

    default_checkpoint = os.environ.get("CLAP_CHECKPOINT", f"models/{DEFAULT_CLAP_CHECKPOINT}")
    checkpoint_input = st.sidebar.text_input("CLAP checkpoint path", value=default_checkpoint)
    device = st.sidebar.selectbox("CLAP device", ["cpu", "cuda"], index=0)
    audio_root_override = st.sidebar.text_input(
        "Audio root override",
        value=bundle.manifest.get("audio_root", ""),
    ).strip()

    query = st.text_input("Describe the kind of music you want", placeholder="warm ambient electronic with soft female vocals")
    metric = st.radio("Similarity metric", ["cosine", "dot"], horizontal=True)
    limit = st.slider("Results to show", min_value=1, max_value=20, value=10)

    if st.button("Search", type="primary"):
        if not query.strip():
            st.warning("Enter a text query first.")
            return

        checkpoint_path = resolve_checkpoint_path(checkpoint_input)
        if not checkpoint_path.exists():
            st.error(f"CLAP checkpoint not found: {checkpoint_path}")
            return

        try:
            model = load_clap_model(str(checkpoint_path), device)
            query_embedding = embed_text_query(model, query.strip())
        except Exception as exc:  # pragma: no cover - dependency-specific
            st.error(str(exc))
            return

        ranking = rank_tracks_by_embedding(
            query_embedding,
            bundle.clap_audio_embeddings,
            bundle.track_ids,
            metric=metric,
            limit=limit,
        )
        playlist_path = write_m3u8(
            [bundle.audio_path(track_id, audio_root_override or None) for track_id in ranking["track_id"]],
            TEXT_PLAYLIST_PATH,
        )

        st.success(f"Wrote ranked text-query playlist to `{project_relative_string(playlist_path)}`.")
        st.dataframe(ranking)

        for _, row in ranking.iterrows():
            track_id = row["track_id"]
            st.caption(f"#{int(row['rank'])} - {pretty_track_label(track_id)} (score={row['score']:.4f})")
            st.audio(str(bundle.audio_path(track_id, audio_root_override or None)), format="audio/mp3", start_time=0)


if __name__ == "__main__":
    main()
