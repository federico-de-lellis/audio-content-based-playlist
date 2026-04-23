from __future__ import annotations

import sys
from pathlib import Path

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


DISCOGS_PLAYLIST_PATH = PLAYLIST_DIR / "similarity_discogs.m3u8"
CLAP_PLAYLIST_PATH = PLAYLIST_DIR / "similarity_clap.m3u8"


@st.cache_resource(show_spinner=False)
def get_bundle():
    return load_analysis_bundle(load_discogs_embeddings=True, load_clap_embeddings=True)


def render_result_column(title: str, ranking, bundle, audio_root_override: str) -> None:
    st.subheader(title)
    if ranking.empty:
        st.warning("No results available.")
        return

    st.dataframe(ranking)
    for _, row in ranking.iterrows():
        track_id = row["track_id"]
        st.caption(f"#{int(row['rank'])} - {pretty_track_label(track_id)} (score={row['score']:.4f})")
        st.audio(str(bundle.audio_path(track_id, audio_root_override or None)), format="audio/mp3", start_time=0)


def main() -> None:
    st.set_page_config(page_title="Similarity Search", layout="wide")
    bundle = get_bundle()

    st.title("Track Similarity Search")
    st.write("Compare nearest-neighbour results from mean Discogs-Effnet embeddings and mean CLAP embeddings.")

    if not bundle.has_discogs_embeddings() or not bundle.has_clap_audio_embeddings():
        st.error(
            "Similarity search requires `analysis/discogs_embeddings.npy` and `analysis/clap_audio_embeddings.npy`. "
            "Run `scripts/analyze_collection.py` first."
        )
        return

    audio_root_override = st.sidebar.text_input(
        "Audio root override",
        value=bundle.manifest.get("audio_root", ""),
    ).strip()

    query_track = st.selectbox("Query track", bundle.track_ids, format_func=pretty_track_label)
    metric = st.radio("Similarity metric", ["cosine", "dot"], horizontal=True)
    limit = st.slider("Results per list", min_value=1, max_value=20, value=10)

    if st.button("Find Similar Tracks", type="primary"):
        query_index = bundle.track_index.index.get_loc(query_track)

        discogs_ranking = rank_tracks_by_embedding(
            bundle.discogs_embeddings[query_index],
            bundle.discogs_embeddings,
            bundle.track_ids,
            metric=metric,
            exclude_track_id=query_track,
            limit=limit,
        )
        clap_ranking = rank_tracks_by_embedding(
            bundle.clap_audio_embeddings[query_index],
            bundle.clap_audio_embeddings,
            bundle.track_ids,
            metric=metric,
            exclude_track_id=query_track,
            limit=limit,
        )

        discogs_playlist = write_m3u8(
            [bundle.audio_path(track_id, audio_root_override or None) for track_id in discogs_ranking["track_id"]],
            DISCOGS_PLAYLIST_PATH,
        )
        clap_playlist = write_m3u8(
            [bundle.audio_path(track_id, audio_root_override or None) for track_id in clap_ranking["track_id"]],
            CLAP_PLAYLIST_PATH,
        )

        st.success(
            "Wrote similarity playlists to "
            f"`{project_relative_string(discogs_playlist)}` and `{project_relative_string(clap_playlist)}`."
        )

        st.subheader("Query track")
        st.caption(pretty_track_label(query_track))
        st.audio(str(bundle.audio_path(query_track, audio_root_override or None)), format="audio/mp3", start_time=0)

        left, right = st.columns(2)
        with left:
            render_result_column("Discogs-Effnet neighbours", discogs_ranking, bundle, audio_root_override)
        with right:
            render_result_column("CLAP neighbours", clap_ranking, bundle, audio_root_override)


if __name__ == "__main__":
    main()
