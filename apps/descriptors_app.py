from __future__ import annotations

import random
import sys
from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from playlist_tools import (
    PLAYLIST_DIR,
    combine_key_and_scale,
    load_analysis_bundle,
    pretty_track_label,
    project_relative_string,
    top_style_per_track,
    write_m3u8,
)


PLAYLIST_PATH = PLAYLIST_DIR / "descriptors_playlist.m3u8"


@st.cache_resource(show_spinner=False)
def get_bundle():
    return load_analysis_bundle()


def available_key_profiles(bundle) -> list[str]:
    profiles = []
    for profile in ("temperley", "krumhansl", "edma"):
        if bundle.has_column(f"key_{profile}") and bundle.has_column(f"scale_{profile}"):
            profiles.append(profile)
    return profiles


def main() -> None:
    st.set_page_config(page_title="Descriptor Playlists", layout="wide")
    bundle = get_bundle()

    st.title("Descriptor-Based Playlist Generator")
    st.write(f"Loaded analysis for {len(bundle.track_index)} tracks.")

    if bundle.manifest.get("uses_legacy_bundle"):
        st.info(
            "The app is currently using the repository's legacy Discogs style artifact. "
            "Run `scripts/analyze_collection.py` to unlock tempo, danceability, key, "
            "voice/instrumental, and embedding-based features."
        )

    audio_root_override = st.sidebar.text_input(
        "Audio root override",
        value=bundle.manifest.get("audio_root", ""),
        help="Override the audio root recorded in the analysis manifest when needed.",
    ).strip()

    styles = bundle.style_labels
    st.subheader("Style Filters")
    style_select = st.multiselect("Select styles by activation range", styles)
    style_select_range = (0.0, 1.0)
    if style_select:
        style_select_range = st.slider(
            "Activation range for the selected styles",
            min_value=0.0,
            max_value=1.0,
            value=(0.5, 1.0),
            step=0.01,
        )
        st.write(bundle.styles[style_select].describe())

    st.subheader("Descriptor Filters")
    tempo_range = None
    if bundle.has_column("tempo_bpm"):
        tempo_series = bundle.tracks["tempo_bpm"].dropna()
        tempo_range = st.slider(
            "Tempo range (BPM)",
            min_value=float(tempo_series.min()),
            max_value=float(tempo_series.max()),
            value=(float(tempo_series.min()), float(tempo_series.max())),
            step=1.0,
        )

    danceability_range = None
    if bundle.has_column("danceability_prob"):
        danceability_series = bundle.tracks["danceability_prob"].dropna()
        danceability_range = st.slider(
            "Danceability range",
            min_value=float(danceability_series.min()),
            max_value=float(danceability_series.max()),
            value=(float(danceability_series.min()), float(danceability_series.max())),
            step=0.01,
        )

    voice_filter = "Any"
    voice_threshold = 0.5
    if bundle.has_column("voice_prob") and bundle.has_column("instrumental_prob"):
        voice_filter = st.selectbox("Voice / instrumental", ["Any", "With voice", "Instrumental"])
        voice_threshold = st.slider("Voice classifier threshold", 0.0, 1.0, 0.5, 0.05)

    key_profiles = available_key_profiles(bundle)
    selected_key_profile = None
    selected_key = "Any"
    selected_scale = "Any"
    if key_profiles:
        selected_key_profile = st.selectbox("Key profile", key_profiles)
        key_values = sorted(bundle.tracks[f"key_{selected_key_profile}"].dropna().unique().tolist())
        selected_key = st.selectbox("Key", ["Any"] + key_values)
        selected_scale = st.selectbox("Scale", ["Any", "major", "minor"])

    st.subheader("Ranking and Post-processing")
    style_rank = st.multiselect(
        "Rank by style activations (multiply selected activations)",
        styles,
    )
    max_tracks = int(
        st.number_input("Maximum number of tracks (0 for all)", min_value=0, value=25, step=1)
    )
    shuffle = st.checkbox("Random shuffle after ranking/filtering", value=False)

    if st.button("Generate Playlist", type="primary"):
        result_ids = pd.Index(bundle.track_ids)

        if style_select:
            style_subset = bundle.styles.reindex(result_ids)[style_select]
            style_mask = pd.Series(True, index=result_ids)
            for style in style_select:
                style_mask &= style_subset[style].between(style_select_range[0], style_select_range[1], inclusive="both")
            result_ids = style_mask[style_mask].index

        track_subset = bundle.tracks.reindex(result_ids)

        if tempo_range is not None:
            tempo_mask = track_subset["tempo_bpm"].between(tempo_range[0], tempo_range[1], inclusive="both")
            result_ids = tempo_mask[tempo_mask].index
            track_subset = bundle.tracks.reindex(result_ids)

        if danceability_range is not None:
            dance_mask = track_subset["danceability_prob"].between(
                danceability_range[0],
                danceability_range[1],
                inclusive="both",
            )
            result_ids = dance_mask[dance_mask].index
            track_subset = bundle.tracks.reindex(result_ids)

        if voice_filter == "With voice":
            voice_mask = track_subset["voice_prob"].fillna(0.0) >= voice_threshold
            result_ids = voice_mask[voice_mask].index
            track_subset = bundle.tracks.reindex(result_ids)
        elif voice_filter == "Instrumental":
            instrumental_mask = track_subset["instrumental_prob"].fillna(0.0) >= voice_threshold
            result_ids = instrumental_mask[instrumental_mask].index
            track_subset = bundle.tracks.reindex(result_ids)

        if selected_key_profile is not None:
            if selected_key != "Any":
                key_mask = track_subset[f"key_{selected_key_profile}"] == selected_key
                result_ids = key_mask[key_mask].index
                track_subset = bundle.tracks.reindex(result_ids)
            if selected_scale != "Any":
                scale_mask = track_subset[f"scale_{selected_key_profile}"] == selected_scale
                result_ids = scale_mask[scale_mask].index
                track_subset = bundle.tracks.reindex(result_ids)

        ranking_scores = pd.Series(dtype="float64")
        if style_rank:
            rank_frame = bundle.styles.reindex(result_ids)[style_rank].copy()
            ranking_scores = rank_frame[style_rank[0]].copy()
            for style in style_rank[1:]:
                ranking_scores *= rank_frame[style]
            ranking_scores = ranking_scores.sort_values(ascending=False)
            result_ids = ranking_scores.index
            track_subset = bundle.tracks.reindex(result_ids)

        result_ids = list(result_ids)
        if shuffle:
            random.shuffle(result_ids)
        if max_tracks > 0:
            result_ids = result_ids[:max_tracks]

        result_tracks = bundle.tracks.reindex(result_ids).copy()
        if not bundle.styles.empty and result_ids:
            top_styles = top_style_per_track(bundle.styles.reindex(result_ids))
            result_tracks["top_style"] = top_styles
        if not ranking_scores.empty:
            result_tracks["style_rank_score"] = ranking_scores.reindex(result_ids)
        if selected_key_profile is not None and result_ids:
            result_tracks["selected_key_label"] = result_tracks.apply(
                lambda row: combine_key_and_scale(
                    row.get(f"key_{selected_key_profile}"),
                    row.get(f"scale_{selected_key_profile}"),
                ),
                axis=1,
            )

        audio_paths = [bundle.audio_path(track_id, audio_root_override or None) for track_id in result_ids]
        playlist_path = write_m3u8(audio_paths, PLAYLIST_PATH)

        st.success(
            f"Generated {len(result_ids)} tracks and wrote `{project_relative_string(playlist_path)}`."
        )
        if result_tracks.empty:
            st.warning("No tracks matched the current query.")
            return

        display_columns = [
            column
            for column in [
                "tempo_bpm",
                "danceability_prob",
                "voice_prob",
                "instrumental_prob",
                "voice_label",
                "top_style",
                "style_rank_score",
                "selected_key_label",
            ]
            if column in result_tracks.columns
        ]
        st.dataframe(result_tracks[display_columns] if display_columns else result_tracks)

        st.subheader("Audio previews")
        for track_id in result_ids[:10]:
            st.caption(pretty_track_label(track_id))
            st.audio(str(bundle.audio_path(track_id, audio_root_override or None)), format="audio/mp3", start_time=0)


if __name__ == "__main__":
    main()
