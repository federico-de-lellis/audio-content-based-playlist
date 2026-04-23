from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
ANALYSIS_DIR = PROJECT_ROOT / "analysis"
PLAYLIST_DIR = PROJECT_ROOT / "playlists"
LEGACY_STYLE_PICKLE = PROJECT_ROOT / "data" / "files_essentia_effnet-discogs.jsonl.pickle"
METADATA_DIR = PROJECT_ROOT / "metadata"
DEFAULT_TRACK_INDEX = ANALYSIS_DIR / "track_index.parquet"
DEFAULT_TRACKS = ANALYSIS_DIR / "tracks.parquet"
DEFAULT_STYLES = ANALYSIS_DIR / "style_activations.parquet"
DEFAULT_DISCOGS_EMBEDDINGS = ANALYSIS_DIR / "discogs_embeddings.npy"
DEFAULT_CLAP_AUDIO_EMBEDDINGS = ANALYSIS_DIR / "clap_audio_embeddings.npy"
DEFAULT_MANIFEST = ANALYSIS_DIR / "manifest.json"


@dataclass
class AnalysisBundle:
    track_index: pd.DataFrame
    tracks: pd.DataFrame
    styles: pd.DataFrame
    manifest: dict[str, Any]
    discogs_embeddings: np.ndarray | None = None
    clap_audio_embeddings: np.ndarray | None = None

    def audio_root(self, override_root: str | Path | None = None) -> Path:
        if override_root:
            override_path = Path(override_root).expanduser()
            if not override_path.is_absolute():
                override_path = PROJECT_ROOT / override_path
            return override_path.resolve()

        configured_root = self.manifest.get("audio_root", ".")
        resolved = (PROJECT_ROOT / configured_root).resolve()
        if resolved.exists():
            return resolved

        fallback_roots = [
            PROJECT_ROOT / "MusAV" / "audio_chunks",
            PROJECT_ROOT / "audio",
            PROJECT_ROOT,
        ]
        for candidate in fallback_roots:
            if candidate.exists():
                return candidate.resolve()
        return resolved

    def audio_path(self, track_id: str, override_root: str | Path | None = None) -> Path:
        relative_path = self.track_index.loc[track_id, "relative_path"]
        return (self.audio_root(override_root) / relative_path).resolve()

    @property
    def track_ids(self) -> list[str]:
        return list(self.track_index.index)

    @property
    def style_labels(self) -> list[str]:
        return list(self.styles.columns)

    def has_column(self, name: str) -> bool:
        return name in self.tracks.columns and self.tracks[name].notna().any()

    def has_discogs_embeddings(self) -> bool:
        return self.discogs_embeddings is not None and len(self.discogs_embeddings) == len(self.track_index)

    def has_clap_audio_embeddings(self) -> bool:
        return self.clap_audio_embeddings is not None and len(self.clap_audio_embeddings) == len(self.track_index)


def _read_json(path: Path, default: dict[str, Any] | None = None) -> dict[str, Any]:
    if not path.exists():
        return {} if default is None else default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def project_relative_string(path: str | Path) -> str:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        return candidate.as_posix()

    try:
        return candidate.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return candidate.resolve().as_posix()


def _read_parquet(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_parquet(path)


def _read_npy(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    return np.load(path)


def _ensure_track_index_index(track_index: pd.DataFrame) -> pd.DataFrame:
    if "track_id" in track_index.columns:
        track_index = track_index.set_index("track_id")
    track_index.index.name = "track_id"
    return track_index.sort_index()


def _ensure_tracks_index(tracks: pd.DataFrame, fallback_index: pd.Index) -> pd.DataFrame:
    if tracks is None:
        return pd.DataFrame(index=fallback_index)
    if "track_id" in tracks.columns:
        tracks = tracks.set_index("track_id")
    tracks.index.name = "track_id"
    return tracks.reindex(fallback_index)


def _ensure_styles_index(styles: pd.DataFrame, fallback_index: pd.Index) -> pd.DataFrame:
    if styles is None:
        return pd.DataFrame(index=fallback_index)
    if "track_id" in styles.columns:
        styles = styles.set_index("track_id")
    styles.index.name = "track_id"
    return styles.reindex(fallback_index)


def load_default_style_labels() -> list[str]:
    metadata_path = METADATA_DIR / "genre_discogs400-discogs-effnet-1.json"
    if metadata_path.exists():
        metadata = _read_json(metadata_path)
        classes = metadata.get("classes")
        if isinstance(classes, list) and classes:
            return [str(label) for label in classes]

    if not LEGACY_STYLE_PICKLE.exists():
        return []
    legacy_styles = pd.read_pickle(LEGACY_STYLE_PICKLE)
    return list(legacy_styles.columns)


def _load_legacy_bundle() -> AnalysisBundle:
    styles = pd.read_pickle(LEGACY_STYLE_PICKLE).sort_index()
    track_index = pd.DataFrame({"relative_path": styles.index}, index=styles.index)
    track_index.index.name = "track_id"
    tracks = pd.DataFrame({"relative_path": styles.index}, index=styles.index)
    tracks.index.name = "track_id"
    manifest = {
        "analysis_version": "legacy-discogs-style-demo",
        "audio_root": ".",
        "track_count": len(styles),
        "generated_from": str(LEGACY_STYLE_PICKLE.relative_to(PROJECT_ROOT)),
        "uses_legacy_bundle": True,
    }
    return AnalysisBundle(track_index=track_index, tracks=tracks, styles=styles, manifest=manifest)


def load_analysis_bundle(
    analysis_dir: str | Path = ANALYSIS_DIR,
    *,
    load_discogs_embeddings: bool = False,
    load_clap_embeddings: bool = False,
    allow_legacy: bool = True,
) -> AnalysisBundle:
    analysis_dir = Path(analysis_dir)
    manifest = _read_json(analysis_dir / "manifest.json", default={"audio_root": "MusAV/audio_chunks"})
    track_index = _read_parquet(analysis_dir / "track_index.parquet")
    tracks = _read_parquet(analysis_dir / "tracks.parquet")
    styles = _read_parquet(analysis_dir / "style_activations.parquet")

    if track_index is None and allow_legacy and LEGACY_STYLE_PICKLE.exists():
        return _load_legacy_bundle()

    if track_index is None:
        raise FileNotFoundError(
            "Analysis artifacts not found. Run scripts/analyze_collection.py first or keep the legacy style demo data."
        )

    track_index = _ensure_track_index_index(track_index)
    tracks = _ensure_tracks_index(tracks, track_index.index)
    styles = _ensure_styles_index(styles, track_index.index)

    if "relative_path" not in track_index.columns and "relative_path" in tracks.columns:
        track_index["relative_path"] = tracks["relative_path"]
    if "relative_path" not in track_index.columns:
        track_index["relative_path"] = track_index.index
    if "relative_path" not in tracks.columns:
        tracks["relative_path"] = track_index["relative_path"]

    discogs_embeddings = None
    if load_discogs_embeddings:
        discogs_embeddings = _read_npy(analysis_dir / "discogs_embeddings.npy")

    clap_audio_embeddings = None
    if load_clap_embeddings:
        clap_audio_embeddings = _read_npy(analysis_dir / "clap_audio_embeddings.npy")

    return AnalysisBundle(
        track_index=track_index,
        tracks=tracks,
        styles=styles,
        manifest=manifest,
        discogs_embeddings=discogs_embeddings,
        clap_audio_embeddings=clap_audio_embeddings,
    )


def write_m3u8(audio_paths: Iterable[str | Path], destination: str | Path) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    playlist_root = destination.parent.resolve()
    lines = []
    for path in audio_paths:
        audio_path = Path(path).expanduser()
        if not audio_path.is_absolute():
            audio_path = (PROJECT_ROOT / audio_path).resolve()
        else:
            audio_path = audio_path.resolve()
        relative_line = os.path.relpath(audio_path, start=playlist_root)
        lines.append(Path(relative_line).as_posix())
    destination.write_text("\n".join(lines), encoding="utf-8")
    return destination


def cosine_scores(query_embedding: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
    query = np.asarray(query_embedding, dtype=np.float32)
    matrix = np.asarray(embeddings, dtype=np.float32)
    query_norm = np.linalg.norm(query)
    matrix_norms = np.linalg.norm(matrix, axis=1)
    safe_denominator = np.maximum(matrix_norms * query_norm, 1e-12)
    return (matrix @ query) / safe_denominator


def dot_scores(query_embedding: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
    query = np.asarray(query_embedding, dtype=np.float32)
    matrix = np.asarray(embeddings, dtype=np.float32)
    return matrix @ query


def score_embeddings(query_embedding: np.ndarray, embeddings: np.ndarray, metric: str = "cosine") -> np.ndarray:
    if metric == "cosine":
        return cosine_scores(query_embedding, embeddings)
    if metric == "dot":
        return dot_scores(query_embedding, embeddings)
    raise ValueError(f"Unsupported similarity metric: {metric}")


def rank_tracks_by_embedding(
    query_embedding: np.ndarray,
    embeddings: np.ndarray,
    track_ids: Sequence[str],
    *,
    metric: str = "cosine",
    exclude_track_id: str | None = None,
    limit: int = 10,
) -> pd.DataFrame:
    scores = score_embeddings(query_embedding, embeddings, metric=metric)
    ranking = pd.DataFrame({"track_id": list(track_ids), "score": scores})
    if exclude_track_id is not None:
        ranking = ranking.loc[ranking["track_id"] != exclude_track_id]
    ranking = ranking.sort_values("score", ascending=False).head(limit).reset_index(drop=True)
    ranking["rank"] = np.arange(1, len(ranking) + 1)
    return ranking[["rank", "track_id", "score"]]


def pretty_track_label(track_id: str) -> str:
    path = Path(track_id)
    return f"{path.stem} - {track_id}"


def parent_genre(style_label: str) -> str:
    return style_label.split("---", 1)[0]


def combine_key_and_scale(key: str | float | None, scale: str | float | None) -> str | None:
    if key is None or scale is None:
        return None
    if pd.isna(key) or pd.isna(scale):
        return None
    return f"{key} {scale}"


def top_style_per_track(styles: pd.DataFrame) -> pd.Series:
    if styles.empty:
        return pd.Series(dtype="object")
    return styles.idxmax(axis=1)


def top_parent_genre_per_track(styles: pd.DataFrame) -> pd.Series:
    top_styles = top_style_per_track(styles)
    return top_styles.map(parent_genre)
