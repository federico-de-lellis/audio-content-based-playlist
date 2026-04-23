from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from playlist_tools import load_default_style_labels


KEY_PROFILES = ("temperley", "krumhansl", "edma")
DEFAULT_CLAP_CHECKPOINT = "music_speech_epoch_15_esc_89.25.pt"
DEFAULT_CLAP_AMODEL = "HTSAT-base"
METADATA_DIR = PROJECT_ROOT / "metadata"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze a nested MP3 collection with Essentia and CLAP.")
    parser.add_argument("--audio-root", default="MusAV/audio_chunks", help="Root folder containing MP3 files.")
    parser.add_argument("--output-dir", default="analysis", help="Directory for cached and consolidated outputs.")
    parser.add_argument("--models-dir", default="models", help="Directory containing Essentia and CLAP model files.")
    parser.add_argument("--device", default="cpu", help="CLAP device, for example cpu or cuda.")
    parser.add_argument("--max-files", type=int, default=None, help="Optional smoke-test limit.")
    parser.add_argument("--force", action="store_true", help="Recompute tracks even when cached JSON exists.")
    parser.add_argument(
        "--clap-checkpoint",
        default=DEFAULT_CLAP_CHECKPOINT,
        help="Filename or path for the CLAP checkpoint.",
    )
    parser.add_argument(
        "--clap-amodel",
        default=DEFAULT_CLAP_AMODEL,
        help="CLAP audio backbone, for example HTSAT-base or HTSAT-tiny.",
    )
    return parser.parse_args()


def discover_mp3s(audio_root: Path) -> list[Path]:
    return sorted(path for path in audio_root.rglob("*.mp3") if path.is_file())


def json_cache_path(per_track_dir: Path, relative_path: Path) -> Path:
    return per_track_dir / relative_path.with_suffix(".json")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def project_relative_string(path: str | Path) -> str:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        return candidate.as_posix()

    try:
        return candidate.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return candidate.resolve().as_posix()


def resolve_model_file(models_dir: Path, candidates: list[str]) -> Path:
    for candidate in candidates:
        candidate_path = Path(candidate)
        if candidate_path.is_absolute() and candidate_path.exists():
            return candidate_path

        direct = models_dir / candidate
        if direct.exists():
            return direct

        glob_matches = sorted(models_dir.glob(candidate))
        if glob_matches:
            return glob_matches[0]

    raise FileNotFoundError(f"Could not find any of the expected model files in {models_dir}: {candidates}")


def load_metadata_classes(filename: str) -> list[str]:
    path = METADATA_DIR / filename
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    classes = payload.get("classes", [])
    if not isinstance(classes, list):
        return []
    return [str(item) for item in classes]


def load_metadata_io(
    filename: str,
    *,
    default_input: str | None = None,
    default_output: str | None = None,
    preferred_output_purpose: str = "predictions",
) -> tuple[str | None, str | None]:
    path = METADATA_DIR / filename
    if not path.exists():
        return default_input, default_output

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    schema = payload.get("schema", {})
    inputs = schema.get("inputs", [])
    outputs = schema.get("outputs", [])

    input_name = default_input
    if isinstance(inputs, list) and inputs:
        first_input = inputs[0]
        if isinstance(first_input, dict):
            input_name = first_input.get("name", input_name)

    output_name = default_output
    if isinstance(outputs, list) and outputs:
        preferred_output = None
        for output in outputs:
            if isinstance(output, dict) and output.get("output_purpose") == preferred_output_purpose:
                preferred_output = output
                break
        if preferred_output is None and isinstance(outputs[0], dict):
            preferred_output = outputs[0]
        if preferred_output is not None:
            output_name = preferred_output.get("name", output_name)

    return input_name, output_name


def ensure_runtime_dependencies() -> tuple[Any, Any]:
    try:
        import essentia.standard as estd
    except Exception as exc:  # pragma: no cover - environment-specific
        raise RuntimeError(
            "Essentia is required for analysis. Install the dependencies from requirements.txt "
            "and ensure the TensorFlow-enabled Essentia package is available."
        ) from exc

    try:
        import laion_clap
    except Exception as exc:  # pragma: no cover - environment-specific
        raise RuntimeError(
            "laion-clap is required for analysis. Install the dependencies from requirements.txt."
        ) from exc

    return estd, laion_clap


def mean_over_time(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 0:
        return array.reshape(1)
    if array.ndim == 1:
        return array
    return array.mean(axis=0)


class AnalysisRuntime:
    def __init__(
        self,
        estd: Any,
        laion_clap: Any,
        models_dir: Path,
        device: str,
        clap_checkpoint: str,
        clap_amodel: str,
    ) -> None:
        self.estd = estd
        self.models_dir = models_dir
        self.device = device
        self.clap_amodel = clap_amodel
        self.signal_sample_rate = 44100
        self.effnet_sample_rate = 16000
        self.clap_sample_rate = 48000
        self.style_labels = load_default_style_labels()
        self.voice_classes = load_metadata_classes("voice_instrumental-discogs-effnet-1.json")
        self.danceability_classes = load_metadata_classes("danceability-discogs-effnet-1.json")

        self.rhythm = estd.RhythmExtractor2013(method="multifeature")
        self.key_extractors = {
            profile: estd.KeyExtractor(profileType=profile, sampleRate=self.signal_sample_rate)
            for profile in KEY_PROFILES
        }
        self.loudness = estd.LoudnessEBUR128(sampleRate=self.signal_sample_rate)

        discogs_effnet_path = resolve_model_file(models_dir, ["discogs-effnet-bs64-1.pb"])
        genre_model_path = resolve_model_file(
            models_dir,
            ["genre_discogs400-discogs-effnet-1.pb", "*genre*discogs400*effnet*.pb"],
        )
        voice_model_path = resolve_model_file(
            models_dir,
            ["voice_instrumental-discogs-effnet-1.pb", "*voice*instrumental*effnet*.pb"],
        )
        dance_model_path = resolve_model_file(
            models_dir,
            ["danceability-discogs-effnet-1.pb", "*danceability*effnet*.pb"],
        )
        clap_checkpoint_path = resolve_model_file(models_dir, [clap_checkpoint, f"*{Path(clap_checkpoint).name}"])
        effnet_input_name, effnet_output_name = load_metadata_io(
            "discogs-effnet-bs64-1.json",
            default_output="PartitionedCall:1",
            preferred_output_purpose="embeddings",
        )
        genre_input_name, genre_output_name = load_metadata_io(
            "genre_discogs400-discogs-effnet-1.json",
            default_output="PartitionedCall:0",
        )
        voice_input_name, voice_output_name = load_metadata_io(
            "voice_instrumental-discogs-effnet-1.json",
            default_input="model/Placeholder",
            default_output="model/Softmax",
        )
        dance_input_name, dance_output_name = load_metadata_io(
            "danceability-discogs-effnet-1.json",
            default_input="model/Placeholder",
            default_output="model/Softmax",
        )

        self.discogs_embedding_model = estd.TensorflowPredictEffnetDiscogs(
            graphFilename=str(discogs_effnet_path),
            output=effnet_output_name,
        )
        genre_kwargs = {"graphFilename": str(genre_model_path)}
        if genre_input_name is not None:
            genre_kwargs["input"] = genre_input_name
        if genre_output_name is not None:
            genre_kwargs["output"] = genre_output_name
        self.genre_model = estd.TensorflowPredict2D(**genre_kwargs)

        voice_kwargs = {"graphFilename": str(voice_model_path)}
        if voice_input_name is not None:
            voice_kwargs["input"] = voice_input_name
        if voice_output_name is not None:
            voice_kwargs["output"] = voice_output_name
        self.voice_model = estd.TensorflowPredict2D(**voice_kwargs)

        dance_kwargs = {"graphFilename": str(dance_model_path)}
        if dance_input_name is not None:
            dance_kwargs["input"] = dance_input_name
        if dance_output_name is not None:
            dance_kwargs["output"] = dance_output_name
        self.danceability_model = estd.TensorflowPredict2D(**dance_kwargs)

        self.clap_model = laion_clap.CLAP_Module(
            enable_fusion=False,
            device=device,
            amodel=clap_amodel,
            tmodel="roberta",
        )
        self.clap_model.load_ckpt(str(clap_checkpoint_path))

        self._mono_resamplers: dict[tuple[int, int], Any] = {}
        self._stereo_resamplers: dict[tuple[int, int], Any] = {}

    def _mono_resampler(self, input_rate: int, output_rate: int) -> Any:
        key = (input_rate, output_rate)
        if key not in self._mono_resamplers:
            self._mono_resamplers[key] = self.estd.Resample(
                inputSampleRate=input_rate,
                outputSampleRate=output_rate,
            )
        return self._mono_resamplers[key]

    def resample_mono(self, audio: np.ndarray, input_rate: int, output_rate: int) -> np.ndarray:
        if input_rate == output_rate:
            return np.asarray(audio, dtype=np.float32)
        return np.asarray(self._mono_resampler(input_rate, output_rate)(audio), dtype=np.float32)

    def resample_stereo(self, audio: np.ndarray, input_rate: int, output_rate: int) -> np.ndarray:
        if input_rate == output_rate:
            return np.asarray(audio, dtype=np.float32)
        left = self.resample_mono(audio[:, 0], input_rate, output_rate)
        right = self.resample_mono(audio[:, 1], input_rate, output_rate)
        return np.stack([left, right], axis=1).astype(np.float32)

    def load_audio(self, path: Path) -> tuple[np.ndarray, np.ndarray, int]:
        output = self.estd.AudioLoader(filename=str(path))()
        stereo_audio = np.asarray(output[0], dtype=np.float32)
        sample_rate = int(output[1]) if len(output) > 1 else self.signal_sample_rate

        if stereo_audio.ndim == 1:
            stereo_audio = np.stack([stereo_audio, stereo_audio], axis=1)
        elif stereo_audio.ndim == 2 and stereo_audio.shape[0] in (1, 2) and stereo_audio.shape[1] > stereo_audio.shape[0]:
            stereo_audio = stereo_audio.T

        mono_audio = stereo_audio.mean(axis=1).astype(np.float32)
        return stereo_audio, mono_audio, sample_rate

    def extract_loudness_lufs(self, stereo_signal: np.ndarray) -> float:
        loudness_output = self.loudness(stereo_signal)
        if isinstance(loudness_output, tuple):
            # Essentia returns (momentary, short_term, integrated, loudness_range).
            integrated = loudness_output[2]
        else:
            integrated = loudness_output
        return float(np.asarray(integrated).reshape(-1)[0])

    def extract_key_data(self, mono_signal: np.ndarray) -> dict[str, dict[str, float | str]]:
        key_data: dict[str, dict[str, float | str]] = {}
        for profile, extractor in self.key_extractors.items():
            key, scale, strength = extractor(mono_signal)
            key_data[profile] = {
                "key": str(key),
                "scale": str(scale),
                "strength": float(strength),
            }
        return key_data

    def classify_voice(self, embeddings: np.ndarray) -> tuple[float, float]:
        embeddings = self._validate_discogs_embeddings(embeddings)
        averaged = mean_over_time(self.voice_model(embeddings))
        probabilities = {label: float(value) for label, value in zip(self.voice_classes, averaged, strict=False)}
        if probabilities:
            instrumental_prob = float(probabilities.get("instrumental", 0.0))
            voice_prob = float(probabilities.get("voice", 0.0))
        elif len(averaged) == 1:
            voice_prob = float(averaged[0])
            instrumental_prob = float(1.0 - voice_prob)
        else:
            instrumental_prob = float(averaged[0])
            voice_prob = float(averaged[1])
        return instrumental_prob, voice_prob

    def classify_danceability(self, embeddings: np.ndarray) -> float:
        embeddings = self._validate_discogs_embeddings(embeddings)
        averaged = mean_over_time(self.danceability_model(embeddings))
        probabilities = {label: float(value) for label, value in zip(self.danceability_classes, averaged, strict=False)}
        if probabilities:
            return float(probabilities.get("danceable", max(probabilities.values())))
        if len(averaged) == 1:
            return float(averaged[0])
        return float(averaged[-1])

    def classify_styles(self, embeddings: np.ndarray) -> dict[str, float]:
        embeddings = self._validate_discogs_embeddings(embeddings)
        averaged = mean_over_time(self.genre_model(embeddings))
        labels = self.style_labels or [f"style_{index:03d}" for index in range(len(averaged))]
        return {label: float(value) for label, value in zip(labels, averaged, strict=False)}

    def _validate_discogs_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        array = np.asarray(embeddings, dtype=np.float32)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.ndim != 2:
            raise ValueError(f"Discogs embeddings must be a 2D array, got shape {array.shape}")
        if array.shape[1] != 1280:
            raise ValueError(f"Discogs embeddings must have 1280 features, got shape {array.shape}")
        return array

    def embed_discogs(self, mono_16k: np.ndarray) -> np.ndarray:
        embeddings = np.asarray(self.discogs_embedding_model(mono_16k), dtype=np.float32)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        return embeddings

    def embed_clap_audio(self, mono_48k: np.ndarray) -> np.ndarray:
        embedding = self.clap_model.get_audio_embedding_from_data(x=[mono_48k], use_tensor=False)
        embedding = np.asarray(embedding, dtype=np.float32)
        if embedding.ndim == 2:
            return embedding[0]
        return embedding.reshape(-1)

    def analyze_track(self, audio_path: Path, relative_path: Path) -> dict[str, Any]:
        stereo_audio, mono_audio, sample_rate = self.load_audio(audio_path)

        stereo_signal = self.resample_stereo(stereo_audio, sample_rate, self.signal_sample_rate)
        mono_signal = self.resample_mono(mono_audio, sample_rate, self.signal_sample_rate)
        mono_16k = self.resample_mono(mono_audio, sample_rate, self.effnet_sample_rate)
        mono_48k = self.resample_mono(mono_audio, sample_rate, self.clap_sample_rate)

        bpm, *_ = self.rhythm(mono_signal)
        key_data = self.extract_key_data(mono_signal)
        loudness_lufs = self.extract_loudness_lufs(stereo_signal)

        discogs_embeddings = self.embed_discogs(mono_16k)
        discogs_embedding_mean = mean_over_time(discogs_embeddings)
        style_activations = self.classify_styles(discogs_embeddings)
        instrumental_prob, voice_prob = self.classify_voice(discogs_embeddings)
        danceability_prob = self.classify_danceability(discogs_embeddings)
        clap_embedding_mean = self.embed_clap_audio(mono_48k)

        return {
            "track_id": relative_path.as_posix(),
            "relative_path": relative_path.as_posix(),
            "duration_seconds": float(len(mono_audio) / sample_rate),
            "source_sample_rate": sample_rate,
            "tempo_bpm": float(bpm),
            "loudness_lufs": float(loudness_lufs),
            "danceability_prob": float(danceability_prob),
            "voice_prob": float(voice_prob),
            "instrumental_prob": float(instrumental_prob),
            "voice_label": "voice" if voice_prob >= instrumental_prob else "instrumental",
            "key_profiles": key_data,
            "style_activations": style_activations,
            "discogs_embedding_mean": discogs_embedding_mean.tolist(),
            "clap_audio_embedding_mean": clap_embedding_mean.tolist(),
            "analyzed_at_utc": datetime.now(timezone.utc).isoformat(),
        }


def consolidate_outputs(output_dir: Path, audio_root: Path, processed_files: list[Path], args: argparse.Namespace) -> None:
    per_track_dir = output_dir / "per_track"
    records: list[dict[str, Any]] = []
    for json_file in sorted(per_track_dir.rglob("*.json")):
        with json_file.open("r", encoding="utf-8") as handle:
            records.append(json.load(handle))

    if not records:
        print("No successful analyses found, skipping consolidated artifact generation.")
        return

    records.sort(key=lambda record: record["track_id"])
    track_ids = [record["track_id"] for record in records]

    tracks_rows: list[dict[str, Any]] = []
    style_rows: list[dict[str, float]] = []
    discogs_embeddings: list[list[float]] = []
    clap_embeddings: list[list[float]] = []

    for record in records:
        key_profiles = record.get("key_profiles", {})
        tracks_rows.append(
            {
                "track_id": record["track_id"],
                "relative_path": record["relative_path"],
                "duration_seconds": record.get("duration_seconds"),
                "source_sample_rate": record.get("source_sample_rate"),
                "tempo_bpm": record.get("tempo_bpm"),
                "loudness_lufs": record.get("loudness_lufs"),
                "danceability_prob": record.get("danceability_prob"),
                "voice_prob": record.get("voice_prob"),
                "instrumental_prob": record.get("instrumental_prob"),
                "voice_label": record.get("voice_label"),
                "key_temperley": key_profiles.get("temperley", {}).get("key"),
                "scale_temperley": key_profiles.get("temperley", {}).get("scale"),
                "key_strength_temperley": key_profiles.get("temperley", {}).get("strength"),
                "key_krumhansl": key_profiles.get("krumhansl", {}).get("key"),
                "scale_krumhansl": key_profiles.get("krumhansl", {}).get("scale"),
                "key_strength_krumhansl": key_profiles.get("krumhansl", {}).get("strength"),
                "key_edma": key_profiles.get("edma", {}).get("key"),
                "scale_edma": key_profiles.get("edma", {}).get("scale"),
                "key_strength_edma": key_profiles.get("edma", {}).get("strength"),
                "analyzed_at_utc": record.get("analyzed_at_utc"),
            }
        )
        style_rows.append(record.get("style_activations", {}))
        discogs_embeddings.append(record.get("discogs_embedding_mean", []))
        clap_embeddings.append(record.get("clap_audio_embedding_mean", []))

    track_index = pd.DataFrame({"track_id": track_ids, "relative_path": track_ids}).set_index("track_id")
    tracks_df = pd.DataFrame(tracks_rows).set_index("track_id").reindex(track_ids)
    styles_df = pd.DataFrame(style_rows, index=track_ids).sort_index(axis=1).reindex(track_ids)
    track_index["relative_path"] = tracks_df["relative_path"]

    discogs_matrix = np.asarray(discogs_embeddings, dtype=np.float32)
    clap_matrix = np.asarray(clap_embeddings, dtype=np.float32)

    output_dir.mkdir(parents=True, exist_ok=True)
    track_index.to_parquet(output_dir / "track_index.parquet")
    tracks_df.to_parquet(output_dir / "tracks.parquet")
    styles_df.to_parquet(output_dir / "style_activations.parquet")
    np.save(output_dir / "discogs_embeddings.npy", discogs_matrix)
    np.save(output_dir / "clap_audio_embeddings.npy", clap_matrix)

    manifest = {
        "analysis_version": 1,
        "audio_root": project_relative_string(audio_root),
        "models_dir": project_relative_string(models_dir),
        "device": args.device,
        "clap_amodel": args.clap_amodel,
        "track_count": len(track_index),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "style_label_count": len(styles_df.columns),
        "artifacts": {
            "track_index": "track_index.parquet",
            "tracks": "tracks.parquet",
            "styles": "style_activations.parquet",
            "discogs_embeddings": "discogs_embeddings.npy",
            "clap_audio_embeddings": "clap_audio_embeddings.npy",
        },
    }
    save_json(output_dir / "manifest.json", manifest)

    print(f"Wrote consolidated artifacts for {len(track_index)} tracks to {output_dir}")


def main() -> None:
    args = parse_args()
    os.environ.setdefault("HF_HOME", str((PROJECT_ROOT / ".cache" / "huggingface").resolve()))
    os.environ.setdefault("MPLCONFIGDIR", str((PROJECT_ROOT / ".cache" / "matplotlib").resolve()))
    audio_root = (PROJECT_ROOT / args.audio_root).resolve()
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    models_dir = (PROJECT_ROOT / args.models_dir).resolve()

    if not audio_root.exists():
        raise FileNotFoundError(f"Audio root does not exist: {audio_root}")
    if not models_dir.exists():
        raise FileNotFoundError(f"Models directory does not exist: {models_dir}")

    mp3_files = discover_mp3s(audio_root)
    if args.max_files is not None:
        mp3_files = mp3_files[: args.max_files]

    print(f"Discovered {len(mp3_files)} MP3 files under {audio_root}")
    per_track_dir = output_dir / "per_track"
    error_log = output_dir / "errors.jsonl"

    estd, laion_clap = ensure_runtime_dependencies()
    runtime = AnalysisRuntime(
        estd,
        laion_clap,
        models_dir=models_dir,
        device=args.device,
        clap_checkpoint=args.clap_checkpoint,
        clap_amodel=args.clap_amodel,
    )

    processed_files: list[Path] = []
    for audio_path in tqdm(mp3_files, desc="Analyzing tracks"):
        relative_path = audio_path.relative_to(audio_root)
        cache_file = json_cache_path(per_track_dir, relative_path)

        if cache_file.exists() and not args.force:
            processed_files.append(audio_path)
            continue

        try:
            record = runtime.analyze_track(audio_path, relative_path)
            save_json(cache_file, record)
            processed_files.append(audio_path)
        except KeyboardInterrupt:  # pragma: no cover - interactive safety
            raise
        except Exception as exc:  # pragma: no cover - runtime path depends on external models
            append_jsonl(
                error_log,
                {
                    "track_id": relative_path.as_posix(),
                    "audio_path": project_relative_string(audio_path),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "traceback": traceback.format_exc(),
                    "logged_at_utc": datetime.now(timezone.utc).isoformat(),
                },
            )

    consolidate_outputs(output_dir, audio_root, processed_files, args)


if __name__ == "__main__":
    main()
