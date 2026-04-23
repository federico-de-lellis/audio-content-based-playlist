from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from playlist_tools import AnalysisBundle, combine_key_and_scale, load_analysis_bundle, parent_genre, top_style_per_track


KEY_PROFILES = ("temperley", "krumhansl", "edma")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate statistical overview outputs for the music collection.")
    parser.add_argument("--analysis-dir", default="analysis", help="Folder containing consolidated analysis artifacts.")
    parser.add_argument("--output-dir", default="reports", help="Folder for Markdown, TSV, and figures.")
    return parser.parse_args()


def ensure_plotting_dependencies():
    try:
        cache_dir = PROJECT_ROOT / ".cache" / "matplotlib"
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except Exception as exc:  # pragma: no cover - environment-specific
        raise RuntimeError(
            "matplotlib and seaborn are required to build the overview report. Install requirements.txt first."
        ) from exc
    return plt, sns


def save_numeric_distribution(plt, sns, series: pd.Series, title: str, xlabel: str, output_path: Path) -> None:
    clean = series.dropna()
    if clean.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.histplot(clean, kde=True, bins=30, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Track count")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_category_distribution(plt, sns, series: pd.Series, title: str, output_path: Path, top_n: int | None = None) -> pd.Series:
    clean = series.dropna()
    if clean.empty:
        return pd.Series(dtype="int64")

    counts = clean.value_counts()
    if top_n is not None:
        counts = counts.head(top_n)

    fig_height = max(4, min(12, int(len(counts) * 0.35) + 1))
    fig, ax = plt.subplots(figsize=(10, fig_height))
    sns.barplot(x=counts.values, y=counts.index, orient="h", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Track count")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return counts


def build_key_distribution_figure(plt, sns, tracks: pd.DataFrame, output_path: Path) -> dict[str, pd.Series]:
    distributions: dict[str, pd.Series] = {}
    available_profiles = [
        profile
        for profile in KEY_PROFILES
        if f"key_{profile}" in tracks.columns and f"scale_{profile}" in tracks.columns
    ]
    if not available_profiles:
        return distributions

    fig, axes = plt.subplots(len(available_profiles), 1, figsize=(10, 4 * len(available_profiles)))
    if len(available_profiles) == 1:
        axes = [axes]

    for axis, profile in zip(axes, available_profiles, strict=False):
        labels = tracks.apply(
            lambda row: combine_key_and_scale(row.get(f"key_{profile}"), row.get(f"scale_{profile}")),
            axis=1,
        ).dropna()
        counts = labels.value_counts().head(15)
        distributions[profile] = counts
        sns.barplot(x=counts.values, y=counts.index, orient="h", ax=axis)
        axis.set_title(f"Key and scale distribution ({profile})")
        axis.set_xlabel("Track count")
        axis.set_ylabel("")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return distributions


def compute_key_agreement(tracks: pd.DataFrame) -> tuple[float | None, str | None, dict[str, float]]:
    profile_labels: dict[str, pd.Series] = {}
    for profile in KEY_PROFILES:
        key_column = f"key_{profile}"
        scale_column = f"scale_{profile}"
        if key_column not in tracks.columns or scale_column not in tracks.columns:
            continue
        labels = tracks.apply(
            lambda row: combine_key_and_scale(row.get(key_column), row.get(scale_column)),
            axis=1,
        )
        profile_labels[profile] = labels

    if len(profile_labels) < 3:
        return None, None, {}

    label_frame = pd.DataFrame(profile_labels).dropna()
    if label_frame.empty:
        return None, None, {}

    full_agreement = float(
        ((label_frame["temperley"] == label_frame["krumhansl"]) & (label_frame["temperley"] == label_frame["edma"])).mean()
    )

    pairwise_agreement: dict[str, float] = {}
    for profile in KEY_PROFILES:
        others = [other for other in KEY_PROFILES if other != profile]
        scores = []
        for other in others:
            scores.append(float((label_frame[profile] == label_frame[other]).mean()))
        pairwise_agreement[profile] = sum(scores) / len(scores)

    selected_profile = max(pairwise_agreement, key=pairwise_agreement.get)
    return full_agreement, selected_profile, pairwise_agreement


def write_style_distribution_tsv(style_counts: pd.Series, output_path: Path) -> None:
    table = style_counts.rename_axis("style").reset_index(name="count")
    table.to_csv(output_path, sep="\t", index=False)


def build_report_markdown(
    bundle: AnalysisBundle,
    parent_genre_counts: pd.Series,
    style_counts: pd.Series,
    full_key_agreement: float | None,
    selected_profile: str | None,
    pairwise_agreement: dict[str, float],
) -> str:
    lines = [
        "# Music Collection Overview",
        "",
        "## Summary",
        f"- Tracks available in the loaded analysis bundle: {len(bundle.track_index)}",
        f"- Analysis source: `{bundle.manifest.get('analysis_version', 'unknown')}`",
        f"- Audio root: `{bundle.manifest.get('audio_root', 'unknown')}`",
        "",
        "## Style Diversity",
        "- Parent genre distribution is computed from the top-1 Discogs style prediction per track.",
    ]

    if not parent_genre_counts.empty:
        lines.append("- Top parent genres:")
        for genre, count in parent_genre_counts.head(10).items():
            lines.append(f"  - {genre}: {count}")
    else:
        lines.append("- Parent genre distribution is unavailable because style activations were not found.")

    if not style_counts.empty:
        top_styles = style_counts.head(10)
        lines.append("- Top detailed styles:")
        for style, count in top_styles.items():
            lines.append(f"  - {style}: {count}")

    lines.extend(["", "## Descriptor Observations"])
    descriptor_columns = {
        "tempo_bpm": "Tempo distribution plot available in `reports/figures/tempo_distribution.png`.",
        "danceability_prob": "Danceability distribution plot available in `reports/figures/danceability_distribution.png`.",
        "loudness_lufs": "Loudness distribution plot available in `reports/figures/loudness_distribution.png`.",
    }

    has_any_descriptor = False
    for column, message in descriptor_columns.items():
        if bundle.has_column(column):
            has_any_descriptor = True
            series = bundle.tracks[column].dropna()
            lines.append(
                f"- `{column}`: min={series.min():.2f}, median={series.median():.2f}, max={series.max():.2f}. {message}"
            )
    if not has_any_descriptor:
        lines.append("- Scalar descriptors are not available in the current bundle. Run the full analysis pipeline first.")

    if bundle.has_column("voice_label"):
        voice_share = bundle.tracks["voice_label"].value_counts(normalize=True)
        lines.append(
            f"- Voice vs instrumental share: voice={voice_share.get('voice', 0.0):.1%}, "
            f"instrumental={voice_share.get('instrumental', 0.0):.1%}."
        )

    lines.extend(["", "## Key Profile Comparison"])
    if full_key_agreement is None:
        lines.append("- Key-profile comparison is unavailable because the required key columns are missing.")
    else:
        lines.append(f"- Tracks where all three profiles agree: {full_key_agreement:.1%}.")
        lines.append(f"- Recommended default profile for the UI: `{selected_profile}`.")
        for profile, agreement in pairwise_agreement.items():
            lines.append(f"- Mean pairwise agreement for `{profile}`: {agreement:.1%}.")

    if bundle.manifest.get("uses_legacy_bundle"):
        lines.extend(
            [
                "",
                "## Legacy Bundle Note",
                "- The current report was built from the repository's legacy Discogs style artifact.",
                "- Re-run `scripts/analyze_collection.py` to populate tempo, loudness, key, danceability, voice/instrumental, and embedding outputs.",
            ]
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_analysis_bundle(
        analysis_dir=(PROJECT_ROOT / args.analysis_dir).resolve(),
        load_discogs_embeddings=False,
        load_clap_embeddings=False,
        allow_legacy=True,
    )
    plt, sns = ensure_plotting_dependencies()

    style_counts = pd.Series(dtype="int64")
    parent_genre_counts = pd.Series(dtype="int64")
    if not bundle.styles.empty:
        top_styles = top_style_per_track(bundle.styles)
        parent_genre_counts = top_styles.map(parent_genre).value_counts()
        style_counts = top_styles.value_counts()

        save_category_distribution(
            plt,
            sns,
            top_styles.map(parent_genre),
            "Parent genre distribution",
            figures_dir / "parent_genre_distribution.png",
            top_n=15,
        )
        write_style_distribution_tsv(style_counts, output_dir / "style_distribution.tsv")

    if bundle.has_column("tempo_bpm"):
        save_numeric_distribution(
            plt,
            sns,
            bundle.tracks["tempo_bpm"],
            "Tempo distribution",
            "BPM",
            figures_dir / "tempo_distribution.png",
        )
    if bundle.has_column("danceability_prob"):
        save_numeric_distribution(
            plt,
            sns,
            bundle.tracks["danceability_prob"],
            "Danceability distribution",
            "Danceability probability",
            figures_dir / "danceability_distribution.png",
        )
    if bundle.has_column("loudness_lufs"):
        save_numeric_distribution(
            plt,
            sns,
            bundle.tracks["loudness_lufs"],
            "Integrated loudness distribution",
            "LUFS",
            figures_dir / "loudness_distribution.png",
        )
    if bundle.has_column("voice_label"):
        save_category_distribution(
            plt,
            sns,
            bundle.tracks["voice_label"],
            "Voice versus instrumental share",
            figures_dir / "voice_share.png",
        )

    build_key_distribution_figure(plt, sns, bundle.tracks, figures_dir / "key_scale_distribution.png")
    full_key_agreement, selected_profile, pairwise_agreement = compute_key_agreement(bundle.tracks)

    report_markdown = build_report_markdown(
        bundle=bundle,
        parent_genre_counts=parent_genre_counts,
        style_counts=style_counts,
        full_key_agreement=full_key_agreement,
        selected_profile=selected_profile,
        pairwise_agreement=pairwise_agreement,
    )
    (output_dir / "collection_overview.md").write_text(report_markdown, encoding="utf-8")
    print(f"Wrote overview report assets to {output_dir}")


if __name__ == "__main__":
    main()
