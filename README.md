# Audio Content Based Playlists

This repository contains an end-to-end assignment scaffold for analyzing a local music collection, generating overview reports, and building three Streamlit playlist interfaces:

- descriptor-based playlists
- similarity search by example track
- freeform text search with CLAP

## Repository Layout

- `scripts/analyze_collection.py`: extract Essentia and CLAP features for every MP3 in a collection
- `scripts/build_overview_report.py`: generate collection statistics, figures, and a Markdown overview
- `apps/descriptors_app.py`: playlist generation by descriptors and style activations
- `apps/similarity_app.py`: playlist generation by track similarity
- `apps/text_query_app.py`: playlist generation by freeform text queries
- `playlist_tools.py`: shared loading, scoring, and playlist-export helpers
- `reports/final_report.md`: report write-up and implementation notes

## Setup

Create and activate a Python virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Model Files

Place the required model checkpoints in a local `models/` directory, or point the scripts to another folder with `--models-dir`.

Expected files:

- `discogs-effnet-bs64-1.pb`
- `genre_discogs400-discogs-effnet-1.pb`
- `voice_instrumental-discogs-effnet-1.pb`
- `danceability-discogs-effnet-1.pb`
- `music_speech_epoch_15_esc_89.25.pt`

The validated CLAP configuration for `music_speech_epoch_15_esc_89.25.pt` is `HTSAT-base`.

## Run the Analysis Pipeline

Analyze the collection rooted at `MusAV/audio_chunks` and write outputs into `analysis/`:

```bash
python scripts/analyze_collection.py \
  --audio-root MusAV/audio_chunks \
  --output-dir analysis \
  --models-dir models \
  --device cpu \
  --clap-amodel HTSAT-base
```

Useful optional flags:

- `--max-files 25`: smoke-test on a subset
- `--force`: recompute already-cached tracks
- `--clap-amodel HTSAT-base`: explicit CLAP audio backbone for the validated music-speech checkpoint

Generated artifacts:

- `analysis/tracks.parquet`
- `analysis/style_activations.parquet`
- `analysis/track_index.parquet`
- `analysis/discogs_embeddings.npy`
- `analysis/clap_audio_embeddings.npy`
- `analysis/per_track/*.json`
- `analysis/errors.jsonl`
- `analysis/manifest.json`

## Build the Overview Report

```bash
python scripts/build_overview_report.py --analysis-dir analysis --output-dir reports
```

This produces:

- `reports/collection_overview.md`
- `reports/style_distribution.tsv`
- `reports/figures/*.png`

## Run the Streamlit Apps

Default descriptor app:

```bash
./run.sh
```

Similarity app:

```bash
./run.sh similarity
```

Text-query app:

```bash
./run.sh text
```

I possible to run a specific file directly:

```bash
streamlit run apps/descriptors_app.py
```

## Notes

- The repo still contains the original legacy Discogs style artifact in `data/`. The new utility loader can fall back to that file for the descriptor demo when the full `analysis/` outputs have not been generated yet.
- Full quantitative report results require running the analysis pipeline first.
