# Music Collection Overview

## Summary
- Tracks available in the loaded analysis bundle: 2100
- Analysis source: `1`
- Audio root: `MusAV/audio_chunks`

## Style Diversity
- Parent genre distribution is computed from the top-1 Discogs style prediction per track.
- Top parent genres:
  - Rock: 549
  - Electronic: 463
  - Hip Hop: 271
  - Folk, World, & Country: 204
  - Pop: 155
  - Latin: 117
  - Classical: 86
  - Funk / Soul: 64
  - Jazz: 56
  - Reggae: 40
- Top detailed styles:
  - Rock---Alternative Rock: 82
  - Folk, World, & Country---Folk: 77
  - Rock---Punk: 76
  - Electronic---House: 61
  - Electronic---Ambient: 57
  - Rock---Pop Rock: 56
  - Hip Hop---Trap: 54
  - Hip Hop---Cloud Rap: 52
  - Pop---Ballad: 47
  - Rock---Indie Rock: 47

## Descriptor Observations
- `tempo_bpm`: min=60.19, median=119.84, max=184.57. Tempo distribution plot available in `reports/figures/tempo_distribution.png`.
- `danceability_prob`: min=0.00, median=0.68, max=1.00. Danceability distribution plot available in `reports/figures/danceability_distribution.png`.
- `loudness_lufs`: min=-19.97, median=-10.19, max=-4.98. Loudness distribution plot available in `reports/figures/loudness_distribution.png`.
- Voice vs instrumental share: voice=71.4%, instrumental=28.6%.

## Key Profile Comparison
- Tracks where all three profiles agree: 49.1%.
- Recommended default profile for the UI: `krumhansl`.
- Mean pairwise agreement for `temperley`: 54.9%.
- Mean pairwise agreement for `krumhansl`: 69.9%.
- Mean pairwise agreement for `edma`: 66.9%.
