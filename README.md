# MicroLM-Audio

MicroLM-Audio is a two-stage project:

1. Reproduce the original MicroLM language model workflow, including pretrain, SFT, and LoRA / Qwen smoke runs.
2. Adapt the lightweight Transformer idea to GTZAN music genre classification with log-mel frame tokens.

See [Target.md](Target.md) for the step-by-step implementation tutorial.

## Repository Layout

```text
external/MicroLM/              # Upstream MicroLM clone placeholder
configs/audio_gtzan_small.json # Main audio classification config
microlm_audio/                 # MicroLM-Audio Python package
scripts/                       # CLI entry points
data/                          # Local datasets and generated splits
outputs/                       # Local experiment outputs
```

Large datasets, checkpoints, generated features, and experiment outputs are intentionally ignored by Git.
