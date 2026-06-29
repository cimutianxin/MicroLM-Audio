# MicroLM Reproduction Summary

## Environment

| Item | Value |
| --- | --- |
| Conda env | `microlm` |
| Python | `3.11.15` |
| PyTorch | `2.5.1+cu121` |
| TorchVision | `0.20.1+cu121` |
| TorchAudio | `2.5.1+cu121` |
| CUDA runtime | `12.1` |
| CUDA available | `True` |
| GPU | `NVIDIA GeForce RTX 4060 Ti` |
| GPU memory | `16380 MiB` |
| NVIDIA driver | `580.142` |

## Source

| Item | Value |
| --- | --- |
| Upstream repository | `https://github.com/jiaran-king/MicroLM.git` |
| Upstream commit | `782ae02` |
| Local path | `external/MicroLM` |
| Environment clone command | `conda create --name microlm --clone microlm-audio -y` |
| Install command | `python -m pip install -e "external/MicroLM[all]"` |

## Checks

| Check | Result | Notes |
| --- | --- | --- |
| CUDA import check | Passed | `torch.cuda.is_available() == True` |
| Transformer import check | Passed | `TransformerLM` instantiated successfully |
| Test suite | Passed | `32 passed, 1 warning` |
| Pretrain smoke | Passed | `configs/pretrain_smoke.json` |

## Pretrain Smoke Result

```text
训练集大小： 4096 tokens
验证集大小 1024 tokens
Model params: 459,392
Model Config: Norm=pre, UseNorm=True, FFN=swiglu, RoPE=True
Iter 0: train_loss 6.4086, val_loss 6.5344, lr 0.00e+00
Iter 19: train_loss 5.3352, val_loss 6.4113, lr 7.32e-05
```

## Outputs

| File | Purpose |
| --- | --- |
| `outputs/microlm_reproduce/pytest.log` | Full pytest output |
| `outputs/microlm_reproduce/pretrain_smoke.log` | Pretrain smoke log |
| `outputs/microlm_reproduce/checkpoints/pretrain_smoke_ckpt_final.pt` | Final smoke checkpoint |
| `outputs/microlm_reproduce/checkpoints/pretrain_smoke_model_config.json` | Model config from smoke run |
| `outputs/microlm_reproduce/checkpoints/pretrain_smoke_resolved_train_config.json` | Resolved train config from smoke run |

## Notes

No OOM occurred. The upstream `pyproject.toml` requires Python `>=3.11`. The existing `microlm-audio` environment was upgraded to Python `3.11.15`, then cloned to a dedicated `microlm` environment for the full upstream MicroLM reproduction workflow. Future upstream reproduction steps should use `conda activate microlm`; the later audio classification work can return to `conda activate microlm-audio`.
