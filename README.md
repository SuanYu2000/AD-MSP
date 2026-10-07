# Multi-PSP

Few-shot learning with Visformer pretrain + Semantic Prompt / Multi-Prompt + optional regularizers.

## Main entrypoints

| Script | Role |
|--------|------|
| `train_vit.py` | Pre-train Visformer on base classes |
| `train_FSL_reg.py` | Episode FSL fine-tune with optional regularizers |
| `train_semantic4_reg.py` | Semantic4 multi-prompt training / test (fusion sweep via `--test`) |

## Requirements

- Python >= 3.8
- PyTorch >= 1.7.1
- torchvision, einops, tensorboard
- Local `clip/` package is included (OpenAI CLIP)

## Dataset setup

Download datasets and extract into `./dataset` (links in `dataset/数据集地址.txt`). Paths are defined in `data/dataset.py`.

## Example commands

```bash
# 1) Pre-train
python train_vit.py --gpu 0 --dataset miniImageNet --exp pre-train --rand_aug --repeat_aug

# 2) FSL + regularizers
python train_FSL_reg.py --gpu 0 --dataset miniImageNet \
  --init checkpoint/miniImageNet/visformer-t/pre-train/checkpoint_epoch_800.pth \
  --supcon_weight 0.1 --text_dropout_p 0.3 --clip_kd_weight 0.5

# 3) Semantic4 multi-prompt + regularizers
python train_semantic4_reg.py --gpu 0 --dataset miniImageNet \
  --init checkpoint/miniImageNet/visformer-t/pre-train-1shot/checkpoint_epoch_800.pth \
  --supcon_weight 0.3

# 4) Semantic4 test / fusion sweep
python train_semantic4_reg.py --gpu 0 --dataset miniImageNet --test \
  --resume checkpoint/.../checkpoint_epoch_best.pth --shot 1
```

Regularizer flags (`--supcon_weight`, `--text_dropout_p`, `--clip_kd_weight`) default to off when set to `0`.
