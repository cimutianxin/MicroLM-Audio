# MicroLM-Audio 项目实现教程

## 0. 项目目标

本项目分为两个阶段：

```text
阶段 1：复现原 MicroLM 语言模型完整流程
阶段 2：在复现基础上改造为 MicroLM-Audio 音乐风格分类器
```

最终项目名称：

```text
MicroLM-Audio: A Lightweight Transformer for Music Genre Classification
```

阶段 1 保留原 MicroLM 的核心训练与微调流程：

```text
原 MicroLM 仓库
-> 环境配置
-> 语言模型预训练 smoke run
-> SFT smoke run
-> LoRA / Qwen 相关 smoke run
-> 推理与评估检查
-> 记录复现结果
```

阶段 2 的核心任务是把轻量级 Transformer 从文本序列建模迁移到音频分类：

```text
audio waveform
-> log-mel spectrogram
-> frame sequence
-> Transformer encoder
-> pooling
-> classifier head
-> genre label
```

项目最终保留两条必要主线：

```text
主线 1：原 MicroLM 语言模型复现
主线 2：MicroLM-Audio GTZAN 音乐分类改造
```

音频分类主线为：

```text
GTZAN 数据集
-> 3 秒音频切片
-> log-mel 特征
-> MicroLM-Audio 分类模型
-> 训练、验证、测试
-> 输出 accuracy、macro-F1、confusion matrix
```

---

## 1. 推荐硬件与模型配置

当前目标硬件按 16GB GPU 设计。

推荐第一版配置：

```text
sample_rate = 22050
clip_length = 3
n_fft = 1024
hop_length = 512
n_mels = 128
d_model = 256
num_layers = 4
num_heads = 4
d_ff = 1024
batch_size = 32
epochs = 50
```

如果出现显存不足，按这个顺序降低：

```text
batch_size: 32 -> 16 -> 8
d_model: 256 -> 192 -> 128
num_layers: 4 -> 3 -> 2
clip_length: 3 -> 2
```

每个参数的作用：

| 参数 | 作用 |
| --- | --- |
| `sample_rate` | 统一音频采样率，避免不同文件长度和频率尺度不一致 |
| `clip_length` | 每个训练样本的音频秒数，越长信息越多但显存越高 |
| `n_fft` | STFT 窗口大小，影响频率分辨率 |
| `hop_length` | 相邻帧之间的步长，影响时间分辨率和序列长度 |
| `n_mels` | mel 频带数量，也是每个音频帧的输入维度 |
| `d_model` | Transformer 隐藏维度 |
| `num_layers` | Transformer 层数 |
| `num_heads` | 多头注意力头数 |
| `d_ff` | 前馈网络隐藏维度 |
| `batch_size` | 每次训练输入的样本数量 |

---

## 2. 项目目录结构

建议从这个结构开始搭建：

```text
MicroLM-Audio/
├── README.md
├── Target.md
├── requirements.txt
│
├── external/
│   └── MicroLM/
│
├── configs/
│   └── audio_gtzan_small.json
│
├── microlm_audio/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataset.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── classifier.py
│   ├── training/
│   │   ├── __init__.py
│   │   └── trainer.py
│   └── utils/
│       ├── __init__.py
│       ├── audio.py
│       └── metrics.py
│
├── scripts/
│   ├── prepare_gtzan.py
│   ├── train_gtzan_classifier.py
│   └── eval_gtzan_classifier.py
│
├── data/
│   ├── raw/
│   │   └── gtzan/
│   ├── processed/
│   │   └── gtzan_mel/
│   └── splits/
│       ├── train.csv
│       ├── valid.csv
│       └── test.csv
│
└── outputs/
    ├── microlm_reproduce/
    └── gtzan_classifier/
```

各目录作用：

| 目录 | 作用 |
| --- | --- |
| `external/MicroLM/` | 原 MicroLM 仓库，用于复现语言模型、SFT、LoRA/Qwen 流程 |
| `configs/` | 保存实验配置，保证训练可复现 |
| `microlm_audio/data/` | 数据集读取、clip 加载、mel 特征加载 |
| `microlm_audio/models/` | MicroLM-Audio 模型定义 |
| `microlm_audio/training/` | 训练循环、验证循环、checkpoint 保存 |
| `microlm_audio/utils/` | 音频处理、指标计算等工具函数 |
| `scripts/` | 用户实际运行的入口脚本 |
| `data/raw/` | 原始 GTZAN 音频 |
| `data/processed/` | 预处理后的 log-mel 特征 |
| `data/splits/` | train / valid / test 划分 |
| `outputs/` | 训练日志、模型权重、评估结果 |

---

## 3. Step 1：创建环境

### 要做什么

创建 Python 环境并安装原 MicroLM 和音频改造都需要的依赖。

```bash
conda create -n microlm-audio python=3.10 -y
conda activate microlm-audio
```

安装 PyTorch：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

安装项目依赖：

```bash
pip install numpy pandas tqdm scikit-learn matplotlib librosa soundfile
```

进入原 MicroLM 仓库后，如果它提供 `requirements.txt`，继续安装：

```bash
pip install -r requirements.txt
```

建议写入 `requirements.txt`：

```text
torch
torchaudio
numpy
pandas
tqdm
scikit-learn
matplotlib
librosa
soundfile
```

### 为什么做

音频项目需要同时处理：

```text
1. 音频读取
2. 特征提取
3. 深度学习训练
4. 指标计算
5. 结果可视化
```

这些依赖分别覆盖上述需求。

### 验证命令

```bash
python -c "import torch; print(torch.cuda.is_available())"
python -c "import librosa, sklearn, pandas; print('ok')"
```

---

## 4. Step 2：复现原 MicroLM 语言模型流程

### 要做什么

把原 MicroLM 仓库放到：

```text
external/MicroLM/
```

推荐命令：

```bash
mkdir -p external
git clone https://github.com/jiaran-king/MicroLM.git external/MicroLM
cd external/MicroLM
```

先做基础导入和测试：

```bash
python -c "import torch; print(torch.cuda.is_available())"
pytest tests/
```

如果原仓库没有完整测试，至少做模型导入检查。具体模块路径以原仓库实际代码为准：

```bash
python -c "from microlm.model.transformer import TransformerLM; print('ok')"
```

然后运行原项目提供的最小训练或 smoke 配置：

```bash
python scripts/train_pretrain.py --config configs/pretrain_smoke.json
```

### 为什么做

这一阶段的作用不是追求语言模型效果，而是确认：

```text
1. 原项目依赖可安装
2. 原 Transformer 主体可实例化
3. 原训练脚本可启动
4. checkpoint、日志、配置读取流程可用
```

这些结果会给第二阶段音频改造提供可信基础。

### 输出文件

建议保存到：

```text
outputs/microlm_reproduce/
├── pretrain_smoke.log
├── checkpoints/
└── reproduce_summary.md
```

`reproduce_summary.md` 记录：

```text
1. GPU 型号与显存
2. Python / PyTorch / CUDA 版本
3. pretrain smoke 是否成功
4. 最终 loss
5. 是否出现 OOM
6. 修改过哪些配置
```

---

## 5. Step 3：复现 SFT / LoRA / Qwen 相关流程

### 要做什么

在原 MicroLM 仓库中继续运行 SFT 与 LoRA / Qwen 相关 smoke 流程：

```bash
cd external/MicroLM
python scripts/train_sft.py --config configs/sft_smoke.json
python scripts/train_qwen_lora.py --config configs/qwen_lora_structured_smoke.json
```

如果脚本或配置名和原仓库不一致，以实际文件为准，原则是保留三类检查：

```text
1. SFT 训练入口能启动
2. LoRA 参数高效微调入口能启动
3. Qwen 相关结构化微调入口能启动
```

### 为什么做

SFT / LoRA / Qwen 流程属于原 MicroLM 项目的重要能力。保留它们可以证明复现不是只跑了一个预训练脚本，而是覆盖了原项目的主要训练范式。

### 输出文件

建议保存到：

```text
outputs/microlm_reproduce/
├── sft_smoke.log
├── qwen_lora_smoke.log
└── reproduce_summary.md
```

`reproduce_summary.md` 中继续补充：

```text
1. SFT smoke 是否成功
2. LoRA / Qwen smoke 是否成功
3. 每个流程的最终 loss 或关键日志
4. 遇到的依赖、显存、配置问题
5. 对原仓库做过哪些最小修改
```

---

## 6. Step 4：准备 GTZAN 数据集

### 要做什么

把 GTZAN 数据集放到：

```text
data/raw/gtzan/
```

推荐结构：

```text
data/raw/gtzan/
├── blues/
├── classical/
├── country/
├── disco/
├── hiphop/
├── jazz/
├── metal/
├── pop/
├── reggae/
└── rock/
```

GTZAN 一共有 10 个类别：

```text
blues
classical
country
disco
hiphop
jazz
metal
pop
reggae
rock
```

### 为什么做

分类任务需要从目录名获得标签。例如：

```text
data/raw/gtzan/blues/blues.00001.wav -> label = blues
```

保持目录结构规整，可以让预处理脚本自动扫描所有音频并生成标签。

---

## 7. Step 5：划分 train / valid / test

### 要做什么

写 `scripts/prepare_gtzan.py`，完成三件事：

```text
1. 扫描 data/raw/gtzan/ 下所有音频文件
2. 按歌曲级别划分 train / valid / test
3. 把每首歌切成多个 3 秒 clip
```

推荐划分比例：

```text
train: 70%
valid: 15%
test: 15%
```

关键原则：

```text
同一首歌切出来的所有 clip 必须放在同一个 split 中。
```

错误做法：

```text
song_001_clip_0 -> train
song_001_clip_1 -> test
```

正确做法：

```text
song_001_clip_0 -> train
song_001_clip_1 -> train
song_002_clip_0 -> valid
song_002_clip_1 -> valid
```

### 为什么做

如果同一首歌的不同片段同时出现在训练集和测试集，模型会“见过”测试歌曲的一部分，测试准确率会虚高。这叫数据泄漏。

### 输出文件

```text
data/splits/train.csv
data/splits/valid.csv
data/splits/test.csv
```

每个 CSV 建议包含：

```text
song_id,audio_path,label,clip_index,start_time,end_time
```

字段含义：

| 字段 | 作用 |
| --- | --- |
| `song_id` | 原始歌曲 ID，用于 song-level 聚合 |
| `audio_path` | 原始音频路径 |
| `label` | 类别名称 |
| `clip_index` | 第几个 3 秒片段 |
| `start_time` | clip 开始时间 |
| `end_time` | clip 结束时间 |

---

## 8. Step 6：提取 log-mel 特征

### 要做什么

在 `prepare_gtzan.py` 中继续实现 log-mel 提取，或者单独封装到：

```text
microlm_audio/utils/audio.py
```

推荐参数：

```text
sample_rate = 22050
clip_length = 3
n_fft = 1024
hop_length = 512
n_mels = 128
fmin = 0
fmax = 8000
```

对每个 clip 生成一个 `.npy` 文件：

```text
data/processed/gtzan_mel/
├── blues.00001_clip000.npy
├── blues.00001_clip001.npy
└── ...
```

3 秒音频大约得到：

```text
mel shape = [time_frames, n_mels]
time_frames ≈ 129
n_mels = 128
```

模型输入形状：

```text
[batch_size, time_frames, n_mels]
```

### 为什么做

原始 waveform 是一维振幅序列，不适合直接输入当前轻量 Transformer。log-mel spectrogram 把音频转换成“时间帧序列”，每一帧是一个 128 维频率特征：

```text
文本任务: token_1, token_2, token_3, ...
音频任务: mel_frame_1, mel_frame_2, mel_frame_3, ...
```

这样就可以把音频看成一种连续 token 序列。

### 输出文件

除了 `.npy` 特征文件，还需要一个总索引：

```text
data/processed/gtzan_mel/metadata.csv
```

建议字段：

```text
split,song_id,feature_path,label,label_id,clip_index
```

---

## 9. Step 7：实现 Dataset

### 要做什么

创建：

```text
microlm_audio/data/dataset.py
```

实现 `GTZANMelDataset`：

```python
class GTZANMelDataset(torch.utils.data.Dataset):
    def __init__(self, metadata_csv, split):
        pass

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass
```

每次返回：

```python
{
    "mel": FloatTensor[time_frames, n_mels],
    "label": LongTensor[],
    "song_id": str,
    "clip_index": int,
}
```

### 为什么做

`Dataset` 是 PyTorch 训练流程的数据入口。它把磁盘上的 `.npy` 特征和 CSV 标签转换成模型能直接使用的 tensor。

### 注意事项

第一版可以让所有 clip 固定为 3 秒，因此 time_frames 基本一致。如果后续出现长度不一致，再加入 padding 和 `attention_mask`。

---

## 10. Step 8：实现 MicroLM-Audio 模型

### 要做什么

创建：

```text
microlm_audio/models/classifier.py
```

实现 `MicroLMAudioClassifier`：

```python
class MicroLMAudioClassifier(nn.Module):
    def __init__(
        self,
        n_mels=128,
        d_model=256,
        num_layers=4,
        num_heads=4,
        d_ff=1024,
        num_classes=10,
        dropout=0.1,
    ):
        pass

    def forward(self, mel, attention_mask=None):
        pass
```

模型结构：

```text
mel [B, T, 128]
-> Linear(128, d_model)
-> positional embedding
-> TransformerEncoder
-> mean pooling
-> Linear(d_model, 10)
-> logits [B, 10]
```

### 为什么做

原 MicroLM 的思想是用轻量 Transformer 处理 token 序列。这里的迁移点是：

| 文本语言模型 | 音频分类模型 |
| --- | --- |
| token id | log-mel frame |
| token embedding | linear projection |
| causal Transformer | bidirectional Transformer encoder |
| LM head | classification head |
| next-token prediction | genre classification |

分类任务需要看到完整音频片段，因此使用双向注意力，而不是因果注意力。

---

## 11. Step 9：实现训练脚本

### 要做什么

创建：

```text
scripts/train_gtzan_classifier.py
```

训练脚本负责：

```text
1. 读取 configs/audio_gtzan_small.json
2. 构建 train / valid Dataset
3. 构建 DataLoader
4. 初始化 MicroLMAudioClassifier
5. 使用 CrossEntropyLoss
6. 使用 AdamW 优化器
7. 每个 epoch 后在 valid 上评估
8. 保存 best.pt 和 last.pt
```

损失函数：

```python
loss = F.cross_entropy(logits, labels)
```

### 为什么做

训练脚本把数据、模型、优化器和评估连接起来。它是项目从“代码组件”变成“可运行实验”的入口。

### 推荐输出

```text
outputs/gtzan_classifier/
├── config.json
├── train.log
├── checkpoints/
│   ├── best.pt
│   └── last.pt
└── metrics_valid.json
```

---

## 12. Step 10：实现评估脚本

### 要做什么

创建：

```text
scripts/eval_gtzan_classifier.py
```

评估脚本负责：

```text
1. 加载 config
2. 加载 test Dataset
3. 加载 best.pt
4. 输出 clip-level 指标
5. 输出 song-level 指标
6. 保存预测 CSV
7. 保存 confusion matrix
```

clip-level 预测：

```text
每个 3 秒 clip 独立预测一次类别。
```

song-level 预测：

```text
同一首歌的多个 clip logits 求平均，然后 argmax 得到整首歌类别。
```

### 为什么做

GTZAN 原始样本是一首 30 秒歌曲，而训练时使用的是 3 秒 clip。clip-level 能观察局部片段分类能力，song-level 更接近真实任务评价。

### 输出文件

```text
outputs/gtzan_classifier/
├── metrics_test.json
├── clip_predictions.csv
├── song_predictions.csv
└── confusion_matrix.png
```

---

## 13. Step 11：编写配置文件

### 要做什么

创建：

```text
configs/audio_gtzan_small.json
```

推荐内容：

```json
{
  "dataset": "gtzan",
  "metadata_csv": "data/processed/gtzan_mel/metadata.csv",
  "output_dir": "outputs/gtzan_classifier",

  "sample_rate": 22050,
  "clip_length": 3,
  "n_fft": 1024,
  "hop_length": 512,
  "n_mels": 128,

  "num_classes": 10,
  "d_model": 256,
  "num_layers": 4,
  "num_heads": 4,
  "d_ff": 1024,
  "dropout": 0.1,

  "batch_size": 32,
  "epochs": 50,
  "learning_rate": 0.0003,
  "weight_decay": 0.01,
  "seed": 42
}
```

### 为什么做

配置文件可以让训练命令保持简单，也能记录实验条件。以后改 batch size、层数、学习率时，不需要修改训练代码。

---

## 14. Step 12：运行完整流程

### 1. 准备数据

```bash
python scripts/prepare_gtzan.py --config configs/audio_gtzan_small.json
```

期望得到：

```text
data/splits/train.csv
data/splits/valid.csv
data/splits/test.csv
data/processed/gtzan_mel/metadata.csv
```

### 2. 训练模型

```bash
python scripts/train_gtzan_classifier.py --config configs/audio_gtzan_small.json
```

期望得到：

```text
outputs/gtzan_classifier/checkpoints/best.pt
outputs/gtzan_classifier/checkpoints/last.pt
outputs/gtzan_classifier/train.log
outputs/gtzan_classifier/metrics_valid.json
```

### 3. 评估模型

```bash
python scripts/eval_gtzan_classifier.py \
  --config configs/audio_gtzan_small.json \
  --checkpoint outputs/gtzan_classifier/checkpoints/best.pt
```

期望得到：

```text
outputs/gtzan_classifier/metrics_test.json
outputs/gtzan_classifier/clip_predictions.csv
outputs/gtzan_classifier/song_predictions.csv
outputs/gtzan_classifier/confusion_matrix.png
```

---

## 15. 指标说明

至少输出四个指标：

```text
clip-level accuracy
clip-level macro-F1
song-level accuracy
song-level macro-F1
```

指标作用：

| 指标 | 作用 |
| --- | --- |
| `clip-level accuracy` | 每个 3 秒片段的分类准确率 |
| `clip-level macro-F1` | 每个类别同等权重的片段级 F1 |
| `song-level accuracy` | 每首歌聚合后的分类准确率 |
| `song-level macro-F1` | 每个类别同等权重的歌曲级 F1 |
| `confusion matrix` | 查看哪些类别容易互相混淆 |

推荐最终主要报告 song-level 指标，因为 GTZAN 的自然单位是一首歌。

---

## 16. 最终验收标准

项目完成后应满足：

```text
1. 原 MicroLM 仓库能完成环境安装和基础导入
2. 原 MicroLM pretrain smoke 能运行并保存日志
3. 原 MicroLM SFT smoke 能运行并保存日志
4. 原 MicroLM LoRA / Qwen smoke 能运行并保存日志
5. reproduce_summary.md 能记录复现环境、结果和问题
6. GTZAN 原始数据能被脚本扫描
7. train / valid / test split 能正确生成
8. 同一首歌的所有 clip 不会跨 split
9. log-mel 特征能保存为 .npy
10. metadata.csv 能记录 split、label、feature_path、song_id
11. GTZANMelDataset 能正常返回 mel 和 label
12. MicroLMAudioClassifier 前向传播能输出 [B, 10]
13. 训练脚本能完成至少 1 个 epoch
14. best.pt 和 last.pt 能保存
15. 评估脚本能加载 best.pt
16. 能输出 clip-level 和 song-level 指标
17. 能输出 confusion_matrix.png
```

---

## 17. 推荐实现顺序

按下面顺序实现，最不容易卡住：

```text
Step 1. 创建 requirements.txt
Step 2. 创建项目目录结构
Step 3. 克隆原 MicroLM 到 external/MicroLM
Step 4. 安装原 MicroLM 依赖并运行基础导入检查
Step 5. 运行 pretrain smoke
Step 6. 运行 SFT smoke
Step 7. 运行 LoRA / Qwen smoke
Step 8. 编写 outputs/microlm_reproduce/reproduce_summary.md
Step 9. 编写 configs/audio_gtzan_small.json
Step 10. 编写 microlm_audio/utils/audio.py
Step 11. 编写 scripts/prepare_gtzan.py
Step 12. 生成 split CSV 和 metadata.csv
Step 13. 编写 microlm_audio/data/dataset.py
Step 14. 用 DataLoader 测试一个 batch
Step 15. 编写 microlm_audio/models/classifier.py
Step 16. 测试模型 forward 输出形状
Step 17. 编写 microlm_audio/utils/metrics.py
Step 18. 编写 scripts/train_gtzan_classifier.py
Step 19. 先跑 1 个 epoch 验证训练流程
Step 20. 跑完整训练
Step 21. 编写 scripts/eval_gtzan_classifier.py
Step 22. 输出最终测试指标和混淆矩阵
```

---

## 18. 最小可运行检查

在正式训练前，必须先做这些检查。

### 检查 mel 特征形状

```bash
python - <<'PY'
import numpy as np
path = "data/processed/gtzan_mel/metadata.csv"
print("metadata exists:", path)
PY
```

### 检查 Dataset

```bash
python - <<'PY'
from microlm_audio.data.dataset import GTZANMelDataset
ds = GTZANMelDataset("data/processed/gtzan_mel/metadata.csv", split="train")
item = ds[0]
print(item["mel"].shape, item["label"])
PY
```

### 检查模型 forward

```bash
python - <<'PY'
import torch
from microlm_audio.models.classifier import MicroLMAudioClassifier

model = MicroLMAudioClassifier()
x = torch.randn(2, 129, 128)
y = model(x)
print(y.shape)
PY
```

期望输出：

```text
torch.Size([2, 10])
```

---

## 19. 报告中的核心表述

可以在最终报告中这样描述：

```text
本项目首先复现原 MicroLM 的轻量级语言模型流程，包括预训练 smoke、SFT smoke 以及 LoRA / Qwen 相关 smoke 检查，以确认原项目环境、训练入口和 Transformer 主体可正常运行。随后，在保留轻量 Transformer 序列建模思想的基础上，构建面向 GTZAN 音乐风格分类的 MicroLM-Audio。模型将音频 waveform 转换为 log-mel spectrogram，并把每个 mel frame 视为连续音频 token，通过双向 self-attention 建模片段内的时序关系，最后使用 mean pooling 和分类头输出 10 类音乐风格预测。评估阶段同时报告 clip-level 和 song-level 指标，其中 song-level 预测通过聚合同一首歌曲的多个 clip logits 得到。
```

---

## 20. 后续可选扩展

主项目完成后，再考虑这些扩展：

```text
1. 加入 SpecAugment 提升鲁棒性
2. 尝试更长 clip_length，例如 5 秒或 10 秒
3. 尝试 attention pooling
4. 使用 FMA-small 做更大规模分类
5. 接入 pretrained audio encoder，例如 AST 或 wav2vec2
```

这些都不是第一版必需内容。第一版只需要把 MicroLM-Audio 主线完整跑通。

---

## 21. 项目完成后的最终形态

完成后，项目应能按两个阶段复现。

阶段 1：原 MicroLM 复现：

```bash
cd external/MicroLM
python scripts/train_pretrain.py --config configs/pretrain_smoke.json
python scripts/train_sft.py --config configs/sft_smoke.json
python scripts/train_qwen_lora.py --config configs/qwen_lora_structured_smoke.json
```

阶段 2：MicroLM-Audio 音频分类：

```bash
cd ../../
python scripts/prepare_gtzan.py --config configs/audio_gtzan_small.json
python scripts/train_gtzan_classifier.py --config configs/audio_gtzan_small.json
python scripts/eval_gtzan_classifier.py --config configs/audio_gtzan_small.json --checkpoint outputs/gtzan_classifier/checkpoints/best.pt
```

最终产物：

```text
MicroLM-Audio/
├── external/MicroLM/
├── configs/audio_gtzan_small.json
├── microlm_audio/
├── scripts/
├── data/processed/gtzan_mel/metadata.csv
└── outputs/
    ├── microlm_reproduce/
    │   ├── pretrain_smoke.log
    │   ├── sft_smoke.log
    │   ├── qwen_lora_smoke.log
    │   └── reproduce_summary.md
    └── gtzan_classifier/
        ├── checkpoints/best.pt
        ├── metrics_test.json
        ├── clip_predictions.csv
        ├── song_predictions.csv
        └── confusion_matrix.png
```

这就是第一版 MicroLM-Audio 的完整实现目标：先复现原 MicroLM 的语言模型与微调流程，再完成 GTZAN 音频分类迁移；保留必要主线，同时不被不必要的 baseline 分散注意力。
