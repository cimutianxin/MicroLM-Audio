# MicroLM-Audio 项目实现教程

## 0. 项目目标

本项目分为三个阶段：

```text
阶段 1：复制并使用独立 microlm 环境
阶段 2：完整复现 jiaran-king/MicroLM 全流程
阶段 3：在复现基础上改造为 MicroLM-Audio 音乐风格分类器
```

最终项目名称：

```text
MicroLM-Audio: A Lightweight Transformer for Music Genre Classification
```

阶段 1 先从当前音频项目环境复制一个独立环境：

```text
microlm-audio -> microlm
```

环境分工：

```text
microlm       : 专门用于复现 jiaran-king/MicroLM
microlm-audio : 后续用于 GTZAN 音频分类改造
```

阶段 2 先完整实现原 MicroLM 项目的整个过程，而不是只跑 smoke：

```text
jiaran-king/MicroLM
-> 环境配置
-> pytest / smoke 验证
-> MiniMind 数据下载
-> 预训练语料清洗
-> BPE tokenizer 训练
-> 语料 tokenization
-> 正式 pretrain
-> 正式 SFT
-> 正式 LoRA SFT
-> KV cache 推理 / chat / eval 检查
-> InstructIE 数据 pipeline
-> Qwen LoRA 结构化微调
-> 评测与结果归档
-> 记录复现结果
```

阶段 3 的核心任务是把轻量级 Transformer 从文本序列建模迁移到音频分类：

```text
audio waveform
-> log-mel spectrogram
-> frame sequence
-> Transformer encoder
-> pooling
-> classifier head
-> genre label
```

项目最终保留两条必要主线，按顺序执行：

```text
主线 1：原 MicroLM 完整复现
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

资源预期：

```text
MicroLM smoke / tokenizer / 小规模 pretrain / GTZAN 分类：16GB GPU 足够
完整 MiniMind pretrain / SFT / LoRA：可在 16GB GPU 上尝试，训练时间取决于数据规模
Qwen2.5-1.5B LoRA：16GB GPU 有机会运行，但更依赖显存余量、batch size、量化策略和磁盘空间
vLLM 部署检查：如果资源不足，先保留为 smoke / help / 配置检查，并在 reproduce_summary.md 记录原因
```

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

本项目使用两个 Conda 环境：

```text
microlm       : 复现 jiaran-king/MicroLM 全流程
microlm-audio : 后续实现 MicroLM-Audio 音频分类
```

如果已经有 `microlm-audio` 环境，直接复制出 `microlm`：

```bash
conda create --name microlm --clone microlm-audio -y
```

如果从零开始，也可以直接创建：

```bash
conda create -n microlm python=3.11 -y
conda activate microlm
```

原 MicroLM 当前 `pyproject.toml` 要求：

```text
python >= 3.11
```

因此 `microlm` 环境必须使用 Python 3.11 或更高版本。

安装 PyTorch：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

安装音频项目依赖：

```bash
pip install numpy pandas tqdm scikit-learn matplotlib librosa soundfile
```

进入原 MicroLM 仓库后安装上游项目依赖：

```bash
cd external/MicroLM
pip install -e ".[all]"
```

本仓库的 `requirements.txt` 至少包含：

```text
torch
torchvision
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

环境拆分的原因：

```text
1. 原 MicroLM 全流程依赖 transformers / peft / datasets / wandb 等 LLM 工具
2. MicroLM-Audio 后续还会加入音频处理、GTZAN 数据、分类训练逻辑
3. 两条线分环境可以避免依赖互相污染
4. 复现上游项目时始终使用 microlm，音频改造时再切回 microlm-audio
```

### 验证命令

```bash
conda activate microlm
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "from microlm.model.transformer import TransformerLM; print('MicroLM ok')"

conda activate microlm-audio
python -c "import librosa, sklearn, pandas; print('ok')"
```

---

## 4. Step 2：搭建并 smoke 验证原 MicroLM

### 要做什么

把上游仓库放到：

```text
external/MicroLM/
```

推荐命令：

```bash
mkdir -p external
git clone https://github.com/jiaran-king/MicroLM.git external/MicroLM
cd external/MicroLM
```

激活专用环境：

```bash
conda activate microlm
```

安装上游项目：

```bash
pip install -e ".[all]"
```

先做基础导入和测试：

```bash
python -c "import torch; print(torch.cuda.is_available())"
pytest tests/
```

模型导入检查：

```bash
python -c "from microlm.model.transformer import TransformerLM; print('ok')"
```

运行最小 pretrain smoke：

```bash
python scripts/train_pretrain.py --config configs/pretrain_smoke.json
```

### 为什么做

这一阶段不追求最终效果，只确认上游项目能在本机稳定启动：

```text
1. 原项目依赖可安装
2. 原 Transformer 主体可实例化
3. 原训练脚本可启动
4. checkpoint、日志、配置读取流程可用
```

这些结果会给 Step 3 的完整复现和后续音频改造提供可信基础。

### 输出文件

保存到：

```text
outputs/microlm_reproduce/
├── pytest.log
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

## 5. Step 3：完整复现 jiaran-king/MicroLM 全流程

### 要做什么

Step 2 只证明项目能跑；Step 3 要按上游 README 的主线，把 MicroLM 项目从数据到训练产物完整走一遍。

完整流程分为两条线：

```text
A/B 线：自研 MicroLM
MiniMind 数据
-> 清洗
-> tokenizer 训练
-> tokenization
-> pretrain
-> SFT
-> LoRA SFT
-> 推理 / KV cache / eval

C/D 线：Qwen 迁移
InstructIE 数据 + Qwen2.5-1.5B-Instruct
-> 六步数据 pipeline
-> Qwen LoRA 结构化微调
-> 自动评测 / 导出 / 部署检查
```

### A0：下载 MiniMind 数据

```bash
conda activate microlm
cd external/MicroLM
pip install huggingface_hub

mkdir -p data data/minimind_sft/gongjy/minimind_dataset

python - <<'PY'
from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="jingyaogong/minimind_dataset",
    repo_type="dataset",
    filename="pretrain_t2t_mini.jsonl",
    local_dir="data",
)
hf_hub_download(
    repo_id="jingyaogong/minimind_dataset",
    repo_type="dataset",
    filename="sft_t2t_mini.jsonl",
    local_dir="data/minimind_sft/gongjy/minimind_dataset",
)
PY
```

### A1：清洗预训练语料

```bash
python scripts/prepare_pretrain_jsonl.py \
  --input-path data/pretrain_t2t_mini.jsonl \
  --output-dir data/pretrain_clean \
  --document-separator "<|endoftext|>" \
  --replace-literal "<|im_end|>=\n" \
  --replace-literal "<|im_start|>=\n" \
  --replace-literal "<think>=\n" \
  --replace-literal "</think>=\n" \
  --clean-html
```

产物：

```text
data/pretrain_clean/train.txt
data/pretrain_clean/valid.txt
data/pretrain_clean/tokenizer_corpus.txt
data/pretrain_clean/metadata.json
```

### A2：构造 tokenizer 训练样本

```bash
python - <<'PY'
from pathlib import Path

src = Path("data/pretrain_clean/tokenizer_corpus.txt")
dst = Path("data/pretrain_clean/tokenizer_sample.txt")
sample_bytes = 15 * 1024 * 1024

with src.open("rb") as fsrc, dst.open("wb") as fdst:
    fdst.write(fsrc.read(sample_bytes))
PY
```

### A3：训练 BPE tokenizer

```bash
python scripts/train_tokenizer.py --config configs/tokenizer_full_clean.json
```

产物：

```text
outputs/tokenizer_full_clean/vocab.json
outputs/tokenizer_full_clean/merge.txt
```

### A4：将预训练语料编码成 token IDs

```bash
python scripts/tokenize_corpus.py --config configs/tokenize_full_corpus.json
```

产物：

```text
data/pretrain_clean/tokenized_full/train_ids.npy
data/pretrain_clean/tokenized_full/valid_ids.npy
data/pretrain_clean/tokenized_full/metadata.json
```

### B1：正式 pretrain

```bash
python scripts/train_pretrain.py --config configs/pretrain_full_corpus.json
```

产物：

```text
outputs/pretrain_full_corpus/ckpt_final.pt
outputs/pretrain_full_corpus/model_config.json
```

### B2：正式 SFT 全参微调

```bash
python scripts/train_sft.py --config configs/sft_baseline.json
```

产物：

```text
outputs/sft_baseline/ckpt_final.pt
outputs/sft_baseline/train_log.jsonl
```

### B3：正式 SFT LoRA 微调

```bash
python scripts/train_sft.py --config configs/sft_lora.json
```

产物：

```text
outputs/sft_lora/ckpt_final.pt
outputs/sft_lora/lora_adaptor.pt
```

### B4：推理、聊天和 KV cache 检查

使用训练产物做基础推理：

```bash
python scripts/generate_text.py --help
python scripts/chat.py --help
python scripts/benchmark_kvcache.py
```

这一阶段用于确认：

```text
1. checkpoint 能加载
2. 文本生成链路可用
3. chat.py 交互入口可用
4. KV cache 推理优化脚本可运行
```

### C0：下载 InstructIE 数据与 Qwen 基座模型

```bash
pip install huggingface_hub

mkdir -p data/instructie Qwen2.5-1.5B-Instruct

python - <<'PY'
from huggingface_hub import hf_hub_download, snapshot_download

for filename in ["train_zh.json", "valid_zh.json", "test_zh.json", "schema_zh.json"]:
    hf_hub_download(
        repo_id="zjunlp/InstructIE",
        repo_type="dataset",
        filename=filename,
        local_dir="data/instructie",
    )

snapshot_download(
    repo_id="Qwen/Qwen2.5-1.5B-Instruct",
    local_dir="Qwen2.5-1.5B-Instruct",
)
PY
```

### C1：运行 InstructIE 六步数据 pipeline

```bash
python scripts/01_normalize.py
python scripts/02_filter.py
python scripts/03_quality_tier.py
python scripts/04_derive_tasks.py
python scripts/05_stratified_sample.py
python scripts/06_to_chat_jsonl.py
```

产物：

```text
data/processed/*.jsonl
data/sft_candidate/train.jsonl
data/sft_candidate/valid.jsonl
data/sft_candidate/metadata.json
```

### D1：Qwen LoRA 结构化微调

```bash
python scripts/train_qwen_lora.py --config configs/qwen_lora_structured.json
```

产物：

```text
outputs/qwen_lora/adaptor_final/
outputs/qwen_lora/best_adaptor/
outputs/qwen_lora/train_log.jsonl
```

### D2：Qwen 评测、导出和部署检查

```bash
python scripts/run_instructie_eval.py
python scripts/check_structured_stability.py
python scripts/export_final_model.py --help
python scripts/smoke_vllm.py --help
```

如果本机显存或网络不适合完整下载 Qwen2.5-1.5B-Instruct，至少先运行 smoke 配置：

```bash
cd external/MicroLM
python scripts/train_sft.py --config configs/sft_smoke.json
python scripts/train_qwen_lora.py --config configs/qwen_lora_structured_smoke.json
```

### 为什么做

这一阶段完整复现上游项目的价值是：

```text
1. 理解 MicroLM 从 tokenizer 到 pretrain 的完整数据链路
2. 验证 SFT 和 LoRA 微调的训练协议
3. 理解 KV cache、chat、eval 等推理侧能力
4. 理解 Qwen 迁移线的数据 pipeline 与结构化输出训练
5. 为后续把 Transformer 主体迁移到音频分类建立完整上下文
```

### 输出文件

建议保存到：

```text
outputs/microlm_reproduce/
├── full_pretrain.log
├── sft_smoke.log
├── qwen_lora_smoke.log
├── full_sft.log
├── full_lora.log
├── qwen_lora.log
├── eval_summary.md
└── reproduce_summary.md
```

`reproduce_summary.md` 中继续补充：

```text
1. MiniMind 数据是否下载成功
2. tokenizer 是否训练成功
3. pretrain / SFT / LoRA 是否完成
4. InstructIE 数据 pipeline 是否完成
5. Qwen LoRA 是否完成或是否因资源原因只跑 smoke
6. 每个流程的最终 loss 或关键日志
7. 遇到的依赖、显存、配置问题
8. 对原仓库做过哪些最小修改
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
1. 已从 microlm-audio 复制出独立 microlm 环境
2. microlm 环境使用 Python >= 3.11
3. 原 MicroLM 仓库能完成环境安装和基础导入
4. 原 MicroLM pytest 能通过
5. 原 MicroLM pretrain smoke 能运行并保存日志
6. MiniMind 数据能下载并完成清洗
7. BPE tokenizer 能训练并保存 vocab / merge
8. 预训练语料能 tokenization 成 train_ids.npy / valid_ids.npy
9. 正式 pretrain 能启动并保存 checkpoint
10. 正式 SFT 全参微调能启动并保存 checkpoint
11. 正式 LoRA SFT 能启动并保存 adaptor
12. 推理 / chat / KV cache 检查能运行
13. InstructIE 六步数据 pipeline 能运行
14. Qwen LoRA 正式训练能完成，或在资源不足时完成 smoke 并记录原因
15. reproduce_summary.md 能记录完整复现环境、命令、结果和问题
16. GTZAN 原始数据能被脚本扫描
17. train / valid / test split 能正确生成
18. 同一首歌的所有 clip 不会跨 split
19. log-mel 特征能保存为 .npy
20. metadata.csv 能记录 split、label、feature_path、song_id
21. GTZANMelDataset 能正常返回 mel 和 label
22. MicroLMAudioClassifier 前向传播能输出 [B, 10]
23. 训练脚本能完成至少 1 个 epoch
24. best.pt 和 last.pt 能保存
25. 评估脚本能加载 best.pt
26. 能输出 clip-level 和 song-level 指标
27. 能输出 confusion_matrix.png
```

---

## 17. 推荐实现顺序

按下面顺序实现，最不容易卡住：

```text
Step 1. 创建 requirements.txt
Step 2. 创建项目目录结构
Step 3. 从 microlm-audio 复制 microlm 环境
Step 4. 克隆原 MicroLM 到 external/MicroLM
Step 5. 在 microlm 环境中安装原 MicroLM 依赖
Step 6. 运行 pytest 和 pretrain smoke
Step 7. 下载 MiniMind 数据
Step 8. 清洗预训练语料
Step 9. 训练 BPE tokenizer
Step 10. tokenization 生成 train_ids.npy / valid_ids.npy
Step 11. 运行正式 pretrain
Step 12. 运行正式 SFT 全参微调
Step 13. 运行正式 LoRA SFT
Step 14. 跑推理、chat、KV cache 检查
Step 15. 下载 InstructIE 数据和 Qwen 基座模型
Step 16. 运行 InstructIE 六步数据 pipeline
Step 17. 运行 Qwen LoRA 结构化微调
Step 18. 汇总 outputs/microlm_reproduce/reproduce_summary.md
Step 19. 编写 configs/audio_gtzan_small.json
Step 20. 编写 microlm_audio/utils/audio.py
Step 21. 编写 scripts/prepare_gtzan.py
Step 22. 生成 split CSV 和 metadata.csv
Step 23. 编写 microlm_audio/data/dataset.py
Step 24. 用 DataLoader 测试一个 batch
Step 25. 编写 microlm_audio/models/classifier.py
Step 26. 测试模型 forward 输出形状
Step 27. 编写 microlm_audio/utils/metrics.py
Step 28. 编写 scripts/train_gtzan_classifier.py
Step 29. 先跑 1 个 epoch 验证训练流程
Step 30. 跑完整训练
Step 31. 编写 scripts/eval_gtzan_classifier.py
Step 32. 输出最终测试指标和混淆矩阵
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
本项目首先复制独立的 microlm 环境，并完整复现 jiaran-king/MicroLM 项目流程，包括 MiniMind 数据准备、预训练语料清洗、BPE tokenizer 训练、正式 pretrain、SFT、LoRA SFT、KV cache 推理检查，以及 InstructIE 数据 pipeline 和 Qwen LoRA 结构化微调。该阶段用于完整理解 MicroLM 从数据、模型、训练、微调到推理评测的闭环。随后，在保留轻量 Transformer 序列建模思想的基础上，构建面向 GTZAN 音乐风格分类的 MicroLM-Audio。模型将音频 waveform 转换为 log-mel spectrogram，并把每个 mel frame 视为连续音频 token，通过双向 self-attention 建模片段内的时序关系，最后使用 mean pooling 和分类头输出 10 类音乐风格预测。
```

---

## 20. 后续可选扩展

原 MicroLM 完整流程和 MicroLM-Audio 主线完成后，再考虑这些扩展：

```text
1. 加入 SpecAugment 提升鲁棒性
2. 尝试更长 clip_length，例如 5 秒或 10 秒
3. 尝试 attention pooling
4. 使用 FMA-small 做更大规模分类
5. 接入 pretrained audio encoder，例如 AST 或 wav2vec2
```

这些都不是第一版必需内容。第一版优先级是：先完整复现上游 MicroLM，再完成 MicroLM-Audio 主线。

---

## 21. 项目完成后的最终形态

完成后，项目应能按三个阶段复现。

阶段 1：复制并验证 microlm 环境：

```bash
conda create --name microlm --clone microlm-audio -y
conda activate microlm
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

阶段 2：完整复现原 MicroLM：

```bash
cd external/MicroLM
pip install -e ".[all]"
pytest tests/
python scripts/train_pretrain.py --config configs/pretrain_smoke.json

# 完整自研 MicroLM 主线
python scripts/prepare_pretrain_jsonl.py \
  --input-path data/pretrain_t2t_mini.jsonl \
  --output-dir data/pretrain_clean \
  --document-separator "<|endoftext|>" \
  --replace-literal "<|im_end|>=\n" \
  --replace-literal "<|im_start|>=\n" \
  --replace-literal "<think>=\n" \
  --replace-literal "</think>=\n" \
  --clean-html
python scripts/train_tokenizer.py --config configs/tokenizer_full_clean.json
python scripts/tokenize_corpus.py --config configs/tokenize_full_corpus.json
python scripts/train_pretrain.py --config configs/pretrain_full_corpus.json
python scripts/train_sft.py --config configs/sft_baseline.json
python scripts/train_sft.py --config configs/sft_lora.json

# Qwen 迁移主线
python scripts/01_normalize.py
python scripts/02_filter.py
python scripts/03_quality_tier.py
python scripts/04_derive_tasks.py
python scripts/05_stratified_sample.py
python scripts/06_to_chat_jsonl.py
python scripts/train_qwen_lora.py --config configs/qwen_lora_structured.json
```

阶段 3：MicroLM-Audio 音频分类：

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
    │   ├── full_pretrain.log
    │   ├── sft_smoke.log
    │   ├── qwen_lora_smoke.log
    │   ├── full_sft.log
    │   ├── full_lora.log
    │   ├── qwen_lora.log
    │   └── reproduce_summary.md
    └── gtzan_classifier/
        ├── checkpoints/best.pt
        ├── metrics_test.json
        ├── clip_predictions.csv
        ├── song_predictions.csv
        └── confusion_matrix.png
```

这就是第一版 MicroLM-Audio 的完整实现目标：先在独立 microlm 环境中完整复现 jiaran-king/MicroLM，再完成 GTZAN 音频分类迁移；保留必要主线，同时不被不必要的 baseline 分散注意力。
