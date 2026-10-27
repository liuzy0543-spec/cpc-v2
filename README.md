# cpc-embedding-v1

Cell Painting 单细胞嵌入的**统一输入/输出接口**：一行代码，一个方法名换一个 Embedding。

```python
from cellpaint_pipeline import models

vec = models.embed('ca_mae', crops).as_numpy()   # (N, 384)
```

本包是**发布单元**，只含运行所需的代码、方法清单和本文档；
不含训练代码、不含审核记录、**不含模型权重**（体积与许可证都不允许）。

| 项 | 值 |
|---|---|
| 版本 | `cpc-embedding-v1` |
| 方法数 | 4（官方权重 4 个，替身 0 个） |
| 默认 profile | `v2`（即官方四方法） |
| 许可证 | 代码 MIT；**各方法权重许可见下表，逐个为准** |

---

## 1. 装

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows
pip install -r requirements.txt
pip install -e .                # 装成本包，之后随便哪个目录都能 import
```

**要跑嵌入就必须装深度学习框架**，而框架分两派，一个环境里往往装不全：

| 方法 | 需要 |
|---|---|
| `deepprofiler` | TensorFlow 2.x + h5py |
| `ca_mae` | torch + transformers + einops + safetensors |
| `cellclip` | torch + safetensors |
| `o2vae` | torch + e2cnn |


`requirements.txt` 只列了最小公共集；上面这几行按需装。
**没装框架不会静默失败**：`models.probe(name)` 会明确告诉你 `status='wrong-interpreter'`。

## 2. 放权重（唯一需要手工的一步）

权重不进仓库。全部放进一个根目录，用环境变量指过去：

```bash
export CPC_WEIGHTS_ROOT=/path/to/weights      # Windows: set CPC_WEIGHTS_ROOT=E:\cpc_weights
```

目录布局（`source` 里的子目录要保留）：

```
$CPC_WEIGHTS_ROOT/
├── Cell_Painting_CNN_v1.hdf5
├── openphenom_pkg/openphenom/model.safetensors
├── cellclip/model.safetensors
├── o2vae/o2vae_mefs_official.pt
```

解析顺序：显式 `weights_roots=` 参数 > `CPC_WEIGHTS_ROOT` > 内置默认列表
（`E:\cpc_weights`、`D:\cpc_weights`、`D:\模型权重`、`~/.cellpainting-weights`）。
**设了环境变量就是覆盖，不是追加。**

### 下载地址

### `deepprofiler`

- 放到：`$CPC_WEIGHTS_ROOT/Cell_Painting_CNN_v1.hdf5`
- sha256（`官方`）：`87b756f579336d606fd3d387b04bd2d422f9d410e5635a5d6b2cda7615833e73`
- 下载：<https://zenodo.org/records/7114558/files/Cell_Painting_CNN_v1.hdf5?download=1>

### `ca_mae`

- 放到：`$CPC_WEIGHTS_ROOT/openphenom_pkg/openphenom/model.safetensors`
- sha256（`官方`）：`f6e5f1c97101331b1574c9e4b99623260191c55eea5d98e40460849c0e4c4d47`
- 下载：<https://hf-mirror.com/recursionpharma/OpenPhenom/resolve/main/model.safetensors>
- 镜像：<https://huggingface.co/recursionpharma/OpenPhenom/resolve/main/model.safetensors>
- 体积：约 679 MB

### `cellclip`

- 放到：`$CPC_WEIGHTS_ROOT/cellclip/model.safetensors`
- sha256（`官方`）：`66c2d32a44a3e7d68deb489047df0df7676940308683b3285b1c8d28c471b550`
- 下载：<https://hf-mirror.com/suinleelab/CellCLIP/resolve/main/model.safetensors>
- 镜像：<https://huggingface.co/suinleelab/CellCLIP/resolve/main/model.safetensors>
- 体积：约 1716 MB

### `o2vae`

- 放到：`$CPC_WEIGHTS_ROOT/o2vae/o2vae_mefs_official.pt`
- sha256（`官方`）：`aa70cfa8b080c9d9989c11de316a2f7f074e8bef29366c4aaec24c807c5dfc0f`
- 下载：<https://zenodo.org/records/14988837/files/checkpoints.zip>
- 压缩包内路径：`checkpoints/o2vae_mefs_official.pt`（解压后取出即可）
- 体积：约 7 MB

下完建议核一次哈希：

```python
from cellpaint_pipeline.models.weights import verify_sha256
from pathlib import Path
print(verify_sha256(Path(r'$CPC_WEIGHTS_ROOT/...'))[:16])   # 与上表前缀一致即可
```

## 3. 用法

### 3.1 单方法

```python
import numpy as np
from cellpaint_pipeline import models

crops = np.load('crops.npy')          # (N, H, W, 5) uint16，原始裁剪
vec = models.embed('ca_mae', crops, assume='raw_crop')
print(vec.dim, vec.as_numpy().shape)  # 384 (N, 384)
print(vec.metadata['weights_provenance'])
```

`assume` 告诉接口"我给的是什么"：

| `assume` | 含义 |
|---|---|
| `'raw_crop'` | 原始裁剪，按 manifest 的预处理计划自动缩放/换布局 |
| 不传 | 已经预处理过（BCHW、float32、值域已就位），接口只做形状守卫 |

**不确定该传什么就先问接口**：`models.manifest('ca_mae').preprocess` 会把
scale、value_range、layout、source_map 全部列出来。

### 3.2 多个方法一起跑

```python
batch = models.embed_all(
    ['deepprofiler', 'ca_mae', 'cellclip'],
    {'deepprofiler': dp_crops, 'ca_mae': ca_crops, 'cellclip': vectors},
    assume={'deepprofiler': 'raw_crop', 'ca_mae': 'raw_crop'},
    on_error='collect',      # 一个失败不影响其它；'raise' 则立刻抛
)
print(list(batch.results), batch.errors)
```

`unit` / `channels` / `assume` / `masks` 四个参数都接受"统一值"或"按方法名给的映射"。

### 3.3 先体检再跑

```bash
python verify_install.py          # 列出每个方法 ready / 缺依赖 / 缺权重
```

## 4. 方法清单

| 方法 | 输出宽度 | 输入单元 | 输入布局 | 通道 | 权重来源 | 许可证 |
|---|---|---|---|---|---|---|
| `deepprofiler` | 672 | single_cell_crop | BHWC | 5 | 官方 | BSD-3-Clause (code) / CC-BY-4.0 (weigh |
| `ca_mae` | 384 | single_cell_crop | BCHW | 5 | 官方 | Recursion Non-Commercial EULA |
| `cellclip` | 512 | channel_embeddings | BCHW | 5 | 官方 | CC BY-NC-ND 4.0 |
| `o2vae` | 512 | single_cell_crop | BCHW | 1 | 官方 | MIT |

机器可读的同一份信息在 `MANIFEST.json`（含 sha256、下载地址、预处理全字段）。

## 5. 输入约定

- **通道顺序固定**为 `DNA, ER, RNA, AGP, Mito`（Cell Painting 标准）。
  数据源命名不同（如 BBBC022 的 `OrigHoechst/OrigSyto/OrigER/OrigPh_golgi/OrigMito`）时，
  用 `channels=` 显式声明映射，不要靠猜。
- `prepare()` 吃 **BHWC**、吐 **BCHW**；`embed()` 的"已预处理"入参是 **BCHW**。
  两种布局在同一个 API 里并存，传错会以 `channel-count` 报错 —— 这不是 bug，是守卫。
- 值域守卫是硬的：uint16 原始值送进声明 `[0,1]` 的方法会被拒（`reason='value-range'`），
  防的是"除以 255 又除以 65535"这种静默错。

## 6. 已知边界（不要当成 bug）

- `o2vae` 单张与批量曾差 2.6% std（e2cnn `NormMaxPool` 的 argmax 翻转），已通过
  "单行补成两行再取首行"修好；验证：单张 vs 批量的最大差 3e-6。
- 官方权重的 `o2vae` 训练于细胞剪影，本包喂的是核剪影，**分布不一致**，
  其分数不可与论文直接比较。
- 嵌入的绝对数值只在"同一份权重 + 同一个批量形状"下逐位可复现。

## 7. 与 `cpc-embedding-v2` 的区别

| | cpc-embedding-v1 | cpc-embedding-v2 |
|---|---|---|
| 方法数 | 4 | 7 |
| 权重 | 全部官方 | 4 官方 + 3 替身 |
| 适合 | 出可发表的数字 | 接口完整性与对比实验 |

两个包的代码**只差一行**（`registry.py` 里默认 profile 的兜底值），其余逐字节相同。

## 8. 目录

```
cpc-embedding-v1/
├── README.md
├── MANIFEST.json          方法清单（机器可读）
├── requirements.txt
├── verify_install.py      自检
├── pyproject.toml
└── src/cellpaint_pipeline/
    ├── __init__.py
    ├── errors.py
    └── models/
        ├── contract.py      契约：manifest / 预处理规格 / 兼容性判定（纯数据）
        ├── registry.py      七个方法的声明与 profile
        ├── weights.py       权重解析与哈希校验（纯标准库）
        ├── preprocessing.py 预处理执行
        ├── embedding.py     embed / embed_many / embed_all / 缓存
        └── adapters/        每个方法一个适配器
```
