# 本地扫雷 AI 训练项目

这个项目只用于本地扫雷训练，不连接、不控制任何公开扫雷网站。

## 项目结构

```text
minesweeper_ai_training/
├─ minesweeper_ai/
│  ├─ env.py          # 本地扫雷环境
│  ├─ solver.py       # 逻辑求解器：基础推理、集合包含、边界枚举概率
│  ├─ dataset.py      # 自动采集训练数据
│  └─ model.py        # 小型 CNN 模型
├─ scripts/
│  ├─ collect_dataset.py   # 生成训练数据
│  ├─ train_cnn.py         # 训练模型
│  ├─ evaluate.py          # 评估纯逻辑 / 逻辑+模型胜率
│  └─ smoke_test.py        # 快速测试安装是否正常
└─ requirements.txt
```

## 安装

建议使用 Python 3.10 或以上版本。

```bash
pip install -r requirements.txt
```

如果你的电脑已经装好 PyTorch，也可以只安装 numpy：

```bash
pip install numpy
```

## 快速测试

```bash
python scripts/smoke_test.py
```

## 第一步：生成训练数据

先用高级局生成少量数据测试：

```bash
python scripts/collect_dataset.py --width 30 --height 16 --mines 99 --games 200 --out data_expert_200.npz
```

想多训练一些，可以把 `--games` 改成 2000、5000 或更多：

```bash
python scripts/collect_dataset.py --width 30 --height 16 --mines 99 --games 5000 --out data_expert_5000.npz
```

数据来自本地模拟局面。程序会在逻辑求解器推不动、需要猜的时候记录棋盘状态，并用隐藏雷图生成监督标签。

## 第二步：训练 CNN

```bash
python scripts/train_cnn.py --data data_expert_5000.npz --epochs 8 --batch-size 64 --out model_expert.pt
```

模型学习的不是“必胜外挂”，而是：

```text
在当前可见棋盘下，未知格是雷的概率大概是多少。
```

## 第三步：评估胜率

只用逻辑求解器：

```bash
python scripts/evaluate.py --games 200 --width 30 --height 16 --mines 99
```

逻辑求解器 + 训练模型：

```bash
python scripts/evaluate.py --games 200 --width 30 --height 16 --mines 99 --model model_expert.pt
```

## 参数建议

电脑配置一般时：

```bash
python scripts/collect_dataset.py --games 1000 --out data_1000.npz
python scripts/train_cnn.py --data data_1000.npz --epochs 5 --batch-size 32 --out model.pt
python scripts/evaluate.py --games 100 --model model.pt
```

电脑配置较好时：

```bash
python scripts/collect_dataset.py --games 10000 --out data_10000.npz
python scripts/train_cnn.py --data data_10000.npz --epochs 10 --batch-size 128 --out model.pt
python scripts/evaluate.py --games 500 --model model.pt
```

## 重要说明

普通随机扫雷存在信息不足局面，所以 AI 不能保证 100% 成功。这个项目的目标是提高“必须猜”的局面中的选择质量。
