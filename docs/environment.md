# ASAP_official 环境要求

> 基于 `.sh/run_q1_motion_tracking.sh` 和 `setup.py` 整理，conda 环境已验证可用。

## Conda 环境

- **路径**: `/root/autodl-tmp/conda_envs/q1_motion`
- **Python**: 3.8.20

## 核心依赖

| 包 | 版本 | 说明 |
|---|---|---|
| Python | 3.8.20 | IsaacGym 仅支持 3.8 |
| PyTorch | 2.1.0+cu121 | CUDA 12.1 |
| torchvision | 0.16.0+cu121 | |
| torchaudio | 2.1.0+cu121 | |
| CUDA | 12.1 | 系统级 |
| IsaacGym | Preview 4 (1.0rc4) | 本地安装，路径 `/root/autodl-tmp/isaacgym/python` |
| hydra-core | 1.3.2 | 配置管理 |
| numpy | 1.23.5 | 精确锁定版本 |
| onnx | 1.17.0 | 模型导出 |
| onnxruntime | 1.19.2 | ONNX 推理 |
| wandb | 0.24.2 | 训练日志 |
| tensorboard | - | 训练监控 |

## 项目包（本地 editable install）

```bash
pip install -e /root/autodl-tmp/ASAP_official/.           # asap (humanoidverse/)
pip install -e /root/autodl-tmp/ASAP_official/isaac_utils  # 数学/旋转工具
```

## pip 依赖（setup.py）

```
hydra-core>=1.2.0
numpy==1.23.5
rich, ipdb, matplotlib, termcolor, wandb, plotly, tqdm
loguru, meshcat, pynput, scipy, tensorboard
onnx, onnxruntime, opencv-python, joblib
easydict, lxml, numpy-stl, open3d
```

## 关键注意事项

1. **IsaacGym 不是 pip 包**：从 NVIDIA 官方下载 Preview 4，本地 `pip install -e` 安装。绑定文件为 `gym_38.so`（Python 3.8）。
2. **导入顺序**：`import isaacgym` 必须在 `import torch` 之前。
3. **numpy 版本锁定**：`==1.23.5`，高版本可能与 IsaacGym 不兼容。
4. **Python 版本锁定**：必须是 3.8，IsaacGym Preview 4 的预编译 `.so` 仅支持 3.8。
