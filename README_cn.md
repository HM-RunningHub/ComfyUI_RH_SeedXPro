# ComfyUI SeedXPro 翻译节点

## 功能特性

- 支持多语言翻译，基于 ByteDance-Seed/Seed-X-PPO-7B 模型
- **自动模型下载**: 首次使用时自动从 Hugging Face 下载模型到 `models/Seed-X-PPO-7B` 目录
- 无需手动下载模型文件

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用说明

1. 将此插件放入 ComfyUI 的 `custom_nodes` 目录
2. 首次运行时，系统会自动检查并下载 `ByteDance-Seed/Seed-X-PPO-7B` 模型
3. 模型会下载到 ComfyUI 的 `models/Seed-X-PPO-7B` 目录
4. 下载完成后即可正常使用翻译功能

## 注意事项

- 模型文件较大（约13GB），首次下载需要一定时间和网络带宽
- 确保有足够的磁盘空间存储模型文件
- 需要 CUDA 环境运行模型 