# 人体姿态识别演示

这是从农业产权交易项目素材中整理出的独立姿态识别示例，包含图片姿态分析和摄像头实时姿态识别脚本。

## 运行

建议使用 Python 3.10。安装依赖：

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
pip install -r requirements.txt
```

实时摄像头演示：

```bash
python poseVideo.py
```

图片演示脚本默认读取 `img/1.jpg`，并将结果写入 `output/`。请自行放入图片；项目不公开原始人像样例。Ultralytics 会在首次运行时下载脚本所需的轻量姿态模型。`poseIMG_Akimbo.py` 使用轻量 `yolo11n-pose.pt`，避免依赖体积较大的模型文件。

如需使用 NVIDIA GPU，请按本机 CUDA 版本安装对应的 PyTorch；默认依赖也可用于 CPU 推理。
