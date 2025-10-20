# 🚀 YOLOv9 ONNX & RKNN Deployment

[![Platform](https://img.shields.io/badge/platform-RK3568-orange)](https://www.rock-chips.com/)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)

This repository provides end-to-end scripts to:
- Export **YOLOv9** models (from [WongKinYiu/yolov9](https://github.com/WongKinYiu/yolov9)) to **ONNX**
- Convert ONNX models to **RKNN** format for **Rockchip NPU** (e.g., RK3588, RK****)
- Run inference on **Linux + RKNN Toolkit2**

> ⚠️ **Note**: Pre-trained model weights are **NOT included** due to size and license. Please download them from the official source.

##  Models used in the project DownLoad
```bash
https://pan.quark.cn/s/4f35e2bb44aa
```
---
## 📁 Project Structure
```bash
YOLOv9-onnx-rknn/
├── step1_pt2onnx_Linux/ # Step 1: PyTorch → ONNX (Linux)
├── step2_onnx_inference_Linux/ # Step 2: ONNX inference test (Linux)
├── step3_onnx2rknn_Linux/ # Step 3: ONNX → RKNN (RKNN Toolkit2 Linux)
├── step4_rknn_infer_edge/ # Step 4: Deploy on Rockchip device(RK devices)
└── README.md # ← You are here
```
---

## 🧪 Step-by-Step Usage

### 🔹 Step 1: Export YOLOv9 to ONNX (Ubuntu + GPU/CPU)

This step converts the official YOLOv9 PyTorch model (`.pt`) to ONNX format on **Linux**.

#### 🛠️ Setup
```bash
cd step1_pt2onnx_Linux
pip install -r requirements.txt
```
#### Download the pre-trained weight (e.g., yolov9-s.pt) from the official YOLOv9 releases and place it in this folder.
```bash
python detect.py --source './data/images/horses.jpg' --img 640 --device cpu --weights 'yolov9-s.pt'
```
#### What to Expect
An ONNX model (yolov9-s.onnx) will be generated in the same directory.
You may see warnings or non-fatal errors in the terminal (e.g., about unused layers or opset compatibility).

#### → Don’t worry! 
As long as the .onnx file is created, the export succeeded.
The script also runs a quick inference to verify the model.

### 🔹 Step 2: Test ONNX Inference (Linux)

Verify that the exported ONNX model runs correctly using ONNX Runtime.
#### 🛠️ Setup
```bash
cd step2_onnx_inference_Linux
```
#### Copy the .onnx file from Step 1 into this folder.
#### Open yolov9_onnx_infer.py and modify line 13 to point to your model.
```bash
python yolov9_onnx_infer.py
```
#### Expected Result
The script loads the ONNX model using onnxruntime
Runs inference on a test image (e.g., test.jpg or hardcoded path)
Outputs detection results or saves a result image

### 🔹 Step 3: Convert ONNX to RKNN (Linux + RKNN Toolkit2)
Use Rockchip official RKNN Toolkit2 to convert ONNX to .rknn and test inference.

#### Download files
```bash
ON LINUX:https://github.com/airockchip/rknn-toolkit2
cd step3_onnx2rknn_Linux
python onnx2rknn.py  # remember change the model path(onnx and rknn) and data path.
```
### 🔹 Step 4: Run on Device
```bash
ON RK DEVICES:https://github.com/airockchip/rknn-toolkit2/tree/master/rknn-toolkit-lite2
python rknninfer.py  # remember change the model path(onnx and rknn) and data path.
```


## 📬 Contact

If you encounter any issues, need the pre-converted model files (`.onnx`, `.rknn`), or have questions about deployment:

> **Please send me a private message with your email address.**  
> I'll get back to you as soon as possible.

*(Note: Model weights like `yolov9-s.pt` must be downloaded from the [official YOLOv9 repo](https://github.com/WongKinYiu/yolov9) due to license restrictions.)*


## 🙏 Acknowledgements

This project builds upon the excellent work of the following open-source projects.  
Huge thanks to the authors and contributors!

- **[YOLOv9](https://github.com/WongKinYiu/yolov9)** by WongKinYiu – The original YOLOv9 implementation  
- **[RKNN Toolkit2](https://github.com/airockchip/rknn-toolkit2)** by Rockchip – Official toolkit for NPU deployment  
- **[ONNX](https://onnx.ai/)** – Open standard for model interoperability  
- **[ONNX Simplifier](https://github.com/daquexian/onnx-simplifier)** – Clean and optimize ONNX models
- **[CSDN PIONEER](https://blog.csdn.net/zhangqian_1?spm=1018.2118.3001.5148)** – Clear steps
- **[CSDN PIONEER Articles](https://blog.csdn.net/zhangqian_1/article/details/136321979?spm=1001.2014.3001.5501 )** – Clear steps

Without these tools, efficient edge deployment would not be possible. 🙌
