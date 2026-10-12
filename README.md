# PPE_Detection_Model

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![YOLO](https://img.shields.io/badge/Model-YOLOv11-green)
![Ultralytics](https://img.shields.io/badge/Trained%20with-Ultralytics-orange)
![Colab](https://img.shields.io/badge/Trained%20on-Google%20Colab-F9AB00?logo=googlecolab&logoColor=white)
![License](https://img.shields.io/badge/License-AGPL--3.0-blue)
![Offline](https://img.shields.io/badge/Mode-Offline-red)

An edge-computing solution for real-time workplace safety monitoring. This system detects Personal Protective Equipment (PPE) compliance and identifies fire/smoke hazards **without requiring an internet connection**.

Designed for privacy, low latency, and reliability in remote or high-security industrial environments.

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Tech Stack](#️-tech-stack)
- [Project Structure](#-project-structure)
- [Installation Guide](#-installation-guide)
  - [Prerequisites](#prerequisites)
  - [Step 1: Install Python](#step-1-install-python)
  - [Step 2: Install Git](#step-2-install-git)
  - [Step 3: Clone the Repository](#step-3-clone-the-repository)
  - [Step 4: Create a Virtual Environment](#step-4-create-a-virtual-environment)
  - [Step 5: Install Dependencies](#step-5-install-dependencies)
  - [Step 6: Verify Installation](#step-6-verify-installation)
- [Usage](#-usage)
- [Model Performance](#-model-performance)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)
- [Disclaimer](#-disclaimer)

---

## 📋 Overview

This project uses Deep Learning (Computer Vision) to monitor video feeds from cameras (USB, IP, or CCTV) and alerts safety officers in real-time if:
1. Workers are missing required safety gear (Helmets, Vests, Masks).
2. Fire or Smoke is detected in the frame.

The custom object detection model was trained using the **Ultralytics** framework with the **YOLOv11** architecture. Training was performed entirely on **Google Colab**, utilizing its free GPU (Tesla T4) to accelerate the training process on a custom dataset.

**Key differentiator:** The inference engine runs entirely on local hardware (CPU/GPU/Edge TPU). No data is sent to the cloud, ensuring 100% privacy and operation in air-gapped facilities.

---

## ✨ Features

- **100% Offline Inference:** Runs locally on NVIDIA Jetson, Raspberry Pi, or standard x86 PCs.
- **YOLOv11 Backbone:** State-of-the-art object detection for higher accuracy and faster FPS.
- **Trained with Ultralytics:** Leverages the industry-standard Ultralytics library for optimal model performance.
- **Cloud-Trained (Google Colab):** Model was trained efficiently using Google Colab's T4 GPU environment.
- **Dual Detection Pipeline:**
    - **PPE Compliance:** Detects Helmets, Safety Vests, Goggles, and Masks.
    - **Hazard Detection:** Detects Open Flames and Smoke plumes.
- **Real-Time Alerts:**
    - Audible alarms via local buzzer/speaker.
    - Visual bounding boxes with confidence scores.
    - Automated logging of violations to a local database.
- **Low Latency:** Optimized for real-time FPS using TensorRT/ONNX runtime.
- **Web Interface:** Built with FastAPI for easy monitoring and control.

---

## 🛠️ Tech Stack

- **Language:** Python 3.9+
- **Web Framework:** FastAPI, Uvicorn
- **Training Framework:** [Ultralytics](https://github.com/ultralytics/ultralytics) (YOLOv11)
- **Training Environment:** Google Colab (Tesla T4 GPU)
- **Deep Learning Framework:** PyTorch / ONNX Runtime
- **Model:** YOLOv11 (You Only Look Once) - Custom trained
- **Computer Vision:** OpenCV
- **Hardware Support:** CPU, NVIDIA CUDA, Google Coral Edge TPU

---

## 📂 Project Structure

```text
├── .github/workflows/      # GitHub Actions
├── models/                 # Model related files
├── static/                 # Web interface files (HTML/CSS/JS)
├── .gitattributes          # Git attributes
├── .gitignore              # Git ignore rules
├── README.md               # Project documentation
├── app.py                  # Main application entry point (FastAPI)
├── best.onnx               # Optimized ONNX model weights
├── best.pt                 # PyTorch model weights
├── requirements.txt        # Python dependencies
└── run.bat                 # Windows quick-start script
```

---

## 🚀 Installation Guide

This guide will walk you through setting up the PPE Detection Model from scratch. Follow the steps in order.

### Prerequisites

Before you begin, ensure you have the following:

- A computer running **Windows 10/11**, **Linux (Ubuntu 20.04+)**, or **macOS 11+**.
- A webcam or IP camera for live testing (optional but recommended).
- At least **4 GB of RAM** (8 GB+ recommended).
- **5 GB of free disk space** for dependencies and models.
- An internet connection **only for the initial setup** (downloading dependencies). After setup, the system runs completely offline.

### Step 1: Install Python

Python 3.8 or higher is required. We recommend Python 3.9 or 3.10.

#### Windows
1. Download the official installer from [python.org](https://www.python.org/downloads/).
2. Run the installer. **Important:** Check the box **"Add Python to PATH"** at the bottom of the installer window.
3. Click **"Install Now"** and wait for completion.
4. Verify the installation by opening Command Prompt and running:
   ```bash
   python --version
   ```
   You should see something like `Python 3.10.11`.

#### Linux (Ubuntu/Debian)
Open a terminal and run:
```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv -y
```
Verify:
```bash
python3 --version
pip3 --version
```

#### macOS
1. Install Homebrew if you haven't already: [brew.sh](https://brew.sh/)
2. Install Python:
   ```bash
   brew install python@3.10
   ```
3. Verify:
   ```bash
   python3 --version
   ```

### Step 2: Install Git

Git is needed to clone the repository.

#### Windows
Download and install Git from [git-scm.com](https://git-scm.com/download/win). Use the default settings.

#### Linux
```bash
sudo apt install git -y
```

#### macOS
```bash
brew install git
```

Verify Git installation:
```bash
git --version
```

### Step 3: Clone the Repository

Open a terminal (or Command Prompt on Windows) and run:

```bash
git clone https://github.com/Ashis7116/PPE_Detection_Model.git
cd PPE_Detection_Model
```

### Step 4: Create a Virtual Environment

A virtual environment isolates the project's dependencies from your system Python.

#### Windows
```bash
python -m venv venv
venv\Scripts\activate
```

#### Linux / macOS
```bash
python3 -m venv venv
source venv/bin/activate
```

After activation, your terminal prompt should be prefixed with `(venv)`.

### Step 5: Install Dependencies

With the virtual environment activated, install all required Python packages:

> **Note for GPU users:** By default, PyTorch installs as a CPU version. If you have an NVIDIA GPU and want CUDA acceleration, run the following command **before** running `pip install -r requirements.txt`:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
> ```

Now install the rest of the requirements:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 6: Verify Installation

The trained model weights (`best.onnx` and `best.pt`) are already included in the repository. 

Run a quick test to ensure everything is working. Start the FastAPI server:

```bash
python app.py
```

You should see output logs indicating that the model loaded successfully and the server is running. Open your web browser and navigate to the local host address shown in the terminal (usually `http://127.0.0.1:8000`) to access the web interface.

**Windows Users:** You can also simply double-click `run.bat` to start the detection automatically.

---

## ⚙️ Usage

To start the application:

```bash
python app.py
```

Once the server is running, open your web browser and go to:
```text
http://127.0.0.1:8000
```

From the web interface, you can:
- Start/Stop the detection feed.
- View real-time bounding boxes for PPE, fire, and smoke.
- Configure alert settings.

---

## 📊 Model Performance

| Class | Precision | Recall | mAP@0.5 |
| :--- | :--- | :--- | :--- |
| Helmet | 0.93 | 0.90 | 0.92 |
| Vest | 0.91 | 0.89 | 0.91 |
| Fire | 0.96 | 0.94 | 0.95 |
| Smoke | 0.89 | 0.86 | 0.87 |

*Trained using the Ultralytics framework (YOLOv11) on Google Colab with a custom dataset of 10,000+ annotated images.*

---

## 🛠️ Troubleshooting

| Problem | Solution |
| :--- | :--- |
| `ModuleNotFoundError: No module named 'cv2'` | Run `pip install opencv-python` inside your activated virtual environment. |
| `ImportError: libGL.so.1: cannot open shared object file` (Linux) | Install `libgl1` and `libglib2.0-0`: `sudo apt install libgl1 libglib2.0-0 -y` |
| Model fails to load | Ensure `best.onnx` or `best.pt` exists in the root directory. |
| Webcam not detected | Try a different camera index in the web interface. On Linux, ensure you have permission to access `/dev/video*`. |
| Slow inference on CPU | Use a GPU if available, or reduce the input resolution. |
| `CUDA out of memory` | Lower the batch size or run on CPU. |
| Port 8000 already in use | Run `python app.py --port 8001` (or another free port). |

---

## 🤝 Contributing

Contributions are welcome! If you have a dataset to improve the model or want to add a feature (like MQTT integration for IoT), please follow these steps:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.

---

## 📜 License

This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**. 

This is because the project utilizes the [Ultralytics](https://github.com/ultralytics/ultralytics) YOLOv11 framework, which is licensed under AGPL-3.0. Any derivative work must also be open-sourced under the same license.

See the `LICENSE` file for more information.

---

## 📧 Contact

For questions or support, please open an issue on GitHub.

Project Link: [https://github.com/Ashis7116/PPE_Detection_Model](https://github.com/Ashis7116/PPE_Detection_Model)

---

## ⚠️ Disclaimer

*This system is designed to assist safety protocols but should not replace human supervision or standard fire safety equipment. Always follow local safety regulations and use this tool as a supplementary measure.*
