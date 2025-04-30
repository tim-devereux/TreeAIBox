
## TreeAIBox CloudCompare Plugin

A CloudCompare Python plugin providing a unified web-style GUI for a suite of LiDAR processing modules targeting forest and tree analysis.


### 📖 Overview

TreeAIBox brings together four core LiDAR-processing workflows in a single GUI:

- **TreeFiltering**  
  Supervised deep-learning filtering to separate understory and overstory points .  
- **TreeisoNet**  
  End-to-end crown segmentation pipeline (StemCls → TreeLoc → TreeOff → CrownOff3D), with optional manual editing .  
- **WoodCls**  
  3D stem & branch classification on TLS data .  
- **QSM**  
  Plot-level skeletonization and export of tree structure to XML/OBJ .

---

### 🚀 Features

- **Pretrained model management**  
  Download from a remote server; cached under  
  `~/.local/share/CloudCompare/TreeAIBox/models` & `logs` .  
- **GPU acceleration toggle**  
- **Interactive parameter controls**  
- **Progress bar** and **notifications**  
- Results as scalar fields, child clouds, or exported files.

---

### 🛠️ Installation

#### 1. Via Windows Installer

A ready-to-run installer is provided:

1. Download or copy **TreeAIBox_Plugin_Installer.exe** into any folder.  
2. **Right-click → Run as administrator**.  
3. Follow the prompts; by default it installs into your CloudCompare folder (`%PROGRAMFILES%\CloudCompare`).  
4. The installer will:
   - Copy all Python scripts, UI files, images, and modules into  
     `…\CloudCompare\plugins\Python\Plugins\TreeAIBox\`  
   - Generate a helper batch to detect NVIDIA GPU and install the correct PyTorch wheel.  
   - Launch `pip` to install required Python packages (PyQt6, torch, requests, etc.).  

Once it finishes, restart CloudCompare and launch the plugin from the Python console.

> **Note:** The NSIS script (`CloudCompare_Python_Plugin.nsi`) can be edited if you need to customize install paths or package versions.

#### 2. Manual (Git + pip)

```bash
cd %PROGRAMFILES%\CloudCompare\plugins\Python\Plugins
git clone https://github.com/truebelief/cc-TreeAIBox-plugin-test TreeAIBox
pip install PyQt6 PyQt6-WebEngine numpy torch requests
```

---

### ▶️ Usage

In CloudCompare, register the TreeAIBox.py under Script Register, and click the TreeAIBox.

Then select a point cloud, pick your module tab, choose/download a model, adjust settings, and click **Apply**.

![TreeFiltering](https://github.com/user-attachments/assets/ee4f3558-6535-448b-8279-d0a4cff2158f)
![TreeIsoNet](https://github.com/user-attachments/assets/d82e3bf6-8db1-49a4-a7c2-134ce4760fec)
![WoodCls](https://github.com/user-attachments/assets/2cf1288e-d9e8-4cf8-8251-4e8e2dcd17ec)
![QSM](https://github.com/user-attachments/assets/aa1a0bc8-febe-41d1-8bdb-f952970b5017)


---

### ⚙️ Configuration

- **`model_zoo.json`** lists available model names.  
- Logs & outputs in `C:\Users\YOURNAME\AppData\Local\CloudCompare\TreeAIBox\`.

The table below summarizes the voxel resolution and GPU memory requirements of current AI models, categorized by sensor type, task, component, and scene:

| Sensor                           | Task           | Component           | Scene       | Resolution | VRAM  |
|:---------------------------------|:---------------|:--------------------|:------------|:-----------|:------|
| ALS (or UAV without stems)       | Classification | Vegetation layer    | Mountainous | 80 cm      | 3 GB  |
| ALS (or UAV without stems)       | Classification | Vegetation layer    | Regular     | 50 cm      | 3 GB  |
| ALS (or UAV without stems)       | Classification | Vegetation layer    | Wellsite    | 15 cm      | 3 GB  |
| UAV (with stems)                 | Classification | Vegetation layer    | Regular     | 12 cm      | 3 GB  |
| UAV (with stems)                 | Classification | Stems               | Mixedwood   | 8 cm       | 3 GB  |
| TLS                              | Classification | Vegetation layer    | Regular     | 8 cm       | 3 GB  |
| TLS                              | Classification | Stems               | Boreal      | 10 cm      | 3 GB  |
| TLS                              | Classification | Stems               | Boreal      | 4 cm       | 3 GB  |
| TLS                              | Classification | Stems               | Boreal      | 20 cm      | 8 GB  |
| TLS                              | Classification | Stems               | Regular     | 4 cm       | 12 GB |
| TLS                              | Classification | Stems + branches    | Regular     | 4 cm       | 2 GB  |
| TLS                              | Classification | Stems + branches    | Regular     | 2.5 cm     | 3 GB  |
| ALS (or UAV without stems)       | Clustering     | Tree tops           | Wellsite    | 10 cm      | 4 GB  |
| ALS (or UAV without stems)       | Clustering     | Tree segments       | Wellsite    | 10 cm      | 4 GB  |
| UAV (with stems)                 | Clustering     | Tree bases          | Mixedwood   | 10 cm      | 3 GB  |
| UAV (with stems)                 | Clustering     | Tree segments       | Mixedwood   | 15 cm      | 4 GB  |
| TLS                              | Clustering     | Tree bases          | Boreal      | 10 cm      | 3 GB  |
| TLS                              | Clustering     | Tree segments       | Boreal      | 15 cm      | 4 GB  |


---

### 🤝 Contributing

1. Fork → feature branch → PR.  
2. Follow existing style and add tests as needed.

### 📄 License

GNU GPL v3 (same as CloudCompare).

---
![image](https://github.com/user-attachments/assets/2cac174d-f874-4a4a-bc4d-93c6ee9d4905)


