
## TreeAIBox CloudCompare Plugin

A CloudCompare Python plugin providing a unified web-style GUI for a suite of LiDAR processing modules targeting forest and tree analysis.


### 📖 Overview

TreeAIBox brings together four core LiDAR-processing workflows in a single GUI:

- **TreeFiltering**  
  Supervised deep-learning filtering to separate understory and overstory points.
  
- **TreeisoNet**  
  End-to-end crown segmentation pipeline (StemCls → TreeLoc → TreeOff → CrownOff3D), allowing manual editing.
  
- **WoodCls**  
  3D stem & branch classification on TLS data.
  
- **QSM**  
  Plot-level skeletonization and export of tree structure to XML/OBJ.

---

### 🚀 Features

- **20+ pretrained AI models**  
  Downloadable from a remote server; mostly lightweight or distilled versions, fine-tuned with carefully annotated datasets.
  
- **3D targeted**  
  Operates directly on raw 3D point clouds—no CHM or raster inputs—using voxel-based AI architectures for both training and inference.
  
- **Sensor, scene, and resolution options**  
  Supports TLS, ALS, and UAV LiDAR across boreal, mixedwood, and reclamation forest types.
  
- **GPU acceleration toggle**  
  Runs on either GPU (CUDA) or CPU for flexibility.
  
- **Web-style UI framework**  
  Features resizable windows and modular UI components.
  
- **Interactive parameter controls**  
  Allow certain result customization using adjustable parameters.
  
- **Open source**  
  Fully Python-based (except for pretrained model files); outputs include scalar fields, point clouds, and exportable files.
  
- **Windows installer**  
  Automatically installs required packages.
---

### 🛠️ Installation

#### 1. Via Windows Installer

A ready-to-run online installer is provided. Ensure **internet access** is enabled:

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
- Logs & outputs in `C:\Users\USERNAME\AppData\Local\CloudCompare\TreeAIBox\`.

The table below summarizes the voxel resolution and GPU memory used by the current AI models, categorized by sensor type, task, component, and scene:

<table border="1" cellpadding="5" cellspacing="0" style="border-collapse: collapse; border: 1px solid #ddd;">
  <thead>
    <tr style="background-color: #f2f2f2;">
      <th align="center" style="border: 1px solid #ddd;">Sensor</th>
      <th align="center" style="border: 1px solid #ddd;">Task</th>
      <th align="center" style="border: 1px solid #ddd;">Component</th>
      <th align="center" style="border: 1px solid #ddd;">Scene</th>
      <th align="center" style="border: 1px solid #ddd;">Resolution</th>
      <th align="center" style="border: 1px solid #ddd;">VRAM</th>
    </tr>
  </thead>
  <tbody>
    <!-- ALS Classification -->
    <tr style="background-color: #f8f9fa;">
      <td align="center" rowspan="3" style="border: 1px solid #ddd;">ALS (or UAV without stems)</td>
      <td align="center" rowspan="3" style="border: 1px solid #ddd;">Classification</td>
      <td align="center" rowspan="3" style="border: 1px solid #ddd;">Vegetation layer</td>
      <td align="center" style="border: 1px solid #ddd;">Mountainous</td>
      <td align="center" style="border: 1px solid #ddd;">80 cm</td>
      <td align="center" style="border: 1px solid #ddd;">3 GB</td>
    </tr>
    <tr style="background-color: #f8f9fa;">
      <td align="center" style="border: 1px solid #ddd;">Regular</td>
      <td align="center" style="border: 1px solid #ddd;">50 cm</td>
      <td align="center" style="border: 1px solid #ddd;">3 GB</td>
    </tr>
    <tr style="background-color: #f8f9fa;">
      <td align="center" style="border: 1px solid #ddd;">Wellsite</td>
      <td align="center" style="border: 1px solid #ddd;">15 cm</td>
      <td align="center" style="border: 1px solid #ddd;">3 GB</td>
    </tr>
    <!-- UAV Classification -->
    <tr style="background-color: #e8f4f9;">
      <td align="center" rowspan="2" style="border: 1px solid #ddd;">UAV (with stems)</td>
      <td align="center" rowspan="2" style="border: 1px solid #ddd;">Classification</td>
      <td align="center" style="border: 1px solid #ddd;">Vegetation layer</td>
      <td align="center" style="border: 1px solid #ddd;">Regular</td>
      <td align="center" style="border: 1px solid #ddd;">12 cm</td>
      <td align="center" style="border: 1px solid #ddd;">3 GB</td>
    </tr>
    <tr style="background-color: #e8f4f9;">
      <td align="center" style="border: 1px solid #ddd;">Stems</td>
      <td align="center" style="border: 1px solid #ddd;">Mixedwood</td>
      <td align="center" style="border: 1px solid #ddd;">8 cm</td>
      <td align="center" style="border: 1px solid #ddd;">3 GB</td>
    </tr>
    <!-- TLS Classification -->
    <tr style="background-color: #f0ebf8;">
      <td align="center" rowspan="7" style="border: 1px solid #ddd;">TLS</td>
      <td align="center" rowspan="7" style="border: 1px solid #ddd;">Classification</td>
      <td align="center" style="border: 1px solid #ddd;">Vegetation layer</td>
      <td align="center" style="border: 1px solid #ddd;">Regular</td>
      <td align="center" style="border: 1px solid #ddd;">8 cm</td>
      <td align="center" style="border: 1px solid #ddd;">3 GB</td>
    </tr>
    <tr style="background-color: #f0ebf8;">
      <td align="center" rowspan="3" style="border: 1px solid #ddd;">Stems</td>
      <td align="center" rowspan="3" style="border: 1px solid #ddd;">Boreal</td>
      <td align="center" style="border: 1px solid #ddd;">10 cm</td>
      <td align="center" style="border: 1px solid #ddd;">3 GB</td>
    </tr>
    <tr style="background-color: #f0ebf8;">
      <td align="center" style="border: 1px solid #ddd;">4 cm</td>
      <td align="center" style="border: 1px solid #ddd;">3 GB</td>
    </tr>
    <tr style="background-color: #f0ebf8;">
      <td align="center" style="border: 1px solid #ddd;">20 cm</td>
      <td align="center" style="border: 1px solid #ddd;">8 GB</td>
    </tr>
    <tr style="background-color: #f0ebf8;">
      <td align="center" style="border: 1px solid #ddd;">Stems</td>
      <td align="center" style="border: 1px solid #ddd;">Regular</td>
      <td align="center" style="border: 1px solid #ddd;">4 cm</td>
      <td align="center" style="border: 1px solid #ddd;">12 GB</td>
    </tr>
    <tr style="background-color: #f0ebf8;">
      <td align="center" rowspan="2" style="border: 1px solid #ddd;">Stems + branches</td>
      <td align="center" rowspan="2" style="border: 1px solid #ddd;">Regular</td>
      <td align="center" style="border: 1px solid #ddd;">4 cm</td>
      <td align="center" style="border: 1px solid #ddd;">2 GB</td>
    </tr>
    <tr style="background-color: #f0ebf8;">
      <td align="center" style="border: 1px solid #ddd;">2.5 cm</td>
      <td align="center" style="border: 1px solid #ddd;">3 GB</td>
    </tr>
    <!-- ALS Clustering -->
    <tr style="background-color: #f8f9fa;">
      <td align="center" rowspan="2" style="border: 1px solid #ddd;">ALS (or UAV without stems)</td>
      <td align="center" rowspan="2" style="border: 1px solid #ddd;">Clustering</td>
      <td align="center" style="border: 1px solid #ddd;">Tree tops</td>
      <td align="center" style="border: 1px solid #ddd;">Wellsite</td>
      <td align="center" style="border: 1px solid #ddd;">10 cm</td>
      <td align="center" style="border: 1px solid #ddd;">4 GB</td>
    </tr>
    <tr style="background-color: #f8f9fa;">
      <td align="center" style="border: 1px solid #ddd;">Tree segments</td>
      <td align="center" style="border: 1px solid #ddd;">Wellsite</td>
      <td align="center" style="border: 1px solid #ddd;">10 cm</td>
      <td align="center" style="border: 1px solid #ddd;">4 GB</td>
    </tr>
    <!-- UAV Clustering -->
    <tr style="background-color: #e8f4f9;">
      <td align="center" rowspan="2" style="border: 1px solid #ddd;">UAV (with stems)</td>
      <td align="center" rowspan="2" style="border: 1px solid #ddd;">Clustering</td>
      <td align="center" style="border: 1px solid #ddd;">Tree bases</td>
      <td align="center" style="border: 1px solid #ddd;">Mixedwood</td>
      <td align="center" style="border: 1px solid #ddd;">10 cm</td>
      <td align="center" style="border: 1px solid #ddd;">3 GB</td>
    </tr>
    <tr style="background-color: #e8f4f9;">
      <td align="center" style="border: 1px solid #ddd;">Tree segments</td>
      <td align="center" style="border: 1px solid #ddd;">Mixedwood</td>
      <td align="center" style="border: 1px solid #ddd;">15 cm</td>
      <td align="center" style="border: 1px solid #ddd;">4 GB</td>
    </tr>
    <!-- TLS Clustering -->
    <tr style="background-color: #f0ebf8;">
      <td align="center" rowspan="2" style="border: 1px solid #ddd;">TLS</td>
      <td align="center" rowspan="2" style="border: 1px solid #ddd;">Clustering</td>
      <td align="center" style="border: 1px solid #ddd;">Tree bases</td>
      <td align="center" style="border: 1px solid #ddd;">Boreal</td>
      <td align="center" style="border: 1px solid #ddd;">10 cm</td>
      <td align="center" style="border: 1px solid #ddd;">3 GB</td>
    </tr>
    <tr style="background-color: #f0ebf8;">
      <td align="center" style="border: 1px solid #ddd;">Tree segments</td>
      <td align="center" style="border: 1px solid #ddd;">Boreal</td>
      <td align="center" style="border: 1px solid #ddd;">15 cm</td>
      <td align="center" style="border: 1px solid #ddd;">4 GB</td>
    </tr>
  </tbody>
</table>

---

### 🤝 Contributing

1. Fork → feature branch → PR.  
2. Follow existing style and add tests as needed.

### 📄 License

GNU GPL v3 (same as CloudCompare).

---
![image](https://github.com/user-attachments/assets/2cac174d-f874-4a4a-bc4d-93c6ee9d4905)


