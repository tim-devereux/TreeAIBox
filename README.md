
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
  Automatically installs required packages and registers the main script as a Python plugin.
---

### 🛠️ Installation

#### 1. Via Windows installer (Suggested)

A ready-to-run online installer is provided. Ensure **internet access** is enabled:

1. Download or copy **TreeAIBox_Plugin_Installer.exe** into any folder.  
2. **Right-click → Run as administrator** (suggested).  
3. Follow the prompts; by default it detects your CloudCompare folder (e.g., `%PROGRAMFILES%\CloudCompare`) based on the registry.  
4. The installer will:
   - Copy all Python scripts, UI files, images, and modules into  
     `…\CloudCompare\plugins\Python\Plugins\TreeAIBox\`  
   - Generate a helper batch to detect NVIDIA GPU and install the correct PyTorch wheel. (Please keep patient)
   - Launch `pip` to install required Python packages (PyQt6, torch, requests, etc.).  

Once it finishes, restart CloudCompare and launch the plugin from the Python console.

> **Note:** The NSIS script (`CloudCompare_Python_Plugin.nsi`) can be edited if you need to customize install paths or package versions.

#### 2. Alternatively, manual (Git + pip)

```bash
cd %PROGRAMFILES%\CloudCompare\plugins\Python\Plugins
git clone https://github.com/truebelief/cc-TreeAIBox-plugin-test TreeAIBox
pip install PyQt6 PyQt6-WebEngine numpy torch requests
```
In CloudCompare, register the TreeAIBox.py by clicking the Add Script button under the Script Register menu.

---

### ▶️ Usage

In CloudCompare, under Script Register, click the TreeAIBox.

![image](https://github.com/user-attachments/assets/f76865eb-ba91-4516-85b7-42c08d10ccb5)

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

<table>
  <thead>
    <tr>
      <th align="center">Sensor</th>
      <th align="center">Task</th>
      <th align="center">Component</th>
      <th align="center">Scene</th>
      <th align="center">Resolution</th>
      <th align="center">VRAM</th>
    </tr>
  </thead>
  <tbody>
    <!-- ALS Classification -->
    <tr>
      <td align="center" rowspan="3"><strong>ALS (or UAV without stems)</strong></td>
      <td align="center" rowspan="3">Classification</td>
      <td align="center" rowspan="3">Vegetation layer</td>
      <td align="center">Mountainous</td>
      <td align="center">80 cm</td>
      <td align="center">3 GB</td>
    </tr>
    <tr>
      <td align="center">Regular</td>
      <td align="center">50 cm</td>
      <td align="center">3 GB</td>
    </tr>
    <tr>
      <td align="center">Wellsite</td>
      <td align="center">15 cm</td>
      <td align="center">3 GB</td>
    </tr>
    <!-- UAV Classification -->
    <tr>
      <td align="center" rowspan="2"><strong>UAV (with stems)</strong></td>
      <td align="center" rowspan="2">Classification</td>
      <td align="center">Vegetation layer</td>
      <td align="center">Regular</td>
      <td align="center">12 cm</td>
      <td align="center">3 GB</td>
    </tr>
    <tr>
      <td align="center">Stems</td>
      <td align="center">Mixedwood</td>
      <td align="center">8 cm</td>
      <td align="center">3 GB</td>
    </tr>
    <!-- TLS Classification -->
    <tr>
      <td align="center" rowspan="7"><strong>TLS</strong></td>
      <td align="center" rowspan="7">Classification</td>
      <td align="center">Vegetation layer</td>
      <td align="center">Regular</td>
      <td align="center">8 cm</td>
      <td align="center">3 GB</td>
    </tr>
    <tr>
      <td align="center" rowspan="3">Stems</td>
      <td align="center" rowspan="3">Boreal</td>
      <td align="center">10 cm</td>
      <td align="center">3 GB</td>
    </tr>
    <tr>
      <td align="center">4 cm</td>
      <td align="center">3 GB</td>
    </tr>
    <tr>
      <td align="center">20 cm</td>
      <td align="center">8 GB</td>
    </tr>
    <tr>
      <td align="center">Stems</td>
      <td align="center">Regular</td>
      <td align="center">4 cm</td>
      <td align="center">12 GB</td>
    </tr>
    <tr>
      <td align="center" rowspan="2">Stems + branches</td>
      <td align="center" rowspan="2">Regular</td>
      <td align="center">4 cm</td>
      <td align="center">2 GB</td>
    </tr>
    <tr>
      <td align="center">2.5 cm</td>
      <td align="center">3 GB</td>
    </tr>
    <!-- ALS Clustering -->
    <tr>
      <td align="center" rowspan="2"><strong>ALS (or UAV without stems)</strong></td>
      <td align="center" rowspan="2">Clustering</td>
      <td align="center">Tree tops</td>
      <td align="center">Wellsite</td>
      <td align="center">10 cm</td>
      <td align="center">4 GB</td>
    </tr>
    <tr>
      <td align="center">Tree segments</td>
      <td align="center">Wellsite</td>
      <td align="center">10 cm</td>
      <td align="center">4 GB</td>
    </tr>
    <!-- UAV Clustering -->
    <tr>
      <td align="center" rowspan="2"><strong>UAV (with stems)</strong></td>
      <td align="center" rowspan="2">Clustering</td>
      <td align="center">Tree bases</td>
      <td align="center">Mixedwood</td>
      <td align="center">10 cm</td>
      <td align="center">3 GB</td>
    </tr>
    <tr>
      <td align="center">Tree segments</td>
      <td align="center">Mixedwood</td>
      <td align="center">15 cm</td>
      <td align="center">4 GB</td>
    </tr>
    <!-- TLS Clustering -->
    <tr>
      <td align="center" rowspan="2"><strong>TLS</strong></td>
      <td align="center" rowspan="2">Clustering</td>
      <td align="center">Tree bases</td>
      <td align="center">Boreal</td>
      <td align="center">10 cm</td>
      <td align="center">3 GB</td>
    </tr>
    <tr>
      <td align="center">Tree segments</td>
      <td align="center">Boreal</td>
      <td align="center">15 cm</td>
      <td align="center">4 GB</td>
    </tr>
  </tbody>
</table>

---

### Folder structure

```
TreeAIBox-main
│   TreeAIBox_Plugin_Installer.exe                                  # Windows installer for the plugin
│   CloudCompare_Python_Plugin.nsi                                  # Configuration of the plugin installer
│   treeaibox-header.jpg                                            # Installer icon
│   treeaibox-welcome.jpg                                           # Installer icon
│   dl_visualization.svg                                            # The main DL network structure illustration
│   LICENSE.txt                                                     # License file
│   model_zoo.json                                                  # List of available trained DL model file names
│   README.md                                                       # README
│   TODO.md                                                         # To-do list
│   TreeAIBox.py                                                    # Main python program of TreeAIBox
│   treeaibox_logo.ico                                              # Plugin logo
│   treeaibox_ui.html                                               # Main GUI (web view)
├───img                                                             # Icons and images used by the plugin GUI
└───modules                                                         # Submodules of TreeAIBox
    ├───filter                                                      # TreeFiltering and WoodCls modules
    │       componentFilter.py                                      # Functions of filtering tree layer, branch, and stem components
    │       createDTM.py                                            # Functions of creating DTM grid points based on the filtered tree and ground layers
    │       *.json                                                  # Definition of DL model parameters
    │       vox3DESegFormer.py                                      # DL model structure (version 2)
    │       vox3DSegFormer.py                                       # DL model structure (version 1)
    │       __init__.py
    │
    ├───qsm                                                         # QSM module
    │       applyQSM.py                                             # Functions of skeletonizing and reconstructing 3D tree geometries
    │       __init__.py
    │
    └───treeisonet
            cleanSmallerClusters.py                                 # Functions of roughly removing small clusters based on point number
            treeLoc.py                                              # Functions of extracing stem locations or tree tops
            treeOff.py                                              # Functions of clustering crown points to the extracted tree tops (for the stem-invisible scene)
            stemCluster.py                                          # Functions of clustering stem points to the stem locations based on the shortest path rule
            crownCluster.py                                         # Functions of clustering crown points to the segmented stem points based on the shortest path rule
            crownOff.py                                             # Functions of clustering crown points to the segmented stem points based on deep learning clustering
            treeStat.py                                             # Functions of extracting individual-tree and plot-level statistics
            *.json                                                  # Definition of DL model parameters
            vox3DSegFormerDetection.py                              # DL model structure of tree location detection
            vox3DSegFormerRegression.py                             # DL model structure of point offset regression
            __init__.py
```
---

### 🤝 Contributing

1. Fork → feature branch → PR.  
2. Follow existing style and add tests as needed.

### 📄 License

GNU GPL v3 (same as CloudCompare).

Developed by Zhouxin Xi, tested by Charumitha Selvaraj

---
*Born from over a decade of LiDAR research with support from dedicated collaborators.*

![image](https://github.com/user-attachments/assets/2cac174d-f874-4a4a-bc4d-93c6ee9d4905)


