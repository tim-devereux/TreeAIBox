
## TreeAIBox CloudCompare Plugin

A PyQt6-based CloudCompare plugin providing a unified web-style GUI for a suite of LiDAR processing modules targeting forest and tree analysis.


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
- **`model_server_path`** in `TreeAIBox.py` points at your model hosting URL.  
- Logs & outputs in `~/.local/share/CloudCompare/TreeAIBox/`.

---

### 🤝 Contributing

1. Fork → feature branch → PR.  
2. Follow existing style and add tests as needed.

---

### 📄 License

GNU GPL v3 (same as CloudCompare).

---

*Built & maintained by Zhouxin Xi & collaborators.*  
```
