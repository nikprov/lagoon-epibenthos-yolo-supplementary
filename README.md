"# lagoon-epibenthos-yolo-supplementary" 

# Lagoon Epibenthos YOLO (lagoon-epibenthos-yolo)

[![Python 3.14.3](https://img.shields.io/badge/python-3.14.3-blue.svg)](https://www.python.org/downloads/)
[![Ultralytics YOLOv11](https://img.shields.io/badge/YOLO-v11-orange.svg)](https://github.com/ultralytics/ultralytics)

Welcome to the official code repository for the paper:  
**"Implementing Optimized Computer Vision Algorithm To Underwater Imagery For Identification and Spatial Analysis Of Epibenthic Fauna In Shallow Lagoon Waters"**

## 📖 About the repository
This repository contains the supplementary code, datasets, and model used to detect and analyze epibenthic fauna (*Paranemonia* sp., *Anemonia* sp., and *Brachyura* sp.) in the highly turbid, shallow waters of Logarou Lagoon. 

While the custom-trained \(for the above 3 classes\) YOLOv11s model is provided along with the exported metric files upon the completion of the training, we have also included the ready-to-use extracted data `.xlsx` files so readers can quickly reproduce the statistical tests and validate our findings.

---

## ⚙️ Prerequisites for reviewers

To run these scripts, you will need to install Python and a code editor. We recommend **Visual Studio Code (VS Code)**.

### 1. Install Python
1. Download **Python 3.14.3** (or newer) from the [official Python website](https://www.python.org/downloads/). We highly recommend installing the latest stable release for optimal security.
2. Run the installer. 
3. **CRITICAL:** During installation, ensure you check the box that says **"Add Python to PATH"** before clicking Install.

### 2. Install Visual Studio Code (IDE)
1. Download VS Code from [code.visualstudio.com](https://code.visualstudio.com/).
2. Install the software with default settings.
3. Open VS Code, go to the Extensions tab (`Ctrl+Shift+X`), and install the official **Python extension** (provided by Microsoft).

---

## 🛠️ Installation & Setup

**Step 1: Clone or Download the Repository**
Download this repository as a `.zip` file and extract it, or clone it via git:
```bash
git clone https://github.com/yourusername/lagoon-epibenthos-yolo.git
cd lagoon-epibenthos-yolo
```

**Step 2: Open the Folder in VS Code**
Open VS Code, click `File > Open Folder...` and select the extracted `lagoon-epibenthos-yolo` folder. Open a new terminal in VS Code by clicking `Terminal > New Terminal` from the top menu.

**Step 3: Create a Virtual Environment (Recommended)**
In the terminal, create and activate a virtual environment to avoid package conflicts:
```bash
# For Windows:
python -m venv venv
.\venv\Scripts\activate

# For macOS/Linux:
python3 -m venv venv
source venv/bin/activate
```

**Step 4: Install Requirements**
With your virtual environment activated, install the required packages (Ultralytics, OpenCV, Pandas, SciPy, etc.):
```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Pipeline

All sample data and pre-extracted results are located in the `data-files/` folder.

### 1. Run the Image Enhancement Script
This script applies the 3-stage radiometric correction (Adaptive Gray-World correction, CLAHE, and Histogram Stretching) to the raw turbid sample images.
```bash
python scripts/CLAHE_underwater_preprocessor_github_v2.py
```
Run and follow the console instructions.

### 2. Run the Statistical Analysis 
To allow readers to quickly view the study's core results, we provide the ready-made extracted data file (`YOLO_detection_results.xlsx`). This script generates the $\chi^2$ habitat selectivity summaries, size distribution metrics, and Mann-Whitney U test results.
```bash
python scripts/unified_statistical_analysis_github_v2.py
```
Again, run and follow the console instructions.


### 3. Run YOLOv11s Inference (Optional)
If you wish to test the object detection itself, you can run the custom-trained YOLO model on the enhanced images to detect the epibenthic fauna. The custom weights (`best.pt`) are located in the `models/` directory. Inference-ready scripts are provided in the `scripts/For-inference/ ` directory. Before running open in VScode and adjust properly.
```bash
python scripts/For-inference/YOLO_on_pics_to_table_and_annot.py
```



---

## 📬 Contact & Citation
If you use this code \(or logic\) in the methodological sequence of your research, please cite our paper (citation details pending publication)! For code-related issues, please open an issue on GitHub.