# Microscopy Image Processing Pipeline

## Image Registration Scripts - Installation and Usage

### Source and Author Information
- **Modified script:** `ashlar_processing.py`  
- **Original source:** [Färkkilä Lab - Ashlar Workflow](https://github.com/farkkilab/image_processing/blob/main/pipeline/1_stitching/ashlar_workflow.py)  
- **Author:** Cruz Osuna ([cruzosuna2003@gmail.com](mailto:cruzosuna2003@gmail.com))

---

## Environment Setup

### 1. Create Conda Environment
```bash
conda env create -f image_registration.yml
```

### 2. Activate Environment
```bash
conda activate image_registration
```

### 3. Verify Installation
Ensure Ashlar is available in your PATH:
```bash
which ashlar
```

---

## Configuration

### Path Setup (Required)
Edit the following paths in `stitching.py`:

```python
input_path = "/path/to/01_Registration/RCPNL/"           # Raw microscopy files  
output_path = "/path/to/02_Visualization/Images_IC/"     # Processed OME-TIFFs  
illumination_base = "/path/to/00_Illumination/IC_files"  # Correction profiles
```

---

## Expected Folder Structure

```
00_Illumination/IC_files/
├─ Sample1/
│   ├─ file1-ffp.tif
│   └─ file1-dfp.tif

01_Registration/RCPNL/
├─ Sample1/
│   └─ file1.rcpnl

02_Visualization/Images_IC/
└─ Sample1.ome.tif
```

---

## Usage

### Basic Processing
```bash
python stitching.py -c 8
```

- `-c` / `--threads`: Number of parallel processes (default: 4)

### Force Reprocessing
```bash
python stitching.py -c 8 -f
```

- `-f` / `--force`: Overwrite existing output files

---

## Key Parameters

- **Illumination Correction:** Automatically applied when matching FFP/DFP files are found  
- **CZI Handling:** Automatic channel alignment and pyramid generation  
- **Memory:** ~4–6 GB RAM per thread (increase for large datasets)

---

## Illumination Correction Setup

### File Naming Convention

| Raw File                | Illumination Files                  |
|-------------------------|-------------------------------------|
| `CYCLE1_SampleA.rcpnl`  | `CYCLE1_SampleA-ffp.tif`            |
|                         | `CYCLE1_SampleA-dfp.tif`            |

- FFP/DFP files must be in matching subfolders under `IC_files`  
- File base names must match exactly  
- TIFF format required for correction profiles

---

## Monitoring & Troubleshooting

### Expected Output
- OME-TIFF files with pyramid structure  
- Metadata preservation from original files  
- Channel-aligned images (for CZI inputs)

### Common Issues
- **Missing illumination files:** Script will warn and process without correction  
- **Path mismatches:** Ensure your folder structure matches the configuration  
- **Memory errors:** Reduce thread count (`-c 4`)

### Progress Tracking
- Real-time progress bar using `tqdm`  
- Detailed logs for each processed sample  
- Error messages with command output included
