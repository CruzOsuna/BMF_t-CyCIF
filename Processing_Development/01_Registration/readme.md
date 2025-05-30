## Image Registration Scripts - Installation and Usage

### Source and Author Information
- **Modified from:** [Färkkilä Lab - Ashlar Workflow](https://github.com/farkkilab/image_processing/blob/main/pipeline/1_stitching/ashlar_workflow.py)
- **Author:** Cruz Osuna (cruzosuna2003@gmail.com)

---

## Creating the Conda Environment

To set up the required Conda environment, run the following command:
```bash
conda env create -f image_registration.yml
```

---

## Running the Stitching Script

### 1. Activate the Conda Environment
Before running the script, activate the Conda environment:
```bash
conda activate image_registration
```

### 2. Verify and Modify the Script Configuration
Ensure that the correct paths are specified in `stitching.py` before execution.

### 3. Execute the Script
Run the stitching script using:
```bash
python stitching.py -c <n>
```
Replace `<n>` with the desired number of images you want to be processed at the same time in parallel. If no value is specified, the script defaults to 4 threads.

Note: Processing each image in parallel requires approximately 4 to 6 GB of RAM, but it could be more in case of large images.

---

These instructions ensure a smooth setup and execution of the image registration scripts. Modify the paths as needed for your specific setup.

