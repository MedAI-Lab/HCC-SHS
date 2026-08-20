## 🛠️ Scripts Overview

This repository contains a pipeline for WSI patch processing, JSON filtering, and spatial feature extraction. Below is a brief description of each script:

*   **`process_patch_wsi.py`**
    A script for calculating JSON files using DeepLIIF. 
    > ⚠️ **Note:** This script must be placed in the **root directory** of the DeepLIIF project. It requires WSI patch `.h5` files (segmented by CLAM) as input for block-wise map calculation.

*   **`json_edi.py`**
    A JSON editing tool based on outputs from HoverNet or DeepLIIF. It is designed to filter out unwanted cell types (e.g., IHC-negative cells) from the generated JSON files.

*   **`json_combine.py`**
    A utility script to merge multiple processed cell JSON files into a single file, consolidating target cells such as IHC-positive cells and tumor cells.

*   **`spatial analysis_all.py`**
    A comprehensive spatial analysis script. It extracts various spatial features based on the merged JSON files.