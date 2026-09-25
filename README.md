# Spatial hypoxia signature guides adjuvant therapy in hepatocellular carcinoma after hepatectomy

## System Requirements

### Software

The software requires Python 3.11 and the following Python packages:

| Package         | Version |
| --------------- | ------: |
| Python          | 3.11.14 |
| NumPy           |   2.3.5 |
| pandas          |   2.3.3 |
| SciPy           |  1.16.3 |
| scikit-learn    |   1.7.2 |
| scikit-survival |  0.26.0 |
| lifelines       |  0.30.0 |
| Optuna          |   4.6.0 |

All required dependencies are also provided in `requirements.txt`.

### Operating system

The software has been tested on:

* Operating system: Windows 10

Other operating systems may work but have not been systematically tested.

## Installation

Clone the repository:

git clone https://github.com/MedAI-Lab/HCC-SHS.git
cd HCC-SHS

Install the required dependencies:

pip install -r requirements.txt

Installation typically takes approximately 1 minutes on a standard desktop computer with a stable internet connection. The actual installation time may vary depending on the operating system, network speed, and local Python environment.

## Demonstration

### Demonstration data

A small synthetic dataset is provided in:

data/

The demonstration dataset is provided to verify that the software can be executed and to illustrate the required input data format.

The synthetic demonstration dataset is **not intended to reproduce the results, selected features, model coefficients, or performance reported in the original study**.

No original patient-level research data are required to run the demonstration.

### Running the demonstration

After installing the required dependencies, run:

python scripts/run_pipeline.py

The analysis results are saved to:

outputs/

### Expected output

After successful execution, the `outputs/` directory will contain the results generated from the demonstration dataset.

For example:

outputs/
├── performance.csv
├── clinical_train_risk.csv.csv

The exact output files depend on the analysis configuration.

The demonstration results are intended to verify successful execution of the software and are not expected to reproduce the numerical results of the original study.

### Demonstration runtime

The complete demonstration typically takes approximately 15 minutes on a standard desktop computer using the tested environment.

## Instructions for Use

The software can be applied to user-provided spatial features and clinical data.

### Input data

Users should prepare their data according to the format illustrated by the files in:

data/

The input data should contain the spatial features and clinical variables required by the corresponding analysis.

Sample identifiers should be consistent across the spatial-feature and clinical data tables.

The demonstration data can be used as a template for the required data structure and variable format.

### Configuration

The main analysis configuration is defined in:

shsc/config.py

Users should modify the relevant input/output paths and analysis parameters according to their own dataset.

### Running the analysis on user data

After preparing the input data and configuration, run:

python scripts/run_pipeline.py

The generated results will be saved to the configured output directory.

For analyses intended to reproduce the procedures described in the manuscript, users should follow the preprocessing, feature harmonization, feature selection, model development, and evaluation procedures described in the manuscript.

## Repository Structure

HCC-SHS/
├── README.md
├── LICENSE
├── requirements.txt
│
├── shsc/
│   ├── config.py
│   ├── harmonize.py
│   ├── feature_selection.py
│   ├── clinical.py
│   ├── model.py
│   ├── evaluate.py
│   └── pipeline.py
│
├── scripts/
│   └── run_pipeline.py
│
├── data/
│
└── outputs/