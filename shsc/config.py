from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


ID_COL = "filename"
CENTER_COL = "center"
BATCH_COL = "batch"

CENTER_BATCH_COL = "center_batch"
#影像表格
EVENT_COL = "recurrence"          
TIME_COL = "time"                 
AGE_COL = "age"
GENDER_COL = "gender"

# 临床表格
SLIDE_COL = "slide"
CLIN_EVENT_COL = "event"
CLIN_TIME_COL = "survival_months"

#临床特征
CLINICAL_CANDIDATES: List[str] = [
    "Albumin",
    "Tumor diameter",
    "Total bilirubin",
    "AFP",
    "ALBI score",
    "PLT",
    "MVI",
    "AST",
    "HBV-DNA",
    "Edmondson-Steiner Grade",
    "Tumor number",
    "Cirrhosis",
    "Age",
    "Viral hepatitis",
    "Child-Pugh",
    "Satellite nodules",
    "Sex",
    "ALT",
]


L1_COX_PENALIZER = 0.05          
L1_COX_L1_RATIO = 1.0
L1_COEFF_EPS = 1e-6              
RFE_CV_SPLITS = 5                
RFE_MIN_FEATURES = 2             
KS_THRESHOLD = 0.25              
WASSERSTEIN_THRESHOLD = 0.5      


SEED = 42
N_BOOT = 1500                    
ALPHA = 0.05                    

#optuna
N_TRIALS = 80
N_STARTUP_TRIALS = 15



@dataclass
class Paths:
    root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1])
    train_csv: Path = field(default=None)
    internal_csv: Path = field(default=None)
    external_csv: Path = field(default=None)
    clinical_csv: Path = field(default=None)
    output_dir: Path = field(default=None)

    def __post_init__(self) -> None:
        data = self.root / "data"
        if self.train_csv is None:
            self.train_csv = data / "demo_train.csv"
        if self.internal_csv is None:
            self.internal_csv = data / "demo_internal_val.csv"
        if self.external_csv is None:
            self.external_csv = data / "demo_external_test.csv"
        if self.clinical_csv is None:
            self.clinical_csv = data / "demo_clinical.csv"
        if self.output_dir is None:
            self.output_dir = self.root / "outputs"
