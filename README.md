# iFeature for Antimicrobial Peptide Analysis

A practical guide and toolkit for using iFeature to extract sequence-based features from antimicrobial peptides (AMPs) for machine learning applications.

[![Python Version](https://img.shields.io/badge/python-3.7%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![iFeature](https://img.shields.io/badge/based%20on-iFeature-orange)](https://github.com/Superzchen/iFeature)

---

## Overview

This repository provides a comprehensive workflow for analyzing antimicrobial peptides using **iFeature** - a Python toolkit for extracting protein sequence features. This is a tutorial and analysis framework built on top of the original iFeature package.

### What This Repository Offers

-  Step-by-step guide for AMP feature extraction
-  Ready-to-use Python scripts for ML pipelines
-  Sample datasets and example analyses
-  Jupyter notebooks with complete workflows
-  Visualization tools for feature analysis

### Applications

- AMP activity prediction (antibacterial, antifungal, antiviral)
- Peptide toxicity assessment
- Feature engineering for deep learning models
- Virtual screening of peptide libraries

---

## Quick Start

### 1. Clone This Repository

```bash
git clone https://github.com/YourBioinfo_code_analytics/iFeature-AMP-Analysis_tutorial.git
cd iFeature-AMP-Analysis_tutorial
```

### 2. Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 3. Clone iFeature

```bash
# Clone the original iFeature repository
git clone https://github.com/Superzchen/iFeature.git
```

### 4. Run Example Analysis

```bash
python scripts/extract_features.py
python scripts/train_classifier.py
```

---

## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Requirements

Create a `requirements.txt` file:

```txt
numpy>=1.19.0
pandas>=1.1.0
scikit-learn>=0.23.0
biopython>=1.78
matplotlib>=3.3.0
seaborn>=0.11.0
scipy>=1.5.0
```

Install:
```bash
pip install -r requirements.txt
```

---

## Repository Structure

```
iFeature-AMP-Analysis_tutorial/
│
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── LICENSE                        # MIT License
│
├── data/                          # Sample datasets
│   ├── sample_positive_AMPs.fasta
│   ├── sample_negative_AMPs.fasta
│   └── combined_dataset.fasta
│
├── scripts/                       # Analysis scripts
│   ├── extract_features.py       # Feature extraction
│   ├── train_classifier.py       # ML model training
│   └── visualize_features.py     # Feature visualization
│
├── results/                       # Output directory
│   ├── features/
│   ├── models/
│   └── plots/
│
└── iFeature/                      # Clone original iFeature here
    └── (git clone from original repo)
```

---

## Feature Extraction

### Available Feature Types

| Feature | Dimensions | Description | Use Case |
|---------|-----------|-------------|----------|
| **AAC** | 20 | Amino acid composition | Basic composition analysis |
| **DPC** | 400 | Dipeptide composition | Local sequence patterns |
| **TPC** | 8000 | Tripeptide composition | Detailed patterns (high-dim) |
| **CTD** | 147 | Composition-Transition-Distribution | Physicochemical properties |
| **PAAC** | 20+λ | Pseudo amino acid composition | Sequence order + composition |
| **APAAC** | 20+λ | Amphiphilic PAAC | Membrane interaction |
| **CTriad** | 343 | Conjoint triad | Protein-protein interaction |

### Example Usage

```python
import sys
sys.path.append('./iFeature')
from iFeature import AAC, DPC, CTD, readFasta
import pandas as pd

# Read sequences
sequences = readFasta('data/sample_positive_AMPs.fasta')

# Extract features
aac_features = AAC(sequences)
dpc_features = DPC(sequences)
ctd_features = CTD(sequences)

# Save to CSV
pd.DataFrame(aac_features).to_csv('results/features/AAC_features.csv', index=False)
```

---

## Machine Learning Pipeline

### Complete Classification Example

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

# Load features
df = pd.read_csv('results/features/AAC_features.csv')
X = df.iloc[:, 1:].values  # Features
y = df.iloc[:, 0].values   # Labels

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Normalize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_scaled, y_train)

# Evaluate
y_pred = clf.predict(X_test_scaled)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"AUC-ROC: {roc_auc_score(y_test, clf.predict_proba(X_test_scaled)[:, 1]):.4f}")
```

---

## Scripts Overview

### 1. `extract_features.py`
Extracts AAC, DPC, CTD, and PAAC features from FASTA files.

**Usage:**
```bash
python scripts/extract_features.py
```

### 2. `train_classifier.py`
Trains Random Forest and SVM classifiers with cross-validation.

**Usage:**
```bash
python scripts/train_classifier.py
```

### 3. `visualize_features.py`
Creates PCA plots and feature distribution visualizations.

**Usage:**
```bash
python scripts/visualize_features.py
```

---

## AMP Databases

Collect AMP sequences from these public databases:

- **[DBAASP](https://dbaasp.org/)** - Database of Antimicrobial Activity and Structure of Peptides
- **[APD3](https://aps.unmc.edu/)** - Antimicrobial Peptide Database
- **[CAMP](http://www.camp.bicnirrh.res.in/)** - Collection of Anti-Microbial Peptides
- **[DRAMP](http://dramp.cpu-bioinfor.org/)** - Data Repository of Antimicrobial Peptides

---

## Troubleshooting

### Common Issues

**1. Module not found error:**
```bash
# Make sure iFeature is cloned in the correct location
git clone https://github.com/Superzchen/iFeature.git
# Update path in scripts if needed
```

**2. Invalid amino acid characters:**
```python
# Clean sequences before extraction
def clean_sequence(seq):
    standard_aa = set('ACDEFGHIKLMNPQRSTVWY')
    return ''.join([aa if aa in standard_aa else 'X' for aa in seq])
```

**3. Memory issues with large datasets:**
```python
# Process in batches
batch_size = 1000
for i in range(0, len(sequences), batch_size):
    batch = sequences[i:i+batch_size]
    process_batch(batch)
```

---

## Citation

If you use this workflow in your research, please cite:

**iFeature:**
```
Chen, Z., Zhao, P., Li, F., Leier, A., Marquez-Lago, T. T., Wang, Y., ... & Song, J. (2018). 
iFeature: a Python package and web server for features extraction and selection from protein and peptide sequences. 
Bioinformatics, 34(14), 2499-2502.
```

**BibTeX:**
```bibtex
@article{chen2018ifeature,
  title={iFeature: a Python package and web server for features extraction and selection from protein and peptide sequences},
  author={Chen, Zhen and Zhao, Peng and Li, Fuyi and Leier, Andre and Marquez-Lago, Tatiana T and Wang, Yanan and Webb, Geoffrey I and Smith, A Ian and Daly, Roger J and Chou, Kuo-Chen and others},
  journal={Bioinformatics},
  volume={34},
  number={14},
  pages={2499--2502},
  year={2018},
  publisher={Oxford University Press}
}
```

---

## License

This project is licensed under the MIT License.

**MIT License**

```
Copyright (c) 2025 [Your Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## Acknowledgments

- **iFeature** - Original feature extraction toolkit by Chen et al.
- **AMP Databases** - DBAASP, APD3, CAMP, DRAMP for providing public datasets
- **Community** - Contributors and users who improve this resource

---

## Contact & Support

- **Issues:** Open an issue on GitHub for bug reports or feature requests
- **Original iFeature:** [github.com/Superzchen/iFeature](https://github.com/Superzchen/iFeature)
- **iFeature Web Server:** [ifeature.erc.monash.edu](http://ifeature.erc.monash.edu/)

---

## Disclaimer

This is a tutorial and analysis framework built on top of iFeature. All credit for the core feature extraction algorithms goes to the original iFeature authors. This repository provides practical workflows and examples for applying iFeature to antimicrobial peptide research.

---

**Last Updated:** November 2025  
**Status:** Active Development

---

## Star This Repository

If you find this resource helpful, please consider giving it a star! It helps others discover this work.

**Happy AMP Analysis!**
