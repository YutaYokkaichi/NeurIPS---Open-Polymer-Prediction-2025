# NeurIPS - Open Polymer Prediction 2025

**Predicting polymer properties with machine learning to accelerate sustainable materials research**

[![Competition](https://img.shields.io/badge/Kaggle-Competition-blue)](https://www.kaggle.com/competitions/neurips-open-polymer-prediction-2025/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://www.mit.edu/~amini/LICENSE.md)
[![Prize Pool](https://img.shields.io/badge/Prize%20Pool-$50,000-gold)](https://www.kaggle.com/competitions/neurips-open-polymer-prediction-2025/)

## 🎯 Competition Overview

Can your model unlock the secrets of polymers? This competition challenges participants to predict fundamental polymer properties to accelerate sustainable materials research. Polymers are essential building blocks in medicine, electronics, and sustainability, but progress has been limited by the lack of accessible, high-quality data.

This competition introduces a **game-changing, large-scale open-source dataset** – ten times larger than any existing resource – to unlock the vast potential of sustainable materials through machine learning.

## 🧪 Problem Statement

Your mission is to predict a polymer's real-world performance directly from its chemical structure. Given a polymer's structure as a SMILES (Simplified Molecular Input Line Entry System) string, build a model that accurately forecasts **five key properties**:

### Target Properties

1. **Glass Transition Temperature (Tg)** - °C
   - Critical temperature where polymer transitions from glassy to rubbery state
   
2. **Fractional Free Volume (FFV)** - Dimensionless
   - Measure of packing efficiency and molecular mobility
   
3. **Thermal Conductivity (Tc)** - W/m·K
   - Heat transfer capability of the polymer
   
4. **Density** - g/cm³
   - Mass per unit volume of the polymer
   
5. **Radius of Gyration (Rg)** - Å
   - Measure of molecular size and compactness

All ground truth values are derived from **molecular dynamics simulation** averages.

## 🗂️ Dataset Structure

```
dataset/
├── train.csv           # Training data with features and targets
├── test.csv           # Test data (features only)
└── sample_submission.csv  # Submission format example
```

### Data Files

#### `train.csv` (7,975 entries)
- `id`: Unique polymer identifier
- `SMILES`: Chemical structure notation
- `Tg`, `FFV`, `Tc`, `Density`, `Rg`: Target properties (some may be missing)

#### `test.csv` (4 entries visible + ~1,500 in hidden test set)
- `id`: Unique polymer identifier  
- `SMILES`: Chemical structure notation

#### `sample_submission.csv`
- Required format: `id,Tg,FFV,Tc,Density,Rg`
- All predictions must be provided for each test sample

## 📊 Evaluation Metric

The competition uses **weighted Mean Absolute Error (wMAE)** across all five properties:

```
wMAE = (1/|X|) × Σ(X∈X) Σ(i∈I(X)) w_i × |ŷ_i(X) - y_i(X)|
```

Where the weighting factor ensures:
- **Scale normalization**: Properties with different scales contribute equally
- **Inverse square-root scaling**: Rare properties get higher weights
- **Weight normalization**: Total weight across all properties equals K (number of tasks)

## 🏆 Prizes & Timeline

### Prize Pool: $50,000
- 🥇 **1st Place**: $12,000
- 🥈 **2nd Place**: $10,000  
- 🥉 **3rd Place**: $10,000
- 🏅 **4th Place**: $8,000
- 🏅 **5th Place**: $5,000
- 🎓 **Top Student Group**: $5,000

### Key Dates
- **Start**: June 16, 2025
- **Entry Deadline**: September 8, 2025
- **Team Merger Deadline**: September 8, 2025
- **Final Submission**: September 15, 2025
- *All deadlines at 11:59 PM UTC*

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas numpy scikit-learn rdkit-pypi matplotlib seaborn
```

### Data Loading
```python
import pandas as pd

# Load datasets
train_df = pd.read_csv('dataset/train.csv')
test_df = pd.read_csv('dataset/test.csv')
sample_submission = pd.read_csv('dataset/sample_submission.csv')

print(f"Training samples: {len(train_df)}")
print(f"Test samples: {len(test_df)}")
print(f"Features: {train_df.columns.tolist()}")
```

### Submission Format
```python
# Create submission
submission = pd.DataFrame({
    'id': test_df['id'],
    'Tg': predictions[:, 0],
    'FFV': predictions[:, 1], 
    'Tc': predictions[:, 2],
    'Density': predictions[:, 3],
    'Rg': predictions[:, 4]
})
submission.to_csv('submission.csv', index=False)
```

## 🔬 Approach Ideas

### Feature Engineering
- **Molecular Descriptors**: Extract chemical features from SMILES
- **Graph Neural Networks**: Represent molecules as graphs
- **Fingerprints**: Morgan, MACCS, or custom molecular fingerprints
- **3D Conformations**: Generate and analyze 3D molecular structures

### Modeling Strategies
- **Multi-task Learning**: Predict all five properties simultaneously
- **Ensemble Methods**: Combine multiple models for robust predictions
- **Deep Learning**: Transformer models for SMILES sequences
- **Traditional ML**: Random Forest, XGBoost with engineered features

### Data Challenges
- **Missing Values**: Handle sparse target matrices strategically
- **Multi-scale Targets**: Properties have vastly different ranges
- **Chemical Validity**: Ensure SMILES parsing and validation

## 📚 Useful Resources

### Chemical Informatics
- [SMILES Notation](https://en.wikipedia.org/wiki/Simplified_Molecular_Input_Line_Entry_System)
- [RDKit Documentation](https://www.rdkit.org/docs/)
- [Molecular Descriptors](https://www.rdkit.org/docs/GettingStartedInPython.html#list-of-available-descriptors)

### Polymer Science
- [Polymer Density](https://omnexus.specialchem.com/polymer-property/density)
- [Glass Transition Temperature](https://www.protolabs.com/resources/design-tips/glass-transition-temperature-of-polymers/)
- [Thermal Conductivity](https://web.mit.edu/nanoengineering/research/polymers.shtml)

### Machine Learning
- [Graph Neural Networks for Chemistry](https://github.com/deepchem/deepchem)
- [Molecular Property Prediction](https://arxiv.org/abs/1609.02907)

## 🏛️ Competition Host

**University of Notre Dame**

## 📜 Citation

```bibtex
@misc{neurips-polymer-2025,
    title={NeurIPS - Open Polymer Prediction 2025},
    author={Gang Liu and Jiaxin Xu and Eric Inae and Yihan Zhu and Ying Li and Tengfei Luo and Meng Jiang and Yao Yan and Walter Reade and Sohier Dane and Addison Howard and María Cruz},
    year={2025},
    url={https://kaggle.com/competitions/neurips-open-polymer-prediction-2025},
    publisher={Kaggle}
}
```

## 📊 Competition Stats

- **Entrants**: 535
- **Active Participants**: 18  
- **Teams**: 18
- **Submissions**: 46
- **Dataset Size**: 691.6 kB

---

*This project aims to accelerate sustainable polymer research through virtual screening and drive significant advancements in materials science. Your contributions have the potential to redefine polymer discovery!* 🌱