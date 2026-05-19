# 🧬 AMP Discovery Pipeline
### An End-to-End Machine Learning Pipeline for Antimicrobial Peptide Prediction and Explainability

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-AUC--ROC%200.9725-brightgreen?style=flat)](https://xgboost.readthedocs.io)
[![SHAP](https://img.shields.io/badge/Explainability-SHAP-orange?style=flat)](https://shap.readthedocs.io)
[![Streamlit](https://img.shields.io/badge/App-Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Tests](https://img.shields.io/badge/Tests-42%20passing-success?style=flat)](tests/)
[![License](https://img.shields.io/badge/License-MIT-blue?style=flat)](LICENSE)

---

## 🌍 Problem Statement

Antimicrobial resistance (AMR) kills **1.3 million people annually** and is
projected to cause **10 million deaths per year by 2050** — surpassing cancer.
The global antibiotic pipeline is failing: traditional discovery takes 15 years
and costs over $1 billion per drug.

Antimicrobial peptides (AMPs) are a promising alternative. Found naturally in
frog skin, human immune cells, and bee venom, AMPs kill bacteria by physically
disrupting their membranes — a mechanism bacteria struggle to resist
evolutionarily.

**The bottleneck:** Out of millions of possible peptide sequences, identifying
which ones are antimicrobial requires expensive, slow wet lab screening.
This pipeline addresses that bottleneck computationally.

---

## 🔬 What This Project Does

> A research-grade ML pipeline that predicts whether a peptide sequence is
> antimicrobial, explains the biological reasoning using SHAP values, and
> provides multi-model consensus — deployed as an open-access web application.

### Novelty Over Existing Tools

| Feature | iAMPpred | AMPfun | CAMP | **This Pipeline** |
|---|---|---|---|---|
| AMP Prediction | ✅ | ✅ | ✅ | ✅ |
| SHAP Explainability | ❌ | ❌ | ❌ | ✅ |
| Multi-model Consensus | ❌ | ❌ | ❌ | ✅ |
| Toxicity Awareness | ❌ | ❌ | ❌ | ✅ |
| Live Web App | ❌ | ❌ | ❌ | ✅ |
| Open Source | ❌ | ❌ | ❌ | ✅ |
| Downloadable Reports | ❌ | ❌ | ❌ | ✅ |

---

## 📊 Results

### Model Performance — 5-Fold Stratified Cross Validation

| Model | Accuracy | AUC-ROC | Precision | Recall | F1 Score |
|---|---|---|---|---|---|
| **XGBoost** | **91.68%** | **0.9725** | **92.61%** | **90.66%** | **91.58%** |
| Random Forest | 91.24% | 0.9705 | 91.64% | 90.79% | 91.19% |
| Logistic Regression | 83.73% | 0.9003 | 84.30% | 82.97% | 83.58% |

> All metrics reported as mean across 5 CV folds.
> AUC-ROC is the primary metric for AMP classification tasks.
> 0.5 = random chance · 1.0 = perfect separation.

### Top Predictive Features (SHAP Analysis)

| Rank | Feature | Mean \|SHAP\| | Biological Meaning |
|---|---|---|---|
| 1 | aa_C (Cysteine) | 2.0015 | Defensin family marker |
| 2 | Length | 0.8577 | AMPs are shorter than average proteins |
| 3 | Charge | 0.3588 | AMPs are positively charged |
| 4 | aa_K (Lysine) | 0.3335 | Primary cationic residue in AMPs |
| 5 | Positive Fraction | 0.2827 | K+R+H content drives membrane binding |

---

## 🗂 Dataset

- **Source:** UniProt SwissProt (manually curated, gold standard)
- **Size:** 1,586 sequences (793 AMP + 793 non-AMP)
- **Perfectly balanced** — equal positive and negative classes
- **Deduplicated** — no sequence leakage between train and test

### AMP Sources (Positive Class)
- UniProt KW-0929 — experimentally verified antimicrobial peptides
- Defensins — cysteine-rich immune defense AMPs
- Bacteriocins — bacteria-derived AMPs
- Antibiotic peptides KW-0045
- Cathelicidins — mammalian innate immunity

### Non-AMP Sources (Negative Class)
- Neuropeptides KW-0547
- Hormones KW-0134
- Growth Factors KW-0349
- Transcription regulators KW-0167

---

## ⚙️ Pipeline Architecture

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/iamvarshag/amp_discovery_pipeline.git
cd amp_discovery_pipeline
```

### 2. Create environment

```bash
conda create -n amp-discovery python=3.10 -y
conda activate amp-discovery
pip install -r requirements.txt
```

### 3. Download data and train models

```bash
python data/download_data.py
python src/feature_engineering.py
python src/model_training.py
python src/explainability.py
```

### 4. Run the web app

```bash
streamlit run app/main.py
```

Open `http://localhost:8501` in your browser.

### 5. Run tests

```bash
pytest tests/ -v
```

Expected: **42 tests passing**

---

## 📁 Project Structure

---

## 🧬 How To Use The Web App

1. **Paste** any peptide sequence in single-letter amino acid code
2. **Click Analyze** to run prediction
3. **View** AMP probability with confidence level
4. **Explore SHAP tab** to understand WHY the model made this decision
5. **Check Model Consensus** — do all 3 models agree?
6. **Download** full feature report as CSV

### Example Sequences To Try

| Sequence | Source | Expected |
|---|---|---|
| `GIGKFLHSAKKFGKAFVGEIMNS` | Magainin-2, frog skin | AMP ✅ |
| `KLLLKWLLKWLKK` | Synthetic cationic AMP | AMP ✅ |
| `ILPWKWPWWPWRR` | Indolicidin, bovine | AMP ✅ |
| `LLGDFFRKSKEKIGKEFKRIVQRIKDFLRNLVPRTES` | LL-37, human | AMP ✅ |

---

## 🔬 Biological Interpretation

### Why Charge Matters
AMPs are positively charged (+2 to +9 at pH 7.4). Bacterial membranes are
negatively charged. Opposite charges attract — the AMP binds selectively to
bacteria while ignoring neutral human cell membranes. SHAP analysis confirms
charge is one of the top predictive features.

### Why Length Matters
AMPs are typically 10–50 amino acids. They need to be small enough to rapidly
insert into and disrupt bacterial membranes. Longer proteins rarely have this
property. Length is the second most important SHAP feature.

### Why SHAP Explainability Matters
A prediction without explanation is a black box. A researcher cannot act on
a yes/no answer. SHAP values tell the researcher exactly which amino acid
properties drive the prediction — making results actionable for peptide
engineering and optimization.

---

## ⚠️ Limitations

- **Cysteine bias:** Defensins represent ~20% of positive sequences,
  causing mild overweighting of cysteine-rich sequences. Version 1.1
  will apply family-balanced sampling.
- **Dataset size:** 1,586 sequences is sufficient for ML but larger
  datasets (DRAMP, 4000+ sequences) would improve generalization.
- **No wet lab validation:** All predictions are computational.
  Experimental confirmation is required before any research application.
- **Length restriction:** Sequences outside 10–100 amino acids are
  excluded. Extended AMPs are not covered.

---

## 🗺 Roadmap

- [x] Version 1.0 — Core prediction pipeline + SHAP + web app
- [ ] Version 1.1 — Family-balanced dataset, cysteine bias fix
- [ ] Version 1.2 — DRAMP integration (4,000+ sequences)
- [ ] Version 1.3 — Toxicity scoring (ToxinPred integration)
- [ ] Version 1.4 — Novel AMP generation module
- [ ] Version 2.0 — FastAPI backend + React frontend

---

## 🛠 Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.10 |
| ML Models | XGBoost, scikit-learn |
| Explainability | SHAP TreeExplainer |
| Biology | BioPython, peptides |
| Web App | Streamlit |
| Data Source | UniProt REST API |
| Testing | pytest (42 tests) |
| Containerization | Docker |
| Deployment | HuggingFace Spaces |

---

## 📖 References

1. Wang, G. et al. (2016). APD3: the antimicrobial peptide database.
   *Nucleic Acids Research*, 44(D1), D1087-D1093.
2. Bhadra, P. et al. (2018). iAMP-2L: A two-level multi-label
   classifier for identifying antimicrobial peptides.
   *Oncotarget*, 9(29), 20392.
3. Lundberg, S. M. & Lee, S. I. (2017). A unified approach to
   interpreting model predictions. *NeurIPS*, 30.
4. UniProt Consortium (2023). UniProt: the universal protein
   knowledgebase. *Nucleic Acids Research*, 51(D1), D523-D531.
5. The World Health Organization (2023). Antimicrobial Resistance
   Global Action Plan.

---

## ⚕️ Disclaimer

This is a **research prototype** for computational biomarker discovery.
It is not validated for clinical use and must not be used for medical
decisions. All predictions require experimental validation before
any research or therapeutic application.

---

## 👨‍🔬 Author

**Biotechnology Engineering Student**
RV College of Engineering, Bengaluru, India · 2026

---

*Built with biological rigor, engineering discipline, and scientific honesty.*