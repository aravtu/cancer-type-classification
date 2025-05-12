# 🧬 Multi-class Cancer Type Classification Using TCGA RNA-Seq Data

This repository contains a complete machine learning pipeline for classifying cancer types using RNA-seq gene expression data from The Cancer Genome Atlas (TCGA), with model interpretability via SHAP and validation on external GEO datasets.

---

## 🚀 Overview

- **Goal**: Predict cancer type based on gene expression profiles
- **Data**: TCGA RNA-seq (FPKM, log2-transformed) + GEO validation (GSE270769)
- **Model**: XGBoost classifier trained on the top 10 most common cancer types
- **Accuracy**: ~98% on test set
- **Interpretability**: Global and per-sample SHAP analysis
- **Generalization**: Validated on unseen external dataset

---

## 📁 Files

- `cancer_classifier_model.joblib`: Trained XGBoost model
- `notebook.ipynb`: Main Jupyter Notebook with code for:
  - Data preparation
  - EDA (PCA, UMAP)
  - Model training (LogReg, RF, XGBoost)
  - SHAP interpretability
  - GEO validation
- `README.md`: Project summary and usage
- `data/`: (optional) Folder for processed TCGA/GEO expression files

---

## 📊 Results

- **Train/Test Split**: Stratified, 80/20
- **Accuracy**: 98%
- **ROC AUC**: >0.97 (multiclass)
- **Top Biomarkers**: GATA3, PRLR, LMX1B (for BRCA, LGG, etc.)
- **External Validation**: All GEO samples correctly classified (GSE270769)

---

## 🔍 SHAP Interpretability

- SHAP summary plots for top 1000 genes
- Per-class SHAP value extraction
- UMAP clustering of SHAP profiles
- Force plots for individual samples
- Shared vs unique genes across cancer types

---

## 🔄 External Validation

Validated the model on GEO dataset `GSE270769` (breast invasive carcinoma).  
Top predictors (GATA3, PRLR, LMX1B) remained consistent.

---

## 🧪 Model Inference

To use the pretrained model:

## 🛠 Requirements

- Python 3.8+
- pandas, numpy, scikit-learn, xgboost
- shap, umap-learn, matplotlib, seaborn
- joblib

Install with:

```bash
pip install -r requirements.txt
```

---

## 📜 Citation

If you use this project, please cite:

> *Multi-class Cancer Type Classification Using TCGA RNA-Seq Data and Machine Learning*, 2025.

---

