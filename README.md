# 🔮 AI Career Path Predictor using Semi-Supervised Self-Training

[![Python Version](https://img.shields.io/badge/Python-3.13+-blue.svg)](https://www.python.org/)
[![ML Framework](https://img.shields.io/badge/Framework-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

An end-to-end Machine Learning pipeline that predicts tech career paths (Software Engineer, Data Scientist, Product Manager) based on custom professional skill profiles. This project utilizes **Semi-Supervised Learning (Self-Training)** to iteratively expand its training domain from a minimal pool of initially labeled records.

---

## 📌 Project Overview & Problem Statement
In real-world data science, labeling data requires massive human effort, time, and budget. This project mimics this bottleneck by creating a data-scarce scenario. 

Instead of downloading a massive pre-labeled CSV, we use `scikit-learn`'s native dataset generation tools to spin up a 1,000-profile database on the fly. We then **intentionally hide 75% of the career targets**, forcing a semi-supervised wrapper to bridge the gap and pseudo-label the remaining data points based on feature similarities.

---

## 🧠 Methodology & Algorithm Architecture

### 1. Data Synthesis & Engineering
A 5-dimensional feature matrix is generated using `make_classification`. To align with scikit-learn's internal feature rules ($n_{\text{informative}} + n_{\text{redundant}} + n_{\text{repeated}} < n_{\text{features}}$), redundant features are explicitly set to zero, mapping neatly into realistic professional scorecards:
* `Coding_Score`
* `Math_Stats_Score`
* `System_Design_Score`
* `Product_Strategy`
* `Communication_Score`

### 2. The Semi-Supervised Loop
Unlabeled samples are flagged with a token value of `-1`. A standard `RandomForestClassifier` is encapsulated inside a `SelfTrainingClassifier` loop:

```text
    [ 25% Labeled Profiles ] ──> Train Base Classifier (Random Forest)
                                              │
                                              ▼
    [ 75% Unlabeled (-1) ]  ──> Predict Probability Threshold (>= 75%)
                                              │
    [ Pseudo-Labeled Data ] <─── Add highly confident profiles back


    precision    recall  f1-score   support

Software Engineer       0.80      0.80      0.80        64
   Data Scientist       0.79      0.80      0.79        65
  Product Manager       0.79      0.77      0.78        71

         accuracy                           0.79       200
        macro avg       0.79      0.79      0.79       200
     weighted avg       0.79      0.79      0.79       200