# Bankruptcy Prediction System

Machine learning project for predicting corporate bankruptcy using an imbalanced financial dataset, with a strong focus on recall optimization, cost-sensitive learning, and threshold tuning.

---

## Overview

Early detection of financially distressed companies is critical for reducing losses, improving risk management, and supporting proactive decision-making. This project builds a bankruptcy prediction system designed to maximize recall, since missing a bankrupt company can have serious financial consequences.

The study compares multiple machine learning models under a highly imbalanced class distribution and evaluates them using recall, precision, F1-score, accuracy, AUC, and tuned decision thresholds.

---

## Project Goal

The objective of this project is to identify companies likely to go bankrupt with high sensitivity.

Key priorities:
- Maximize recall for bankrupt firms  
- Reduce false negatives  
- Handle severe class imbalance  
- Compare classical and ensemble models  
- Tune decision thresholds for improved sensitivity

---

## Dataset

This project uses the **Taiwan Economic Journal Bankruptcy Dataset**.

### Dataset Properties
- **6,819 companies**
- **95 financial indicators**
- Binary target variable:
  - `0` — Non-bankrupt
  - `1` — Bankrupt

The dataset is highly imbalanced, with bankrupt firms representing **less than 0.48%** of all samples.

Financial indicators include:
- Profitability ratios  
- Liquidity measures  
- Solvency indicators  
- Leverage metrics  
- Operating efficiency variables

---

## Methodology

The project follows the pipeline below:

```text
Raw Financial Data
→ Data Preprocessing
→ Class Imbalance Handling
→ Model Training
→ Hyperparameter Tuning
→ Threshold Tuning
→ Evaluation
```

### Preprocessing
The workflow includes:
- Removing low-variance features  
- Dropping missing values  
- Standardizing features using `StandardScaler`  
- Stratified train/validation/test splits  

---

## Handling Class Imbalance

Because bankrupt firms are extremely rare, cost-sensitive learning was used:

### Logistic Regression
```python
class_weight = {0:1,1:5}
```

### Random Forest
```python
class_weight = "balanced"
```

### XGBoost
```python
scale_pos_weight = imbalance_ratio * 1.5
```

### Neural Network
- Class weights computed from training distribution

---

## Models Evaluated

This project benchmarks:

- Logistic Regression  
- Random Forest  
- XGBoost  
- Neural Network

Hyperparameters were tuned using **GridSearchCV** with **recall** as the optimization metric.

---

## Threshold Tuning

Instead of relying on the default `0.5` decision threshold, thresholds were tuned on validation data to maximize recall.

This improves sensitivity and reduces the risk of missed bankruptcies.

---

## Results

### Test Set Performance

| Model | Accuracy | Precision | Recall | F1 | AUC | Threshold |
|------|----------:|----------:|------:|----:|----:|----------:|
| Logistic Regression | 0.0391 | 0.0325 | 1.0000 | 0.0629 | 0.8908 | 0.0000 |
| Random Forest | 0.7703 | 0.1231 | 1.0000 | 0.2193 | 0.9458 | 0.0366 |
| Neural Network | 0.0323 | 0.0304 | 0.9394 | 0.0589 | 0.8500 | 0.0000 |
| XGBoost | 0.8759 | 0.1948 | 0.9091 | 0.3209 | 0.9382 | 0.0067 |

---

## Key Findings

### Logistic Regression
- Perfect recall  
- Very low precision  
- Over-predicts bankruptcy cases

### Random Forest
- Perfect recall  
- Better precision  
- Strong AUC performance

### Neural Network
- High recall  
- Lower stability than tree-based models

### XGBoost
- Best balance of recall, precision and AUC  
- Strongest practical deployment candidate

---

## Why Recall Matters

In bankruptcy prediction:

- False positives may cause unnecessary caution  
- False negatives may miss distressed firms entirely  

Because missing bankrupt companies is much more costly, recall was prioritized throughout this project.

---

## Practical Deployment Recommendations

Based on results, the project recommends:

1. Deploy XGBoost as the primary model  
2. Use threshold tuning to adjust sensitivity  
3. Continuously monitor:
   - Recall  
   - Precision  
   - F1 Score  
   - ROC-AUC  

4. Retrain models periodically using updated financial data

---

## Repository Contents

This repository includes:

- Project notebook with experiments  
- Source code for preprocessing and model training  
- Written project report (PDF)  
- Project presentation slides (PPTX)

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Sumant-varanasi/taiwan-bankruptcy-risk-modeling.git
cd taiwan-bankruptcy-risk-modeling
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

Run the model pipeline:

```bash
python train_models.py
```

Or launch the notebook:

```bash
jupyter notebook
```

---

## Future Work

Potential extensions:

- Ensemble stacking  
- Cost-sensitive boosting variants  
- Deep learning for tabular finance  
- Graph-based financial risk models  
- Real-time early warning bankruptcy systems

---

## Conclusion

This project demonstrates that recall-optimized bankruptcy prediction can be improved substantially using cost-sensitive learning and threshold tuning.

Among all models tested, **XGBoost** provided the strongest practical balance between:
- High recall  
- Better precision  
- Strong discrimination (AUC)

making it the preferred model for deployment.

---

