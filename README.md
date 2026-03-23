# 📉 Customer Churn Predictor

An end-to-end Machine Learning application that predicts **customer churn for telecom companies** using a Random Forest classifier — with an interactive live prediction interface.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red)
![RandomForest](https://img.shields.io/badge/Model-Random_Forest-green)

---

## 🚀 Live Demo
[👉 Click here to try it live](#) *(Deploy on Streamlit Cloud — free)*

---

## 🎯 Model Performance

| Metric | Score |
|---|---|
| Accuracy | ~85% |
| ROC-AUC | ~0.88 |
| Model | Random Forest (100 trees) |
| Dataset | Telecom Churn (7,000+ records) |

---

## 📊 Features

- 📊 **Model Performance Tab** — Confusion matrix, ROC curve, feature importances
- 🔍 **Data Insights Tab** — Churn distribution, tenure analysis
- 🎯 **Live Prediction Tab** — Enter customer details and get instant churn prediction with risk level

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Scikit-Learn | Random Forest model training |
| Pandas / NumPy | Data processing |
| Matplotlib / Seaborn | Visualizations |
| Streamlit | Interactive web app |

---

## ⚙️ Setup & Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/chauhan-03/customer-churn-predictor
cd customer-churn-predictor

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

---

## 📁 Project Structure

```
project3-churn-predictor/
│
├── app.py              # Streamlit app with model + UI
├── requirements.txt    # Dependencies
└── README.md           # This file
```

---

## 👨‍💻 Author

**Jatin Chauhan** — Engineer at Samsung R&D
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/jatin-chauhan-a07153171/)
