# 🏦 Universal Bank — Personal Loan Intelligence Hub

A full-stack Streamlit dashboard for personal loan campaign targeting, built for the Head of Marketing at Universal Bank.

---

## 🚀 Live Demo (Streamlit Cloud)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo → set **Main file path** to `app.py`
4. Click **Deploy**

---

## 📁 Project Structure

```
universal_bank_app/
├── app.py                  # Main Streamlit application
├── UniversalBank.csv       # Training dataset
├── requirements.txt        # Python dependencies
└── README.md
```

---

## 📊 Dashboard Sections

| Section | What it covers |
|---|---|
| 🏠 Executive Overview | KPIs, model summary, top predictors |
| 📊 Descriptive Analytics | Demographics, income, education, product ownership |
| 🔍 Diagnostic Analytics | Why customers accept/reject loans — cross-segment deep dives |
| 🤖 Predictive Models | Decision Tree, Random Forest, Gradient Boosted Tree — metrics, ROC, confusion matrix |
| 🎯 Prescriptive Analytics | Campaign targeting tiers, ICP, ROI estimator |
| 📁 Predict New Customers | Upload CSV → download predictions with probability scores |

---

## 🤖 Models Used

- **Decision Tree** (max_depth=6)
- **Random Forest** (100 estimators, max_depth=8)
- **Gradient Boosted Tree** (100 estimators, learning_rate=0.1)

All models evaluated on: Accuracy, Precision, Recall, F1 Score, AUC-ROC

---

## 📥 Test File Format

Upload a CSV with these columns to predict new customers:

```
Age, Experience, Income, Family, CCAvg, Education, Mortgage,
Securities Account, CD Account, Online, CreditCard
```

The app will return:
- `Predicted_Personal_Loan` (0/1)
- `Predicted_Personal_Loan_Label` (Will Accept / Will NOT Accept)
- `Loan_Probability_%`
- `Priority_Tier` (Low / Medium / High / Priority)

---

## 🛠 Local Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

*Built with Streamlit · scikit-learn · matplotlib · seaborn*
