# Pricing Intelligence Engine

> **Live Demo:** [https://yuktha-pricing-intelligence.streamlit.app/]  
> **Tech Stack:** Python · XGBoost · DoWhy · SHAP · Isolation Forest · Streamlit · Plotly

A production-grade competitive pricing analytics system that goes beyond dashboards —
combining ML forecasting, causal inference, and anomaly detection into a live 
decision-making tool.

---

##  Business Problem

Retailers lose millions annually by mispricing against competitors. 
This system answers 3 questions most analytics tools can't:

1. **What** is our price position vs competitors right now?
2. **Why** does price change affect revenue? (causally, not just correlation)
3. **What happens** if we change price by X% tomorrow?

---

## Results

| Metric | Value |
|--------|-------|
| XGBoost MAPE | 8.19% |
| R² Score | 0.9426 |
| Revenue Opportunity Identified | $9,758 |
| Anomalous Products Flagged | 34 (5% of catalog) |
| Causal Effect of $1 Price Increase | -$0.90 revenue |

---

##  What Makes This Different

### 1. Causal Inference (DoWhy)
Most pricing models find correlation. This one proves **causality**.

Using Microsoft's DoWhy library with backdoor linear regression,
controlling for quantity, competitor prices, and lag price:

> A $1 increase in unit price **causes** $0.90 decrease in revenue

This is the same methodology used by pricing teams at Netflix, Uber, and Amazon.

### 2. Anomaly Detection (Isolation Forest)
Automatically flags products with abnormal pricing patterns — 
turning a one-time analysis into a **live monitoring system**.
34 products (5%) flagged as pricing anomalies requiring immediate attention.

### 3. What-If Price Simulator
Interactive slider lets any business user simulate revenue impact 
of pricing decisions **without writing a single line of code.**

---

##  Architecture

retail_price.csv
│
▼
x01_eda.py          → EDA, cleaning, visualizations
│
▼
02_models.py        → XGBoost · SHAP · Isolation Forest · DoWhy
│
▼
app.py              → Streamlit dashboard (live)

---

## 📈 Dashboard Features

- **KPI Cards** — MAPE, R², revenue opportunity at a glance
- **Price Distribution** — our price vs 3 competitors
- **SHAP Analysis** — top 15 revenue drivers explained
- **Revenue Gap Scatter** — where pricing opportunities exist
- **Anomaly Alerts** — 34 flagged products with drill-down table
- **What-If Simulator** — real-time revenue impact of price changes
- **Causal Inference Panel** — proven price-revenue causality

---

## 🚀 Run Locally

```bash
git clone https://github.com/Yuktha29/pricing-intelligence
cd pricing-intelligence

conda create -n pricing python=3.11 -y
conda activate pricing
pip install -r requirements.txt

python x01_eda.py
python 02_models.py
streamlit run app.py
```

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| ML & Modeling | XGBoost, Scikit-learn |
| Explainability | SHAP |
| Causal Inference | DoWhy |
| Anomaly Detection | Isolation Forest |
| Dashboard | Streamlit, Plotly |
| Data | Pandas, NumPy |

---

## 👩‍💻 Author

**Yuktha Reddy**  
Data Analyst · CS Graduate · University at Buffalo  
[LinkedIn](https://www.linkedin.com/in/yukthareddy29/) · [Portfolio](https://yuktha29.github.io/)