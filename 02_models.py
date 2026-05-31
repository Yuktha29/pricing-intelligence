import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, r2_score
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import json
import warnings
warnings.filterwarnings('ignore')

# Load cleaned data
df = pd.read_csv('retail_price_clean.csv')

# ── Feature Engineering ──────────────────────────────────────────
df['avg_comp_price'] = (df['comp_1'] + df['comp_2'] + df['comp_3']) / 3
df['price_premium'] = df['unit_price'] / df['avg_comp_price']
df['price_vs_comp1'] = df['unit_price'] - df['comp_1']
df['price_vs_comp2'] = df['unit_price'] - df['comp_2']
df['price_vs_comp3'] = df['unit_price'] - df['comp_3']
df['comp_min_price'] = df[['comp_1','comp_2','comp_3']].min(axis=1)
df['price_vs_cheapest'] = df['unit_price'] - df['comp_min_price']

# ── Target & Features ────────────────────────────────────────────
target = 'total_price'
drop_cols = [c for c in ['total_price','product_id','month_year'] if c in df.columns]
cat_cols = df.select_dtypes(include='object').columns.tolist()
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

feature_cols = [c for c in df.columns if c not in drop_cols]
X = df[feature_cols]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ── XGBoost Model ────────────────────────────────────────────────
xgb_model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05,
                               max_depth=6, random_state=42, verbosity=0)
xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)

mape = mean_absolute_percentage_error(y_test, y_pred) * 100
r2 = r2_score(y_test, y_pred)
print(f"XGBoost → MAPE: {mape:.2f}%  |  R²: {r2:.4f}")

# ── SHAP Analysis ────────────────────────────────────────────────
print("Running SHAP analysis...")
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)

plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test, max_display=15, show=False)
plt.tight_layout()
plt.savefig('shap_summary.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: shap_summary.png")

# Top SHAP features
shap_importance = pd.DataFrame({
    'feature': X_test.columns,
    'importance': np.abs(shap_values).mean(axis=0)
}).sort_values('importance', ascending=False)
print("\nTop 5 revenue drivers:")
print(shap_importance.head())
shap_importance.to_csv('shap_importance.csv', index=False)

# ── Revenue Opportunity ──────────────────────────────────────────
df['predicted_revenue'] = xgb_model.predict(X)
df['revenue_gap'] = df['predicted_revenue'] - df[target]
total_opportunity = df['revenue_gap'][df['revenue_gap'] > 0].sum()
print(f"\nRevenue recovery opportunity: ${total_opportunity:,.0f}")

# ── Save Outputs ─────────────────────────────────────────────────
df[['unit_price','avg_comp_price','price_premium',
    'predicted_revenue','revenue_gap']].to_csv('model_results.csv', index=False)

metrics = {
    "mape": round(mape, 2),
    "r2": round(r2, 4),
    "total_records": len(df),
    "revenue_opportunity": round(total_opportunity, 2),
    "avg_price_premium": round(df['price_premium'].mean(), 4)
}
with open('metrics.json', 'w') as f:
    json.dump(metrics, f)

print("\nAll files saved!")
print("Metrics:", metrics)

# ── Anomaly Detection ────────────────────────────────────────────
from sklearn.ensemble import IsolationForest

anomaly_features = df[['unit_price', 'avg_comp_price', 'price_premium',
                        'price_vs_comp1', 'price_vs_comp2', 'price_vs_comp3']].copy()

iso = IsolationForest(contamination=0.05, random_state=42)
df['anomaly'] = iso.fit_predict(anomaly_features)
df['is_anomaly'] = df['anomaly'] == -1

anomaly_count = df['is_anomaly'].sum()
print(f"\nAnomalies detected: {anomaly_count} ({anomaly_count/len(df)*100:.1f}% of products)")

df[['unit_price', 'avg_comp_price', 'price_premium',
    'price_vs_comp1', 'price_vs_comp2', 'price_vs_comp3',
    'is_anomaly']].to_csv('anomaly_results.csv', index=False)
print("Saved: anomaly_results.csv")

# ── Causal Inference ─────────────────────────────────────────────
import dowhy
from dowhy import CausalModel

causal_df = pd.read_csv('retail_price_clean.csv')[
    ['unit_price', 'total_price', 'qty', 'comp_1', 'comp_2', 'comp_3', 'lag_price']
].dropna()

model = CausalModel(
    data=causal_df,
    treatment='unit_price',
    outcome='total_price',
    common_causes=['qty', 'comp_1', 'comp_2', 'comp_3', 'lag_price']
)

identified = model.identify_effect(proceed_when_unidentifiable=True)
estimate = model.estimate_effect(
    identified,
    method_name="backdoor.linear_regression"
)

causal_effect = estimate.value
print(f"\nCausal Effect of price on revenue: {causal_effect:.4f}")
print("Interpretation: A $1 increase in unit price causes a",
      f"${causal_effect:.2f} change in total revenue")

# Save causal result
causal_result = {"causal_effect": round(float(causal_effect), 4)}
with open('causal_result.json', 'w') as f:
    json.dump(causal_result, f)
print("Saved: causal_result.json")