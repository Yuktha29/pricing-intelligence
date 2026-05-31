import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json

st.set_page_config(page_title="Pricing Intelligence Engine", layout="wide", page_icon="💰")

# ── Load Data ────────────────────────────────────────────────────
df = pd.read_csv('retail_price_clean.csv')
results = pd.read_csv('model_results.csv')
shap_df = pd.read_csv('shap_importance.csv')
with open('metrics.json') as f:
    metrics = json.load(f)

# ── Header ───────────────────────────────────────────────────────
st.title("💰 Pricing Intelligence Engine")
st.markdown("*Real-time competitive pricing analysis powered by XGBoost + SHAP*")
st.divider()

# ── KPI Cards ────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Model Accuracy (MAPE)", f"{metrics['mape']}%", "Lower is better")
c2.metric("R² Score", f"{metrics['r2']}", "94% variance explained")
c3.metric("Products Analyzed", f"{metrics['total_records']:,}")
c4.metric("Revenue Opportunity", f"${metrics['revenue_opportunity']:,.0f}", "Pricing gap identified")

st.divider()

# ── Row 1: Price Distribution + Competitor Gap ───────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Price Distribution vs Competitors")
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=df['unit_price'], name='Our Price',
                               marker_color='steelblue', opacity=0.7, nbinsx=30))
    fig.add_trace(go.Histogram(x=df['comp_1'], name='Competitor 1',
                               marker_color='coral', opacity=0.7, nbinsx=30))
    fig.add_trace(go.Histogram(x=df['comp_2'], name='Competitor 2',
                               marker_color='green', opacity=0.7, nbinsx=30))
    fig.update_layout(barmode='overlay', height=350,
                      xaxis_title="Price", yaxis_title="Count")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("📈 Price Premium Over Competitors")
    fig2 = px.histogram(results, x='price_premium', nbins=30,
                        color_discrete_sequence=['steelblue'],
                        labels={'price_premium': 'Price Premium Ratio'})
    fig2.add_vline(x=1.0, line_dash="dash", line_color="red",
                   annotation_text="Break-even")
    fig2.update_layout(height=350)
    st.plotly_chart(fig2, use_container_width=True)

# ── Row 2: SHAP + Revenue Gap ────────────────────────────────────
col3, col4 = st.columns(2)

with col3:
    st.subheader("🔍 Top Revenue Drivers (SHAP)")
    top_shap = shap_df.head(10).sort_values('importance')
    fig3 = px.bar(top_shap, x='importance', y='feature', orientation='h',
                  color='importance', color_continuous_scale='Blues',
                  labels={'importance': 'SHAP Importance', 'feature': 'Feature'})
    fig3.update_layout(height=350, showlegend=False)
    st.plotly_chart(fig3, use_container_width=True)

with col4:
    st.subheader("💸 Revenue Gap Analysis")
    fig4 = px.scatter(results, x='unit_price', y='revenue_gap',
                      color='price_premium',
                      color_continuous_scale='RdYlGn',
                      labels={'unit_price': 'Unit Price',
                              'revenue_gap': 'Revenue Gap ($)',
                              'price_premium': 'Price Premium'},
                      opacity=0.6)
    fig4.add_hline(y=0, line_dash="dash", line_color="black")
    fig4.update_layout(height=350)
    st.plotly_chart(fig4, use_container_width=True)

# ── Row 3: Competitor Price Comparison ───────────────────────────
st.subheader("🏪 Our Price vs Each Competitor")
col5, col6, col7 = st.columns(3)

for col, comp, name in zip([col5, col6, col7],
                            ['comp_1','comp_2','comp_3'],
                            ['Competitor 1','Competitor 2','Competitor 3']):
    with col:
        avg_diff = (df['unit_price'] - df[comp]).mean()
        pct_above = (df['unit_price'] > df[comp]).mean() * 100
        fig = px.scatter(df, x=comp, y='unit_price', opacity=0.4,
                         color_discrete_sequence=['steelblue'],
                         labels={comp: f'{name} Price', 'unit_price': 'Our Price'})
        fig.add_shape(type='line', x0=df[comp].min(), y0=df[comp].min(),
                      x1=df[comp].max(), y1=df[comp].max(),
                      line=dict(color='red', dash='dash'))
        fig.update_layout(height=280, title=f"Avg gap: ${avg_diff:.1f} | {pct_above:.0f}% above")
        st.plotly_chart(fig, use_container_width=True)

# ── Row 4: Raw Data Explorer ─────────────────────────────────────
st.divider()
st.subheader("🗂️ Data Explorer")
cols_show = ['unit_price','avg_comp_price','price_premium','predicted_revenue','revenue_gap']
st.dataframe(results[cols_show].round(2), use_container_width=True, height=250)

st.caption("Built by Yuktha Reddy | XGBoost + SHAP Pricing Intelligence Engine")