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

# ── What-If Price Simulator ──────────────────────────────────────
st.divider()
st.subheader("🎯 What-If Price Simulator")
st.markdown("*Simulate revenue impact of pricing changes in real-time*")

col_sim1, col_sim2 = st.columns([1, 2])

with col_sim1:
    price_change = st.slider("Price Change (%)", min_value=-30, max_value=30, value=0, step=1)
    st.markdown(f"**Current avg price:** ${df['unit_price'].mean():.2f}")
    new_price = df['unit_price'].mean() * (1 + price_change/100)
    st.markdown(f"**Simulated avg price:** ${new_price:.2f}")

    baseline_revenue = df['total_price'].sum()
    # Simple elasticity-based estimate (price elasticity ~ -1.5 is typical retail)
    elasticity = -1.5
    revenue_change_pct = elasticity * (price_change / 100)
    simulated_revenue = baseline_revenue * (1 + revenue_change_pct)
    delta = simulated_revenue - baseline_revenue

    st.metric("Baseline Revenue", f"${baseline_revenue:,.0f}")
    st.metric("Simulated Revenue", f"${simulated_revenue:,.0f}",
              delta=f"${delta:,.0f}")

    if delta > 0:
        st.success(f"✅ Price decrease could recover ${delta:,.0f} in revenue")
    elif delta < 0:
        st.warning(f"⚠️ Price increase may reduce revenue by ${abs(delta):,.0f}")
    else:
        st.info("No change in pricing")

with col_sim2:
    price_range = list(range(-30, 31, 5))
    revenues = [baseline_revenue * (1 + (elasticity * p/100)) for p in price_range]
    sim_df = pd.DataFrame({'Price Change (%)': price_range, 'Projected Revenue': revenues})
    fig_sim = px.line(sim_df, x='Price Change (%)', y='Projected Revenue',
                      markers=True, color_discrete_sequence=['steelblue'],
                      title='Revenue Sensitivity to Price Changes')
    fig_sim.add_vline(x=price_change, line_dash="dash", line_color="red",
                      annotation_text="Current selection")
    fig_sim.add_vline(x=0, line_dash="dot", line_color="gray",
                      annotation_text="Baseline")
    fig_sim.update_layout(height=380)
    st.plotly_chart(fig_sim, use_container_width=True)

    # ── Anomaly Detection ────────────────────────────────────────────
st.divider()
st.subheader("🚨 Competitor Price Anomaly Alerts")
st.markdown("*Products with abnormal pricing patterns detected via Isolation Forest*")

anomaly_df = pd.read_csv('anomaly_results.csv')
anomalies = anomaly_df[anomaly_df['is_anomaly'] == True]
normal = anomaly_df[anomaly_df['is_anomaly'] == False]

col_a1, col_a2, col_a3 = st.columns(3)
col_a1.metric("🚨 Anomalies Detected", len(anomalies))
col_a2.metric("✅ Normal Products", len(normal))
col_a3.metric("Anomaly Rate", f"{len(anomalies)/len(anomaly_df)*100:.1f}%")

col_b1, col_b2 = st.columns(2)

with col_b1:
    fig_a = px.scatter(anomaly_df, x='unit_price', y='avg_comp_price',
                       color='is_anomaly',
                       color_discrete_map={True: 'red', False: 'steelblue'},
                       labels={'unit_price': 'Our Price',
                               'avg_comp_price': 'Avg Competitor Price',
                               'is_anomaly': 'Anomaly'},
                       title='Anomaly Detection: Our Price vs Competitors')
    fig_a.update_layout(height=350)
    st.plotly_chart(fig_a, use_container_width=True)

with col_b2:
    fig_b = px.scatter(anomaly_df, x='price_premium', y='price_vs_comp1',
                       color='is_anomaly',
                       color_discrete_map={True: 'red', False: 'steelblue'},
                       labels={'price_premium': 'Price Premium Ratio',
                               'price_vs_comp1': 'Gap vs Competitor 1',
                               'is_anomaly': 'Anomaly'},
                       title='Price Premium vs Competitor Gap')
    fig_b.update_layout(height=350)
    st.plotly_chart(fig_b, use_container_width=True)

st.markdown("**🔴 Flagged Products Needing Attention:**")
st.dataframe(anomalies[['unit_price','avg_comp_price','price_premium',
                          'price_vs_comp1','price_vs_comp2','price_vs_comp3']].round(2),
             use_container_width=True, height=200)

# ── Causal Inference ─────────────────────────────────────────────
st.divider()
st.subheader("⚗️ Causal Inference: True Price Impact")
st.markdown("*Beyond correlation — quantifying the causal effect of price on revenue using DoWhy*")

with open('causal_result.json') as f:
    causal = json.load(f)

effect = causal['causal_effect']

col_c1, col_c2 = st.columns([1, 2])

with col_c1:
    st.metric("Causal Effect (per $1 price increase)", f"${effect:.2f}")
    st.markdown("""
    **How to read this:**
    - This is **not just correlation** — it's causal
    - Controlling for qty, competitors & lag price
    - A **$1 price increase** causes **$0.90 revenue loss**
    - Recommendation: prioritize volume over margin
    """)

    st.info(f"""
    💡 **Business Insight**
    
    To recover the **$9,758 opportunity**:
    - Reduce price by ~$10 on flagged products
    - Expected causal revenue gain: **${abs(effect)*10*34:,.0f}**
    - Affects {34} anomalous products identified
    """)

with col_c2:
    # Causal effect visualization
    price_deltas = list(range(-20, 21, 2))
    causal_revenues = [effect * d * len(df) for d in price_deltas]
    causal_sim_df = pd.DataFrame({
        'Price Change ($)': price_deltas,
        'Causal Revenue Impact ($)': causal_revenues
    })
    fig_c = px.bar(causal_sim_df, x='Price Change ($)', y='Causal Revenue Impact ($)',
                   color='Causal Revenue Impact ($)',
                   color_continuous_scale='RdYlGn',
                   title='Causal Revenue Impact by Price Change')
    fig_c.add_hline(y=0, line_color='black', line_dash='dash')
    fig_c.update_layout(height=380, showlegend=False)

st.caption("Built by Yuktha Reddy | XGBoost + SHAP Pricing Intelligence Engine")