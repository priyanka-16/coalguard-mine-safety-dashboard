import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="CoalGuard", page_icon="⚫", layout="wide")

st.title("⚫ **CoalGuard** — AI Mine Calamity Prediction")
st.markdown("**India loses a miner every 36 hours. CoalGuard predicts the next incident before it happens.**")

@st.cache_data
def load_data():
    df = pd.read_csv('dataset.csv')
    return df

df = load_data()

# Sidebar filters
st.sidebar.header("🔍 Filters")
state_filter = st.sidebar.multiselect("State", options=df['state'].unique(), default=df['state'].unique())
mine_type_filter = st.sidebar.multiselect("Mine Type", options=df['mine_type'].unique(), default=df['mine_type'].unique())

filtered_df = df[df['state'].isin(state_filter) & df['mine_type'].isin(mine_type_filter)]

# KPI Metrics
col1, col2, col3, col4 = st.columns(4)
total_shifts = len(filtered_df)
incidents = filtered_df['target_incident'].sum()
incident_rate = (incidents / total_shifts * 100).round(2)

with col1:
    st.metric("Total Shifts", total_shifts)
with col2:
    st.metric("Incidents", incidents, delta=f"{incident_rate}%")
with col3:
    st.metric("Workers Exposed", filtered_df['workers_underground'].sum())
with col4:
    st.metric("High Risk Shifts", len(filtered_df[filtered_df['methane_ppm'] > 2000]))

# Risk Heatmap
st.subheader("🔥 Current Mine Risk Heatmap")
fig_heatmap = px.density_heatmap(filtered_df, x='mine_id', y='shift', 
                                z='methane_ppm', color_continuous_scale='Reds',
                                title="Methane Levels by Mine & Shift")
st.plotly_chart(fig_heatmap, use_container_width=True)

# Top Risk Factors
st.subheader("⚠️ Top 5 Risk Alerts")
risk_cols = ['methane_ppm', 'co_ppm', 'water_seepage_m3h', 'roof_stability']
risk_data = []
for col in risk_cols:
    threshold = filtered_df[col].quantile(0.9)
    risky_shifts = len(filtered_df[filtered_df[col] > threshold])
    risk_data.append({'Risk Factor': col.replace('_', ' ').title(), 'Risky Shifts': risky_shifts})

risk_df = pd.DataFrame(risk_data)
fig_risk = px.bar(risk_df, x='Risk Factor', y='Risky Shifts', color='Risky Shifts', 
                  color_continuous_scale='Reds')
st.plotly_chart(fig_risk, use_container_width=True)

# Incident Prediction Model
st.subheader("🤖 AI Incident Prediction")
X = filtered_df[['methane_ppm', 'co_ppm', 'roof_stability', 'pillar_stress_kpa', 'workers_underground']]
y = filtered_df['target_incident']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

predictions = model.predict(X_scaled)
probabilities = model.predict_proba(X_scaled)[:,1]

col1, col2 = st.columns(2)
with col1:
    st.metric("Model Accuracy", f"{model.score(X_scaled, y):.1%}")
with col2:
    st.metric("Next Shift Risk", f"{probabilities.mean()*100:.1f}%")

# Risk Distribution
fig_prob = px.histogram(x=probabilities*100, nbins=20, title="Predicted Incident Probability Distribution")
st.plotly_chart(fig_prob, use_container_width=True)

# Association Rules
st.subheader("🔗 Danger Patterns (Apriori)")
binary_data = pd.get_dummies(filtered_df[['mine_type', 'shift', 'target_incident']].astype(bool))
frequent_itemsets = apriori(binary_data, min_support=0.1, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence']].head(10))

st.markdown("---")
st.markdown("*Built for Data Analytics Project by Payal Manwani | Data: DGMS India*")
