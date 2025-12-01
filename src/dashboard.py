"""Streamlit dashboard for SECOM yield analysis."""

import os
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pickle
from datetime import datetime, timedelta

PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed")


def load_processed():
    return pd.read_csv(os.path.join(PROCESSED_DIR, "secom_processed.csv"))


st.set_page_config(layout="wide", page_title="Semiconductor Manufacturing Yield Dashboard", initial_sidebar_state="expanded")

st.markdown("""<style>h2 { margin-top: 0; margin-bottom: 0.5rem; }</style>""", unsafe_allow_html=True)

def main():
    st.title("Semiconductor Manufacturing Yield Dashboard")
    st.divider()
    
    df = load_processed()

    # Sidebar filters
    st.sidebar.header("Filters & Analysis")
    
    with st.sidebar.expander("Location & Product", expanded=True):
        plant = st.selectbox("Plant Location", options=["All"] + sorted(df["plant_location"].unique().tolist()), help="Filter by manufacturing plant")
        product = st.selectbox("Product Family", options=["All"] + sorted(df["product_family"].unique().tolist()), help="Filter by product type")
    
    mask = pd.Series(True, index=df.index)
    if plant != "All":
        mask &= df["plant_location"] == plant
    if product != "All":
        mask &= df["product_family"] == product

    df_f = df[mask].copy()

    # Ensure date column exists (create synthetic if needed)
    if "date" not in df_f.columns:
        df_f["date"] = pd.to_datetime("2020-01-01") + pd.to_timedelta(df_f.reset_index().index, unit="D")
    else:
        df_f["date"] = pd.to_datetime(df_f["date"], errors="coerce")
        # Fill any NaT with synthetic sequence
        if df_f["date"].isna().any():
            missing_idx = df_f["date"].isna()
            seq = pd.to_datetime("2020-01-01") + pd.to_timedelta(np.arange(missing_idx.sum()), unit="D")
            df_f.loc[missing_idx, "date"] = seq.values

    # Date range filter in sidebar
    min_date = df_f["date"].min().date()
    max_date = df_f["date"].max().date()
    with st.sidebar.expander("Time Range", expanded=True):
        start_date, end_date = st.date_input("Select date range", [min_date, max_date])
    # apply date filter
    df_f = df_f[(df_f["date"] >= pd.to_datetime(start_date)) & (df_f["date"] <= pd.to_datetime(end_date))]
    
    st.sidebar.divider()

    # Simple KPIs - use Pass/Fail column directly (UCI SECOM: -1=fail, 1=pass) from FILTERED data
    total = len(df_f)
    if "Pass/Fail" in df_f.columns:
        fails = (df_f["Pass/Fail"] == -1).sum()
        passes = (df_f["Pass/Fail"] == 1).sum()
    elif "class" in df_f.columns:
        fails = (df_f["class"] == -1).sum()
        passes = (df_f["class"] == 1).sum()
    else:
        # Fallback
        fails = 0
        passes = total

    yield_rate = passes / total * 100 if total > 0 else 0
    dpm = fails / total * 1_000_000 if total > 0 else 0

    st.markdown("## Key Performance Indicators")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric("Yield Rate", f"{yield_rate:.2f}%", delta=f"{passes:,} pass", delta_color="off")
    with kpi2:
        st.metric("DPM", f"{dpm:,.0f}", delta=f"{fails:,} fail", delta_color="inverse")
    with kpi3:
        st.metric("Total Units", f"{total:,}", delta="in selected range", delta_color="off")
    with kpi4:
        st.metric("Pass Rate", f"{passes:,}", delta=f"{(passes/total*100):.1f}% of total" if total > 0 else "N/A", delta_color="off")
    
    st.divider()
    
    st.markdown("## Production Analysis")
    
    label_col = "Pass/Fail" if "Pass/Fail" in df_f.columns else ("class" if "class" in df_f.columns else df_f.columns[-1])
    yield_df = df_f.groupby(pd.Grouper(key="date", freq="D")).apply(lambda g: pd.Series({"total": len(g), "fails": int((g[label_col] == -1).sum())})).reset_index()
    if not yield_df.empty:
        yield_df["yield_rate"] = (yield_df["total"] - yield_df["fails"]) / yield_df["total"] * 100
        fig_y = px.line(yield_df, x="date", y="yield_rate", 
                       title="Daily Yield Rate Trend",
                       markers=True,
                       labels={"date": "Date", "yield_rate": "Yield Rate (%)"},
                       template="plotly_white")
        fig_y.update_traces(line=dict(color="#667eea", width=3), marker=dict(size=6))
        st.plotly_chart(fig_y, use_container_width=True)
    
    st.divider()

    # Sensor correlation heatmap
    st.markdown("## Sensor Diagnostics")
    
    col_heatmap, col_spc = st.columns(2)
    
    with col_heatmap:
        numeric = df_f.select_dtypes(include=["number"]).iloc[:, :60]  # limit columns for display
        if numeric.shape[1] >= 2:
            corr = numeric.corr()
            fig = px.imshow(corr, 
                           title="Sensor Correlation Matrix (First 60 Sensors)",
                           color_continuous_scale="RdBu",
                           template="plotly_white")
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

    with col_spc:
        # SPC chart (control limits) for a selected sensor
        numeric_cols = df_f.select_dtypes(include=["number"]).columns.tolist()
        if numeric_cols:
            spc_col = st.selectbox("Select Sensor for SPC Chart", options=numeric_cols, index=0)
            series = df_f[spc_col].dropna().astype(float)
            if not series.empty:
                mu = series.mean()
                sigma = series.std()
                ucl = mu + 3 * sigma
                lcl = mu - 3 * sigma
                fig_spc = go.Figure()
                fig_spc.add_trace(go.Scatter(x=df_f["date"], y=df_f[spc_col], 
                                            mode="lines+markers", 
                                            name=spc_col,
                                            line=dict(color="#667eea")))
                fig_spc.add_hline(y=mu, line_dash="dash", line_color="green", 
                                 annotation_text="Mean", annotation_position="right")
                fig_spc.add_hline(y=ucl, line_dash="dot", line_color="red", 
                                 annotation_text="UCL (±3σ)", annotation_position="right")
                fig_spc.add_hline(y=lcl, line_dash="dot", line_color="red", 
                                 annotation_text="LCL (±3σ)", annotation_position="right")
                fig_spc.update_layout(title=f"Statistical Process Control: {spc_col}",
                                     template="plotly_white",
                                     hovermode="x unified")
                st.plotly_chart(fig_spc, use_container_width=True)
    
    st.divider()

    # Feature Importance - Pareto Chart (Model 2 Output)
    st.markdown("## Feature Importance Analysis")
    
    model_path = os.path.join(PROCESSED_DIR, "rf_model.pkl")
    if os.path.exists(model_path):
        try:
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            
            if hasattr(model, "feature_importances_"):
                # Get feature importances from the trained Random Forest
                importances = model.feature_importances_
                
                # Get available numeric columns (in same order as model was trained)
                numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
                if len(numeric_cols) >= len(importances):
                    feature_names = numeric_cols[:len(importances)]
                else:
                    feature_names = [f"Feature_{i}" for i in range(len(importances))]
                
                # Create DataFrame and sort by importance
                importance_df = pd.DataFrame({
                    "Feature": feature_names,
                    "Importance": importances
                }).sort_values("Importance", ascending=False).head(20)  # Top 20 features
                
                # Calculate cumulative percentage
                importance_df["Cumulative_%"] = importance_df["Importance"].cumsum() / importance_df["Importance"].sum() * 100
                
                # Create Pareto chart (bar + line)
                fig_pareto = go.Figure()
                
                # Bar chart for importance
                fig_pareto.add_trace(go.Bar(
                    x=importance_df["Feature"],
                    y=importance_df["Importance"],
                    name="Importance Score",
                    marker_color="#667eea",
                    yaxis="y1"
                ))
                
                # Line chart for cumulative %
                fig_pareto.add_trace(go.Scatter(
                    x=importance_df["Feature"],
                    y=importance_df["Cumulative_%"],
                    name="Cumulative %",
                    line=dict(color="red", width=3),
                    mode="lines+markers",
                    marker=dict(size=8),
                    yaxis="y2"
                ))
                
                # Update layout with dual y-axes
                fig_pareto.update_layout(
                    title="Feature Importance - Pareto Chart (Top 20 Sensors)",
                    xaxis=dict(title="Sensor Name", tickangle=-45),
                    yaxis=dict(
                        title="Importance Score",
                        titlefont=dict(color="#667eea"),
                        tickfont=dict(color="#667eea")
                    ),
                    yaxis2=dict(
                        title="Cumulative %",
                        titlefont=dict(color="red"),
                        tickfont=dict(color="red"),
                        overlaying="y",
                        side="right"
                    ),
                    hovermode="x unified",
                    template="plotly_white",
                    legend=dict(x=0.5, y=1.1, orientation="h"),
                    height=500
                )
                
                st.plotly_chart(fig_pareto, use_container_width=True)
                
                # Display key insights
                col_insights1, col_insights2 = st.columns(2)
                with col_insights1:
                    top_5_importance = importance_df["Importance"].head(5).sum() / importance_df["Importance"].sum() * 100
                    st.metric("Top 5 Sensors Explain", f"{top_5_importance:.1f}%", delta="of total importance", delta_color="off")
                with col_insights2:
                    top_10_importance = importance_df["Importance"].head(10).sum() / importance_df["Importance"].sum() * 100
                    st.metric("Top 10 Sensors Explain", f"{top_10_importance:.1f}%", delta="of total importance", delta_color="off")
                
                # Detailed feature table
                with st.expander("View Detailed Feature Rankings"):
                    display_df = importance_df.copy()
                    display_df["Importance"] = display_df["Importance"].apply(lambda x: f"{x:.6f}")
                    display_df["Cumulative_%"] = display_df["Cumulative_%"].apply(lambda x: f"{x:.2f}%")
                    st.dataframe(display_df, use_container_width=True, hide_index=True)
            else:
                st.info("Feature importances not available in the model.")
        except Exception as e:
            st.warning(f"Could not extract feature importances: {e}")
    else:
        st.info("Train the model first by running: `python src/models.py`")
    
    st.divider()

    # Download filtered CSV
    try:
        csv_data = df_f.to_csv(index=False)
        st.download_button("Download Filtered Data (CSV)", 
                          data=csv_data, 
                          file_name=f"secom_filtered_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv", 
                          mime="text/csv",
                          help="Export current filtered dataset")
    except Exception:
        pass

    # ML model predictions (if model exists)
    model_path = os.path.join(PROCESSED_DIR, "rf_model.pkl")
    if os.path.exists(model_path):
        st.markdown("## Machine Learning Model Predictions")
        try:
            # Try to load with pickle
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            
            # select numeric features that match model input length
            if hasattr(model, "n_features_in_"):
                needed = int(model.n_features_in_)
                avail = df_f.select_dtypes(include=["number"]).columns.tolist()
                if len(avail) >= needed:
                    Xp = df_f[avail[:needed]].fillna(0).values
                    try:
                        probs = model.predict_proba(Xp)[:, 1] if hasattr(model, "predict_proba") else model.predict(Xp)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Avg Failure Probability", f"{probs.mean():.3f}", help="Average predicted failure probability across filtered data")
                        with col2:
                            high_risk = (probs > 0.5).sum()
                            st.metric("High-Risk Units", f"{high_risk:,}", delta=f"{high_risk/len(probs)*100:.1f}% of selected" if len(probs) > 0 else "N/A", delta_color="inverse")
                        
                        figp = px.histogram(probs, nbins=30, 
                                          title="Distribution of Predicted Failure Probabilities",
                                          labels={"value": "Failure Probability", "count": "Frequency"},
                                          template="plotly_white")
                        figp.update_traces(marker_color="#667eea")
                        st.plotly_chart(figp, use_container_width=True)
                    except Exception as score_err:
                        st.warning(f"Model loaded but scoring failed: {score_err}")
                        st.info("Try retraining by running: `python src/models.py`")
                else:
                    st.info("Model exists but not enough numeric columns available to score.")
            else:
                st.info("Model loaded but does not expose expected attributes for automatic scoring.")
        except Exception as e:
            st.warning(f"Could not load model: {e}")
            st.info("Try retraining by running: `python src/models.py`")
    
    st.divider()
    st.markdown("---")
    st.caption(f"Dashboard updated at {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')} | Showing {len(df_f):,} records out of {len(df):,} total")


if __name__ == "__main__":
    main()
