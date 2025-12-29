"""
Recovery Predictor - Streamlit Application
==========================================
Interactive tool for clinicians and line managers.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from dataclasses import asdict
import json

from config import (
    RecoveryConfig, DEFAULT_CONFIG,
    InjuryType, BodyRegion, JMESStatus, RecoveryBand, Trade,
    get_age_band, get_recovery_band
)
from recovery_model import RecoveryPredictor, CaseInput, RecoveryPrediction


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="SEKHMET Recovery Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Session state
if 'config' not in st.session_state:
    st.session_state.config = RecoveryConfig()
if 'predictor' not in st.session_state:
    st.session_state.predictor = RecoveryPredictor(st.session_state.config)


# ============================================================
# SIDEBAR - CONFIGURATION
# ============================================================

def render_sidebar():
    """Configuration sidebar"""
    
    st.sidebar.title("⚙️ Configuration")
    
    with st.sidebar.expander("📊 Recovery Bands", expanded=False):
        st.markdown("Define thresholds (months)")
        
        fast_max = st.slider("Fast: 0 to", 1, 6, 3, key="fast")
        medium_max = st.slider(f"Medium: {fast_max} to", fast_max + 1, 12, 6, key="medium")
        slow_max = st.slider(f"Slow: {medium_max} to", medium_max + 1, 24, 12, key="slow")
        
        st.session_state.config.band_thresholds = {
            "Fast": (0, fast_max),
            "Medium": (fast_max, medium_max),
            "Slow": (medium_max, slow_max),
            "Complex": (slow_max, 36)
        }
        
        # Show bands
        st.markdown("**Current Bands:**")
        for band, (lo, hi) in st.session_state.config.band_thresholds.items():
            color = {"Fast": "🟢", "Medium": "🟡", "Slow": "🟠", "Complex": "🔴"}[band]
            st.markdown(f"{color} {band}: {lo}-{hi} months")
    
    with st.sidebar.expander("🦴 Body Region Modifiers", expanded=False):
        st.markdown("Recovery time multipliers")
        
        for region in BodyRegion:
            current = st.session_state.config.body_region_modifiers.get(region.value, 1.0)
            new_val = st.slider(
                region.value.replace("_", " "),
                0.5, 2.0, current, 0.05,
                key=f"region_{region.value}"
            )
            st.session_state.config.body_region_modifiers[region.value] = new_val
    
    with st.sidebar.expander("👤 Age Modifiers", expanded=False):
        st.markdown("Recovery time multipliers by age")
        
        for age_band, current in st.session_state.config.age_modifiers.items():
            new_val = st.slider(
                age_band,
                0.5, 2.0, current, 0.05,
                key=f"age_{age_band}"
            )
            st.session_state.config.age_modifiers[age_band] = new_val
    
    with st.sidebar.expander("🩹 Injury Profiles", expanded=False):
        injury_type = st.selectbox(
            "Select injury type to edit",
            [i.value for i in InjuryType]
        )
        
        profile = st.session_state.config.injury_profiles[injury_type]
        
        col1, col2 = st.columns(2)
        with col1:
            min_months = st.number_input(
                "Min recovery (months)",
                0.5, 24.0, float(profile.base_recovery_months[0]), 0.5,
                key=f"inj_min_{injury_type}"
            )
        with col2:
            max_months = st.number_input(
                "Max recovery (months)",
                1.0, 36.0, float(profile.base_recovery_months[1]), 0.5,
                key=f"inj_max_{injury_type}"
            )
        
        recur_risk = st.slider(
            "Recurrence risk",
            0.0, 1.0, profile.recurrence_risk, 0.05,
            key=f"inj_recur_{injury_type}"
        )
        
        mld_prob = st.slider(
            "MLD probability",
            0.0, 1.0, profile.mld_probability, 0.05,
            key=f"inj_mld_{injury_type}"
        )
        
        # Update profile
        profile.base_recovery_months = (min_months, max_months)
        profile.recurrence_risk = recur_risk
        profile.mld_probability = mld_prob
    
    # Update predictor
    st.session_state.predictor = RecoveryPredictor(st.session_state.config)
    
    st.sidebar.markdown("---")
    
    # Export/Import config
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("📥 Export Config"):
            config_dict = {
                "band_thresholds": st.session_state.config.band_thresholds,
                "body_region_modifiers": st.session_state.config.body_region_modifiers,
                "age_modifiers": st.session_state.config.age_modifiers
            }
            st.sidebar.download_button(
                "Download JSON",
                json.dumps(config_dict, indent=2),
                "recovery_config.json",
                "application/json"
            )
    with col2:
        if st.button("🔄 Reset"):
            st.session_state.config = RecoveryConfig()
            st.session_state.predictor = RecoveryPredictor(st.session_state.config)
            st.rerun()


# ============================================================
# TAB 1: INDIVIDUAL PREDICTION
# ============================================================

def render_individual_tab():
    """Individual case prediction"""
    
    st.header("🧑‍⚕️ Individual Recovery Prediction")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Case Details")
        
        age = st.number_input("Age", 18, 60, 32)
        
        trade = st.selectbox(
            "Trade",
            [t.value for t in Trade],
            index=0
        )
        
        injury_type = st.selectbox(
            "Injury Type",
            [i.value for i in InjuryType],
            format_func=lambda x: f"{x} - {st.session_state.config.injury_profiles[x].description}"
        )
        
        body_region = st.selectbox(
            "Body Region",
            [b.value for b in BodyRegion]
        )
        
        severity = st.slider("Severity Score", 1, 10, 5)
        
        prior_injuries = st.number_input("Prior Injury Count", 0, 20, 0)
        
        prior_same_region = st.checkbox("Prior injury to same region?")
        
        current_jmes = st.selectbox(
            "Current JMES",
            [j.value for j in JMESStatus],
            index=1  # Default to MLD
        )
        
        months_since = st.number_input("Months since injury", 0, 24, 0)
        
        receiving_treatment = st.checkbox("Receiving treatment?", value=True)
        
        predict_btn = st.button("🔮 Predict Recovery", type="primary", use_container_width=True)
    
    with col2:
        if predict_btn:
            # Create case
            case = CaseInput(
                age=age,
                trade=Trade(trade),
                injury_type=InjuryType(injury_type),
                body_region=BodyRegion(body_region),
                severity_score=severity,
                prior_injury_count=prior_injuries,
                prior_same_region=prior_same_region,
                current_jmes=JMESStatus(current_jmes),
                months_since_injury=months_since,
                receiving_treatment=receiving_treatment
            )
            
            # Get prediction
            prediction = st.session_state.predictor.predict(case)
            
            # Display results
            st.subheader("📊 Prediction Results")
            
            # Key metrics
            col_a, col_b, col_c, col_d = st.columns(4)
            
            band_colors = {
                "Fast": "green",
                "Medium": "orange",
                "Slow": "red",
                "Complex": "purple"
            }
            
            with col_a:
                st.metric(
                    "Expected Recovery",
                    f"{prediction.expected_recovery_months} months"
                )
            with col_b:
                st.metric(
                    "Recovery Band",
                    prediction.recovery_band.value
                )
            with col_c:
                st.metric(
                    "Full Recovery Prob",
                    f"{prediction.prob_full_recovery:.0%}"
                )
            with col_d:
                st.metric(
                    "Confidence",
                    prediction.confidence_level
                )
            
            st.markdown("---")
            
            # Timeline
            st.subheader("📅 Recovery Timeline")
            
            col_t1, col_t2, col_t3 = st.columns(3)
            with col_t1:
                st.markdown(f"**🟢 Optimistic:** {prediction.optimistic_months} months")
            with col_t2:
                st.markdown(f"**🟡 Realistic:** {prediction.realistic_months} months")
            with col_t3:
                st.markdown(f"**🔴 Pessimistic:** {prediction.pessimistic_months} months")
            
            # Recovery curve
            curve_data = st.session_state.predictor.generate_recovery_curve(case, 24)
            curve_df = pd.DataFrame(curve_data)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=curve_df['month'],
                y=curve_df['cumulative_recovery_prob'],
                mode='lines+markers',
                name='Recovery Probability',
                line=dict(color='#2E86AB', width=3),
                fill='tozeroy',
                fillcolor='rgba(46, 134, 171, 0.2)'
            ))
            
            # Add threshold lines
            fig.add_hline(y=0.5, line_dash="dash", line_color="orange", 
                         annotation_text="50%")
            fig.add_hline(y=0.75, line_dash="dash", line_color="green",
                         annotation_text="75%")
            fig.add_hline(y=0.9, line_dash="dash", line_color="darkgreen",
                         annotation_text="90%")
            
            fig.update_layout(
                title="Cumulative Recovery Probability Over Time",
                xaxis_title="Months",
                yaxis_title="Probability of Recovery",
                yaxis_range=[0, 1],
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Probability milestones
            st.subheader("🎯 Recovery Milestones")
            
            milestone_df = pd.DataFrame({
                "Timeframe": ["3 months", "6 months", "12 months", "24 months"],
                "Probability": [
                    prediction.prob_recovery_3mo,
                    prediction.prob_recovery_6mo,
                    prediction.prob_recovery_12mo,
                    prediction.prob_recovery_24mo
                ]
            })
            
            fig_bar = px.bar(
                milestone_df,
                x="Timeframe",
                y="Probability",
                color="Probability",
                color_continuous_scale=["red", "orange", "green"],
                range_color=[0, 1]
            )
            fig_bar.update_layout(height=250, showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)
            
            # Contributing factors
            st.subheader("⚖️ Contributing Factors")
            
            factors_df = pd.DataFrame([
                {"Factor": k, "Impact": v, "Effect": "Slower" if v > 1 else "Faster" if v < 1 else "Neutral"}
                for k, v in prediction.contributing_factors.items()
            ])
            
            fig_factors = px.bar(
                factors_df,
                x="Impact",
                y="Factor",
                orientation='h',
                color="Effect",
                color_discrete_map={"Slower": "#e74c3c", "Faster": "#2ecc71", "Neutral": "#95a5a6"}
            )
            fig_factors.add_vline(x=1.0, line_dash="dash", line_color="black")
            fig_factors.update_layout(height=300, xaxis_range=[0.5, 2.0])
            st.plotly_chart(fig_factors, use_container_width=True)
            
            # JMES outcomes
            st.subheader("🏥 JMES Outcome Probabilities")
            
            jmes_df = pd.DataFrame({
                "Outcome": ["Full Recovery (MFD)", "Partial Recovery", "Medical Discharge"],
                "Probability": [
                    prediction.prob_full_recovery,
                    prediction.prob_partial_recovery,
                    prediction.prob_medical_discharge
                ]
            })
            
            fig_jmes = px.pie(
                jmes_df,
                values="Probability",
                names="Outcome",
                color="Outcome",
                color_discrete_map={
                    "Full Recovery (MFD)": "#2ecc71",
                    "Partial Recovery": "#f39c12",
                    "Medical Discharge": "#e74c3c"
                }
            )
            fig_jmes.update_layout(height=300)
            st.plotly_chart(fig_jmes, use_container_width=True)
            
            # Manager summary
            st.subheader("📋 Line Manager Summary")
            
            st.info(f"""
            **Expected Return to Duty:** {prediction.realistic_months} months (range: {prediction.optimistic_months}-{prediction.pessimistic_months})
            
            **Planning Recommendation:** 
            - Earliest possible: {prediction.optimistic_months} months
            - Plan for: {prediction.realistic_months} months  
            - Contingency: {prediction.pessimistic_months} months
            
            **Recovery Band:** {prediction.recovery_band.value}
            
            **Confidence:** {prediction.confidence_level}
            """)


# ============================================================
# TAB 2: COHORT PLANNING
# ============================================================

def render_cohort_tab():
    """Cohort planning view for managers"""
    
    st.header("📊 Cohort Planning")
    
    st.markdown("""
    Upload a CSV of injured personnel or use sample data to forecast 
    recovery timelines across a team.
    """)
    
    # Sample data option
    if st.button("Generate Sample Cohort (10 cases)"):
        np.random.seed(42)
        
        cases = []
        for i in range(10):
            case = CaseInput(
                age=np.random.randint(22, 48),
                trade=Trade(np.random.choice([t.value for t in Trade])),
                injury_type=InjuryType(np.random.choice([
                    "MSKI_minor", "MSKI_moderate", "MSKI_major", "MH_mild", "MH_moderate"
                ])),
                body_region=BodyRegion(np.random.choice([
                    "Lower_back", "Knee", "Shoulder", "Mental", "Other"
                ])),
                severity_score=np.random.randint(2, 9),
                prior_injury_count=np.random.randint(0, 4),
                prior_same_region=np.random.random() < 0.2,
                current_jmes=JMESStatus(np.random.choice(["MLD", "MND"], p=[0.7, 0.3])),
                months_since_injury=np.random.randint(0, 3),
                receiving_treatment=True
            )
            cases.append(case)
        
        # Get cohort summary
        summary = st.session_state.predictor.predict_cohort(cases)
        
        # Display
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Cases", summary["total_cases"])
        with col2:
            st.metric("Mean Recovery", f"{summary['mean_recovery_months']:.1f} mo")
        with col3:
            st.metric("Median Recovery", f"{summary['median_recovery_months']:.1f} mo")
        with col4:
            st.metric("Avg Full Recovery", f"{summary['avg_prob_full_recovery']:.0%}")
        
        # Band distribution
        st.subheader("Recovery Band Distribution")
        
        band_df = pd.DataFrame([
            {"Band": k, "Count": v}
            for k, v in summary["band_distribution"].items()
        ])
        
        fig = px.pie(
            band_df,
            values="Count",
            names="Band",
            color="Band",
            color_discrete_map={
                "Fast": "#2ecc71",
                "Medium": "#f39c12",
                "Slow": "#e74c3c",
                "Complex": "#9b59b6"
            }
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Individual predictions table
        st.subheader("Individual Predictions")
        
        pred_data = []
        for i, (case, pred) in enumerate(zip(cases, summary["predictions"])):
            pred_data.append({
                "ID": f"SP-{i+1:03d}",
                "Age": case.age,
                "Trade": case.trade.value,
                "Injury": case.injury_type.value,
                "Region": case.body_region.value,
                "JMES": case.current_jmes.value,
                "Expected (mo)": pred.expected_recovery_months,
                "Band": pred.recovery_band.value,
                "Full Recovery %": f"{pred.prob_full_recovery:.0%}",
                "Confidence": pred.confidence_level
            })
        
        pred_df = pd.DataFrame(pred_data)
        st.dataframe(pred_df, use_container_width=True)
        
        # Gantt-style timeline
        st.subheader("📅 Recovery Timeline Forecast")
        
        timeline_data = []
        for i, pred in enumerate(summary["predictions"]):
            timeline_data.append({
                "ID": f"SP-{i+1:03d}",
                "Start": 0,
                "Optimistic": pred.optimistic_months,
                "Realistic": pred.realistic_months,
                "Pessimistic": pred.pessimistic_months
            })
        
        timeline_df = pd.DataFrame(timeline_data)
        
        fig_timeline = go.Figure()
        
        for i, row in timeline_df.iterrows():
            # Optimistic bar
            fig_timeline.add_trace(go.Bar(
                y=[row["ID"]],
                x=[row["Optimistic"]],
                orientation='h',
                marker_color='#2ecc71',
                name='Optimistic' if i == 0 else None,
                showlegend=i == 0
            ))
            # Extension to realistic
            fig_timeline.add_trace(go.Bar(
                y=[row["ID"]],
                x=[row["Realistic"] - row["Optimistic"]],
                orientation='h',
                marker_color='#f39c12',
                name='Realistic' if i == 0 else None,
                showlegend=i == 0,
                base=row["Optimistic"]
            ))
            # Extension to pessimistic
            fig_timeline.add_trace(go.Bar(
                y=[row["ID"]],
                x=[row["Pessimistic"] - row["Realistic"]],
                orientation='h',
                marker_color='#e74c3c',
                name='Pessimistic' if i == 0 else None,
                showlegend=i == 0,
                base=row["Realistic"]
            ))
        
        fig_timeline.update_layout(
            barmode='stack',
            title="Recovery Timeline by Personnel",
            xaxis_title="Months",
            height=400
        )
        st.plotly_chart(fig_timeline, use_container_width=True)


# ============================================================
# TAB 3: MODEL SETTINGS
# ============================================================

def render_settings_tab():
    """Advanced model settings"""
    
    st.header("⚙️ Model Settings")
    
    st.markdown("""
    View and adjust the underlying model parameters.
    These settings affect all predictions.
    """)
    
    # Current config summary
    st.subheader("Current Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Recovery Bands**")
        for band, (lo, hi) in st.session_state.config.band_thresholds.items():
            st.markdown(f"- {band}: {lo}-{hi} months")
        
        st.markdown("**Age Modifiers**")
        for age, mod in st.session_state.config.age_modifiers.items():
            effect = "↑ slower" if mod > 1 else "↓ faster" if mod < 1 else "baseline"
            st.markdown(f"- {age}: {mod}x ({effect})")
    
    with col2:
        st.markdown("**Body Region Modifiers**")
        for region, mod in st.session_state.config.body_region_modifiers.items():
            effect = "↑ slower" if mod > 1 else "↓ faster" if mod < 1 else "baseline"
            st.markdown(f"- {region.replace('_', ' ')}: {mod}x ({effect})")
    
    st.markdown("---")
    
    # Injury profiles
    st.subheader("Injury Type Profiles")
    
    injury_data = []
    for name, profile in st.session_state.config.injury_profiles.items():
        injury_data.append({
            "Type": name,
            "Recovery (min)": profile.base_recovery_months[0],
            "Recovery (max)": profile.base_recovery_months[1],
            "Variance": profile.variance,
            "Recurrence Risk": f"{profile.recurrence_risk:.0%}",
            "MLD Prob": f"{profile.mld_probability:.0%}",
            "MND Prob": f"{profile.mnd_probability:.0%}"
        })
    
    injury_df = pd.DataFrame(injury_data)
    st.dataframe(injury_df, use_container_width=True)
    
    # Trade physical demand
    st.subheader("Trade Physical Demand")
    
    trade_data = []
    for trade, demand in st.session_state.config.trade_physical_demand.items():
        mod = st.session_state.config.physical_demand_modifiers[demand]
        trade_data.append({
            "Trade": trade,
            "Demand Level": demand,
            "Recovery Modifier": f"{mod}x"
        })
    
    trade_df = pd.DataFrame(trade_data)
    st.dataframe(trade_df, use_container_width=True)


# ============================================================
# MAIN APP
# ============================================================

def main():
    """Main application"""
    
    # Sidebar
    render_sidebar()
    
    # Header
    st.title("🏥 SEKHMET Recovery Predictor")
    st.markdown("""
    **Predict recovery trajectories for injured service personnel.**
    
    Adjust configuration in the sidebar, then use the tabs below.
    """)
    
    # Tabs
    tab1, tab2, tab3 = st.tabs([
        "🧑‍⚕️ Individual Prediction",
        "📊 Cohort Planning",
        "⚙️ Model Settings"
    ])
    
    with tab1:
        render_individual_tab()
    
    with tab2:
        render_cohort_tab()
    
    with tab3:
        render_settings_tab()
    
    # Footer
    st.markdown("---")
    st.caption("SEKHMET Recovery Predictor v1.0 | Configuration is session-based")


if __name__ == "__main__":
    main()
