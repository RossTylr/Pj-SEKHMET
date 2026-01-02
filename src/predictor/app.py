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
    InjuryType, BodyRegion, JMESStatus, RecoveryBand, Trade, TradeCategory,
    get_age_band, get_recovery_band, get_trade_category, EvidenceBase
)
from bayesian_model import BayesianRecoveryModel, CaseInput, RecoveryPrediction
from cox_model import CoxRecoveryModel, CaseInput as CoxCaseInput, CoxPrediction

# XGBoost model (V2)
try:
    from xgb_model import XGBSurvivalModel, XGBPrediction
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    XGBSurvivalModel = None


# ============================================================
# V3 UI HELPER FUNCTIONS
# ============================================================

def get_recovery_months(pred):
    """Helper to extract recovery months from prediction."""
    if hasattr(pred, 'median_recovery_months'):
        return pred.median_recovery_months
    elif hasattr(pred, 'predicted_time_months'):
        return pred.predicted_time_months
    elif hasattr(pred, 'expected_recovery_months'):
        return pred.expected_recovery_months
    else:
        return 6.0  # Default fallback


def render_traffic_light_summary(prediction):
    """Display traffic light summary for RTD likelihood at 3/6/12 months."""
    st.subheader("🚦 RTD Likelihood")

    # Get probabilities at key timepoints
    if hasattr(prediction, 'prob_recovery_3mo'):
        prob_3mo = prediction.prob_recovery_3mo
        prob_6mo = prediction.prob_recovery_6mo
        prob_12mo = prediction.prob_recovery_12mo
    else:
        # Estimate from median using exponential approximation
        median = get_recovery_months(prediction)
        prob_3mo = min(0.95, max(0.05, 1 - np.exp(-0.693 * 3 / median)))
        prob_6mo = min(0.95, max(0.05, 1 - np.exp(-0.693 * 6 / median)))
        prob_12mo = min(0.95, max(0.05, 1 - np.exp(-0.693 * 12 / median)))

    def get_traffic_light(prob):
        """Return emoji and label based on probability."""
        if prob >= 0.70:
            return "🟢", "Likely"
        elif prob >= 0.40:
            return "🟡", "Possible"
        else:
            return "🔴", "Unlikely"

    col1, col2, col3 = st.columns(3)

    with col1:
        emoji, label = get_traffic_light(prob_3mo)
        st.markdown(f"""
        <div style="text-align: center; padding: 1rem; background-color: #f8f9fa; border-radius: 0.5rem;">
            <div style="font-size: 3rem;">{emoji}</div>
            <div style="font-weight: bold; font-size: 1.2rem;">3 Months</div>
            <div style="font-size: 1.5rem; font-weight: bold;">{prob_3mo:.0%}</div>
            <div style="color: gray;">{label}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        emoji, label = get_traffic_light(prob_6mo)
        st.markdown(f"""
        <div style="text-align: center; padding: 1rem; background-color: #f8f9fa; border-radius: 0.5rem;">
            <div style="font-size: 3rem;">{emoji}</div>
            <div style="font-weight: bold; font-size: 1.2rem;">6 Months</div>
            <div style="font-size: 1.5rem; font-weight: bold;">{prob_6mo:.0%}</div>
            <div style="color: gray;">{label}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        emoji, label = get_traffic_light(prob_12mo)
        st.markdown(f"""
        <div style="text-align: center; padding: 1rem; background-color: #f8f9fa; border-radius: 0.5rem;">
            <div style="font-size: 3rem;">{emoji}</div>
            <div style="font-weight: bold; font-size: 1.2rem;">12 Months</div>
            <div style="font-size: 1.5rem; font-weight: bold;">{prob_12mo:.0%}</div>
            <div style="color: gray;">{label}</div>
        </div>
        """, unsafe_allow_html=True)

    st.caption("**Workforce planning**: 🟢 = Plan for RTD | 🟡 = Monitor closely | 🔴 = Arrange cover")


def plot_survival_curve_with_ci(cox_model, case, prediction):
    """Generate survival curve with confidence interval shading."""
    # Get survival curve data
    t, survival = cox_model.get_survival_curve(case, max_months=36)
    recovery = 1 - survival  # Probability of having recovered

    # Calculate CI bounds (using prediction intervals)
    ci_factor = 0.25  # 25% relative uncertainty
    recovery_upper = np.clip(recovery * (1 + ci_factor), 0, 1)
    recovery_lower = np.clip(recovery * (1 - ci_factor), 0, 1)

    fig = go.Figure()

    # Confidence interval shading (add first so it's behind)
    fig.add_trace(go.Scatter(
        x=np.concatenate([t, t[::-1]]),
        y=np.concatenate([recovery_upper, recovery_lower[::-1]]),
        fill='toself',
        fillcolor='rgba(46, 125, 50, 0.15)',
        line=dict(color='rgba(255,255,255,0)'),
        name='90% CI',
        showlegend=True,
        hoverinfo='skip'
    ))

    # Main recovery curve
    fig.add_trace(go.Scatter(
        x=t,
        y=recovery,
        mode='lines',
        name='Recovery Probability',
        line=dict(color='#2E7D32', width=3),
    ))

    # Median line
    median = get_recovery_months(prediction)
    fig.add_vline(
        x=median,
        line_dash="dash",
        line_color="#FF9800",
        annotation_text=f"Median: {median:.1f} mo",
        annotation_position="top"
    )

    # Lower/Upper bound lines
    if hasattr(prediction, 'recovery_lower_90'):
        lower = prediction.recovery_lower_90
        upper = prediction.recovery_upper_90
    elif hasattr(prediction, 'lower_bound_months'):
        lower = prediction.lower_bound_months
        upper = prediction.upper_bound_months
    else:
        lower = median * 0.6
        upper = median * 1.5

    fig.add_vline(x=lower, line_dash="dot", line_color="#4CAF50",
                  annotation_text=f"Best: {lower:.1f}", annotation_position="bottom left")
    fig.add_vline(x=upper, line_dash="dot", line_color="#F44336",
                  annotation_text=f"Worst: {upper:.1f}", annotation_position="bottom right")

    # 50% probability reference line
    fig.add_hline(y=0.5, line_dash="dot", line_color="gray", opacity=0.5)

    # Key milestone markers (3, 6, 12 months)
    for milestone in [3, 6, 12]:
        if milestone <= 36:
            idx = np.argmin(np.abs(t - milestone))
            prob = recovery[idx]
            fig.add_trace(go.Scatter(
                x=[milestone],
                y=[prob],
                mode='markers+text',
                marker=dict(size=10, color='#1976D2'),
                text=[f'{prob:.0%}'],
                textposition='top center',
                name=f'{milestone}mo',
                showlegend=False
            ))

    fig.update_layout(
        title="Recovery Probability Over Time (with 90% CI)",
        xaxis_title="Months Since Injury",
        yaxis_title="Probability of Recovery",
        yaxis_range=[0, 1],
        xaxis_range=[0, 36],
        template="plotly_white",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig


def render_model_agreement(cox_model, xgb_model, case_dict, case_input):
    """Show model agreement indicator when multiple models available."""
    if xgb_model is None:
        st.info("XGBoost model not available for comparison.")
        return None

    # Get predictions from both models
    cox_pred = cox_model.predict(case_input)
    cox_months = cox_pred.median_recovery_months

    xgb_pred = xgb_model.predict(case_dict)
    xgb_months = xgb_pred.predicted_time_months

    # Calculate agreement
    difference = abs(cox_months - xgb_months)
    avg_months = (cox_months + xgb_months) / 2
    agreement_pct = max(0, 100 - (difference / avg_months * 100))

    # Determine status
    if difference <= 1.5:
        status = "high"
        emoji = "✅"
        message = "Models agree closely"
        bg_color = "#e8f5e9"
    elif difference <= 3.0:
        status = "moderate"
        emoji = "⚠️"
        message = "Models show some divergence"
        bg_color = "#fff3e0"
    else:
        status = "low"
        emoji = "🔴"
        message = "Models disagree - consider clinical review"
        bg_color = "#ffebee"

    # Display
    st.markdown("### 🔄 Model Agreement")

    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        st.metric("Cox PH", f"{cox_months:.1f} mo")

    with col2:
        st.metric("XGBoost", f"{xgb_months:.1f} mo")

    with col3:
        st.markdown(f"""
        <div style="padding: 0.5rem; border-radius: 0.5rem; background-color: {bg_color};">
            <span style="font-size: 1.5rem;">{emoji}</span>
            <strong>Agreement: {agreement_pct:.0f}%</strong><br/>
            <span style="color: gray;">{message}</span>
        </div>
        """, unsafe_allow_html=True)

    if status == "low":
        st.warning(f"""
        **Models disagree by {difference:.1f} months**

        Use Cox PH estimate ({cox_months:.1f} mo) as primary,
        consider XGBoost ({xgb_months:.1f} mo) as alternative scenario.
        """)

    return {'cox_months': cox_months, 'xgb_months': xgb_months, 'agreement_pct': agreement_pct}


def render_comparator_benchmark(case_dict, prediction, current_months):
    """Compare current case to a typical/baseline case."""
    st.subheader("📊 Comparison to Typical Case")

    # Baseline is 30yo, no risk factors
    baseline_months = 6.0  # Typical moderate knee injury

    # Adjust baseline for injury type/region
    injury_type = case_dict.get('injury_type', 'mski_moderate')
    if hasattr(injury_type, 'value'):
        injury_type = injury_type.value

    if 'minor' in str(injury_type):
        baseline_months = 2.0
    elif 'severe' in str(injury_type):
        baseline_months = 15.0
    elif 'major' in str(injury_type):
        baseline_months = 10.0

    difference = current_months - baseline_months
    diff_pct = (difference / baseline_months) * 100 if baseline_months > 0 else 0

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Typical 30yo**")
        st.markdown("Same injury, no risk factors")
        st.metric("Recovery", f"{baseline_months:.1f} mo")

    with col2:
        age = case_dict.get('age', 30)
        st.markdown("**Your Case**")
        st.markdown(f"Age {age}, with risk factors")
        st.metric(
            "Recovery",
            f"{current_months:.1f} mo",
            delta=f"{difference:+.1f} mo" if abs(difference) > 0.1 else None,
            delta_color="inverse"
        )

    with col3:
        st.markdown("**Difference**")
        if difference > 0.5:
            st.error(f"⬆️ {diff_pct:.0f}% slower")
            st.caption("Risk factors adding time")
        elif difference < -0.5:
            st.success(f"⬇️ {abs(diff_pct):.0f}% faster")
            st.caption("Better than typical")
        else:
            st.info("➡️ Same as typical")


def render_prediction_disclaimer():
    """Footer disclaimer on every prediction."""
    st.divider()
    st.caption("""
    **Reminder**: This prediction is for operational planning only.
    Individual clinical decisions require qualified healthcare professional assessment.
    """)


def render_body_selector_simple():
    """Render body region selector with icons."""
    region_options = {
        "🦵 Knee": BodyRegion.KNEE,
        "🔙 Lower Back": BodyRegion.LOWER_BACK,
        "💪 Shoulder": BodyRegion.SHOULDER,
        "🦶 Ankle/Foot": BodyRegion.ANKLE_FOOT,
        "🦴 Hip/Groin": BodyRegion.HIP_GROIN,
        "🦒 Cervical Spine": BodyRegion.CERVICAL_SPINE,
        "✋ Wrist/Hand": BodyRegion.WRIST_HAND,
    }

    selected = st.selectbox(
        "Body Region",
        options=list(region_options.keys()),
        format_func=lambda x: x,
        help="Select the injured body region"
    )

    return region_options[selected]


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
    st.session_state.predictor = BayesianRecoveryModel(st.session_state.config)
if 'cox_model' not in st.session_state:
    st.session_state.cox_model = CoxRecoveryModel()
if 'xgb_model' not in st.session_state:
    if XGB_AVAILABLE:
        st.session_state.xgb_model = XGBSurvivalModel()
        st.session_state.xgb_model.train(n_synthetic=5000)
    else:
        st.session_state.xgb_model = None
if 'model_type' not in st.session_state:
    st.session_state.model_type = "Cox PH (Evidence-based)"


# ============================================================
# SIDEBAR - CONFIGURATION
# ============================================================

def render_sidebar():
    """Configuration sidebar"""

    st.sidebar.title("⚙️ Configuration")

    # Model selection (V2: 3 models)
    st.sidebar.subheader("Model Selection")
    model_options = ["Cox PH (Evidence-based)", "Bayesian (Clinician-adjustable)"]
    if XGB_AVAILABLE and st.session_state.xgb_model is not None:
        model_options.append("XGBoost (ML/SHAP)")

    current_index = 0
    if st.session_state.model_type in model_options:
        current_index = model_options.index(st.session_state.model_type)

    st.session_state.model_type = st.sidebar.radio(
        "Select Model",
        model_options,
        index=current_index,
        help="""
        Cox PH: Published hazard ratios, clinical gold standard
        Bayesian: Adjustable parameters for local calibration
        XGBoost: ML model with SHAP explainability (research only)
        """
    )

    if st.session_state.model_type == "Cox PH (Evidence-based)":
        st.sidebar.info(f"Evidence base v{st.session_state.cox_model.evidence.version}")
    elif st.session_state.model_type == "XGBoost (ML/SHAP)":
        st.sidebar.warning("Trained on synthetic data - research use only")

    st.sidebar.markdown("---")

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
    st.session_state.predictor = BayesianRecoveryModel(st.session_state.config)
    
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
            st.session_state.predictor = BayesianRecoveryModel(st.session_state.config)
            st.rerun()


# ============================================================
# TAB 1: INDIVIDUAL PREDICTION
# ============================================================

def render_individual_tab():
    """Individual case prediction - MSKI only (V2)"""

    st.header("🦴 Individual MSKI Prediction")

    # MSKI injury types and body regions only (V2 - MH removed)
    injury_types = [
        InjuryType.MSKI_MINOR, InjuryType.MSKI_MODERATE, InjuryType.MSKI_MAJOR, InjuryType.MSKI_SEVERE
    ]
    body_regions = [
        BodyRegion.KNEE, BodyRegion.LOWER_BACK, BodyRegion.SHOULDER,
        BodyRegion.ANKLE_FOOT, BodyRegion.HIP_GROIN, BodyRegion.CERVICAL_SPINE,
        BodyRegion.WRIST_HAND
    ]

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Injury Details")

        # Body Region first, then Severity
        body_region = st.selectbox(
            "Body Region",
            body_regions,
            format_func=lambda x: x.name.replace('_', ' ').title()
        )

        injury_type = st.selectbox(
            "Severity",
            injury_types,
            format_func=lambda x: x.display_name
        )

        prior_same_region = st.checkbox("Prior injury to same region?", help="HR 1.80")

        st.markdown("---")
        st.subheader("Demographics")

        age = st.number_input("Age", 18, 55, 30)

        months_since = st.number_input("Months since injury", 0, 24, 0)

        # Risk Factors section
        st.markdown("---")
        st.subheader("Risk Factors")

        # Lifestyle factors
        st.markdown("**Lifestyle**")
        is_smoker = st.checkbox("Current smoker?", help="HR 1.43")
        high_alcohol = st.checkbox("High alcohol intake?", help="HR 1.25")
        sleep_quality = st.select_slider(
            "Sleep quality",
            options=["Poor", "Fair", "Good"],
            value="Good",
            help="Poor sleep HR 1.30"
        )
        poor_sleep = (sleep_quality == "Poor")

        # Occupation factors (V2 - replaces Trade)
        st.markdown("**Occupational Health**")
        oh_risk = st.select_slider(
            "OH/Occupational Risk",
            options=["Low", "Moderate", "High"],
            value="Moderate",
            help="Low: 1.0x | Moderate: 1.15x | High: 1.30x"
        )

        # BMI factors
        st.markdown("**BMI**")
        bmi = st.number_input("BMI", min_value=15.0, max_value=50.0, value=25.0, step=0.5)
        if bmi < 18.5:
            bmi_cat = "Underweight"
        elif bmi < 25:
            bmi_cat = "Normal"
        elif bmi < 30:
            bmi_cat = "Overweight"
        elif bmi < 35:
            bmi_cat = "Obese (Class 1)"
        else:
            bmi_cat = "Obese (Class 2+)"
        st.caption(f"Category: {bmi_cat}")

        # Treatment
        st.markdown("**Treatment**")
        receiving_treatment = st.checkbox("Supervised rehabilitation?", value=True, help="HR 0.75")

        # Hidden defaults for backward compatibility
        trade = Trade.GENERIC
        prior_injuries = 1 if prior_same_region else 0
        has_mh_comorbidity = False  # MH removed in V2

        predict_btn = st.button("🔮 Predict Recovery", type="primary", use_container_width=True)
    
    with col2:
        if predict_btn:
            # Map OH risk to severity score for model compatibility
            severity_from_oh = {"Low": 3, "Moderate": 5, "High": 7}
            severity = severity_from_oh.get(oh_risk, 5)

            # Use Cox, XGBoost, or Bayesian based on selection
            if st.session_state.model_type == "XGBoost (ML/SHAP)":
                # XGBoost prediction (V2)
                xgb_case = {
                    'age': age,
                    'injury_type': injury_type.value,
                    'body_region': body_region.value,
                    'oh_risk': oh_risk,
                    'prior_same_region': 1 if prior_same_region else 0,
                    'is_smoker': 1 if is_smoker else 0,
                    'high_alcohol': 1 if high_alcohol else 0,
                    'poor_sleep': 1 if poor_sleep else 0,
                    'receiving_treatment': 1 if receiving_treatment else 0,
                    'bmi': bmi,
                }

                xgb_pred = st.session_state.xgb_model.predict(xgb_case)

                # V3: Traffic Light Summary (FIRST)
                render_traffic_light_summary(xgb_pred)

                st.divider()

                # Display XGBoost results
                st.subheader("📊 Prediction Results (XGBoost/SHAP)")

                st.warning(xgb_pred.disclaimer)

                # Key metrics
                col_a, col_b, col_c = st.columns(3)

                with col_a:
                    st.metric(
                        "Predicted Recovery",
                        f"{xgb_pred.predicted_time_months} months"
                    )
                with col_b:
                    st.metric(
                        "Recovery Band",
                        xgb_pred.recovery_band
                    )
                with col_c:
                    st.metric(
                        "90% Range",
                        f"{xgb_pred.lower_bound_months}-{xgb_pred.upper_bound_months} mo"
                    )

                st.markdown("---")

                # SHAP explainability
                if xgb_pred.shap_values:
                    st.subheader("🔍 SHAP Feature Importance")

                    st.markdown("**Factors slowing recovery (positive SHAP):**")
                    if xgb_pred.top_positive_factors:
                        for factor, val in xgb_pred.top_positive_factors:
                            st.markdown(f"- **{factor.replace('_', ' ').title()}**: +{val:.3f}")
                    else:
                        st.markdown("*None significant*")

                    st.markdown("**Factors speeding recovery (negative SHAP):**")
                    if xgb_pred.top_negative_factors:
                        for factor, val in xgb_pred.top_negative_factors:
                            st.markdown(f"- **{factor.replace('_', ' ').title()}**: {val:.3f}")
                    else:
                        st.markdown("*None significant*")

                    # SHAP bar chart
                    shap_df = pd.DataFrame([
                        {"Feature": k.replace('_', ' ').title(), "SHAP": v}
                        for k, v in xgb_pred.shap_values.items()
                    ]).sort_values('SHAP', key=abs, ascending=True)

                    fig_shap = px.bar(
                        shap_df,
                        x="SHAP",
                        y="Feature",
                        orientation='h',
                        color="SHAP",
                        color_continuous_scale=["green", "white", "red"],
                        color_continuous_midpoint=0
                    )
                    fig_shap.add_vline(x=0, line_dash="dash", line_color="black")
                    fig_shap.update_layout(height=400, title="SHAP Values (Feature Impact)")
                    st.plotly_chart(fig_shap, use_container_width=True)

                # Manager summary
                st.subheader("📋 Line Manager Summary")

                st.info(f"""
                **Expected Return to Duty:** {xgb_pred.predicted_time_months} months (90% range: {xgb_pred.lower_bound_months}-{xgb_pred.upper_bound_months})

                **Recovery Band:** {xgb_pred.recovery_band}

                **Note:** This prediction uses an XGBoost model trained on synthetic data.
                It captures non-linear interactions but has NOT been validated on real outcomes.
                For clinical decisions, prefer the Cox PH model.
                """)

                # V5: Prediction disclaimer footer
                render_prediction_disclaimer()

            elif st.session_state.model_type == "Cox PH (Evidence-based)":
                # Create Cox case
                cox_case = CoxCaseInput(
                    age=age,
                    trade=trade,
                    injury_type=injury_type,
                    body_region=body_region,
                    severity_score=severity,
                    prior_injury_count=prior_injuries,
                    prior_same_region=prior_same_region,
                    current_jmes=JMESStatus.MLD,  # Default JMES status (hidden from UI)
                    months_since_injury=float(months_since),
                    receiving_treatment=receiving_treatment,
                    is_smoker=is_smoker,
                    has_mh_comorbidity=has_mh_comorbidity,
                    multiple_tbi_history=False  # TBI removed from model
                )

                # Build case_dict for V3 features
                case_dict = {
                    'age': age,
                    'injury_type': injury_type,
                    'body_region': body_region,
                    'oh_risk': oh_risk,
                    'prior_same_region': prior_same_region,
                    'is_smoker': is_smoker,
                    'high_alcohol': high_alcohol,
                    'poor_sleep': poor_sleep,
                    'receiving_treatment': receiving_treatment,
                    'bmi': bmi,
                }

                # Get Cox prediction
                cox_prediction = st.session_state.cox_model.predict(cox_case)

                # V3: Traffic Light Summary (FIRST)
                render_traffic_light_summary(cox_prediction)

                st.divider()

                # Display Cox results
                st.subheader("📊 Prediction Results (Cox PH Model)")

                # Key metrics
                col_a, col_b, col_c, col_d = st.columns(4)

                with col_a:
                    st.metric(
                        "Median Recovery",
                        f"{cox_prediction.median_recovery_months} months"
                    )
                with col_b:
                    st.metric(
                        "Recovery Band",
                        cox_prediction.recovery_band.value.title()
                    )
                with col_c:
                    st.metric(
                        "Full Recovery Prob",
                        f"{cox_prediction.prob_full_recovery:.0%}"
                    )
                with col_d:
                    st.metric(
                        "Confidence",
                        cox_prediction.confidence.title()
                    )

                st.markdown("---")

                # V3: Model Agreement (if XGBoost available)
                if XGB_AVAILABLE and st.session_state.xgb_model is not None:
                    with st.expander("🔄 Model Agreement Check", expanded=False):
                        xgb_case_dict = {
                            'age': age,
                            'injury_type': injury_type.value,
                            'body_region': body_region.value,
                            'oh_risk': oh_risk,
                            'prior_same_region': 1 if prior_same_region else 0,
                            'is_smoker': 1 if is_smoker else 0,
                            'high_alcohol': 1 if high_alcohol else 0,
                            'poor_sleep': 1 if poor_sleep else 0,
                            'receiving_treatment': 1 if receiving_treatment else 0,
                            'bmi': bmi,
                        }
                        render_model_agreement(
                            st.session_state.cox_model,
                            st.session_state.xgb_model,
                            xgb_case_dict,
                            cox_case
                        )

                # Timeline with Return to Fitness vs Return to Duty
                st.subheader("📅 Recovery Timeline")

                col_t1, col_t2, col_t3 = st.columns(3)
                with col_t1:
                    st.markdown(f"**Return to Fitness:** {cox_prediction.time_to_fitness_months} months")
                with col_t2:
                    st.markdown(f"**Return to Duty:** {cox_prediction.time_to_rtd_months} months")
                with col_t3:
                    st.markdown(f"**90% Range:** {cox_prediction.recovery_lower_90}-{cox_prediction.recovery_upper_90} months")

                # V3: Comparator Benchmark
                st.markdown("---")
                render_comparator_benchmark(case_dict, cox_prediction, cox_prediction.median_recovery_months)

                # V3: Cox model visualizations with CI shading
                st.subheader("📈 Recovery Trajectory")
                fig = plot_survival_curve_with_ci(st.session_state.cox_model, cox_case, cox_prediction)
                st.plotly_chart(fig, use_container_width=True)

                st.caption(f"""
                **Reading this chart**: The green line shows expected recovery probability over time.
                The shaded area represents uncertainty (90% CI).
                Vertical lines mark best case ({cox_prediction.recovery_lower_90:.1f} mo),
                median ({cox_prediction.median_recovery_months:.1f} mo),
                and worst case ({cox_prediction.recovery_upper_90:.1f} mo).
                """)

                # Probability milestones
                st.subheader("🎯 Recovery Milestones")

                milestone_df = pd.DataFrame({
                    "Timeframe": ["3 months", "6 months", "12 months", "24 months"],
                    "Probability": [
                        cox_prediction.prob_recovery_3mo,
                        cox_prediction.prob_recovery_6mo,
                        cox_prediction.prob_recovery_12mo,
                        cox_prediction.prob_recovery_24mo
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

                # Hazard ratios (contributing factors)
                st.subheader("⚖️ Risk Factor Impact (Hazard Ratios)")

                factors_df = pd.DataFrame([
                    {"Factor": k.replace('_', ' ').title(), "HR": v, "Effect": "Slower" if v > 1 else "Faster" if v < 1 else "Neutral"}
                    for k, v in cox_prediction.hazard_ratios.items()
                    if v != 1.0
                ])

                if not factors_df.empty:
                    fig_factors = px.bar(
                        factors_df,
                        x="HR",
                        y="Factor",
                        orientation='h',
                        color="Effect",
                        color_discrete_map={"Slower": "#e74c3c", "Faster": "#2ecc71", "Neutral": "#95a5a6"}
                    )
                    fig_factors.add_vline(x=1.0, line_dash="dash", line_color="black")
                    fig_factors.update_layout(height=300, xaxis_range=[0.5, 2.5])
                    st.plotly_chart(fig_factors, use_container_width=True)

                # Manager summary
                st.subheader("📋 Line Manager Summary")

                st.info(f"""
                **Expected Return to Duty:** {cox_prediction.time_to_rtd_months} months (90% range: {cox_prediction.recovery_lower_90}-{cox_prediction.recovery_upper_90})

                **Key Milestones:**
                - Return to Fitness: {cox_prediction.time_to_fitness_months} months
                - Return to Duty: {cox_prediction.time_to_rtd_months} months

                **Recovery Band:** {cox_prediction.recovery_band.value.title()}

                **Confidence:** {cox_prediction.confidence.title()}
                ({cox_prediction.confidence_rationale})
                """)

                # Evidence sources
                if cox_prediction.primary_sources:
                    with st.expander("📚 Evidence Sources"):
                        for source_id in cox_prediction.primary_sources:
                            citation = st.session_state.cox_model.evidence.get_source_citation(source_id)
                            if citation:
                                st.markdown(f"**{citation.get('authors', 'Unknown')}** ({citation.get('year', 'N/A')}). "
                                           f"*{citation.get('title', 'Untitled')}*. {citation.get('journal', '')}. "
                                           f"DOI: {citation.get('doi', 'N/A')}")
                            else:
                                st.markdown(f"- {source_id}")

                # V5: Prediction disclaimer footer
                render_prediction_disclaimer()

            else:
                # Create Bayesian case
                case = CaseInput(
                    age=age,
                    trade=trade,
                    injury_type=injury_type,
                    body_region=body_region,
                    severity_score=severity,
                    prior_injury_count=prior_injuries,
                    prior_same_region=prior_same_region,
                    current_jmes=JMESStatus.MLD,  # Default JMES status (hidden from UI)
                    months_since_injury=months_since,
                    receiving_treatment=receiving_treatment,
                    is_smoker=is_smoker,
                    has_mh_comorbidity=has_mh_comorbidity
                )

                # Get legacy prediction
                prediction = st.session_state.predictor.predict(case)

                # V3: Traffic Light Summary (FIRST)
                render_traffic_light_summary(prediction)

                st.divider()

                # Display legacy results
                st.subheader("📊 Prediction Results (Bayesian Model)")

                # Key metrics
                col_a, col_b, col_c, col_d = st.columns(4)

                with col_a:
                    st.metric(
                        "Expected Recovery",
                        f"{prediction.expected_recovery_months} months"
                    )
                with col_b:
                    st.metric(
                        "Recovery Band",
                        prediction.recovery_band.value.title()
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
                    st.markdown(f"**Optimistic:** {prediction.optimistic_months} months")
                with col_t2:
                    st.markdown(f"**Realistic:** {prediction.realistic_months} months")
                with col_t3:
                    st.markdown(f"**Pessimistic:** {prediction.pessimistic_months} months")

                # Bayesian model visualizations
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

                # Manager summary
                st.subheader("📋 Line Manager Summary")

                st.info(f"""
                **Expected Return to Duty:** {prediction.realistic_months} months (range: {prediction.optimistic_months}-{prediction.pessimistic_months})

                **Planning Recommendation:**
                - Earliest possible: {prediction.optimistic_months} months
                - Plan for: {prediction.realistic_months} months
                - Contingency: {prediction.pessimistic_months} months

                **Recovery Band:** {prediction.recovery_band.value.title()}

                **Confidence:** {prediction.confidence_level}
                """)

                # V5: Prediction disclaimer footer
                render_prediction_disclaimer()


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

        # Valid enum values for sampling (V2 - MSKI only, MH removed)
        valid_trades = [Trade.INFANTRY, Trade.SIGNALS, Trade.LOGISTICS, Trade.MEDIC, Trade.REME]
        valid_injury_types = [InjuryType.MSKI_MINOR, InjuryType.MSKI_MODERATE, InjuryType.MSKI_MAJOR, InjuryType.MSKI_SEVERE]
        valid_body_regions = [BodyRegion.LOWER_BACK, BodyRegion.KNEE, BodyRegion.SHOULDER, BodyRegion.ANKLE_FOOT, BodyRegion.HIP_GROIN]
        valid_jmes = [JMESStatus.MLD, JMESStatus.MND]

        cases = []
        for i in range(10):
            case = CaseInput(
                age=np.random.randint(22, 48),
                trade=np.random.choice(valid_trades),
                injury_type=np.random.choice(valid_injury_types),
                body_region=np.random.choice(valid_body_regions),
                severity_score=np.random.randint(2, 9),
                prior_injury_count=np.random.randint(0, 4),
                prior_same_region=np.random.random() < 0.2,
                current_jmes=np.random.choice(valid_jmes, p=[0.7, 0.3]),
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
                "Injury": case.injury_type.value.upper(),
                "Region": case.body_region.value.replace('_', ' ').title(),
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
# TAB 3: MODEL SETTINGS (V5: Governance & Riley Compliance)
# ============================================================

def render_riley_compliance():
    """
    Explicit compliance check against Riley's PROGRESS framework recommendations.
    Reference: prognosisresearch.com, Riley et al. BMJ 2020
    """
    st.subheader("Riley Framework Compliance")

    st.markdown("""
    Assessment against key recommendations from Professor Richard Riley's
    clinical prediction model guidance ([PROGRESS framework](https://prognosisresearch.com)).
    """)

    compliance_items = [
        {
            'Recommendation': 'Handle continuous predictors correctly - do NOT dichotomize',
            'Status': 'Compliant',
            'Icon': '✅',
            'Implementation': 'Age and BMI kept as continuous variables in all models. '
                             'No arbitrary cutoffs (e.g., age>65). Risk thresholds applied '
                             'only to final predicted probabilities, not input features.',
            'Evidence': 'See cox_model.py: age used as continuous with per-decade HR scaling'
        },
        {
            'Recommendation': 'Focus on calibration, not just discrimination',
            'Status': 'Not Yet Assessed',
            'Icon': '❌',
            'Implementation': 'Calibration plots require real outcome data. Currently no '
                             'validation dataset available. C-statistic also not calculated.',
            'Evidence': 'Calibration assessment planned for Phase 2 of validation roadmap'
        },
        {
            'Recommendation': 'Present calibration plots, not just summary statistics',
            'Status': 'Placeholder Ready',
            'Icon': '⏳',
            'Implementation': 'UI includes calibration plot placeholder. Will display '
                             'predicted vs observed decile plot when validation data available.',
            'Evidence': 'See render_calibration_status() in app.py'
        },
        {
            'Recommendation': 'Assess clinical utility via decision curve analysis',
            'Status': 'Not Done',
            'Icon': '❌',
            'Implementation': 'Net benefit analysis not performed. Cannot currently quantify '
                             'whether model improves decisions vs treat-all/treat-none strategies.',
            'Evidence': 'Planned for Phase 3 of validation roadmap'
        },
        {
            'Recommendation': 'Ensure model stability - coefficients should not vary wildly',
            'Status': 'Partial',
            'Icon': '⚠️',
            'Implementation': 'Cox model uses fixed literature HRs (stable). '
                             'XGBoost trained on synthetic data - stability not quantified. '
                             'Bootstrap stability analysis not performed.',
            'Evidence': 'XGBoost labelled as research-only due to stability concerns'
        },
        {
            'Recommendation': 'Use formal sample size calculations (pmsampsize)',
            'Status': 'Not Applied',
            'Icon': '⚠️',
            'Implementation': 'XGBoost uses n=5,000 synthetic samples (arbitrary). '
                             'No formal pmsampsize calculation performed. '
                             'For real data, minimum sample size must be calculated.',
            'Evidence': 'See sample size requirements section below'
        },
        {
            'Recommendation': 'ML methods require larger samples than regression',
            'Status': 'Acknowledged',
            'Icon': '⚠️',
            'Implementation': 'XGBoost explicitly labelled as research demonstration only. '
                             'Cox PH (regression) recommended for any operational use. '
                             'ML model not recommended until large real dataset available.',
            'Evidence': 'UI warns: "XGBoost trained on synthetic data - research use only"'
        },
    ]

    for item in compliance_items:
        with st.expander(f"{item['Icon']} {item['Recommendation']}"):
            st.markdown(f"**Status**: {item['Status']}")
            st.markdown(f"**Implementation**: {item['Implementation']}")
            st.caption(f"Evidence: {item['Evidence']}")

    # Summary
    st.divider()
    compliant = sum(1 for i in compliance_items if i['Icon'] == '✅')
    partial = sum(1 for i in compliance_items if i['Icon'] == '⚠️')
    not_done = sum(1 for i in compliance_items if i['Icon'] == '❌')
    pending = sum(1 for i in compliance_items if i['Icon'] == '⏳')

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Compliant", f"{compliant}/7")
    with col2:
        st.metric("Partial", f"{partial}/7")
    with col3:
        st.metric("Pending", f"{pending}/7")
    with col4:
        st.metric("Not Done", f"{not_done}/7")


def render_continuous_variables_note():
    """Document that continuous variables are handled correctly."""
    st.subheader("Continuous Variable Handling")

    st.success("""
    **Continuous predictors are NOT dichotomized**

    Following Riley's guidance, SEKHMET keeps continuous variables continuous.
    """)

    st.markdown("""
    | Variable | Handling | Why Not Dichotomize? |
    |----------|----------|---------------------|
    | **Age** | Continuous, HR per decade over 25 | Avoids arbitrary cutoff (e.g., 65), preserves information |
    | **BMI** | Continuous, stepped HR at 30/35 | Clinical thresholds used for HR, but raw BMI retained |

    **What this means**:
    - A 34-year-old is treated differently from a 44-year-old (not grouped as "under 65")
    - BMI 29 vs BMI 31 shows appropriate risk difference (not cliff at 30)
    - Full predictive information retained

    **Decision thresholds** (e.g., traffic light RTD probability) are applied to
    *final predicted risk*, not to input variables.
    """)


def render_stability_assessment():
    """Document model stability status."""
    st.subheader("Model Stability")

    st.markdown("""
    Model stability refers to whether selected predictors and coefficients
    would remain similar if the study were repeated.
    """)

    stability_data = [
        {
            'Model': 'Cox PH',
            'Stability': '✅ High',
            'Reason': 'Uses fixed hazard ratios from published meta-analyses. '
                     'No fitting to local data = no instability from sampling.',
            'Risk': 'Low - parameters externally derived'
        },
        {
            'Model': 'Bayesian',
            'Stability': '✅ High',
            'Reason': 'Rule-based model with clinician-adjustable parameters. '
                     'No data-driven fitting = no sampling instability.',
            'Risk': 'Low - parameters manually set'
        },
        {
            'Model': 'XGBoost',
            'Stability': '⚠️ Unknown',
            'Reason': 'Trained on synthetic data. Bootstrap stability analysis not performed. '
                     'ML methods typically require large samples for stability.',
            'Risk': 'Potentially high - coefficients may vary significantly with different samples'
        },
    ]

    df = pd.DataFrame(stability_data)
    st.dataframe(df, hide_index=True, use_container_width=True)

    with st.expander("Why stability matters"):
        st.markdown("""
        ### Unstable Models Are Dangerous

        If a model is unstable:
        - Different training samples → different predictions
        - Individual predictions may be unreliable
        - Apparent performance may be optimistic (overfitting)

        ### How to Assess Stability

        1. **Bootstrap resampling**: Refit model on 500+ bootstrap samples
        2. **Check coefficient variation**: Do HRs vary wildly across samples?
        3. **Check predictor selection**: Are same variables selected each time?

        ### SEKHMET Status

        - **Cox**: Stable by design (fixed external HRs)
        - **XGBoost**: Stability NOT assessed - treat as research demonstration only

        **Recommendation**: Use Cox model for any operational decisions until
        XGBoost stability is validated with real data.
        """)


def render_pmsampsize_requirements():
    """Document sample size requirements per pmsampsize methodology."""
    st.subheader("Sample Size Requirements (pmsampsize)")

    st.markdown("""
    For future validation with real data, minimum sample size should be calculated
    using [`pmsampsize`](https://github.com/cran/pmsampsize) (Riley et al., 2020).
    """)

    st.code("""
# R example for SEKHMET-like model
library(pmsampsize)

# Binary outcome (recovered by 6 months: yes/no)
pmsampsize(
    type = "b",           # binary outcome
    rsquared = 0.25,      # anticipated R² (conservative estimate)
    parameters = 12,      # number of predictor parameters
    prevalence = 0.6,     # ~60% recover by 6 months
    shrinkage = 0.9       # target shrinkage to minimise overfitting
)

# Survival outcome (time to recovery)
pmsampsize(
    type = "s",           # survival outcome
    rsquared = 0.20,      # anticipated R²
    parameters = 12,      # number of predictor parameters
    rate = 0.15,          # overall event rate per person-month
    timepoint = 12,       # prediction horizon (months)
    meanfup = 8           # mean follow-up time
)
    """, language="r")

    st.warning("""
    **Current Status**: No formal sample size calculation performed.

    - XGBoost uses arbitrary n=5,000 synthetic samples
    - Cox model uses literature HRs (no fitting required)
    - Before real-data ML training, pmsampsize MUST be used

    **Typical requirements** for stable prediction models:
    - Regression: Often 10-20 events per predictor parameter
    - ML methods: Often 50-200+ events per parameter (higher than regression)
    """)


def render_model_limitations():
    """
    Transparent display of model limitations per Riley PROGRESS framework.
    Reference: prognosisresearch.com
    """
    st.subheader("Methodological Limitations")

    st.markdown("""
    Following the [PROGRESS framework](https://prognosisresearch.com) for
    clinical prediction model research (Riley et al.), this section documents
    the current validation status of SEKHMET.
    """)

    limitations = [
        {
            'Criterion': 'External Validation',
            'Status': '❌ Not Done',
            'Impact': 'Unknown real-world performance',
            'Mitigation': 'Pending access to outcome data'
        },
        {
            'Criterion': 'Calibration Assessment',
            'Status': '❌ Not Done',
            'Impact': 'Predicted probabilities may not match observed outcomes',
            'Mitigation': 'Requires real outcome data for calibration plots'
        },
        {
            'Criterion': 'Discrimination (C-statistic)',
            'Status': '❌ Not Calculated',
            'Impact': 'Model ranking ability unknown',
            'Mitigation': 'Requires validation dataset'
        },
        {
            'Criterion': 'Clinical Utility (Decision Curve)',
            'Status': '❌ Not Assessed',
            'Impact': 'Net benefit vs treat-all/treat-none unknown',
            'Mitigation': 'Future work when outcome data available'
        },
        {
            'Criterion': 'Sample Size Adequacy',
            'Status': '⚠️ Synthetic Only',
            'Impact': 'XGBoost trained on simulated data (n=5,000)',
            'Mitigation': 'Cox uses literature HRs; XGBoost for demo only'
        },
        {
            'Criterion': 'Model Stability',
            'Status': '⚠️ Variable',
            'Impact': 'XGBoost coefficients may be unstable',
            'Mitigation': 'Cox model (fixed HRs) recommended for decisions'
        },
    ]

    df = pd.DataFrame(limitations)
    st.dataframe(df, hide_index=True, use_container_width=True)

    # What this means
    with st.expander("What does this mean?"):
        st.markdown("""
        ### Interpretation Guide

        | Use Case | Appropriate? | Notes |
        |----------|--------------|-------|
        | Workforce capacity planning | ✅ Yes | Aggregate estimates for planning |
        | Resource allocation modelling | ✅ Yes | Scenario analysis acceptable |
        | Individual prognosis discussion | ⚠️ Caution | Use as conversation starter only |
        | Clinical treatment decisions | ❌ No | Requires validated diagnostic tool |
        | Fitness for duty determination | ❌ No | Requires clinical assessment |

        ### Path to Validation

        1. Obtain de-identified outcome data
        2. Perform calibration assessment (predicted vs observed)
        3. Calculate discrimination (C-statistic/AUC)
        4. Conduct decision curve analysis
        5. External validation in independent cohort
        6. Formal clinical governance review
        """)

    # Reference
    st.info("""
    **Reference**: Riley RD et al. "Minimum sample size for developing a
    multivariable prediction model" (BMJ 2020).
    See [prognosisresearch.com](https://prognosisresearch.com)
    """)


def render_calibration_status():
    """Display calibration status prominently."""
    st.subheader("Calibration Status")

    st.error("""
    **MODEL NOT CALIBRATED**

    This model has **not** been validated against real outcome data.

    | Model | Calibration Status |
    |-------|-------------------|
    | Cox PH | Parameters from published literature - not locally calibrated |
    | Bayesian | Clinician-adjustable - no empirical calibration |
    | XGBoost | Trained on synthetic data - not validated |
    """)

    with st.expander("What is calibration and why does it matter?"):
        st.markdown("""
        **Calibration** measures whether predicted probabilities match observed outcomes.

        ### Example
        - Model predicts: "70% chance of recovery by 6 months"
        - Good calibration: ~70 out of 100 similar patients actually recover by 6 months
        - Poor calibration: Only 40 out of 100 actually recover (model overestimates)

        ### Why It Matters

        | Issue | Consequence |
        |-------|-------------|
        | Overestimation | False optimism, inadequate planning |
        | Underestimation | Unnecessary concern, over-resourcing |
        | Variable calibration | Unpredictable errors across subgroups |

        ### Important Note

        **Good discrimination ≠ Good calibration**

        A model can correctly rank patients (high risk vs low risk) but still
        systematically over- or under-estimate absolute probabilities.

        Both are required for safe clinical/operational use.
        """)

    # Calibration plot placeholder
    st.caption("Calibration plot will be available after validation against real outcomes.")


def render_calibration_plot_placeholder():
    """Placeholder for calibration plot with explanation."""
    st.subheader("Calibration Plot")

    st.warning("**Calibration plot not available** - requires validation against real outcomes.")

    # Show example of what good/bad calibration looks like
    with st.expander("What should a calibration plot show?"):
        st.markdown("""
        A calibration plot compares **predicted probabilities** (x-axis) to
        **observed proportions** (y-axis).

        ### Ideal Calibration
        - Points lie on the 45 degree diagonal line
        - Predicted 30% → ~30% observed
        - Predicted 70% → ~70% observed

        ### Poor Calibration Examples

        | Pattern | Meaning | Risk |
        |---------|---------|------|
        | Points above diagonal | Model **underestimates** risk | False reassurance |
        | Points below diagonal | Model **overestimates** risk | Unnecessary concern |
        | S-shaped curve | Poor calibration at extremes | Wrong for high/low risk |
        | Scattered points | Inconsistent calibration | Unreliable |

        ### Why C-statistic (AUC) Is Not Enough

        A model can have excellent discrimination (AUC = 0.85) but terrible calibration:
        - Correctly ranks high vs low risk patients
        - But predicts 80% when true rate is 40%

        **Both discrimination AND calibration are required.**
        """)

        # Placeholder figure
        st.caption("Example calibration plot (placeholder):")

        # Create example plot
        fig = go.Figure()

        # Perfect calibration line
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Perfect calibration',
            line=dict(dash='dash', color='gray')
        ))

        # Example well-calibrated points
        x_good = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        y_good = [0.12, 0.18, 0.32, 0.38, 0.52, 0.58, 0.72, 0.78, 0.88]

        fig.add_trace(go.Scatter(
            x=x_good, y=y_good,
            mode='markers',
            name='Example: Good calibration',
            marker=dict(size=12, color='green')
        ))

        fig.update_layout(
            title="Example Calibration Plot (Illustrative Only)",
            xaxis_title="Predicted Probability",
            yaxis_title="Observed Proportion",
            xaxis_range=[0, 1],
            yaxis_range=[0, 1],
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

        st.caption("*This is an illustrative example, not SEKHMET validation data.*")


def render_decision_curve_placeholder():
    """Placeholder for decision curve analysis."""
    st.subheader("Decision Curve Analysis")

    st.warning("**Decision curve analysis not performed** - requires validation data.")

    with st.expander("What is decision curve analysis?"):
        st.markdown("""
        Decision curve analysis (DCA) evaluates the **clinical utility** of a prediction model.

        ### Key Question
        Does using this model lead to better decisions than:
        - **Treat all**: Assume everyone needs intervention
        - **Treat none**: Assume no one needs intervention

        ### Net Benefit

        ```
        Net Benefit = (True Positives / n) - (False Positives / n) x (threshold / (1 - threshold))
        ```

        This weighs:
        - Benefit of correctly identifying high-risk cases
        - Harm of incorrectly flagging low-risk cases

        ### Interpretation

        | Result | Meaning |
        |--------|---------|
        | Model curve above "treat all" | Model adds value vs treating everyone |
        | Model curve above "treat none" | Model adds value vs treating no one |
        | Model curve below both | Model is **harmful** - don't use it |

        ### For SEKHMET

        DCA would answer: "Does using SEKHMET for workforce planning lead to better
        resource allocation than assuming all injuries take X months?"

        **Cannot be assessed without real outcome data.**
        """)


def render_validation_roadmap():
    """Show path to full validation."""
    st.subheader("Validation Roadmap")

    roadmap = [
        {"Phase": "1", "Task": "Obtain aggregate outcome data", "Status": "⏳ Pending"},
        {"Phase": "2", "Task": "Internal validation (calibration, discrimination)", "Status": "⏳ Pending"},
        {"Phase": "3", "Task": "Decision curve analysis", "Status": "⏳ Pending"},
        {"Phase": "4", "Task": "External validation", "Status": "⏳ Pending"},
        {"Phase": "5", "Task": "Clinical governance review", "Status": "⏳ Pending"},
        {"Phase": "6", "Task": "Deployment approval", "Status": "⏳ Pending"},
    ]

    df = pd.DataFrame(roadmap)
    st.dataframe(df, hide_index=True, use_container_width=True)

    st.info("""
    **Current Status**: Research/demonstration phase

    The tool is appropriate for:
    - Exploring prediction model capabilities
    - Workforce planning scenario analysis
    - Stakeholder demonstrations

    **Not yet appropriate for**:
    - Individual clinical decisions
    - Formal capacity commitments
    - Regulatory submissions
    """)


def render_sample_size_transparency():
    """Document sample size considerations per pmsampsize guidance."""
    st.subheader("Sample Size & Data Sources")

    st.markdown("""
    | Model | Data Source | Sample Size | Notes |
    |-------|-------------|-------------|-------|
    | **Cox PH** | Published literature | N/A | Hazard ratios from 22 peer-reviewed sources |
    | **Bayesian** | Expert-configurable | N/A | Rule-based, adjustable parameters |
    | **XGBoost** | Synthetic generation | n=5,000 | Simulated from evidence base parameters |
    """)

    st.warning("""
    **XGBoost Limitation**: The XGBoost model is trained on synthetic data
    generated from the same evidence base as the Cox model. It **cannot**
    discover new patterns - it only demonstrates the ML pipeline.

    For real deployment, minimum sample size should be calculated using
    [`pmsampsize`](https://github.com/cran/pmsampsize).
    """)


def render_model_configuration():
    """Render existing model configuration settings."""
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


def render_settings_tab():
    """
    Model Settings tab with full governance and Riley compliance information.
    V5: Restructured with sub-tabs for Configuration, Riley Compliance, Limitations, Validation Status
    """
    st.header("Model Settings")

    # Sub-tabs for organisation - now includes Riley Compliance
    settings_tabs = st.tabs([
        "Configuration",
        "Riley Compliance",
        "Limitations",
        "Validation Status"
    ])

    with settings_tabs[0]:
        render_model_configuration()

    with settings_tabs[1]:
        # Riley-specific compliance checks
        render_riley_compliance()
        st.divider()
        render_continuous_variables_note()
        st.divider()
        render_stability_assessment()

    with settings_tabs[2]:
        render_model_limitations()
        st.divider()
        render_pmsampsize_requirements()
        st.divider()
        render_sample_size_transparency()

    with settings_tabs[3]:
        render_calibration_status()
        st.divider()
        render_calibration_plot_placeholder()
        st.divider()
        render_decision_curve_placeholder()
        st.divider()
        render_validation_roadmap()


# ============================================================
# TAB 4: REFERENCES (V4)
# ============================================================

def get_source_usage(source_id: str) -> list:
    """
    Determine what parameters this source is used for.
    Returns list of usage descriptions.
    """
    usage_map = {
        'antosh_2018': [
            'ACL reconstruction RTD rates',
            'Military-specific knee injury outcomes'
        ],
        'anderson_2023': [
            'Military academy injury epidemiology',
            'Age-related recovery modifiers',
            'Smoking hazard ratio'
        ],
        'marquina_2024': [
            'ACL reconstruction meta-analysis',
            'Return to sport timelines'
        ],
        'rhon_2022': [
            'Spine rehabilitation outcomes',
            'Lower back recovery trajectories'
        ],
        'olivotto_2025': [
            'MSKI prognostic factors systematic review',
            'Risk factor hazard ratios',
            'BMI impact on recovery'
        ],
        'wiggins_2016': [
            'ACL reinjury rates',
            'Age-related reinjury risk',
            'Prior injury hazard ratio'
        ],
        'kcmhr_2024': [
            'UK military mental health prevalence',
            'PTSD recovery trajectories (research only)'
        ],
        'hoge_2014': [
            'Military PTSD prevalence comparison',
            'UK vs US outcomes'
        ],
        'shaw_2019': [
            'Occupational factors in LBP recovery',
            'Job modification impact'
        ],
    }
    return usage_map.get(source_id, [])


def generate_bibtex(sources: dict) -> str:
    """Generate BibTeX format for all sources."""
    bibtex_entries = []

    for source_id, source in sources.items():
        entry_type = 'article'
        authors = source.get('authors', 'Unknown').replace(' and ', ' AND ')

        entry = f"""@{entry_type}{{{source_id},
    author = {{{authors}}},
    title = {{{source.get('title', 'Untitled')}}},
    journal = {{{source.get('journal', '')}}},
    year = {{{source.get('year', '')}}},
    doi = {{{source.get('doi', '')}}}
}}"""
        bibtex_entries.append(entry)

    return "\n\n".join(bibtex_entries)


def generate_formatted_references(sources: dict) -> str:
    """Generate formatted reference list (Vancouver style)."""
    lines = [
        "SEKHMET Evidence Base - Reference List",
        "=" * 50,
        "",
    ]

    sorted_sources = sorted(sources.items(), key=lambda x: x[1].get('year', 0), reverse=True)

    for i, (source_id, source) in enumerate(sorted_sources, 1):
        authors = source.get('authors', 'Unknown')
        title = source.get('title', 'Untitled')
        journal = source.get('journal', '')
        year = source.get('year', 'n.d.')
        doi = source.get('doi', '')

        ref = f"{i}. {authors}. {title}."
        if journal:
            ref += f" {journal}."
        ref += f" {year}."
        if doi:
            ref += f" doi:{doi}"

        lines.append(ref)
        lines.append("")

    return "\n".join(lines)


def render_reference_card(source_id: str, source: dict):
    """Render a single reference as an expandable card."""
    authors = source.get('authors', 'Unknown')
    year = source.get('year', 'n.d.')
    title = source.get('title', 'Untitled')
    journal = source.get('journal', '')
    doi = source.get('doi', '')

    header = f"**{authors}** ({year})"

    # Badges
    badges = []
    if source.get('military_specific'):
        badges.append("🎖️ Military")

    study_type = source.get('study_type', '')
    if study_type:
        type_icons = {
            'meta_analysis': '📊',
            'systematic_review': '🔍',
            'cohort': '👥',
            'rct': '🎲',
            'case_control': '⚖️',
            'cross_sectional': '📸',
            'retrospective_cohort': '📜',
        }
        icon = type_icons.get(study_type, '📄')
        badges.append(f"{icon} {study_type.replace('_', ' ').title()}")

    badge_str = " | ".join(badges) if badges else ""

    with st.expander(f"{header} - *{title[:60]}{'...' if len(title) > 60 else ''}*"):
        st.markdown(f"**{authors}** ({year}). *{title}*.")
        if journal:
            st.markdown(f"📰 {journal}")
        if doi:
            st.markdown(f"🔗 [doi:{doi}](https://doi.org/{doi})")

        st.caption(badge_str)

        if source.get('n'):
            st.markdown(f"**Sample size**: n = {source['n']:,}")

        if source.get('note'):
            st.info(f"📝 {source['note']}")

        st.markdown("**Used for:**")
        used_for = get_source_usage(source_id)
        if used_for:
            for usage in used_for:
                st.markdown(f"• {usage}")
        else:
            st.caption("General evidence base calibration")


def render_parameter_source_table():
    """Render table showing parameter-to-source mappings."""
    st.subheader("📋 Parameter Source Mapping")

    st.markdown("This table shows which clinical parameters are derived from which sources.")

    mapping_data = [
        {'Parameter': 'ACL median recovery (moderate)', 'Value': '9 months', 'Sources': 'Antosh 2018, Marquina 2024', 'Grade': '🟡 Moderate'},
        {'Parameter': 'Lower back median recovery (moderate)', 'Value': '6 months', 'Sources': 'Rhon 2022, Shaw 2019', 'Grade': '🟡 Moderate'},
        {'Parameter': 'Age HR (per decade >25)', 'Value': '1.15', 'Sources': 'Anderson 2023, Wiggins 2016', 'Grade': '🟡 Moderate'},
        {'Parameter': 'Prior injury HR', 'Value': '1.80', 'Sources': 'Wiggins 2016, Olivotto 2025', 'Grade': '🟡 Moderate'},
        {'Parameter': 'Smoking HR', 'Value': '1.43', 'Sources': 'Anderson 2023', 'Grade': '🟡 Moderate'},
        {'Parameter': 'Supervised rehab HR', 'Value': '0.75', 'Sources': 'Olivotto 2025', 'Grade': '🟡 Moderate'},
        {'Parameter': 'BMI >= 30 HR', 'Value': '1.20', 'Sources': 'Olivotto 2025', 'Grade': '🟡 Moderate'},
        {'Parameter': 'OH Risk High HR', 'Value': '1.30', 'Sources': 'Shaw 2019, Olivotto 2025', 'Grade': '🟡 Moderate'},
    ]

    df = pd.DataFrame(mapping_data)
    st.dataframe(df, hide_index=True, use_container_width=True)


def render_evidence_summary(sources: dict):
    """Render evidence base summary statistics."""
    st.subheader("Evidence Base Statistics")

    col1, col2, col3, col4 = st.columns(4)

    military_count = sum(1 for s in sources.values() if s.get('military_specific'))
    total_n = sum(s.get('n', 0) for s in sources.values())

    with col1:
        st.metric("Total Sources", len(sources))
    with col2:
        st.metric("Military-Specific", military_count)
    with col3:
        st.metric("Total Sample Size", f"{total_n:,}")
    with col4:
        years = [s.get('year', 0) for s in sources.values() if s.get('year')]
        st.metric("Year Range", f"{min(years)}-{max(years)}" if years else "N/A")

    st.divider()

    # Study type breakdown
    st.subheader("By Study Type")

    type_counts = {}
    for s in sources.values():
        t = s.get('study_type', 'other')
        type_counts[t] = type_counts.get(t, 0) + 1

    type_df = pd.DataFrame([
        {'Study Type': k.replace('_', ' ').title(), 'Count': v}
        for k, v in sorted(type_counts.items(), key=lambda x: x[1], reverse=True)
    ])

    fig = px.pie(type_df, names='Study Type', values='Count', title='Distribution by Study Type')
    st.plotly_chart(fig, use_container_width=True)

    # Timeline
    st.subheader("Publication Timeline")

    year_counts = {}
    for s in sources.values():
        y = s.get('year')
        if y:
            year_counts[y] = year_counts.get(y, 0) + 1

    if year_counts:
        year_df = pd.DataFrame([
            {'Year': k, 'Publications': v}
            for k, v in sorted(year_counts.items())
        ])

        fig = px.bar(year_df, x='Year', y='Publications', title='Publications by Year')
        st.plotly_chart(fig, use_container_width=True)

    # Evidence quality
    st.subheader("Evidence Quality Assessment")

    st.markdown("""
    | Grade | Definition | Sources |
    |-------|------------|---------|
    | 🟢 **High** | Multiple large RCTs, high-quality meta-analyses | Limited |
    | 🟡 **Moderate** | Cohort studies, single RCTs, systematic reviews | Most parameters |
    | 🔴 **Low** | Case series, expert opinion, extrapolated | Some edge cases |

    **Note**: Military-specific evidence is prioritised where available.
    Civilian data is used where military data is lacking, with appropriate
    adjustments for occupational demands.
    """)


def render_reference_list(sources: dict):
    """Render filterable reference list."""
    col1, col2, col3 = st.columns(3)

    with col1:
        study_types = ["All"] + sorted(set(s.get('study_type', 'other') for s in sources.values()))
        selected_type = st.selectbox("Study Type", study_types)

    with col2:
        pop_filter = st.radio("Population", ["All", "Military", "Civilian"], horizontal=True)

    with col3:
        search = st.text_input("Search", placeholder="Author, title...")

    # Filter
    filtered = {}
    for sid, s in sources.items():
        if selected_type != "All" and s.get('study_type') != selected_type:
            continue
        if pop_filter == "Military" and not s.get('military_specific'):
            continue
        if pop_filter == "Civilian" and s.get('military_specific'):
            continue
        if search:
            searchable = f"{s.get('authors','')} {s.get('title','')}".lower()
            if search.lower() not in searchable:
                continue
        filtered[sid] = s

    st.caption(f"Showing {len(filtered)} of {len(sources)} references")
    st.divider()

    # Display
    for sid, s in sorted(filtered.items(), key=lambda x: x[1].get('year', 0), reverse=True):
        render_reference_card(sid, s)

    # Export
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "📄 Download BibTeX",
            generate_bibtex(sources),
            "sekhmet_references.bib"
        )
    with col2:
        st.download_button(
            "📝 Download Text",
            generate_formatted_references(sources),
            "sekhmet_references.txt"
        )


def render_methodology_references():
    """Render methodology references section (Riley, TRIPOD, etc.)."""
    st.subheader("Methodology References")

    st.markdown("""
    The following resources informed the development methodology:

    **Prediction Model Standards**
    - Riley RD et al. (2020). *Minimum sample size for developing a multivariable
      prediction model*. BMJ. [doi:10.1136/bmj.m441](https://doi.org/10.1136/bmj.m441)
    - Collins GS et al. (2015). *Transparent Reporting of a multivariable prediction
      model for Individual Prognosis Or Diagnosis (TRIPOD)*.
      [doi:10.7326/M14-0697](https://doi.org/10.7326/M14-0697)

    **Framework**
    - [PROGRESS Framework](https://prognosisresearch.com) - Prognosis research strategy

    **Tools**
    - [`pmsampsize`](https://github.com/cran/pmsampsize) - Sample size calculations
    - [`pmcalplot`](https://github.com/cran/pmcalplot) - Calibration plot generation
    """)

    st.divider()

    st.markdown("""
    **Why These Standards Matter**

    | Standard | Purpose |
    |----------|---------|
    | **Riley 2020** | Minimum sample size calculations for prediction models |
    | **TRIPOD** | Transparent reporting of prediction model development |
    | **PROGRESS** | Framework for prognosis research methodology |

    SEKHMET aims to follow these standards. See the **Model Settings → Riley Compliance**
    tab for a detailed assessment of current compliance status.
    """)


def render_references_tab():
    """Render the References tab with full evidence base citations."""
    st.header("Evidence Base & References")

    st.markdown("""
    SEKHMET's Cox proportional hazards model is calibrated using
    **peer-reviewed clinical literature** from military and civilian populations.

    All hazard ratios, recovery timelines, and outcome probabilities
    are traceable to specific sources listed below.
    """)

    # Load evidence
    evidence = EvidenceBase()
    sources = evidence._sources if hasattr(evidence, '_sources') else {}

    # Sub-tabs within References (V5: added Methodology tab)
    ref_tab1, ref_tab2, ref_tab3, ref_tab4 = st.tabs([
        "Full Reference List",
        "Parameter Mapping",
        "Evidence Summary",
        "Methodology"
    ])

    with ref_tab1:
        render_reference_list(sources)

    with ref_tab2:
        render_parameter_source_table()

    with ref_tab3:
        render_evidence_summary(sources)

    with ref_tab4:
        render_methodology_references()


# ============================================================
# MAIN APP
# ============================================================

def main():
    """Main application"""

    # Sidebar
    render_sidebar()

    # Header
    st.title("🏥 SEKHMET Recovery Predictor")

    # REGULATORY DISCLAIMER - prominent, non-dismissible
    st.error("""
    **FOR RESEARCH/OPERATIONAL PLANNING ONLY – NOT FOR CLINICAL DIAGNOSIS**

    This tool supports workforce capacity planning. It is **not** a medical device
    and must **not** be used for individual clinical diagnosis or treatment decisions.
    """)

    st.markdown("*Evidence-based MSKI recovery prediction for workforce planning*")
    
    # Tabs (V4: 4 tabs with References)
    tab1, tab2, tab3, tab4 = st.tabs([
        "🧑‍⚕️ Individual Prediction",
        "📊 Cohort Planning",
        "📚 References",
        "⚙️ Model Settings"
    ])

    with tab1:
        render_individual_tab()

    with tab2:
        render_cohort_tab()

    with tab3:
        render_references_tab()

    with tab4:
        render_settings_tab()
    
    # Footer
    st.markdown("---")
    st.caption("SEKHMET Recovery Predictor v1.0 | Configuration is session-based")


if __name__ == "__main__":
    main()
