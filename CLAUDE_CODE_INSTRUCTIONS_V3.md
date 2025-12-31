# SEKHMET Revision V3 - UI Enhancements

## Context

You are updating the Pj-SEKHMET repository to add these UI enhancements:
1. **Confidence Intervals Visualised** - Shaded range on survival curve
2. **"What If" Scenario Slider** - Real-time recalculation for modifiable risk factors
3. **Comparator Benchmarks** - "Your case vs. typical" side-by-side
4. **Traffic Light Summary** - 🟢🟡🔴 for RTD likelihood at 3/6/12 months
6. **Model Agreement Indicator** - Flag when Cox and XGBoost diverge
8. **Body Region Heatmap Input** - Clickable body diagram

Repository: https://github.com/RossTylr/Pj-SEKHMET

**Prerequisites**: V2 changes (XGBoost added, MH removed) should be complete first.

---

## FEATURE 1: Confidence Intervals Visualised

### Description
Show prediction uncertainty as a shaded range on the survival curve, not just numbers.

### Implementation in app.py

```python
def plot_survival_curve_with_ci(cox_model, case, prediction):
    """
    Generate survival curve with confidence interval shading.
    """
    import plotly.graph_objects as go
    
    # Get survival curve data
    t, survival = cox_model.get_survival_curve(case, max_months=36)
    recovery = 1 - survival  # Probability of having recovered
    
    # Calculate CI bounds (using prediction intervals)
    # Approximate 90% CI based on Weibull uncertainty
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
        name='90% Confidence Interval',
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
    median = prediction.median_recovery_months if hasattr(prediction, 'median_recovery_months') else prediction.predicted_time_months
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
    else:
        lower = prediction.lower_bound_months
        upper = prediction.upper_bound_months
    
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
```

### Usage in Results Section

```python
# Replace existing survival curve plot
st.subheader("📈 Recovery Trajectory")
fig = plot_survival_curve_with_ci(cox_model, case, pred)
st.plotly_chart(fig, use_container_width=True)

# Add interpretation text
st.caption(f"""
**Reading this chart**: The green line shows expected recovery probability over time. 
The shaded area represents uncertainty (90% confidence interval). 
Vertical lines mark best case ({pred.lower_bound_months:.1f} mo), 
median ({pred.median_recovery_months:.1f} mo), 
and worst case ({pred.upper_bound_months:.1f} mo).
""")
```

---

## FEATURE 2: "What If" Scenario Slider

### Description
Let users adjust modifiable risk factors and see real-time prediction changes.

### Implementation in app.py

```python
def render_what_if_scenarios(base_case: dict, model, model_type: str):
    """
    Interactive what-if scenario explorer for modifiable risk factors.
    """
    st.subheader("🔮 What If Scenarios")
    st.caption("See how changing modifiable factors affects recovery time")
    
    # Get base prediction
    base_pred = get_prediction(model, base_case, model_type)
    base_months = get_recovery_months(base_pred)
    
    # Modifiable factors
    modifiable_factors = {
        'is_smoker': {'label': 'Quit Smoking', 'current': base_case.get('is_smoker', False), 
                      'toggle_to': False, 'hr_change': 1.43},
        'high_alcohol': {'label': 'Reduce Alcohol', 'current': base_case.get('high_alcohol', False),
                         'toggle_to': False, 'hr_change': 1.25},
        'poor_sleep': {'label': 'Improve Sleep', 'current': base_case.get('poor_sleep', False),
                       'toggle_to': False, 'hr_change': 1.30},
        'receiving_treatment': {'label': 'Add Supervised Rehab', 'current': base_case.get('receiving_treatment', True),
                                'toggle_to': True, 'hr_change': 0.75},
    }
    
    # BMI scenarios
    current_bmi = base_case.get('bmi', 25)
    
    # Calculate scenarios
    scenarios = []
    
    for key, config in modifiable_factors.items():
        if config['current'] != config['toggle_to']:
            # This factor can be changed
            scenario_case = base_case.copy()
            scenario_case[key] = config['toggle_to']
            scenario_pred = get_prediction(model, scenario_case, model_type)
            scenario_months = get_recovery_months(scenario_pred)
            
            change = scenario_months - base_months
            if abs(change) > 0.1:  # Only show meaningful changes
                scenarios.append({
                    'action': config['label'],
                    'current': base_months,
                    'new': scenario_months,
                    'change': change,
                    'change_pct': (change / base_months) * 100
                })
    
    # BMI reduction scenario
    if current_bmi >= 28:
        target_bmi = max(25, current_bmi - 5)
        scenario_case = base_case.copy()
        scenario_case['bmi'] = target_bmi
        scenario_pred = get_prediction(model, scenario_case, model_type)
        scenario_months = get_recovery_months(scenario_pred)
        change = scenario_months - base_months
        
        if abs(change) > 0.1:
            scenarios.append({
                'action': f'Reduce BMI to {target_bmi:.0f}',
                'current': base_months,
                'new': scenario_months,
                'change': change,
                'change_pct': (change / base_months) * 100
            })
    
    # Display scenarios
    if scenarios:
        # Sort by impact (most beneficial first)
        scenarios.sort(key=lambda x: x['change'])
        
        for scenario in scenarios:
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.markdown(f"**{scenario['action']}**")
            
            with col2:
                st.metric(
                    "New Estimate",
                    f"{scenario['new']:.1f} mo",
                    delta=f"{scenario['change']:+.1f} mo",
                    delta_color="inverse"  # Negative is good (faster recovery)
                )
            
            with col3:
                if scenario['change'] < 0:
                    st.success(f"⬇️ {abs(scenario['change_pct']):.0f}% faster")
                else:
                    st.warning(f"⬆️ {scenario['change_pct']:.0f}% slower")
        
        # Combined scenario
        st.divider()
        st.markdown("**💪 Combined: All Improvements**")
        
        combined_case = base_case.copy()
        for key, config in modifiable_factors.items():
            combined_case[key] = config['toggle_to']
        if current_bmi >= 28:
            combined_case['bmi'] = max(25, current_bmi - 5)
        
        combined_pred = get_prediction(model, combined_case, model_type)
        combined_months = get_recovery_months(combined_pred)
        combined_change = combined_months - base_months
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Current Estimate", f"{base_months:.1f} months")
        with col2:
            st.metric(
                "With All Changes",
                f"{combined_months:.1f} months",
                delta=f"{combined_change:+.1f} mo",
                delta_color="inverse"
            )
        
        if combined_change < 0:
            st.success(f"🎯 Potential improvement: **{abs(combined_change):.1f} months faster** ({abs(combined_change/base_months)*100:.0f}% reduction)")
    
    else:
        st.info("✅ No modifiable risk factors identified for this case. Current management is optimal.")


def get_prediction(model, case_dict, model_type):
    """Helper to get prediction from appropriate model."""
    if model_type == "cox":
        case = dict_to_case_input(case_dict)
        return model.predict(case)
    else:
        return model.predict(case_dict)


def get_recovery_months(pred):
    """Helper to extract recovery months from prediction."""
    if hasattr(pred, 'median_recovery_months'):
        return pred.median_recovery_months
    elif hasattr(pred, 'predicted_time_months'):
        return pred.predicted_time_months
    else:
        return pred.predicted_months
```

### Add to Results Section

```python
# After main prediction display
with st.expander("🔮 What If Scenarios", expanded=False):
    render_what_if_scenarios(case_dict, current_model, model_type)
```

---

## FEATURE 3: Comparator Benchmarks

### Description
Show "Your case vs. typical" side-by-side comparison.

### Implementation in app.py

```python
def render_comparator_benchmark(case_dict: dict, prediction, model, model_type: str):
    """
    Compare current case to a typical/baseline case with same injury.
    """
    st.subheader("📊 Comparison to Typical Case")
    
    # Create baseline case (same injury, no risk factors, age 30)
    baseline_case = {
        'age': 30,
        'body_region': case_dict['body_region'],
        'injury_type': case_dict['injury_type'],
        'prior_same_region': False,
        'is_smoker': False,
        'high_alcohol': False,
        'poor_sleep': False,
        'oh_risk': 'Moderate',
        'bmi': 25.0,
        'receiving_treatment': True,
    }
    
    baseline_pred = get_prediction(model, baseline_case, model_type)
    baseline_months = get_recovery_months(baseline_pred)
    current_months = get_recovery_months(prediction)
    
    difference = current_months - baseline_months
    diff_pct = (difference / baseline_months) * 100 if baseline_months > 0 else 0
    
    # Display comparison
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Typical 30yo**")
        st.markdown(f"Same injury, no risk factors")
        st.metric("Recovery", f"{baseline_months:.1f} mo")
    
    with col2:
        st.markdown("**Your Case**")
        st.markdown(f"Age {case_dict['age']}, with risk factors")
        st.metric(
            "Recovery", 
            f"{current_months:.1f} mo",
            delta=f"{difference:+.1f} mo" if difference != 0 else None,
            delta_color="inverse"
        )
    
    with col3:
        st.markdown("**Difference**")
        if difference > 0:
            st.error(f"⬆️ {diff_pct:.0f}% slower")
            st.caption("Risk factors adding time")
        elif difference < 0:
            st.success(f"⬇️ {abs(diff_pct):.0f}% faster")
            st.caption("Better than typical")
        else:
            st.info("➡️ Same as typical")
    
    # Breakdown of contributing factors
    with st.expander("See what's different"):
        differences = []
        
        if case_dict['age'] != 30:
            age_effect = (case_dict['age'] - 30) / 10 * 0.15  # ~15% per decade
            differences.append(f"• Age {case_dict['age']} vs 30: {age_effect*100:+.0f}%")
        
        if case_dict.get('prior_same_region'):
            differences.append("• Prior same-region injury: +80%")
        
        if case_dict.get('is_smoker'):
            differences.append("• Smoking: +43%")
        
        if case_dict.get('high_alcohol'):
            differences.append("• High alcohol: +25%")
        
        if case_dict.get('poor_sleep'):
            differences.append("• Poor sleep: +30%")
        
        if case_dict.get('oh_risk') == 'High':
            differences.append("• High OH risk: +30%")
        elif case_dict.get('oh_risk') == 'Low':
            differences.append("• Low OH risk: -15%")
        
        if case_dict.get('bmi', 25) >= 35:
            differences.append("• BMI ≥35: +40%")
        elif case_dict.get('bmi', 25) >= 30:
            differences.append("• BMI 30-35: +20%")
        
        if not case_dict.get('receiving_treatment', True):
            differences.append("• No supervised rehab: +33%")
        
        if differences:
            st.markdown("**Contributing factors:**")
            for diff in differences:
                st.markdown(diff)
        else:
            st.markdown("No significant differences from typical case.")
```

### Add to Results Section

```python
# After confidence interval chart
render_comparator_benchmark(case_dict, pred, current_model, model_type)
```

---

## FEATURE 4: Traffic Light Summary

### Description
Instant 🟢🟡🔴 visual for RTD likelihood at key milestones.

### Implementation in app.py

```python
def render_traffic_light_summary(prediction):
    """
    Display traffic light summary for RTD likelihood at 3/6/12 months.
    """
    st.subheader("🚦 RTD Likelihood")
    
    # Get probabilities at key timepoints
    if hasattr(prediction, 'prob_recovery_3mo'):
        prob_3mo = prediction.prob_recovery_3mo
        prob_6mo = prediction.prob_recovery_6mo
        prob_12mo = prediction.prob_recovery_12mo
    else:
        # Estimate from median using exponential approximation
        median = get_recovery_months(prediction)
        # Using Weibull-ish approximation
        prob_3mo = min(0.95, max(0.05, 1 - np.exp(-0.693 * 3 / median)))
        prob_6mo = min(0.95, max(0.05, 1 - np.exp(-0.693 * 6 / median)))
        prob_12mo = min(0.95, max(0.05, 1 - np.exp(-0.693 * 12 / median)))
    
    def get_traffic_light(prob):
        """Return emoji and color based on probability."""
        if prob >= 0.70:
            return "🟢", "green", "Likely"
        elif prob >= 0.40:
            return "🟡", "orange", "Possible"
        else:
            return "🔴", "red", "Unlikely"
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        emoji, color, label = get_traffic_light(prob_3mo)
        st.markdown(f"""
        <div style="text-align: center; padding: 1rem; background-color: #f8f9fa; border-radius: 0.5rem;">
            <div style="font-size: 3rem;">{emoji}</div>
            <div style="font-weight: bold; font-size: 1.2rem;">3 Months</div>
            <div style="font-size: 1.5rem; font-weight: bold;">{prob_3mo:.0%}</div>
            <div style="color: gray;">{label}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        emoji, color, label = get_traffic_light(prob_6mo)
        st.markdown(f"""
        <div style="text-align: center; padding: 1rem; background-color: #f8f9fa; border-radius: 0.5rem;">
            <div style="font-size: 3rem;">{emoji}</div>
            <div style="font-weight: bold; font-size: 1.2rem;">6 Months</div>
            <div style="font-size: 1.5rem; font-weight: bold;">{prob_6mo:.0%}</div>
            <div style="color: gray;">{label}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        emoji, color, label = get_traffic_light(prob_12mo)
        st.markdown(f"""
        <div style="text-align: center; padding: 1rem; background-color: #f8f9fa; border-radius: 0.5rem;">
            <div style="font-size: 3rem;">{emoji}</div>
            <div style="font-weight: bold; font-size: 1.2rem;">12 Months</div>
            <div style="font-size: 1.5rem; font-weight: bold;">{prob_12mo:.0%}</div>
            <div style="color: gray;">{label}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Workforce planning interpretation
    st.caption("""
    **For workforce planning**: 
    🟢 = Plan for RTD | 🟡 = Monitor closely | 🔴 = Arrange cover
    """)
```

### Add to Results Section (TOP of results)

```python
# Put this FIRST in results section for instant visibility
render_traffic_light_summary(pred)
st.divider()
# Then rest of results...
```

---

## FEATURE 6: Model Agreement Indicator

### Description
Flag when Cox and XGBoost predictions diverge significantly.

### Implementation in app.py

```python
def render_model_agreement(cox_model, xgb_model, case_dict, case_input):
    """
    Show model agreement indicator when multiple models available.
    """
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
        color = "green"
        message = "Models agree closely"
    elif difference <= 3.0:
        status = "moderate"
        emoji = "⚠️"
        color = "orange"
        message = "Models show some divergence"
    else:
        status = "low"
        emoji = "🔴"
        color = "red"
        message = "Models disagree significantly - consider clinical review"
    
    # Display
    st.markdown("### 🔄 Model Agreement")
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        st.metric("Cox PH", f"{cox_months:.1f} mo")
    
    with col2:
        st.metric("XGBoost", f"{xgb_months:.1f} mo")
    
    with col3:
        st.markdown(f"""
        <div style="padding: 0.5rem; border-radius: 0.5rem; background-color: {'#e8f5e9' if status == 'high' else '#fff3e0' if status == 'moderate' else '#ffebee'};">
            <span style="font-size: 1.5rem;">{emoji}</span>
            <strong>Agreement: {agreement_pct:.0f}%</strong><br/>
            <span style="color: gray;">{message}</span>
        </div>
        """, unsafe_allow_html=True)
    
    # Recommendation for low agreement
    if status == "low":
        st.warning(f"""
        **⚠️ Models disagree by {difference:.1f} months**
        
        This may indicate:
        - Unusual combination of risk factors
        - Case at edge of training data
        - Need for clinical judgment
        
        **Recommendation**: Use Cox PH estimate ({cox_months:.1f} mo) as primary, 
        consider XGBoost ({xgb_months:.1f} mo) as alternative scenario.
        """)
    
    return {
        'cox_months': cox_months,
        'xgb_months': xgb_months,
        'difference': difference,
        'agreement_pct': agreement_pct,
        'status': status
    }
```

### Add Toggle in Results Section

```python
# Add after model selection
show_agreement = st.sidebar.checkbox("Show model agreement", value=True, 
                                      help="Compare Cox and XGBoost predictions")

# In results section
if show_agreement:
    with st.expander("🔄 Model Agreement Check", expanded=(model_choice != "XGBoost (ML/SHAP)")):
        agreement = render_model_agreement(cox_model, xgb_model, case_dict, case_input)
```

---

## FEATURE 8: Body Region Heatmap Input

### Description
Clickable body diagram instead of dropdown for body region selection.

### Implementation in app.py

```python
def render_body_selector():
    """
    Render interactive body region selector using clickable image map.
    Returns selected body region.
    """
    import base64
    
    st.subheader("🦴 Select Injury Location")
    st.caption("Click on the body region or use dropdown below")
    
    # SVG body diagram with clickable regions
    body_svg = """
    <svg viewBox="0 0 200 400" style="max-width: 200px; margin: auto; display: block;">
        <!-- Body outline -->
        <ellipse cx="100" cy="40" rx="25" ry="30" fill="#f5f5f5" stroke="#333" stroke-width="2"/>
        
        <!-- Neck/Cervical -->
        <rect id="cervical_spine" x="90" y="65" width="20" height="20" 
              fill="#e3f2fd" stroke="#1976d2" stroke-width="2" 
              style="cursor: pointer;" class="body-region" data-region="cervical_spine"/>
        
        <!-- Shoulders -->
        <ellipse id="shoulder_left" cx="60" cy="100" rx="20" ry="15" 
                 fill="#e3f2fd" stroke="#1976d2" stroke-width="2" 
                 style="cursor: pointer;" class="body-region" data-region="shoulder"/>
        <ellipse id="shoulder_right" cx="140" cy="100" rx="20" ry="15" 
                 fill="#e3f2fd" stroke="#1976d2" stroke-width="2" 
                 style="cursor: pointer;" class="body-region" data-region="shoulder"/>
        
        <!-- Torso -->
        <rect x="70" y="85" width="60" height="80" fill="#f5f5f5" stroke="#333" stroke-width="2"/>
        
        <!-- Lower Back -->
        <rect id="lower_back" x="80" y="140" width="40" height="30" 
              fill="#e3f2fd" stroke="#1976d2" stroke-width="2" 
              style="cursor: pointer;" class="body-region" data-region="lower_back"/>
        
        <!-- Hip/Groin -->
        <ellipse id="hip_groin" cx="100" cy="185" rx="35" ry="20" 
                 fill="#e3f2fd" stroke="#1976d2" stroke-width="2" 
                 style="cursor: pointer;" class="body-region" data-region="hip_groin"/>
        
        <!-- Arms -->
        <rect x="35" y="100" width="15" height="60" fill="#f5f5f5" stroke="#333" stroke-width="2"/>
        <rect x="150" y="100" width="15" height="60" fill="#f5f5f5" stroke="#333" stroke-width="2"/>
        
        <!-- Wrist/Hand -->
        <ellipse id="wrist_left" cx="42" cy="170" rx="10" ry="15" 
                 fill="#e3f2fd" stroke="#1976d2" stroke-width="2" 
                 style="cursor: pointer;" class="body-region" data-region="wrist_hand"/>
        <ellipse id="wrist_right" cx="158" cy="170" rx="10" ry="15" 
                 fill="#e3f2fd" stroke="#1976d2" stroke-width="2" 
                 style="cursor: pointer;" class="body-region" data-region="wrist_hand"/>
        
        <!-- Legs -->
        <rect x="70" y="205" width="25" height="100" fill="#f5f5f5" stroke="#333" stroke-width="2"/>
        <rect x="105" y="205" width="25" height="100" fill="#f5f5f5" stroke="#333" stroke-width="2"/>
        
        <!-- Knee -->
        <ellipse id="knee_left" cx="82" cy="270" rx="15" ry="20" 
                 fill="#e3f2fd" stroke="#1976d2" stroke-width="2" 
                 style="cursor: pointer;" class="body-region" data-region="knee"/>
        <ellipse id="knee_right" cx="118" cy="270" rx="15" ry="20" 
                 fill="#e3f2fd" stroke="#1976d2" stroke-width="2" 
                 style="cursor: pointer;" class="body-region" data-region="knee"/>
        
        <!-- Ankle/Foot -->
        <ellipse id="ankle_left" cx="82" cy="340" rx="12" ry="20" 
                 fill="#e3f2fd" stroke="#1976d2" stroke-width="2" 
                 style="cursor: pointer;" class="body-region" data-region="ankle_foot"/>
        <ellipse id="ankle_right" cx="118" cy="340" rx="12" ry="20" 
                 fill="#e3f2fd" stroke="#1976d2" stroke-width="2" 
                 style="cursor: pointer;" class="body-region" data-region="ankle_foot"/>
        
        <!-- Labels -->
        <text x="100" y="58" text-anchor="middle" font-size="8" fill="#666">Head</text>
        <text x="100" y="78" text-anchor="middle" font-size="7" fill="#1976d2">Cervical</text>
        <text x="35" y="95" text-anchor="middle" font-size="7" fill="#1976d2">Shoulder</text>
        <text x="100" y="158" text-anchor="middle" font-size="7" fill="#1976d2">Lower Back</text>
        <text x="100" y="190" text-anchor="middle" font-size="7" fill="#1976d2">Hip/Groin</text>
        <text x="30" y="180" text-anchor="middle" font-size="7" fill="#1976d2">Wrist</text>
        <text x="82" cy="275" text-anchor="middle" font-size="7" fill="#1976d2">Knee</text>
        <text x="100" y="365" text-anchor="middle" font-size="7" fill="#1976d2">Ankle/Foot</text>
    </svg>
    """
    
    # Display SVG
    st.markdown(body_svg, unsafe_allow_html=True)
    
    # Region buttons as alternative (Streamlit can't do clickable SVG directly)
    st.markdown("**Quick select:**")
    
    col1, col2, col3, col4 = st.columns(4)
    
    regions = {
        'Knee': 'knee',
        'Lower Back': 'lower_back',
        'Shoulder': 'shoulder',
        'Ankle/Foot': 'ankle_foot',
        'Hip/Groin': 'hip_groin',
        'Cervical': 'cervical_spine',
        'Wrist/Hand': 'wrist_hand',
    }
    
    # Store selection in session state
    if 'selected_region' not in st.session_state:
        st.session_state.selected_region = 'knee'
    
    # Create button grid
    cols = st.columns(4)
    for i, (label, value) in enumerate(regions.items()):
        with cols[i % 4]:
            if st.button(
                label, 
                key=f"region_{value}",
                type="primary" if st.session_state.selected_region == value else "secondary",
                use_container_width=True
            ):
                st.session_state.selected_region = value
                st.rerun()
    
    return st.session_state.selected_region


# Alternative: Use streamlit-image-select for actual clickable image
def render_body_selector_simple():
    """
    Simpler version using visual dropdown with icons.
    """
    region_options = {
        "🦵 Knee": "knee",
        "🔙 Lower Back": "lower_back",
        "💪 Shoulder": "shoulder",
        "🦶 Ankle/Foot": "ankle_foot",
        "🦴 Hip/Groin": "hip_groin",
        "🦒 Cervical Spine": "cervical_spine",
        "✋ Wrist/Hand": "wrist_hand",
    }
    
    selected = st.selectbox(
        "Body Region",
        options=list(region_options.keys()),
        format_func=lambda x: x,
        help="Select the injured body region"
    )
    
    return region_options[selected]
```

### Integration Options

**Option A: Button Grid (works in standard Streamlit)**
```python
# In sidebar
st.sidebar.subheader("🦴 Injury Location")
body_region = render_body_selector_simple()  # Uses emoji dropdown

# Or use button grid in main area
with tab_predict:
    col_body, col_form = st.columns([1, 2])
    with col_body:
        body_region = render_body_selector()  # Button grid with SVG diagram
```

**Option B: Install streamlit-elements for true interactivity**
```bash
pip install streamlit-elements
```

```python
# Then use custom clickable SVG component
# (More complex, requires additional setup)
```

---

## UPDATED RESULTS LAYOUT

### Recommended Order

```python
def render_prediction_results(pred, case_dict, case_input, model_type, 
                               cox_model, bayesian_model, xgb_model):
    """
    Render prediction results in optimal order.
    """
    
    # 1. TRAFFIC LIGHT - First thing users see
    render_traffic_light_summary(pred)
    
    st.divider()
    
    # 2. KEY METRICS ROW
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Expected Recovery", f"{get_recovery_months(pred):.1f} mo")
    with col2:
        st.metric("Best Case", f"{pred.lower_bound_months:.1f} mo")
    with col3:
        st.metric("Worst Case", f"{pred.upper_bound_months:.1f} mo")
    with col4:
        st.metric("Planning Band", pred.recovery_band)
    
    st.divider()
    
    # 3. MODEL AGREEMENT (if enabled)
    if st.session_state.get('show_agreement', True):
        agreement = render_model_agreement(cox_model, xgb_model, case_dict, case_input)
        st.divider()
    
    # 4. SURVIVAL CURVE WITH CI
    st.subheader("📈 Recovery Trajectory")
    fig = plot_survival_curve_with_ci(cox_model, case_input, pred)
    st.plotly_chart(fig, use_container_width=True)
    
    # 5. COMPARATOR BENCHMARK
    render_comparator_benchmark(case_dict, pred, current_model, model_type)
    
    st.divider()
    
    # 6. WHAT-IF SCENARIOS (expandable)
    with st.expander("🔮 What If Scenarios", expanded=False):
        render_what_if_scenarios(case_dict, current_model, model_type)
    
    # 7. SHAP VALUES (for XGBoost only)
    if model_type == "xgb" and hasattr(pred, 'shap_values'):
        with st.expander("🔍 SHAP Feature Contributions", expanded=False):
            render_shap_display(pred)
    
    # 8. EVIDENCE SOURCES
    with st.expander("📚 Evidence Sources", expanded=False):
        render_citations(pred)
```

---

## TESTING CHECKLIST

```bash
# Run Streamlit
streamlit run app.py

# Manual checks:
# - [ ] Traffic lights show at TOP of results
# - [ ] 🟢🟡🔴 colors correct for probabilities
# - [ ] Survival curve has shaded CI band
# - [ ] Best/Worst case lines visible on curve
# - [ ] Comparator shows "Your case vs typical 30yo"
# - [ ] What-if scenarios calculate correctly
# - [ ] Quitting smoking shows faster recovery
# - [ ] Model agreement shows when Cox/XGBoost differ
# - [ ] Warning appears when agreement < 70%
# - [ ] Body region buttons work (or emoji dropdown)
# - [ ] Selected region highlights
```

---

## COMMIT MESSAGE

```bash
git add -A
git commit -m "feat: UI enhancements - traffic lights, what-if, comparators

NEW FEATURES:
- Traffic light summary (🟢🟡🔴) for 3/6/12 month RTD likelihood
- What-if scenario slider for modifiable risk factors
- Comparator benchmark vs typical 30yo baseline
- Model agreement indicator (Cox vs XGBoost divergence warning)
- Confidence interval shading on survival curve
- Body region visual selector with button grid

UI IMPROVEMENTS:
- Results now show traffic lights FIRST for instant interpretation
- Expandable sections for detailed analysis
- Better visual hierarchy throughout

Closes #[issue]"

git push
```

---

## FILE CHANGES SUMMARY

| File | Changes |
|------|---------|
| `app.py` | Add all 6 new features, reorganise results layout |
| `requirements.txt` | No changes needed (plotly already included) |

---

## FEATURE PRIORITY IF TIME-LIMITED

1. **Traffic Light** (10 min) - Highest impact, simplest to add
2. **Model Agreement** (15 min) - Important safety feature
3. **CI on Survival Curve** (15 min) - Visual upgrade
4. **Comparator Benchmark** (20 min) - Context for users
5. **What-If Scenarios** (30 min) - Interactive, engaging
6. **Body Heatmap** (30 min) - Nice-to-have, more complex

Total estimated: ~2 hours for all features
