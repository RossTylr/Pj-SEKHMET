"""
JMES Synthetic Workforce MVP - Streamlit Dashboard
===================================================
Spiral Development Phase 1: Core Visualization & Generation UI

Features:
1. Synthetic data generation controls
2. Population overview dashboard
3. JMES monitoring visualizations
4. Validation & verification panel
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yaml
from pathlib import Path
from datetime import datetime
import io

# Local imports
from generator import SyntheticDataGenerator
from models import (
    PERSONNEL_MASTER_SCHEMA, PERSON_MONTH_SCHEMA,
    JMESStatus, ServiceBranch
)


# ============================================================
# PAGE CONFIG & SESSION STATE
# ============================================================

st.set_page_config(
    page_title="JMES Workforce Simulation",
    page_icon="assets/images/logo.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'master_df' not in st.session_state:
    st.session_state.master_df = None
if 'month_df' not in st.session_state:
    st.session_state.month_df = None
if 'generator' not in st.session_state:
    st.session_state.generator = None
if 'config' not in st.session_state:
    st.session_state.config = None


# ============================================================
# SIDEBAR - CONTROLS
# ============================================================

def render_sidebar():
    """Render sidebar controls"""
    st.sidebar.image("assets/images/logo.png", width=80)
    st.sidebar.title("JMES Simulation")
    st.sidebar.markdown("---")
    
    # Simulation Parameters
    st.sidebar.subheader("⚙️ Simulation Parameters")
    
    population_size = st.sidebar.number_input(
        "Initial Population",
        min_value=1000,
        max_value=100000,
        value=10000,
        step=1000,
        help="Number of synthetic personnel to generate (use 10K for testing)"
    )
    
    simulation_months = st.sidebar.slider(
        "Simulation Duration (months)",
        min_value=12,
        max_value=120,
        value=60,
        step=12,
        help="Duration of longitudinal simulation"
    )
    
    seed = st.sidebar.number_input(
        "Random Seed",
        min_value=1,
        max_value=9999,
        value=42,
        help="For reproducibility"
    )
    
    st.sidebar.markdown("---")
    
    # Turnover adjustment
    st.sidebar.subheader("📊 Turnover Model")
    annual_turnover = st.sidebar.slider(
        "Annual Turnover Rate",
        min_value=0.05,
        max_value=0.20,
        value=0.10,
        step=0.01,
        format="%.0f%%",
        help="Target annual workforce turnover"
    )
    
    st.sidebar.markdown("---")
    
    # Generate button
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        generate_clicked = st.button(
            "🚀 Generate",
            type="primary",
            use_container_width=True
        )
    
    with col2:
        clear_clicked = st.button(
            "🗑️ Clear",
            use_container_width=True
        )
    
    if clear_clicked:
        st.session_state.master_df = None
        st.session_state.month_df = None
        st.session_state.generator = None
        st.rerun()
    
    return {
        'generate': generate_clicked,
        'population_size': population_size,
        'simulation_months': simulation_months,
        'seed': seed,
        'annual_turnover': annual_turnover
    }


# ============================================================
# DATA GENERATION
# ============================================================

def generate_data(params: dict):
    """Generate synthetic data with progress tracking"""
    
    # Create temporary config
    config = create_config(params)
    config_path = Path("/tmp/simulation_config.yaml")
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("Initializing generator...")
        generator = SyntheticDataGenerator(
            config_path=str(config_path),
            seed=params['seed']
        )
        progress_bar.progress(10)
        
        status_text.text("Generating initial cohort...")
        master_df = generator.generate_initial_cohort(params['population_size'])
        progress_bar.progress(30)
        
        status_text.text("Running monthly simulation...")
        all_month_records = []
        n_months = params['simulation_months']
        
        for month in range(1, n_months + 1):
            month_records = generator.simulate_month(month)
            all_month_records.extend(month_records)
            
            if month % 6 == 0:
                progress = 30 + int(60 * month / n_months)
                progress_bar.progress(progress)
                status_text.text(f"Simulating month {month}/{n_months}...")
        
        status_text.text("Finalizing data...")
        master_df = pd.DataFrame(generator.master_records)
        month_df = pd.DataFrame(all_month_records)
        progress_bar.progress(95)
        
        # Store in session state
        st.session_state.master_df = master_df
        st.session_state.month_df = month_df
        st.session_state.generator = generator
        st.session_state.config = config
        
        progress_bar.progress(100)
        status_text.text("✅ Generation complete!")
        
        return True
        
    except Exception as e:
        st.error(f"Generation failed: {str(e)}")
        return False


def create_config(params: dict) -> dict:
    """Create configuration dictionary from UI parameters"""
    monthly_rate = 1 - (1 - params['annual_turnover']) ** (1/12)
    
    return {
        'population': {
            'initial_size': params['population_size'],
            'simulation_months': params['simulation_months']
        },
        'turnover': {
            'annual_rate': params['annual_turnover'],
            'monthly_rate': monthly_rate,
            'los_weights': {'0-4': 1.3, '5-12': 0.9, '13+': 0.8}
        },
        'service_mix': {'Army': 0.60, 'RN': 0.20, 'RAF': 0.20},
        'gender': {
            'overall_female_rate': 0.175,
            'trade_adjustments': {
                'CMT': 0.25, 'Paramedic': 0.22, 'AHP': 0.35, 'ODP': 0.30, 'Other': 0.12
            }
        },
        'age': {
            'min': 18, 'max': 55, 'mean': 32.5, 'std': 8.0,
            'inflow': {'min': 18, 'max': 35, 'mean': 22, 'std': 4.0}
        },
        'length_of_service': {
            'bands': {'0-4': 0.40, '5-12': 0.35, '13-25': 0.25},
            'max_years': 30
        },
        'rank_bands': {
            'OR2-OR4': 0.45, 'OR5-OR7': 0.30, 'OR8-OR9': 0.10,
            'OF1-OF3': 0.10, 'OF4-OF5': 0.05
        },
        'trades': {
            'CMT': 0.15, 'Paramedic': 0.10, 'AHP': 0.12, 'ODP': 0.08, 'Other': 0.55
        },
        'jmes': {
            'baseline_distribution': {'MFD': 0.85, 'MLD': 0.12, 'MND': 0.03},
            'transitions': {
                'MFD_to_MLD': 0.008, 'MFD_to_MND': 0.001,
                'MLD_to_MND': 0.015, 'MLD_to_MFD': 0.025,
                'MND_to_MLD': 0.010, 'MND_to_MFD': 0.002
            }
        },
        'injuries': {
            'baseline_monthly_mski': 0.02,
            'trade_multipliers': {
                'CMT': 1.8, 'Paramedic': 1.6, 'AHP': 1.0, 'ODP': 1.2, 'Other': 1.3
            },
            'deployment_multiplier': 1.5,
            'high_risk_training_multiplier': 3.0,
            'types': {
                'MSKI-minor': 0.50, 'MSKI-major': 0.25, 'MH-episode': 0.15, 'Other': 0.10
            }
        },
        'deployment': {
            'baseline_monthly_rate': 0.08,
            'service_rates': {'Army': 1.2, 'RN': 1.1, 'RAF': 0.8},
            'intensity': {'Low_tempo': 0.60, 'High_tempo': 0.40}
        },
        'training': {
            'monthly_rate': 0.15,
            'intensity': {'Low_risk': 0.70, 'High_risk': 0.30}
        },
        'pregnancy': {
            'annual_conception_rate': 0.065,
            'monthly_conception_rate': 0.0055,
            'duration': {'pregnancy_months': 9, 'postpartum_months': 4}
        },
        'engagement_types': {'PC': 0.35, 'IC': 0.25, 'FE': 0.30, 'UCM-H': 0.10},
        'unit_environment': {'Standard': 0.70, 'High-Risk': 0.20, 'Hot/Cold': 0.10},
        'validation': {
            'turnover_tolerance': 0.02,
            'population_stability': 0.05,
            'jmes_mfd_floor': 0.80,
            'injury_rate_ceiling': 0.10
        }
    }


# ============================================================
# DASHBOARD VIEWS
# ============================================================

def render_overview_tab():
    """Render population overview"""
    master_df = st.session_state.master_df
    month_df = st.session_state.month_df
    
    if master_df is None:
        st.info("Generate data to view overview")
        return
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Personnel",
            f"{len(master_df):,}",
            help="Unique individuals in simulation"
        )
    
    with col2:
        mfd_rate = (master_df['baseline_jmes'] == 'MFD').mean()
        st.metric(
            "Baseline MFD Rate",
            f"{mfd_rate:.1%}",
            help="Medically Fully Deployable at t=0"
        )
    
    with col3:
        female_rate = (master_df['gender'] == 'Female').mean()
        st.metric(
            "Female %",
            f"{female_rate:.1%}"
        )
    
    with col4:
        mean_age = master_df['age_start'].mean()
        st.metric(
            "Mean Age",
            f"{mean_age:.1f}"
        )
    
    st.markdown("---")
    
    # Distribution charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Service branch distribution
        fig = px.pie(
            master_df,
            names='service_branch',
            title='Service Branch Distribution',
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Age distribution
        fig = px.histogram(
            master_df,
            x='age_start',
            nbins=30,
            title='Age Distribution at Baseline',
            color_discrete_sequence=['#2E86AB']
        )
        fig.update_layout(height=350, xaxis_title='Age', yaxis_title='Count')
        st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Trade distribution
        trade_counts = master_df['trade'].value_counts()
        fig = px.bar(
            x=trade_counts.index,
            y=trade_counts.values,
            title='Trade Distribution',
            color=trade_counts.index,
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig.update_layout(height=350, xaxis_title='Trade', yaxis_title='Count', showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # JMES baseline
        jmes_counts = master_df['baseline_jmes'].value_counts()
        colors = {'MFD': '#28a745', 'MLD': '#ffc107', 'MND': '#dc3545'}
        fig = px.bar(
            x=jmes_counts.index,
            y=jmes_counts.values,
            title='Baseline JMES Distribution',
            color=jmes_counts.index,
            color_discrete_map=colors
        )
        fig.update_layout(height=350, xaxis_title='JMES Status', yaxis_title='Count', showlegend=False)
        st.plotly_chart(fig, use_container_width=True)


def render_jmes_tab():
    """Render JMES monitoring dashboard"""
    month_df = st.session_state.month_df
    
    if month_df is None:
        st.info("Generate data to view JMES analysis")
        return
    
    # Time series of JMES distribution
    st.subheader("📈 JMES Status Over Time")
    
    jmes_time = month_df.groupby(['month', 'jmes_current']).size().unstack(fill_value=0)
    jmes_pct = jmes_time.div(jmes_time.sum(axis=1), axis=0) * 100
    
    fig = go.Figure()
    colors = {'MFD': '#28a745', 'MLD': '#ffc107', 'MND': '#dc3545'}
    
    for status in ['MFD', 'MLD', 'MND']:
        if status in jmes_pct.columns:
            fig.add_trace(go.Scatter(
                x=jmes_pct.index,
                y=jmes_pct[status],
                name=status,
                mode='lines',
                line=dict(color=colors[status], width=2),
                stackgroup='one'
            ))
    
    fig.update_layout(
        title='JMES Status Distribution Over Time',
        xaxis_title='Month',
        yaxis_title='Percentage',
        height=400,
        legend=dict(orientation='h', yanchor='bottom', y=1.02)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # JMES deterioration events
    st.subheader("⚠️ Deterioration Events")
    
    col1, col2 = st.columns(2)
    
    with col1:
        events_by_month = month_df.groupby('month')['jmes_event_this_month'].sum()
        fig = px.line(
            x=events_by_month.index,
            y=events_by_month.values,
            title='Monthly JMES Deterioration Events',
            labels={'x': 'Month', 'y': 'Events'}
        )
        fig.update_traces(line_color='#dc3545')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Cumulative deteriorations
        cumulative = events_by_month.cumsum()
        fig = px.area(
            x=cumulative.index,
            y=cumulative.values,
            title='Cumulative Deterioration Events',
            labels={'x': 'Month', 'y': 'Cumulative Events'}
        )
        fig.update_traces(line_color='#dc3545', fillcolor='rgba(220, 53, 69, 0.3)')
        st.plotly_chart(fig, use_container_width=True)
    
    # Service-specific JMES
    st.subheader("🎖️ JMES by Service Branch")
    
    # Get latest month data
    latest_month = month_df['month'].max()
    latest_data = month_df[month_df['month'] == latest_month]
    
    # Merge with master to get service branch
    master_df = st.session_state.master_df
    merged = latest_data.merge(
        master_df[['person_id', 'service_branch']],
        on='person_id'
    )
    
    jmes_by_service = merged.groupby(['service_branch', 'jmes_current']).size().unstack(fill_value=0)
    jmes_by_service_pct = jmes_by_service.div(jmes_by_service.sum(axis=1), axis=0) * 100
    
    fig = px.bar(
        jmes_by_service_pct,
        barmode='group',
        title=f'JMES Distribution by Service (Month {latest_month})',
        color_discrete_map=colors
    )
    fig.update_layout(xaxis_title='Service Branch', yaxis_title='Percentage')
    st.plotly_chart(fig, use_container_width=True)


def render_turnover_tab():
    """Render turnover analysis"""
    month_df = st.session_state.month_df
    generator = st.session_state.generator
    
    if month_df is None:
        st.info("Generate data to view turnover analysis")
        return
    
    st.subheader("📊 Workforce Turnover Analysis")
    
    # Inflow/Outflow over time
    inflow_by_month = month_df.groupby('month')['inflow_flag'].sum()
    outflow_by_month = month_df.groupby('month')['outflow_flag'].sum()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=inflow_by_month.index,
        y=inflow_by_month.values,
        name='Inflow',
        mode='lines',
        line=dict(color='#28a745', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=outflow_by_month.index,
        y=outflow_by_month.values,
        name='Outflow',
        mode='lines',
        line=dict(color='#dc3545', width=2)
    ))
    fig.update_layout(
        title='Monthly Inflow vs Outflow',
        xaxis_title='Month',
        yaxis_title='Count',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Population trend
    col1, col2 = st.columns(2)
    
    with col1:
        pop_by_month = month_df.groupby('month')['person_id'].nunique()
        fig = px.line(
            x=pop_by_month.index,
            y=pop_by_month.values,
            title='Active Population Over Time',
            labels={'x': 'Month', 'y': 'Active Personnel'}
        )
        fig.update_traces(line_color='#2E86AB')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Outflow reasons (if captured)
        if 'outflow_reason' in month_df.columns:
            outflow_data = month_df[month_df['outflow_flag'] == 1]
            if len(outflow_data) > 0:
                reason_counts = outflow_data['outflow_reason'].value_counts()
                fig = px.pie(
                    values=reason_counts.values,
                    names=reason_counts.index,
                    title='Outflow Reasons'
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Verification metrics
    st.subheader("✅ Turnover Verification")
    
    if generator:
        log = generator.verification_log
        config = st.session_state.config
        
        total_months = config['population']['simulation_months']
        avg_monthly_outflow = log['total_outflows'] / total_months
        implied_annual = 1 - (1 - avg_monthly_outflow / config['population']['initial_size']) ** 12
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            target_turnover = config['turnover']['annual_rate']
            delta = implied_annual - target_turnover
            st.metric(
                "Implied Annual Turnover",
                f"{implied_annual:.1%}",
                delta=f"{delta:+.1%}",
                delta_color="inverse"
            )
        
        with col2:
            st.metric(
                "Total Inflows",
                f"{log['total_inflows']:,}"
            )
        
        with col3:
            st.metric(
                "Total Outflows",
                f"{log['total_outflows']:,}"
            )


def render_injuries_tab():
    """Render injury analysis"""
    month_df = st.session_state.month_df
    master_df = st.session_state.master_df
    
    if month_df is None:
        st.info("Generate data to view injury analysis")
        return
    
    st.subheader("🩹 Injury Analysis")
    
    # Injury events over time
    injury_data = month_df[month_df['injury_type'] != 'None']
    injuries_by_month = injury_data.groupby('month').size()
    
    fig = px.line(
        x=injuries_by_month.index,
        y=injuries_by_month.values,
        title='Monthly Injury Events',
        labels={'x': 'Month', 'y': 'Injuries'}
    )
    fig.update_traces(line_color='#e74c3c')
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Injury type distribution
        injury_types = injury_data['injury_type'].value_counts()
        fig = px.pie(
            values=injury_types.values,
            names=injury_types.index,
            title='Injury Type Distribution',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Severity distribution
        fig = px.histogram(
            injury_data,
            x='injury_severity_score',
            nbins=10,
            title='Injury Severity Distribution',
            color_discrete_sequence=['#e74c3c']
        )
        fig.update_layout(xaxis_title='Severity Score', yaxis_title='Count')
        st.plotly_chart(fig, use_container_width=True)
    
    # Injuries by trade
    st.subheader("📊 Injuries by Trade")
    
    # Merge with master for trade info
    injury_merged = injury_data.merge(
        master_df[['person_id', 'trade']],
        on='person_id'
    )
    
    injuries_by_trade = injury_merged.groupby('trade').size()
    pop_by_trade = master_df['trade'].value_counts()
    
    # Calculate rate per 1000 person-months
    total_months = month_df['month'].max()
    injury_rate = (injuries_by_trade / (pop_by_trade * total_months / 12)) * 100
    
    fig = px.bar(
        x=injury_rate.index,
        y=injury_rate.values,
        title='Annual Injury Rate by Trade (%)',
        color=injury_rate.values,
        color_continuous_scale='Reds'
    )
    fig.update_layout(xaxis_title='Trade', yaxis_title='Annual Rate (%)', showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


def render_export_tab():
    """Render data export options"""
    master_df = st.session_state.master_df
    month_df = st.session_state.month_df
    
    if master_df is None:
        st.info("Generate data to enable export")
        return
    
    st.subheader("📥 Export Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### Master Table")
        st.write(f"Rows: {len(master_df):,}")
        st.write(f"Columns: {len(master_df.columns)}")
        
        # CSV download
        csv_master = master_df.to_csv(index=False)
        st.download_button(
            "📄 Download Master CSV",
            csv_master,
            "personnel_master.csv",
            "text/csv",
            use_container_width=True
        )
    
    with col2:
        st.markdown("### Longitudinal Table")
        st.write(f"Rows: {len(month_df):,}")
        st.write(f"Columns: {len(month_df.columns)}")
        
        # CSV download (warning for large files)
        if len(month_df) > 1_000_000:
            st.warning("Large file - download may take time")
        
        csv_month = month_df.to_csv(index=False)
        st.download_button(
            "📄 Download Monthly CSV",
            csv_month,
            "person_month.csv",
            "text/csv",
            use_container_width=True
        )
    
    with col3:
        st.markdown("### Configuration")
        config = st.session_state.config
        if config:
            config_yaml = yaml.dump(config, default_flow_style=False)
            st.download_button(
                "⚙️ Download Config YAML",
                config_yaml,
                "simulation_config.yaml",
                "text/yaml",
                use_container_width=True
            )
    
    # Data preview
    st.markdown("---")
    st.subheader("👀 Data Preview")
    
    preview_tab1, preview_tab2 = st.tabs(["Master Table", "Monthly Table"])
    
    with preview_tab1:
        st.dataframe(master_df.head(100), use_container_width=True)
    
    with preview_tab2:
        st.dataframe(month_df.head(100), use_container_width=True)


# ============================================================
# MAIN APP
# ============================================================

def main():
    """Main application entry point"""
    
    # Sidebar
    params = render_sidebar()
    
    # Header
    st.image("assets/images/logo.png", width=100)
    st.title("JMES Defence Workforce Simulation")
    st.markdown("""
    **Synthetic Tri-Service Dataset Generator** for JMES prediction modelling.
    
    Generate realistic workforce data with configurable:
    - Population size and simulation duration
    - Turnover dynamics (inflow/outflow)
    - JMES transitions and deterioration events
    - Injury, deployment, and pregnancy episodes
    """)
    
    # Generate data if requested
    if params['generate']:
        with st.spinner("Generating synthetic data..."):
            success = generate_data(params)
            if success:
                st.success("✅ Data generated successfully!")
                st.rerun()
    
    st.markdown("---")
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview",
        "🏥 JMES Monitoring",
        "🔄 Turnover",
        "🩹 Injuries",
        "📥 Export"
    ])
    
    with tab1:
        render_overview_tab()
    
    with tab2:
        render_jmes_tab()
    
    with tab3:
        render_turnover_tab()
    
    with tab4:
        render_injuries_tab()
    
    with tab5:
        render_export_tab()
    
    # Footer
    st.markdown("---")
    st.caption(
        "JMES Synthetic Workforce MVP v0.1.0 | "
        "Chain-of-Verification Enabled | "
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    )


if __name__ == "__main__":
    main()
