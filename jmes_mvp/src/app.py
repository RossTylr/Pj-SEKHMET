"""
JMES Streamlit Dashboard - Interactive data exploration
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from generator import SyntheticDataGenerator


def main():
    """Main application entry point"""
    st.set_page_config(page_title="JMES Synthetic Workforce", layout="wide")

    st.title("JMES Synthetic Workforce Simulation")
    st.markdown("Generates synthetic Defence personnel data for medical employment modeling")

    # Initialize session state
    if 'master_df' not in st.session_state:
        st.session_state.master_df = None
    if 'monthly_df' not in st.session_state:
        st.session_state.monthly_df = None

    # Sidebar configuration
    render_sidebar()

    # Main content
    if st.session_state.master_df is not None:
        # Create tabs
        tabs = st.tabs(["Overview", "JMES Monitoring", "Turnover", "Injuries", "Export"])

        with tabs[0]:
            render_overview_tab()

        with tabs[1]:
            render_jmes_tab()

        with tabs[2]:
            render_turnover_tab()

        with tabs[3]:
            render_injuries_tab()

        with tabs[4]:
            render_export_tab()
    else:
        st.info("Configure simulation parameters in the sidebar and click 'Run Simulation'")


def render_sidebar():
    """Render sidebar controls"""
    st.sidebar.header("Simulation Configuration")

    # Population parameters
    st.sidebar.subheader("Population")
    population_size = st.sidebar.number_input(
        "Initial Size",
        min_value=100,
        max_value=100000,
        value=1000,
        step=100
    )

    simulation_months = st.sidebar.slider(
        "Simulation Duration (months)",
        min_value=6,
        max_value=120,
        value=12,
        step=6
    )

    # Random seed
    seed = st.sidebar.number_input(
        "Random Seed",
        min_value=0,
        max_value=9999,
        value=42
    )

    # Run button
    if st.sidebar.button("Run Simulation", type="primary"):
        with st.spinner("Generating synthetic data..."):
            run_simulation(population_size, simulation_months, seed)


def run_simulation(population_size: int, simulation_months: int, seed: int):
    """Execute simulation and store results"""
    try:
        # Create generator
        gen = SyntheticDataGenerator('configs/simulation_config.yaml', seed=seed)
        gen.config['population']['initial_size'] = population_size
        gen.config['population']['simulation_months'] = simulation_months

        # Run simulation
        master_df, monthly_df = gen.run_simulation()

        # Store in session state
        st.session_state.master_df = master_df
        st.session_state.monthly_df = monthly_df

        st.sidebar.success(f"Simulation complete: {len(master_df):,} personnel, {len(monthly_df):,} records")

    except Exception as e:
        st.sidebar.error(f"Simulation failed: {str(e)}")


def render_overview_tab():
    """Render overview statistics and visualizations"""
    st.header("Population Overview")

    master_df = st.session_state.master_df
    monthly_df = st.session_state.monthly_df

    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Personnel", f"{len(master_df):,}")

    with col2:
        final_pop = monthly_df[monthly_df['month'] == monthly_df['month'].max()]
        mfd_rate = (final_pop['jmes_current'] == 'MFD').mean()
        st.metric("Final MFD Rate", f"{mfd_rate:.1%}")

    with col3:
        avg_age = master_df['age_start'].mean()
        st.metric("Average Age", f"{avg_age:.1f}")

    with col4:
        female_rate = (master_df['gender'] == 'Female').mean()
        st.metric("Female Rate", f"{female_rate:.1%}")

    st.markdown("---")

    # Charts
    col1, col2 = st.columns(2)

    with col1:
        # Service branch distribution
        service_counts = master_df['service_branch'].value_counts()
        fig = px.pie(
            values=service_counts.values,
            names=service_counts.index,
            title="Service Branch Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Trade distribution
        trade_counts = master_df['trade'].value_counts()
        fig = px.bar(
            x=trade_counts.index,
            y=trade_counts.values,
            title="Trade Distribution",
            labels={'x': 'Trade', 'y': 'Count'}
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Age distribution
        fig = px.histogram(
            master_df,
            x='age_start',
            nbins=20,
            title="Age Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

        # JMES baseline
        jmes_counts = master_df['baseline_jmes'].value_counts()
        fig = px.bar(
            x=jmes_counts.index,
            y=jmes_counts.values,
            title="Baseline JMES Distribution",
            labels={'x': 'JMES Status', 'y': 'Count'}
        )
        st.plotly_chart(fig, use_container_width=True)


def render_jmes_tab():
    """Render JMES monitoring visualizations"""
    st.header("JMES Monitoring")

    monthly_df = st.session_state.monthly_df

    # JMES over time
    jmes_by_month = monthly_df.groupby(['month', 'jmes_current']).size().reset_index(name='count')
    jmes_pivot = jmes_by_month.pivot(index='month', columns='jmes_current', values='count').fillna(0)

    fig = go.Figure()
    for col in jmes_pivot.columns:
        fig.add_trace(go.Scatter(
            x=jmes_pivot.index,
            y=jmes_pivot[col],
            mode='lines',
            stackgroup='one',
            name=col
        ))

    fig.update_layout(
        title="JMES Status Over Time (Stacked)",
        xaxis_title="Month",
        yaxis_title="Personnel Count",
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)

    # JMES events over time
    jmes_events = monthly_df.groupby('month')['jmes_event_this_month'].sum().reset_index()

    fig = px.line(
        jmes_events,
        x='month',
        y='jmes_event_this_month',
        title="JMES Deterioration Events by Month",
        labels={'jmes_event_this_month': 'Events', 'month': 'Month'}
    )
    st.plotly_chart(fig, use_container_width=True)

    # JMES by service
    master_df = st.session_state.master_df
    service_jmes = master_df.groupby(['service_branch', 'baseline_jmes']).size().reset_index(name='count')

    fig = px.bar(
        service_jmes,
        x='service_branch',
        y='count',
        color='baseline_jmes',
        title="Baseline JMES by Service Branch",
        barmode='group'
    )
    st.plotly_chart(fig, use_container_width=True)


def render_turnover_tab():
    """Render turnover analysis"""
    st.header("Turnover Analysis")

    monthly_df = st.session_state.monthly_df

    # Inflows and outflows over time
    turnover_data = monthly_df.groupby('month').agg({
        'inflow_flag': 'sum',
        'outflow_flag': 'sum'
    }).reset_index()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=turnover_data['month'],
        y=turnover_data['inflow_flag'],
        mode='lines+markers',
        name='Inflows'
    ))
    fig.add_trace(go.Scatter(
        x=turnover_data['month'],
        y=turnover_data['outflow_flag'],
        mode='lines+markers',
        name='Outflows'
    ))

    fig.update_layout(
        title="Inflows and Outflows Over Time",
        xaxis_title="Month",
        yaxis_title="Count",
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)

    # Population trend
    pop_by_month = monthly_df.groupby('month')['person_id'].nunique().reset_index()

    fig = px.line(
        pop_by_month,
        x='month',
        y='person_id',
        title="Active Population Over Time",
        labels={'person_id': 'Active Personnel', 'month': 'Month'}
    )
    st.plotly_chart(fig, use_container_width=True)

    # Outflow reasons
    outflow_reasons = monthly_df[monthly_df['outflow_flag'] == 1]['outflow_reason'].value_counts()

    if len(outflow_reasons) > 0:
        fig = px.pie(
            values=outflow_reasons.values,
            names=outflow_reasons.index,
            title="Outflow Reasons Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)


def render_injuries_tab():
    """Render injury analysis"""
    st.header("Injury Analysis")

    monthly_df = st.session_state.monthly_df

    # Injuries over time
    injuries_by_month = monthly_df[monthly_df['injury_type'] != 'None'].groupby('month').size().reset_index(name='count')

    fig = px.line(
        injuries_by_month,
        x='month',
        y='count',
        title="Monthly Injury Count",
        labels={'count': 'Injuries', 'month': 'Month'}
    )
    st.plotly_chart(fig, use_container_width=True)

    # Injury types
    injury_types = monthly_df[monthly_df['injury_type'] != 'None']['injury_type'].value_counts()

    fig = px.pie(
        values=injury_types.values,
        names=injury_types.index,
        title="Injury Types Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Get trade from master
    master_df = st.session_state.master_df
    monthly_with_trade = monthly_df.merge(
        master_df[['person_id', 'trade']],
        on='person_id',
        how='left'
    )

    # Injury rate by trade
    injury_by_trade = monthly_with_trade[monthly_with_trade['injury_type'] != 'None'].groupby('trade').size()
    total_by_trade = monthly_with_trade.groupby('trade').size()
    injury_rate = (injury_by_trade / total_by_trade).sort_values(ascending=False).reset_index(name='rate')

    fig = px.bar(
        injury_rate,
        x='trade',
        y='rate',
        title="Injury Rate by Trade",
        labels={'rate': 'Injury Rate', 'trade': 'Trade'}
    )
    fig.update_yaxes(tickformat='.1%')
    st.plotly_chart(fig, use_container_width=True)


def render_export_tab():
    """Render data export options"""
    st.header("Data Export")

    master_df = st.session_state.master_df
    monthly_df = st.session_state.monthly_df

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Master Table")
        st.write(f"Rows: {len(master_df):,}")
        st.write(f"Columns: {len(master_df.columns)}")

        csv = master_df.to_csv(index=False)
        st.download_button(
            label="Download Master CSV",
            data=csv,
            file_name="jmes_master.csv",
            mime="text/csv"
        )

    with col2:
        st.subheader("Monthly Table")
        st.write(f"Rows: {len(monthly_df):,}")
        st.write(f"Columns: {len(monthly_df.columns)}")

        csv = monthly_df.to_csv(index=False)
        st.download_button(
            label="Download Monthly CSV",
            data=csv,
            file_name="jmes_monthly.csv",
            mime="text/csv"
        )

    # Data preview
    st.markdown("---")
    st.subheader("Data Preview")

    tab1, tab2 = st.tabs(["Master Table", "Monthly Table"])

    with tab1:
        st.dataframe(master_df.head(100), use_container_width=True)

    with tab2:
        st.dataframe(monthly_df.head(100), use_container_width=True)


if __name__ == "__main__":
    main()
