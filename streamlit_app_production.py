"""
Energy Price Forecasting Dashboard
===================================

Production-grade electricity price prediction using physics-informed ML.
Real-time data from ENTSO-E Transparency Platform.

Author: Suchita Kulkarni
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Energy Price Forecasting",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional slate theme CSS
st.markdown("""
    <style>
    /* Main background */
    .main {
        background-color: #1e1e1e;
        color: #ffffff;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #2d2d2d;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        font-family: 'Inter', sans-serif;
    }
    
    /* Main title */
    .main-title {
        font-size: 2.8rem;
        font-weight: 700;
        color: #4a9eff;
        margin-bottom: 0.3rem;
        letter-spacing: -0.5px;
    }
    
    .sub-title {
        font-size: 1.1rem;
        color: #b0b0b0;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* Metric cards */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        color: #4a9eff;
    }
    
    [data-testid="stMetricLabel"] {
        color: #b0b0b0;
        font-size: 0.9rem;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: #2d2d2d;
        padding: 8px;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #3d3d3d;
        border-radius: 8px;
        color: #b0b0b0;
        font-weight: 500;
        padding: 0px 24px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #4a9eff;
        color: #ffffff;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #4a9eff;
        color: #ffffff;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 2rem;
        font-weight: 500;
    }
    
    .stButton>button:hover {
        background-color: #3a8eef;
    }
    
    /* Selectbox */
    .stSelectbox {
        color: #ffffff;
    }
    
    /* Text */
    p, span, div {
        color: #d0d0d0;
    }
    
    /* Info boxes */
    .stAlert {
        background-color: #2d2d2d;
        color: #ffffff;
        border-radius: 8px;
    }
    
    /* Plotly charts - dark theme */
    .js-plotly-plot {
        background-color: #1e1e1e !important;
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def load_entso_fetcher():
    """Load ENTSO-E data fetcher with API key from secrets."""
    from entsoe_data_fetcher import EntsoeDataFetcher
    
    api_key = st.secrets["ENTSOE_API_KEY"]
    return EntsoeDataFetcher(api_key=api_key)


@st.cache_resource
def load_forecaster(country):
    """Load pretrained model if exists, otherwise return None."""
    from model_training_pipeline import EnergyPriceForecaster
    
    model_path = Path(f"models/{country}_forecaster")
    
    if model_path.with_name(f"{country}_forecaster_stat.pkl").exists():
        try:
            forecaster = EnergyPriceForecaster.load(model_path)
            return forecaster
        except Exception as e:
            st.sidebar.error(f"Error loading model: {e}")
            return None
    else:
        return None


@st.cache_data(ttl=900)  # 15 min cache - ENTSO-E actual load publishes with ~1hr lag
def fetch_latest_data(country, days=30):
    """Fetch latest data from ENTSO-E."""
    fetcher = load_entso_fetcher()

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    df = fetcher.fetch_complete_dataset(
        country=country,
        start_date=start_date.strftime('%Y-%m-%d %H:%M'),
        end_date=end_date.strftime('%Y-%m-%d %H:%M'),
        include_forecasts=False
    )

    df = fetcher.compute_physics_features(df)

    return df


def make_predictions(forecaster, df, hours_ahead=24):
    """
    Make predictions with uncertainty bands.
    
    Returns:
        dict with predictions, quantiles, and actuals
    """
    # Get latest data
    latest = df.iloc[-hours_ahead:]
    
    # Engineer features
    df_feat = forecaster.engineer_features(df)
    latest_feat = df_feat.iloc[-hours_ahead:]
    
    # Prepare features for both models
    X_stat = latest_feat[forecaster.feature_names_stat].copy()
    X_phys = latest_feat[forecaster.feature_names_phys].copy()
    
    # Handle NaNs
    X_stat = X_stat.ffill().bfill().fillna(0)
    X_phys = X_phys.ffill().bfill().fillna(0)
    
    # Point predictions
    y_pred_stat = forecaster.stat_model.predict(X_stat)
    y_pred_phys = forecaster.phys_model.predict(X_phys)
    
    # Quantile predictions
    quantiles_stat = {}
    quantiles_phys = {}
    
    for q, model in forecaster.quantile_models_stat.items():
        quantiles_stat[q] = model.predict(X_stat)
    
    for q, model in forecaster.quantile_models_phys.items():
        quantiles_phys[q] = model.predict(X_phys)
    
    return {
        'times': latest.index,
        'actual': latest['price'].values,
        'pred_stat': y_pred_stat,
        'pred_phys': y_pred_phys,
        'quantiles_stat': quantiles_stat,
        'quantiles_phys': quantiles_phys,
    }


def plot_forecast_with_uncertainty(results, title="Price Forecast"):
    """Create forecast plot with uncertainty bands."""
    fig = go.Figure()
    
    times = results['times']
    
    # Physics model with uncertainty
    if 0.9 in results['quantiles_phys'] and 0.1 in results['quantiles_phys']:
        fig.add_trace(go.Scatter(
            x=times,
            y=results['quantiles_phys'][0.9],
            fill=None,
            mode='lines',
            line_color='rgba(74, 158, 255, 0)',
            showlegend=False,
            name='Physics 90%'
        ))
        fig.add_trace(go.Scatter(
            x=times,
            y=results['quantiles_phys'][0.1],
            fill='tonexty',
            mode='lines',
            line_color='rgba(74, 158, 255, 0)',
            fillcolor='rgba(74, 158, 255, 0.15)',
            name='Physics 80% CI',
            showlegend=True
        ))
    
    # Statistical model with uncertainty
    if 0.9 in results['quantiles_stat'] and 0.1 in results['quantiles_stat']:
        fig.add_trace(go.Scatter(
            x=times,
            y=results['quantiles_stat'][0.9],
            fill=None,
            mode='lines',
            line_color='rgba(255, 107, 107, 0)',
            showlegend=False,
            name='Statistical 90%'
        ))
        fig.add_trace(go.Scatter(
            x=times,
            y=results['quantiles_stat'][0.1],
            fill='tonexty',
            mode='lines',
            line_color='rgba(255, 107, 107, 0)',
            fillcolor='rgba(255, 107, 107, 0.15)',
            name='Statistical 80% CI',
            showlegend=True
        ))
    
    # Point predictions
    fig.add_trace(go.Scatter(
        x=times,
        y=results['pred_phys'],
        mode='lines',
        name='Physics-Informed',
        line=dict(color='#4a9eff', width=2.5)
    ))
    
    fig.add_trace(go.Scatter(
        x=times,
        y=results['pred_stat'],
        mode='lines',
        name='Statistical',
        line=dict(color='#ff6b6b', width=2.5, dash='dash')
    ))
    
    # Actual
    fig.add_trace(go.Scatter(
        x=times,
        y=results['actual'],
        mode='lines+markers',
        name='Actual',
        line=dict(color='#ffffff', width=3),
        marker=dict(size=6)
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Price (EUR/MWh)",
        height=500,
        template='plotly_dark',
        paper_bgcolor='#1e1e1e',
        plot_bgcolor='#2d2d2d',
        font=dict(color='#ffffff'),
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(45, 45, 45, 0.8)'
        )
    )
    
    return fig


def plot_physics_context(df, results):
    """
    Plot residual demand and renewable penetration alongside price,
    showing the physical drivers of price over the forecast window.
    """
    times = results['times']
    window = df.loc[times[0]:times[-1]] if times[0] in df.index else df.iloc[-len(times):]

    fig = go.Figure()

    # Residual demand on primary y-axis
    if 'residual_demand' in window.columns:
        fig.add_trace(go.Scatter(
            x=window.index,
            y=window['residual_demand'] / 1000,  # GW
            mode='lines',
            name='Residual Demand (GW)',
            line=dict(color='#f39c12', width=2),
            yaxis='y1'
        ))

    # Renewable penetration on secondary y-axis
    if 'renewable_penetration' in window.columns:
        fig.add_trace(go.Scatter(
            x=window.index,
            y=window['renewable_penetration'] * 100,
            mode='lines',
            name='Renewable Penetration (%)',
            line=dict(color='#2ecc71', width=2, dash='dot'),
            yaxis='y2'
        ))

    # Actual price on tertiary axis
    fig.add_trace(go.Scatter(
        x=window.index,
        y=window['price'],
        mode='lines',
        name='Actual Price (EUR/MWh)',
        line=dict(color='#ffffff', width=2),
        yaxis='y3'
    ))

    fig.update_layout(
        xaxis=dict(title='Time', domain=[0.08, 1.0]),
        yaxis=dict(
            title='Residual Demand (GW)',
            title_font=dict(color='#f39c12'),
            tickfont=dict(color='#f39c12'),
            side='left'
        ),
        yaxis2=dict(
            title='Renewable Penetration (%)',
            title_font=dict(color='#2ecc71'),
            tickfont=dict(color='#2ecc71'),
            overlaying='y',
            side='right',
            position=1.0
        ),
        yaxis3=dict(
            title='Price (EUR/MWh)',
            title_font=dict(color='#ffffff'),
            tickfont=dict(color='#ffffff'),
            overlaying='y',
            side='left',
            position=0.0,
            anchor='free'
        ),
        template='plotly_dark',
        paper_bgcolor='#1e1e1e',
        plot_bgcolor='#2d2d2d',
        font=dict(color='#ffffff'),
        hovermode='x unified',
        height=380,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(45,45,45,0.8)'
        )
    )
    return fig


def build_annotation_panel(df, results):
    """
    Build a text annotation explaining which physics features
    are most active in the current forecast window.
    Returns a list of (label, value, note) tuples.
    """
    times = results['times']
    window = df.loc[times[0]:times[-1]] if times[0] in df.index else df.iloc[-len(times):]

    annotations = []

    if 'residual_demand' in window.columns and 'demand' in window.columns:
        residual_mean = window['residual_demand'].mean() / 1000
        demand_mean = window['demand'].mean() / 1000
        frac = residual_mean / demand_mean if demand_mean > 0 else 0
        annotations.append((
            "Residual Demand",
            f"{residual_mean:.1f} GW ({frac*100:.0f}% of total)",
            "High residual demand -> gas/coal price setting -> physics constraints most active"
            if frac > 0.6 else
            "Low residual demand -> renewables displacing thermal -> reduced physics advantage"
        ))

    if 'residual_ramp' in window.columns:
        max_ramp = window['residual_ramp'].abs().max() / 1000
        annotations.append((
            "Max Residual Ramp",
            f"{max_ramp:.1f} GW/hr",
            "Large ramp detected -> ramp rate constraint is active in physics model"
            if max_ramp > 2.0 else
            "Ramp rates stable -> ramp constraint less influential this window"
        ))

    if 'renewable_penetration' in window.columns:
        ren_mean = window['renewable_penetration'].mean() * 100
        annotations.append((
            "Avg Renewable Penetration",
            f"{ren_mean:.1f}%",
            "High renewable share -> duck curve effects likely -> physics model has advantage at ramp hours"
            if ren_mean > 40 else
            "Moderate renewable share -> merit order physics still relevant"
        ))

    if 'evening_solar_drop' in window.columns:
        evening_stress = window['evening_solar_drop'].max() / 1000
        if evening_stress > 0.5:
            annotations.append((
                "Evening Solar Drop",
                f"{evening_stress:.1f} GW peak",
                "Significant solar ramp-down detected -> duck curve stress -> physics model captures this explicitly"
            ))

    return annotations


def main():
    """Main dashboard."""
    
    # Header
    st.markdown('<div class="main-title">Energy Price Forecasting</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Physics-Informed Machine Learning for Electricity Markets</div>', 
                unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Configuration")
    
    country_names = {
        'DE': 'Germany',
        'ES': 'Spain',
        'FR': 'France',
        'NL': 'Netherlands',
        'DK': 'Denmark',
        'GB': 'Great Britain',
    }
    
    selected_country = st.sidebar.selectbox(
        "Market",
        options=list(country_names.keys()),
        format_func=lambda x: country_names[x],
        index=1
    )
    
    days_history = st.sidebar.slider("Historical Data (days)", 7, 90, 30)
    
    # Load model
    forecaster = load_forecaster(selected_country)
    
    if forecaster:
        st.sidebar.success(f"Model Loaded")
        train_date = forecaster.metadata['train_date'][:10]
        st.sidebar.caption(f"Trained: {train_date}")
    else:
        st.sidebar.warning("No pretrained model found")
        st.sidebar.caption("Train a model using model_training_pipeline.py")
    
    st.sidebar.markdown("---")
    
    # About section
    with st.sidebar.expander("About This Dashboard"):
        st.markdown("""
        **Two Approaches:**
        
        **Statistical ML**: Pure data-driven forecasting using historical patterns.
        
        **Physics-Informed ML**: Incorporates domain constraints:
        - Ramp rate limits
        - Capacity bounds
        - Merit order dispatch
        - Duck curve dynamics
        
        **Data**: ENTSO-E Transparency Platform (real-time European grid data)
        """)
    
    # Fetch data
    try:
        with st.spinner(f'Fetching data for {country_names[selected_country]}...'):
            df = fetch_latest_data(selected_country, days=days_history)
        
        if len(df) < 100:
            st.error(f"Insufficient data for {country_names[selected_country]}. Try a different market.")
            return
        
        # Make predictions if model exists
        if forecaster:
            results = make_predictions(forecaster, df, hours_ahead=min(24, len(df)//2))
            
            mae_stat = np.mean(np.abs(results['actual'] - results['pred_stat']))
            mae_phys = np.mean(np.abs(results['actual'] - results['pred_phys']))
            improvement = (mae_stat - mae_phys) / mae_stat * 100
        else:
            results = None
            mae_stat = mae_phys = improvement = 0
        
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        current_price = df['price'].dropna().iloc[-1]
        price_24h_ago = df['price'].dropna().iloc[-25] if len(df['price'].dropna()) > 25 else current_price
        demand_series = df['demand'].dropna()
        current_demand = demand_series.iloc[-1] / 1000 if len(demand_series) > 0 else float('nan')
        
        # Get utilization if available
        if 'utilization' in df.columns:
            current_util = df['utilization'].iloc[-1]
            if current_util > 0.8:
                util_status = "High"
                util_color = "inverse"
            elif current_util > 0.6:
                util_status = "Normal"
                util_color = "off"
            else:
                util_status = "Low"
                util_color = "normal"
        else:
            current_util = 0
            util_status = "N/A"
            util_color = "off"
        
        with col1:
            st.metric(
                "Current Price",
                f"{current_price:.1f} EUR/MWh",
                delta=f"{current_price - price_24h_ago:.1f}",
                help="Most recent day-ahead market price. Delta shows change vs same hour yesterday."
            )
        
        with col2:
            st.metric(
                "Demand",
                f"{current_demand:.1f} GW" if not np.isnan(current_demand) else "N/A",
                help="Last reported actual load (MW -> GW). ENTSO-E publishes with ~1hr lag so this may trail price by an hour."
            )
        
        with col3:
            if results:
                st.metric(
                    "Statistical MAE",
                    f"{mae_stat:.1f} EUR/MWh",
                    help="Mean Absolute Error of the baseline statistical model (no physics) on the last 24hr window. "
                    "Lower is better. Typical day-ahead MAE for European markets: 3-10 EUR/MWh."
                )
            else:
                st.metric("Statistical MAE", "N/A")
        
        with col4:
            if results:
                st.metric(
                    "Physics MAE",
                    f"{mae_phys:.1f} EUR/MWh",
                    delta=f"{improvement:.1f}%" if improvement != 0 else "0%",
                    delta_color="inverse" if improvement > 0 else "normal",
                    help="MAE of the physics-informed model. Delta shows % improvement over the statistical baseline. "
                    "Positive delta (green) = physics constraints helped. "
                    "Negative delta (red) = statistical model was more accurate — common in nuclear-dominated grids (e.g. FR) "
                    "where physical dispatch is flat and historical patterns dominate."
                )
            else:
                st.metric("Physics MAE", "N/A")
        
        # Tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "Live Forecast",
            "Performance Analysis",
            "Generation Mix",
            "Drift Monitoring"
        ])
        
        with tab1:
            st.markdown("### Price Forecast with Uncertainty")
            
            if results:
                fig = plot_forecast_with_uncertainty(results, title="")
                st.plotly_chart(fig, use_container_width=True)

                # Physics drivers annotation panel
                st.markdown("### Physical Drivers in This Window")
                annotations = build_annotation_panel(df, results)
                if annotations:
                    ann_cols = st.columns(len(annotations))
                    for col, (label, value, note) in zip(ann_cols, annotations):
                        with col:
                            st.metric(label, value)
                            st.caption(note)
                else:
                    st.info("Physics feature data not available for annotation")

                st.markdown("### Residual Demand & Renewable Penetration vs Price")
                st.caption(
                    "Residual demand (demand minus renewables) is the key driver of which thermal plant "
                    "sets the marginal price. High residual demand with large ramps is where physics constraints "
                    "add the most forecast value."
                )
                fig_ctx = plot_physics_context(df, results)
                st.plotly_chart(fig_ctx, use_container_width=True)

                # Forecast table
                st.markdown("### Detailed Predictions")
                forecast_df = pd.DataFrame({
                    'Time': results['times'],
                    'Actual': results['actual'],
                    'Statistical': results['pred_stat'],
                    'Physics': results['pred_phys'],
                    'Stat Error': np.abs(results['actual'] - results['pred_stat']),
                    'Phys Error': np.abs(results['actual'] - results['pred_phys'])
                })
                st.dataframe(forecast_df.style.format({
                    'Actual': '{:.1f}',
                    'Statistical': '{:.1f}',
                    'Physics': '{:.1f}',
                    'Stat Error': '{:.1f}',
                    'Phys Error': '{:.1f}'
                }), use_container_width=True)
            else:
                st.info("Train a model to see forecasts with uncertainty bands")
        
        with tab2:
            st.markdown("### Model Performance Analysis")
            
            if forecaster and results:
                # Performance by regime
                st.markdown("#### Performance by Price Regime")
                
                metadata = forecaster.metadata
                
                regime_data = {
                    'Regime': ['Normal Prices', 'Price Spikes', 'Low Prices'],
                    'Statistical MAE': [
                        metadata.get('mae_stat_normal', 0),
                        metadata.get('mae_stat_spike', 0),
                        metadata.get('mae_stat_low', 0)
                    ],
                    'Physics MAE': [
                        metadata.get('mae_phys_normal', 0),
                        metadata.get('mae_phys_spike', 0),
                        metadata.get('mae_phys_low', 0)
                    ],
                    'Improvement': [
                        metadata.get('improvement_normal', 0),
                        metadata.get('improvement_spike', 0),
                        metadata.get('improvement_low', 0)
                    ]
                }
                
                regime_df = pd.DataFrame(regime_data)

                # CV std error bars - pull from metadata if available
                # These are stored as mae_std_stat / mae_std_phys from time-series CV
                err_stat = metadata.get('mae_std_stat', None)
                err_phys = metadata.get('mae_std_phys', None)

                # Build per-regime error bar arrays (uniform across regimes if only overall CV std available)
                def make_err(val, n=3):
                    return [val] * n if val is not None else None

                fig_regime = go.Figure()
                fig_regime.add_trace(go.Bar(
                    name='Statistical',
                    x=regime_df['Regime'],
                    y=regime_df['Statistical MAE'],
                    marker_color='#ff6b6b',
                    error_y=dict(
                        type='data',
                        array=make_err(err_stat),
                        visible=err_stat is not None,
                        color='rgba(255,107,107,0.6)',
                        thickness=2,
                        width=6
                    )
                ))
                fig_regime.add_trace(go.Bar(
                    name='Physics',
                    x=regime_df['Regime'],
                    y=regime_df['Physics MAE'],
                    marker_color='#4a9eff',
                    error_y=dict(
                        type='data',
                        array=make_err(err_phys),
                        visible=err_phys is not None,
                        color='rgba(74,158,255,0.6)',
                        thickness=2,
                        width=6
                    )
                ))
                
                fig_regime.update_layout(
                    barmode='group',
                    yaxis_title='MAE (EUR/MWh)',
                    template='plotly_dark',
                    paper_bgcolor='#1e1e1e',
                    plot_bgcolor='#2d2d2d',
                    font=dict(color='#ffffff'),
                    height=400
                )
                
                st.plotly_chart(fig_regime, use_container_width=True)
                
                # Show improvement percentages
                col1, col2, col3 = st.columns(3)

                def improvement_metric(col, label, key, help_text):
                    val = metadata.get(key, 0)
                    delta_label = f"physics better by {val:.1f}%" if val > 0 else (
                        f"stat better by {abs(val):.1f}%" if val < 0 else "no difference"
                    )
                    col.metric(
                        label,
                        f"{val:.1f}%",
                        delta=delta_label,
                        delta_color="normal" if val > 0 else ("inverse" if val < 0 else "off"),
                        help=help_text
                    )

                improvement_metric(
                    col1, "Normal Prices (10th-90th pct)", "improvement_normal",
                    "Physics model MAE improvement over statistical baseline during typical price hours. "
                    "Positive = physics model is more accurate. "
                    "Merit order and residual demand constraints drive this regime."
                )
                improvement_metric(
                    col2, "Price Spikes (>90th pct)", "improvement_spike",
                    "Improvement during high-price events. Physics constraints on ramp rates and capacity "
                    "limits should help here — but only if spikes are physically driven. "
                    "Gas/CO2 price shocks (not modelled) can limit improvement."
                )
                improvement_metric(
                    col3, "Low Prices (<10th pct)", "improvement_low",
                    "Improvement during low/negative price events, typically caused by renewable oversupply. "
                    "Physics model captures minimum stable generation constraints, "
                    "which matter when supply exceeds demand."
                )
                
                # Error distribution
                st.markdown("#### Error Distribution")
                
                errors_stat = results['actual'] - results['pred_stat']
                errors_phys = results['actual'] - results['pred_phys']
                
                fig_err = go.Figure()
                fig_err.add_trace(go.Histogram(
                    x=errors_stat,
                    name='Statistical',
                    opacity=0.6,
                    marker_color='#ff6b6b',
                    nbinsx=30
                ))
                fig_err.add_trace(go.Histogram(
                    x=errors_phys,
                    name='Physics',
                    opacity=0.6,
                    marker_color='#4a9eff',
                    nbinsx=30
                ))
                
                fig_err.update_layout(
                    barmode='overlay',
                    xaxis_title='Forecast Error (EUR/MWh)',
                    yaxis_title='Frequency',
                    template='plotly_dark',
                    paper_bgcolor='#1e1e1e',
                    plot_bgcolor='#2d2d2d',
                    font=dict(color='#ffffff'),
                    height=400
                )
                
                st.plotly_chart(fig_err, use_container_width=True)
                
            else:
                st.info("Train a model to see performance analysis")
        
        with tab3:
            st.markdown("### Generation Mix")
            
            # Weekly generation stack
            week_data = df.iloc[-168:] if len(df) >= 168 else df
            
            gen_cols = [c for c in ['nuclear', 'coal', 'gas', 'wind_onshore', 'solar'] 
                       if c in df.columns and df[c].sum() > 0]
            
            if gen_cols:
                fig_gen = go.Figure()
                
                colors = {
                    'nuclear': '#9b59b6',
                    'coal': '#95a5a6',
                    'gas': '#e74c3c',
                    'wind_onshore': '#3498db',
                    'solar': '#f39c12'
                }
                
                for col in gen_cols:
                    c = colors.get(col, '#ffffff')
                    fig_gen.add_trace(go.Scatter(
                        x=week_data.index,
                        y=week_data[col],
                        mode='lines',
                        name=col.replace('_', ' ').title(),
                        stackgroup='one',
                        fillcolor=c,
                        line=dict(color=c, width=0.5)
                    ))
                
                fig_gen.add_trace(go.Scatter(
                    x=week_data.index,
                    y=week_data['demand'],
                    mode='lines',
                    name='Demand',
                    line=dict(color='#ffffff', width=2, dash='dash')
                ))
                
                fig_gen.update_layout(
                    xaxis_title="Time",
                    yaxis_title="Power (MW)",
                    template='plotly_dark',
                    paper_bgcolor='#1e1e1e',
                    plot_bgcolor='#2d2d2d',
                    font=dict(color='#ffffff'),
                    hovermode='x unified',
                    height=500
                )
                
                st.plotly_chart(fig_gen, use_container_width=True)
                
                # Generation mix pie
                st.markdown("### Average Generation Mix")
                
                gen_avg = week_data[gen_cols].mean()
                
                fig_pie = go.Figure(data=[go.Pie(
                    labels=[c.replace('_', ' ').title() for c in gen_cols],
                    values=gen_avg.values,
                    marker=dict(colors=[colors.get(c, '#ffffff') for c in gen_cols])
                )])
                
                fig_pie.update_layout(
                    template='plotly_dark',
                    paper_bgcolor='#1e1e1e',
                    font=dict(color='#ffffff'),
                    height=400
                )
                
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.warning("Generation data not available for this market")
        
        with tab4:
            st.markdown("### Model Drift Monitoring")
            
            if forecaster and results:
                # Calculate rolling MAE
                window_size = min(7 * 24, len(results['actual']) // 2)
                
                if window_size > 24:
                    rolling_mae_stat = []
                    rolling_mae_phys = []
                    
                    for i in range(window_size, len(results['actual'])):
                        window_actual = results['actual'][i-window_size:i]
                        window_pred_stat = results['pred_stat'][i-window_size:i]
                        window_pred_phys = results['pred_phys'][i-window_size:i]
                        
                        rolling_mae_stat.append(np.mean(np.abs(window_actual - window_pred_stat)))
                        rolling_mae_phys.append(np.mean(np.abs(window_actual - window_pred_phys)))
                    
                    # Drift detection
                    baseline_mae_stat = forecaster.metadata['mae_stat']
                    baseline_mae_phys = forecaster.metadata['mae_phys']
                    
                    current_mae_stat = rolling_mae_stat[-1] if rolling_mae_stat else mae_stat
                    current_mae_phys = rolling_mae_phys[-1] if rolling_mae_phys else mae_phys
                    
                    drift_stat = ((current_mae_stat - baseline_mae_stat) / baseline_mae_stat) * 100
                    drift_phys = ((current_mae_phys - baseline_mae_phys) / baseline_mae_phys) * 100
                    
                    # Status indicators
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if abs(drift_phys) < 10:
                            st.success("Model Healthy")
                        elif abs(drift_phys) < 20:
                            st.warning("Minor Drift Detected")
                        else:
                            st.error("Significant Drift - Retrain Recommended")
                    
                    with col2:
                        st.metric(
                            "Statistical Drift",
                            f"{drift_stat:+.1f}%",
                            delta=f"{current_mae_stat - baseline_mae_stat:.1f} EUR/MWh",
                            help="How much the statistical model's rolling MAE has changed vs its training-time baseline. "
                            "Positive = model is degrading. Typically caused by seasonal regime shifts or structural market changes."
                        )
                    
                    with col3:
                        st.metric(
                            "Physics Drift",
                            f"{drift_phys:+.1f}%",
                            delta=f"{current_mae_phys - baseline_mae_phys:.1f} EUR/MWh",
                            help="Same as statistical drift but for the physics-informed model. "
                            "Physics models tend to drift less during renewable mix changes because "
                            "the residual demand feature adapts to new generation patterns implicitly."
                        )
                    
                    # Rolling MAE plot
                    fig_drift = go.Figure()
                    
                    times_drift = results['times'][window_size:]
                    
                    fig_drift.add_trace(go.Scatter(
                        x=times_drift,
                        y=rolling_mae_stat,
                        mode='lines',
                        name='Statistical Rolling MAE',
                        line=dict(color='#ff6b6b', width=2)
                    ))
                    
                    fig_drift.add_trace(go.Scatter(
                        x=times_drift,
                        y=rolling_mae_phys,
                        mode='lines',
                        name='Physics Rolling MAE',
                        line=dict(color='#4a9eff', width=2)
                    ))
                    
                    # Baseline reference
                    fig_drift.add_hline(
                        y=baseline_mae_stat,
                        line_dash="dash",
                        line_color="#ff6b6b",
                        opacity=0.5,
                        annotation_text="Stat Baseline"
                    )
                    
                    fig_drift.add_hline(
                        y=baseline_mae_phys,
                        line_dash="dash",
                        line_color="#4a9eff",
                        opacity=0.5,
                        annotation_text="Phys Baseline"
                    )
                    
                    fig_drift.update_layout(
                        title="Rolling MAE (7-day window)",
                        xaxis_title="Time",
                        yaxis_title="MAE (EUR/MWh)",
                        template='plotly_dark',
                        paper_bgcolor='#1e1e1e',
                        plot_bgcolor='#2d2d2d',
                        font=dict(color='#ffffff'),
                        height=400
                    )
                    
                    st.plotly_chart(fig_drift, use_container_width=True)
                    
                    # Recommendations
                    st.markdown("### Recommendations")
                    
                    if abs(drift_phys) < 10:
                        st.success("Model is performing well. No action needed.")
                    elif abs(drift_phys) < 20:
                        st.warning("Minor drift detected. Continue monitoring.")
                    else:
                        st.error("Significant drift detected. Retrain recommended.")
                        if st.button("Retrain Model"):
                            st.info("Retraining functionality coming soon. Use model_training_pipeline.py manually for now.")
                else:
                    st.info("Insufficient data for drift monitoring. Need at least 7 days of data.")
            else:
                st.info("Train a model to enable drift monitoring")
        
        # Footer
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.caption("Data: ENTSO-E Transparency Platform")
        
        with col2:
            if len(df) > 0:
                last_ts = df['price'].dropna().index[-1]
                lag_minutes = int((datetime.now() - last_ts.to_pydatetime().replace(tzinfo=None)).total_seconds() / 60)
                if lag_minutes < 90:
                    st.caption(f"Last price: {last_ts.strftime('%Y-%m-%d %H:%M UTC')} ({lag_minutes} min ago)")
                else:
                    st.caption(f"Last price: {last_ts.strftime('%Y-%m-%d %H:%M UTC')} — ENTSO-E publish lag expected")
        
        with col3:
            st.caption(f"Samples: {len(df):,} hours | Cache refreshes every 15 min")
    
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.info("Please verify:\n- ENTSO-E API key is valid\n- Selected market has data\n- Internet connection is stable")


if __name__ == "__main__":
    main()
