## -> change to stoxx 50 instead of msci world and change occurrencies and titles.##
import streamlit as st 
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import io
import datetime
from dateutil.relativedelta import relativedelta
from pandas.tseries.offsets import MonthEnd, MonthBegin
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Set page configuration with custom theme
st.set_page_config(
    layout="wide",
    page_title='Dynamic Portfolio Strategy (Europe)',
    page_icon=':chart_with_upwards_trend:',
    initial_sidebar_state='expanded'
)

# Apply custom CSS for fonts and backgrounds
st.markdown(
    """
    <style>
    /* Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap');
    html, body, [class*="css"]  {
        font-family: 'Roboto', sans-serif;
        font-size: 16px;
    }
    /* Background */
    body {
        background: linear-gradient(90deg, #f0f2f6 0%, #e2e6ee 100%);
    }
    /* Sidebar */
    .css-1d391kg {
        background-color: #1f4e79 !important;
    }
    .css-1d391kg .css-1v3fvcr {
        color: white;
    }
    .css-1d391kg .css-1v3fvcr .css-1nnyp7m {
        color: white;
    }
    /* Headers */
    h1, h2, h3, h4 {
        color: #1f4e79;
        font-weight: 700;
    }
    /* Text */
    p, div, label, span {
        color: #333333;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title and Introduction
st.title('Factor Investing: Adapting to Economic Phases (Europe Version)')

st.markdown('''
## Introduction

This app presents the **Dynamic Portfolio Strategy (European)**, an approach designed to optimize returns by dynamically allocating assets based on economic phases.

Using the **OECD CLI Diffusion Index** as a macroeconomic indicator, this strategy identifies four distinct economic phases:
- **Recovery**: Early growth after a contraction.
- **Expansion**: Sustained growth phase.
- **Slowdown**: Decline from peak economic activity.
- **Contraction**: Widespread economic decline.

By aligning factor-based ETFs with these phases, the strategy seeks to:
1. Outperform benchmarks (STOXX 600, MSCI World).
2. Minimize drawdowns during adverse market conditions.

The analysis evaluates this strategy’s performance, highlighting its ability to leverage factors such as Value, Quality, Momentum, and Low Volatility across economic cycles.
''')

st.markdown('---')

# Sidebar for navigation
st.sidebar.title('Navigation')
sections = [
    'Methodology',
    'Portfolio Construction',
    'Mean Portfolio Evolution'
]
selected_section = st.sidebar.radio('Go to', sections)

# ------------------------------------------------------------------------------
# 1) OECD CLI Data Function
@st.cache_data
def get_oecd_data(countries):
    database = '@DF_CLI'
    frequency = 'M'
    indicator = 'LI..'
    unit_of_measure = 'AA...'
    start_period = '1990-01'
    
    country_code = "+".join(countries)
    query_text = f"{database}/{country_code}.{frequency}.{indicator}.{unit_of_measure}?startPeriod={start_period}&dimensionAtObservation=AllDimensions"
    url = f"https://sdmx.oecd.org/public/rest/data/OECD.SDD.STES,DSD_STES{query_text}"

    headers = { 
        'User-Agent': 'Mozilla/5.0', 
        'Accept': 'application/vnd.sdmx.data+csv; charset=utf-8' 
    }
    
    download = requests.get(url=url, headers=headers)
    df = pd.read_csv(io.StringIO(download.text))
    return df

# List of (European + some global) countries
countries = [
    'AUS', 'CAN', 'FRA', 'DEU', 'ITA', 'JPN', 'DNK', 'KOR', 'MEX', 'ESP',
    'TUR', 'GBR', 'USA', 'BRA', 'CHN', 'IND', 'IDN', 'ZAF'
]

with st.spinner('Fetching OECD CLI data...'):
    cli_data = get_oecd_data(countries)

# Reshape the data
pivot_data = cli_data.pivot(index='TIME_PERIOD', columns='REF_AREA', values='OBS_VALUE')
pivot_data.columns.name = None
pivot_data.fillna(method='ffill', inplace=True)
pivot_data.index = pd.to_datetime(pivot_data.index).tz_localize(None)

# Shift by 1 month for OECD CLI lag
pivot_data.index = pivot_data.index + pd.DateOffset(months=1)

# Calculate Diffusion Index
pivot_data_change = pivot_data.diff()
diffusion_index   = (pivot_data_change > 0).sum(axis=1) / len(pivot_data.columns)
pivot_data['DI']  = diffusion_index
pivot_data['DI_change'] = pivot_data['DI'].diff()
pivot_data['DI_direction'] = np.where(pivot_data['DI_change'] > 0, 'Rising', 'Falling')

def determine_phase(row):
    if row['DI'] < 0.5 and row['DI_direction'] == 'Rising':
        return 'Recovery'
    elif row['DI'] >= 0.5 and row['DI_direction'] == 'Rising':
        return 'Expansion'
    elif row['DI'] >= 0.5 and row['DI_direction'] == 'Falling':
        return 'Slowdown'
    elif row['DI'] < 0.5 and row['DI_direction'] == 'Falling':
        return 'Contraction'
    else:
        return 'Unknown'

pivot_data['Phase'] = pivot_data.apply(determine_phase, axis=1)
pivot_data.dropna(subset=['Phase'], inplace=True)

# ------------------------------------------------------------------------------
# 2) European Factor ETFs
tickers = [
    'IEVL.MI',  # Value
    'IEQU.MI',  # Quality
    'IEMO.MI',  # Momentum
    'IFSE.MI',  # Multifactor
    'IESZ.MI',  # Size
    'MVEU.MI',  # Minimum Volatility
    'EHF1.DE',  # High Dividend
    'LEONIA.MI' # Cash
]

etf_labels = {
    'IEVL.MI': 'Value',
    'IEQU.MI': 'Quality',
    'IEMO.MI': 'Momentum',
    'IFSE.MI': 'Multifactor',
    'MVEU.MI': 'Minimum Volatility',
    'IESZ.MI': 'Size',
    'EHF1.DE': 'High Dividend',
    'LEONIA.MI': 'Cash'
}

labels_df = pd.DataFrame(list(etf_labels.items()), columns=['Ticker','Label'])
labels_df['Labels'] = labels_df['Label'].apply(lambda x: [lbl.strip() for lbl in x.split(',')])

start_date = '2016-01-01'
end_date   = datetime.datetime.today().strftime('%Y-%m-%d')

@st.cache_data
def get_etf_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)['Close']
    return data

with st.spinner('Fetching ETF data from Yahoo Finance...'):
    data = get_etf_data(tickers, start_date, end_date)

data.fillna(method='ffill', inplace=True)
data.dropna(inplace=True)
data.index = data.index.tz_localize(None)

available_tickers = data.columns.tolist()
labels_df = labels_df[labels_df['Ticker'].isin(available_tickers)]

# ------------------------------------------------------------------------------
# 3) Daily Returns & Phase Alignment
daily_returns = data.pct_change().fillna(0)
pivot_data.index = pivot_data.index.tz_localize(None)

# Reindex pivot_data to daily
required_columns = ['DI','DI_change','DI_direction','Phase']
pivot_data_daily = pivot_data[required_columns].reindex(daily_returns.index, method='ffill')
pivot_data_daily.fillna(method='bfill', inplace=True)
daily_phases = pivot_data_daily['Phase']

# Factor -> ETF mapping
factor_etf_mapping = {}
for _, row in labels_df.iterrows():
    for factor_label in row['Labels']:
        factor_etf_mapping.setdefault(factor_label, []).append(row['Ticker'])

unique_phases = daily_phases.dropna().unique()
factor_performance = pd.DataFrame()

# ------------------------------------------------------------------------------
# Factor Performance by Phase
for phase in unique_phases:
    phase_dates = daily_phases[daily_phases == phase].index
    phase_dates = phase_dates.intersection(daily_returns.index)
    if phase_dates.empty:
        continue
    
    factor_cum_returns = {}
    for factor, etfs_in_factor in factor_etf_mapping.items():
        valid_etfs = [etf for etf in etfs_in_factor if etf in daily_returns.columns]
        if not valid_etfs:
            continue
        
        phase_returns = daily_returns.loc[phase_dates, valid_etfs]
        cum_returns   = (1 + phase_returns).cumprod()
        mean_cum_ret  = cum_returns.iloc[-1].mean() - 1
        factor_cum_returns[factor] = mean_cum_ret
    
    df_factors = pd.DataFrame.from_dict(factor_cum_returns, orient='index', columns=[phase])
    factor_performance = pd.concat([factor_performance, df_factors], axis=1)

factor_performance.fillna(0, inplace=True)

# ------------------------------------------------------------------------------
# Pick Top 3 ETFs each phase
best_etfs_per_phase = {}
for phase in unique_phases:
    phase_dates = daily_phases[daily_phases == phase].index
    phase_dates = phase_dates.intersection(daily_returns.index)
    if phase_dates.empty:
        continue
    
    phase_returns = daily_returns.loc[phase_dates]
    cum_returns   = (1 + phase_returns).cumprod().iloc[-1] - 1
    sorted_etfs   = cum_returns.sort_values(ascending=False)
    top_etfs      = sorted_etfs.head(3)
    best_etfs_per_phase[phase] = top_etfs.index.tolist()

# ------------------------------------------------------------------------------
# Build Dynamic Weights
weights_df = pd.DataFrame(0, index=daily_returns.index, columns=daily_returns.columns)
for date in daily_returns.index:
    ph = daily_phases.loc[date]
    picks = best_etfs_per_phase.get(ph, [])
    if picks:
        w = np.repeat(1.0 / len(picks), len(picks))
        weights_df.loc[date, picks] = w

weights_df.fillna(method='ffill', inplace=True)

# Prepare portfolio data
latest_date = weights_df.index.max()
current_weights = weights_df.loc[latest_date]
current_weights = current_weights[current_weights>0]
etfs_in_portfolio = weights_df.columns[(weights_df!=0).any()].tolist()
weights_over_time = weights_df[etfs_in_portfolio]
mean_weights = weights_over_time.mean()
mean_weights = mean_weights[mean_weights>0]

# ------------------------------------------------------------------------------
# 4) Portfolio Returns & Benchmarks
portfolio_returns = (daily_returns * weights_df.shift(1)).sum(axis=1)
portfolio_cum_returns = (1 + portfolio_returns).cumprod()

# STOXX 600
stoxx600 = yf.download('XSX6.MI', start=portfolio_returns.index.min(), end=end_date)['Close']
stoxx600.fillna(method='ffill', inplace=True)
stoxx600.dropna(inplace=True)
stoxx600.index = stoxx600.index.tz_localize(None)
stoxx600_returns = stoxx600.pct_change().fillna(0)
stoxx600_cum_returns = (1 + stoxx600_returns).cumprod()

# MSCI World
msci_world = yf.download('LYP6.DE', start=portfolio_returns.index.min(), end=end_date)['Close']
msci_world.fillna(method='ffill', inplace=True)
msci_world.dropna(inplace=True)
msci_world.index = msci_world.index.tz_localize(None)
msci_world_returns = msci_world.pct_change().fillna(0)
msci_world_cum_returns = (1 + msci_world_returns).cumprod()

# Align indexes
common_idx = portfolio_cum_returns.index \
    .intersection(stoxx600_cum_returns.index) \
    .intersection(msci_world_cum_returns.index)

portfolio_cum_returns_aligned   = portfolio_cum_returns.loc[common_idx]
stoxx600_cum_returns_aligned    = stoxx600_cum_returns.loc[common_idx]
msci_world_cum_returns_aligned  = msci_world_cum_returns.loc[common_idx]

portfolio_returns_aligned  = portfolio_returns.loc[common_idx]
stoxx600_returns_aligned   = stoxx600_returns.loc[common_idx]
msci_world_returns_aligned = msci_world_returns.loc[common_idx]

window_size = 252
def calculate_drawdown(cumulative_returns):
    peak = cumulative_returns.cummax()
    return (cumulative_returns - peak)/peak

# ------------------------------------------------------------------------------
# STREAMLIT SECTIONS
if selected_section == 'Methodology':
    st.markdown('---')
    st.markdown('''
    ## Methodology

    1. **OECD CLI Data (Shifted 1 Month)**:
       - The Diffusion Index (DI) is calculated monthly to categorize phases:
         - **Recovery**: DI < 0.5 & Rising
         - **Expansion**: DI ≥ 0.5 & Rising
         - **Slowdown**: DI ≥ 0.5 & Falling
         - **Contraction**: DI < 0.5 & Falling

    2. **ETF Selection**:
       - European Factor ETFs mapped to Value, Quality, Momentum, Size, etc.
       - Top 3 by cumulative return in each phase → equal weights.

    3. **Benchmarks**:
       - STOXX 600 (XSX6.MI) and MSCI World (URTH).
    ''')

    st.markdown('### Economic Phases Data')
    st.dataframe(pivot_data[['DI','DI_change','Phase']].tail(15).style.background_gradient(cmap='Blues'), use_container_width=True)

    st.markdown('### OECD CLI Diffusion Index (Shifted)')
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pivot_data.index, y=pivot_data['DI'], 
        mode='lines', name='Diffusion Index', line=dict(color='#1f4e79', width=2)
    ))
    fig.add_hline(y=0.5, line_dash='dash', line_color='red', annotation_text='Threshold=0.5', annotation_position='top left')
    fig.update_layout(
        title='OECD CLI Diffusion Index',
        xaxis_title='Date',
        yaxis_title='Diffusion Index',
        font=dict(size=14),
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)

elif selected_section == 'Portfolio Construction':
    st.markdown('---')
    st.markdown('## Factor Performance by Phase')

    # Display factor performance table
    st.markdown('### Factor Performance Data')
    st.dataframe(factor_performance.style.background_gradient(cmap='Blues'), use_container_width=True)

    # Heatmap
    st.markdown('### Factor Performance Heatmap')
    factor_performance_t = factor_performance.T
    plt.figure(figsize=(10, 6))
    sns.heatmap(factor_performance_t*100, annot=True, fmt='.2f', cmap='Blues', cbar_kws={'label': 'Return (%)'})
    st.pyplot(plt.gcf())

    st.markdown('---')
    st.markdown('### Top 3 ETFs per Phase')
    for ph in unique_phases:
        st.write(f"**{ph}**: {best_etfs_per_phase.get(ph, [])}")

    # Last Portfolio Weights
    weights_monthly = weights_df.resample('M').first()
    st.markdown('### Last Portfolio Weights (Monthly)')
    st.dataframe(weights_monthly.tail().style.background_gradient(cmap='Blues'), use_container_width=True)

    # Portfolio vs. Benchmarks
    st.markdown('### Portfolio Performance vs. STOXX 600 & MSCI World')
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=portfolio_cum_returns_aligned.index,
        y=portfolio_cum_returns_aligned,
        mode='lines',
        name='Dynamic Portfolio',
        line=dict(width=3)
    ))
    fig.add_trace(go.Scatter(
        x=stoxx600_cum_returns_aligned.index,
        y=stoxx600_cum_returns_aligned,
        mode='lines',
        name='STOXX 600 (XSX6.MI)',
        line=dict(dash='dash', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=msci_world_cum_returns_aligned.index,
        y=msci_world_cum_returns_aligned,
        mode='lines',
        name='MSCI World (URTH)',
        line=dict(dash='dot', width=2)
    ))
    fig.update_layout(
        title='Portfolio Performance',
        xaxis_title='Date',
        yaxis_title='Cumulative Returns',
        width=1100,
        height=600,
        font=dict(size=14)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Portfolio Allocations
    st.markdown('---')
    st.markdown('## Portfolio Allocations Over Time')
    weights_monthly = weights_over_time.resample('M').first()
    if latest_date not in weights_monthly.index:
        last_month_end = latest_date.replace(day=1)+MonthEnd(1)
        weights_monthly.loc[last_month_end] = weights_over_time.loc[latest_date]

    fig = make_subplots(
        rows=3, cols=1,
        row_heights=[0.5, 0.25, 0.25],
        vertical_spacing=0.08,
        subplot_titles=(
            'Portfolio Allocations (Monthly)',
            f'Current Allocation ({latest_date.strftime("%Y-%m-%d")})',
            'Mean Allocation (Entire Period)'
        ),
        specs=[[{"type":"xy"}],[{"type":"domain"}],[{"type":"domain"}]]
    )
    # Stacked area
    for etf in etfs_in_portfolio:
        fig.add_trace(
            go.Scatter(
                x=weights_monthly.index,
                y=weights_monthly[etf],
                mode='lines',
                stackgroup='one',
                name=etf
            ),
            row=1, col=1
        )
    fig.update_xaxes(row=1, col=1, tickformat='%Y-%m', tickangle=45, nticks=20)
    fig.update_yaxes(title_text='Weight', row=1, col=1)

    # Current pie
    fig.add_trace(
        go.Pie(
            labels=current_weights.index,
            values=current_weights.values,
            hole=0.4,
            sort=False
        ),
        row=2, col=1
    )
    # Mean pie
    fig.add_trace(
        go.Pie(
            labels=mean_weights.index,
            values=mean_weights.values,
            hole=0.4,
            sort=False
        ),
        row=3, col=1
    )

    fig.update_layout(
        height=1500,
        hovermode='x unified',
        legend_title='ETF',
        font=dict(size=14)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Rolling Sharpe Ratio
    st.markdown('---')
    st.markdown('## Rolling Sharpe Ratio Comparison')

    portfolio_rolling_sharpe = portfolio_returns_aligned.rolling(window_size).apply(
        lambda x: (x.mean() / x.std()) * np.sqrt(252) if x.std()!=0 else np.nan
    )
    msci_rolling_sharpe = msci_world_returns_aligned.rolling(window_size).apply(
        lambda x: (x.mean() / x.std()) * np.sqrt(252) if x.std()!=0 else np.nan
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=portfolio_rolling_sharpe.index, y=portfolio_rolling_sharpe,
        mode='lines', name='Dynamic Portfolio'
    ))
    fig.add_trace(go.Scatter(
        x=msci_rolling_sharpe.index, y=msci_rolling_sharpe,
        mode='lines', name='MSCI World (URTH)', line=dict(dash='dash')
    ))
    fig.update_layout(
        title='Rolling Sharpe Ratio',
        xaxis_title='Date',
        yaxis_title='Sharpe Ratio',
        hovermode='x unified',
        font=dict(size=14),
        width=1100,
        height=600
    )
    st.plotly_chart(fig, use_container_width=True)

    # Rolling Drawdown
    st.markdown('---')
    st.markdown('## Rolling Drawdown Comparison')
    portfolio_drawdown = calculate_drawdown(portfolio_cum_returns_aligned)
    msci_drawdown      = calculate_drawdown(msci_world_cum_returns_aligned)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=portfolio_drawdown.index, y=portfolio_drawdown,
        mode='lines', name='Dynamic Portfolio'
    ))
    fig.add_trace(go.Scatter(
        x=msci_drawdown.index, y=msci_drawdown,
        mode='lines', name='MSCI World (URTH)', line=dict(dash='dash')
    ))
    fig.update_layout(
        title='Rolling Drawdown',
        xaxis_title='Date',
        yaxis_title='Drawdown',
        yaxis_tickformat='%',
        hovermode='x unified',
        font=dict(size=14),
        width=1100,
        height=600
    )
    st.plotly_chart(fig, use_container_width=True)

    # Annual Percentage Returns
    st.markdown('---')
    st.markdown('## Annual Percentage Returns')

    portfolio_annual_returns = (1 + portfolio_returns_aligned).resample('Y').prod() - 1
    msci_annual_returns      = (1 + msci_world_returns_aligned).resample('Y').prod() - 1
    stoxx_annual_returns     = (1 + stoxx600_returns_aligned).resample('Y').prod() - 1

    # Squeeze to ensure 1D
    if isinstance(portfolio_annual_returns, pd.DataFrame):
        portfolio_annual_returns = portfolio_annual_returns.iloc[:,0]
    if isinstance(msci_annual_returns, pd.DataFrame):
        msci_annual_returns = msci_annual_returns.iloc[:,0]
    if isinstance(stoxx_annual_returns, pd.DataFrame):
        stoxx_annual_returns = stoxx_annual_returns.iloc[:,0]

    annual_returns_df = pd.DataFrame({
        'Dynamic Portfolio': portfolio_annual_returns,
        'MSCI World (URTH)': msci_annual_returns,
        'STOXX 600 (XSX6.MI)': stoxx_annual_returns
    })
    annual_returns_df.index = annual_returns_df.index.year
    annual_returns_df.index.name = 'Year'

    melted_annual = annual_returns_df.reset_index().melt(id_vars='Year', var_name='Portfolio', value_name='Annual Return')
    fig_bar = px.bar(
        melted_annual,
        x='Year',
        y='Annual Return',
        color='Portfolio',
        barmode='group',
        text='Annual Return',
        labels={'Annual Return':'Annual Return (%)'},
        color_discrete_sequence=px.colors.qualitative.Set1
    )
    fig_bar.update_traces(texttemplate='%{text:.2%}', textposition='outside')
    fig_bar.update_layout(
        title='Annual Percentage Returns',
        yaxis_tickformat='%',
        xaxis=dict(type='category'),
        font=dict(size=14),
        width=1100,
        height=600
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown('---')

elif selected_section == 'Mean Portfolio Evolution':
    st.markdown('---')
    st.markdown('## Mean Portfolio Evolution')

    weights_monthly = weights_df.resample('M').first()
    mean_weights_monthly = pd.DataFrame(index=weights_monthly.index, columns=weights_monthly.columns)

    # Compute cumulative average
    for i in range(len(weights_monthly)):
        subset = weights_monthly.iloc[:i+1]
        mean_weights_monthly.iloc[i] = subset.mean()

    mean_weights_df = mean_weights_monthly.reindex(daily_returns.index, method='ffill').fillna(0)
    mean_portfolio_returns = (daily_returns * mean_weights_df.shift(1)).sum(axis=1)
    mean_portfolio_cum_returns = (1 + mean_portfolio_returns).cumprod()

    # Align
    mean_common_idx = mean_portfolio_cum_returns.index \
        .intersection(stoxx600_cum_returns.index) \
        .intersection(msci_world_cum_returns.index)
    mean_cum_returns_aligned  = mean_portfolio_cum_returns.loc[mean_common_idx]
    stoxx_cum_aligned         = stoxx600_cum_returns.loc[mean_common_idx]
    msci_cum_aligned          = msci_world_cum_returns.loc[mean_common_idx]

    st.markdown('### Mean Portfolio vs. STOXX 600 & MSCI World')
    fig_mean = go.Figure()
    fig_mean.add_trace(go.Scatter(
        x=mean_cum_returns_aligned.index,
        y=mean_cum_returns_aligned,
        mode='lines',
        name='Mean Portfolio'
    ))
    fig_mean.add_trace(go.Scatter(
        x=stoxx_cum_aligned.index,
        y=stoxx_cum_aligned,
        mode='lines',
        name='STOXX 600',
        line=dict(dash='dash')
    ))
    fig_mean.add_trace(go.Scatter(
        x=msci_cum_aligned.index,
        y=msci_cum_aligned,
        mode='lines',
        name='MSCI World',
        line=dict(dash='dot')
    ))
    fig_mean.update_layout(
        title='Mean Portfolio Performance',
        xaxis_title='Date',
        yaxis_title='Cumulative Returns',
        font=dict(size=14),
        width=1100,
        height=600
    )
    st.plotly_chart(fig_mean, use_container_width=True)

    st.markdown('### Mean Portfolio Weights Sample (Monthly)')
    st.dataframe(mean_weights_monthly.tail().style.background_gradient(cmap='Blues'), use_container_width=True)

    fig_allocation = make_subplots(rows=1, cols=1, subplot_titles=['Evolving Mean Portfolio Allocations'])
    for etf in mean_weights_monthly.columns[mean_weights_monthly.sum()>0]:
        fig_allocation.add_trace(go.Scatter(
            x=mean_weights_monthly.index,
            y=mean_weights_monthly[etf],
            stackgroup='one',
            mode='lines',
            name=etf
        ))
    fig_allocation.update_layout(height=600, hovermode='x unified', legend_title='ETF', font=dict(size=14))
    st.plotly_chart(fig_allocation, use_container_width=True)

    st.markdown('---')
    st.markdown('## Rolling Sharpe Ratio (Mean Portfolio)')

    mean_returns_aligned = mean_portfolio_returns.loc[mean_common_idx]
    msci_returns_aligned = msci_world_returns.loc[mean_common_idx]

    mean_rolling_sharpe = mean_returns_aligned.rolling(window_size).apply(
        lambda x: (x.mean()/x.std())*np.sqrt(252) if x.std()!=0 else np.nan
    )
    msci_rolling_sharpe_mean = msci_returns_aligned.rolling(window_size).apply(
        lambda x: (x.mean()/x.std())*np.sqrt(252) if x.std()!=0 else np.nan
    )

    fig_sharpe = go.Figure()
    fig_sharpe.add_trace(go.Scatter(
        x=mean_rolling_sharpe.index, y=mean_rolling_sharpe,
        mode='lines', name='Mean Portfolio'
    ))
    fig_sharpe.add_trace(go.Scatter(
        x=msci_rolling_sharpe_mean.index, y=msci_rolling_sharpe_mean,
        mode='lines', name='MSCI World', line=dict(dash='dash')
    ))
    fig_sharpe.update_layout(
        title='Rolling Sharpe Ratio (Mean Portfolio)',
        xaxis_title='Date',
        yaxis_title='Sharpe Ratio',
        hovermode='x unified',
        font=dict(size=14),
        width=1100,
        height=600
    )
    st.plotly_chart(fig_sharpe, use_container_width=True)

    st.markdown('---')
    st.markdown('## Rolling Drawdown (Mean Portfolio)')

    mean_drawdown = calculate_drawdown(mean_cum_returns_aligned)
    msci_drawdown_mean = calculate_drawdown(msci_cum_aligned)

    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        x=mean_drawdown.index, y=mean_drawdown,
        mode='lines', name='Mean Portfolio'
    ))
    fig_dd.add_trace(go.Scatter(
        x=msci_drawdown_mean.index, y=msci_drawdown_mean,
        mode='lines', name='MSCI World', line=dict(dash='dash')
    ))
    fig_dd.update_layout(
        title='Rolling Drawdown (Mean Portfolio)',
        xaxis_title='Date',
        yaxis_title='Drawdown',
        yaxis_tickformat='%',
        hovermode='x unified',
        font=dict(size=14),
        width=1100,
        height=600
    )
    st.plotly_chart(fig_dd, use_container_width=True)

    st.markdown('---')
    st.markdown('## Annual Percentage Returns (Mean Portfolio)')

    mean_annual_ret = (1 + mean_returns_aligned).resample('Y').prod() - 1
    stoxx_annual_ret = (1 + stoxx600_returns.loc[mean_common_idx]).resample('Y').prod() - 1
    msci_annual_ret  = (1 + msci_world_returns.loc[mean_common_idx]).resample('Y').prod() - 1

    if isinstance(mean_annual_ret, pd.DataFrame):
        mean_annual_ret = mean_annual_ret.iloc[:,0]
    if isinstance(stoxx_annual_ret, pd.DataFrame):
        stoxx_annual_ret = stoxx_annual_ret.iloc[:,0]
    if isinstance(msci_annual_ret, pd.DataFrame):
        msci_annual_ret = msci_annual_ret.iloc[:,0]

    df_annual_mean = pd.DataFrame({
        'Mean Portfolio': mean_annual_ret,
        'STOXX 600': stoxx_annual_ret,
        'MSCI World': msci_annual_ret
    })
    df_annual_mean.index = df_annual_mean.index.year
    df_annual_mean.index.name='Year'

    melt_mean = df_annual_mean.reset_index().melt(id_vars='Year', var_name='Portfolio', value_name='Annual Return')
    fig_bar2 = px.bar(
        melt_mean,
        x='Year',
        y='Annual Return',
        color='Portfolio',
        barmode='group',
        text='Annual Return'
    )
    fig_bar2.update_traces(texttemplate='%{text:.2%}', textposition='outside')
    fig_bar2.update_layout(
        title='Annual Percentage Returns (Mean Portfolio)',
        yaxis_tickformat='%',
        xaxis=dict(type='category'),
        font=dict(size=14),
        width=1100,
        height=600
    )
    st.plotly_chart(fig_bar2, use_container_width=True)

    st.markdown('---')

