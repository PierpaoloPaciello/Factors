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
    page_title='Dynamic Portfolio Strategy - Europe',
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

This app presents the **Dynamic Portfolio Strategy (European Version)**, an approach designed to optimize returns by dynamically allocating assets based on economic phases.

Using the **OECD CLI Diffusion Index** as a macroeconomic indicator, this strategy identifies four distinct economic phases:
- **Recovery**: Early growth after a contraction.
- **Expansion**: Sustained growth phase.
- **Slowdown**: Decline from peak economic activity.
- **Contraction**: Widespread economic decline.

By aligning factor-based ETFs with these phases, the strategy seeks to:
1. Outperform benchmarks (we show a global example with MSCI World (URTH) and SPY).
2. Minimize drawdowns during adverse market conditions.

The analysis evaluates the strategy’s performance, highlighting its ability to leverage factors such as Momentum, Quality, Value, and Low/Minimum Volatility across economic cycles.
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

# ----------------- 1) FETCHING OECD DATA -----------------------

# Function to download CLI data from OECD
@st.cache_data
def get_oecd_data(countries):
    database = '@DF_CLI'
    frequency = 'M'
    indicator = 'LI..'
    unit_of_measure = 'AA...'
    start_period = '1990-01'
    
    # Join all country codes
    country_code = "+".join(countries)
    
    # Create the query URL
    query_text = f"{database}/{country_code}.{frequency}.{indicator}.{unit_of_measure}?startPeriod={start_period}&dimensionAtObservation=AllDimensions"
    url = f"https://sdmx.oecd.org/public/rest/data/OECD.SDD.STES,DSD_STES{query_text}"
    
    headers = { 
        'User-Agent': 'Mozilla/5.0', 
        'Accept': 'application/vnd.sdmx.data+csv; charset=utf-8' 
    }
    
    # Fetch the data
    download = requests.get(url=url, headers=headers)
    df = pd.read_csv(io.StringIO(download.text))
    
    return df

# List of countries for DI calculation (European + Some Global)
countries = [
    'AUS', 'CAN', 'FRA', 'DEU', 'ITA', 'JPN', 'DNK', 'KOR', 'MEX', 'ESP',
    'TUR', 'GBR', 'USA', 'BRA', 'CHN', 'IND', 'IDN', 'ZAF'
]

# Fetch the CLI data for all countries
with st.spinner('Fetching OECD CLI data...'):
    cli_data = get_oecd_data(countries)

# Reshape the data for easier processing
pivot_data = cli_data.pivot(index='TIME_PERIOD', columns='REF_AREA', values='OBS_VALUE')

# Remove the column name to avoid KeyError when selecting columns
pivot_data.columns.name = None

# Ensure missing values are handled (fill with previous values)
pivot_data.fillna(method='ffill', inplace=True)

# Convert TIME_PERIOD to datetime and ensure timezone-naive
pivot_data.index = pd.to_datetime(pivot_data.index).tz_localize(None)

# ----------------- SHIFT DATA BY 1 MONTH (OECD CLI Lag) -----------------------
pivot_data.index = pivot_data.index + pd.DateOffset(months=1)

# ----------------- 2) CALCULATE DI AND PHASES -----------------------
pivot_data_change = pivot_data.diff()
diffusion_index = (pivot_data_change > 0).sum(axis=1) / len(pivot_data.columns)
pivot_data['DI'] = diffusion_index
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
pivot_data = pivot_data.dropna(subset=['Phase'])

# ----------------- 3) PREPARE ETF DATA (Europe Tickers) -----------------------
tickers = [
    'IEVL.MI', 'IEQU.MI', 'IEMO.MI', 'IFSE.MI',
    'IESZ.MI', 'MVEU.MI', 'EHF1.DE', 'LEONIA.MI'
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

labels_df = pd.DataFrame(list(etf_labels.items()), columns=['Ticker', 'Label'])
labels_df['Labels'] = labels_df['Label'].apply(lambda x: [label.strip() for label in x.split(',')])

start_date = '2016-01-01'
end_date = datetime.datetime.today().strftime('%Y-%m-%d')

@st.cache_data
def get_etf_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)['Close']
    return data

with st.spinner('Fetching ETF data from Yahoo Finance...'):
    data = get_etf_data(tickers, start_date, end_date)

data = data.fillna(method='ffill').dropna()
data.index = data.index.tz_localize(None)

available_tickers = data.columns.tolist()
labels_df = labels_df[labels_df['Ticker'].isin(available_tickers)]

# ----------------- 4) DAILY RETURNS AND PHASE ALIGNMENT -----------------------
daily_returns = data.pct_change().fillna(0)
daily_returns.index = daily_returns.index.tz_localize(None)

# We'll keep only the columns we need for phases
required_columns = ['DI', 'DI_change', 'DI_direction', 'Phase']
pivot_data_daily = pivot_data[required_columns]

# Reindex CLI data to daily for merging with daily returns
pivot_data_daily = pivot_data_daily.reindex(daily_returns.index, method='ffill')
pivot_data_daily = pivot_data_daily.fillna(method='bfill')

daily_phases = pivot_data_daily['Phase']

# ----------------- 5) FACTOR -> ETF MAPPING -----------------------
factor_etf_mapping = {}
for _, row in labels_df.iterrows():
    for label in row['Labels']:
        factor_etf_mapping.setdefault(label.strip(), []).append(row['Ticker'])

# ----------------- 6) FACTOR PERFORMANCE PER PHASE -----------------------
factor_performance = pd.DataFrame()
unique_phases = daily_phases.dropna().unique()

for phase in unique_phases:
    # Get dates corresponding to the phase
    phase_dates = daily_phases[daily_phases == phase].index
    phase_dates = phase_dates.intersection(daily_returns.index)
    
    if phase_dates.empty:
        continue
    
    factor_cum_returns = {}
    for factor, etfs in factor_etf_mapping.items():
        etfs_in_data = [etf for etf in etfs if etf in daily_returns.columns]
        if not etfs_in_data:
            continue
        phase_returns = daily_returns.loc[phase_dates, etfs_in_data]
        cum_returns = (1 + phase_returns).cumprod()
        mean_cum_return = cum_returns.iloc[-1].mean() - 1
        factor_cum_returns[factor] = mean_cum_return
    
    factor_cum_returns_df = pd.DataFrame.from_dict(factor_cum_returns, orient='index', columns=[phase])
    factor_performance = pd.concat([factor_performance, factor_cum_returns_df], axis=1)

factor_performance.fillna(0, inplace=True)

# ----------------- 7) SELECT TOP ETFs PER PHASE -----------------------
best_etfs_per_phase = {}
for phase in unique_phases:
    phase_dates = daily_phases[daily_phases == phase].index
    phase_dates = phase_dates.intersection(daily_returns.index)
    if phase_dates.empty:
        continue
    
    phase_returns = daily_returns.loc[phase_dates]
    cum_returns = (1 + phase_returns).cumprod().iloc[-1] - 1
    sorted_etfs = cum_returns.sort_values(ascending=False)
    top_etfs = sorted_etfs.head(3)
    best_etfs_per_phase[phase] = top_etfs.index.tolist()

# ----------------- 8) DYNAMIC PORTFOLIO WEIGHTS -----------------------
weights_df = pd.DataFrame(index=daily_returns.index, columns=daily_returns.columns).fillna(0)

for date in daily_returns.index:
    phase = daily_phases.loc[date]
    etfs = best_etfs_per_phase.get(phase, [])
    
    if not etfs:
        continue
    
    # Equal weighting among selected ETFs
    weights = np.repeat(1/len(etfs), len(etfs))
    weights_df.loc[date, etfs] = weights

# Forward-fill the weights to handle any NaN values
weights_df.fillna(method='ffill', inplace=True)

# Prepare data for allocations over time
latest_date = weights_df.index.max()
current_weights = weights_df.loc[latest_date]
current_weights = current_weights[current_weights > 0]
etfs_in_portfolio = weights_df.columns[(weights_df != 0).any()].tolist()
weights_over_time = weights_df[etfs_in_portfolio]

# Calculate mean allocations over entire period
mean_weights = weights_over_time.mean()
mean_weights = mean_weights[mean_weights > 0]

# ----------------- 9) PORTFOLIO RETURNS AND BENCHMARKS -----------------------
portfolio_returns = (daily_returns * weights_df.shift(1)).sum(axis=1)
portfolio_cum_returns = (1 + portfolio_returns).cumprod()

# Benchmark 1: MSCI World (URTH)
msci_world = yf.download('URTH', start=portfolio_returns.index.min(), end=end_date)['Close']
msci_world = msci_world.fillna(method='ffill').dropna()
msci_world.index = msci_world.index.tz_localize(None)
msci_world_returns = msci_world.pct_change().fillna(0)
msci_world_cum_returns = (1 + msci_world_returns).cumprod()

# Benchmark 2: S&P 500 (SPY)
spy = yf.download('SPY', start=portfolio_returns.index.min(), end=end_date)['Close']
spy = spy.fillna(method='ffill').dropna()
spy.index = spy.index.tz_localize(None)
spy_returns = spy.pct_change().fillna(0)
spy_cum_returns = (1 + spy_returns).cumprod()

# Rolling window size (e.g. 252 trading days ~ 1 year)
window_size = 252

def calculate_drawdown(cumulative_returns):
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    return drawdown

# ----------------- 10) STREAMLIT APP SECTIONS -----------------------
if selected_section == 'Methodology':
    st.markdown('---')
    st.markdown('''
    ## Methodology

    1. **Economic Phase Identification**:
       - The Diffusion Index (DI) is calculated monthly (with a 1-month lag adjustment) and categorized into four phases:
         - **Recovery**: DI < 0.5 and rising.
         - **Expansion**: DI ≥ 0.5 and rising.
         - **Slowdown**: DI ≥ 0.5 and falling.
         - **Contraction**: DI < 0.5 and falling.

    2. **ETF Selection and Portfolio Construction**:
       - ETFs are mapped to factors, and top performers (based on cumulative return in each phase) are selected.
       - Portfolios are constructed using equal weights for the top 3 ETFs in that phase and rebalanced monthly (or daily in code, with forward-fill).

    3. **Performance Benchmarking**:
       - Portfolio performance is compared to MSCI World ETF (URTH) and SPY via cumulative returns, Sharpe Ratios, and drawdowns.
    ''')

    st.markdown('### Economic Phases Data Sample')
    st.dataframe(pivot_data[['DI', 'DI_change', 'Phase']].tail(15).style.background_gradient(cmap='Blues'), use_container_width=True)

    st.markdown('### OECD CLI Diffusion Index (Shifted 1 Month)')
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=pivot_data.index,
            y=pivot_data['DI'],
            mode='lines',
            name='Diffusion Index',
            line=dict(color='#1f4e79', width=2)
        )
    )
    fig.add_hline(
        y=0.5,
        line_dash='dash',
        line_color='red',
        annotation_text='Threshold (0.5)',
        annotation_position='top left'
    )
    fig.update_layout(
        title='OECD CLI Diffusion Index (Europe + Select Global)',
        xaxis_title='Date',
        yaxis_title='Diffusion Index',
        font=dict(size=14),
        hovermode='x unified', 
        legend=dict(x=0.01, y=0.99)
    )
    st.plotly_chart(fig, use_container_width=True)

elif selected_section == 'Portfolio Construction':
    st.markdown('---')
    st.markdown('''
    ## Factor Performance by Phase
    The table below shows the cumulative **factor** outperformance during each economic phase.
    ''')
    st.dataframe(factor_performance.style.background_gradient(cmap='Blues'), use_container_width=True)

    st.markdown('### Factor Performance Heatmap')
    factor_performance_t = factor_performance.T
    plt.figure(figsize=(10, 6))
    sns.heatmap(factor_performance_t * 100, annot=True, fmt=".2f", cmap='Blues', cbar_kws={'label': 'Return (%)'})
    plt.xlabel('Factor')
    plt.ylabel('Economic Phase')
    st.pyplot(plt.gcf())

    st.markdown('---')
    st.markdown('### Top 3 ETFs During Each Phase')
    for phase in unique_phases:
        st.markdown(f"**{phase}**:")
        top_etfs = best_etfs_per_phase.get(phase, [])
        for etf in top_etfs:
            st.write(f"- {etf}")

    st.markdown('---')
    st.markdown('## Dynamic Portfolio Construction')

    # Show monthly weights sample
    weights_monthly = weights_df.resample('M').first()
    st.markdown('### Last 5 Months of Portfolio Weights')
    st.dataframe(weights_monthly.tail().style.background_gradient(cmap='Blues'), use_container_width=True)

    # Align for plotting
    common_index = portfolio_cum_returns.index.intersection(msci_world_cum_returns.index).intersection(spy_cum_returns.index)
    portfolio_cum_returns_aligned = portfolio_cum_returns.loc[common_index]
    msci_world_cum_returns_aligned = msci_world_cum_returns.loc[common_index]
    spy_cum_returns_aligned = spy_cum_returns.loc[common_index]

    st.markdown('### Portfolio Performance vs. Benchmarks')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=portfolio_cum_returns_aligned.index, 
                             y=portfolio_cum_returns_aligned, 
                             mode='lines', 
                             name='Dynamic Portfolio', 
                             line=dict(width=3)))
    fig.add_trace(go.Scatter(x=spy_cum_returns_aligned.index, 
                             y=spy_cum_returns_aligned, 
                             mode='lines', 
                             name='SPY (US)', 
                             line=dict(dash='dash', width=2)))
    fig.add_trace(go.Scatter(x=msci_world_cum_returns_aligned.index, 
                             y=msci_world_cum_returns_aligned, 
                             mode='lines', 
                             name='MSCI World (URTH)', 
                             line=dict(dash='dot', width=2)))
    fig.update_layout(
        title='Cumulative Returns Comparison',
        xaxis_title='Date',
        yaxis_title='Cumulative Returns',
        font=dict(size=14),
        legend=dict(x=0.02, y=0.98),
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)

    # Portfolio allocations over time
    st.markdown('---')
    st.markdown('## Portfolio Allocations Over Time')

    # Resample to monthly again for stacked area
    weights_monthly = weights_over_time.resample('M').first()

    # Ensure the latest date is included if not in monthly index
    if latest_date not in weights_monthly.index:
        last_month_end = latest_date.replace(day=1) + MonthEnd(1)
        weights_monthly.loc[last_month_end] = weights_over_time.loc[latest_date]

    fig = make_subplots(
        rows=3, cols=1,
        row_heights=[0.5, 0.25, 0.25],
        vertical_spacing=0.08,
        subplot_titles=(
            'Portfolio Allocations (Monthly)',
            f'Current Portfolio Allocation ({latest_date.strftime("%Y-%m-%d")})',
            'Mean Portfolio Allocation (Entire Period)'
        ),
        specs=[
            [{"type": "xy"}],
            [{"type": "domain"}],
            [{"type": "domain"}]
        ]
    )

    # 1) Stacked area
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

    fig.update_xaxes(tickformat='%Y-%m', tickangle=45, nticks=20, row=1, col=1)
    fig.update_yaxes(title_text='Weight', row=1, col=1)

    # 2) Current portfolio (pie)
    fig.add_trace(
        go.Pie(
            labels=current_weights.index,
            values=current_weights.values,
            hole=0.4,
            sort=False
        ),
        row=2, col=1
    )

    # 3) Mean portfolio (pie)
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
    portfolio_rolling_sharpe = portfolio_returns.rolling(window=window_size).apply(
        lambda x: (x.mean() / x.std()) * np.sqrt(252)
    )
    msci_rolling_sharpe = msci_world_returns.reindex(portfolio_returns.index).rolling(window=window_size).apply(
        lambda x: (x.mean() / x.std()) * np.sqrt(252)
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=portfolio_rolling_sharpe.index, y=portfolio_rolling_sharpe,
        mode='lines', name='Dynamic Portfolio', line=dict(width=2)
    ))
    fig.add_trace(go.Scatter(
        x=msci_rolling_sharpe.index, y=msci_rolling_sharpe,
        mode='lines', name='MSCI World (URTH)', line=dict(dash='dash', width=2)
    ))
    fig.update_layout(
        title='Rolling Sharpe Ratio',
        xaxis_title='Date',
        yaxis_title='Sharpe Ratio',
        hovermode='x unified',
        font=dict(size=14)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Rolling Drawdown
    st.markdown('---')
    st.markdown('## Rolling Drawdown Comparison')
    portfolio_drawdown = calculate_drawdown(portfolio_cum_returns)
    msci_drawdown = calculate_drawdown(msci_world_cum_returns)

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
        font=dict(size=14)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Annual Returns
    st.markdown('---')
    st.markdown('## Annual Percentage Returns')

    portfolio_annual_returns = (1 + portfolio_returns).resample('Y').prod() - 1
    msci_annual_returns = (1 + msci_world_returns).resample('Y').prod() - 1
    spy_annual_returns = (1 + spy_returns).resample('Y').prod() - 1

    annual_returns = pd.DataFrame({
        'Dynamic Portfolio': portfolio_annual_returns,
        'MSCI World': msci_annual_returns,
        'SPY': spy_annual_returns
    })
    annual_returns.index = annual_returns.index.year
    annual_returns.index.name = 'Year'

    annual_returns_melted = annual_returns.reset_index().melt(id_vars='Year', var_name='Portfolio', value_name='Annual Return')

    fig = px.bar(
        annual_returns_melted,
        x='Year',
        y='Annual Return',
        color='Portfolio',
        barmode='group',
        text='Annual Return',
        labels={'Annual Return': 'Annual Return (%)'},
        color_discrete_sequence=px.colors.qualitative.Set1
    )
    fig.update_traces(texttemplate='%{text:.2%}', textposition='outside')
    fig.update_layout(
        title='Annual Percentage Returns',
        yaxis_tickformat='%',
        xaxis=dict(type='category'),
        font=dict(size=14)
    )
    st.plotly_chart(fig, use_container_width=True)

elif selected_section == 'Mean Portfolio Evolution':
    st.markdown('---')
    st.markdown('''
    ## Mean Portfolio Evolution

    This section explores a **Mean Portfolio** approach that averages all historical allocations up to each point in time, 
    then rebalances monthly with these average weights.
    ''')

    # 1) Calculate monthly weights
    weights_monthly = weights_df.resample('M').first()

    # 2) Compute cumulative average weights each month
    mean_weights_monthly = pd.DataFrame(index=weights_monthly.index, columns=weights_monthly.columns)
    for i in range(len(weights_monthly)):
        mean_weights_monthly.iloc[i] = weights_monthly.iloc[:i+1].mean()

    # 3) Expand back to daily
    mean_weights_df = mean_weights_monthly.reindex(daily_returns.index, method='ffill').fillna(0)

    # 4) Mean Portfolio returns
    mean_portfolio_returns = (daily_returns * mean_weights_df.shift(1)).sum(axis=1)
    mean_portfolio_cum_returns = (1 + mean_portfolio_returns).cumprod()

    # Align with benchmarks
    common_index = mean_portfolio_cum_returns.index.intersection(msci_world_cum_returns.index).intersection(spy_cum_returns.index)
    mean_portfolio_cum_returns_aligned = mean_portfolio_cum_returns.loc[common_index]
    msci_world_cum_returns_aligned = msci_world_cum_returns.loc[common_index]
    spy_cum_returns_aligned = spy_cum_returns.loc[common_index]

    st.markdown('### Mean Portfolio Performance vs. Benchmarks')
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=mean_portfolio_cum_returns_aligned.index, y=mean_portfolio_cum_returns_aligned,
        mode='lines', name='Mean Portfolio', line=dict(width=3)
    ))
    fig.add_trace(go.Scatter(
        x=spy_cum_returns_aligned.index, y=spy_cum_returns_aligned,
        mode='lines', name='SPY', line=dict(dash='dash', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=msci_world_cum_returns_aligned.index, y=msci_world_cum_returns_aligned,
        mode='lines', name='MSCI World (URTH)', line=dict(dash='dot', width=2)
    ))
    fig.update_layout(
        title='Mean Portfolio Cumulative Returns',
        xaxis_title='Date',
        yaxis_title='Cumulative Return',
        font=dict(size=14)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('### Sample of Mean Weights (Monthly)')
    st.dataframe(mean_weights_monthly.head().style.background_gradient(cmap='Blues'), use_container_width=True)

    st.markdown('### Evolving Mean Portfolio Allocation Over Time')
    fig = make_subplots(rows=1, cols=1)
    for etf in mean_weights_monthly.columns[mean_weights_monthly.sum() > 0]:
        fig.add_trace(
            go.Scatter(
                x=mean_weights_monthly.index,
                y=mean_weights_monthly[etf],
                mode='lines',
                stackgroup='one',
                name=etf
            ),
            row=1, col=1
        )
    fig.update_layout(
        title='Mean Portfolio Allocation (Monthly)',
        hovermode='x unified',
        yaxis_title='Weight',
        font=dict(size=14)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('---')
    st.markdown('## Rolling Sharpe & Drawdown for Mean Portfolio')

    # Rolling Sharpe
    mean_portfolio_rolling_sharpe = mean_portfolio_returns.rolling(window=window_size).apply(
        lambda x: (x.mean() / x.std()) * np.sqrt(252)
    )
    msci_rolling_sharpe_aligned = msci_world_returns.loc[mean_portfolio_returns.index].rolling(window=window_size).apply(
        lambda x: (x.mean() / x.std()) * np.sqrt(252)
    )

    st.markdown('### Rolling Sharpe Ratio')
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=mean_portfolio_rolling_sharpe.index, y=mean_portfolio_rolling_sharpe,
        mode='lines', name='Mean Portfolio', line=dict(color='green', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=msci_rolling_sharpe_aligned.index, y=msci_rolling_sharpe_aligned,
        mode='lines', name='MSCI World (URTH)', line=dict(color='red', dash='dash', width=2)
    ))
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Sharpe Ratio',
        hovermode='x unified',
        font=dict(size=14)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('### Rolling Drawdown')
    mean_portfolio_drawdown = calculate_drawdown(mean_portfolio_cum_returns_aligned)
    msci_drawdown_aligned = calculate_drawdown(msci_world_cum_returns_aligned)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=mean_portfolio_drawdown.index, y=mean_portfolio_drawdown,
        mode='lines', name='Mean Portfolio', line=dict(color='green', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=msci_drawdown_aligned.index, y=msci_drawdown_aligned,
        mode='lines', name='MSCI World (URTH)', line=dict(color='red', dash='dash', width=2)
    ))
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Drawdown',
        yaxis_tickformat='%',
        hovermode='x unified',
        font=dict(size=14)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('### Annual Percentage Returns')
    mean_portfolio_annual_returns = (1 + mean_portfolio_returns).resample('Y').prod() - 1
    msci_annual_returns_aligned = (1 + msci_world_returns.loc[mean_portfolio_returns.index]).resample('Y').prod() - 1
    spy_annual_returns_aligned = (1 + spy_returns.loc[mean_portfolio_returns.index]).resample('Y').prod() - 1

    annual_returns_mean_portfolio = pd.DataFrame({
        'Mean Portfolio': mean_portfolio_annual_returns,
        'MSCI World': msci_annual_returns_aligned,
        'SPY': spy_annual_returns_aligned
    })
    annual_returns_mean_portfolio.index = annual_returns_mean_portfolio.index.year
    annual_returns_mean_portfolio.index.name = 'Year'

    annual_returns_melted_mean_portfolio = annual_returns_mean_portfolio.reset_index().melt(
        id_vars='Year', var_name='Portfolio', value_name='Annual Return'
    )

    fig = px.bar(
        annual_returns_melted_mean_portfolio,
        x='Year',
        y='Annual Return',
        color='Portfolio',
        barmode='group',
        text='Annual Return',
        labels={'Annual Return': 'Annual Return (%)'},
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig.update_traces(texttemplate='%{text:.2%}', textposition='outside')
    fig.update_layout(
        yaxis_tickformat='%',
        xaxis=dict(type='category'),
        font=dict(size=14)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('---')
    st.markdown('## Conclusion')
    st.markdown('''
    In this **European** adaptation of the Dynamic Portfolio Strategy, we see how factor exposures can be aligned
    with macroeconomic cycles, as indicated by the OECD CLI (shifted for data lag). Both the **dynamic** and **mean** 
    portfolios show how factor-timing approaches can enhance returns or reduce drawdowns relative to broad benchmarks.
    ''')


