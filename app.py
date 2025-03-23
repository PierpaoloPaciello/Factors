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
    page_title='Dynamic Portfolio Strategy',
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
st.title('Factor Investing: Adapting to Economic Phases')

st.markdown('''
## Introduction

This app presents the **Dynamic Portfolio Strategy**. It uses the **OECD CLI Diffusion Index** to classify economic phases and then maps factor‐based ETFs to those phases.

**Key points:**
- **Economic phases:** Recovery, Expansion, Slowdown, and Contraction.
- **ETF Selection:**  
  - *Before 2023*: For demonstration, the top 3 ETFs are selected using the full available data (look‐ahead bias).  
  - *From 2023 onward*: Only historical data up to the rebalancing date is used (no look‐ahead bias).
- **Benchmarks:** The strategy is compared with the MSCI World ETF (URTH) and SPY.
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

# --------------------------
# 1. Download and Prepare Data
# --------------------------

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

countries = [
    'AUS', 'CAN', 'FRA', 'DEU', 'ITA', 'JPN', 'DNK', 'KOR', 'MEX', 'ESP',
    'TUR', 'GBR', 'USA', 'BRA', 'CHN', 'IND', 'IDN', 'ZAF'
]

with st.spinner('Fetching OECD CLI data...'):
    cli_data = get_oecd_data(countries)

# Reshape CLI data
pivot_data = cli_data.pivot(index='TIME_PERIOD', columns='REF_AREA', values='OBS_VALUE')
pivot_data.columns.name = None
pivot_data.fillna(method='ffill', inplace=True)
pivot_data.index = pd.to_datetime(pivot_data.index).tz_localize(None)
pivot_data.index = pivot_data.index + pd.DateOffset(months=1)

# Calculate Diffusion Index (DI)
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

# --------------------------
# 2. ETFs, Returns, and Factor Mapping
# --------------------------
tickers = [
    'QUAL', 'USMV', 'DYNF', 'MTUM', 'VLUE',
    'LRGF', 'SMLF', 'SIZE', 'SMMV', 'IGS', 'FOVL',
    'HDV', 'DGRO', 'DVY', 'IJH',
]

etf_labels = {
    'QUAL': 'Quality',
    'USMV': 'Low Volatility',
    'DYNF': 'Multifactor',
    'MTUM': 'Momentum',
    'VLUE': 'Value',
    'LRGF': 'Multifactor',
    'SMLF': 'Size',
    'SIZE': 'Size',
    'SMMV': 'Low Volatility',
    'IJS': 'Size',
    'FVOL': 'Size',
    'HDV': 'High Dividend',
    'DHRO': 'High Dividend',
    'DVY': 'High Dividend',
    'IJH': 'Size',
}

labels_df = pd.DataFrame(list(etf_labels.items()), columns=['Ticker', 'Label'])
labels_df['Labels'] = labels_df['Label'].apply(lambda x: [label.strip() for label in x.split(',')])

# Set ETF data start date (we keep a longer history so we can show the biased period pre-2023)
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

daily_returns = data.pct_change().fillna(0)
daily_returns.index = daily_returns.index.tz_localize(None)
pivot_data.index = pivot_data.index.tz_localize(None)

required_columns = ['DI', 'DI_change', 'DI_direction', 'Phase']
pivot_data_daily = pivot_data[required_columns].reindex(daily_returns.index, method='ffill').fillna(method='bfill')
daily_phases = pivot_data_daily['Phase']

# Map ETFs to factors
factor_etf_mapping = {}
for idx, row in labels_df.iterrows():
    for label in row['Labels']:
        factor_etf_mapping.setdefault(label.strip(), []).append(row['Ticker'])

# --------------------------
# 3. Precompute Look-Ahead Biased Selections & Factor Performance
# (These are used for periods before 2023 and for display purposes)
# --------------------------
unique_phases = daily_phases.dropna().unique()

# (A) Best ETFs per phase using the full sample (look-ahead bias)
best_etfs_per_phase = {}
for phase in unique_phases:
    phase_dates = daily_phases[daily_phases == phase].index.intersection(daily_returns.index)
    if phase_dates.empty:
        continue
    phase_returns = daily_returns.loc[phase_dates]
    cum_returns = (1 + phase_returns).cumprod().iloc[-1] - 1
    sorted_etfs = cum_returns.sort_values(ascending=False)
    top_etfs = sorted_etfs.head(3)
    best_etfs_per_phase[phase] = top_etfs.index.tolist()

# (B) Factor Performance (look-ahead biased, full sample)
factor_performance = pd.DataFrame()
for phase in unique_phases:
    phase_dates = daily_phases[daily_phases == phase].index.intersection(daily_returns.index)
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

# --------------------------
# 4. Portfolio Weights Calculation with Conditional Look-Ahead Bias
# --------------------------
# Identify rebalancing dates as the 14th day of each month
rebalancing_dates = daily_returns.index[daily_returns.index.day == 14]
weights_df = pd.DataFrame(index=daily_returns.index, columns=daily_returns.columns).fillna(0)

for i, rebal_date in enumerate(rebalancing_dates):
    # Determine the signal date (first trading day of the month)
    current_month_dates = daily_returns.index[(daily_returns.index.year == rebal_date.year) &
                                               (daily_returns.index.month == rebal_date.month)]
    month_start = current_month_dates.min()
    phase = daily_phases.loc[month_start]
    
    # For dates before 2023 use the precomputed (biased) selection,
    # and for 2023 onward recalculate using only historical data.
    if rebal_date < pd.Timestamp('2023-01-01'):
        etfs = best_etfs_per_phase.get(phase, [])
    else:
        historical_dates = daily_returns.index[daily_returns.index <= rebal_date]
        phase_historical_dates = daily_phases.loc[historical_dates][daily_phases.loc[historical_dates] == phase].index
        if phase_historical_dates.empty:
            continue
        phase_returns = daily_returns.loc[phase_historical_dates]
        cum_returns = (1 + phase_returns).cumprod().iloc[-1] - 1
        sorted_etfs = cum_returns.sort_values(ascending=False)
        etfs = sorted_etfs.head(3).index.tolist()
    
    if not etfs:
        continue

    weights = np.repeat(1/len(etfs), len(etfs))
    
    if i < len(rebalancing_dates) - 1:
        next_rebal_date = rebalancing_dates[i + 1]
    else:
        next_rebal_date = weights_df.index[-1] + pd.Timedelta(days=1)
    
    weights_df.loc[rebal_date:next_rebal_date, etfs] = weights

weights_df = weights_df.ffill()

# Prepare portfolio allocations over time
latest_date = weights_df.index.max()
current_weights = weights_df.loc[latest_date]
current_weights = current_weights[current_weights > 0]
etfs_in_portfolio = weights_df.columns[(weights_df != 0).any()].tolist()
weights_over_time = weights_df[etfs_in_portfolio]
mean_weights = weights_over_time.mean()
mean_weights = mean_weights[mean_weights > 0]

# --------------------------
# 5. Benchmark Data and Portfolio Returns
# --------------------------
portfolio_returns = (daily_returns * weights_df.shift(1)).sum(axis=1)
portfolio_cum_returns = (1 + portfolio_returns).cumprod()

msci_world = yf.download('URTH', start=portfolio_returns.index.min(), end=end_date)['Close']
msci_world = msci_world.fillna(method='ffill').dropna()
msci_world.index = msci_world.index.tz_localize(None)
msci_world_returns = msci_world.pct_change().fillna(0)
msci_world_cum_returns = (1 + msci_world_returns).cumprod()

spy = yf.download('SPY', start=portfolio_returns.index.min(), end=end_date)['Close']
spy = spy.fillna(method='ffill').dropna()
spy.index = spy.index.tz_localize(None)
spy_returns = spy.pct_change().fillna(0)
spy_cum_returns = (1 + spy_returns).cumprod()

window_size = 252  # one year of trading days

def calculate_drawdown(cumulative_returns):
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    return drawdown

# --------------------------
# 6. Display Sections and Graphs
# --------------------------
if selected_section == 'Methodology':
    st.markdown('---')
    st.markdown('''
    ## Methodology

    - **Economic Phase Identification:**  
      The Diffusion Index (DI) is computed monthly and used to assign one of four phases:
      Recovery, Expansion, Slowdown, or Contraction.
      
    - **ETF Selection & Portfolio Construction:**  
      *Before 2023* the top ETFs for each phase are selected using the full future data (introducing look‐ahead bias).  
      *From 2023 onward* only historical data up to the rebalancing date is used (no look‐ahead bias).

    - **Performance Benchmarking:**  
      The dynamic portfolio is compared against the MSCI World ETF (URTH) and SPY ETF.
    ''')

    st.markdown('### Economic Phases Data')
    st.dataframe(pivot_data[['DI', 'DI_change', 'Phase']].tail(15).style.background_gradient(cmap='Blues'), use_container_width=True)

    st.markdown('### OECD CLI Diffusion Index')
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
        title='OECD CLI Diffusion Index',
        xaxis_title='Date',
        yaxis_title='DI',
        font=dict(size=14),
        hovermode='x unified',
        legend=dict(x=0.01, y=0.99)
    )
    st.plotly_chart(fig, use_container_width=True)

elif selected_section == 'Portfolio Construction':
    st.markdown('---')
    st.markdown('''
    ## Portfolio Construction

    The portfolio rebalances on the 14th day of each month.  
    - *Before 2023*: The top 3 ETFs for the current phase are chosen using the full available data (look‐ahead bias).  
    - *From 2023 onward*: Only historical data up to the rebalancing date is used (no look‐ahead bias).
    ''')

    st.markdown('### Factor Performance by Phase (Look-Ahead Biased)')
    # Plot factor performance heatmap
    factor_performance_t = factor_performance.T
    plt.figure(figsize=(12, 8))
    sns.heatmap(factor_performance_t * 100, annot=True, fmt=".2f", cmap='Blues', cbar_kws={'label': 'Return (%)'})
    plt.xlabel('Factor')
    plt.ylabel('Economic Phase')
    st.pyplot(plt.gcf())
    
    st.markdown('### Top ETFs During Each Phase (Look-Ahead Biased)')
    for phase in unique_phases:
        etf_list = best_etfs_per_phase.get(phase, [])
        st.markdown(f"**{phase} Phase:** " + (", ".join(etf_list) if etf_list else "No data"))
    
    st.markdown('---')
    st.markdown('### Last Portfolio Weights')
    weights_monthly = weights_df.resample('M').first()
    st.dataframe(weights_monthly.tail().style.background_gradient(cmap='Blues'), use_container_width=True)
    
    common_index = portfolio_cum_returns.index.intersection(msci_world_cum_returns.index).intersection(spy_cum_returns.index)
    portfolio_cum_returns_aligned = portfolio_cum_returns.loc[common_index]
    msci_world_cum_returns_aligned = msci_world_cum_returns.loc[common_index].squeeze()
    spy_cum_returns_aligned = spy_cum_returns.loc[common_index].squeeze()

    st.markdown('### Portfolio Performance vs. Benchmarks')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=portfolio_cum_returns_aligned.index, y=portfolio_cum_returns_aligned,
                             mode='lines', name='Dynamic Portfolio', line=dict(width=3)))
    fig.add_trace(go.Scatter(x=spy_cum_returns_aligned.index, y=spy_cum_returns_aligned,
                             mode='lines', name='SPY ETF', line=dict(dash='dash', width=2)))
    fig.add_trace(go.Scatter(x=msci_world_cum_returns_aligned.index, y=msci_world_cum_returns_aligned,
                             mode='lines', name='MSCI World ETF (URTH)', line=dict(dash='dot', width=2)))
    fig.update_layout(
        title='Portfolio Performance',
        xaxis_title='Date',
        yaxis_title='Cumulative Returns',
        width=1200,
        height=600,
        legend=dict(x=0.02, y=0.98),
        font=dict(size=14)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('---')
    st.markdown('### Portfolio Allocations')
    weights_monthly = weights_over_time.resample('M').first()
    if latest_date not in weights_monthly.index:
        latest_month_end = latest_date.replace(day=1) + MonthEnd(1)
        weights_monthly.loc[latest_month_end] = weights_over_time.loc[latest_date]
    
    fig = make_subplots(
        rows=3, cols=1,
        row_heights=[0.5, 0.25, 0.25],
        vertical_spacing=0.08,
        subplot_titles=(
            'Portfolio Allocations Over Time (Monthly)',
            f'Current Allocation as of {latest_date.strftime("%Y-%m-%d")}',
            'Mean Allocation Over Entire Period'
        ),
        specs=[
            [{"type": "xy"}],
            [{"type": "domain"}],
            [{"type": "domain"}]
        ]
    )
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
    fig.update_xaxes(
        row=1, col=1,
        tickformat='%Y-%m',
        tickangle=45,
        nticks=20
    )
    fig.add_trace(
        go.Pie(
            labels=current_weights.index,
            values=current_weights.values,
            name='Current Allocation',
            textinfo='percent+label',
            hole=0.4,
            sort=False
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Pie(
            labels=mean_weights.index,
            values=mean_weights.values,
            name='Mean Allocation',
            textinfo='percent+label',
            hole=0.4,
            sort=False
        ),
        row=3, col=1
    )
    fig.update_layout(
        height=1500,
        showlegend=True,
        hovermode='x unified',
        legend_title='ETF',
        font=dict(size=14)
    )
    fig.update_yaxes(title_text='Portfolio Weight', row=1, col=1)
    fig.update_xaxes(title_text='Date', row=1, col=1)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('---')
    st.markdown('### Rolling Sharpe Ratio Comparison')
    portfolio_rolling_sharpe = portfolio_returns.rolling(window=window_size).apply(
        lambda x: (x.mean() / x.std()) * np.sqrt(252)
    ).squeeze()
    msci_rolling_sharpe = msci_world_returns.loc[portfolio_returns.index].rolling(window=window_size).apply(
        lambda x: (x.mean() / x.std()) * np.sqrt(252)
    ).squeeze()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=portfolio_rolling_sharpe.index, y=portfolio_rolling_sharpe,
                             mode='lines', name='Dynamic Portfolio', line=dict(color='blue', width=2)))
    fig.add_trace(go.Scatter(x=msci_rolling_sharpe.index, y=msci_rolling_sharpe,
                             mode='lines', name='MSCI World ETF (URTH)', line=dict(color='red', dash='dash', width=2)))
    fig.update_layout(
        title='Rolling Sharpe Ratio',
        xaxis_title='Date',
        yaxis_title='Rolling Sharpe Ratio',
        hovermode='x unified',
        width=1200,
        height=600,
        font=dict(size=14)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('---')
    st.markdown('### Rolling Drawdown Comparison')
    portfolio_drawdown = calculate_drawdown(portfolio_cum_returns)
    msci_drawdown = calculate_drawdown(msci_world_cum_returns_aligned)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=portfolio_drawdown.index, y=portfolio_drawdown,
                             mode='lines', name='Dynamic Portfolio', line=dict(color='blue', width=2)))
    fig.add_trace(go.Scatter(x=msci_drawdown.index, y=msci_drawdown,
                             mode='lines', name='MSCI World ETF (URTH)', line=dict(color='red', dash='dash', width=2)))
    fig.update_layout(
        title='Rolling Drawdown',
        xaxis_title='Date',
        yaxis_title='Drawdown',
        yaxis_tickformat='%',
        hovermode='x unified',
        width=1200,
        height=600,
        font=dict(size=14)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('---')
    st.markdown('### Annual Percentage Returns')
    portfolio_annual_returns = (1 + portfolio_returns).resample('Y').prod() - 1
    msci_annual_returns = (1 + msci_world_returns).resample('Y').prod() - 1
    spy_annual_returns = (1 + spy_returns).resample('Y').prod() - 1
    portfolio_annual_returns = portfolio_annual_returns.squeeze()
    msci_annual_returns = msci_annual_returns.squeeze()
    spy_annual_returns = spy_annual_returns.squeeze()
    annual_returns = pd.DataFrame({
        'Dynamic Portfolio': portfolio_annual_returns,
        'MSCI World ETF': msci_annual_returns,
        'SPY ETF': spy_annual_returns
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
        width=1200,
        height=600,
        font=dict(size=14)
    )
    st.plotly_chart(fig, use_container_width=True)

elif selected_section == 'Mean Portfolio Evolution':
    st.markdown('---')
    st.markdown('''
    ## Mean Portfolio Evolution

    This section shows the performance of a **Mean Portfolio** that rebalances monthly using the cumulative average allocation
    of the dynamic portfolio up to that date.
    ''')
    weights_monthly = weights_df.resample('M').first()
    mean_weights_monthly = pd.DataFrame(index=weights_monthly.index, columns=weights_monthly.columns)
    for i in range(len(weights_monthly)):
        weights_up_to_month = weights_monthly.iloc[:i+1]
        cumulative_avg_weights = weights_up_to_month.mean()
        mean_weights_monthly.iloc[i] = cumulative_avg_weights
    mean_weights_df = mean_weights_monthly.reindex(daily_returns.index, method='ffill').fillna(0)
    mean_portfolio_returns = (daily_returns * mean_weights_df.shift(1)).sum(axis=1)
    mean_portfolio_cum_returns = (1 + mean_portfolio_returns).cumprod()
    
    common_index = mean_portfolio_cum_returns.index.intersection(msci_world_cum_returns.index).intersection(spy_cum_returns.index)
    mean_portfolio_cum_returns_aligned = mean_portfolio_cum_returns.loc[common_index].squeeze()
    spy_cum_returns_aligned = spy_cum_returns.loc[common_index].squeeze()
    msci_world_cum_returns_aligned = msci_world_cum_returns.loc[common_index].squeeze()
    
    st.markdown('### Mean Portfolio Performance vs. Benchmarks')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=mean_portfolio_cum_returns_aligned.index, 
                             y=mean_portfolio_cum_returns_aligned, 
                             mode='lines', name='Mean Portfolio', line=dict(width=3)))
    fig.add_trace(go.Scatter(x=spy_cum_returns_aligned.index, 
                             y=spy_cum_returns_aligned, 
                             mode='lines', name='SPY ETF', line=dict(dash='dash', width=2)))
    fig.add_trace(go.Scatter(x=msci_world_cum_returns_aligned.index, 
                             y=msci_world_cum_returns_aligned, 
                             mode='lines', name='MSCI World ETF (URTH)', line=dict(dash='dot', width=2)))
    fig.update_layout(
        title='Mean Portfolio Performance',
        xaxis_title='Date',
        yaxis_title='Cumulative Returns',
        width=1200,
        height=600,
        legend=dict(x=0.02, y=0.98),
        font=dict(size=14)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('### Mean Portfolio Weights Sample (Monthly)')
    st.dataframe(mean_weights_monthly.head().style.background_gradient(cmap='Blues'), use_container_width=True)
    
    fig = make_subplots(
        rows=1, cols=1,
        subplot_titles=('Evolving Mean Portfolio Allocations Over Time',),
        specs=[[{"type": "xy"}]]
    )
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
        height=600,
        showlegend=True,
        hovermode='x unified',
        legend_title='ETF',
        font=dict(size=14)
    )
    fig.update_yaxes(title_text='Mean Portfolio Weight', row=1, col=1)
    fig.update_xaxes(title_text='Date', row=1, col=1)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('---')
    st.markdown('### Rolling Sharpe Ratio Comparison')
    mean_portfolio_rolling_sharpe = mean_portfolio_returns.rolling(window=window_size).apply(
        lambda x: (x.mean() / x.std()) * np.sqrt(252)
    ).squeeze()
    msci_rolling_sharpe_aligned = msci_world_returns.loc[mean_portfolio_returns.index].rolling(window=window_size).apply(
        lambda x: (x.mean() / x.std()) * np.sqrt(252)
    ).squeeze()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=mean_portfolio_rolling_sharpe.index, 
                             y=mean_portfolio_rolling_sharpe,
                             mode='lines', name='Mean Portfolio', line=dict(color='green', width=2)))
    fig.add_trace(go.Scatter(x=msci_rolling_sharpe_aligned.index, 
                             y=msci_rolling_sharpe_aligned,
                             mode='lines', name='MSCI World ETF (URTH)', line=dict(color='red', dash='dash', width=2)))
    fig.update_layout(
        title='Rolling Sharpe Ratio Comparison',
        xaxis_title='Date',
        yaxis_title='Rolling Sharpe Ratio',
        hovermode='x unified',
        width=1200,
        height=600,
        font=dict(size=14)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('---')
    st.markdown('### Rolling Drawdown Comparison')
    mean_portfolio_drawdown = calculate_drawdown(mean_portfolio_cum_returns_aligned)
    msci_drawdown_aligned = calculate_drawdown(msci_world_cum_returns_aligned)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=mean_portfolio_drawdown.index, y=mean_portfolio_drawdown,
                             mode='lines', name='Mean Portfolio', line=dict(color='green', width=2)))
    fig.add_trace(go.Scatter(x=msci_drawdown_aligned.index, y=msci_drawdown_aligned,
                             mode='lines', name='MSCI World ETF (URTH)', line=dict(color='red', dash='dash', width=2)))
    fig.update_layout(
        title='Rolling Drawdown',
        xaxis_title='Date',
        yaxis_title='Drawdown',
        yaxis_tickformat='%',
        hovermode='x unified',
        width=1200,
        height=600,
        font=dict(size=14)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('---')
    st.markdown('### Annual Percentage Returns')
    mean_portfolio_annual_returns = (1 + mean_portfolio_returns).resample('Y').prod() - 1
    msci_annual_returns_aligned = (1 + msci_world_returns.loc[mean_portfolio_returns.index]).resample('Y').prod() - 1
    spy_annual_returns_aligned = (1 + spy_returns.loc[mean_portfolio_returns.index]).resample('Y').prod() - 1
    mean_portfolio_annual_returns = mean_portfolio_annual_returns.squeeze()
    msci_annual_returns_aligned = msci_annual_returns_aligned.squeeze()
    spy_annual_returns_aligned = spy_annual_returns_aligned.squeeze()
    annual_returns_mean_portfolio = pd.DataFrame({
        'Mean Portfolio': mean_portfolio_annual_returns,
        'MSCI World ETF': msci_annual_returns_aligned,
        'SPY ETF': spy_annual_returns_aligned
    })
    annual_returns_mean_portfolio.index = annual_returns_mean_portfolio.index.year
    annual_returns_mean_portfolio.index.name = 'Year'
    annual_returns_melted_mean_portfolio = annual_returns_mean_portfolio.reset_index().melt(id_vars='Year', var_name='Portfolio', value_name='Annual Return')
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
        title='Annual Percentage Returns',
        yaxis_tickformat='%',
        xaxis=dict(type='category'),
        width=1200,
        height=600,
        font=dict(size=14)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('---')
