import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# Page configuration
st.set_page_config(
    page_title="AI Trading Strategy",
    layout="wide",
    page_icon="🤖",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .main .block-container {
        max-width: 1400px;
        padding: 2rem 5rem;
    }
    
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .main-content {
        background: white;
        border-radius: 30px;
        padding: 3rem;
        box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        margin: 2rem auto;
    }
    
    .hero {
        text-align: center;
        padding: 3rem 2rem 2rem 2rem;
        margin-bottom: 3rem;
    }
    
    .hero h1 {
        font-size: 4rem;
        font-weight: 900;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        line-height: 1.2;
    }
    
    .hero p {
        font-size: 1.4rem;
        color: #666;
        margin-top: 1rem;
        font-weight: 500;
    }
    
    .section-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: #667eea;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #667eea;
    }
    
    .stMetric {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3);
    }
    
    .stMetric label {
        color: rgba(255,255,255,0.95) !important;
        font-size: 0.95rem !important;
        font-weight: 600 !important;
    }
    
    .stMetric [data-testid="stMetricValue"] {
        color: white !important;
        font-size: 2.2rem !important;
        font-weight: 700 !important;
    }
    
    .stMetric [data-testid="stMetricDelta"] {
        color: rgba(255,255,255,0.9) !important;
    }
    
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 700;
        border: none;
        padding: 1.2rem 3rem;
        border-radius: 50px;
        font-size: 1.3rem;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
        margin-top: 1rem;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 35px rgba(102, 126, 234, 0.6);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: #f8f9fa;
        border-radius: 15px;
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 0.8rem 1.5rem;
        font-weight: 600;
        color: #333 !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
    }
    
    .info-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        margin: 2rem 0;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    
    .info-box h3 {
        margin-top: 0;
        color: white;
    }
    
    .feature-card {
        background: #f8f9fa;
        border: 2px solid #e9ecef;
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        height: 100%;
    }
    
    .feature-card:hover {
        border-color: #667eea;
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.2);
    }
    
    .feature-card h3 {
        color: #667eea;
        font-size: 1.3rem;
        margin-bottom: 1rem;
    }
    
    .feature-card p {
        color: #555;
        line-height: 1.6;
    }
    
    .stats-box {
        background: #f8f9fa;
        border-radius: 15px;
        padding: 1.5rem;
        border: 2px solid #e9ecef;
        margin: 1rem 0;
    }
    
    .stats-box h3 {
        color: #667eea;
        margin-top: 0;
    }
    
    .guide-section {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 5px solid #667eea;
    }
    
    .guide-section h4 {
        color: #667eea;
        margin-top: 0;
    }
    
    .guide-section p {
        color: #555;
        line-height: 1.8;
    }
    
    .output-summary {
        background: #f0f7ff;
        border: 3px solid #667eea;
        border-radius: 20px;
        padding: 2rem;
        margin: 2rem 0;
    }
    
    .output-summary h2 {
        color: #667eea;
        margin-top: 0;
    }
    
    .summary-metric {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-content">', unsafe_allow_html=True)

# Hero Section
st.markdown("""
<div class='hero'>
    <h1>🤖 AI Trading Strategy</h1>
    <p>Machine Learning Meets Quantitative Finance</p>
</div>
""", unsafe_allow_html=True)

# Predefined stock lists
POPULAR_STOCKS = {
    "🚀 Tech Giants": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "NFLX", "ADBE", "CRM"],
    "💼 Blue Chips": ["JPM", "JNJ", "PG", "WMT", "V", "UNH", "HD", "DIS", "BA", "CAT"],
    "📈 Growth Stocks": ["TSLA", "NVDA", "AMD", "SHOP", "SQ", "ROKU", "PLTR", "COIN", "SNOW", "NET"],
    "🏆 S&P 500 Top 50": [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "UNH", "JNJ",
        "V", "WMT", "JPM", "PG", "MA", "HD", "CVX", "MRK", "ABBV", "KO",
        "PEP", "COST", "AVGO", "LLY", "TMO", "MCD", "CSCO", "ACN", "ABT", "DHR",
        "NKE", "CRM", "TXN", "PM", "NEE", "ORCL", "WFC", "VZ", "BMY", "UPS",
        "MS", "RTX", "HON", "QCOM", "INTU", "LOW", "AMGN", "T", "IBM", "CAT"
    ],
    "💰 Financial Sector": ["JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW", "AXP", "USB"],
    "⚡ Energy Sector": ["XOM", "CVX", "COP", "SLB", "EOG", "PXD", "MPC", "VLO", "PSX", "OXY"],
    "🏥 Healthcare": ["JNJ", "UNH", "PFE", "ABBV", "TMO", "MRK", "ABT", "DHR", "BMY", "LLY"]
}

# Control Panel
st.markdown('<p class="section-title">⚙️ Configure Your Strategy</p>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📊 Basic Settings", "📁 Data Management"])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        stock_universe = st.selectbox(
            "📊 Stock Universe",
            list(POPULAR_STOCKS.keys()),
            index=0,
            help="Choose a predefined set of stocks"
        )
        
        n_clusters = st.slider(
            "🎯 Number of Clusters",
            min_value=3,
            max_value=8,
            value=4,
            help="More clusters = finer segmentation"
        )
        
        start_date = st.date_input(
            "📅 Start Date",
            datetime(2022, 1, 1),
            help="Beginning of backtest period"
        )
    
    with col2:
        top_n_stocks = st.slider(
            "💼 Portfolio Size",
            min_value=3,
            max_value=15,
            value=5,
            help="Number of stocks to hold"
        )
        
        end_date = st.date_input(
            "📅 End Date",
            datetime(2024, 1, 1),
            help="End of backtest period"
        )

with tab2:
    data_mode = st.radio(
        "📁 Data Source",
        ["Download Fresh Data"],
        help="Download stock data from Yahoo Finance"
    )
    
    st.info(f"📊 Will download {len(POPULAR_STOCKS[stock_universe])} stocks from Yahoo Finance")

# Big action button
st.markdown("<br>", unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    run_backtest = st.button("🚀 RUN BACKTEST NOW", use_container_width=True)

if run_backtest:
    st.session_state.run = True

# Functions
@st.cache_data(show_spinner=False, ttl=3600)
def download_stock_data(tickers, start, end):
    """Download stock data"""
    import time
    
    all_data = {}
    failed = []
    
    progress_text = st.empty()
    
    for i, ticker in enumerate(tickers):
        try:
            progress_text.text(f"⏳ Downloading {ticker} ({i+1}/{len(tickers)})...")
            ticker_data = yf.download(ticker, start=start, end=end, progress=False)
            
            if not ticker_data.empty and 'Adj Close' in ticker_data.columns:
                all_data[ticker] = ticker_data['Adj Close']
            
            time.sleep(0.05)
            
        except:
            failed.append(ticker)
    
    progress_text.empty()
    
    if failed:
        st.warning(f"⚠️ Failed: {', '.join(failed[:3])}")
    
    if all_data:
        data = pd.DataFrame(all_data)
        threshold = len(data) * 0.7
        data = data.dropna(axis=1, thresh=int(threshold))
        return data
    
    return None

def calculate_features(data):
    """Calculate technical features"""
    features_dict = {}
    
    for ticker in data.columns:
        try:
            prices = data[ticker].dropna()
            if len(prices) < 50:
                continue
                
            returns = prices.pct_change().dropna()
            
            if len(returns) == 0:
                continue
            
            mean_ret = returns.mean() * 252
            vol = returns.std() * np.sqrt(252)
            sharpe = (mean_ret / vol) if vol > 0 else 0
            
            features_dict[ticker] = {
                'mean_return': mean_ret,
                'volatility': vol,
                'sharpe_ratio': sharpe,
                'momentum_1m': prices.pct_change(21).iloc[-1] if len(prices) > 21 else 0,
                'momentum_3m': prices.pct_change(63).iloc[-1] if len(prices) > 63 else 0,
                'momentum_6m': prices.pct_change(126).iloc[-1] if len(prices) > 126 else 0,
                'momentum_12m': prices.pct_change(252).iloc[-1] if len(prices) > 252 else 0,
                'rsi': calculate_rsi(prices),
                'max_drawdown': calculate_max_drawdown(prices)
            }
        except:
            continue
    
    if not features_dict:
        return None
        
    return pd.DataFrame(features_dict).T

def calculate_rsi(prices, period=14):
    """Calculate RSI"""
    try:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if len(rsi) > 0 and not np.isnan(rsi.iloc[-1]) else 50
    except:
        return 50

def calculate_max_drawdown(prices):
    """Calculate max drawdown"""
    try:
        cumulative = (1 + prices.pct_change()).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    except:
        return 0

def perform_clustering(features, n_clusters):
    """Perform K-Means clustering"""
    features_clean = features.dropna()
    
    if len(features_clean) < n_clusters:
        raise ValueError(f"Not enough stocks for {n_clusters} clusters")
    
    numeric_features = features_clean.select_dtypes(include=[np.number])
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(numeric_features)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    features_clean['cluster'] = clusters
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    return features_clean, X_pca, kmeans, scaler, pca

def calculate_portfolio_performance(data, selected_stocks):
    """Calculate portfolio returns"""
    portfolio_data = data[selected_stocks].dropna()
    
    if portfolio_data.empty:
        return None, None
        
    portfolio_returns = portfolio_data.pct_change().mean(axis=1)
    cumulative_returns = (1 + portfolio_returns).cumprod()
    
    return portfolio_returns, cumulative_returns

# Main execution
if 'run' in st.session_state and st.session_state.run:
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Download data
        tickers = POPULAR_STOCKS[stock_universe]
        status_text.markdown(f"### 📊 Downloading {len(tickers)} stocks...")
        progress_bar.progress(20)
        data = download_stock_data(tickers, start_date, end_date)
        
        if data is None or len(data) < 5:
            st.error("❌ Insufficient data. Try different dates.")
            st.stop()
        
        status_text.markdown(f"### ✅ Processing {len(data.columns)} stocks")
        progress_bar.progress(40)
        
        # Calculate features
        features = calculate_features(data)
        
        if features is None or len(features) < n_clusters:
            st.error(f"❌ Not enough stocks. Reduce clusters to {max(3, len(features)-1)}")
            st.stop()
            
        progress_bar.progress(60)
        
        # Clustering
        status_text.markdown("### 🤖 Running K-Means clustering...")
        features_clustered, X_pca, kmeans, scaler, pca = perform_clustering(features, n_clusters)
        progress_bar.progress(80)
        
        # Select best cluster
        cluster_stats = features_clustered.groupby('cluster').agg({
            'mean_return': 'mean',
            'volatility': 'mean',
            'sharpe_ratio': 'mean',
            'momentum_12m': 'mean'
        })
        cluster_stats['count'] = features_clustered.groupby('cluster').size()
        best_cluster = cluster_stats['sharpe_ratio'].idxmax()
        
        cluster_stocks = features_clustered[features_clustered['cluster'] == best_cluster]
        actual_top_n = min(top_n_stocks, len(cluster_stocks))
        top_stocks = cluster_stocks.nlargest(actual_top_n, 'sharpe_ratio').index.tolist()
        
        # Portfolio performance
        portfolio_returns, cumulative_returns = calculate_portfolio_performance(data, top_stocks)
        
        if portfolio_returns is None:
            st.error("❌ Could not calculate returns.")
            st.stop()
        
        # Benchmark
        status_text.markdown("### 📊 Fetching S&P 500...")
        sp500 = yf.download("^GSPC", start=start_date, end=end_date, progress=False)['Adj Close']
        sp500_returns = sp500.pct_change()
        sp500_cumulative = (1 + sp500_returns).cumprod()
        
        # Align dates
        common_dates = portfolio_returns.index.intersection(sp500_returns.index)
        portfolio_returns = portfolio_returns.loc[common_dates]
        sp500_returns = sp500_returns.loc[common_dates]
        cumulative_returns = cumulative_returns.loc[common_dates]
        sp500_cumulative = sp500_cumulative.loc[common_dates]
        
        progress_bar.progress(100)
        status_text.empty()
        progress_bar.empty()
        
        st.success("🎉 Analysis Complete!")
        
        # ============ OUTPUT SUMMARY ============
        st.markdown("""
        <div class="output-summary">
            <h2>📋 BACKTEST RESULTS SUMMARY</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Calculate metrics
        total_return = (cumulative_returns.iloc[-1] - 1) * 100
        sp500_total_return = (sp500_cumulative.iloc[-1] - 1) * 100
        ann_return = portfolio_returns.mean() * 252 * 100
        ann_vol = portfolio_returns.std() * np.sqrt(252) * 100
        sharpe = (portfolio_returns.mean() / portfolio_returns.std()) * np.sqrt(252)
        sp500_ann_return = sp500_returns.mean() * 252 * 100
        sp500_sharpe = (sp500_returns.mean() / sp500_returns.std()) * np.sqrt(252)
        max_dd = ((cumulative_returns / cumulative_returns.cummax()) - 1).min() * 100
        
        # Summary text
        st.markdown(f"""
        <div class="summary-metric">
            <strong>📊 Strategy Performance:</strong> Your AI-selected portfolio of {actual_top_n} stocks from Cluster {best_cluster} 
            achieved a total return of <strong style="color: {'green' if total_return > 0 else 'red'};">{total_return:.2f}%</strong> 
            compared to S&P 500's {sp500_total_return:.2f}% (outperformance: {total_return - sp500_total_return:.2f}%)
        </div>
        
        <div class="summary-metric">
            <strong>📈 Risk-Adjusted Returns:</strong> Sharpe Ratio of <strong style="color: #667eea;">{sharpe:.3f}</strong> 
            vs S&P 500's {sp500_sharpe:.3f}, indicating {'better' if sharpe > sp500_sharpe else 'lower'} risk-adjusted performance
        </div>
        
        <div class="summary-metric">
            <strong>⚠️ Risk Metrics:</strong> Maximum drawdown was <strong>{max_dd:.2f}%</strong> with 
            annualized volatility of {ann_vol:.2f}%
        </div>
        
        <div class="summary-metric">
            <strong>🤖 ML Insights:</strong> K-Means identified {n_clusters} distinct clusters. 
            The best performing cluster contained {len(cluster_stocks)} stocks with average Sharpe ratio of {cluster_stats.loc[best_cluster, 'sharpe_ratio']:.3f}
        </div>
        
        <div class="summary-metric">
            <strong>💼 Selected Portfolio:</strong> {', '.join(top_stocks)}
        </div>
        """, unsafe_allow_html=True)
        
        # PERFORMANCE METRICS
        st.markdown('<p class="section-title">📊 Performance Metrics</p>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Return", f"{total_return:.2f}%", 
                     delta=f"{total_return - sp500_total_return:.2f}% vs S&P")
        with col2:
            st.metric("Annual Return", f"{ann_return:.2f}%",
                     delta=f"{ann_return - sp500_ann_return:.2f}% vs S&P")
        with col3:
            st.metric("Sharpe Ratio", f"{sharpe:.3f}",
                     delta=f"{sharpe - sp500_sharpe:.3f} vs S&P")
        with col4:
            st.metric("Max Drawdown", f"{max_dd:.2f}%")
        
        # CUMULATIVE RETURNS CHART
        st.markdown('<p class="section-title">💹 Cumulative Returns Over Time</p>', unsafe_allow_html=True)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=cumulative_returns.index, y=(cumulative_returns - 1) * 100,
            name='AI Strategy', line=dict(color='#667eea', width=3),
            fill='tozeroy', fillcolor='rgba(102, 126, 234, 0.1)'
        ))
        fig.add_trace(go.Scatter(
            x=sp500_cumulative.index, y=(sp500_cumulative - 1) * 100,
            name='S&P 500', line=dict(color='#764ba2', width=3, dash='dash')
        ))
        fig.update_layout(
            height=500, 
            xaxis_title='Date', 
            yaxis_title='Cumulative Return (%)',
            hovermode='x unified', 
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis=dict(showgrid=True, gridcolor='#f0f0f0'),
            yaxis=dict(showgrid=True, gridcolor='#f0f0f0'),
            legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)')
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # CLUSTER ANALYSIS
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📈 All Clusters Performance")
            cluster_display = cluster_stats.copy()
            cluster_display['Return'] = (cluster_display['mean_return'] * 100).round(2)
            cluster_display['Vol'] = (cluster_display['volatility'] * 100).round(2)
            cluster_display['Sharpe'] = cluster_display['sharpe_ratio'].round(3)
            cluster_display['Mom'] = (cluster_display['momentum_12m'] * 100).round(2)
            cluster_display = cluster_display[['Return', 'Vol', 'Sharpe', 'Mom', 'count']]
            cluster_display.columns = ['Return %', 'Vol %', 'Sharpe', 'Momentum %', 'Stocks']
            
            st.dataframe(
                cluster_display.style.background_gradient(cmap='RdYlGn', subset=['Sharpe']).format(precision=2),
                use_container_width=True
            )
            st.caption(f"⭐ Best Cluster: #{best_cluster} with Sharpe Ratio of {cluster_stats.loc[best_cluster, 'sharpe_ratio']:.3f}")
        
        with col2:
            st.markdown("### 🎯 PCA Visualization")
            fig = go.Figure()
            colors = px.colors.qualitative.Set2[:n_clusters]
            
            for i, cluster in enumerate(range(n_clusters)):
                mask = features_clustered['cluster'] == cluster
                is_best = (cluster == best_cluster)
                fig.add_trace(go.Scatter(
                    x=X_pca[mask, 0], 
                    y=X_pca[mask, 1],
                    mode='markers', 
                    name=f'Cluster {cluster}' + (' ⭐' if is_best else ''),
                    marker=dict(
                        size=12 if is_best else 8, 
                        opacity=0.8,
                        color=colors[i],
                        line=dict(width=2, color='white') if is_best else dict(width=0)
                    )
                ))
            
            fig.update_layout(
                height=400, 
                plot_bgcolor='white', 
                paper_bgcolor='white',
                xaxis=dict(showgrid=True, gridcolor='#f0f0f0', title='Principal Component 1'),
                yaxis=dict(showgrid=True, gridcolor='#f0f0f0', title='Principal Component 2'),
                legend=dict(x=0.01, y=0.99)
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # SELECTED PORTFOLIO
        st.markdown(f'<p class="section-title">💼 Selected Portfolio Details (Cluster {best_cluster})</p>', unsafe_allow_html=True)
        
        selected_df = cluster_stocks.loc[top_stocks][['mean_return', 'volatility', 'sharpe_ratio', 'momentum_12m', 'rsi']].copy()
        selected_df['Return %'] = (selected_df['mean_return'] * 100).round(2)
        selected_df['Vol %'] = (selected_df['volatility'] * 100).round(2)
        selected_df['Sharpe'] = selected_df['sharpe_ratio'].round(3)
        selected_df['Momentum %'] = (selected_df['momentum_12m'] * 100).round(2)
        selected_df['RSI'] = selected_df['rsi'].round(1)
        selected_df = selected_df[['Return %', 'Vol %', 'Sharpe', 'Momentum %', 'RSI']]
        
        st.dataframe(
            selected_df.style.background_gradient(cmap='RdYlGn', subset=['Sharpe', 'Return %']).format(precision=2),
            use_container_width=True
        )
        
        # DOWNLOAD SECTION
        st.markdown('<p class="section-title">📥 Download Results</p>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv = selected_df.to_csv()
            st.download_button("📄 Portfolio CSV", csv, "portfolio.csv", "text/csv")
        
        with col2:
            results_csv = pd.DataFrame({
                'Date': cumulative_returns.index,
                'Strategy Return %': (cumulative_returns - 1) * 100,
                'SP500 Return %': (sp500_cumulative.loc[common_dates] - 1) * 100
            }).to_csv(index=False)
            st.download_button("📈 Returns CSV", results_csv, "returns.csv", "text/csv")
        
        with col3:
            data_csv = data.to_csv()
            st.download_button("💾 Price Data", data_csv, "price_data.csv", "text/csv")
        
        # FINAL SUMMARY
        st.markdown("""
        <div class="info-box">
            <h3>✅ Analysis Complete!</h3>
            <p><strong>What Just Happened:</strong></p>
            <ul style="text-align: left; line-height: 1.8;">
                <li>
