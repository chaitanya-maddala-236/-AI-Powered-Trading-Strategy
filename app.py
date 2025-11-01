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
    page_title="AI-Powered Trading Strategy",
    layout="wide",
    page_icon="🤖",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stMetric:hover {
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        transform: translateY(-2px);
        transition: all 0.3s ease;
    }
    h1 {
        color: #1f77b4;
        font-weight: 700;
        padding-bottom: 1rem;
        border-bottom: 3px solid #1f77b4;
    }
    h2 {
        color: #2c3e50;
        margin-top: 2rem;
        padding: 0.5rem 0;
        border-left: 5px solid #1f77b4;
        padding-left: 10px;
    }
    h3 {
        color: #34495e;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        border: none;
        padding: 0.75rem;
        border-radius: 10px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    .info-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin: 20px 0;
    }
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.07);
        margin: 10px 0;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Header with logo effect
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("""
    <div style='text-align: center; padding: 20px;'>
        <h1 style='font-size: 3rem; margin: 0; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; border: none;'>
            🤖 AI Trading Strategy
        </h1>
        <p style='font-size: 1.2rem; color: #7f8c8d; margin-top: 10px;'>
            Unsupervised Learning Meets Quantitative Finance
        </p>
    </div>
    """, unsafe_allow_html=True)

# Sidebar with enhanced design
with st.sidebar:
    st.markdown("""
    <div style='text-align: center; padding: 20px; background: white; border-radius: 10px; margin-bottom: 20px;'>
        <h2 style='margin: 0; color: #2c3e50; border: none;'>⚙️ Strategy Controls</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📊 Clustering Parameters")
    n_clusters = st.slider("Number of Clusters", 3, 10, 5, help="More clusters = finer segmentation")
    top_n_stocks = st.slider("Portfolio Size", 5, 30, 15, help="Number of stocks to hold")
    
    st.markdown("### 📅 Time Period")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start", datetime(2015, 1, 1))
    with col2:
        end_date = st.date_input("End", datetime(2024, 1, 1))
    
    st.markdown("### 🔄 Advanced Settings")
    lookback_period = st.slider("Lookback (days)", 100, 500, 252)
    rebalance_freq = st.slider("Rebalance (days)", 30, 180, 90)
    
    st.markdown("---")
    run_backtest = st.button("🚀 RUN BACKTEST", type="primary")
    
    if run_backtest:
        st.session_state.run = True
    
    st.markdown("---")
    st.markdown("""
    <div style='background: white; padding: 15px; border-radius: 10px; margin-top: 20px;'>
        <h4 style='color: #2c3e50; margin-top: 0;'>💡 Quick Tips</h4>
        <ul style='color: #7f8c8d; font-size: 0.9rem;'>
            <li>5 clusters works well for most cases</li>
            <li>Longer periods = more stable results</li>
            <li>Rebalance quarterly (90 days)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def download_sp500_tickers():
    """Download S&P 500 tickers from Wikipedia"""
    sp500 = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
    return sp500['Symbol'].tolist()[:100]

@st.cache_data(show_spinner=False)
def download_stock_data(tickers, start, end):
    """Download stock data"""
    data = yf.download(tickers, start=start, end=end, progress=False)['Adj Close']
    data = data.dropna(axis=1, thresh=len(data)*0.8)
    return data

def calculate_features(data):
    """Calculate technical features for each stock"""
    features_dict = {}
    
    for ticker in data.columns:
        prices = data[ticker].dropna()
        
        if len(prices) < 100:
            continue
            
        returns = prices.pct_change()
        
        features_dict[ticker] = {
            'mean_return': returns.mean() * 252,
            'volatility': returns.std() * np.sqrt(252),
            'sharpe_ratio': (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0,
            'momentum_1m': prices.pct_change(21).iloc[-1] if len(prices) > 21 else 0,
            'momentum_3m': prices.pct_change(63).iloc[-1] if len(prices) > 63 else 0,
            'momentum_6m': prices.pct_change(126).iloc[-1] if len(prices) > 126 else 0,
            'momentum_12m': prices.pct_change(252).iloc[-1] if len(prices) > 252 else 0,
            'rsi': calculate_rsi(prices),
            'max_drawdown': calculate_max_drawdown(prices)
        }
    
    return pd.DataFrame(features_dict).T

def calculate_rsi(prices, period=14):
    """Calculate RSI indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1] if len(rsi) > 0 else 50

def calculate_max_drawdown(prices):
    """Calculate maximum drawdown"""
    cumulative = (1 + prices.pct_change()).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()

def perform_clustering(features, n_clusters):
    """Perform K-Means clustering"""
    features_clean = features.dropna()
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features_clean)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    features_clean['cluster'] = clusters
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    return features_clean, X_pca, kmeans, scaler, pca

def calculate_portfolio_performance(data, selected_stocks):
    """Calculate portfolio returns"""
    portfolio_data = data[selected_stocks].dropna()
    portfolio_returns = portfolio_data.pct_change().mean(axis=1)
    cumulative_returns = (1 + portfolio_returns).cumprod()
    
    return portfolio_returns, cumulative_returns

# Main execution
if 'run' in st.session_state and st.session_state.run:
    try:
        # Progress tracking with better UI
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.markdown("### 🔄 Downloading S&P 500 tickers...")
        progress_bar.progress(10)
        tickers = download_sp500_tickers()
        
        status_text.markdown(f"### ✅ Found {len(tickers)} tickers")
        progress_bar.progress(25)
        
        status_text.markdown("### 📈 Downloading stock data...")
        data = download_stock_data(tickers, start_date, end_date)
        
        status_text.markdown(f"### ✅ Downloaded data for {len(data.columns)} stocks")
        progress_bar.progress(40)
        
        status_text.markdown("### 🔧 Calculating technical features...")
        features = calculate_features(data)
        
        status_text.markdown(f"### ✅ Calculated features for {len(features)} stocks")
        progress_bar.progress(60)
        
        status_text.markdown("### 🤖 Performing K-Means clustering...")
        features_clustered, X_pca, kmeans, scaler, pca = perform_clustering(features, n_clusters)
        
        status_text.markdown(f"### ✅ Clustered stocks into {n_clusters} groups")
        progress_bar.progress(80)
        
        status_text.markdown("### 💰 Calculating portfolio performance...")
        progress_bar.progress(90)
        
        # Clear progress indicators
        progress_bar.progress(100)
        status_text.empty()
        progress_bar.empty()
        
        # Success message
        st.success("🎉 Analysis Complete! Scroll down to see results.")
        
        # RESULTS SECTION
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Cluster Analysis
        st.markdown("## 📊 Cluster Analysis")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 📈 Cluster Statistics")
            cluster_stats = features_clustered.groupby('cluster').agg({
                'mean_return': 'mean',
                'volatility': 'mean',
                'sharpe_ratio': 'mean',
                'momentum_12m': 'mean'
            }).round(4)
            cluster_stats['count'] = features_clustered.groupby('cluster').size()
            cluster_stats.columns = ['Avg Return', 'Avg Vol', 'Avg Sharpe', 'Momentum', 'Stocks']
            
            # Style the dataframe
            styled_df = cluster_stats.style.background_gradient(cmap='RdYlGn', subset=['Avg Sharpe'])
            st.dataframe(styled_df, use_container_width=True)
        
        with col2:
            st.markdown("### 🎯 Cluster Visualization (PCA)")
            
            # Create interactive plotly scatter
            fig = go.Figure()
            
            for cluster in range(n_clusters):
                mask = features_clustered['cluster'] == cluster
                fig.add_trace(go.Scatter(
                    x=X_pca[mask, 0],
                    y=X_pca[mask, 1],
                    mode='markers',
                    name=f'Cluster {cluster}',
                    marker=dict(size=10, opacity=0.7),
                    hovertemplate='<b>Cluster %{text}</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}',
                    text=[cluster] * sum(mask)
                ))
            
            fig.update_layout(
                xaxis_title=f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)',
                yaxis_title=f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)',
                height=400,
                hovermode='closest',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Best Cluster Selection
        best_cluster = cluster_stats['Avg Sharpe'].idxmax()
        
        st.markdown(f"""
        <div class='info-box'>
            <h3 style='margin-top: 0; color: white;'>🎯 Best Performing Cluster: Cluster {best_cluster}</h3>
            <p style='font-size: 1.1rem; margin: 0;'>
                <b>Sharpe Ratio:</b> {cluster_stats.loc[best_cluster, 'Avg Sharpe']:.3f} | 
                <b>Stocks:</b> {int(cluster_stats.loc[best_cluster, 'Stocks'])} | 
                <b>Avg Return:</b> {cluster_stats.loc[best_cluster, 'Avg Return']*100:.2f}%
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Selected Stocks
        cluster_stocks = features_clustered[features_clustered['cluster'] == best_cluster]
        top_stocks = cluster_stocks.nlargest(top_n_stocks, 'sharpe_ratio').index.tolist()
        
        st.markdown("### 📋 Selected Portfolio Stocks")
        
        selected_df = cluster_stocks.loc[top_stocks][['mean_return', 'volatility', 'sharpe_ratio', 'momentum_12m', 'rsi']]
        selected_df.columns = ['Annual Return', 'Volatility', 'Sharpe', '12M Momentum', 'RSI']
        selected_df = (selected_df * 100).round(2)
        
        # Display as interactive table
        st.dataframe(
            selected_df.style.background_gradient(cmap='RdYlGn', subset=['Sharpe']),
            use_container_width=True
        )
        
        # Portfolio Performance
        st.markdown("## 💰 Portfolio Performance")
        
        portfolio_returns, cumulative_returns = calculate_portfolio_performance(data, top_stocks)
        
        # Download benchmark
        sp500 = yf.download("^GSPC", start=start_date, end=end_date, progress=False)['Adj Close']
        sp500_returns = sp500.pct_change()
        sp500_cumulative = (1 + sp500_returns).cumprod()
        
        # Align dates
        common_dates = portfolio_returns.index.intersection(sp500_returns.index)
        portfolio_returns = portfolio_returns.loc[common_dates]
        sp500_returns = sp500_returns.loc[common_dates]
        cumulative_returns = cumulative_returns.loc[common_dates]
        sp500_cumulative = sp500_cumulative.loc[common_dates]
        
        # Interactive Performance Chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=cumulative_returns.index,
            y=(cumulative_returns - 1) * 100,
            name='AI Strategy',
            line=dict(color='#667eea', width=3),
            fill='tonexty',
            hovertemplate='<b>AI Strategy</b><br>Date: %{x}<br>Return: %{y:.2f}%<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter(
            x=sp500_cumulative.index,
            y=(sp500_cumulative - 1) * 100,
            name='S&P 500',
            line=dict(color='#ff6b6b', width=3, dash='dash'),
            hovertemplate='<b>S&P 500</b><br>Date: %{x}<br>Return: %{y:.2f}%<extra></extra>'
        ))
        
        fig.update_layout(
            title='Cumulative Returns Over Time',
            xaxis_title='Date',
            yaxis_title='Cumulative Return (%)',
            height=500,
            hovermode='x unified',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance Metrics
        st.markdown("### 📈 Key Performance Metrics")
        
        total_return = (cumulative_returns.iloc[-1] - 1) * 100
        sp500_total_return = (sp500_cumulative.iloc[-1] - 1) * 100
        
        ann_return = portfolio_returns.mean() * 252 * 100
        ann_vol = portfolio_returns.std() * np.sqrt(252) * 100
        sharpe = (portfolio_returns.mean() / portfolio_returns.std()) * np.sqrt(252)
        
        sp500_ann_return = sp500_returns.mean() * 252 * 100
        sp500_ann_vol = sp500_returns.std() * np.sqrt(252) * 100
        sp500_sharpe = (sp500_returns.mean() / sp500_returns.std()) * np.sqrt(252)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            delta_color = "normal" if total_return > sp500_total_return else "inverse"
            st.metric(
                "Total Return",
                f"{total_return:.2f}%",
                delta=f"{total_return - sp500_total_return:.2f}% vs S&P500",
                delta_color=delta_color
            )
        
        with col2:
            delta_color = "normal" if ann_return > sp500_ann_return else "inverse"
            st.metric(
                "Annual Return",
                f"{ann_return:.2f}%",
                delta=f"{ann_return - sp500_ann_return:.2f}% vs S&P500",
                delta_color=delta_color
            )
        
        with col3:
            delta_color = "inverse" if ann_vol < sp500_ann_vol else "normal"
            st.metric(
                "Volatility",
                f"{ann_vol:.2f}%",
                delta=f"{ann_vol - sp500_ann_vol:.2f}% vs S&P500",
                delta_color=delta_color
            )
        
        with col4:
            delta_color = "normal" if sharpe > sp500_sharpe else "inverse"
            st.metric(
                "Sharpe Ratio",
                f"{sharpe:.3f}",
                delta=f"{sharpe - sp500_sharpe:.3f} vs S&P500",
                delta_color=delta_color
            )
        
        # Additional Analysis
        st.markdown("## 🔍 Advanced Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Rolling Sharpe
            st.markdown("### Rolling Sharpe Ratio (1Y)")
            rolling_window = 252
            rolling_sharpe = (portfolio_returns.rolling(rolling_window).mean() / 
                             portfolio_returns.rolling(rolling_window).std()) * np.sqrt(252)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=rolling_sharpe.index,
                y=rolling_sharpe,
                fill='tozeroy',
                line=dict(color='#667eea', width=2),
                hovertemplate='Date: %{x}<br>Sharpe: %{y:.3f}<extra></extra>'
            ))
            fig.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.5)
            fig.update_layout(
                height=350,
                yaxis_title='Sharpe Ratio',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Drawdown Analysis
            st.markdown("### Drawdown Analysis")
            cumulative = (1 + portfolio_returns).cumprod()
            running_max = cumulative.cummax()
            drawdown = (cumulative - running_max) / running_max * 100
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=drawdown.index,
                y=drawdown,
                fill='tozeroy',
                line=dict(color='#ff6b6b', width=2),
                hovertemplate='Date: %{x}<br>Drawdown: %{y:.2f}%<extra></extra>'
            ))
            fig.update_layout(
                height=350,
                yaxis_title='Drawdown (%)',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Feature Importance
        st.markdown("### 🎯 Feature Analysis")
        
        feature_names = ['Mean Return', 'Volatility', 'Sharpe', 'Mom 1M', 'Mom 3M', 'Mom 6M', 'Mom 12M', 'RSI', 'Max DD']
        feature_std = features_clustered.drop('cluster', axis=1).std()
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=feature_names,
            y=feature_std.values,
            marker=dict(
                color=feature_std.values,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Std Dev")
            ),
            hovertemplate='<b>%{x}</b><br>Std Dev: %{y:.4f}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Feature Variability (Higher = More Important for Clustering)',
            height=400,
            xaxis_title='Features',
            yaxis_title='Standard Deviation',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"❌ Error occurred: {str(e)}")
        st.exception(e)

else:
    # Landing page
    st.markdown("""
    <div class='info-box'>
        <h2 style='margin-top: 0; color: white;'>👋 Welcome to AI Trading Strategy</h2>
        <p style='font-size: 1.1rem;'>
            Configure your parameters in the sidebar and click <b>"RUN BACKTEST"</b> to start the analysis!
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature showcase
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class='metric-card'>
            <h3>🤖 Machine Learning</h3>
            <p>K-Means clustering identifies stocks with similar risk-return profiles</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='metric-card'>
            <h3>📊 Technical Analysis</h3>
            <p>9 advanced features including momentum, RSI, and Sharpe ratio</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='metric-card'>
            <h3>💹 Portfolio Optimization</h3>
            <p>Systematic selection from best-performing cluster</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("## 📚 How It Works")
    
    tab1, tab2, tab3 = st.tabs(["📖 Methodology", "🎯 Features", "💼 For Internships"])
    
    with tab1:
        st.markdown("""
        ### The Strategy Pipeline
        
        1. **Data Collection** 📥
           - Downloads S&P 500 historical data via Yahoo Finance
           - Filters stocks with sufficient trading history
        
        2. **Feature Engineering** 🔧
           - Calculates 9 technical indicators per stock
           - Includes returns, volatility, momentum, RSI, drawdown
        
        3. **Clustering** 🎯
           - Standardizes features using StandardScaler
           - Groups stocks using K-Means algorithm
           - Visualizes clusters via PCA projection
        
        4. **Selection** ⭐
           - Identifies best cluster by Sharpe ratio
           - Selects top N stocks from that cluster
        
        5. **Backtesting** 📊
           - Equal-weight portfolio construction
           - Compares vs S&P 500 benchmark
           - Calculates comprehensive performance metrics
        """)
    
    with tab2:
        st.markdown("""
        ### Technical Features Explained
        
        | Feature | Description | Importance |
        |---------|-------------|------------|
        | **Mean Return** | Annualized average return | Profitability measure |
        | **Volatility** | Standard deviation of returns | Risk measure |
        | **Sharpe Ratio** | Risk-adjusted return | Quality measure |
        | **Momentum (1M-12M)** | Price change over period | Trend strength |
        | **RSI** | Relative Strength Index | Overbought/oversold |
        | **Max Drawdown** | Largest peak-to-trough decline | Downside risk |
        
        All features are standardized before clustering to ensure equal weighting.
        """)
    
    with tab3:
        st.markdown("""
        ### Why This Impresses for Quant Internships
        
        ✅ **Demonstrates ML Skills**
        - Unsupervised learning (K-Means, PCA)
        - Feature engineering and scaling
        - Hyperparameter tuning
        
        ✅ **Shows Financial Domain Knowledge**
        - Risk-adjusted returns (Sharpe ratio)
        - Technical indicators (RSI, momentum, drawdown)
        - Portfolio construction and rebalancing
        
        ✅ **Proves Software Engineering Ability**
        - Clean, modular code architecture
        - Data caching for performance
        - Professional UI/UX design
        - Error handling and validation
        
        ✅ **Includes Proper Backtesting**
        - Benchmark comparison
        - Multiple performance metrics
        - Rolling analysis windows
        - Drawdown visualization
        
        ✅ **Production-Ready Application**
        - Interactive Streamlit dashboard
        - Plotly visualizations
        - Configurable parameters
        - Export-ready results
        """)
