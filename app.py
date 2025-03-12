import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
import os
from openai import OpenAI
from typing import List, Dict
from advanced_analytics import AdvancedAnalytics
from enhanced_visualizations import EnhancedVisualizations
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure OpenAI
client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))

class DatasetChat:
    def __init__(self):
        self.model = "gpt-4o-mini"
        self.data_insights = {}
        self.analytics = None
        self.visualizations = None
    
    def analyze_dataset(self, df: pd.DataFrame):
        """Analyze the dataset and extract key insights."""
        insights = {}
        
        # Basic statistics
        insights["row_count"] = len(df)
        insights["column_count"] = len(df.columns)
        
        # Initialize analytics and visualization modules
        self.analytics = AdvancedAnalytics()
        self.visualizations = EnhancedVisualizations()
        
        # Check for specific columns in the dataset
        if 'Name' in df.columns and 'Global_Sales' in df.columns:
            # Video game dataset analysis
            
            # Top selling games
            top_games = df.sort_values(by="Global_Sales", ascending=False).head(5)
            insights["top_selling_games"] = []
            
            for _, row in top_games.iterrows():
                game_info = {
                    "name": row["Name"],
                    "global_sales": row["Global_Sales"],
                }
                
                # Add platform if available
                if "Platform" in row:
                    game_info["platform"] = row["Platform"]
                
                # Add year if available
                if "Year" in row:
                    game_info["year"] = row["Year"]
                
                insights["top_selling_games"].append(game_info)
            
            # Sales by platform
            if "Platform" in df.columns:
                platform_sales = df.groupby("Platform")["Global_Sales"].sum().sort_values(ascending=False).to_dict()
                insights["platform_sales"] = platform_sales
            
            # Sales by genre
            if "Genre" in df.columns:
                genre_sales = df.groupby("Genre")["Global_Sales"].sum().sort_values(ascending=False).to_dict()
                insights["genre_sales"] = genre_sales
            
            # Sales by publisher
            if "Publisher" in df.columns:
                publisher_sales = df.groupby("Publisher")["Global_Sales"].sum().sort_values(ascending=False).to_dict()
                insights["publisher_sales"] = publisher_sales
                
        # Add advanced analytics insights
        if len(df.select_dtypes(include=['int64', 'float64']).columns) > 0:
            # Get descriptive statistics for numeric columns
            insights["descriptive_stats"] = AdvancedAnalytics.get_descriptive_statistics(df)
            
            # Attempt to detect time series data
            time_cols = AdvancedAnalytics.detect_time_series(df)
            if time_cols and len(time_cols) > 0:
                insights["has_time_series"] = True
                insights["time_columns"] = time_cols
                
                # Try to analyze the first detected time series with first numeric column
                numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
                if len(numeric_cols) > 0:
                    try:
                        time_analysis = AdvancedAnalytics.analyze_time_series(df, time_cols[0], numeric_cols[0])
                        if "error" not in time_analysis:
                            insights["time_analysis"] = {
                                "column": numeric_cols[0],
                                "trend": time_analysis["trend_direction"],
                                "growth_rate": time_analysis["growth_rate"]
                            }
                    except Exception:
                        # Skip time series analysis if it fails
                        pass
        
        self.data_insights = insights
        return insights
    
    def get_chat_response(self, df: pd.DataFrame, user_message: str) -> str:
        """Generate a chat response based on the dataset and user's message."""
        # Analyze the dataset if not already analyzed
        if not self.data_insights:
            self.analyze_dataset(df)
            
        # Prepare a context message with dataset insights
        insights_text = "Dataset information:\n"
        
        if "row_count" in self.data_insights and "column_count" in self.data_insights:
            insights_text += f"- {self.data_insights['row_count']} rows and {self.data_insights['column_count']} columns\n"
        
        # Add info about top items if available
        if "top_selling_games" in self.data_insights and self.data_insights["top_selling_games"]:
            insights_text += "- Top items:\n"
            for game in self.data_insights["top_selling_games"][:3]:  # Only include top 3
                name = game["name"]
                sales = game["global_sales"]
                insights_text += f"  * {name}: {sales} million global sales\n"
            
        # Add platform info if available
        if "platform_sales" in self.data_insights:
            top_platforms = list(self.data_insights["platform_sales"].items())[:3]
            insights_text += "- Top platforms: " + ", ".join([f"{p} ({s:.2f}M)" for p, s in top_platforms]) + "\n"
            
        # Add genre info if available
        if "genre_sales" in self.data_insights:
            top_genres = list(self.data_insights["genre_sales"].items())[:3]
            insights_text += "- Top genres: " + ", ".join([f"{g} ({s:.2f}M)" for g, s in top_genres]) + "\n"
        
        # Add advanced analytics insights
        if "descriptive_stats" in self.data_insights:
            stats = self.data_insights["descriptive_stats"]
            numeric_cols = list(stats.keys())
            
            if numeric_cols:
                # Get a sample column to show stats for
                sample_col = numeric_cols[0]
                col_stats = stats[sample_col]
                
                insights_text += f"- Statistical insights for {sample_col}: "
                insights_text += f"Mean={col_stats['mean']:.2f}, "
                insights_text += f"Median={col_stats['median']:.2f}, "
                insights_text += f"Min={col_stats['min']:.2f}, "
                insights_text += f"Max={col_stats['max']:.2f}\n"
                
                # Add outlier info
                if "outlier_count" in col_stats and col_stats["outlier_count"] > 0:
                    insights_text += f"- Detected {col_stats['outlier_count']} outliers in {sample_col} "
                    insights_text += f"({col_stats['outlier_percentage']:.1f}% of data)\n"
        
        # Add time series insights if available
        if "time_analysis" in self.data_insights:
            time_info = self.data_insights["time_analysis"]
            insights_text += f"- Time series analysis for {time_info['column']}: "
            insights_text += f"{time_info['trend']} trend with {time_info['growth_rate']*100:.1f}% growth rate\n"
            
        # Prepare system message with information about dataset
        system_message = f"""
        You are an AI assistant named Dataset Chat. You help analyze and answer questions about datasets.
        Respond in a brief, informative way based on the dataset information provided.
        
        {insights_text}
        
        Keep answers concise and focus on the data. Mention if you don't have enough information to answer.
        """
        
        # Generate response
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature=0.1,
            max_tokens=450,
        )
        
        return response.choices[0].message.content
    
    def get_dataset_info(self, df: pd.DataFrame) -> str:
        """Generate initial dataset information"""
        info = []
        info.append(f"Number of rows: {len(df)}")
        info.append(f"Number of columns: {len(df.columns)}")
        info.append("\nColumns and sample data:")
        for col in df.columns:
            dtype = str(df[col].dtype)
            n_unique = df[col].nunique()
            sample = df[col].head(3).tolist()
            info.append(f"- {col} ({dtype}): {n_unique} unique values")
            info.append(f"  Sample values: {sample}")
        return "\n".join(info)
    
def main():
    st.set_page_config(
        page_title="Dataset Chat Assistant",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title('Dataset Chat Assistant')
    
    # Initialize session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'dataset_chat' not in st.session_state:
        st.session_state.dataset_chat = DatasetChat()
        
    if 'df' not in st.session_state:
        st.session_state.df = None
        
    # Create two columns for layout
    viz_col, chat_col = st.columns([2, 1])
    
    # Visualization interface in left column
    with viz_col:
        # File upload with emoji
        uploaded_file = st.file_uploader("📂 Upload Dataset (CSV/Excel)", type=['csv', 'xlsx'])
        
        if uploaded_file:
            try:
                if uploaded_file.name.endswith('.csv'):
                    st.session_state.df = pd.read_csv(uploaded_file)
                else:
                    st.session_state.df = pd.read_excel(uploaded_file)
                
                # Create tabs for different views
                tabs = st.tabs(["📊 Data Preview", "🔍 Insights", "📈 Advanced Visualizations"])
                
                # Data Preview Tab
                with tabs[0]:
                    st.dataframe(st.session_state.df.head(), use_container_width=True)
                
                # Insights Tab - Show automatic insights
                with tabs[1]:
                    if st.session_state.df is not None:
                        with st.spinner("Generating insights..."):
                            # Make sure dataset_chat has analyzed the dataset
                            if not hasattr(st.session_state.dataset_chat, 'data_insights') or not st.session_state.dataset_chat.data_insights:
                                st.session_state.dataset_chat.analyze_dataset(st.session_state.df)
                            
                            insights = st.session_state.dataset_chat.data_insights
                            
                            # Basic dataset info
                            st.subheader("📋 Dataset Summary")
                            st.write(f"📊 Rows: {insights.get('row_count', 'N/A')} | Columns: {insights.get('column_count', 'N/A')}")
                            
                            # Show top selling games if available
                            if "top_selling_games" in insights and insights["top_selling_games"]:
                                st.subheader("🏆 Top Items")
                                for i, game in enumerate(insights["top_selling_games"], 1):
                                    st.write(f"{i}. **{game['name']}** ({game.get('platform', 'Unknown')}, {game.get('year', 'Unknown')}): {game['global_sales']} million global sales")
                            
                            # Show key statistics by category
                            col1, col2 = st.columns(2)
                            
                            # Platform sales if available
                            if "platform_sales" in insights:
                                with col1:
                                    st.subheader("🎮 Top Platforms")
                                    for platform, sales in list(insights["platform_sales"].items())[:5]:
                                        st.write(f"- **{platform}**: {sales:.2f}M")
                            
                            # Genre sales if available
                            if "genre_sales" in insights:
                                with col2:
                                    st.subheader("🎭 Top Genres")
                                    for genre, sales in list(insights["genre_sales"].items())[:5]:
                                        st.write(f"- **{genre}**: {sales:.2f}M")
                            
                            # Publisher sales if available
                            if "publisher_sales" in insights:
                                st.subheader("🏢 Top Publishers")
                                for publisher, sales in list(insights["publisher_sales"].items())[:5]:
                                    st.write(f"- **{publisher}**: {sales:.2f}M")
                
                # Advanced Visualizations Tab
                with tabs[2]:
                    st.subheader("📈 Enhanced Visualizations")
                    
                    # Get column types
                    df = st.session_state.df
                    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
                    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
                    
                    # Create visualization options
                    viz_type = st.selectbox(
                        "Select Visualization Type",
                        ["Correlation Analysis", "Outlier Detection", "Time Series Analysis", "Distribution Analysis", 
                         "Multi-Chart Dashboard"]
                    )
                    
                    if viz_type == "Correlation Analysis" and len(numeric_cols) >= 2:
                        if st.button("Generate Correlation Analysis"):
                            correlation = AdvancedAnalytics.generate_correlation_analysis(df)
                            if "error" not in correlation:
                                st.write("### Correlation Heatmap")
                                fig = EnhancedVisualizations.create_correlation_heatmap(correlation["correlation_matrix"])
                                st.plotly_chart(fig, use_container_width=True)
                                
                                if correlation["top_positive_correlations"]:
                                    st.write("#### 🔺 Top Positive Correlations")
                                    for col1, col2, value in correlation["top_positive_correlations"]:
                                        st.write(f"- **{col1}** and **{col2}**: {value:.2f}")
                                
                                if correlation["top_negative_correlations"]:
                                    st.write("#### 🔻 Top Negative Correlations")
                                    for col1, col2, value in correlation["top_negative_correlations"]:
                                        st.write(f"- **{col1}** and **{col2}**: {value:.2f}")
                            else:
                                st.error(correlation["error"])
                    
                    elif viz_type == "Outlier Detection" and len(numeric_cols) > 0:
                        outlier_col = st.selectbox("Select column for outlier analysis:", numeric_cols)
                        if st.button("Detect Outliers"):
                            with st.spinner("Detecting outliers..."):
                                result_df, outlier_indices = AdvancedAnalytics.detect_outliers(df, outlier_col)
                                if len(outlier_indices) > 0:
                                    st.write(f"Found {len(outlier_indices)} outliers in column **{outlier_col}**")
                                    fig = EnhancedVisualizations.create_outlier_visualization(result_df, outlier_col)
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.info(f"No significant outliers detected in column **{outlier_col}**")
                    
                    elif viz_type == "Time Series Analysis":
                        # Try to detect time columns
                        time_cols = AdvancedAnalytics.detect_time_series(df)
                        
                        if time_cols:
                            time_col = st.selectbox("Select time column:", time_cols)
                            value_col = st.selectbox("Select value column:", numeric_cols)
                            
                            if st.button("Analyze Time Series"):
                                with st.spinner("Analyzing time series..."):
                                    time_analysis = AdvancedAnalytics.analyze_time_series(df, time_col, value_col)
                                    if "error" not in time_analysis:
                                        fig = EnhancedVisualizations.create_time_series_plot(
                                            time_analysis["time_series"],
                                            time_col,
                                            value_col,
                                            time_analysis.get("trend_values")
                                        )
                                        st.plotly_chart(fig, use_container_width=True)
                                        
                                        # Show time series insights
                                        st.write(f"**Trend Direction:** {time_analysis['trend_direction'].capitalize()}")
                                        st.write(f"**Growth Rate:** {time_analysis['growth_rate']*100:.2f}%")
                                    else:
                                        st.error(time_analysis["error"])
                        else:
                            st.warning("No time columns detected in this dataset")
                    
                    elif viz_type == "Distribution Analysis" and len(numeric_cols) > 0:
                        dist_col = st.selectbox("Select column for distribution analysis:", numeric_cols)
                        if st.button("Analyze Distribution"):
                            fig = EnhancedVisualizations.create_distribution_plot(df, dist_col)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Show descriptive statistics
                            stats = AdvancedAnalytics.get_descriptive_statistics(df[[dist_col]])
                            if dist_col in stats:
                                col_stats = stats[dist_col]
                                st.write("### 📊 Descriptive Statistics")
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.write(f"**Mean:** {col_stats['mean']:.2f}")
                                    st.write(f"**Median:** {col_stats['median']:.2f}")
                                with col2:
                                    st.write(f"**Min:** {col_stats['min']:.2f}")
                                    st.write(f"**Max:** {col_stats['max']:.2f}")
                                with col3:
                                    st.write(f"**Std Dev:** {col_stats['std']:.2f}")
                                    st.write(f"**Outliers:** {col_stats['outlier_count']} ({col_stats['outlier_percentage']:.1f}%)")
                    
                    elif viz_type == "Multi-Chart Dashboard":
                        if st.button("Generate Dashboard"):
                            if len(numeric_cols) >= 2:
                                fig = EnhancedVisualizations.create_multi_chart_dashboard(
                                    df, 
                                    numeric_cols[:6]  # Limit to 6 columns for dashboard
                                )
                                if fig:
                                    st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.error("Need at least 2 numeric columns for dashboard")
                
                # Legacy Visualization Controls in Sidebar
                st.sidebar.subheader("🎨 Visualization")
                
                # Plot type selection with emojis
                plot_types = {
                    '📊 Bar Chart': px.bar,
                    '📈 Line Plot': px.line,
                    '🔵 Scatter Plot': px.scatter,
                    '📊 Histogram': px.histogram,
                    '📦 Box Plot': px.box,
                    '🎻 Violin Plot': px.violin,
                    '🥧 Pie Chart': px.pie,
                    '🌡️ Heatmap': px.imshow
                }
                
                plot_type = st.sidebar.selectbox("Select Plot Type", list(plot_types.keys()))
                
                df = st.session_state.df  # Local reference for visualization
                numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
                categorical_cols = df.select_dtypes(include=['object']).columns
                
                if '📊 Bar Chart' in plot_type or '📈 Line Plot' in plot_type or '🔵 Scatter Plot' in plot_type:
                    x_col = st.sidebar.selectbox("📏 X-axis", df.columns)
                    y_col = st.sidebar.selectbox("📐 Y-axis", df.columns)
                    color_col = st.sidebar.selectbox("🎨 Color by", ['None'] + list(df.columns))
                    
                    if st.sidebar.button('✨ Generate'):
                        if color_col != 'None':
                            fig = plot_types[plot_type](df, x=x_col, y=y_col, color=color_col)
                        else:
                            fig = plot_types[plot_type](df, x=x_col, y=y_col)
                        fig.update_layout(title=f"{y_col} vs {x_col}")
                        st.plotly_chart(fig, use_container_width=True)
                
                elif '📦 Box Plot' in plot_type or '🎻 Violin Plot' in plot_type:
                    x_col = st.sidebar.selectbox("📏 Category", categorical_cols)
                    y_col = st.sidebar.selectbox("📐 Value", numeric_cols)
                    
                    if st.sidebar.button('✨ Generate'):
                        fig = plot_types[plot_type](df, x=x_col, y=y_col)
                        fig.update_layout(title=f"{y_col} Distribution by {x_col}")
                        st.plotly_chart(fig, use_container_width=True)
                
                elif '📊 Histogram' in plot_type:
                    col = st.sidebar.selectbox("📏 Column", numeric_cols)
                    bins = st.sidebar.slider("📊 Bins", 5, 100, 30)
                    
                    if st.sidebar.button('✨ Generate'):
                        fig = px.histogram(df, x=col, nbins=bins)
                        fig.update_layout(title=f"Distribution of {col}")
                        st.plotly_chart(fig, use_container_width=True)
                
                elif '🥧 Pie Chart' in plot_type:
                    names_col = st.sidebar.selectbox("📏 Category", categorical_cols)
                    values_col = st.sidebar.selectbox("📐 Values", numeric_cols)
                    
                    if st.sidebar.button('✨ Generate'):
                        fig = px.pie(df, names=names_col, values=values_col)
                        fig.update_layout(title=f"{values_col} by {names_col}")
                        st.plotly_chart(fig, use_container_width=True)
                        
                elif '🌡️ Heatmap' in plot_type:
                    if len(numeric_cols) < 2:
                        st.sidebar.error("Need at least 2 numeric columns for heatmap")
                    else:
                        corr_matrix = df[numeric_cols].corr()
                        fig = px.imshow(corr_matrix,
                                    labels=dict(color="Correlation"),
                                    x=corr_matrix.columns,
                                    y=corr_matrix.columns)
                        fig.update_layout(title="Correlation Heatmap")
                        st.plotly_chart(fig, use_container_width=True)
                        
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
    
    # Chat interface in right column
    with chat_col:
        st.subheader("💬 Chat")
        
        # Display chat messages
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.write("👤 You:", message["content"])
            else:
                st.write("🤖 Assistant:", message["content"])
        
        # Chat input
        user_input = st.text_input("Ask about the dataset...")
        
        # Send button outside the form
        if st.button("💬 Send", key="send_button"):
            if not user_input:
                st.error("Please enter a message first!")
            elif st.session_state.df is None:
                st.error("Please upload a dataset first!")
            else:
                # Add user message to chat history
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": user_input
                })
                
                # Get bot response
                response = st.session_state.dataset_chat.get_chat_response(
                    st.session_state.df, user_input
                )
                
                # Add bot response to chat history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response
                })
                st.rerun()
        
        # Clear chat button
        if st.button("🔄 Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

if __name__ == "__main__":
    main()
