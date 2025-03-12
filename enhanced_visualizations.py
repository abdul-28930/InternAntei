import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

class EnhancedVisualizations:    
    @staticmethod
    def create_correlation_heatmap(corr_matrix):
        """Create an enhanced correlation heatmap with annotations
        
        Args:
            corr_matrix (pd.DataFrame): Correlation matrix dataframe
            
        Returns:
            go.Figure: Plotly figure object
        """
        # Create heatmap
        fig = px.imshow(
            corr_matrix,
            color_continuous_scale='RdBu_r',
            zmin=-1, zmax=1,
            text_auto='.2f',
            aspect="auto"
        )
        
        # Update layout
        fig.update_layout(
            title="Correlation Matrix Heatmap",
            height=600,
            width=700,
            xaxis_title="Features",
            yaxis_title="Features",
            coloraxis_colorbar=dict(
                title="Correlation Coefficient",
                thicknessmode="pixels", thickness=20,
                lenmode="pixels", len=500,
                yanchor="top", y=1,
                ticks="outside"
            )
        )
        
        return fig
    
    @staticmethod
    def create_outlier_visualization(df, column):
        """Create a box plot visualization highlighting outliers
        
        Args:
            df (pd.DataFrame): Input dataframe with is_outlier column
            column (str): Column name to visualize
            
        Returns:
            go.Figure: Plotly figure object
        """
        if 'is_outlier' not in df.columns:
            return px.box(df, y=column, title=f"Box Plot: {column}")
            
        # Create box plot
        fig = go.Figure()
        
        # Add box plot
        fig.add_trace(go.Box(
            y=df[column],
            name=column,
            boxmean=True,
            boxpoints=False,
            line=dict(color='rgba(8, 81, 156, 0.8)')
        ))
        
        # Add scatter points on top
        normal_points = df[~df['is_outlier']][column]
        outlier_points = df[df['is_outlier']][column]
        
        fig.add_trace(go.Scatter(
            y=normal_points,
            x=np.ones(len(normal_points)),
            mode='markers',
            name='Normal',
            marker=dict(
                color='rgba(8, 81, 156, 0.7)',
                size=6
            ),
            opacity=0.7
        ))
        
        fig.add_trace(go.Scatter(
            y=outlier_points,
            x=np.ones(len(outlier_points)),
            mode='markers',
            name='Outliers',
            marker=dict(
                color='rgba(255, 0, 0, 0.9)',
                size=9,
                line=dict(
                    color='darkred',
                    width=2
                )
            )
        ))
        
        # Update layout
        fig.update_layout(
            title=f"Outlier Analysis: {column}",
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False
            ),
            yaxis_title=column,
            legend_title="Data Points"
        )
        
        return fig
    
    @staticmethod
    def create_cluster_visualization(df, cluster_col, x_col, y_col, centers=None):
        """Create a scatter plot visualization of clusters
        
        Args:
            df (pd.DataFrame): Input dataframe with cluster column
            cluster_col (str): Column containing cluster labels
            x_col (str): Column for x-axis
            y_col (str): Column for y-axis
            centers (np.ndarray): Cluster centers array (optional)
            
        Returns:
            go.Figure: Plotly figure object
        """
        if cluster_col not in df.columns:
            return px.scatter(df, x=x_col, y=y_col, title=f"{y_col} vs {x_col}")
            
        # Remove rows with NaN clusters
        clean_df = df.dropna(subset=[cluster_col])
        
        # Create scatter plot with clusters
        fig = px.scatter(
            clean_df,
            x=x_col,
            y=y_col,
            color=cluster_col,
            color_continuous_scale=px.colors.qualitative.G10,
            title=f"Cluster Analysis: {y_col} vs {x_col}",
            hover_data=[cluster_col]
        )
        
        # Add cluster centers if provided
        if centers is not None and len(centers) > 0:
            fig.add_trace(go.Scatter(
                x=centers[:, df.columns.get_loc(x_col)],
                y=centers[:, df.columns.get_loc(y_col)],
                mode='markers',
                marker=dict(
                    color='black',
                    size=15,
                    symbol='x'
                ),
                name='Cluster Centers'
            ))
        
        return fig
    
    @staticmethod
    def create_time_series_plot(time_series, time_col, value_col, trend_values=None):
        """Create an enhanced time series plot with trend line
        
        Args:
            time_series (pd.DataFrame): Time series dataframe
            time_col (str): Column containing time data
            value_col (str): Column containing values to plot
            trend_values (list): Trend line values (optional)
            
        Returns:
            go.Figure: Plotly figure object
        """
        # Create time series plot
        fig = go.Figure()
        
        # Add actual data
        fig.add_trace(go.Scatter(
            x=time_series[time_col],
            y=time_series[value_col],
            mode='lines+markers',
            name='Actual',
            line=dict(color='royalblue', width=2),
            marker=dict(size=6)
        ))
        
        # Add trend line if provided
        if trend_values is not None:
            fig.add_trace(go.Scatter(
                x=time_series[time_col],
                y=trend_values,
                mode='lines',
                name='Trend',
                line=dict(color='red', width=2, dash='dash')
            ))
        
        # Update layout
        fig.update_layout(
            title=f"Time Series Analysis: {value_col} over Time",
            xaxis_title="Time",
            yaxis_title=value_col,
            legend_title="Data",
            hovermode="x unified"
        )
        
        return fig
    
    @staticmethod
    def create_distribution_plot(df, column):
        """Create an enhanced distribution plot (histogram with KDE)
        
        Args:
            df (pd.DataFrame): Input dataframe
            column (str): Column to visualize
            
        Returns:
            go.Figure: Plotly figure object
        """
        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Get column data without NaN
        data = df[column].dropna()
        
        # Add histogram
        fig.add_trace(
            go.Histogram(
                x=data,
                name="Histogram",
                opacity=0.75,
                nbinsx=30,
                marker_color='rgba(73, 133, 178, 0.7)'
            ),
            secondary_y=False
        )
        
        # Try to add KDE
        try:
            from scipy import stats
            
            # Calculate KDE
            kde_x = np.linspace(data.min(), data.max(), 1000)
            kde = stats.gaussian_kde(data)
            kde_y = kde(kde_x)
            
            # Scale KDE to match histogram
            hist, bin_edges = np.histogram(data, bins=30)
            scaling_factor = max(hist) / max(kde_y)
            kde_y_scaled = kde_y * scaling_factor
            
            # Add KDE curve
            fig.add_trace(
                go.Scatter(
                    x=kde_x,
                    y=kde_y_scaled,
                    name="Density",
                    line=dict(color='rgb(203, 67, 53)', width=2)
                ),
                secondary_y=False
            )
        except:
            # If scipy is not available, skip KDE
            pass
        
        # Add percentile lines
        percentiles = [0.25, 0.5, 0.75]
        percentile_values = [data.quantile(p) for p in percentiles]
        percentile_labels = ["25th", "Median", "75th"]
        colors = ['rgba(44, 160, 44, 0.7)', 'rgba(214, 39, 40, 0.7)', 'rgba(148, 103, 189, 0.7)']
        
        for i, (value, label, color) in enumerate(zip(percentile_values, percentile_labels, colors)):
            fig.add_vline(
                x=value,
                line_width=2,
                line_dash="dash",
                line_color=color,
                annotation_text=f"{label} Percentile",
                annotation_position="top right"
            )
        
        # Update layout
        fig.update_layout(
            title=f"Distribution Analysis: {column}",
            xaxis_title=column,
            yaxis_title="Frequency",
            bargap=0.1,
            hovermode="x unified"
        )
        
        return fig
    
    @staticmethod
    def create_multi_chart_dashboard(df, cols):
        """Create a multi-chart dashboard with multiple visualizations
        
        Args:
            df (pd.DataFrame): Input dataframe
            cols (list): List of columns to visualize
            
        Returns:
            go.Figure: Plotly figure object
        """
        # Filter to numeric columns only
        numeric_cols = [col for col in cols if pd.api.types.is_numeric_dtype(df[col])]
        
        if len(numeric_cols) < 2:
            st.error("Need at least 2 numeric columns for dashboard")
            return None
            
        # Create subplot grid
        rows = min(len(numeric_cols), 3)
        cols_grid = min(2, (len(numeric_cols) + rows - 1) // rows)
        
        # Create figure
        fig = make_subplots(
            rows=rows, 
            cols=cols_grid,
            subplot_titles=[f"Distribution: {col}" for col in numeric_cols[:rows*cols_grid]]
        )
        
        # Add histograms for each column
        for i, col in enumerate(numeric_cols[:rows*cols_grid]):
            row = i // cols_grid + 1
            col_pos = i % cols_grid + 1
            
            # Add histogram
            fig.add_trace(
                go.Histogram(
                    x=df[col].dropna(),
                    name=col,
                    opacity=0.75,
                    marker_color=px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
                ),
                row=row, col=col_pos
            )
        
        # Update layout
        fig.update_layout(
            title="Multi-Chart Dashboard",
            height=300 * rows,
            showlegend=False,
            hovermode="closest"
        )
        
        return fig
    
    @staticmethod
    def suggest_visualizations(df):
        """Suggest appropriate visualizations based on dataframe content
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            list: List of suggested visualization types and relevant columns
        """
        suggestions = []
        
        # Get column types
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        # Correlation heatmap for numeric columns
        if len(numeric_cols) >= 2:
            suggestions.append({
                "type": "Correlation Heatmap",
                "description": "Visualize relationships between numeric variables",
                "columns": numeric_cols,
                "icon": "🌡️"
            })
        
        # Distribution plots for numeric columns
        if len(numeric_cols) > 0:
            suggestions.append({
                "type": "Distribution Analysis",
                "description": "Analyze the distribution of individual numeric variables",
                "columns": numeric_cols,
                "icon": "📊"
            })
        
        # Scatter plots for pairs of numeric columns
        if len(numeric_cols) >= 2:
            suggestions.append({
                "type": "Scatter Plot",
                "description": "Explore relationships between pairs of numeric variables",
                "columns": numeric_cols,
                "icon": "🔵"
            })
        
        # Bar charts for categorical columns
        if len(categorical_cols) > 0 and len(numeric_cols) > 0:
            suggestions.append({
                "type": "Bar Chart",
                "description": "Compare numeric values across categories",
                "categorical_columns": categorical_cols,
                "numeric_columns": numeric_cols,
                "icon": "📊"
            })
        
        # Time series for datetime columns
        if len(datetime_cols) > 0 and len(numeric_cols) > 0:
            suggestions.append({
                "type": "Time Series",
                "description": "Analyze trends over time",
                "time_columns": datetime_cols,
                "value_columns": numeric_cols,
                "icon": "📈"
            })
        elif len(numeric_cols) > 0:
            # Check for columns that might be years
            year_cols = []
            for col in numeric_cols:
                values = df[col].dropna()
                if len(values) > 0:
                    # If most values are between 1900-2100, likely years
                    year_count = ((values >= 1900) & (values <= 2100)).sum()
                    if year_count / len(values) > 0.8:
                        year_cols.append(col)
            
            if len(year_cols) > 0:
                suggestions.append({
                    "type": "Time Series",
                    "description": "Analyze trends over years",
                    "time_columns": year_cols,
                    "value_columns": [col for col in numeric_cols if col not in year_cols],
                    "icon": "📈"
                })
        
        # Box plots for categorical and numeric columns
        if len(categorical_cols) > 0 and len(numeric_cols) > 0:
            suggestions.append({
                "type": "Box Plot",
                "description": "Compare distributions across categories",
                "categorical_columns": categorical_cols,
                "numeric_columns": numeric_cols,
                "icon": "📦"
            })
        
        # Pie charts for categorical columns
        if len(categorical_cols) > 0:
            for col in categorical_cols:
                if df[col].nunique() <= 10:  # Only suggest for columns with reasonable number of categories
                    suggestions.append({
                        "type": "Pie Chart",
                        "description": f"Show proportion of categories in {col}",
                        "column": col,
                        "icon": "🥧"
                    })
        
        return suggestions
