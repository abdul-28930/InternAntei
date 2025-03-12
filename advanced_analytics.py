import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import re

class AdvancedAnalytics:
    """Advanced analytics module for data analysis and visualization"""
    
    @staticmethod
    def detect_outliers(df, column, contamination=0.05):
        """Detect outliers in a numeric column using Isolation Forest
        
        Args:
            df (pd.DataFrame): Input dataframe
            column (str): Column name to analyze
            contamination (float): Expected proportion of outliers
            
        Returns:
            tuple: (DataFrame with outliers flagged, outlier indices)
        """
        if column not in df.columns or not pd.api.types.is_numeric_dtype(df[column]):
            return df, []
            
        # Remove NaN values for isolation forest
        data = df[column].dropna().values.reshape(-1, 1)
        
        # Need at least 10 data points for meaningful outlier detection
        if len(data) < 10:
            return df, []
            
        # Use isolation forest for outlier detection
        model = IsolationForest(contamination=contamination, random_state=42)
        model.fit(data)
        
        # Get outliers
        outlier_indices = np.where(model.predict(data) == -1)[0]
        
        # Create result dataframe
        result_df = df.copy()
        result_df['is_outlier'] = False
        
        # Map outliers back to original dataframe indices
        original_indices = df[column].dropna().index
        outlier_original_indices = [original_indices[i] for i in outlier_indices]
        result_df.loc[outlier_original_indices, 'is_outlier'] = True
        
        return result_df, outlier_original_indices
    
    @staticmethod
    def generate_correlation_analysis(df):
        """Generate detailed correlation analysis for numeric columns
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            dict: Dictionary with correlation insights
        """
        numeric_df = df.select_dtypes(include=['int64', 'float64'])
        if len(numeric_df.columns) < 2:
            return {"error": "Need at least 2 numeric columns for correlation analysis"}
            
        # Calculate correlation matrix
        corr_matrix = numeric_df.corr()
        
        # Find strongest positive and negative correlations
        corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]
                corr_value = corr_matrix.iloc[i, j]
                corr_pairs.append((col1, col2, corr_value))
        
        # Sort by absolute correlation value
        corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        
        # Get top positive and negative correlations
        top_positive = [pair for pair in corr_pairs if pair[2] > 0][:5]
        top_negative = [pair for pair in corr_pairs if pair[2] < 0][:5]
        
        return {
            "correlation_matrix": corr_matrix,
            "top_positive_correlations": top_positive,
            "top_negative_correlations": top_negative
        }
    
    @staticmethod
    def perform_clustering(df, columns=None, n_clusters=3):
        """Perform K-means clustering on selected numeric columns
        
        Args:
            df (pd.DataFrame): Input dataframe
            columns (list): List of columns to use for clustering, defaults to all numeric
            n_clusters (int): Number of clusters to create
            
        Returns:
            tuple: (DataFrame with cluster labels, cluster centers)
        """
        # Select numeric columns if not specified
        if columns is None:
            numeric_df = df.select_dtypes(include=['int64', 'float64'])
            columns = numeric_df.columns.tolist()
        else:
            numeric_df = df[columns].select_dtypes(include=['int64', 'float64'])
            columns = numeric_df.columns.tolist()
            
        if len(columns) < 2:
            return df, None
            
        # Remove rows with NaN values
        clean_df = numeric_df[columns].dropna()
        if len(clean_df) < n_clusters:
            return df, None
            
        # Scale the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(clean_df)
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(scaled_data)
        
        # Get cluster centers
        centers = kmeans.cluster_centers_
        
        # Map back to original scale
        centers_original = scaler.inverse_transform(centers)
        
        # Create result dataframe
        result_df = df.copy()
        result_df['cluster'] = np.nan
        
        # Map cluster labels back to original dataframe indices
        original_indices = clean_df.index
        for i, idx in enumerate(original_indices):
            result_df.loc[idx, 'cluster'] = cluster_labels[i]
            
        return result_df, centers_original
    
    @staticmethod
    def detect_time_series(df):
        """Detect time series columns in dataframe
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            list: List of potential time series columns
        """
        time_columns = []
        
        # Check for datetime columns
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        time_columns.extend(datetime_cols)
        
        # Check for string columns that might contain dates
        for col in df.select_dtypes(include=['object']).columns:
            # Check first 5 non-null values
            sample = df[col].dropna().head(5).astype(str).tolist()
            
            # Simple regex patterns for date detection
            date_patterns = [
                r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
                r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
                r'\d{2}-\d{2}-\d{4}',  # DD-MM-YYYY
                r'\d{2}\.\d{2}\.\d{4}'  # DD.MM.YYYY
            ]
            
            # Check if most values match date patterns
            matches = 0
            for value in sample:
                for pattern in date_patterns:
                    if re.search(pattern, value):
                        matches += 1
                        break
            
            # If most values look like dates, add to time columns
            if matches >= min(3, len(sample)):
                time_columns.append(col)
                
        # Check for year columns (numeric columns with mostly values between 1900-2100)
        for col in df.select_dtypes(include=['int64', 'float64']).columns:
            values = df[col].dropna()
            if len(values) > 0:
                # If most values are between 1900-2100, likely years
                year_count = ((values >= 1900) & (values <= 2100)).sum()
                if year_count / len(values) > 0.8:
                    time_columns.append(col)
                    
        return time_columns
    
    @staticmethod
    def analyze_time_series(df, time_col, value_col):
        """Analyze time series data
        
        Args:
            df (pd.DataFrame): Input dataframe
            time_col (str): Column containing time data
            value_col (str): Column containing values to analyze
            
        Returns:
            dict: Dictionary with time series analytics
        """
        if time_col not in df.columns or value_col not in df.columns:
            return {"error": "Columns not found in dataframe"}
            
        # Create a copy to avoid modifying original
        ts_df = df[[time_col, value_col]].copy()
        
        # Handle different time column formats
        if pd.api.types.is_datetime64_dtype(ts_df[time_col]):
            time_series = ts_df
        else:
            # Try to convert to datetime
            try:
                ts_df[time_col] = pd.to_datetime(ts_df[time_col])
                time_series = ts_df
            except:
                # If it's a year column, convert to datetime
                if pd.api.types.is_numeric_dtype(ts_df[time_col]):
                    try:
                        # Assume it's a year column
                        ts_df[time_col] = pd.to_datetime(ts_df[time_col], format='%Y')
                        time_series = ts_df
                    except:
                        return {"error": "Could not convert time column to datetime"}
                else:
                    return {"error": "Could not convert time column to datetime"}
        
        # Remove rows with NaN values
        time_series = time_series.dropna()
        if len(time_series) < 3:
            return {"error": "Not enough data points for time series analysis"}
            
        # Sort by time
        time_series = time_series.sort_values(by=time_col)
        
        # Calculate trend
        try:
            # Simple linear regression for trend
            x = np.arange(len(time_series))
            y = time_series[value_col].values
            
            # Calculate trend using polyfit
            trend_coef = np.polyfit(x, y, 1)
            trend = np.poly1d(trend_coef)
            
            # Calculate trend values
            trend_values = trend(x)
            
            # Determine trend direction
            if trend_coef[0] > 0:
                trend_direction = "increasing"
            elif trend_coef[0] < 0:
                trend_direction = "decreasing"
            else:
                trend_direction = "stable"
                
            # Calculate statistics
            min_value = time_series[value_col].min()
            max_value = time_series[value_col].max()
            mean_value = time_series[value_col].mean()
            
            # Calculate growth rate
            if len(time_series) >= 2:
                first_value = time_series[value_col].iloc[0]
                last_value = time_series[value_col].iloc[-1]
                if first_value != 0:
                    growth_rate = (last_value - first_value) / first_value
                else:
                    growth_rate = float('inf') if last_value > 0 else 0
            else:
                growth_rate = 0
                
            return {
                "time_series": time_series,
                "trend_direction": trend_direction,
                "trend_coefficient": float(trend_coef[0]),
                "trend_values": trend_values.tolist(),
                "min_value": float(min_value),
                "max_value": float(max_value),
                "mean_value": float(mean_value),
                "growth_rate": float(growth_rate)
            }
            
        except Exception as e:
            return {"error": f"Error in trend analysis: {str(e)}"}
    
    @staticmethod
    def get_descriptive_statistics(df):
        """Get enhanced descriptive statistics for all numeric columns
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            dict: Dictionary with descriptive statistics for each numeric column
        """
        # Select numeric columns
        numeric_df = df.select_dtypes(include=['int64', 'float64'])
        if len(numeric_df.columns) == 0:
            return {}
            
        stats = {}
        for col in numeric_df.columns:
            col_stats = {}
            
            # Basic statistics
            values = numeric_df[col].dropna()
            if len(values) == 0:
                continue
                
            col_stats["count"] = len(values)
            col_stats["min"] = float(values.min())
            col_stats["max"] = float(values.max())
            col_stats["mean"] = float(values.mean())
            col_stats["median"] = float(values.median())
            col_stats["std"] = float(values.std())
            
            # Percentiles
            col_stats["25th_percentile"] = float(values.quantile(0.25))
            col_stats["75th_percentile"] = float(values.quantile(0.75))
            col_stats["iqr"] = col_stats["75th_percentile"] - col_stats["25th_percentile"]
            
            # Skewness and kurtosis if pandas has them
            try:
                col_stats["skewness"] = float(values.skew())
                col_stats["kurtosis"] = float(values.kurt())
            except:
                pass
                
            # Number of unique values
            col_stats["unique_count"] = values.nunique()
            
            # Outlier boundaries using IQR
            col_stats["lower_bound"] = col_stats["25th_percentile"] - 1.5 * col_stats["iqr"]
            col_stats["upper_bound"] = col_stats["75th_percentile"] + 1.5 * col_stats["iqr"]
            
            # Count of potential outliers
            outliers = ((values < col_stats["lower_bound"]) | (values > col_stats["upper_bound"])).sum()
            col_stats["outlier_count"] = int(outliers)
            col_stats["outlier_percentage"] = float(outliers / len(values) * 100)
            
            stats[col] = col_stats
            
        return stats
