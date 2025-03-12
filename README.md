# Dataset Chat Assistant

An interactive data analytics and visualization application that leverages OpenAI's GPT-4o-Mini to provide intelligent dataset analysis, insights generation, and advanced visualizations.

![Dataset Chat Assistant](https://img.shields.io/badge/Dataset_Chat-Assistant-blue)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32.0-red)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-green)

## Features

### Advanced Analytics
- **Outlier Detection**: Identify anomalies in numeric data using Isolation Forest algorithm
- **Correlation Analysis**: Generate and visualize correlation matrices with significant relationship highlighting
- **Clustering**: Apply K-means clustering to discover patterns in multidimensional data
- **Time Series Analysis**: Detect trends and patterns in time-based data
- **Descriptive Statistics**: Calculate and visualize key statistical measures for numeric columns

### Enhanced Visualizations
- **Interactive Charts**: Rich, interactive visualizations built with Plotly
- **AI-Suggested Visualizations**: Get chart recommendations based on your dataset's structure
- **Multi-Chart Dashboards**: View multiple aspects of your data in a comprehensive dashboard
- **Emoji-Based Controls**: Intuitive UI with emoji icons for better user experience

### Chat Interface
- **Dataset-Aware Conversations**: Ask questions about your data using natural language
- **Intelligent Insights**: Receive auto-generated insights based on dataset content
- **Context-Sensitive Responses**: Get relevant answers based on the uploaded dataset

## Repository Structure

```
dataset-chat-assistant/
├── app.py                     # Main application file with Streamlit UI
├── advanced_analytics.py      # Advanced analytics module (outlier detection, clustering, etc.)
├── enhanced_visualizations.py # Enhanced visualization module (heatmaps, dashboards, etc.)
├── ANALYTICAL_REPORT.md       # Comparative analysis with similar technologies
├── .env                       # Environment variables (not in repo)
├── .env.example               # Example environment variables template
├── requirements.txt           # Project dependencies
└── README.md                  # This documentation
```

### Key Files Description

- **app.py**: Contains the main `DatasetChat` class and Streamlit interface code. This file handles UI components, user interactions, and integrates the analytics and visualization modules.

- **advanced_analytics.py**: Implements the `AdvancedAnalytics` class with methods for:
  - `detect_outliers()`: Uses Isolation Forest to find anomalies in data
  - `generate_correlation_analysis()`: Calculates correlations between numeric columns
  - `perform_clustering()`: Applies K-means clustering to find patterns
  - `analyze_time_series()`: Detects trends in time-series data
  - `get_descriptive_statistics()`: Calculates key statistical measures

- **enhanced_visualizations.py**: Implements the `EnhancedVisualizations` class with methods for:
  - `create_correlation_heatmap()`: Generates interactive correlation matrices
  - `create_outlier_visualization()`: Visualizes detected outliers
  - `create_distribution_plot()`: Shows distributions with kernel density estimation
  - `create_time_series_plot()`: Generates time series with trend lines
  - `create_multi_chart_dashboard()`: Creates a comprehensive dashboard
  - `suggest_visualizations()`: Recommends appropriate charts based on data

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- Git (for cloning the repository)
- OpenAI API key

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/abdul-28930/InternAntei.git
   cd InternAntei
   ```

2. **Create and activate a virtual environment** (optional but recommended):
   ```bash
   # On Windows
   python -m venv .anteiintern
   .anteiintern\Scripts\activate

   # On macOS/Linux
   python -m venv .anteiintern
   source .anteiintern/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   - Copy the example environment file:
     ```bash
     cp .env.example .env
     ```
   - Edit the `.env` file and add your OpenAI API key:
     ```
     OPENAI_API_KEY=your_api_key_here
     DEFAULT_MODEL=gpt-4o-mini
     ```

## Running the Application

1. **Start the Streamlit server**:
   ```bash
   streamlit run app.py
   ```

2. **Access the application**:
   Open your browser and navigate to `http://localhost:8501`

## How to Use

1. **Upload Your Dataset**:
   - Click the "📂 Upload Dataset" button in the sidebar
   - Select a CSV or Excel file from your computer

2. **Explore the Dataset**:
   - The "📊 Data Preview" tab shows the first few rows of your data
   - The "🔍 Insights" tab provides automatic analysis of key metrics

3. **Generate Visualizations**:
   - Navigate to the "📈 Advanced Visualizations" tab
   - Select from correlation analysis, outlier detection, time series analysis, etc.
   - Click the generate button to create interactive visualizations

4. **Chat with Your Data**:
   - Type questions about your dataset in the chat box
   - Get AI-powered responses based on the actual content of your data

## Examples

### Example Questions

- "What are the top-selling games in this dataset?"
- "Show me the correlation between sales and critic scores"
- "Which platforms have the highest average sales?"
- "Is there a trend in game sales over time?"
- "Identify outliers in the global sales column"

## Troubleshooting

- **API Key Issues**: If you see authentication errors, verify your OpenAI API key in the `.env` file
- **File Upload Errors**: Make sure your CSV or Excel file is properly formatted
- **Memory Errors**: For large datasets, try reducing the size or using a subset of the data
- **Visualization Errors**: Ensure you have selected appropriate columns for each visualization type
