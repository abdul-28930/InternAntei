# Dataset Chat Assistant

An interactive data analytics and visualization application that leverages OpenAI's GPT-4o-Mini to provide intelligent dataset analysis, insights generation, and advanced visualizations.

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

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure environment variables:
   - Copy `.env.example` to `.env`
   - Add your OpenAI API key to the `.env` file:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Running the Application

```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

## How to Use

1. **Upload Your Dataset**: Support for CSV and Excel files
2. **Explore Data**: View your data in the preview tab
3. **Get Insights**: Check the Insights tab for automatic analysis of your data
4. **Create Visualizations**: Generate advanced visualizations in the Visualizations tab
5. **Chat with Your Data**: Ask questions in natural language about your dataset

## Technology Stack

- **OpenAI GPT-4o-Mini**: For natural language understanding and dataset insights
- **Streamlit**: For the interactive web interface
- **Plotly**: For interactive data visualizations
- **Pandas**: For data manipulation and analysis
- **Scikit-learn**: For advanced analytics (outlier detection, clustering)

## Requirements

- Python 3.8 or higher
- Internet connection for OpenAI API access
- OpenAI API key
