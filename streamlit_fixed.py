import streamlit as st
import pandas as pd
import requests
import warnings
import re
import base64
from PIL import Image
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_community.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType
import io

# 📌 OpenAI API key
api_key = "api-key

def run_code_with_judge0(code: str):
    """Execute code using Judge0 API (has matplotlib support)"""
    plot_wrapper = """
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
import io

# Original code
"""
    
    plot_saver = """

# Save any plots as base64
try:
    if plt.get_fignums():
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        print(f"PLOT_DATA:{image_base64}")
        buffer.close()
        plt.close('all')
except Exception as e:
    print(f"PLOT_ERROR: {str(e)}")
"""
    
    modified_code = plot_wrapper + code + plot_saver
    
    url = "https://judge0-ce.p.rapidapi.com/submissions"
    headers = {
        "Content-Type": "application/json",
        "X-RapidAPI-Host": "judge0-ce.p.rapidapi.com",
        "X-RapidAPI-Key": "demo"  # Free tier
    }
    payload = {
        "language_id": 71,  # Python 3
        "source_code": modified_code
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        submission = response.json()
        
        if "token" in submission:
            # Wait for execution
            import time
            time.sleep(2)
            
            # Get result
            result_url = f"https://judge0-ce.p.rapidapi.com/submissions/{submission['token']}"
            result_response = requests.get(result_url, headers=headers)
            return result_response.json()
        else:
            return {"error": "Submission failed"}
            
    except Exception as e:
        return {"error": str(e)}

def run_code_with_piston(code: str, language: str = "python", version: str = "3.10.0"):
    """Execute code using Piston API with plot handling"""
    # Modify code to save plots as base64
    plot_wrapper = """
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import base64
    import io
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("PLOT_ERROR: matplotlib not available")

# Original code
"""
    
    plot_saver = """

# Save any plots as base64
if MATPLOTLIB_AVAILABLE:
    try:
        if plt.get_fignums():
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            print(f"PLOT_DATA:{image_base64}")
            buffer.close()
            plt.close('all')
    except Exception as e:
        print(f"PLOT_ERROR: {str(e)}")
"""
    
    modified_code = plot_wrapper + code + plot_saver
    
    url = "https://emkc.org/api/v2/piston/execute"
    payload = {
        "language": language,
        "version": version,
        "files": [
            {
                "name": "main.py",
                "content": modified_code
            }
        ]
    }
    
    try:
        response = requests.post(url, json=payload)
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def extract_code_from_text(text):
    """Extract Python code blocks from text"""
    # Look for code blocks with ```python or ```
    code_pattern = r'```(?:python)?\n(.*?)\n```'
    matches = re.findall(code_pattern, text, re.DOTALL)
    return matches

def format_execution_result(result):
    """Format the execution result for display"""
    if "error" in result:
        return f"❌ **Error:** {result['error']}"
    
    run_info = result.get("run", {})
    if run_info.get("stderr"):
        return f"❌ **Error:**\n```\n{run_info['stderr']}\n```"
    
    if run_info.get("stdout"):
        stdout = run_info['stdout']
        
        # Check for plot data or errors
        if "PLOT_DATA:" in stdout or "PLOT_ERROR:" in stdout:
            lines = stdout.split('\n')
            output_lines = []
            plot_data = None
            plot_error = None
            
            for line in lines:
                if line.startswith("PLOT_DATA:"):
                    plot_data = line.replace("PLOT_DATA:", "")
                elif line.startswith("PLOT_ERROR:"):
                    plot_error = line.replace("PLOT_ERROR:", "")
                else:
                    output_lines.append(line)
            
            # Display regular output
            if output_lines and any(line.strip() for line in output_lines):
                st.markdown("✅ **Output:**")
                st.code('\n'.join(output_lines))
            
            # Display plot or error
            if plot_data:
                try:
                    import base64
                    image_bytes = base64.b64decode(plot_data)
                    st.markdown("📊 **Generated Plot:**")
                    st.image(image_bytes, use_column_width=True)
                except Exception as e:
                    st.error(f"Error displaying plot: {str(e)}")
            elif plot_error:
                st.warning(f"⚠️ **Plot Error:** {plot_error}")
            
            return None  # Already displayed above
        else:
            return f"✅ **Output:**\n```\n{stdout}\n```"
    
    return "✅ **Code executed successfully (no output)**"

# Page config
st.set_page_config(
    page_title="AI Data Analyst", 
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    .feature-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    .stMetric {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e9ecef;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown("""
<div class="main-header">
    <h1>🤖 AI Data Analyst</h1>
    <p>Upload your data and let AI create visualizations, build ML models, and provide insights!</p>
</div>
""", unsafe_allow_html=True)

# Sidebar with information
with st.sidebar:
    st.markdown("## 🚀 Features")
    
    st.markdown("""
    <div class="feature-card">
        <h4>📊 Data Visualization</h4>
        <p>matplotlib, seaborn, plotly, bokeh</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="feature-card">
        <h4>🤖 Machine Learning</h4>
        <p>scikit-learn, XGBoost, LightGBM</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="feature-card">
        <h4>☁️ Word Clouds</h4>
        <p>Text visualization and analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 💡 Example Queries")
    st.code("Create a histogram of ratings")
    st.code("Build a regression model")
    st.code("Make an interactive plotly chart")
    st.code("Generate word cloud from titles")

# Upload section
st.markdown("## 📁 Data Upload")
uploaded_file = st.file_uploader(
    "Upload your data file",
    type=["csv", "xlsx"],
    help="Supported formats: CSV, Excel (xlsx)"
)

if uploaded_file:
    # Detect file type and read accordingly
    with st.spinner("Loading data..."):
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
    
    # Success message
    st.success(f"✅ Successfully loaded {len(df)} rows and {len(df.columns)} columns!")
    
    # Data overview in columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📋 Data Preview")
        st.dataframe(df.head(), use_container_width=True)
    
    with col2:
        st.markdown("### 📊 Dataset Info")
        st.metric("Rows", f"{len(df):,}")
        st.metric("Columns", len(df.columns))
        st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
        
        # Column types summary
        st.markdown("**Column Types:**")
        type_counts = df.dtypes.value_counts()
        for dtype, count in type_counts.items():
            st.text(f"{str(dtype)}: {count}")
    
    st.markdown("---")
    
    # Generate dynamic example queries based on dataset
    def generate_example_queries(dataframe):
        """Generate example queries based on the dataset columns"""
        queries = []
        columns = dataframe.columns.tolist()
        
        # Numeric columns for visualizations
        numeric_cols = dataframe.select_dtypes(include=['int64', 'float64']).columns.tolist()
        # Text columns for word clouds
        text_cols = dataframe.select_dtypes(include=['object']).columns.tolist()
        # Date columns
        date_cols = []
        for col in columns:
            if 'date' in col.lower() or 'time' in col.lower():
                date_cols.append(col)
        
        # Generate visualization queries
        if numeric_cols:
            queries.append(f"Create a histogram of {numeric_cols[0]}")
            if len(numeric_cols) > 1:
                queries.append(f"Show correlation between {numeric_cols[0]} and {numeric_cols[1]}")
                queries.append(f"Build a regression model to predict {numeric_cols[0]}")
        
        # Text analysis queries
        if text_cols:
            queries.append(f"Generate word cloud from {text_cols[0]}")
        
        # Time series if date columns exist
        if date_cols and numeric_cols:
            queries.append(f"Plot {numeric_cols[0]} over time using {date_cols[0]}")
        
        # Advanced queries
        if len(numeric_cols) >= 3:
            queries.append(f"Create interactive plotly scatter plot")
        
        queries.append("Compare different ML models performance")
        
        return queries[:6]  # Limit to 6 examples
    
    # Update sidebar with dynamic examples
    example_queries = generate_example_queries(df)
    
    # Update sidebar
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🎯 Smart Queries for Your Data")
        for query in example_queries:
            st.code(query)
        
        st.markdown("---")
        st.info("🔥 **Tip:** These queries are generated based on your dataset columns!")
    
    # Create agent with hardcoded API key
    agent = create_pandas_dataframe_agent(
        ChatOpenAI(
            temperature=0,
            model="gpt-3.5-turbo",
            api_key=api_key
        ),
        df,
        verbose=False,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        prefix="You are working with a pandas dataframe called `df`. Always provide the Python code in markdown code blocks (```python) when creating visualizations or analysis. Show the actual code that generates plots, don't just describe them.",
        **{"allow_dangerous_code": True}
    )
    
    # Chat section
    st.markdown("## 💬 AI Assistant")
    st.markdown("Ask questions about your data and get instant visualizations and insights!")
    
    # Chat input with better styling
    prompt = st.chat_input("Ask me anything about your data...")
    
    if prompt:
        # Show user message
        with st.chat_message("user"):
            st.write(prompt)
            
        # Show AI response
        with st.chat_message("assistant"):
            # Check if the prompt is asking for visualization/analysis
            visualization_keywords = [
                'grafik', 'plot', 'chart', 'histogram', 'scatter', 'bar', 'line', 'pie',
                'görselleştir', 'çiz', 'göster', 'analiz', 'dağılım', 'correlation',
                'heatmap', 'boxplot', 'visualization', 'visualize', 'draw', 'wordcloud',
                'word cloud', 'kelime bulutu', 'seaborn', 'sns', 'plotly', 'bokeh',
                'interactive', 'dashboard', 'model', 'predict', 'classification', 
                'regression', 'machine learning', 'ml', 'sklearn', 'train', 'accuracy',
                'random forest', 'svm', 'xgboost', 'lightgbm', 'cross validation'
            ]
            
            needs_code = any(keyword in prompt.lower() for keyword in visualization_keywords)
            
            if needs_code:
                # Check if it's ML related
                ml_keywords = ['model', 'predict', 'classification', 'regression', 'machine learning', 'ml', 'sklearn', 'train', 'accuracy']
                is_ml_request = any(ml_word in prompt.lower() for ml_word in ml_keywords)
                
                if is_ml_request:
                    # For ML requests, add preprocessing instructions
                    modified_prompt = f"""
{prompt}

IMPORTANT: Provide only the Python code in a markdown code block (```python).
- Automatically preprocess the data for machine learning
- Convert date columns to numerical (year, month, etc.)
- Handle categorical variables with LabelEncoder or pd.get_dummies()
- Drop or fill missing values appropriately
- Include train_test_split, model training, and evaluation
- Show model performance metrics
- The dataframe is already available as 'df'.
"""
                else:
                    # For visualization requests
                    modified_prompt = f"""
{prompt}

IMPORTANT: Provide only the Python code in a markdown code block (```python). 
Do not provide explanations or descriptions, just the working code.
The dataframe is already available as 'df'.
"""
            else:
                modified_prompt = prompt
                
            with st.spinner("Thinking..."):
                response = agent.invoke(modified_prompt)
                
                # Check if the response contains code and execute it
                code_blocks = extract_code_from_text(response['output'])
                
                if code_blocks and needs_code:
                    # For visualization requests, show only the plot, not the code
                    for i, code in enumerate(code_blocks):
                        
                        # Choose appropriate spinner message
                        if any(ml_word in code.lower() for ml_word in ["model", "fit(", "predict", "sklearn", "xgb", "lgb", "train"]):
                            spinner_msg = "Training model..."
                        else:
                            spinner_msg = "Creating visualization..."
                            
                        with st.spinner(spinner_msg):
                            try:
                                # Check if code contains matplotlib/plotting/ML
                                if any(keyword in code.lower() for keyword in ["matplotlib", "plt.", "plot", "hist", "scatter", "bar", "wordcloud", "sns.", "seaborn", "px.", "plotly", "go.", "bokeh", "sklearn", "xgb", "lgb", "model", "fit(", "predict"]):
                                    # Create execution environment with dataframe
                                    exec_globals = {
                                        "df": df, 
                                        "pd": pd, 
                                        "plt": plt,
                                        "numpy": None,
                                        "np": None,
                                        "WordCloud": None,
                                        "sns": None,
                                        "seaborn": None,
                                        "px": None,
                                        "plotly": None,
                                        "go": None,
                                        "bokeh": None,
                                        "sklearn": None,
                                        "xgb": None,
                                        "lgb": None,
                                        "joblib": None
                                    }
                                    
                                    # Try to import all libraries
                                    try:
                                        import numpy as np
                                        exec_globals["np"] = np
                                        exec_globals["numpy"] = np
                                    except ImportError:
                                        pass
                                    
                                    try:
                                        from wordcloud import WordCloud
                                        exec_globals["WordCloud"] = WordCloud
                                    except ImportError:
                                        pass
                                    
                                    try:
                                        import seaborn as sns
                                        exec_globals["sns"] = sns
                                        exec_globals["seaborn"] = sns
                                    except ImportError:
                                        pass
                                    
                                    try:
                                        import plotly.express as px
                                        import plotly.graph_objects as go
                                        import plotly
                                        exec_globals["px"] = px
                                        exec_globals["go"] = go
                                        exec_globals["plotly"] = plotly
                                    except ImportError:
                                        pass
                                    
                                    try:
                                        import bokeh
                                        exec_globals["bokeh"] = bokeh
                                    except ImportError:
                                        pass
                                    
                                    try:
                                        import sklearn
                                        from sklearn.model_selection import train_test_split
                                        from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
                                        from sklearn.linear_model import LinearRegression, LogisticRegression
                                        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
                                        from sklearn.svm import SVC, SVR
                                        from sklearn.preprocessing import StandardScaler, LabelEncoder
                                        
                                        exec_globals["sklearn"] = sklearn
                                        exec_globals["train_test_split"] = train_test_split
                                        exec_globals["accuracy_score"] = accuracy_score
                                        exec_globals["mean_squared_error"] = mean_squared_error
                                        exec_globals["classification_report"] = classification_report
                                        exec_globals["LinearRegression"] = LinearRegression
                                        exec_globals["LogisticRegression"] = LogisticRegression
                                        exec_globals["RandomForestClassifier"] = RandomForestClassifier
                                        exec_globals["RandomForestRegressor"] = RandomForestRegressor
                                        exec_globals["SVC"] = SVC
                                        exec_globals["SVR"] = SVR
                                        exec_globals["StandardScaler"] = StandardScaler
                                        exec_globals["LabelEncoder"] = LabelEncoder
                                    except ImportError:
                                        pass
                                    
                                    try:
                                        import xgboost as xgb
                                        exec_globals["xgb"] = xgb
                                    except ImportError:
                                        pass
                                    
                                    try:
                                        import lightgbm as lgb
                                        exec_globals["lgb"] = lgb
                                    except ImportError:
                                        pass
                                    
                                    try:
                                        import joblib
                                        exec_globals["joblib"] = joblib
                                    except ImportError:
                                        pass
                                    
                                    # Execute visualization/ML code locally
                                    if "plotly" in code.lower() or "px." in code.lower() or "go." in code.lower():
                                        # For Plotly, capture the figure
                                        exec_locals = {}
                                        exec(code, exec_globals, exec_locals)
                                        
                                        # Look for plotly figure in locals
                                        for var_name, var_value in exec_locals.items():
                                            if hasattr(var_value, 'show') and 'plotly' in str(type(var_value)):
                                                st.plotly_chart(var_value, use_container_width=True)
                                                break
                                    else:
                                        # For matplotlib/seaborn and ML
                                        import io
                                        import contextlib
                                        
                                        # Capture print outputs
                                        f = io.StringIO()
                                        with contextlib.redirect_stdout(f):
                                            exec(code, exec_globals)
                                        
                                        # Show plot if created
                                        if plt.get_fignums():
                                            st.pyplot(plt.gcf())
                                            plt.close()
                                        
                                        # Show captured metrics immediately after plot
                                        output = f.getvalue().strip()
                                        if output:
                                            st.markdown("**Model Performance:**")
                                            
                                            # Parse metrics
                                            lines = [line.strip() for line in output.split('\n') if line.strip()]
                                            metrics = []
                                            
                                            for line in lines:
                                                if ':' in line:
                                                    metric_name, metric_value = line.split(':', 1)
                                                    metric_name = metric_name.strip()
                                                    metric_value = metric_value.strip()
                                                    
                                                    # Format numbers
                                                    try:
                                                        metric_value = f"{float(metric_value):.4f}"
                                                    except:
                                                        pass
                                                    
                                                    metrics.append((metric_name, metric_value))
                                            
                                            # Display metrics in columns
                                            if metrics:
                                                num_cols = min(len(metrics), 3)
                                                cols = st.columns(num_cols)
                                                
                                                for i, (name, value) in enumerate(metrics):
                                                    with cols[i % num_cols]:
                                                        st.metric(name, value)
                                            
                                            # Show any non-metric text
                                            for line in lines:
                                                if ':' not in line:
                                                    st.text(line)
                                
                                else:
                                    # For non-plot code, use Piston API
                                    result = run_code_with_piston(code)
                                    formatted_result = format_execution_result(result)
                                    if formatted_result:
                                        st.markdown(formatted_result)
                                        
                            except Exception as e:
                                st.error(f"❌ Could not create visualization: {str(e)}")
                
                else:
                    # For non-visualization requests, show the normal response
                    st.success("✅ Answer:")
                    st.markdown(f"> {response['output']}")
