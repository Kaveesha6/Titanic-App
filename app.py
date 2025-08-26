import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns  # Optional - only needed for advanced styling
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="ML Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1f77b4;
    text-align: center;
    padding: 1rem 0;
    border-bottom: 3px solid #1f77b4;
    margin-bottom: 2rem;
}
.section-header {
    font-size: 1.5rem;
    color: #ff7f0e;
    margin: 1rem 0;
    padding: 0.5rem 0;
    border-left: 4px solid #ff7f0e;
    padding-left: 1rem;
}
.metric-container {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 10px;
    margin: 0.5rem 0;
}
.info-box {
    background-color: #2c3e50;
    color: #ffffff;
    padding: 1rem;
    border-radius: 5px;
    border-left: 4px solid #1f77b4;
    margin: 1rem 0;
}
.info-box h3 {
    color: #3498db;
    margin-top: 0;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False

# Helper functions
@st.cache_data
def load_dataset(dataset_name):
    """Load and return the selected dataset"""
    if dataset_name == "Iris":
        data = load_iris()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        df['target_name'] = df['target'].map({i: name for i, name in enumerate(data.target_names)})
        return df, data.feature_names, data.target_names
    elif dataset_name == "Wine":
        data = load_wine()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        df['target_name'] = df['target'].map({i: name for i, name in enumerate(data.target_names)})
        return df, data.feature_names, data.target_names
    elif dataset_name == "Breast Cancer":
        data = load_breast_cancer()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        df['target_name'] = df['target'].map({i: name for i, name in enumerate(data.target_names)})
        return df, data.feature_names, data.target_names

@st.cache_data
def train_models(X_train, X_test, y_train, y_test):
    """Train multiple models and return results"""
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'SVM': SVC(random_state=42, probability=True)
    }
    
    results = {}
    trained_models = {}
    
    # Scale features for SVM and Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    for name, model in models.items():
        if name in ['SVM', 'Logistic Regression']:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled) if hasattr(model, 'predict_proba') else None
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        
        accuracy = accuracy_score(y_test, y_pred)
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        trained_models[name] = model
    
    return results, trained_models, scaler

# Main App
def main():
    # Title and Description
    st.markdown('<h1 class="main-header">🤖 Machine Learning Dashboard</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <h3>Welcome to the ML Dashboard!</h3>
    <p>This interactive application allows you to explore datasets, visualize data patterns, 
    train machine learning models, and make predictions. Use the sidebar to navigate through different sections.</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar Navigation
    st.sidebar.title("📋 Navigation")
    page = st.sidebar.selectbox(
        "Choose a section:",
        ["🏠 Home", "📊 Data Exploration", "📈 Visualizations", "🔮 Model Predictions", "📋 Model Performance"]
    )
    
    # Dataset Selection
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📂 Dataset Selection")
    dataset_choice = st.sidebar.selectbox(
        "Select Dataset:",
        ["Iris", "Wine", "Breast Cancer"],
        help="Choose from three classic machine learning datasets"
    )
    
    # Load data
    try:
        with st.spinner("Loading dataset..."):
            df, feature_names, target_names = load_dataset(dataset_choice)
            st.session_state.df = df
            st.session_state.feature_names = feature_names
            st.session_state.target_names = target_names
            st.session_state.current_dataset = dataset_choice  # Store current dataset name
            st.session_state.data_loaded = True
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return
    
    # Page Content
    if page == "🏠 Home":
        show_home_page()
    elif page == "📊 Data Exploration":
        show_data_exploration()
    elif page == "📈 Visualizations":
        show_visualizations()
    elif page == "🔮 Model Predictions":
        show_model_predictions()
    elif page == "📋 Model Performance":
        show_model_performance()

def show_home_page():
    st.markdown('<h2 class="section-header">🏠 Home</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📋 App Features")
        st.markdown("""
        - **Data Exploration**: View dataset statistics and sample data
        - **Interactive Visualizations**: Multiple chart types with filters
        - **Model Training**: Compare different ML algorithms
        - **Real-time Predictions**: Make predictions with custom inputs
        - **Performance Analysis**: Detailed model evaluation metrics
        """)
    
    with col2:
        st.markdown("### 📊 Current Dataset")
        if st.session_state.data_loaded:
            df = st.session_state.df
            # Get the current dataset name from session state or sidebar
            current_dataset = st.session_state.get('current_dataset', 'Dataset Selected')
            st.metric("Dataset", current_dataset)
            st.metric("Samples", df.shape[0])
            st.metric("Features", len(st.session_state.feature_names))
            st.metric("Classes", len(st.session_state.target_names))

def show_data_exploration():
    st.markdown('<h2 class="section-header">📊 Data Exploration</h2>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.error("Please select a dataset first!")
        return
    
    df = st.session_state.df
    
    # Dataset Overview
    st.markdown("### 📋 Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Samples", df.shape[0])
    with col2:
        st.metric("Features", len(st.session_state.feature_names))
    with col3:
        st.metric("Classes", len(st.session_state.target_names))
    with col4:
        st.metric("Missing Values", df.isnull().sum().sum())
    
    # Data Types and Info
    st.markdown("### 🔍 Data Types")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Feature Data Types:**")
        feature_df = df[st.session_state.feature_names]
        st.dataframe(feature_df.dtypes.to_frame(name='Data Type'))
    
    with col2:
        st.markdown("**Target Distribution:**")
        target_counts = df['target_name'].value_counts()
        st.dataframe(target_counts.to_frame(name='Count'))
    
    # Sample Data
    st.markdown("### 👀 Sample Data")
    n_samples = st.slider("Number of samples to display:", 5, 20, 10)
    st.dataframe(df.head(n_samples))
    
    # Interactive Data Filtering
    st.markdown("### 🔧 Interactive Data Filtering")
    
    # Feature selection for filtering
    selected_features = st.multiselect(
        "Select features to filter by:",
        st.session_state.feature_names[:5],  # Limit to first 5 for better UX
        default=st.session_state.feature_names[:2]
    )
    
    if selected_features:
        filtered_df = df.copy()
        
        for feature in selected_features:
            min_val, max_val = float(df[feature].min()), float(df[feature].max())
            range_val = st.slider(
                f"Filter by {feature}:",
                min_val, max_val, (min_val, max_val),
                key=f"filter_{feature}"
            )
            filtered_df = filtered_df[
                (filtered_df[feature] >= range_val[0]) & 
                (filtered_df[feature] <= range_val[1])
            ]
        
        st.markdown(f"**Filtered Dataset:** {len(filtered_df)} samples")
        st.dataframe(filtered_df.head())
    
    # Statistical Summary
    st.markdown("### 📈 Statistical Summary")
    st.dataframe(df[st.session_state.feature_names].describe())

def show_visualizations():
    st.markdown('<h2 class="section-header">📈 Visualizations</h2>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.error("Please select a dataset first!")
        return
    
    df = st.session_state.df
    
    # Visualization Controls
    st.markdown("### 🎛️ Visualization Controls")
    col1, col2 = st.columns(2)
    
    with col1:
        chart_type = st.selectbox(
            "Select Chart Type:",
            ["Scatter Plot", "Histogram", "Box Plot", "Correlation Heatmap", "Pair Plot"]
        )
    
    with col2:
        color_feature = st.selectbox(
            "Color by:",
            ["target_name"] + list(st.session_state.feature_names[:3])
        )
    
    # Chart 1: Interactive Scatter Plot
    if chart_type == "Scatter Plot":
        st.markdown("### 📊 Interactive Scatter Plot")
        
        col1, col2 = st.columns(2)
        with col1:
            x_feature = st.selectbox("X-axis:", st.session_state.feature_names, key="scatter_x")
        with col2:
            y_feature = st.selectbox("Y-axis:", st.session_state.feature_names, index=1, key="scatter_y")
        
        fig = px.scatter(
            df, x=x_feature, y=y_feature, 
            color=color_feature,
            title=f"Scatter Plot: {x_feature} vs {y_feature}",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Chart 2: Histogram
    elif chart_type == "Histogram":
        st.markdown("### 📊 Feature Distribution")
        
        selected_feature = st.selectbox("Select Feature:", st.session_state.feature_names)
        bins = st.slider("Number of bins:", 10, 50, 30)
        
        fig = px.histogram(
            df, x=selected_feature, 
            color='target_name',
            nbins=bins,
            title=f"Distribution of {selected_feature}",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Chart 3: Box Plot
    elif chart_type == "Box Plot":
        st.markdown("### 📊 Feature Comparison by Target")
        
        selected_features = st.multiselect(
            "Select Features:", 
            st.session_state.feature_names, 
            default=st.session_state.feature_names[:4]
        )
        
        if selected_features:
            fig = go.Figure()
            for feature in selected_features:
                for target in df['target_name'].unique():
                    subset = df[df['target_name'] == target][feature]
                    fig.add_trace(go.Box(
                        y=subset,
                        name=f"{target}",
                        boxmean='sd'
                    ))
            
            fig.update_layout(
                title="Box Plot Comparison",
                yaxis_title="Values",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Chart 4: Correlation Heatmap
    elif chart_type == "Correlation Heatmap":
        st.markdown("### 📊 Feature Correlation Matrix")
        
        correlation_matrix = df[st.session_state.feature_names].corr()
        
        fig = px.imshow(
            correlation_matrix,
            title="Feature Correlation Heatmap",
            color_continuous_scale="RdBu_r",
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Chart 5: Pair Plot (using matplotlib/seaborn)
    elif chart_type == "Pair Plot":
        st.markdown("### 📊 Pair Plot Analysis")
        
        selected_features = st.multiselect(
            "Select Features (max 4 for performance):", 
            st.session_state.feature_names, 
            default=st.session_state.feature_names[:3]
        )
        
        if selected_features and len(selected_features) <= 4:
            with st.spinner("Generating pair plot..."):
                fig, axes = plt.subplots(len(selected_features), len(selected_features), 
                                       figsize=(12, 12))
                
                for i, feature1 in enumerate(selected_features):
                    for j, feature2 in enumerate(selected_features):
                        ax = axes[i][j] if len(selected_features) > 1 else axes
                        
                        if i == j:
                            # Histogram on diagonal
                            for target in df['target_name'].unique():
                                subset = df[df['target_name'] == target]
                                ax.hist(subset[feature1], alpha=0.6, label=target)
                            ax.set_xlabel(feature1)
                            ax.legend()
                        else:
                            # Scatter plot off diagonal
                            for target in df['target_name'].unique():
                                subset = df[df['target_name'] == target]
                                ax.scatter(subset[feature2], subset[feature1], 
                                         label=target, alpha=0.6)
                            ax.set_xlabel(feature2)
                            ax.set_ylabel(feature1)
                
                plt.tight_layout()
                st.pyplot(fig)
        elif len(selected_features) > 4:
            st.warning("Please select maximum 4 features for optimal performance.")

def show_model_predictions():
    st.markdown('<h2 class="section-header">🔮 Model Predictions</h2>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.error("Please select a dataset first!")
        return
    
    df = st.session_state.df
    
    # Train models if not already trained
    if not st.session_state.models_trained:
        with st.spinner("Training models... This may take a moment."):
            X = df[st.session_state.feature_names]
            y = df['target']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            results, trained_models, scaler = train_models(X_train, X_test, y_train, y_test)
            
            st.session_state.results = results
            st.session_state.trained_models = trained_models
            st.session_state.scaler = scaler
            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
            st.session_state.models_trained = True
    
    # Model Selection
    st.markdown("### 🤖 Model Selection")
    selected_model = st.selectbox(
        "Choose a model for predictions:",
        list(st.session_state.trained_models.keys()),
        help="Select which trained model to use for predictions"
    )
    
    # Input Section
    st.markdown("### 📝 Input Features")
    st.markdown("Enter values for each feature to get a prediction:")
    
    # Create input widgets for each feature
    input_values = {}
    
    # Organize inputs in columns for better layout
    n_cols = 2
    cols = st.columns(n_cols)
    
    for i, feature in enumerate(st.session_state.feature_names):
        col_idx = i % n_cols
        
        with cols[col_idx]:
            min_val = float(df[feature].min())
            max_val = float(df[feature].max())
            mean_val = float(df[feature].mean())
            
            input_values[feature] = st.number_input(
                f"{feature}:",
                min_value=min_val * 0.5,  # Allow some range beyond observed values
                max_value=max_val * 1.5,
                value=mean_val,
                step=(max_val - min_val) / 100,
                key=f"input_{feature}",
                help=f"Range in dataset: {min_val:.2f} - {max_val:.2f}"
            )
    
    # Prediction Section
    st.markdown("### 🎯 Prediction Results")
    
    if st.button("🔮 Make Prediction", type="primary"):
        try:
            # Prepare input data
            input_data = np.array([list(input_values.values())])
            
            # Get selected model
            model = st.session_state.trained_models[selected_model]
            
            # Scale input if needed
            if selected_model in ['SVM', 'Logistic Regression']:
                input_data = st.session_state.scaler.transform(input_data)
            
            # Make prediction
            prediction = model.predict(input_data)[0]
            prediction_proba = None
            
            if hasattr(model, 'predict_proba'):
                if selected_model in ['SVM', 'Logistic Regression']:
                    prediction_proba = model.predict_proba(input_data)[0]
                else:
                    X_scaled = st.session_state.scaler.transform(np.array([list(input_values.values())]))
                    prediction_proba = model.predict_proba(np.array([list(input_values.values())]))[0]
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                predicted_class = st.session_state.target_names[prediction]
                st.success(f"**Predicted Class:** {predicted_class}")
                
                # Show confidence if available
                if prediction_proba is not None:
                    confidence = prediction_proba[prediction] * 100
                    st.metric("Confidence", f"{confidence:.1f}%")
            
            with col2:
                if prediction_proba is not None:
                    st.markdown("**Class Probabilities:**")
                    prob_df = pd.DataFrame({
                        'Class': st.session_state.target_names,
                        'Probability': [f"{prob*100:.1f}%" for prob in prediction_proba]
                    })
                    st.dataframe(prob_df, hide_index=True)
            
            # Probability visualization
            if prediction_proba is not None:
                fig = px.bar(
                    x=st.session_state.target_names,
                    y=prediction_proba * 100,
                    title="Prediction Probabilities",
                    labels={'x': 'Class', 'y': 'Probability (%)'}
                )
                fig.update_traces(marker_color=['red' if i == prediction else 'lightblue' 
                                              for i in range(len(prediction_proba))])
                st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
    
    # Sample Predictions Section
    st.markdown("### 📋 Quick Sample Predictions")
    st.markdown("Try some sample data points from the dataset:")
    
    sample_idx = st.selectbox("Select a sample:", range(min(10, len(df))))
    
    if st.button("Use Sample Data"):
        sample_row = df.iloc[sample_idx]
        
        # Update input values
        for feature in st.session_state.feature_names:
            st.session_state[f"input_{feature}"] = float(sample_row[feature])
        
        st.success(f"Loaded sample {sample_idx}. Click 'Make Prediction' to see results!")
        st.markdown(f"**Actual Class:** {sample_row['target_name']}")

def show_model_performance():
    st.markdown('<h2 class="section-header">📋 Model Performance</h2>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.error("Please select a dataset first!")
        return
    
    # Train models if not already trained
    if not st.session_state.models_trained:
        with st.spinner("Training models for evaluation..."):
            df = st.session_state.df
            X = df[st.session_state.feature_names]
            y = df['target']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            results, trained_models, scaler = train_models(X_train, X_test, y_train, y_test)
            
            st.session_state.results = results
            st.session_state.trained_models = trained_models
            st.session_state.scaler = scaler
            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
            st.session_state.models_trained = True
    
    # Model Comparison
    st.markdown("### 🏆 Model Comparison")
    
    # Create comparison metrics
    comparison_data = []
    for model_name, result in st.session_state.results.items():
        comparison_data.append({
            'Model': model_name,
            'Accuracy': f"{result['accuracy']:.4f}",
            'Accuracy (%)': f"{result['accuracy']*100:.2f}%"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, hide_index=True)
    
    # Accuracy comparison chart
    fig = px.bar(
        comparison_df, 
        x='Model', 
        y=[float(acc.strip('%')) for acc in comparison_df['Accuracy (%)']],
        title="Model Accuracy Comparison",
        labels={'y': 'Accuracy (%)'}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed Model Analysis
    st.markdown("### 🔍 Detailed Model Analysis")
    
    selected_model = st.selectbox(
        "Select model for detailed analysis:",
        list(st.session_state.results.keys())
    )
    
    result = st.session_state.results[selected_model]
    
    # Performance Metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Classification Metrics")
        
        # Extract metrics from classification report
        report = result['classification_report']
        
        # Display macro and weighted averages
        st.metric("Accuracy", f"{result['accuracy']:.4f}")
        st.metric("Macro Avg Precision", f"{report['macro avg']['precision']:.4f}")
        st.metric("Macro Avg Recall", f"{report['macro avg']['recall']:.4f}")
        st.metric("Macro Avg F1-Score", f"{report['macro avg']['f1-score']:.4f}")
    
    with col2:
        st.markdown("#### 🎯 Per-Class Performance")
        
        # Create per-class metrics table
        class_metrics = []
        for i, class_name in enumerate(st.session_state.target_names):
            if str(i) in report:
                class_metrics.append({
                    'Class': class_name,
                    'Precision': f"{report[str(i)]['precision']:.4f}",
                    'Recall': f"{report[str(i)]['recall']:.4f}",
                    'F1-Score': f"{report[str(i)]['f1-score']:.4f}",
                    'Support': report[str(i)]['support']
                })
        
        if class_metrics:
            class_df = pd.DataFrame(class_metrics)
            st.dataframe(class_df, hide_index=True)
    
    # Confusion Matrix
    st.markdown("#### 🔲 Confusion Matrix")
    
    cm = result['confusion_matrix']
    
    # Create confusion matrix heatmap
    fig = px.imshow(
        cm,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=st.session_state.target_names,
        y=st.session_state.target_names,
        title=f"Confusion Matrix - {selected_model}",
        color_continuous_scale="Blues",
        text_auto=True
    )
    
    fig.update_layout(
        xaxis_title="Predicted Class",
        yaxis_title="Actual Class",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature Importance (for applicable models)
    if selected_model == 'Random Forest':
        st.markdown("#### 🌟 Feature Importance")
        
        model = st.session_state.trained_models[selected_model]
        importances = model.feature_importances_
        
        importance_df = pd.DataFrame({
            'Feature': st.session_state.feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        fig = px.bar(
            importance_df,
            x='Importance',
            y='Feature',
            orientation='h',
            title="Feature Importance (Random Forest)",
            labels={'Importance': 'Importance Score'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Model Training Information
    st.markdown("### ℹ️ Training Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Training Samples", len(st.session_state.X_train))
    with col2:
        st.metric("Testing Samples", len(st.session_state.X_test))
    with col3:
        st.metric("Train/Test Split", "80/20")

if __name__ == "__main__":
    main()