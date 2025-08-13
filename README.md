# 🤖 Machine Learning Dashboard

A comprehensive interactive web application built with Streamlit for exploring datasets, visualizing data patterns, training machine learning models, and making predictions.

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-239120?style=for-the-badge&logo=plotly&logoColor=white)

## 🌟 Features

### 📊 **Data Exploration**
- Dataset overview with statistics and metadata
- Interactive data filtering with sliders
- Sample data display with customizable rows
- Statistical summaries and data type information
- Target class distribution analysis

### 📈 **Interactive Visualizations**
- **Scatter Plots**: Interactive plots with customizable axes and color coding
- **Histograms**: Feature distribution analysis with adjustable bins
- **Box Plots**: Feature comparison across different target classes
- **Correlation Heatmaps**: Feature correlation matrix visualization
- **Pair Plots**: Multi-feature relationship analysis

### 🔮 **Model Predictions**
- Real-time predictions with user input
- Support for multiple ML algorithms:
  - Random Forest Classifier
  - Logistic Regression
  - Support Vector Machine (SVM)
- Prediction confidence and probability display
- Interactive input widgets with validation
- Quick sample data selection

### 📋 **Model Performance Analysis**
- Comprehensive model comparison
- Detailed performance metrics (Accuracy, Precision, Recall, F1-Score)
- Interactive confusion matrices
- Feature importance visualization (Random Forest)
- Per-class performance breakdown

## 🗂️ **Supported Datasets**

The application comes with three classic machine learning datasets:

| Dataset | Samples | Features | Classes | Description |
|---------|---------|----------|---------|-------------|
| **Iris** | 150 | 4 | 3 | Flower species classification |
| **Wine** | 178 | 13 | 3 | Wine quality classification |
| **Breast Cancer** | 569 | 30 | 2 | Cancer diagnosis classification |

## 🚀 **Quick Start**

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd ml-dashboard
   ```

2. **Create virtual environment (recommended):**
   ```bash
   python -m venv streamlit_env
   
   # Activate virtual environment
   # Windows:
   streamlit_env\Scripts\activate
   # macOS/Linux:
   source streamlit_env/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application:**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser:**
   - The app will automatically open at `http://localhost:8501`
   - If not, navigate to the URL shown in your terminal

## 📋 **Requirements**

```txt
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.15.0
scikit-learn>=1.3.0
```

For the complete list, see [`requirements.txt`](requirements.txt).

## 🎯 **How to Use**

### 1. **Navigation**
Use the sidebar to navigate between different sections:
- 🏠 **Home**: Overview and app information
- 📊 **Data Exploration**: Dataset analysis and filtering
- 📈 **Visualizations**: Interactive charts and plots
- 🔮 **Model Predictions**: Make predictions with custom inputs
- 📋 **Model Performance**: Detailed model evaluation

### 2. **Dataset Selection**
- Choose from Iris, Wine, or Breast Cancer datasets using the sidebar dropdown
- Dataset information updates automatically across all sections

### 3. **Data Exploration**
- View dataset statistics and sample data
- Use interactive filters to explore data subsets
- Analyze feature distributions and correlations

### 4. **Visualizations**
- Select different chart types from the dropdown
- Customize axes, colors, and parameters
- All charts are interactive with zoom and hover capabilities

### 5. **Making Predictions**
- Choose a trained model from the dropdown
- Input feature values using the number inputs
- Click "Make Prediction" to see results
- View prediction confidence and class probabilities

### 6. **Model Performance**
- Compare accuracy across different models
- View detailed metrics and confusion matrices
- Analyze feature importance (for Random Forest)

## 🏗️ **Project Structure**

```
ml-dashboard/
│
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
│
├── data/                 # (Optional) Additional datasets
├── models/               # (Optional) Saved model files
├── utils/                # (Optional) Helper functions
└── assets/               # (Optional) Images and static files
```

## 🛠️ **Technical Details**

### **Machine Learning Pipeline**
- **Data Preprocessing**: Automatic feature scaling for SVM and Logistic Regression
- **Model Training**: 80/20 train-test split with fixed random state
- **Cross-validation**: Built-in model evaluation and comparison
- **Caching**: Streamlit caching for improved performance

### **Key Technologies**
| Technology | Purpose |
|------------|---------|
| **Streamlit** | Web application framework |
| **Pandas** | Data manipulation and analysis |
| **Scikit-learn** | Machine learning algorithms |
| **Plotly** | Interactive visualizations |
| **Matplotlib** | Static plotting |
| **NumPy** | Numerical computing |

### **Performance Features**
- ✅ Session state management for data persistence
- ✅ Caching for expensive operations
- ✅ Loading states for better user experience
- ✅ Error handling and input validation

## 🎨 **Customization**

### **Adding New Datasets**
To add your own dataset, modify the `load_dataset()` function in `app.py`:

```python
def load_dataset(dataset_name):
    if dataset_name == "Your Dataset":
        # Load your data
        df = pd.read_csv("your_data.csv")
        # Process and return
        return df, feature_names, target_names
```

### **Adding New Models**
Extend the `train_models()` function to include additional algorithms:

```python
models = {
    'Random Forest': RandomForestClassifier(),
    'Logistic Regression': LogisticRegression(),
    'SVM': SVC(probability=True),
    'Your Model': YourClassifier()  # Add here
}
```

### **Styling**
Customize the appearance by modifying the CSS in the `st.markdown()` sections:

```python
st.markdown("""
<style>
.your-custom-class {
    background-color: #your-color;
    padding: 1rem;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)
```

## 🐛 **Troubleshooting**

### **Common Issues**

#### 1. Module not found errors:
```bash
pip install -r requirements.txt
```

#### 2. Port already in use:
```bash
streamlit run app.py --server.port 8502
```

#### 3. Memory issues with large datasets:
- Use data sampling or chunking
- Increase available RAM or use smaller datasets

### **Performance Tips**
- 💡 Close unused browser tabs to free memory
- 💡 Use the data filtering options to work with smaller subsets
- 💡 Clear browser cache if visualizations don't load
- 💡 Use Chrome or Firefox for better performance

## 📊 **Screenshots**

### Home Page
> Dashboard overview with dataset metrics

### Data Exploration
> Interactive filtering and statistical analysis

### Visualizations
> Multiple chart types with customization options

### Model Predictions
> Real-time predictions with confidence scores

### Model Performance
> Comprehensive evaluation metrics and comparisons

## 📈 **Future Enhancements**

- [ ] Support for custom dataset uploads
- [ ] Advanced model hyperparameter tuning
- [ ] Time series analysis capabilities
- [ ] Model export and import functionality
- [ ] Advanced visualization types (3D plots, animations)
- [ ] Integration with cloud ML services
- [ ] Automated model selection
- [ ] Real-time data streaming support
- [ ] A/B testing framework for models
- [ ] Mobile-responsive design

## 🤝 **Contributing**

We welcome contributions! Here's how to get started:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. **Commit your changes**
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```
4. **Push to the branch**
   ```bash
   git push origin feature/AmazingFeature
   ```
5. **Open a Pull Request**

### **Development Setup**
```bash
# Clone your fork
git clone https://github.com/yourusername/ml-dashboard.git
cd ml-dashboard

# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # If you have dev requirements

# Run tests
python -m pytest tests/

# Run the app in development mode
streamlit run app.py --server.runOnSave true
```

## 📝 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024 ML Dashboard

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
```

## 📧 **Contact & Support**

- **Developer**: Your Name
- **Email**: your.email@example.com
- **GitHub**: [@yourusername](https://github.com/yourusername)
- **Project Link**: [ML Dashboard Repository](https://github.com/yourusername/ml-dashboard)

### **Getting Help**
- 🐛 **Bug Reports**: [Open an issue](https://github.com/yourusername/ml-dashboard/issues)
- 💡 **Feature Requests**: [Request a feature](https://github.com/yourusername/ml-dashboard/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/yourusername/ml-dashboard/discussions)

## 🙏 **Acknowledgments**

- [Streamlit](https://streamlit.io/) for the amazing web framework
- [Scikit-learn](https://scikit-learn.org/) for machine learning algorithms
- [Plotly](https://plotly.com/) for interactive visualizations
- [UCI ML Repository](https://archive.ics.uci.edu/ml/) for the datasets
- The open-source community for inspiration and tools

## ⭐ **Star History**

[![Star History Chart](https://api.star-history.com/svg?repos=yourusername/ml-dashboard&type=Date)](https://star-history.com/#yourusername/ml-dashboard&Date)

---

## 🚀 **Get Started Now!**

```bash
git clone <repository-url>
cd ml-dashboard
pip install -r requirements.txt
streamlit run app.py
```

**🎉 Enjoy exploring machine learning with this interactive dashboard!**

---

<div align="center">

**Made with ❤️ and Python**

[![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)

</div>