# 🎯 The Whole Truth Foods - Product Success Prediction System

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Live Demo](https://img.shields.io/badge/Live-Demo-success.svg)](https://ds-dashboard-sem7.streamlit.app/)

> **A comprehensive machine learning system for predicting product success using classification models, SHAP/LIME explainability, fairness analysis, and interactive deployment.**

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Live Demo](#-live-demo)
- [Key Features](#-key-features)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Technologies Used](#-technologies-used)
- [Running the Application](#-running-the-application)
- [API Usage](#-api-usage)
- [Docker Deployment](#-docker-deployment)
- [Testing](#-testing)
- [Documentation](#-documentation)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

This project develops an end-to-end machine learning pipeline to predict product success for **The Whole Truth Foods**, a health-focused food brand. The system analyzes 13+ product features to forecast whether a product will succeed or fail in the market.

### 🎓 Project Context

This comprehensive data science project completes **8 experiments** covering the full ML lifecycle:

1. **Introduction & Project Setup** - Dataset exploration and objectives
2. **Data Cleaning & Preprocessing** - Missing values, outliers, feature engineering
3. **Exploratory Data Analysis** - Statistical analysis and visualizations
4. **Machine Learning Modeling** - Model training, tuning, and MLflow tracking
5. **Explainability & Fairness** - SHAP/LIME analysis and bias detection
6. **API Deployment** - FastAPI REST endpoints with containerization
7. **Dashboard & Responsible AI** - Interactive UI and ethical AI practices
8. **Final Portfolio** - Complete documentation and deployment

---

## 🚀 Live Demo

### 🎨 Interactive Dashboard

**🔗 Live Application:** [https://ds-dashboard-sem7.streamlit.app/](https://ds-dashboard-sem7.streamlit.app/)

Explore all 7 experiment pages:
- 📊 Data cleaning and preprocessing results
- 📈 Interactive EDA visualizations
- 🤖 Multi-model comparison interface
- � SHAP & LIME explainability tools
- ⚖️ Fairness and bias analysis
- 🎯 Real-time predictions with explanations

### 📚 API Documentation

**FastAPI Endpoints:** Available for local deployment
- `GET /` - API information
- `GET /health` - Health check
- `POST /predict` - Single prediction
- `POST /batch-predict` - Batch predictions
- `GET /model-info` - Model details

### 🎓 Project Context

This comprehensive data science project completes **8 experiments** covering the full ML lifecycle:

1. Introduction & Project Setup
2. Data Cleaning & Preprocessing
3. Exploratory Data Analysis (EDA)
4. Machine Learning Modeling
5. Explainability & Fairness
6. API Deployment & Containerization
7. Dashboard & Responsible AI
8. Final Portfolio & Documentation

---

## ✨ Key Features

### 🤖 Machine Learning Pipeline
- Multiple classification models (KNN, ANN/DNN, LDA, Naive Bayes)
- Hyperparameter tuning with GridSearchCV
- MLflow experiment tracking
- Dynamic feature importance analysis
- Cross-validation and performance metrics

### 🔍 Model Explainability
- **SHAP Analysis** - Global and local feature explanations
- **LIME Analysis** - Model-agnostic interpretability
- Interactive visualizations (waterfall plots, force plots)
- Feature importance rankings

### ⚖️ Responsible AI
- Bias detection across demographics (gender, age, categories)
- Fairness metrics and analysis
- Privacy-preserving design (no PII collection)
- Comprehensive ethical AI documentation

### 🎨 Interactive Dashboard
- 7 experiment pages covering full ML lifecycle
- Multi-model comparison interface
- Real-time predictions with explanations
- Interactive data visualizations with Plotly
- Deployed on Streamlit Cloud

### 🔌 REST API
- FastAPI microservice architecture
- Multiple endpoints (predict, batch, health, info)
- Automatic Swagger/ReDoc documentation
- Pydantic data validation
- Docker containerization support

### 🐳 Production Ready
- Docker containers for both Streamlit and FastAPI
- Comprehensive test suite (pytest)
- CI/CD pipeline ready
- Git LFS for model versioning
- Complete deployment guides

---

## 🧪 Experiments Completed

| # | Experiment | Status | Key Deliverables |
|---|-----------|--------|------------------|
| 1️⃣ | **Introduction & Setup** | ✅ Complete | Project objectives, dataset overview, environment setup |
| 2️⃣ | **Data Cleaning** | ✅ Complete | Missing value handling, outlier removal, feature engineering |
| 3️⃣ | **EDA & Statistics** | ✅ Complete | Visualizations, correlations, statistical tests (Chi-square, ANOVA) |
| 4️⃣ | **ML Modeling** | ✅ Complete | 4 models, hyperparameter tuning, MLflow tracking, dynamic feature importance |
| 5️⃣ | **Explainability** | ✅ Complete | SHAP global/local analysis, LIME explanations, fairness metrics, bias detection |
| 6️⃣ | **API Deployment** | ✅ Complete | Streamlit app, FastAPI, Docker containers |
| 7️⃣ | **Dashboard & AI** | ✅ Complete | Responsible AI report, comprehensive dashboard |
| 8️⃣ | **Final Portfolio** | ✅ Complete | Documentation, deployment guides, GitHub repo |

**Total Progress:** 8/8 Experiments (100%) ✅

---

## 🏆 Models & Performance



### Model Comparison (All Trained Models)

| Model               | Stage     | Accuracy | Precision | Recall   | F1-Score | Status/Notes                |
|---------------------|-----------|----------|-----------|----------|----------|-----------------------------|
| **KNN**             | Baseline  | 95.30%   | 95.06%    | 98.05%   | 0.9653   | ✅ Best Realistic Model      |
| **KNN**             | Tuned     | 94.92%   | 96.10%    | 96.28%   | 0.9619   | ✅ Production Ready          |
| **ANN_DNN**         | Baseline  | 94.83%   | 92.98%    | 99.76%   | 0.9625   | ✅ High Recall               |
| **ANN_DNN**         | Tuned     | 94.13%   | 97.42%    | 93.66%   | 0.9550   | ✅ Balanced                  |
| **LDA**             | Baseline  | 94.70%   | 92.63%    | 100.00%  | 0.9617   | ✅ Perfect Recall            |
| **LDA**             | Tuned     | 91.33%   | 100.00%   | 86.99%   | 0.9304   | ✅ High Precision            |
| **Naive Bayes**     | Baseline  | 75.56%   | 100.00%   | 63.30%   | 0.7752   | ⚠️ Not Recommended           |
| **Naive Bayes**     | Tuned     | 75.56%   | 100.00%   | 63.30%   | 0.7752   | ⚠️ Not Recommended           |
| **Decision Tree**   | Baseline  | 100.00%  | 1.0000    | 1.0000   | 1.0000   | ⚠️ Overfitting (not deployed)|
| **Decision Tree**   | Tuned     | 100.00%  | 1.0000    | 1.0000   | 1.0000   | ⚠️ Overfitting (not deployed)|
| **SVM**             | Baseline  | 99.97%   | 0.9976    | 1.0000   | 0.9997   | ⚠️ Overfitting (not deployed)|
| **SVM**             | Tuned     | 99.68%   | 0.9976    | 1.0000   | 0.9997   | ⚠️ Overfitting (not deployed)|
| **Logistic Regression** | Baseline | 94.70% | 0.9617    | 0.9263   | 0.9617   | ⚠️ Not deployed              |
| **Logistic Regression** | Tuned    | 99.97%  | 1.0000    | 0.9995   | 0.9998   | ⚠️ Overfitting (not deployed) |
| **Random Forest**   | Baseline  | 100.00%  | 1.0000    | 1.0000   | 1.0000   | ⚠️ Overfitting (not deployed)|
| **Random Forest**   | Tuned     | 100.00%  | 1.0000    | 1.0000   | 1.0000   | ⚠️ Overfitting (not deployed)|
| **XGBoost**         | Baseline  | 100.00%  | 1.0000    | 1.0000   | 1.0000   | ⚠️ Overfitting (not deployed)|
| **XGBoost**         | Tuned     | 100.00%  | 1.0000    | 1.0000   | 1.0000   | ⚠️ Overfitting (not deployed)|
| **LightGBM**        | Baseline  | 100.00%  | 1.0000    | 1.0000   | 1.0000   | ⚠️ Overfitting (not deployed)|
| **LightGBM**        | Tuned     | 100.00%  | 1.0000    | 1.0000   | 1.0000   | ⚠️ Overfitting (not deployed)|

**Note:** Models with 100% accuracy are likely overfitting and are not deployed. Only KNN, ANN_DNN, LDA, and Naive Bayes are used in production.

### 🎯 Best Model: KNN (Baseline) - 95.30%

```python
Model: K-Nearest Neighbors (KNN) - Baseline
Accuracy: 95.30%
Precision: 95.06%  # Minimal false positives
Recall: 98.05%     # Excellent at catching successes
F1-Score: 0.9653   # Great balance

Why KNN (Baseline)?
✅ Highest accuracy (95.30%) without hyperparameter tuning
✅ Excellent recall (98.05% - catches 98% of successful products)
✅ Strong precision (95.06% - minimal false alarms)
✅ Best F1-score (0.9653) - optimal precision-recall balance
✅ Fast inference (~5ms per prediction)
✅ No overfitting (realistic performance on test data)
✅ Simple, interpretable, production-ready

**Alternative Models:**
- **ANN_DNN (Baseline)**: 94.83% - Best for catching ALL successes (99.76% recall)
- **LDA (Baseline)**: 94.70% - Perfect recall (100%) for risk-averse scenarios
- **KNN (Tuned)**: 94.92% - More balanced precision/recall after tuning
```


### 📊 Feature Importance (Dynamic Analysis)

Top features influencing success (calculated using permutation importance):

1. **Price** - Premium pricing signals quality
2. **Has Cocoa** - Premium ingredient indicator
3. **Has Protein** - Health appeal factor
4. **Clean Label** - Trust and transparency
5. **Has Dates** - Natural sweetener preference

*Note: Feature importance is dynamically calculated from actual trained models using permutation analysis, providing real insights into what drives predictions.*

---

## 🚀 Live Deployments

### 🎨 Streamlit Application

**Platform:** Streamlit Cloud  
**Features:**
- 7 interactive experiment pages
- Multi-model comparison (4 models: KNN, ANN_DNN, LDA, Naive Bayes)
- Real-time predictions with SHAP & LIME explanations
- Dynamic feature importance analysis
- Data exploration & visualization
- Fairness analysis dashboard

**Access Locally:**
```bash
streamlit run app.py
# Open: http://localhost:8501
```

**Deploy to Cloud:**
1. Push code to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io/)
3. Connect repository and deploy!

---

### 🔌 FastAPI REST API

**Platform:** Railway / Render / Cloud Run  
**Endpoints:**
- `GET /` - API information
- `GET /health` - Health check
- `GET /model-info` - Model details
- `POST /predict` - Single prediction
- `POST /batch-predict` - Batch predictions

**Access Locally:**
```bash
uvicorn main:app --reload --port 8000
# Swagger: http://localhost:8000/docs
# ReDoc: http://localhost:8000/redoc
```

**Example Request:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "price": 1047,
    "discount": 20,
    "category": "Dark Chocolate",
    "ingredients_count": 8,
    "has_dates": true,
    "has_cocoa": true,
    "has_protein": true,
    "packaging_type": "premium_box",
    "season": "winter",
    "customer_gender": "female",
    "age_numeric": 35,
    "shelf_life": 365,
    "clean_label": true
  }'
```

**Response:**
```json
{
  "prediction": "success",
  "probability": 0.85,
  "confidence": "high",
  "model_used": "knn_model",
  "shap_explanation": {
    "price": 0.32,
    "has_cocoa": 0.25,
    "has_protein": 0.15
  }
}
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+** installed
- **Git** with **Git LFS** (for large model files)
- **Docker** (optional, for containerized deployment)

### Installation Steps

#### 1. Clone the Repository

```bash
git clone https://github.com/Horus0305/DS-Experiments.git
cd DS-Experiments
```

#### 2. Set Up Virtual Environment (Recommended)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python -m venv venv
source venv/bin/activate
```

#### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

#### 4. Pull Model Files (Git LFS)

```bash
git lfs install
git lfs pull
```

This will download the trained model pickle files from Git LFS storage.

---

## 💻 Running the Application

### Option 1: Streamlit Dashboard (Recommended)

Run the interactive web dashboard:

```bash
streamlit run app.py
```

**Access:** Open your browser to `http://localhost:8501`

**Features:**
- Explore all 8 experiments interactively
- Test model predictions in real-time
- View SHAP/LIME explanations
- Analyze fairness metrics
- Compare model performance

### Option 2: FastAPI REST API

Run the API server for programmatic access:

```bash
uvicorn main:app --reload --port 8000
```

**Access:**
- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`
- **API Base:** `http://localhost:8000`

### Option 3: Run Both Services

**PowerShell (Windows):**
```powershell
.\run_both.ps1
```

**Bash (Linux/Mac):**
```bash
chmod +x run_both.sh
./run_both.sh
```

This starts both Streamlit (port 8501) and FastAPI (port 8000) simultaneously.

---

## � Docker Deployment

### Build and Run Containers

#### Streamlit Container

```bash
# Build image
docker build -t ds-experiments:latest .

# Run container
docker run -p 8501:8501 ds-experiments:latest
```

Access: `http://localhost:8501`

#### FastAPI Container

```bash
# Build image
docker build -f Dockerfile.api -t ds-api:latest .

# Run container
docker run -p 8000:8000 ds-api:latest
```

Access: `http://localhost:8000/docs`

#### Docker Compose (Both Services)

```bash
# Start both services
docker-compose up

# Run in detached mode
docker-compose up -d

# Stop services
docker-compose down
```

**Ports:**
- Streamlit: `http://localhost:8501`
- FastAPI: `http://localhost:8000`

---

## 🔌 API Usage

### Make Predictions via API

#### Single Prediction

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "price": 1047,
    "discount": 20,
    "category": "Dark Chocolate",
    "ingredients_count": 8,
    "has_dates": true,
    "has_cocoa": true,
    "has_protein": true,
    "packaging_type": "premium_box",
    "season": "winter",
    "customer_gender": "female",
    "age_numeric": 35,
    "shelf_life": 365,
    "clean_label": true
  }'
```

#### Response Format

```json
{
  "prediction": "success",
  "probability": 0.95,
  "confidence": "high",
  "model_used": "knn_model",
  "shap_explanation": {
    "price": 0.32,
    "has_cocoa": 0.25,
    "has_protein": 0.15
  }
}
```

#### Batch Predictions

```bash
curl -X POST "http://localhost:8000/batch-predict" \
  -H "Content-Type: application/json" \
  -d '{
    "products": [
      { "price": 1047, "discount": 20, ... },
      { "price": 899, "discount": 15, ... }
    ]
  }'
```

#### Health Check

```bash
curl http://localhost:8000/health
```

#### Model Information

```bash
curl http://localhost:8000/model-info
```

---

## �📁 Project Structure

```
DS-Experiments/
│
├── app.py                          # Main Streamlit application
├── main.py                         # FastAPI REST API
│
├── pages/                          # Streamlit experiment pages
│   ├── 1_Introduction.py
│   ├── 2_Data_Cleaning.py
│   ├── 3_EDA_and_Statistical_Analysis.py
│   ├── 4_ML_Modeling_and_Tracking.py
│   ├── 5_Explainability_and_Fairness.py
│   ├── 6_API_Deployment_and_Containerization.py
│   └── 7_Dashboard_and_Responsible_AI.py
│
├── tests/                          # Test suite (pytest)
│   ├── test_application.py         # Streamlit tests
│   ├── test_models.py              # Model prediction tests
│   └── test_main.py                # FastAPI endpoint tests
│
├── dsmodelpickl+preprocessor/      # Trained models & preprocessor
│   ├── preprocessor.pkl            # StandardScaler + encoders
│   ├── knn_model.pkl              # Best model (95.30% accuracy)
│   ├── ann_dnn_model.pkl          # Neural network (94.83% accuracy)
│   ├── lda_model.pkl              # Linear Discriminant Analysis (94.70%)
│   └── naive_bayes_model.pkl      # Naive Bayes (75.56%)
│
├── data/                           # Dataset
│   └── WholeTruthFoodDataset-combined.csv
│
├── experiments/                    # Jupyter notebooks (optional)
│
├── .github/                        # CI/CD workflows
│   └── workflows/
│       └── ci-cd-pipeline.yml      # GitHub Actions
│
├── .streamlit/                     # Streamlit configuration
│   └── config.toml
│
├── Dockerfile                      # Streamlit container
├── Dockerfile.api                  # FastAPI container
├── docker-compose.yml              # Compose configuration
│
├── requirements.txt                # Python dependencies
├── packages.txt                    # System packages
│
├── README.md                       # This file
├── API_README.md                   # API documentation
├── STREAMLIT_README.md             # Streamlit guide
├── DOCKER_README.md                # Docker deployment guide
├── FASTAPI_SETUP_COMPLETE.md       # FastAPI setup summary
├── Responsible_AI.md               # Responsible AI report
│
├── run_both.ps1                    # Run both services (PowerShell)
├── run_both.sh                     # Run both services (Bash)
│
├── .gitignore
├── .gitattributes                  # Git LFS configuration
└── LICENSE                         # MIT License
```

---

## 🛠️ Technologies Used

### Core Technologies
| Category | Technologies |
|----------|-------------|
| **Language** | Python 3.10+ |
| **Web Frameworks** | Streamlit 1.28+, FastAPI 0.104+ |
| **ML Libraries** | Scikit-learn, TensorFlow, XGBoost |
| **Explainability** | SHAP, LIME |
| **Experiment Tracking** | MLflow |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Plotly, Seaborn, Matplotlib |
| **Deployment** | Docker, Uvicorn, Git LFS |
| **Testing** | Pytest |
| **Cloud Platform** | Streamlit Cloud |

### Key Dependencies

```
streamlit>=1.28.0
fastapi>=0.104.0
uvicorn>=0.24.0
scikit-learn>=1.3.0
tensorflow>=2.13.0
shap>=0.43.0
lime>=0.2.0.1
mlflow>=2.8.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.18.0
pydantic>=2.5.0
pytest>=7.4.0
```

---

## � Testing

### Run All Tests

```bash
# Run complete test suite
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=. --cov-report=html

# View coverage report
# Open htmlcov/index.html in browser
```

### Individual Test Modules

```bash
# Test model predictions
pytest tests/test_models.py -v

# Test API endpoints
pytest tests/test_main.py -v

# Test Streamlit application
pytest tests/test_application.py -v
```

### Test Structure

```
tests/
├── test_application.py    # Streamlit app tests
├── test_models.py         # Model prediction tests
└── test_main.py           # FastAPI endpoint tests
```

---

## 📚 Documentation

### Available Guides

| Document | Description |
|----------|-------------|
| **[README.md](README.md)** | Project overview and setup (this file) |
| **[API_README.md](API_README.md)** | FastAPI endpoint documentation |
| **[DOCKER_API_GUIDE.md](DOCKER_API_GUIDE.md)** | Docker containerization guide |

### Interactive Documentation

- **Swagger UI:** `http://localhost:8000/docs` (when FastAPI is running)
- **ReDoc:** `http://localhost:8000/redoc` (alternative API docs)
- **Live Dashboard:** [https://ds-dashboard-sem7.streamlit.app/](https://ds-dashboard-sem7.streamlit.app/)

### Code Documentation

- Inline comments throughout codebase
- Function docstrings with parameter descriptions
- Type hints for better IDE support

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

### How to Contribute

1. **Fork** the repository
2. **Create a feature branch:** `git checkout -b feature/your-feature`
3. **Make your changes** and add tests
4. **Run tests:** `pytest tests/ -v`
5. **Commit:** `git commit -m "Add: feature description"`
6. **Push:** `git push origin feature/your-feature`
7. **Open a Pull Request**

### Guidelines

- Follow **PEP 8** Python style guide
- Add **docstrings** to new functions
- Write **tests** for new features
- Update **documentation** as needed
- Be respectful and collaborative

---

## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Horus0305**

- 🌐 **Live Demo:** [https://ds-dashboard-sem7.streamlit.app/](https://ds-dashboard-sem7.streamlit.app/)
-  **GitHub:** [@Horus0305](https://github.com/Horus0305)
- 📦 **Repository:** [DS-Experiments](https://github.com/Horus0305/DS-Experiments)

---

## 📞 Support

### Get Help

- **🐛 Report Bugs:** [GitHub Issues](https://github.com/Horus0305/DS-Experiments/issues)
- **💬 Discussions:** [GitHub Discussions](https://github.com/Horus0305/DS-Experiments/discussions)
- **📖 Documentation:** See guides in repository

### Common Issues

1. **Git LFS Issues:** Run `git lfs install` and `git lfs pull`
2. **Import Errors:** Ensure all dependencies installed with `pip install -r requirements.txt`
3. **Port Conflicts:** Change ports in run commands if 8501 or 8000 are in use
4. **Model Loading Errors:** Verify model files exist in `dsmodelpickl+preprocessor/`

---

## 🌟 Acknowledgments

- **The Whole Truth Foods** - Dataset and business context
- **Streamlit** - Web framework for data apps
- **FastAPI** - Modern API framework
- **Scikit-learn** - Machine learning library
- **SHAP & LIME** - Explainability tools
- **Open Source Community** - Inspiration and support

---

<div align="center">

**Built with ❤️ using Python, Streamlit, and FastAPI**

[![Made with Python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![Made with Streamlit](https://img.shields.io/badge/Made%20with-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Made with FastAPI](https://img.shields.io/badge/Made%20with-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)

⭐ **Star this repository** if you found it helpful!

[🚀 Live Demo](https://ds-dashboard-sem7.streamlit.app/) · [🐛 Report Bug](https://github.com/Horus0305/DS-Experiments/issues) · [✨ Request Feature](https://github.com/Horus0305/DS-Experiments/issues)

---

© 2024-2025 Horus0305 | MIT License

</div>
