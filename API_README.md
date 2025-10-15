# FastAPI Prediction Service 🚀

## ⚠️ Important: This is OPTIONAL

**Your Streamlit application is completely independent and will deploy to Streamlit Cloud without any issues.**

This FastAPI service is an **additional feature** that provides a REST API for programmatic access to your ML models.

---

## 🎯 Purpose

Provides RESTful API endpoints for:
- Making predictions via HTTP requests
- Integration with other applications
- Automated testing and CI/CD
- External system integrations

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────┐
│         Your DS-Experiments Project          │
├─────────────────────────────────────────────┤
│                                              │
│  ┌──────────────────┐  ┌────────────────┐  │
│  │  Streamlit App    │  │  FastAPI App   │  │
│  │  (app.py)         │  │  (main.py)     │  │
│  │  Port: 8501       │  │  Port: 8000    │  │
│  │  For: Humans      │  │  For: APIs     │  │
│  │  Deploy: Streamlit│  │  Deploy: Cloud │  │
│  └──────────────────┘  └────────────────┘  │
│           │                      │           │
│           └──────────┬───────────┘           │
│                      │                       │
│              ┌───────▼───────┐              │
│              │  Shared Models │              │
│              │  & Preprocessor│              │
│              └────────────────┘              │
└─────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Install Dependencies (if not already installed)

```bash
pip install fastapi uvicorn httpx
```

### 2. Run the API Locally

```bash
# Start the API server
uvicorn main:app --reload --port 8000
```

### 3. Access the API

- **API Root**: http://localhost:8000/
- **Interactive Docs**: http://localhost:8000/docs (Swagger UI)
- **Alternative Docs**: http://localhost:8000/redoc (ReDoc)
- **Health Check**: http://localhost:8000/health

---

## 📡 API Endpoints

### `GET /`
Welcome message and endpoint list

```bash
curl http://localhost:8000/
```

### `GET /health`
Check if model is loaded and API is healthy

```bash
curl http://localhost:8000/health
```

### `GET /model-info`
Get information about the loaded model

```bash
curl http://localhost:8000/model-info
```

### `POST /predict`
Make a single prediction

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "price": 1047,
    "discount": 0,
    "category": "Dark Chocolate",
    "ingredients_count": 4,
    "has_dates": 1,
    "has_cocoa": 1,
    "has_protein": 0,
    "packaging_type": "Paper-based",
    "season": "Winter",
    "customer_gender": 1.0,
    "age_numeric": 55,
    "shelf_life": 12,
    "clean_label": 1
  }'
```

**Response:**
```json
{
  "prediction": 1,
  "probability": 0.8523,
  "confidence": "High",
  "model_used": "KNN"
}
```

### `POST /batch-predict`
Make multiple predictions at once

```bash
curl -X POST "http://localhost:8000/batch-predict" \
  -H "Content-Type: application/json" \
  -d '[
    {
      "price": 1047,
      "discount": 0,
      "category": "Dark Chocolate",
      ...
    },
    {
      "price": 349,
      "discount": 0,
      "category": "Muesli",
      ...
    }
  ]'
```

---

## 🐍 Python Client Example

```python
import requests

# API endpoint
API_URL = "http://localhost:8000/predict"

# Sample input
data = {
    "price": 1047,
    "discount": 0,
    "category": "Dark Chocolate",
    "ingredients_count": 4,
    "has_dates": 1,
    "has_cocoa": 1,
    "has_protein": 0,
    "packaging_type": "Paper-based",
    "season": "Winter",
    "customer_gender": 1.0,
    "age_numeric": 55,
    "shelf_life": 12,
    "clean_label": 1
}

# Make prediction
response = requests.post(API_URL, json=data)
result = response.json()

print(f"Prediction: {result['prediction']}")
print(f"Probability: {result['probability']:.2%}")
print(f"Confidence: {result['confidence']}")
```

---

## 🧪 Testing

```bash
# Run API tests
pytest tests/test_main.py -v

# Run with coverage
pytest tests/test_main.py --cov=main --cov-report=html
```

---

## 🐳 Docker Deployment

### Build and Run with Docker

```bash
# Build the image
docker build -f Dockerfile.api -t ds-experiments-api:latest .

# Run the container
docker run -d -p 8000:8000 --name ds-api ds-experiments-api:latest

# Check logs
docker logs ds-api

# Test the API
curl http://localhost:8000/health
```

### Using Docker Compose

Add to `docker-compose.yml`:

```yaml
services:
  api:
    build:
      context: .
      dockerfile: Dockerfile.api
    ports:
      - "8000:8000"
    environment:
      - MODEL_NAME=KNN
    restart: unless-stopped
```

Then run:
```bash
docker-compose up api
```

---

## ☁️ Cloud Deployment Options

### 1. **Railway** (Easiest)
- Sign up at https://railway.app/
- Connect your GitHub repo
- Set `start` command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- Deploy automatically

### 2. **Render**
- Sign up at https://render.com/
- Create new Web Service
- Set `start` command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- Deploy

### 3. **Google Cloud Run**
```bash
# Build and push to GCR
gcloud builds submit --tag gcr.io/PROJECT_ID/ds-api

# Deploy to Cloud Run
gcloud run deploy ds-api \
  --image gcr.io/PROJECT_ID/ds-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

### 4. **AWS Lambda** (with Mangum)
- Add `mangum` to requirements
- Wrap FastAPI app with Mangum handler
- Deploy via AWS Lambda or Serverless Framework

---

## 🔒 Security Best Practices

For production deployment:

1. **Add API Key Authentication**
```python
from fastapi.security import APIKeyHeader

API_KEY = "your-secret-key"
api_key_header = APIKeyHeader(name="X-API-Key")

@app.post("/predict")
def predict(input_data: PredictionInput, api_key: str = Depends(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    # ... rest of code
```

2. **Add Rate Limiting**
```bash
pip install slowapi

# Add to main.py
from slowapi import Limiter
limiter = Limiter(key_func=get_remote_address)

@app.post("/predict")
@limiter.limit("10/minute")
def predict(...):
    pass
```

3. **CORS Configuration**
Update `allow_origins` in `main.py` to specify allowed domains:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Specific domains only
    ...
)
```

---

## 📊 Monitoring

### Health Checks
```bash
# Check if API is running
curl http://localhost:8000/health

# Expected response
{
  "status": "healthy",
  "model_loaded": true,
  "preprocessor_loaded": true,
  "model_name": "KNN"
}
```

### Logging
Logs are printed to console. In production, use logging services like:
- **Google Cloud Logging**
- **AWS CloudWatch**
- **Datadog**
- **Sentry** (for error tracking)

---

## 🔧 Troubleshooting

### Model Not Loading
```bash
# Check if model files exist
ls dsmodelpickl+preprocessor/

# Expected files:
# - preprocessor.pkl
# - knn_model.pkl
```

### Port Already in Use
```bash
# Change port in startup command
uvicorn main:app --reload --port 8001
```

### Import Errors
```bash
# Reinstall dependencies
pip install -r requirements.txt
```

---

## 📝 Integration with Streamlit

Your Streamlit app (page 6) can optionally call this API:

```python
import requests

def predict_via_api(features):
    """Call FastAPI endpoint from Streamlit"""
    try:
        response = requests.post(
            "http://localhost:8000/predict",
            json=features,
            timeout=5
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"API call failed: {e}")
        return None

# In your Streamlit button handler
if st.button("Predict via API"):
    result = predict_via_api(features_dict)
    if result:
        st.success(f"Prediction: {result['prediction']}")
        st.metric("Probability", f"{result['probability']:.2%}")
```

**Note:** This is optional. Your Streamlit app works perfectly fine loading models directly.

---

## 🎯 When to Use API vs Streamlit

| Use Case | Recommended |
|----------|-------------|
| Human interaction & visualization | **Streamlit** |
| Automated predictions | **FastAPI** |
| Integration with other services | **FastAPI** |
| Mobile app backend | **FastAPI** |
| Data exploration | **Streamlit** |
| Batch processing | **FastAPI** |
| Demo & presentation | **Streamlit** |
| Production ML serving | **FastAPI** |

---

## 🚀 **Your Streamlit App is Still Independent!**

✅ `app.py` and all pages work without `main.py`  
✅ Can deploy to Streamlit Cloud without any changes  
✅ FastAPI is completely optional  
✅ Both can run simultaneously on different ports  
✅ Share the same model files  

---

## 📚 Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Uvicorn Documentation](https://www.uvicorn.org/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [Streamlit Deployment](https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app)

---

**Questions? Check `/docs` endpoint for interactive API documentation!** 🎉
