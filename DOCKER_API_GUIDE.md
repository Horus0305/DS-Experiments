# 🐳 Docker API Guide - FastAPI ML Prediction Service

Complete guide to running and testing the FastAPI ML service using Docker.

---

## 📋 Prerequisites

- Docker Desktop installed and running
- PowerShell or CMD (Windows) / Terminal (Mac/Linux)
- Port 8000 available

---

## 🚀 Step 1: Build the Docker Image

```powershell
# Build the API Docker image
docker build -f Dockerfile.api -t ds-api:latest .
```

**Expected Output:**
```
[+] Building 660.5s (13/13) FINISHED
=> exporting to image
=> => naming to docker.io/library/ds-api:latest
```

---

## ▶️ Step 2: Run the Docker Container

```powershell
# Run the container (detached mode)
docker run -d -p 8000:8000 --name ds-api-container ds-api:latest
```

**Flags Explained:**
- `-d` = Detached mode (runs in background)
- `-p 8000:8000` = Maps port 8000 (host:container)
- `--name ds-api-container` = Names the container
- `ds-api:latest` = Image name and tag

**Check if running:**
```powershell
docker ps
```

You should see:
```
CONTAINER ID   IMAGE            PORTS                    NAMES
abc123def456   ds-api:latest    0.0.0.0:8000->8000/tcp   ds-api-container
```

---

## 🧪 Step 3: Test the API

### Option 1: Using Web Browser

Open: **http://localhost:8000/docs**

This opens the **Swagger UI** with interactive API documentation.

### Option 2: Using PowerShell (curl)

#### Test 1: Root Endpoint
```powershell
curl http://localhost:8000/
```

**Expected Response:**
```json
{
  "message": "Product Success Prediction API",
  "version": "1.0.0",
  "docs": "/docs",
  "health": "/health"
}
```

#### Test 2: Health Check
```powershell
curl http://localhost:8000/health
```

**Expected Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2025-10-16T12:34:56"
}
```

#### Test 3: Model Info
```powershell
curl http://localhost:8000/model-info
```

**Expected Response:**
```json
{
  "model_name": "knn_model",
  "model_type": "K-Nearest Neighbors",
  "features": 13,
  "accuracy": "95.30%"
}
```

---

## 📤 Step 4: Make Predictions

### ✅ Test Case 1: Premium Success Case

**PowerShell:**
```powershell
$body = @{
    price = 1047
    discount = 0
    category = "Dark Chocolate"
    ingredients_count = 4
    has_dates = $true
    has_cocoa = $true
    has_protein = $false
    packaging_type = "Paper-based"
    season = "Winter"
    customer_gender = "Female"
    age_numeric = 55
    shelf_life = 12
    clean_label = $true
} | ConvertTo-Json

curl -X POST "http://localhost:8000/predict" `
  -H "Content-Type: application/json" `
  -d $body
```

**CMD:**
```cmd
curl -X POST "http://localhost:8000/predict" ^
  -H "Content-Type: application/json" ^
  -d "{\"price\": 1047, \"discount\": 0, \"category\": \"Dark Chocolate\", \"ingredients_count\": 4, \"has_dates\": true, \"has_cocoa\": true, \"has_protein\": false, \"packaging_type\": \"Paper-based\", \"season\": \"Winter\", \"customer_gender\": \"Female\", \"age_numeric\": 55, \"shelf_life\": 12, \"clean_label\": true}"
```

**Expected Response:**
```json
{
  "prediction": "success",
  "probability": 0.92,
  "confidence": "high",
  "model_used": "knn_model"
}
```

---

### ❌ Test Case 2: Budget Failure Case

**PowerShell:**
```powershell
$body = @{
    price = 349
    discount = 15
    category = "Muesli"
    ingredients_count = 12
    has_dates = $false
    has_cocoa = $false
    has_protein = $false
    packaging_type = "Recyclable Plastic"
    season = "Summer"
    customer_gender = "Male"
    age_numeric = 25
    shelf_life = 6
    clean_label = $false
} | ConvertTo-Json

curl -X POST "http://localhost:8000/predict" `
  -H "Content-Type: application/json" `
  -d $body
```

**CMD:**
```cmd
curl -X POST "http://localhost:8000/predict" ^
  -H "Content-Type: application/json" ^
  -d "{\"price\": 349, \"discount\": 15, \"category\": \"Muesli\", \"ingredients_count\": 12, \"has_dates\": false, \"has_cocoa\": false, \"has_protein\": false, \"packaging_type\": \"Recyclable Plastic\", \"season\": \"Summer\", \"customer_gender\": \"Male\", \"age_numeric\": 25, \"shelf_life\": 6, \"clean_label\": false}"
```

**Expected Response:**
```json
{
  "prediction": "failure",
  "probability": 0.23,
  "confidence": "high",
  "model_used": "knn_model"
}
```

---

### 💪 Test Case 3: Protein Bar Case

**PowerShell:**
```powershell
$body = @{
    price = 799
    discount = 5
    category = "Protein Bar"
    ingredients_count = 8
    has_dates = $true
    has_cocoa = $true
    has_protein = $true
    packaging_type = "Biodegradable"
    season = "Winter"
    customer_gender = "Female"
    age_numeric = 35
    shelf_life = 9
    clean_label = $true
} | ConvertTo-Json

curl -X POST "http://localhost:8000/predict" `
  -H "Content-Type: application/json" `
  -d $body
```

**Expected Response:**
```json
{
  "prediction": "success",
  "probability": 0.88,
  "confidence": "high",
  "model_used": "knn_model"
}
```

---

## 📦 Step 5: Batch Predictions

**PowerShell:**
```powershell
$body = @{
    predictions = @(
        @{
            price = 1047
            discount = 0
            category = "Dark Chocolate"
            ingredients_count = 4
            has_dates = $true
            has_cocoa = $true
            has_protein = $false
            packaging_type = "Paper-based"
            season = "Winter"
            customer_gender = "Female"
            age_numeric = 55
            shelf_life = 12
            clean_label = $true
        },
        @{
            price = 349
            discount = 15
            category = "Muesli"
            ingredients_count = 12
            has_dates = $false
            has_cocoa = $false
            has_protein = $false
            packaging_type = "Recyclable Plastic"
            season = "Summer"
            customer_gender = "Male"
            age_numeric = 25
            shelf_life = 6
            clean_label = $false
        }
    )
} | ConvertTo-Json -Depth 3

curl -X POST "http://localhost:8000/batch-predict" `
  -H "Content-Type: application/json" `
  -d $body
```

**Expected Response:**
```json
{
  "predictions": [
    {
      "prediction": "success",
      "probability": 0.92,
      "confidence": "high",
      "model_used": "knn_model"
    },
    {
      "prediction": "failure",
      "probability": 0.23,
      "confidence": "high",
      "model_used": "knn_model"
    }
  ],
  "count": 2
}
```

---

## 🔍 Step 6: View Logs

```powershell
# View real-time logs
docker logs -f ds-api-container

# View last 50 lines
docker logs --tail 50 ds-api-container
```

---

## 🛑 Step 7: Stop and Cleanup

### Stop the container:
```powershell
docker stop ds-api-container
```

### Remove the container:
```powershell
docker rm ds-api-container
```

### Remove the image (optional):
```powershell
docker rmi ds-api:latest
```

### Remove all (nuclear option):
```powershell
docker stop ds-api-container
docker rm ds-api-container
docker rmi ds-api:latest
```

---

## 🐛 Troubleshooting

### Issue 1: Port Already in Use
```
Error: bind: address already in use
```

**Solution:**
```powershell
# Find process using port 8000
netstat -ano | findstr :8000

# Kill the process (replace PID with actual number)
taskkill /PID <PID> /F

# Or use a different port
docker run -d -p 8001:8000 --name ds-api-container ds-api:latest
```

### Issue 2: Container Won't Start
```powershell
# Check container logs
docker logs ds-api-container

# Check container status
docker ps -a

# Restart container
docker restart ds-api-container
```

### Issue 3: Model Not Loading
```powershell
# Check if model files are in the image
docker exec -it ds-api-container ls -la dsmodelpickl+preprocessor/

# Should show:
# preprocessor.pkl
# knn_model.pkl
# (other model files)
```

### Issue 4: curl Command Not Found (Windows)
```powershell
# Use Invoke-WebRequest instead
Invoke-WebRequest -Uri "http://localhost:8000/health" -Method GET
```

---

## 📊 API Endpoints Summary

| Endpoint | Method | Description | Auth |
|----------|--------|-------------|------|
| `/` | GET | API information | No |
| `/health` | GET | Health check | No |
| `/model-info` | GET | Model details | No |
| `/predict` | POST | Single prediction | No |
| `/batch-predict` | POST | Batch predictions | No |
| `/docs` | GET | Swagger UI | No |
| `/redoc` | GET | ReDoc UI | No |

---

## 🔐 Adding Authentication (Optional)

For production, add API key authentication:

```python
# Add to main.py
from fastapi.security import APIKeyHeader

API_KEY = "your-secret-key"
api_key_header = APIKeyHeader(name="X-API-Key")

def verify_api_key(api_key: str = Depends(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key

# Then add to endpoints:
@app.post("/predict", dependencies=[Depends(verify_api_key)])
```

**Usage:**
```powershell
curl -X POST "http://localhost:8000/predict" `
  -H "X-API-Key: your-secret-key" `
  -H "Content-Type: application/json" `
  -d $body
```

---

## 📈 Performance Tips

1. **Use Batch Predictions** for multiple items (more efficient)
2. **Enable Caching** for frequently requested predictions
3. **Monitor Memory** usage with `docker stats ds-api-container`
4. **Scale Horizontally** by running multiple containers with load balancer

---

## 🎯 Quick Reference

**Build:**
```bash
docker build -f Dockerfile.api -t ds-api:latest .
```

**Run:**
```bash
docker run -d -p 8000:8000 --name ds-api-container ds-api:latest
```

**Test:**
```bash
curl http://localhost:8000/health
```

**Stop:**
```bash
docker stop ds-api-container
```

---

**📚 Full Documentation:** [API_README.md](./API_README.md)  
**🌐 Swagger UI:** http://localhost:8000/docs  
**📖 ReDoc:** http://localhost:8000/redoc

---

*Last Updated: October 16, 2025*
