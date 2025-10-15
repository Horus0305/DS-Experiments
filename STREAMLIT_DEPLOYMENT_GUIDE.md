# 🚀 Streamlit Cloud Deployment Guide

Complete guide to deploying your ML application to Streamlit Cloud.

---

## 📋 Prerequisites

✅ **Already Completed:**
- GitHub repository created (Horus0305/DS-Experiments)
- Streamlit app developed (`app.py` with 7 experiment pages)
- All dependencies listed in `requirements.txt`
- Configuration in `.streamlit/config.toml`
- Model files ready (tracked by Git LFS)
- Tests passing locally

---

## 🎯 Step 1: Commit and Push to GitHub

Make sure all your changes are committed and pushed:

```powershell
# Stage all changes
git add .

# Commit with message
git commit -m "feat: Complete ML pipeline ready for Streamlit Cloud deployment

- Fixed test_models.py (customer_gender encoding)
- Updated main.py to Pydantic V2 (json_schema_extra)
- Replaced deprecated @app.on_event with lifespan context manager
- Fixed test_main.py to handle 503 status in CI
- Added pytest and testing dependencies to requirements.txt
- Added ML dependencies (shap, mlflow, imbalanced-learn)
- Ready for Streamlit Cloud deployment"

# Push to GitHub
git push origin main
```

**Verify GitHub Actions:** 
- Visit: https://github.com/Horus0305/DS-Experiments/actions
- Ensure all tests pass ✅

---

## 🌐 Step 2: Deploy to Streamlit Cloud

### 2.1 Go to Streamlit Cloud

1. Visit: **https://share.streamlit.io/**
2. Click **"Sign in"** (use your GitHub account)
3. Click **"New app"**

### 2.2 Configure Deployment

Fill in the deployment form:

| Field | Value |
|-------|-------|
| **Repository** | `Horus0305/DS-Experiments` |
| **Branch** | `main` |
| **Main file path** | `app.py` |
| **App URL** | `ds-experiments` (or custom name) |

**Advanced Settings (Optional):**
- **Python version:** 3.10
- **Secrets:** (None needed for this app)

### 2.3 Deploy!

1. Click **"Deploy!"**
2. Wait 5-10 minutes for initial deployment
3. Monitor the deployment logs

---

## 📦 What Happens During Deployment?

Streamlit Cloud will:

1. ✅ Clone your GitHub repository
2. ✅ Pull Git LFS files (model files)
3. ✅ Install system packages from `packages.txt`
4. ✅ Install Python dependencies from `requirements.txt`
5. ✅ Apply configuration from `.streamlit/config.toml`
6. ✅ Start the Streamlit app
7. ✅ Provide a public URL

**Estimated Time:** 5-10 minutes

---

## 🎨 Expected Deployment URL

Your app will be available at:

```
https://ds-experiments-[random].streamlit.app
```

Or with custom subdomain:

```
https://[your-custom-name].streamlit.app
```

---

## ✅ Post-Deployment Checklist

After deployment, verify:

- [ ] App loads without errors
- [ ] All 7 experiment pages accessible
- [ ] Model predictions working
- [ ] SHAP explanations displaying
- [ ] Fairness analysis showing
- [ ] Charts and visualizations rendering
- [ ] No missing dependencies errors

---

## 🐛 Troubleshooting

### Issue 1: Model Files Not Found

**Error:** `FileNotFoundError: dsmodelpickl+preprocessor/knn_model.pkl`

**Solution:**
```powershell
# Ensure Git LFS is tracking model files
git lfs track "dsmodelpickl+preprocessor/*.pkl"
git add .gitattributes
git add dsmodelpickl+preprocessor/
git commit -m "Track model files with Git LFS"
git push origin main

# In Streamlit Cloud, trigger redeploy
```

### Issue 2: Missing Dependencies

**Error:** `ModuleNotFoundError: No module named 'shap'`

**Solution:**
- Check `requirements.txt` includes all dependencies
- Update requirements.txt and push:
```powershell
git add requirements.txt
git commit -m "Add missing dependencies"
git push origin main
```

### Issue 3: Memory Limit Exceeded

**Error:** `Memory limit exceeded`

**Solution:**
- Streamlit Cloud free tier: 1GB RAM
- Optimize by loading only necessary models
- Consider Streamlit Cloud paid tier for more resources

### Issue 4: Slow Loading

**Symptoms:** App takes >30 seconds to load

**Solutions:**
1. Use `@st.cache_resource` for model loading
2. Reduce model file sizes
3. Load models on demand, not on startup

### Issue 5: App Crashes on Startup

**Check Deployment Logs:**
1. Go to Streamlit Cloud dashboard
2. Click on your app
3. View "Manage app" → "Logs"
4. Identify the error and fix

---

## 🔧 Configuration Files

### 1. `requirements.txt`

Already configured with all dependencies:
```plaintext
# Core DVC
dvc>=3.38.1
dvc-gdrive>=3.0.1

# Streamlit
streamlit>=1.28.0

# Data Processing
pandas>=2.0.0
numpy>=1.24.0

# Visualization
plotly>=5.18.0
matplotlib>=3.7.0
seaborn>=0.12.0

# Machine Learning
scikit-learn==1.6.1
tensorflow>=2.13.0
imbalanced-learn>=0.11.0
shap>=0.43.0
mlflow>=2.8.0

# Optional
openpyxl>=3.1.0

# FastAPI (not used in Streamlit deployment)
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
pydantic>=2.5.0
python-multipart>=0.0.6
httpx>=0.25.0

# Testing
pytest>=7.4.0
pytest-cov>=4.1.0
```

### 2. `.streamlit/config.toml`

Already configured:
```toml
[theme]
primaryColor = "#2E7D32"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F5F5F5"
textColor = "#262730"
font = "sans serif"

[server]
headless = true
port = 8501
```

### 3. `packages.txt`

For system-level dependencies (if needed):
```plaintext
# System-level dependencies
# Example: libgomp1
```

---

## 🚀 Optimization Tips

### 1. Cache Model Loading

Update `app.py` to cache models:

```python
import streamlit as st

@st.cache_resource
def load_models():
    """Load models once and cache them"""
    # Your model loading code
    return model, preprocessor

# Use cached models
model, preprocessor = load_models()
```

### 2. Lazy Loading

Load models only when needed:

```python
if st.button("Make Prediction"):
    model = load_model()  # Load on demand
    prediction = model.predict(...)
```

### 3. Reduce Model Sizes

If models are too large:
- Use quantization
- Remove unused models
- Compress pickle files

---

## 📊 Monitoring Your App

### View Analytics

Streamlit Cloud provides:
- **Viewer count** - Number of active users
- **App runs** - Total number of sessions
- **Deployment history** - All deployments
- **Logs** - Real-time application logs

Access via: **Streamlit Cloud Dashboard → Your App → Manage app**

---

## 🔄 Updating Your Deployed App

To update your app after deployment:

```powershell
# Make changes locally
# Test locally: streamlit run app.py

# Commit and push
git add .
git commit -m "Update: description of changes"
git push origin main

# Streamlit Cloud auto-deploys on push to main branch
```

**Auto-deployment:** Enabled by default ✅

---

## 💰 Pricing & Limits

### Free Tier (Suitable for your project)
- ✅ 1 GB RAM
- ✅ 1 CPU core
- ✅ Unlimited apps (public)
- ✅ Auto-deployment from GitHub
- ✅ Community support

### Paid Tiers (If needed)
- **Starter:** $20/month - More resources
- **Team:** Custom pricing - Private apps, more resources

**Your app should work fine on free tier!**

---

## 🎯 Final Deployment Commands

```powershell
# 1. Ensure everything is committed
git status

# 2. Add all files
git add .

# 3. Commit with message
git commit -m "feat: Ready for Streamlit Cloud deployment

- Fixed Pydantic V2 deprecation warnings
- Updated to lifespan context manager
- Fixed test_models.py customer_gender encoding
- Added all ML dependencies (shap, mlflow, imbalanced-learn)
- Added pytest for testing
- Updated test_main.py to handle 503 in CI
- All tests passing
- Ready for production deployment"

# 4. Push to GitHub
git push origin main

# 5. Go to share.streamlit.io and deploy!
```

---

## ✅ Pre-Deployment Checklist

Before deploying, ensure:

- [x] `app.py` exists and runs locally
- [x] `requirements.txt` includes ALL dependencies
- [x] `.streamlit/config.toml` configured
- [x] `packages.txt` created (even if empty)
- [x] Model files tracked by Git LFS
- [x] All tests passing (`pytest tests/ -v`)
- [x] GitHub Actions CI/CD passing
- [x] Repository is public (or Streamlit Cloud has access)
- [x] No hardcoded secrets (use Streamlit secrets if needed)

**Status: ALL READY ✅**

---

## 🌐 Share Your App

Once deployed, share your app:

**Public URL:** `https://[your-app].streamlit.app`

**Embed in README:**
```markdown
[![Streamlit App](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://[your-app].streamlit.app)
```

**Social Media:**
- Twitter: Share with #StreamlitApp #MachineLearning
- LinkedIn: Post about your project
- GitHub: Add URL to repository description

---

## 📚 Resources

- **Streamlit Docs:** https://docs.streamlit.io/streamlit-community-cloud
- **Git LFS:** https://git-lfs.github.com/
- **Streamlit Forum:** https://discuss.streamlit.io/

---

## 🎉 You're Ready!

Your app is fully prepared for deployment. Just follow the steps above and you'll have a live ML application in minutes!

**Good luck! 🚀**

---

**Last Updated:** October 16, 2025  
**Author:** Horus0305  
**Repository:** [DS-Experiments](https://github.com/Horus0305/DS-Experiments)
