import streamlit as st

st.set_page_config(page_title="Introduction", layout="wide")

st.title("Case Study Framing & Dataset Preparation")

st.info("**Objective:** Define a real-world domain problem, benchmark existing solutions, acquire data, and document versioning plan with DVC.")

# Problem Statement
st.header("Problem Statement")

st.subheader("Company Overview")
st.markdown("""
**The Whole Truth Foods** is an Indian clean-label food brand founded in 2019 that focuses on 
creating transparent, healthy snacks using minimal natural ingredients while avoiding artificial 
additives, preservatives, and refined sugars.
""")

st.subheader("Problem Definition")
st.markdown("""
**Domain Problem:** Consumer Preference Prediction for Clean-Label Food Products in the Indian Market

**Business Challenge:** The company faces challenges in:
- Predicting product attributes that drive consumer adoption
- Understanding consumer sentiment across product categories
- Identifying optimal pricing strategies
- Forecasting demand patterns for inventory management
""")

st.subheader("Research Objectives")
st.markdown("""
Develop a data-driven framework to predict consumer preferences by analyzing:
- Social media sentiment
- E-commerce behavior
- Market positioning
""")

with st.expander("View Key Research Questions"):
    st.markdown("""
    1. Which product attributes most strongly influence customer ratings and purchase likelihood?
    2. How can consumer sentiment scores be predicted and compared with competitors?
    3. Can ML models accurately predict commercial success of new products?
    4. What seasonal and demographic patterns exist in clean-label food consumption?
    """)

st.subheader("Expected Business Impact")
col1, col2, col3 = st.columns(3)
col1.metric("Sentiment Analysis Accuracy", "≥85%")
col2.metric("Classification Recall", "≥95%")
col3.metric("Demand Prediction RMSE", "≤15%")

# Benchmark Analysis
st.header("Benchmark Analysis")

st.markdown("""
**Existing Solutions:**
- Traditional: Nielsen Insights, Euromonitor Reports (High cost, limited real-time insights)
- Digital: Brandwatch, Jungle Scout (Platform-specific, lack integration)
- Academic: VADER, ARIMA/Prophet (Generic, not food-industry specific)

**Our Improvements:**
- Integrate multiple data sources with ensemble methods
- Combine structured and unstructured data
- Use advanced NLP and computer vision
- Real-time data pipeline
- Focus on maximizing recall of negative sentiment
""")

# Data Acquisition
st.header("Data Acquisition Strategy")

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**Automated Extraction**")
    st.markdown("Browser scraping tools for structured data")
with col2:
    st.markdown("**Manual Scraping**")
    st.markdown("Manual collection for complex details")
with col3:
    st.markdown("**Synthetic Data**")
    st.markdown("Faker library for missing user data")

# Dataset Description
st.header("Dataset Description")

st.markdown("**10,000 records** with **31 attributes**")

with st.expander("View Dataset Schema"):
    st.markdown("""
    **Product Info:** product_id, product_name, brand, category, ingredients, clean_label, 
    price, discount, units_sold, average_rating, num_reviews, shelf_life, packaging_type
    
    **Review Info:** review_id, review_text, rating, platform, date, reviewer_location, 
    sentiment_score, reviewer_demographic, sentiment_category
    
    **Transaction Info:** transaction_id, purchase_date, quantity, price_per_unit, region, 
    customer_age_group, customer_gender, season, channel
    """)

# Risk Mitigation
st.header("Risk Mitigation")

col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    **Data Access Risks:**
    - Incomplete Data → Validation checks
    - Website Changes → Flexible selectors
    - Manual Errors → Templates & checklists
    """)
with col2:
    st.markdown("""
    **Technical Risks:**
    - Large storage → DVC with cloud
    - Processing time → Parallel processing
    - Reproducibility → Logging & version control
    """)

# Conclusion
st.header("Conclusion")
st.success("""
Established a comprehensive data science framework for analyzing The Whole Truth Foods' 
market position. Multi-dimensional analysis through DVC-versioned pipelines provides 
accurate, cost-effective insights for data-driven decision making.
""")
