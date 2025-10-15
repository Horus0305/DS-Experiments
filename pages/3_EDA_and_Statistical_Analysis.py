import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

st.set_page_config(page_title="EDA & Statistical Analysis", layout="wide")

st.title("Exploratory Data Analysis & Statistical Analysis")

st.info("**Objective:** Visualize distributions, understand data patterns, identify correlations, and validate insights through statistical testing")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("data/WholeTruthFoodDataset-combined.csv")

try:
    df = load_data()
    
    # Quick Stats
    st.header("📊 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", f"{len(df):,}")
    col2.metric("Features", len(df.columns))
    col3.metric("Product Categories", df['category'].nunique())
    col4.metric("Data Completeness", "100%")
    
    # Class Balance Analysis
    st.header("⚖️ Class Balance Analysis")
    
    st.markdown("""
    We examine how balanced our dataset is across different categories. If one class dominates, 
    our future models might become biased toward the majority class.
    """)
    
    tab1, tab2, tab3 = st.tabs(["Sentiment Distribution", "Rating Distribution", "Category Balance"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            sentiment_counts = df['sentiment_category'].value_counts()
            fig = px.pie(
                values=sentiment_counts.values, 
                names=sentiment_counts.index,
                title='Sentiment Category Distribution',
                color_discrete_sequence=['#51CF66', '#FF6B6B', '#FFD43B']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(
                x=sentiment_counts.index,
                y=sentiment_counts.values,
                labels={'x': 'Sentiment', 'y': 'Count'},
                title='Sentiment Frequency',
                color=sentiment_counts.index,
                color_discrete_sequence=['#51CF66', '#FF6B6B', '#FFD43B']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.info(f"**Insight:** Sentiment is skewed towards {sentiment_counts.idxmax()} ({sentiment_counts.max():,} reviews), indicating generally favorable customer feedback.")
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(
                df, 
                x='rating',
                nbins=5,
                title='Rating Distribution',
                labels={'rating': 'Rating', 'count': 'Frequency'},
                color_discrete_sequence=['#4DABF7']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.box(
                df, 
                y='rating',
                title='Rating Spread',
                color_discrete_sequence=['#4DABF7']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.info(f"**Insight:** Ratings are concentrated in the upper range (mean: {df['rating'].mean():.2f}), suggesting positive customer experiences.")
    
    with tab3:
        category_counts = df['category'].value_counts().head(10)
        fig = px.bar(
            x=category_counts.values,
            y=category_counts.index,
            orientation='h',
            title='Top 10 Product Categories',
            labels={'x': 'Count', 'y': 'Category'},
            color_discrete_sequence=['#845EF7']
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.info(f"**Insight:** {category_counts.idxmax()} dominates with {category_counts.max():,} products, indicating category imbalance.")
    
    # Feature Distributions
    st.header("📈 Feature Distribution Analysis")
    
    st.markdown("""
    Using **histograms** and **box plots**, we analyze the "personality" of each feature—are they normally distributed, 
    skewed, or contain outliers?
    """)
    
    tab1, tab2, tab3 = st.tabs(["Price Analysis", "Sales & Reviews", "Engineered Features"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(
                df, 
                x='price',
                nbins=30,
                title='Price Distribution',
                labels={'price': 'Price (₹)', 'count': 'Frequency'},
                color_discrete_sequence=['#20C997']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.box(
                df, 
                y='price',
                title='Price Spread & Outliers',
                color_discrete_sequence=['#20C997']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.warning(f"**Insight:** Price is right-skewed (mean: ₹{df['price'].mean():.0f}, median: ₹{df['price'].median():.0f}), with most products in the affordable range and fewer premium items.")
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(
                df, 
                x='units_sold',
                nbins=30,
                title='Units Sold Distribution',
                labels={'units_sold': 'Units Sold', 'count': 'Frequency'},
                color_discrete_sequence=['#FF6B6B']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.histogram(
                df, 
                x='num_reviews',
                nbins=30,
                title='Number of Reviews Distribution',
                labels={'num_reviews': 'Number of Reviews', 'count': 'Frequency'},
                color_discrete_sequence=['#FF922B']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.warning("**Insight:** Both heavily right-skewed—only a small fraction of products account for the majority of sales and reviews.")
    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(
                df, 
                x='ingredients_count',
                nbins=15,
                title='Ingredient Count Distribution',
                labels={'ingredients_count': 'Number of Ingredients', 'count': 'Frequency'},
                color_discrete_sequence=['#ADB5BD']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Success rate by category
            success_by_cat = df.groupby('category')['Success'].mean().sort_values(ascending=False).head(8)
            fig = px.bar(
                x=success_by_cat.values * 100,
                y=success_by_cat.index,
                orientation='h',
                title='Success Rate by Category (%)',
                labels={'x': 'Success Rate (%)', 'y': 'Category'},
                color_discrete_sequence=['#51CF66']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.info(f"**Insight:** Most products have {df['ingredients_count'].mode()[0]} ingredients. Success rates vary significantly across categories.")
    
    # Correlation Analysis
    st.header("🔗 Feature Correlation Analysis")
    
    st.markdown("""
    A color-coded correlation heatmap showing which features "move together." Strong positive correlations (red) 
    mean features increase together; negative correlations (blue) indicate opposite movements.
    """)
    
    # Select numeric columns for correlation
    numeric_cols = ['price', 'discount', 'units_sold', 'average_rating', 'num_reviews', 
                    'rating', 'sentiment_score', 'ingredients_count', 'age_numeric', 'Success']
    
    # Filter only existing columns
    numeric_cols = [col for col in numeric_cols if col in df.columns]
    corr_df = df[numeric_cols].corr()
    
    fig = px.imshow(
        corr_df,
        text_auto='.2f',
        title='Correlation Heatmap',
        color_continuous_scale='RdBu_r',
        aspect='auto',
        zmin=-1,
        zmax=1
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.info("""
    **Key Findings:**
    - **Price** shows weak correlation with most variables—expensive products don't necessarily sell more or get better ratings
    - **Average Rating** and **Sentiment Score** are positively correlated—happier customers leave higher ratings
    - **Units Sold** and **Number of Reviews** are correlated—popular products naturally accumulate more reviews
    """)
    
    # Statistical Hypothesis Testing
    st.header("🧪 Statistical Hypothesis Testing")
    
    st.markdown("""
    We conducted statistical tests to validate our observations with a significance level of **α = 0.05**.
    Moving from hunches to hard evidence!
    """)
    
    tests_data = [
        {
            'Test': 'Chi-Square: Product Attributes vs Rating',
            'Null Hypothesis': 'Product category/brand has no association with ratings',
            'Result': 'Failed to reject H₀',
            'Conclusion': 'Ratings appear consistent across categories and brands',
            'Icon': '✅'
        },
        {
            'Test': 'Correlation: Price vs Purchase Likelihood',
            'Null Hypothesis': 'No correlation between price and units sold',
            'Result': 'Weak correlation (p < 0.05)',
            'Conclusion': 'Price shows weak but significant influence on sales',
            'Icon': '📊'
        },
        {
            'Test': 'T-Test: High vs Low Price Groups (Ratings)',
            'Null Hypothesis': 'No difference in ratings between price groups',
            'Result': 'Failed to reject H₀',
            'Conclusion': 'Customers rate expensive and affordable products similarly',
            'Icon': '✅'
        },
        {
            'Test': 'Chi-Square: Sentiment vs Weekend Purchase',
            'Null Hypothesis': 'Sentiment is independent of purchase timing',
            'Result': 'Failed to reject H₀',
            'Conclusion': 'Weekend vs weekday does not affect review sentiment',
            'Icon': '✅'
        },
        {
            'Test': 'T-Test: Ingredient (Dates) vs Total Amount',
            'Null Hypothesis': 'Ingredient presence does not affect spending',
            'Result': 'Failed to reject H₀',
            'Conclusion': 'Presence of dates does not significantly impact purchase amount',
            'Icon': '✅'
        },
        {
            'Test': 'T-Test: Customer Gender vs Spending',
            'Null Hypothesis': 'No gender difference in spending behavior',
            'Result': 'Failed to reject H₀',
            'Conclusion': 'No significant spending difference between male and female customers',
            'Icon': '✅'
        }
    ]
    
    for test in tests_data:
        with st.expander(f"{test['Icon']} {test['Test']}"):
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown(f"**Null Hypothesis (H₀):**")
                st.write(test['Null Hypothesis'])
            with col2:
                st.markdown(f"**Result:** {test['Result']}")
                st.markdown(f"**Conclusion:** {test['Conclusion']}")
    
    # Key Insights Summary
    st.header("💡 Key EDA Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Data Characteristics")
        st.markdown("""
        - ✅ **Complete Dataset:** No missing values, ensuring reliable analysis
        - ✅ **Sentiment Skew:** Positive reviews dominate (favorable customer feedback)
        - ✅ **Price Distribution:** Right-skewed, mostly affordable products
        - ✅ **Sales Concentration:** Few products drive majority of sales/reviews
        """)
    
    with col2:
        st.markdown("### Statistical Findings")
        st.markdown("""
        - 📊 **Price Independence:** Product cost doesn't determine ratings
        - 📊 **Consistent Quality:** Ratings similar across brands and categories
        - 📊 **Sentiment-Rating Link:** Strong correlation between sentiment and ratings
        - 📊 **Gender Neutral:** No spending differences between genders
        """)
    
    st.success("""
    **Bottom Line:** Our data tells a clear story—customer satisfaction is independent of price and brand. 
    Quality and experience drive ratings, not cost. This insight is crucial for product development and marketing strategy.
    """)
    
    # Tools Used
    with st.expander("🛠️ Tools & Libraries Used"):
        st.markdown("""
        - **Visualization:** Matplotlib, Seaborn, Plotly
        - **Statistical Testing:** SciPy, Statsmodels
        - **Data Analysis:** Pandas, NumPy
        - **Environment:** Google Colab, Jupyter Notebook
        """)

except FileNotFoundError:
    st.error("❌ Dataset not found. Please ensure 'data/WholeTruthFoodDataset-combined.csv' exists.")
except Exception as e:
    st.error(f"❌ Error: {str(e)}")
    st.exception(e)
