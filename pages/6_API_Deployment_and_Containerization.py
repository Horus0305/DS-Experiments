import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import pickle
import os

st.set_page_config(page_title="Model Testing", layout="wide")

st.title("🚀 Model Testing ")

st.info("**Objective:** Test model predictions interactively and compare results across multiple models")

# Introduction
st.markdown("""
This experiment allows you to:
- 🎮 **Interactive Testing** - Input product features and get instant predictions
- 🔄 **Multi-Model Comparison** - Compare predictions from multiple models side-by-side
- 📊 **Visual Analysis** - See probability distributions and confidence levels
- 🎯 **Real Scenarios** - Test with success and failure case examples
""")

# Load models and preprocessor
@st.cache_resource
def load_models_and_preprocessor():
    """Load all available models and the preprocessor"""
    models = {}
    model_path = "dsmodelpickl+preprocessor"
    
    # Load preprocessor
    preprocessor = None
    try:
        with open(os.path.join(model_path, "preprocessor.pkl"), "rb") as f:
            preprocessor = pickle.load(f)
    except Exception as e:
        st.error(f"Error loading preprocessor: {str(e)}")
        return None, None
    
    # Model file mappings (display name -> file name)
    # Non-overfitting models based on train/test performance comparison
    model_files = {
        "KNN": "knn_model.pkl",
        "ANN_DNN": "ann_dnn_model.pkl",
        "Logistic Regression": "logistic_regression_model.pkl",
        "Naive Bayes": "naive_bayes_model.pkl",
        "SVM": "svm_model.pkl",
        "LDA": "lda_model.pkl"
    }
    
    # Try to load each model
    for model_name, filename in model_files.items():
        try:
            filepath = os.path.join(model_path, filename)
            if os.path.exists(filepath):
                with open(filepath, "rb") as f:
                    models[model_name] = pickle.load(f)
        except Exception as e:
            st.warning(f"Could not load {model_name}: {str(e)}")
    
    return models, preprocessor

# Load models
loaded_models, preprocessor = load_models_and_preprocessor()

if loaded_models is None or preprocessor is None:
    st.error("❌ Failed to load models or preprocessor. Please check the model files.")
    st.stop()

# Model Selection for Comparison
st.header("🎯 Interactive Model Testing")

st.markdown("### Select Models to Compare")

# Available models with their accuracies (baseline models only)
available_models = {
    "KNN": {"accuracy": 0.9530, "recall": 0.9805, "color": "#51CF66"},  # Best baseline
    "ANN_DNN": {"accuracy": 0.9483, "recall": 0.9976, "color": "#FFD43B"},  # Good performance
    "LDA": {"accuracy": 0.9470, "recall": 1.0000, "color": "#CC5DE8"},  # Solid baseline
    "Naive Bayes": {"accuracy": 0.7556, "recall": 0.6330, "color": "#4DABF7"}  # Reference model
}

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("**Choose Models for Comparison:**")
    selected_models = []
    for model_name, info in available_models.items():
        if st.checkbox(f"{model_name} (Acc: {info['accuracy']:.2%})", value=True, key=f"model_{model_name}"):
            selected_models.append(model_name)
    
    if not selected_models:
        st.warning("⚠️ Please select at least one model")
    else:
        st.success(f"✅ {len(selected_models)} model(s) selected")

with col2:
    # st.markdown("**Selected Models Overview:**")
    if selected_models:
        # Create comparison chart
        fig = go.Figure()
        
        for model in selected_models:
            info = available_models[model]
            fig.add_trace(go.Bar(
                x=[model],
                y=[info['accuracy'] * 100],
                name=model,
                marker_color=info['color'],
                text=[f"{info['accuracy']:.2%}"],
                textposition='outside'
            ))
        
        # Dynamically set y-axis range based on selected models
        min_accuracy = min(available_models[m]['accuracy'] * 100 for m in selected_models)
        y_min = max(0, min_accuracy - 10)  # 10% buffer below minimum
        
        fig.update_layout(
            title="Selected Models Accuracy",
            yaxis_title="Accuracy (%)",
            showlegend=False,
            height=400,
            yaxis=dict(range=[y_min, 100])
        )
        
        st.plotly_chart(fig, use_container_width=True)

# Quick preset buttons (before inputs so they can set defaults)
st.markdown("### ⚡ Quick Test Presets")

preset_col1, preset_col2, preset_col3 = st.columns(3)

with preset_col1:
    if st.button("✅ Premium Success Case", use_container_width=True):
        st.session_state.preset_data = {
            "price": 1047, "discount": 0, "category": "Dark Chocolate",
            "ingredients_count": 4, "has_dates": 1, "has_cocoa": 1, "has_protein": 0,
            "packaging_type": "Paper-based", "season": "Winter", "customer_gender": "Female",
            "age_numeric": 55, "shelf_life": 12, "clean_label": 1
        }
        st.rerun()

with preset_col2:
    if st.button("❌ Budget Failure Case", use_container_width=True):
        st.session_state.preset_data = {
            "price": 349, "discount": 0, "category": "Muesli",
            "ingredients_count": 4, "has_dates": 0, "has_cocoa": 0, "has_protein": 0,
            "packaging_type": "Recyclable Plastic", "season": "Summer", "customer_gender": "Male",
            "age_numeric": 21, "shelf_life": 12, "clean_label": 0
        }
        st.rerun()

with preset_col3:
    if st.button("🔄 Protein Bar Case", use_container_width=True):
        st.session_state.preset_data = {
            "price": 1000, "discount": 0, "category": "Protein Bars",
            "ingredients_count": 5, "has_dates": 1, "has_cocoa": 1, "has_protein": 1,
            "packaging_type": "Recyclable Plastic", "season": "Winter", "customer_gender": "Female",
            "age_numeric": 30, "shelf_life": 24, "clean_label": 1
        }
        st.rerun()

# Get preset values or use defaults
if 'preset_data' in st.session_state:
    preset = st.session_state.preset_data
    default_price = preset["price"]
    default_discount = preset["discount"]
    default_category = preset["category"]
    default_ingredients = preset["ingredients_count"]
    default_has_dates = preset["has_dates"]
    default_has_cocoa = preset["has_cocoa"]
    default_has_protein = preset["has_protein"]
    default_packaging = preset["packaging_type"]
    default_season = preset["season"]
    default_gender = preset["customer_gender"]
    default_age = preset["age_numeric"]
    default_shelf_life = preset["shelf_life"]
    default_clean_label = preset["clean_label"]
    # Clear preset after using it
    del st.session_state.preset_data
else:
    # Default values
    default_price = 1047
    default_discount = 0
    default_category = "Dark Chocolate"
    default_ingredients = 4
    default_has_dates = 1
    default_has_cocoa = 1
    default_has_protein = 0
    default_packaging = "Paper-based"
    default_season = "Winter"
    default_gender = "Female"
    default_age = 55
    default_shelf_life = 12
    default_clean_label = 1

# Interactive Input Section
st.header("🎮 Input Product Features")

st.markdown("Enter product and customer details to get predictions from all selected models:")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### 📦 Product Features")
    price = st.number_input("💰 Price (INR)", 0, 2000, default_price, 10, help="Product price in Indian Rupees")
    discount = st.slider("🏷️ Discount (%)", 0, 100, default_discount, 1, help="Discount percentage")
    
    # Get index for category
    categories = ["Dark Chocolate", "Muesli", "Energy Bars", "Peanut Butter", "Protein Bars"]
    category_index = categories.index(default_category) if default_category in categories else 0
    category = st.selectbox("📂 Category", categories, index=category_index)
    
    ingredients_count = st.number_input("🧪 Ingredients Count", 1, 20, default_ingredients, 1)
    shelf_life = st.number_input("📅 Shelf Life (months)", 1, 36, default_shelf_life, 1)

with col2:
    st.markdown("#### 🌟 Product Attributes")
    has_dates = st.selectbox("🌴 Has Dates", [0, 1], index=default_has_dates, format_func=lambda x: "Yes" if x == 1 else "No")
    has_cocoa = st.selectbox("🍫 Has Cocoa", [0, 1], index=default_has_cocoa, format_func=lambda x: "Yes" if x == 1 else "No")
    has_protein = st.selectbox("💪 Has Protein", [0, 1], index=default_has_protein, format_func=lambda x: "Yes" if x == 1 else "No")
    clean_label = st.selectbox("✨ Clean Label", [0, 1], index=default_clean_label, format_func=lambda x: "Yes" if x == 1 else "No")
    
    # Get index for packaging
    packaging_types = ["Paper-based", "Recyclable Plastic", "Glass", "Biodegradable"]
    packaging_index = packaging_types.index(default_packaging) if default_packaging in packaging_types else 0
    packaging_type = st.selectbox("📦 Packaging Type", packaging_types, index=packaging_index)

with col3:
    st.markdown("#### 👥 Customer & Context")
    
    # Get index for season
    seasons = ["Winter", "Summer", "Monsoon", "Spring"]
    season_index = seasons.index(default_season) if default_season in seasons else 0
    season = st.selectbox("🌦️ Season", seasons, index=season_index)
    
    # Get index for gender
    genders = ["Male", "Female"]
    gender_index = genders.index(default_gender) if default_gender in genders else 1
    customer_gender = st.selectbox("👤 Customer Gender", genders, index=gender_index)
    gender_encoded = 1.0 if customer_gender == "Female" else 0.0
    
    age_numeric = st.number_input("🎂 Customer Age", 18, 100, default_age, 1)

st.markdown("---")

# Prediction Button
if st.button("🚀 Get Predictions from All Models", type="primary", use_container_width=True):
    if not selected_models:
        st.error("⚠️ Please select at least one model to get predictions!")
    else:
        st.markdown("---")
        st.header("📊 Multi-Model Prediction Results")
        
        # Calculate prediction for each model using actual models
        def calculate_prediction(model_name, features_dict):
            """Calculate prediction probability using actual trained model"""
            try:
                # Create DataFrame with proper column names
                features_df = pd.DataFrame([features_dict])
                
                # Apply preprocessing
                features_processed = preprocessor.transform(features_df)
                
                # Get model
                model = loaded_models[model_name]
                
                # Get prediction probability
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(features_processed)
                    probability = float(proba[0][1])  # Probability of class 1 (success)
                else:
                    # For models without predict_proba, use decision function
                    prediction = model.predict(features_processed)
                    probability = float(prediction[0])
                
                return probability
                
            except Exception as e:
                st.error(f"Error predicting with {model_name}: {str(e)}")
                return 0.5  # Return neutral probability on error
        
        # Collect features
        features = {
            'price': price,
            'discount': discount,
            'category': category,
            'ingredients_count': ingredients_count,
            'has_dates': has_dates,
            'has_cocoa': has_cocoa,
            'has_protein': has_protein,
            'packaging_type': packaging_type,
            'season': season,
            'customer_gender': gender_encoded,
            'age_numeric': age_numeric,
            'shelf_life': shelf_life,
            'clean_label': clean_label
        }
        
        # Get predictions from all selected models
        predictions = {}
        for model in selected_models:
            prob = calculate_prediction(model, features)
            predictions[model] = {
                'probability': prob,
                'prediction': 1 if prob >= 0.5 else 0,
                'confidence': 'High' if (prob >= 0.8 or prob <= 0.2) else 'Medium' if (prob >= 0.6 or prob <= 0.4) else 'Low'
            }
        
        # Display Input Summary
        st.markdown("### 📝 Input Summary")
        input_col1, input_col2, input_col3 = st.columns(3)
        
        with input_col1:
            st.markdown("**Product:**")
            st.write(f"- Price: ₹{price}")
            st.write(f"- Category: {category}")
            st.write(f"- Discount: {discount}%")
            st.write(f"- Ingredients: {ingredients_count}")
        
        with input_col2:
            st.markdown("**Attributes:**")
            st.write(f"- Cocoa: {'✅' if has_cocoa else '❌'}")
            st.write(f"- Protein: {'✅' if has_protein else '❌'}")
            st.write(f"- Dates: {'✅' if has_dates else '❌'}")
            st.write(f"- Clean Label: {'✅' if clean_label else '❌'}")
        
        with input_col3:
            st.markdown("**Context:**")
            st.write(f"- Season: {season}")
            st.write(f"- Packaging: {packaging_type}")
            st.write(f"- Customer: {customer_gender}, {age_numeric} yrs")
        
        st.markdown("---")
        
        # Side-by-Side Model Comparison
        st.markdown("### 🔄 Model Comparison")
        
        # Create columns based on number of selected models
        if len(selected_models) == 1:
            cols = [st.container()]
        elif len(selected_models) == 2:
            cols = st.columns(2)
        elif len(selected_models) == 3:
            cols = st.columns(3)
        else:
            cols = st.columns(4)
        
        for idx, model in enumerate(selected_models):
            with cols[idx % len(cols)]:
                result = predictions[model]
                prob = result['probability']
                pred_class = result['prediction']
                confidence = result['confidence']
                
                # Model card
                st.markdown(f"#### {model}")
                
                # Gauge chart
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Success %", 'font': {'size': 14}},
                    gauge={
                        'axis': {'range': [None, 100], 'tickwidth': 1},
                        'bar': {'color': available_models[model]['color']},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [0, 40], 'color': '#FFE5E5'},
                            {'range': [40, 70], 'color': '#FFF4E5'},
                            {'range': [70, 100], 'color': '#E7F5FF'}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 3},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                fig.update_layout(height=200, margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(fig, use_container_width=True)
                
                # Metrics
                st.metric("Probability", f"{prob:.1%}")
                
                if pred_class == 1:
                    st.success("✅ **SUCCESS**")
                else:
                    st.error("❌ **FAILURE**")
                
                st.info(f"🎯 Confidence: **{confidence}**")
        
        # Comparison Chart
        st.markdown("---")
        st.markdown("### 📊 Visual Comparison")
        
        comp_col1, comp_col2 = st.columns(2)
        
        with comp_col1:
            # Bar chart comparing probabilities
            fig = go.Figure()
            
            for model in selected_models:
                prob = predictions[model]['probability']
                color = available_models[model]['color']
                
                fig.add_trace(go.Bar(
                    x=[model],
                    y=[prob * 100],
                    name=model,
                    marker_color=color,
                    text=[f"{prob:.1%}"],
                    textposition='outside'
                ))
            
            fig.add_hline(y=50, line_dash="dash", line_color="red", 
                         annotation_text="Decision Threshold (50%)")
            
            fig.update_layout(
                title="Success Probability by Model",
                yaxis_title="Probability (%)",
                xaxis_title="Model",
                showlegend=False,
                height=400,
                yaxis=dict(range=[0, 105])
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with comp_col2:
            # Detailed comparison table
            comparison_data = []
            for model in selected_models:
                result = predictions[model]
                comparison_data.append({
                    "Model": model,
                    "Probability": f"{result['probability']:.2%}",
                    "Prediction": "✅ Success" if result['prediction'] == 1 else "❌ Failure",
                    "Confidence": result['confidence']
                })
            
            df_comparison = pd.DataFrame(comparison_data)
            st.markdown("#### Detailed Results")
            st.dataframe(df_comparison, use_container_width=True, hide_index=True)
            
            # Consensus
            success_count = sum(1 for m in selected_models if predictions[m]['prediction'] == 1)
            failure_count = len(selected_models) - success_count
            
            st.markdown("#### 🤝 Model Consensus")
            
            consensus_fig = go.Figure(data=[
                go.Pie(
                    labels=['Success', 'Failure'],
                    values=[success_count, failure_count],
                    marker=dict(colors=['#51CF66', '#FF6B6B']),
                    hole=0.4,
                    textinfo='label+value',
                    textposition='inside'
                )
            ])
            consensus_fig.update_layout(
                height=250,
                showlegend=False,
                annotations=[dict(text=f'{success_count}/{len(selected_models)}', x=0.5, y=0.5, 
                                font_size=20, showarrow=False)]
            )
            st.plotly_chart(consensus_fig, use_container_width=True)
            
            if success_count == len(selected_models):
                st.success("🎉 **All models predict SUCCESS!** Strong consensus.")
            elif failure_count == len(selected_models):
                st.error("⚠️ **All models predict FAILURE!** Strong consensus.")
            else:
                st.warning(f"⚖️ **Mixed predictions:** {success_count} Success, {failure_count} Failure")
        
        # Recommendation
        st.markdown("---")
        st.markdown("### 💡 Recommendation")
        
        avg_prob = np.mean([predictions[m]['probability'] for m in selected_models])
        
        if avg_prob >= 0.7:
            st.success(f"""
            ✅ **Strong Success Potential ({avg_prob:.1%} avg probability)**
            
            **Action:** Proceed with product launch
            - All indicators suggest high market acceptance
            - Strong performance across multiple models
            - Consider premium positioning strategy
            """)
        elif avg_prob >= 0.5:
            st.info(f"""
            ⚖️ **Moderate Success Potential ({avg_prob:.1%} avg probability)**
            
            **Action:** Consider A/B testing before full launch
            - Mixed signals from models
            - Test with small customer segment first
            - Gather feedback and iterate
            """)
        else:
            st.warning(f"""
            ⚠️ **Low Success Potential ({avg_prob:.1%} avg probability)**
            
            **Action:** Review and improve product strategy
            - Consider increasing quality indicators (price, ingredients)
            - Add premium features (cocoa, protein, dates)
            - Ensure clean label certification
            - Improve packaging sustainability
            """)

# Feature Importance Insights
st.markdown("---")
st.header("🔍 Key Success Factors")

col1, col2 = st.columns(2)

with col1:
    # Feature importance chart
    features_importance = {
        'Price': 35,
        'Has Cocoa': 25,
        'Has Protein': 15,
        'Clean Label': 10,
        'Has Dates': 10,
        'Category': 5
    }
    
    fig = go.Figure(go.Bar(
        x=list(features_importance.values()),
        y=list(features_importance.keys()),
        orientation='h',
        marker=dict(
            color=list(features_importance.values()),
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Importance %")
        ),
        text=[f"{v}%" for v in features_importance.values()],
        textposition='outside'
    ))
    
    fig.update_layout(
        title="Feature Importance for Success Prediction",
        xaxis_title="Importance (%)",
        yaxis_title="Feature",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("### 📈 What Drives Success?")
    
    st.markdown("""
    **High Impact Features:**
    
    1. **💰 Price (35%)**: Premium pricing signals quality
       - ₹1000+ → High success probability
       - ₹700-1000 → Moderate probability
       - <₹450 → Lower probability
    
    2. **🍫 Has Cocoa (25%)**: Premium ingredient
       - Cocoa products perform significantly better
       - Associated with quality and indulgence
    
    3. **💪 Has Protein (15%)**: Health appeal
       - Protein adds functional benefit
       - Appeals to fitness-conscious customers
    
    4. **✨ Clean Label (10%)**: Trust factor
       - Clean label certification boosts confidence
       - Aligns with health-conscious trends
    
    5. **🌴 Has Dates (10%)**: Natural sweetener
       - Dates indicate natural ingredients
       - Premium positioning
    """)

# Model Input Features Reference
st.markdown("---")
st.header("📋 Model Input Features (13 Features)")

feature_col1, feature_col2, feature_col3 = st.columns(3)

with feature_col1:
    st.markdown("### 📦 Product Attributes")
    st.markdown("""
    - **price** (float): Product price in INR
    - **discount** (float): Discount % (0-100)
    - **category** (string): Product category
    - **ingredients_count** (int): # of ingredients
    - **shelf_life** (float): Shelf life in months
    """)

with feature_col2:
    st.markdown("### 🌟 Ingredient Flags")
    st.markdown("""
    - **has_dates** (0/1): Contains dates
    - **has_cocoa** (0/1): Contains cocoa
    - **has_protein** (0/1): Contains protein
    - **clean_label** (0/1): Clean label certified
    """)

with feature_col3:
    st.markdown("### 🎯 Context Features")
    st.markdown("""
    - **packaging_type** (string): Packaging material
    - **season** (string): Launch season
    - **customer_gender** (0/1): Target gender
    - **age_numeric** (float): Target age (18-100)
    """)

# Conclusion
st.markdown("---")
st.header("🏁 Summary")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### ✅ What We Built")
    st.markdown("""
    - **Interactive Testing Interface** for real-time predictions
    - **Multi-Model Comparison** to see consensus across models
    - **Visual Analytics** with gauges, charts, and tables
    - **Quick Presets** for common test scenarios
    - **Feature Importance** insights for decision-making
    """)

with col2:
    st.markdown("### 🚀 Business Value")
    st.markdown("""
    - **Data-Driven Decisions** on product launches
    - **Risk Assessment** before market entry
    - **Success Factors** clearly identified
    - **Confidence Levels** from multiple models
    - **Actionable Recommendations** for each prediction
    """)

st.success("""
### 🎉 Model Testing Complete!

You can now test any product configuration and get instant predictions from multiple ML models. 
Use this tool to make informed decisions about **The Whole Truth Foods** product launches!

**Next Steps:**
1. Test different product configurations
2. Analyze which features drive success
3. Use insights to optimize product strategy
4. Deploy the best performing model to production
""")

st.markdown("---")
st.markdown("*Developed for The Whole Truth Foods - Product Success Prediction System*")
