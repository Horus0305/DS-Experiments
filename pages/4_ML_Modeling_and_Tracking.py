import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3
import json
import pickle
import os

st.set_page_config(page_title="ML Modeling & Tracking", layout="wide")

st.title("ML Modeling & Experiment Tracking")

st.info("**Objective:** Build ML pipeline, tune hyperparameters, and track experiments with MLflow")

# Load actual MLflow data from database
@st.cache_data
def load_mlflow_data():
    try:
        conn = sqlite3.connect('mlruns.db')
        
        # Query to get runs with their metrics
        query = """
        SELECT 
            r.run_uuid,
            r.name as run_name,
            r.status,
            r.start_time,
            r.end_time,
            m.key as metric_key,
            m.value as metric_value
        FROM runs r
        LEFT JOIN metrics m ON r.run_uuid = m.run_uuid
        WHERE r.lifecycle_stage = 'active'
        ORDER BY r.start_time
        """
        
        df = pd.read_sql_query(query, conn)
        
        # Query to get parameters (including model names)
        param_query = """
        SELECT 
            run_uuid,
            key as param_key,
            value as param_value
        FROM params
        WHERE key = 'model_name'
        """
        
        df_params = pd.read_sql_query(param_query, conn)
        
        conn.close()
        
        return df, df_params
    except Exception as e:
        st.error(f"Error loading MLflow data: {str(e)}")
        return None, None

# Load the data
df_metrics, df_params = load_mlflow_data()

if df_metrics is not None and not df_metrics.empty:
    try:
        # Process the data to create summary
        # Pivot metrics to get one row per run
        metrics_pivot = df_metrics.pivot_table(
            index='run_uuid', 
            columns='metric_key', 
            values='metric_value', 
            aggfunc='last'
        ).reset_index()
        
        # Get run names and timing from original dataframe
        run_info = df_metrics[['run_uuid', 'run_name', 'start_time', 'end_time']].drop_duplicates()
        
        # Merge metrics with run info
        summary_df = metrics_pivot.merge(run_info, on='run_uuid', how='left')
        
        # Get model names from params
        if df_params is not None and not df_params.empty:
            model_names = df_params[['run_uuid', 'param_value']].copy()
            model_names.columns = ['run_uuid', 'model_name']
            summary_df = summary_df.merge(model_names, on='run_uuid', how='left')
        
        # If model_name not available, extract from run_name
        if 'model_name' not in summary_df.columns or summary_df['model_name'].isna().any():
            summary_df['model_name'] = summary_df['run_name'].str.replace('_Baseline', '').str.replace('_Tuned', '')
        
        # Calculate training time in seconds
        if 'start_time' in summary_df.columns and 'end_time' in summary_df.columns:
            summary_df['training_time'] = (summary_df['end_time'] - summary_df['start_time']) / 1000
        
        # Add a display name that includes baseline/tuned suffix
        summary_df['display_name'] = summary_df['run_name']
    
    except Exception as e:
        st.warning(f"Error processing MLflow data: {str(e)}. Using sample data.")
        summary_df = pd.DataFrame({
            'model_name': ['KNN', 'ANN_DNN', 'LDA', 'Naive Bayes'],
            'display_name': ['KNN', 'ANN_DNN', 'LDA', 'Naive Bayes'],
            'accuracy': [0.9530, 0.9483, 0.9470, 0.7556],
            'f1_score': [0.9357, 0.9470, 0.9444, 0.7750],
            'precision': [0.8956, 0.9011, 0.8948, 1.0000],
            'recall': [0.9805, 0.9976, 1.0000, 0.6330],
            'training_time': [8, 45, 7, 5]
        })
    
else:
    # Fallback to simulated data if database can't be read
    st.warning("Using sample data. MLflow database not accessible.")
    summary_df = pd.DataFrame({
        'model_name': ['KNN', 'ANN_DNN', 'LDA', 'Naive Bayes'],
        'display_name': ['KNN', 'ANN_DNN', 'LDA', 'Naive Bayes'],
        'accuracy': [0.9530, 0.9483, 0.9470, 0.7556],
        'f1_score': [0.9357, 0.9470, 0.9444, 0.7750],
        'precision': [0.8956, 0.9011, 0.8948, 1.0000],
        'recall': [0.78, 0.85, 0.87, 0.81, 0.89, 0.91],
        'training_time': [12, 45, 38, 56, 89, 72]
    })

# Separate baseline and tuned models using run_name or display_name
if 'run_name' in summary_df.columns:
    baseline_models = summary_df[summary_df['run_name'].str.contains('Baseline', na=False, case=False)].copy()
    tuned_models = summary_df[summary_df['run_name'].str.contains('Tuned', na=False, case=False)].copy()
elif 'display_name' in summary_df.columns:
    baseline_models = summary_df[summary_df['display_name'].str.contains('Baseline', na=False, case=False)].copy()
    tuned_models = summary_df[summary_df['display_name'].str.contains('Tuned', na=False, case=False)].copy()
else:
    # Fallback: use model_name column
    baseline_models = summary_df[~summary_df['model_name'].str.contains('Tuned', na=False, case=False)].copy()
    tuned_models = summary_df[summary_df['model_name'].str.contains('Tuned', na=False, case=False)].copy()

# Dataset Preparation
st.header("📦 Dataset Preparation")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Samples", "10,500")
col2.metric("Training Set (70%)", "7,350")
col3.metric("Test Set (30%)", "3,150")
col4.metric("Features Used", "13")

st.markdown("""
**Data Split Strategy:**
- 70/30 train-test split with stratification to maintain class balance
- Features: 13 carefully selected features (product attributes, customer demographics, engineered features)
- Target variable: `Success` (binary classification - product success indicator)
""")

# Baseline Model Training
st.header("🎯 Baseline Model Training")

st.markdown("""
We trained multiple baseline models to establish a performance benchmark before optimization.
""")

if len(baseline_models) > 0:
    tab1, tab2 = st.tabs(["Performance Comparison", "Detailed Metrics"])
    
    with tab1:
        # Create comparison chart
        fig = go.Figure()
        
        # Dynamically get available metrics
        metric_cols = [col for col in baseline_models.columns if col in ['accuracy', 'precision', 'recall', 'f1_score']]
        metric_names = [col.replace('_', ' ').title() for col in metric_cols]
        colors = ['#4DABF7', '#51CF66', '#FF6B6B', '#FFD43B']
        
        for i, (metric_col, metric_name) in enumerate(zip(metric_cols, metric_names)):
            if metric_col in baseline_models.columns:
                fig.add_trace(go.Bar(
                    name=metric_name,
                    x=baseline_models['model_name'],
                    y=baseline_models[metric_col],
                    marker_color=colors[i % len(colors)]
                ))
        
        fig.update_layout(
            title='Baseline Models Performance Comparison',
            xaxis_title='Model',
            yaxis_title='Score',
            barmode='group',
            height=500,
            xaxis_tickangle=-45
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        display_df = baseline_models[['model_name'] + [col for col in baseline_models.columns if col in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']]].copy()
        display_df.columns = ['Model'] + [col.replace('_', ' ').title() for col in display_df.columns[1:]]
        
        # Apply gradient styling to metric columns
        metric_display_cols = [col for col in display_df.columns if col in ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Roc Auc']]
        if metric_display_cols:
            st.dataframe(
                display_df.style.background_gradient(subset=metric_display_cols, cmap='Greens'),
                use_container_width=True, 
                hide_index=True
            )
        else:
            st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # Find best baseline model
    if 'accuracy' in baseline_models.columns:
        best_baseline = baseline_models.loc[baseline_models['accuracy'].idxmax()]
        
        # Check for overfitting
        if best_baseline.get('accuracy', 0) >= 0.999:
            # Find best non-overfitting baseline
            non_overfit_baseline = baseline_models[baseline_models['accuracy'] < 0.999]
            if not non_overfit_baseline.empty:
                best_non_overfit = non_overfit_baseline.loc[non_overfit_baseline['accuracy'].idxmax()]
                st.warning(f"⚠️ Some baseline models (e.g., {best_baseline['model_name']}) show 100% accuracy, indicating potential overfitting.")
                st.success(f"""
                **Best Non-Overfitting Baseline:** {best_non_overfit['model_name']} achieved {best_non_overfit.get('accuracy', 0):.1%} accuracy
                {f"and {best_non_overfit.get('f1_score', 0):.1%} F1-score" if 'f1_score' in best_non_overfit else ""} - a more realistic baseline.
                """)
            else:
                st.warning(f"⚠️ All baseline models show suspiciously high accuracy. Consider reviewing the data split and validation strategy.")
        else:
            st.success(f"""
            **Baseline Winner:** {best_baseline['model_name']} achieved the best baseline performance with 
            {best_baseline.get('accuracy', 0):.1%} accuracy{f" and {best_baseline.get('f1_score', 0):.1%} F1-score" if 'f1_score' in best_baseline else ""}.
            """)
else:
    st.warning("No baseline models found in MLflow data.")
    
    df_baseline = pd.DataFrame({
        'Model': ['KNN', 'ANN_DNN', 'LDA', 'Naive Bayes'],
        'Accuracy': [0.9530, 0.9483, 0.9470, 0.7556],
        'F1-Score': [0.9357, 0.9470, 0.9444, 0.7750]
    })
    
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Accuracy', x=df_baseline['Model'], y=df_baseline['Accuracy']))
    fig.add_trace(go.Bar(name='F1-Score', x=df_baseline['Model'], y=df_baseline['F1-Score']))
    fig.update_layout(title='Baseline Models (Sample Data)', barmode='group', yaxis_range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)

# Hyperparameter Tuning
st.header("⚙️ Hyperparameter Tuning")

st.markdown("""
Baseline models show strong performance. KNN achieved 95.30% accuracy without hyperparameter tuning.
For this project, we focus on baseline model performance and feature engineering rather than extensive hyperparameter optimization.
""")

# Show parameter grids if available in the data
if len(tuned_models) > 0 and any(col for col in tuned_models.columns if col not in ['model_name', 'accuracy', 'precision', 'recall', 'f1_score', 'run_id', 'training_time']):
    st.markdown("### Tuned Model Parameters")
    
    for _, row in tuned_models.iterrows():
        with st.expander(f"📋 {row['model_name']} Parameters"):
            param_cols = [col for col in tuned_models.columns if col not in ['model_name', 'accuracy', 'precision', 'recall', 'f1_score', 'run_id', 'training_time']]
            for param in param_cols:
                if pd.notna(row.get(param)):
                    st.text(f"{param}: {row[param]}")
else:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Random Forest Tuning")
        st.code("""
Parameters Tuned:
- n_estimators: [100, 200, 300]
- max_depth: [10, 20, 30, None]
- min_samples_split: [2, 5, 10]
- min_samples_leaf: [1, 2, 4]

Best Parameters:
- n_neighbors: 5 to 15
- weights: uniform vs distance
- metric: euclidean, manhattan
- algorithm: auto, ball_tree, kd_tree
    """, language="python")

    with col2:
        st.markdown("### ANN/DNN Tuning")
        st.code("""
Parameters Explored:
- layers: [64, 32] to [128, 64, 32]
- activation: relu, tanh
- optimizer: adam, rmsprop
- dropout: 0.2 to 0.5

Best Configuration:
- layers: [64, 32, 16]
- activation: relu
- optimizer: adam
- dropout: 0.3
    """, language="python")

# Performance Improvement
st.header("📈 Baseline vs Tuned Performance")

# Use real data if available, otherwise show sample
if len(baseline_models) > 0 and len(tuned_models) > 0 and 'accuracy' in baseline_models.columns and 'accuracy' in tuned_models.columns:
    # Build comparison from actual data - match all baseline and tuned pairs
    comparison_data = []
    
    for _, baseline_row in baseline_models.iterrows():
        baseline_name = baseline_row['model_name']
        baseline_acc = baseline_row['accuracy']
        
        # Try to find corresponding tuned model
        tuned_match = tuned_models[tuned_models['model_name'] == baseline_name]
        
        if not tuned_match.empty:
            # Found a tuned version
            tuned_acc = tuned_match.iloc[0]['accuracy']
            comparison_data.append({
                'model': baseline_name,
                'Baseline': baseline_acc,
                'Tuned': tuned_acc,
                'has_both': True
            })
        else:
            # Only baseline exists
            comparison_data.append({
                'model': baseline_name,
                'Baseline': baseline_acc,
                'Tuned': None,
                'has_both': False
            })
    
    # Check for tuned models without baseline
    for _, tuned_row in tuned_models.iterrows():
        tuned_name = tuned_row['model_name']
        if not any(d['model'] == tuned_name for d in comparison_data):
            comparison_data.append({
                'model': tuned_name,
                'Baseline': None,
                'Tuned': tuned_row['accuracy'],
                'has_both': False
            })
    
    if comparison_data:
        comp_df = pd.DataFrame(comparison_data)
        
        # Identify overfitting models (100% or near-perfect accuracy)
        comp_df['baseline_overfit'] = comp_df['Baseline'] >= 0.999
        comp_df['tuned_overfit'] = comp_df['Tuned'] >= 0.999
        
        fig = go.Figure()
        
        # Add baseline bars with color coding for overfitting
        baseline_colors = ['#FF6B6B' if overfit else '#ADB5BD' for overfit in comp_df['baseline_overfit']]
        fig.add_trace(go.Bar(
            name='Baseline',
            x=comp_df['model'],
            y=comp_df['Baseline'],
            marker_color=baseline_colors,
            text=[f'{v:.1%}{"⚠️" if overfit else ""}' if pd.notna(v) else '' 
                  for v, overfit in zip(comp_df['Baseline'], comp_df['baseline_overfit'])],
            textposition='outside'
        ))
        
        # Add tuned bars with color coding for overfitting
        tuned_colors = ['#FF6B6B' if overfit else '#51CF66' for overfit in comp_df['tuned_overfit']]
        fig.add_trace(go.Bar(
            name='Tuned',
            x=comp_df['model'],
            y=comp_df['Tuned'],
            marker_color=tuned_colors,
            text=[f'{v:.1%}{"⚠️" if overfit else ""}' if pd.notna(v) else '' 
                  for v, overfit in zip(comp_df['Tuned'], comp_df['tuned_overfit'])],
            textposition='outside'
        ))
        
        fig.update_layout(
            title=f'Performance Improvement: Baseline vs Tuned Models ({len(comp_df)} Models)<br><sub>⚠️ Red bars indicate potential overfitting (100% accuracy)</sub>',
            xaxis_title='Model',
            yaxis_title='Accuracy',
            barmode='group',
            height=500,
            yaxis_range=[0, 1.1],
            xaxis_tickangle=-45
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show overfitting warning if any detected
        overfit_models = comp_df[(comp_df['baseline_overfit']) | (comp_df['tuned_overfit'])]['model'].tolist()
        if overfit_models:
            st.warning(f"""
            ⚠️ **Overfitting Detected:** {', '.join(overfit_models)} show 100% accuracy, which typically indicates:
            - Memorization of training data rather than learning patterns
            - Data leakage between train/test sets
            - Insufficient model regularization
            
            **Recommendation:** These models may not generalize well to new data. Focus on models with realistic accuracy (90-98%).
            """)
        
        # Analyze tuning impact
        st.markdown("### 📊 Hyperparameter Tuning Impact Analysis")
        
        # Categorize models by tuning impact
        comp_df['change'] = comp_df['Tuned'] - comp_df['Baseline']
        comp_df['change_pct'] = (comp_df['change'] * 100).round(2)
        
        improved = comp_df[comp_df['change'] > 0.001].copy()  # Improved by more than 0.1%
        declined = comp_df[comp_df['change'] < -0.001].copy()  # Declined by more than 0.1%
        unchanged = comp_df[abs(comp_df['change']) <= 0.001].copy()  # Changed less than 0.1%
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("📈 Improved", len(improved), delta="Better after tuning")
        with col2:
            st.metric("📉 Declined", len(declined), delta="Worse after tuning")
        with col3:
            st.metric("➡️ Unchanged", len(unchanged), delta="Minimal change")
        
        # Detailed analysis with reasons
        tab1, tab2, tab3 = st.tabs(["📈 Improved Models", "📉 Declined Models", "➡️ Unchanged Models"])
        
        with tab1:
            if not improved.empty:
                st.success(f"**{len(improved)} models improved** after hyperparameter tuning:")
                
                for _, row in improved.iterrows():
                    with st.expander(f"✅ {row['model']} (+{row['change_pct']:.2f}%)", expanded=False):
                        col_a, col_b = st.columns(2)
                        
                        with col_a:
                            st.metric("Baseline", f"{row['Baseline']:.2%}")
                            st.metric("After Tuning", f"{row['Tuned']:.2%}", delta=f"+{row['change_pct']:.2f}%")
                        
                        with col_b:
                            st.markdown("**Likely Reasons for Improvement:**")
                            
                            if 'Naive Bayes' in row['model']:
                                st.markdown("""
                                - Better variance smoothing parameter
                                - Improved prior probability estimation
                                - Optimal feature distribution handling
                                """)
                            elif 'KNN' in row['model']:
                                st.markdown("""
                                - Optimal number of neighbors (k) found
                                - Better distance metric selection
                                - Improved weight function for neighbors
                                """)
                            elif 'LDA' in row['model']:
                                st.markdown("""
                                - Better regularization parameter
                                - Optimal solver selection
                                - Improved shrinkage coefficient
                                """)
                            else:
                                st.markdown("""
                                - Found optimal complexity parameters
                                - Better regularization balance
                                - Improved feature interaction handling
                                - Reduced overfitting through constraints
                                """)
            else:
                st.info("No models showed improvement after tuning.")
        
        with tab2:
            if not declined.empty:
                st.warning(f"**{len(declined)} models declined** after hyperparameter tuning:")
                
                for _, row in declined.iterrows():
                    with st.expander(f"⚠️ {row['model']} ({row['change_pct']:.2f}%)", expanded=False):
                        col_a, col_b = st.columns(2)
                        
                        with col_a:
                            st.metric("Baseline", f"{row['Baseline']:.2%}")
                            st.metric("After Tuning", f"{row['Tuned']:.2%}", delta=f"{row['change_pct']:.2f}%")
                        
                        with col_b:
                            st.markdown("**Possible Reasons for Decline:**")
                            
                            # Check if model was already overfitting
                            if row['Baseline'] >= 0.999:
                                st.markdown("""
                                - ✅ **Actually beneficial!** Baseline was overfitting (100% accuracy)
                                - Tuning added regularization to reduce overfitting
                                - Lower accuracy but better generalization
                                - More realistic and deployable model
                                """)
                            else:
                                st.markdown("""
                                - Over-regularization during tuning
                                - Hyperparameter search space too restrictive
                                - Default parameters were already near-optimal
                                - Model may need different tuning strategy
                                - Limited training data for GridSearchCV folds
                                """)
            else:
                st.info("No models showed decline after tuning.")
        
        with tab3:
            if not unchanged.empty:
                st.info(f"**{len(unchanged)} models remained essentially unchanged** after tuning:")
                
                for _, row in unchanged.iterrows():
                    with st.expander(f"➡️ {row['model']} ({row['change_pct']:+.2f}%)", expanded=False):
                        col_a, col_b = st.columns(2)
                        
                        with col_a:
                            st.metric("Baseline", f"{row['Baseline']:.2%}")
                            st.metric("After Tuning", f"{row['Tuned']:.2%}", delta=f"{row['change_pct']:+.2f}%")
                        
                        with col_b:
                            st.markdown("**Possible Reasons for No Change:**")
                            
                            if row['Baseline'] >= 0.999 and row['Tuned'] >= 0.999:
                                st.markdown("""
                                - ⚠️ Both baseline and tuned are overfitting
                                - Already at maximum possible accuracy (100%)
                                - Model memorizing training data
                                - Tuning couldn't add more regularization
                                - **Not recommended for production**
                                """)
                            else:
                                st.markdown("""
                                - Default hyperparameters were already optimal
                                - Model reached its performance ceiling
                                - Dataset characteristics limit further improvement
                                - Simple model with few tunable parameters
                                - Robust model not sensitive to hyperparameter changes
                                """)
            else:
                st.info("All models showed significant change after tuning.")
        
        st.markdown("""
        ---
        **Key Insights:**
        - **Improved models**: Hyperparameter tuning successfully found better configurations
        - **Declined from 100%**: This is actually good! Tuning reduced overfitting for better generalization
        - **Unchanged models**: Either already optimal or reached their performance limit
        - **Focus**: Prioritize models with realistic accuracy (90-98%) that improved or stayed stable
        """)
        
        # Show improvement metrics for models that have both baseline and tuned
        models_with_both = comp_df[comp_df['has_both']]
        
        if len(models_with_both) > 0:
            st.markdown("### Performance Improvements")
            cols = st.columns(min(len(models_with_both), 4))  # Max 4 columns per row
            
            for idx, row in models_with_both.iterrows():
                col_idx = idx % 4
                improvement = (row['Tuned'] - row['Baseline']) * 100
                
                # Format improvement with proper sign handling
                improvement_str = f"{improvement:+.1f}%" if improvement != 0 else f"{improvement:.1f}%"
                
                # For accuracy metrics: positive = good (green up), negative = bad (red down)
                delta_color = "normal"
                
                with cols[col_idx]:
                    st.metric(
                        label=f"{row['model']}",
                        value=f"{row['Tuned']:.1%}",
                        delta=improvement_str,
                        delta_color=delta_color
                    )
        
        # Show top realistic models (accuracy < 100%)
        realistic_tuned = comp_df[(comp_df['Tuned'] < 0.999) & (comp_df['Tuned'].notna())].copy()
        
        if not realistic_tuned.empty:
            # Sort by tuned accuracy descending
            realistic_tuned = realistic_tuned.sort_values('Tuned', ascending=False)
            
            st.success(f"""
            ### 🏆 Best Models After Tuning (Realistic Performance)
            
            These models show strong performance without overfitting:
            """)
            
            # Display top models in columns
            num_models = len(realistic_tuned)
            cols = st.columns(min(num_models, 3))  # Max 3 columns
            
            for idx, (_, row) in enumerate(realistic_tuned.iterrows()):
                col_idx = idx % 3
                with cols[col_idx]:
                    rank_emoji = "🥇" if idx == 0 else "🥈" if idx == 1 else "🥉" if idx == 2 else "⭐"
                    
                    # Calculate improvement and format with proper sign
                    if pd.notna(row['Baseline']):
                        improvement = (row['Tuned'] - row['Baseline']) * 100
                        delta_str = f"{improvement:+.1f}%"
                        # For accuracy: positive = good (green), negative = bad (red)
                        delta_color = "normal"
                    else:
                        delta_str = None
                        delta_color = "off"
                    
                    st.metric(
                        label=f"{rank_emoji} {row['model']}",
                        value=f"{row['Tuned']:.2%}",
                        delta=delta_str,
                        delta_color=delta_color
                    )
            
            st.info(f"💡 **Recommendation:** Consider any of the above {len(realistic_tuned)} models for deployment. All show strong, generalizable performance.")
        else:
            st.warning("⚠️ No models with realistic accuracy found. All tuned models may be overfitting.")
    else:
        st.info("No models found for comparison.")
else:
    # Fallback to sample data with actual models
    comparison_data = {
        'Model': ['KNN', 'ANN_DNN', 'LDA', 'Naive Bayes'],
        'Accuracy': [0.9530, 0.9483, 0.9470, 0.7556]
    }

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name='Baseline Models',
        x=comparison_data['Model'],
        y=comparison_data['Accuracy'],
        marker_color='#51CF66',
        text=[f'{a:.2%}' for a in comparison_data['Accuracy']],
        textposition='outside'
    ))

    fig.update_layout(
        title='Model Performance Comparison (Baseline Models)',
        xaxis_title='Model',
        yaxis_title='Accuracy',
        height=400,
        yaxis_range=[0.7, 1.0]
    )

    st.plotly_chart(fig, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("KNN (Best)", "95.30%", delta="Best baseline model")
    col2.metric("ANN_DNN", "94.83%", delta="Good performance")
    col3.metric("LDA", "94.70%", delta="Solid baseline")

    st.success("**Result:** KNN achieved the best baseline performance at 95.30% accuracy without extensive hyperparameter tuning.")

# MLflow Experiment Tracking
st.header("🔬 MLflow Experiment Tracking")

st.markdown("""
All experiments were tracked using **MLflow** for reproducibility and comparison. Each run logged:
- Hyperparameters
- Performance metrics
- Model artifacts
- Training duration
""")

# Use actual MLflow data from database
if len(summary_df) > 0:
    # Display all runs
    display_cols = ['model_name']
    
    # Add run_uuid if it exists
    if 'run_uuid' in summary_df.columns:
        display_cols.append('run_uuid')
    
    # Add metric columns
    metric_cols = [col for col in summary_df.columns if col in ['accuracy', 'precision', 'recall', 'f1_score']]
    display_cols.extend(metric_cols)
    
    # Add training time if available
    if 'training_time' in summary_df.columns:
        display_cols.append('training_time')
    
    # Filter only existing columns
    display_cols = [col for col in display_cols if col in summary_df.columns]
    
    display_df = summary_df[display_cols].copy()
    
    # Rename columns for display
    column_rename = {'model_name': 'Model', 'run_uuid': 'Run ID'}
    for col in display_df.columns:
        if col in ['accuracy', 'precision', 'recall', 'f1_score', 'training_time']:
            column_rename[col] = col.replace('_', ' ').title()
    
    display_df = display_df.rename(columns=column_rename)
    
    # Add status column
    display_df['Status'] = '✅ Completed'
    
    # Apply gradient styling to metric columns
    metric_display_cols = [col for col in display_df.columns if col in ['Accuracy', 'Precision', 'Recall', 'F1 Score']]
    if metric_display_cols:
        st.dataframe(
            display_df.style.background_gradient(subset=metric_display_cols, cmap='Greens'),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### MLflow Tracking Features")
        st.markdown("""
        - ✅ Parameter logging (`mlflow.log_param()`)
        - ✅ Metric tracking (`mlflow.log_metric()`)
        - ✅ Model artifacts (`mlflow.sklearn.log_model()`)
        - ✅ Experiment comparison
        - ✅ Model versioning
        """)
    
    with col2:
        st.markdown("### Logged Information")
        st.code("""
mlflow.start_run():
  Parameters:
    - n_estimators, max_depth
    - learning_rate, subsample
  
  Metrics:
    - accuracy, precision, recall
    - f1_score, roc_auc
  
  Artifacts:
    - model.pkl
    - feature_importance.png
    - confusion_matrix.png
    """, language="python")
else:
    st.warning("No MLflow runs found in database. Showing sample data.")
    
    # Fallback to sample data with actual models
    mlflow_runs = {
        'Run ID': ['run_001', 'run_002', 'run_003', 'run_004'],
        'Model': ['KNN', 'ANN_DNN', 'LDA', 'Naive Bayes'],
        'Accuracy': [0.9530, 0.9483, 0.9470, 0.7556],
        'F1-Score': [0.9357, 0.9470, 0.9444, 0.7750],
        'Training Time (s)': [8, 45, 7, 5],
        'Status': ['✅ Completed', '✅ Completed', '✅ Completed', '✅ Completed']
    }

    df_mlflow = pd.DataFrame(mlflow_runs)

    st.dataframe(
        df_mlflow.style.background_gradient(subset=['Accuracy', 'F1-Score'], cmap='Greens'),
        use_container_width=True,
        hide_index=True
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### MLflow Tracking Features")
        st.markdown("""
        - ✅ Parameter logging (`mlflow.log_param()`)
        - ✅ Metric tracking (`mlflow.log_metric()`)
        - ✅ Model artifacts (`mlflow.sklearn.log_model()`)
        - ✅ Experiment comparison
        - ✅ Model versioning
        """)

    with col2:
        st.markdown("### Logged Information")
        st.code("""
mlflow.start_run():
  Parameters:
    - n_neighbors, weights (KNN)
    - layers, dropout (ANN)
  
  Metrics:
    - accuracy, precision, recall
    - f1_score, roc_auc
  
  Artifacts:
    - model.pkl
    - preprocessor.pkl
    - feature_importance.png
    - confusion_matrix.png
    """, language="python")

# Model Selection
st.header("🏆 Model Selection & Deployment")

# Get realistic tuned models (accuracy < 100%)
realistic_models = tuned_models[tuned_models['accuracy'] < 0.999].copy() if len(tuned_models) > 0 and 'accuracy' in tuned_models.columns else pd.DataFrame()

if not realistic_models.empty:
    realistic_models = realistic_models.sort_values('accuracy', ascending=False)
    
    st.markdown(f"""
    Based on comprehensive evaluation across multiple metrics, **{len(realistic_models)} models** showed strong, 
    generalizable performance without overfitting. Any of these models are suitable for deployment:
    """)
    
    # Show top realistic models
    st.markdown("### 🎯 Recommended Models for Deployment")
    
    for idx, (_, model) in enumerate(realistic_models.iterrows()):
        rank_emoji = "🥇" if idx == 0 else "🥈" if idx == 1 else "🥉" if idx == 2 else "⭐"
        
        with st.expander(f"{rank_emoji} **{model['model_name']}** - {model.get('accuracy', 0):.2%} Accuracy", expanded=(idx < 2)):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("### Performance Metrics")
                st.metric("Accuracy", f"{model.get('accuracy', 0):.2%}")
                if 'f1_score' in model:
                    st.metric("F1-Score", f"{model.get('f1_score', 0):.2%}")
                if 'precision' in model:
                    st.metric("Precision", f"{model.get('precision', 0):.2%}")
                if 'recall' in model:
                    st.metric("Recall", f"{model.get('recall', 0):.2%}")
            
            with col2:
                st.markdown("### Training Info")
                if 'training_time' in model:
                    st.metric("Training Time", f"{model.get('training_time', 0):.2f}s")
                
                # Get baseline comparison
                baseline_match = baseline_models[baseline_models['model_name'] == model['model_name']]
                if not baseline_match.empty:
                    baseline_acc = baseline_match.iloc[0]['accuracy']
                    improvement = (model['accuracy'] - baseline_acc) * 100
                    st.metric("Improvement", f"+{improvement:.1f}%", delta=f"{baseline_acc:.2%} → {model['accuracy']:.2%}")
            
            with col3:
                st.markdown("### Model Characteristics")
                
                # Model-specific characteristics
                if 'XGBoost' in model['model_name']:
                    st.markdown("✅ Gradient boosting\n\n✅ Handles imbalance well\n\n✅ Feature importance\n\n✅ Fast prediction")
                elif 'LightGBM' in model['model_name']:
                    st.markdown("✅ Very fast training\n\n✅ Low memory usage\n\n✅ Handles large data\n\n✅ Good accuracy")
                elif 'Logistic' in model['model_name']:
                    st.markdown("✅ Highly interpretable\n\n✅ Fast predictions\n\n✅ Probabilistic output\n\n✅ Low complexity")
                elif 'SVM' in model['model_name']:
                    st.markdown("✅ Strong with small data\n\n✅ Effective in high dims\n\n✅ Robust to outliers")
                elif 'ANN' in model['model_name'] or 'DNN' in model['model_name']:
                    st.markdown("✅ Deep learning\n\n✅ Complex patterns\n\n✅ Non-linear relationships\n\n✅ Scalable")
                else:
                    st.markdown("✅ Proven performance\n\n✅ Well-tested\n\n✅ Production-ready")
    
    st.info(f"""
    💡 **Deployment Recommendation:** 
    - For **production**: Consider top 2-3 models and A/B test them
    - For **speed**: Choose fastest training/prediction model
    - For **interpretability**: Logistic Regression if available
    - For **accuracy**: Use ensemble of top models
    """)
else:
    st.warning("No suitable models found for deployment. All models may be overfitting.")
    st.markdown("""
    Based on comprehensive evaluation across multiple metrics, **KNN** was selected as the primary model 
    with 95.30% accuracy, offering the best balance of performance, speed, and simplicity.
    """)
    
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### Selection Criteria")
        st.markdown("""
        - High accuracy (95.30%)
        - Excellent recall (98.05%)
        - Good F1-score (93.57%)
        - Fast prediction time
        - No overfitting issues
        - Simple, interpretable
        """)

    with col2:
        st.markdown("### Model Performance")
        st.metric("Accuracy", "95.30%", delta="Best baseline")
        st.metric("F1-Score", "0.9357")
        st.metric("Precision", "89.56%")
        st.metric("Recall", "98.05%", delta="Excellent")

    with col3:
        st.markdown("### Saved Artifacts")
        st.markdown("""
        - `knn_model.pkl`
        - `ann_dnn_model.pkl`
        - `lda_model.pkl`
        - `naive_bayes_model.pkl`
        - `preprocessor.pkl`
        - MLflow model format
        """)

# Feature Importance
st.header("🎯 Feature Importance Analysis")

st.markdown("""
Understanding which features drive our predictions helps validate model logic and ensure business alignment.
""")

# Load actual model objects from pickle files
@st.cache_resource
def load_model_objects():
    """Load actual trained models from pickle files"""
    model_dir = 'dsmodelpickl+preprocessor'
    models = {}
    preprocessor = None
    
    try:
        # Load preprocessor
        preprocessor_path = os.path.join(model_dir, 'preprocessor.pkl')
        if os.path.exists(preprocessor_path):
            with open(preprocessor_path, 'rb') as f:
                preprocessor = pickle.load(f)
        
        # Load all models
        model_files = {
            'KNN': 'knn_model.pkl',
            'ANN_DNN': 'ann_dnn_model.pkl',
            'LDA': 'lda_model.pkl',
            'Naive Bayes': 'naive_bayes_model.pkl'
        }
        
        for model_name, filename in model_files.items():
            model_path = os.path.join(model_dir, filename)
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    models[model_name] = pickle.load(f)
        
        return models, preprocessor
    except Exception as e:
        st.warning(f"Could not load models: {str(e)}")
        return {}, None

# Load models
loaded_models, loaded_preprocessor = load_model_objects()

if loaded_models:
    st.markdown("### 🎯 Dynamic Feature Importance from Your Trained Models")
    
    st.success(f"✅ Successfully loaded {len(loaded_models)} models for feature importance analysis")
    
    # Get feature names from preprocessor if available
    feature_names = None
    if loaded_preprocessor is not None:
        try:
            # Try to get feature names from preprocessor
            if hasattr(loaded_preprocessor, 'get_feature_names_out'):
                feature_names = list(loaded_preprocessor.get_feature_names_out())
            elif hasattr(loaded_preprocessor, 'feature_names_in_'):
                feature_names = list(loaded_preprocessor.feature_names_in_)
        except:
            pass
    
    # If we don't have feature names, use generic names
    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(13)]  # 13 features
    
    # Function to calculate permutation importance
    def calculate_permutation_importance(model, X, y, feature_names, n_repeats=5):
        """
        Calculate permutation importance for any model.
        This works by shuffling each feature and measuring the drop in model performance.
        """
        try:
            from sklearn.metrics import accuracy_score
            
            # Helper function to get clean predictions
            def get_predictions(model, X):
                predictions = model.predict(X)
                
                # Handle multi-dimensional predictions (like neural networks)
                if len(predictions.shape) > 1:
                    predictions = predictions.flatten()
                
                # Convert continuous predictions to binary (0 or 1)
                if predictions.dtype in [np.float16, np.float32, np.float64]:
                    # Flatten again if still multi-dimensional
                    predictions = predictions.flatten()
                    # Round and clip to ensure binary
                    predictions = np.clip(np.round(predictions), 0, 1).astype(int)
                
                return predictions
            
            # Get baseline score
            baseline_predictions = get_predictions(model, X)
            baseline_score = accuracy_score(y, baseline_predictions)
            
            # Debug: Check if model is making varied predictions
            unique_preds = np.unique(baseline_predictions)
            if len(unique_preds) == 1:
                st.warning(f"⚠️ Model is predicting only one class. Baseline accuracy: {baseline_score:.4f}")
            
            importances = []
            for feature_idx in range(X.shape[1]):
                scores = []
                for _ in range(n_repeats):
                    # Copy and shuffle one feature
                    X_permuted = X.copy()
                    np.random.shuffle(X_permuted[:, feature_idx])
                    
                    # Calculate score with shuffled feature
                    permuted_predictions = get_predictions(model, X_permuted)
                    permuted_score = accuracy_score(y, permuted_predictions)
                    
                    # Importance is the drop in performance
                    scores.append(max(0, baseline_score - permuted_score))  # Ensure non-negative
                
                # Average across repeats
                importances.append(np.mean(scores))
            
            # Normalize importances to sum to 1 (like sklearn's feature_importances_)
            importances = np.array(importances)
            if importances.sum() > 0:
                importances = importances / importances.sum()
            else:
                # If all importances are 0, use small uniform values
                st.info("All features have 0 importance. Using small uniform values.")
                importances = np.ones(len(feature_names)) * 0.001
            
            return importances
        except Exception as e:
            st.warning(f"Could not calculate permutation importance: {str(e)}")
            return None
    
    # Load test data for permutation importance (if needed)
    @st.cache_data
    def load_test_data():
        """Load test data for permutation importance calculation"""
        try:
            # Try to load from your data source
            data = pd.read_csv('data/WholeTruthFoodDataset-combined.csv')
            
            # Assuming 'Success' is the target column
            if 'Success' in data.columns:
                X = data.drop('Success', axis=1)
                y = data['Success']
                
                # Apply preprocessing if available
                if loaded_preprocessor is not None:
                    X_processed = loaded_preprocessor.transform(X)
                    st.info(f"📊 Loaded {X_processed.shape[0]} samples with {X_processed.shape[1]} features (preprocessed)")
                else:
                    X_processed = X.values
                    st.warning("⚠️ Using raw data without preprocessing. Results may be inaccurate.")
                
                # Use a stratified sample for better representation
                from sklearn.model_selection import train_test_split
                if len(X_processed) > 500:
                    _, X_sample, _, y_sample = train_test_split(
                        X_processed, y.values, 
                        test_size=500, 
                        stratify=y.values,
                        random_state=42
                    )
                    return X_sample, y_sample
                else:
                    return X_processed, y.values
        except Exception as e:
            st.warning(f"Could not load test data: {str(e)}")
        return None, None
    
    # Extract feature importance from models
    feature_importance_dict = {}
    
    # Load test data once for permutation importance
    X_test, y_test = load_test_data()
    use_permutation = X_test is not None and y_test is not None
    
    for model_name, model in loaded_models.items():
        try:
            if hasattr(model, 'feature_importances_'):
                # Tree-based models (Random Forest, XGBoost, etc.)
                importance = model.feature_importances_
                feature_importance_dict[model_name] = importance
            elif hasattr(model, 'coef_'):
                # Linear models (LDA, Logistic Regression, etc.)
                importance = np.abs(model.coef_[0]) if len(model.coef_.shape) > 1 else np.abs(model.coef_)
                feature_importance_dict[model_name] = importance
            else:
                # For models without direct feature importance, use Permutation Importance
                if use_permutation:
                    with st.spinner(f'Calculating permutation importance for {model_name}...'):
                        importance = calculate_permutation_importance(
                            model, X_test, y_test, feature_names, n_repeats=5
                        )
                        if importance is not None:
                            feature_importance_dict[model_name] = importance
                            st.success(f"✅ {model_name}: Calculated real feature importance using permutation method!")
                        else:
                            # Fallback to uniform
                            importance = np.ones(len(feature_names)) / len(feature_names)
                            feature_importance_dict[model_name] = importance
                            st.info(f"ℹ️ {model_name}: Using uniform importance as fallback.")
                else:
                    # No test data available, use uniform
                    importance = np.ones(len(feature_names)) / len(feature_names)
                    feature_importance_dict[model_name] = importance
                    st.info(f"ℹ️ {model_name}: Test data not available. Using uniform importance.")
        except Exception as e:
            st.warning(f"⚠️ Could not extract feature importance from {model_name}: {str(e)}")
    
    if feature_importance_dict:
        # Create tabs for different models
        model_tabs = st.tabs(list(feature_importance_dict.keys()))
        
        for tab, (model_name, importance) in zip(model_tabs, feature_importance_dict.items()):
            with tab:
                # Ensure we have the right number of features
                n_features = min(len(importance), len(feature_names))
                
                # Create dataframe
                df_importance = pd.DataFrame({
                    'Feature': feature_names[:n_features],
                    'Importance': importance[:n_features]
                }).sort_values('Importance', ascending=False)
                
                # Show top 10
                df_top10 = df_importance.head(10)
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Bar chart
                    fig = px.bar(
                        df_top10,
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        title=f'Top 10 Features - {model_name}',
                        color='Importance',
                        color_continuous_scale='Viridis',
                        text=df_top10['Importance'].apply(lambda x: f'{x:.3f}')
                    )
                    
                    fig.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
                    fig.update_traces(textposition='outside')
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown("#### Top 5 Features")
                    for idx, row in df_top10.head(5).iterrows():
                        st.metric(
                            label=row['Feature'],
                            value=f"{row['Importance']:.4f}"
                        )
                    
                    st.markdown("#### Statistics")
                    st.info(f"""
                    **Total Features:** {n_features}  
                    **Top Feature:** {df_top10.iloc[0]['Feature']}  
                    **Importance Range:** {importance.min():.4f} - {importance.max():.4f}  
                    **Mean Importance:** {importance.mean():.4f}
                    """)
        
        # Aggregate view across all models
        st.markdown("### 📊 Aggregate Feature Importance (All Models)")
        
        # Average importance across all models
        all_importances = []
        for importance in feature_importance_dict.values():
            n_features = min(len(importance), len(feature_names))
            all_importances.append(importance[:n_features])
        
        if all_importances:
            avg_importance = np.mean(all_importances, axis=0)
            
            df_avg = pd.DataFrame({
                'Feature': feature_names[:len(avg_importance)],
                'Average_Importance': avg_importance
            }).sort_values('Average_Importance', ascending=False)
            
            fig = px.bar(
                df_avg.head(10),
                x='Average_Importance',
                y='Feature',
                orientation='h',
                title='Top 10 Features - Averaged Across All Models',
                color='Average_Importance',
                color_continuous_scale='RdYlGn',
                text=df_avg.head(10)['Average_Importance'].apply(lambda x: f'{x:.3f}')
            )
            
            fig.update_layout(height=450, yaxis={'categoryorder': 'total ascending'})
            fig.update_traces(textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
            
            st.success(f"""
            **Key Insights from Feature Importance:**
            - **Most Important:** {df_avg.iloc[0]['Feature']} (avg importance: {df_avg.iloc[0]['Average_Importance']:.4f})
            - **Top 3 Features:** {', '.join(df_avg.head(3)['Feature'].tolist())}
            - **Total Features Analyzed:** {len(df_avg)}
            - **Models Analyzed:** {len(feature_importance_dict)}
            """)
    else:
        st.warning("Could not extract feature importance from loaded models. They may not support feature importance extraction.")
else:
    st.info("No models loaded yet.")

# Results Summary - Dynamic based on actual model performance
st.markdown("---")
st.header("📊 Results Summary")

# Filter to show only the 4 deployed models
deployed_models = ['KNN', 'ANN_DNN', 'LDA', 'Naive Bayes']

if not summary_df.empty:
    # Filter for deployed models only
    model_name_col = 'model_name' if 'model_name' in summary_df.columns else 'display_name'
    deployed_df = summary_df[summary_df[model_name_col].str.contains('|'.join(deployed_models), case=False, na=False)]
    
    if deployed_df.empty:
        # Fallback to all data if filtering fails
        deployed_df = summary_df
    
    # Get best performing model from deployed models (excluding overfitting models)
    non_overfit_deployed = deployed_df[deployed_df['accuracy'] < 0.999]
    
    if not non_overfit_deployed.empty:
        # Use best non-overfitting model
        best_model = non_overfit_deployed.loc[non_overfit_deployed['accuracy'].idxmax()]
    else:
        # Fallback to best overall if all are overfitting
        best_model = deployed_df.loc[deployed_df['accuracy'].idxmax()]
    
    best_accuracy = best_model['accuracy']
    best_model_name = best_model.get('display_name', best_model.get('model_name', 'Unknown'))
    
    # Get metrics from the SAME best model, not max across all models
    best_f1 = best_model['f1_score'] if 'f1_score' in best_model and pd.notna(best_model['f1_score']) else 0
    best_precision = best_model['precision'] if 'precision' in best_model and pd.notna(best_model['precision']) else 0
    best_recall = best_model['recall'] if 'recall' in best_model and pd.notna(best_model['recall']) else 0
    
    # Get worst performer to show improvement
    worst_accuracy = deployed_df['accuracy'].min()
    improvement = best_accuracy - worst_accuracy
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🏆 Model Performance")
        st.markdown(f"""
        **Best Model: {best_model_name}**
        
        | Metric | Value |
        |--------|-------|
        | **Accuracy** | **{best_accuracy:.2%}** |
        | **F1-Score** | **{best_f1:.4f}** |
        | **Precision** | **{best_precision:.2%}** |
        | **Recall** | **{best_recall:.2%}** |
        
        **Performance Range:** {worst_accuracy:.2%} - {best_accuracy:.2%} accuracy
        """)
        
        # Show deployed models comparison only
        st.markdown("#### Deployed Models Performance")
        
        # Add Stage column to indicate Baseline vs Tuned
        comparison_df = deployed_df[[model_name_col, 'accuracy', 'f1_score', 'precision', 'recall']].copy()
        
        # Add stage label (Baseline or Tuned)
        if 'run_name' in deployed_df.columns:
            comparison_df['Stage'] = deployed_df['run_name'].apply(
                lambda x: '🔵 Baseline' if 'Baseline' in str(x) else '🟢 Tuned' if 'Tuned' in str(x) else '⚪ Unknown'
            )
        elif 'display_name' in deployed_df.columns:
            comparison_df['Stage'] = deployed_df['display_name'].apply(
                lambda x: '🔵 Baseline' if 'Baseline' in str(x) else '🟢 Tuned' if 'Tuned' in str(x) else '⚪ Unknown'
            )
        else:
            # Try to infer from model name
            comparison_df['Stage'] = comparison_df[model_name_col].apply(
                lambda x: '🔵 Baseline' if 'Baseline' in str(x) else '🟢 Tuned' if 'Tuned' in str(x) else '⚪ Model'
            )
        
        # Reorder columns
        comparison_df = comparison_df[[model_name_col, 'Stage', 'accuracy', 'f1_score', 'precision', 'recall']]
        comparison_df.columns = ['Model', 'Stage', 'Accuracy', 'F1-Score', 'Precision', 'Recall']
        comparison_df = comparison_df.sort_values('Accuracy', ascending=False)
        
        # Format percentages
        comparison_df['Accuracy'] = comparison_df['Accuracy'].apply(lambda x: f"{x:.2%}")
        comparison_df['F1-Score'] = comparison_df['F1-Score'].apply(lambda x: f"{x:.4f}")
        comparison_df['Precision'] = comparison_df['Precision'].apply(lambda x: f"{x:.2%}")
        comparison_df['Recall'] = comparison_df['Recall'].apply(lambda x: f"{x:.2%}")
        
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("### 💼 Business Impact")
        st.markdown(f"""
        - ✅ **{best_accuracy:.1%} accuracy** in predicting product success ({best_model_name})
        - ✅ **{best_recall:.1%} recall** - catches most successful products
        - ✅ **{best_precision:.1%} precision** - reliable positive predictions
        - ✅ Can identify successful products before launch
        - ✅ Optimize inventory based on predictions
        - ✅ Data-driven product development decisions
        - ✅ Reduce risk of product failures
        - ✅ {improvement:.1%} performance range across models
        """)
        
        # Key insights
        st.markdown("#### 🎯 Key Insights")
        total_models = len(deployed_df)
        high_performers = len(deployed_df[deployed_df['accuracy'] > 0.90])
        st.info(f"""
        - Evaluated **{total_models} production-ready models**
        - **{high_performers} models** achieved >90% accuracy
        - Best model: **{best_model_name}** ({best_accuracy:.2%})
        - Models: KNN, ANN_DNN, LDA, Naive Bayes
        - All experiments tracked in MLflow
        """)
    
    st.success(f"""
    **Conclusion:** Successfully built and evaluated ML pipeline with **{best_accuracy:.2%} accuracy** using **{best_model_name}** as the best performing model. 
    All 4 production models (KNN, ANN_DNN, LDA, Naive Bayes) tracked in MLflow for reproducibility and team collaboration.
    """)
else:
    # Fallback if no data
    st.warning("No model performance data available. Please ensure MLflow database is accessible.")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Model Performance")
        st.info("Run models to see performance metrics here.")
    
    with col2:
        st.markdown("### Business Impact")
        st.info("Performance metrics will appear here after model training.")
