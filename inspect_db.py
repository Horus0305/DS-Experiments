import sqlite3
import pandas as pd

conn = sqlite3.connect('mlruns.db')

print("=" * 80)
print("TUNED MODELS - ACCURACY COMPARISON")
print("=" * 80)

# Get tuned models with their accuracies
query = """
SELECT 
    r.name,
    m.value as accuracy,
    r.status
FROM runs r
JOIN metrics m ON r.run_uuid = m.run_uuid
WHERE r.name LIKE '%Tuned'
AND m.key = 'accuracy'
ORDER BY m.value DESC
"""

df_tuned = pd.read_sql_query(query, conn)
print("\nTUNED MODELS ACCURACY:")
print(df_tuned)
print()

# Compare with baseline
print("=" * 80)
print("BASELINE VS TUNED COMPARISON")
print("=" * 80)

query_comparison = """
SELECT 
    REPLACE(REPLACE(r.name, '_Baseline', ''), '_Tuned', '') as model,
    CASE 
        WHEN r.name LIKE '%Baseline' THEN 'Baseline'
        WHEN r.name LIKE '%Tuned' THEN 'Tuned'
    END as stage,
    m.value as accuracy
FROM runs r
JOIN metrics m ON r.run_uuid = m.run_uuid
WHERE m.key = 'accuracy'
AND (r.name LIKE '%Baseline' OR r.name LIKE '%Tuned')
ORDER BY model, stage
"""

df_comparison = pd.read_sql_query(query_comparison, conn)

# Pivot for easier comparison
pivot = df_comparison.pivot(index='model', columns='stage', values='accuracy')
pivot['difference'] = pivot['Tuned'] - pivot['Baseline']
pivot = pivot.round(4)
print("\n", pivot)
print()

# Get the 6 selected models (KNN, ANN_DNN, Logistic Regression, SVM, Naive Bayes, LDA)
print("=" * 80)
print("SELECTED 6 MODELS FOR DEPLOYMENT")
print("=" * 80)

selected_models = ['KNN', 'ANN_DNN', 'Logistic Regression', 'SVM', 'Naive Bayes', 'LDA']

for model in selected_models:
    query = f"""
    SELECT 
        r.name,
        MAX(m.value) as accuracy
    FROM runs r
    JOIN metrics m ON r.run_uuid = m.run_uuid
    WHERE (r.name LIKE '%{model}%')
    AND m.key = 'accuracy'
    GROUP BY r.name
    ORDER BY m.value DESC
    """
    df_model = pd.read_sql_query(query, conn)
    if not df_model.empty:
        best_run = df_model.iloc[0]
        print(f"{model:25s}: {best_run['accuracy']:.4f} ({best_run['name']})")

conn.close()
