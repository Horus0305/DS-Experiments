"""
Test suite for model loading and prediction functionality
"""
import pytest
import os
import sys
import pickle
import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestModelPrediction:
    """Test model prediction with sample input"""
    
    @pytest.fixture
    def sample_input(self):
        """Create sample input data matching the expected features"""
        return pd.DataFrame({
            'price': [1047],
            'discount': [0],
            'category': ['Dark Chocolate'],
            'ingredients_count': [4],
            'has_dates': [1],
            'has_cocoa': [1],
            'has_protein': [0],
            'packaging_type': ['Paper-based'],
            'season': ['Winter'],
            'customer_gender': ['Female'],
            'age_numeric': [55],
            'shelf_life': [12],
            'clean_label': [1]
        })
    
    def test_model_prediction_with_sample(self, sample_input):
        """Test KNN model prediction with sample input"""
        model_dir = 'dsmodelpickl+preprocessor'
        
        # Check if model directory exists
        if not os.path.exists(model_dir):
            pytest.skip("Model directory not found (may need DVC pull)")
        
        preprocessor_path = os.path.join(model_dir, 'preprocessor.pkl')
        knn_path = os.path.join(model_dir, 'knn_model.pkl')
        
        # Check if required files exist
        if not os.path.exists(preprocessor_path):
            pytest.skip("Preprocessor not found")
        if not os.path.exists(knn_path):
            pytest.skip("KNN model not found")
        
        try:
            # Load preprocessor and model
            with open(preprocessor_path, 'rb') as f:
                preprocessor = pickle.load(f)
            with open(knn_path, 'rb') as f:
                model = pickle.load(f)
            
            # Transform input
            X_transformed = preprocessor.transform(sample_input)
            
            # Make prediction
            prediction = model.predict(X_transformed)
            proba = model.predict_proba(X_transformed)
            
            # Assertions
            assert prediction is not None, "Prediction is None"
            assert len(prediction) == 1, "Expected 1 prediction"
            assert prediction[0] in [0, 1], "Prediction should be 0 or 1"
            
            assert proba is not None, "Probability is None"
            assert proba.shape == (1, 2), "Probability shape should be (1, 2)"
            assert 0 <= proba[0][0] <= 1, "Probability not in [0, 1] range"
            assert 0 <= proba[0][1] <= 1, "Probability not in [0, 1] range"
            assert abs(proba[0][0] + proba[0][1] - 1.0) < 0.001, "Probabilities should sum to 1"
            
            print(f"\n✅ Model prediction successful!")
            print(f"   Prediction: {prediction[0]}")
            print(f"   Probability: {proba[0][1]:.4f}")
            
        except Exception as e:
            pytest.fail(f"Model prediction failed: {e}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

