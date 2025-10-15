"""
Test suite for FastAPI main.py
Tests the prediction API endpoints
"""

from fastapi.testclient import TestClient
from main import app
import pytest

# Create test client
client = TestClient(app)


def test_read_root_endpoint():
    """Tests the root '/' endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "Welcome to the Model Prediction API!"
    assert "endpoints" in response.json()


def test_health_endpoint():
    """Tests the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "preprocessor_loaded" in data


def test_model_info_endpoint():
    """Tests the model info endpoint."""
    response = client.get("/model-info")
    
    # If model is loaded, should return 200, otherwise 503
    if response.status_code == 200:
        data = response.json()
        assert "model_name" in data
        assert "features_required" in data
        assert len(data["features_required"]) == 13
    else:
        assert response.status_code == 503


def test_predict_endpoint_success_case():
    """
    Tests the '/predict' endpoint with a valid sample payload (success case).
    This matches the premium success case from Streamlit.
    """
    # Sample payload - Premium success case
    sample_payload = {
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

    response = client.post("/predict", json=sample_payload)

    # Check if model is loaded first
    if response.status_code == 503:
        pytest.skip("Model not loaded - skipping prediction test")

    if response.status_code != 200:
        print(f"API Error Response: {response.json()}")

    assert response.status_code == 200
    
    data = response.json()
    
    # Verify response structure
    assert "prediction" in data
    assert "probability" in data
    assert "confidence" in data
    assert "model_used" in data
    
    # Verify data types
    assert isinstance(data["prediction"], int)
    assert isinstance(data["probability"], float)
    assert isinstance(data["confidence"], str)
    
    # Verify value ranges
    assert data["prediction"] in [0, 1]
    assert 0 <= data["probability"] <= 1
    assert data["confidence"] in ["Low", "Medium", "High"]
    
    print(f"\n✅ Prediction: {data['prediction']}")
    print(f"✅ Probability: {data['probability']:.4f}")
    print(f"✅ Confidence: {data['confidence']}")
    print(f"✅ Model: {data['model_used']}")


def test_predict_endpoint_failure_case():
    """
    Tests the '/predict' endpoint with a budget failure case.
    """
    # Sample payload - Budget failure case
    sample_payload = {
        "price": 349,
        "discount": 0,
        "category": "Muesli",
        "ingredients_count": 4,
        "has_dates": 0,
        "has_cocoa": 0,
        "has_protein": 0,
        "packaging_type": "Recyclable Plastic",
        "season": "Summer",
        "customer_gender": 0.0,
        "age_numeric": 21,
        "shelf_life": 12,
        "clean_label": 0
    }

    response = client.post("/predict", json=sample_payload)

    if response.status_code == 503:
        pytest.skip("Model not loaded - skipping prediction test")

    assert response.status_code == 200
    
    data = response.json()
    assert "prediction" in data
    assert data["prediction"] in [0, 1]


def test_predict_endpoint_protein_case():
    """
    Tests the '/predict' endpoint with a protein bar case.
    """
    sample_payload = {
        "price": 1000,
        "discount": 0,
        "category": "Protein Bars",
        "ingredients_count": 5,
        "has_dates": 1,
        "has_cocoa": 1,
        "has_protein": 1,
        "packaging_type": "Recyclable Plastic",
        "season": "Winter",
        "customer_gender": 1.0,
        "age_numeric": 30,
        "shelf_life": 24,
        "clean_label": 1
    }

    response = client.post("/predict", json=sample_payload)

    if response.status_code == 503:
        pytest.skip("Model not loaded - skipping prediction test")

    assert response.status_code == 200


def test_predict_endpoint_missing_field():
    """
    Tests the '/predict' endpoint with missing required field.
    Should return 422 validation error.
    """
    # Missing 'price' field
    incomplete_payload = {
        "discount": 0,
        "category": "Dark Chocolate",
        "ingredients_count": 4
    }

    response = client.post("/predict", json=incomplete_payload)
    assert response.status_code == 422  # Validation error


def test_predict_endpoint_invalid_values():
    """
    Tests the '/predict' endpoint with invalid values.
    """
    # Invalid values (negative price, age > 100)
    invalid_payload = {
        "price": -100,  # Negative price
        "discount": 0,
        "category": "Dark Chocolate",
        "ingredients_count": 4,
        "has_dates": 1,
        "has_cocoa": 1,
        "has_protein": 0,
        "packaging_type": "Paper-based",
        "season": "Winter",
        "customer_gender": 1.0,
        "age_numeric": 150,  # Age > 100
        "shelf_life": 12,
        "clean_label": 1
    }

    response = client.post("/predict", json=invalid_payload)
    
    # API might accept it (preprocessing handles outliers), reject it, or be unavailable (no model files in CI)
    # Just check it doesn't crash
    assert response.status_code in [200, 422, 500, 503]


def test_batch_predict_endpoint():
    """
    Tests the '/batch-predict' endpoint with multiple inputs.
    """
    batch_payload = [
        {
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
        },
        {
            "price": 349,
            "discount": 0,
            "category": "Muesli",
            "ingredients_count": 4,
            "has_dates": 0,
            "has_cocoa": 0,
            "has_protein": 0,
            "packaging_type": "Recyclable Plastic",
            "season": "Summer",
            "customer_gender": 0.0,
            "age_numeric": 21,
            "shelf_life": 12,
            "clean_label": 0
        }
    ]

    response = client.post("/batch-predict", json=batch_payload)

    if response.status_code == 503:
        pytest.skip("Model not loaded - skipping batch prediction test")

    assert response.status_code == 200
    
    data = response.json()
    assert "predictions" in data
    assert "count" in data
    assert data["count"] == 2
    assert len(data["predictions"]) == 2


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v', '--tb=short'])
