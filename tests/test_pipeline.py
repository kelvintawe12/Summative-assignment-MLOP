import pytest
from fastapi.testclient import TestClient
from api.main import app
import os

client = TestClient(app)

def test_api_health():
    """Verify that the API health endpoint is functional."""
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()

def test_model_structure():
    """Verify that the model building logic creates the correct output shape."""
    from src.model import build_model
    # Test for 4 classes
    model = build_model(num_classes=4)
    assert model.output_shape == (None, 4)

def test_registry_initialization():
    """Verify that the SQLite database initializes correctly."""
    assert os.path.exists("models/model_metadata.db")
