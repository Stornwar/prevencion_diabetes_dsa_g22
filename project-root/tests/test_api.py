from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {
        "name": "Prevención Diabetes API",
        "api_version": "0.1",
        "model_version": "0.0.1"
    }

def test_predict():
    payload = {
        "inputs": [{
            "HighBP": 1,
            "HighChol": 1,
            "CholCheck": 1,
            "BMI": 25.0,
            "Smoker": 0,
            "Stroke": 0,
            "HeartDiseaseorAttack": 0,
            "PhysActivity": 1,
            "Fruits": 1,
            "Veggies": 1,
            "HvyAlcoholConsump": 0,
            "AnyHealthcare": 1,
            "NoDocbcCost": 0,
            "GenHlth": 3,
            "MentHlth": 5,
            "PhysHlth": 5,
            "DiffWalk": 0,
            "Sex": 1,
            "Age": 2,
            "Education": 3,
            "Income": 2
        }]
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    assert "predictions" in response.json()
