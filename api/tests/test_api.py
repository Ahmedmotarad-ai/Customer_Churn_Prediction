"""API tests: health, valid/invalid predictions, robustness."""


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_shape(client, known_churner):
    r = client.post("/predict", json=known_churner)
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"churn", "churn_label", "churn_probability"}
    assert body["churn"] in (0, 1)
    assert body["churn_label"] == ("Yes" if body["churn"] else "No")
    assert 0.0 <= body["churn_probability"] <= 1.0


def test_predict_known_churner(client, known_churner):
    # first dataset row is a churner the model flags with high probability
    body = client.post("/predict", json=known_churner).json()
    assert body["churn"] == 1
    assert body["churn_probability"] > 0.5


def test_predict_low_risk_stays(client, low_risk):
    body = client.post("/predict", json=low_risk).json()
    assert body["churn"] == 0
    assert body["churn_probability"] < 0.5


def test_predict_negative_tenure_rejected(client, known_churner):
    bad = dict(known_churner, tenure=-5)
    assert client.post("/predict", json=bad).status_code == 422


def test_predict_missing_field_rejected(client, known_churner):
    bad = dict(known_churner)
    bad.pop("Contract")
    assert client.post("/predict", json=bad).status_code == 422


def test_predict_unknown_category_tolerated(client, known_churner):
    # encoder uses handle_unknown="ignore": unseen values must not crash
    weird = dict(known_churner, Contract="Weekly")
    r = client.post("/predict", json=weird)
    assert r.status_code == 200
    assert 0.0 <= r.json()["churn_probability"] <= 1.0
