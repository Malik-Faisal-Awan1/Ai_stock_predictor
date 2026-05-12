from sklearn.neural_network import MLPClassifier
import joblib

def train_mlp(X, y):
    model = MLPClassifier(hidden_layer_sizes=(40,20), max_iter=500)
    model.fit(X, y)
    joblib.dump(model, "mlp_model.joblib")
    return model

def predict_mlp(model, X):
    return model.predict(X)
