from sklearn.neighbors import KNeighborsClassifier
import joblib

def train_knn(X, y):
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X, y)
    joblib.dump(model, "knn_model.joblib")
    return model

def predict_knn(model, X):
    return model.predict(X)
