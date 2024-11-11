from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

models = {
    "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
    "LogisticRegression": LogisticRegression(random_state=42),
    "SVC": SVC(probability=True, random_state=42)
}
