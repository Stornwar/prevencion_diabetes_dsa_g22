from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from config.core import model_config

models = {
    "RandomForest": RandomForestClassifier(
        n_estimators=model_config.n_estimators,
        max_depth=model_config.max_depth,
        random_state=model_config.random_state
    ),
    "LogisticRegression": LogisticRegression(
        random_state=model_config.random_state
    ),
    "XGBoost": XGBClassifier(
        n_estimators=model_config.xgb_n_estimators,
        max_depth=model_config.xgb_max_depth,
        learning_rate=model_config.xgb_learning_rate,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=model_config.random_state
    )
}