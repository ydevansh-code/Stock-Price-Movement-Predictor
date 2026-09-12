import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from src.config import RANDOM_SEED


def select_top_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    top_k: int = 25,
) -> list[str]:
    selector = RandomForestClassifier(
        n_estimators=200,
        max_depth=3,
        min_samples_leaf=15,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    selector.fit(X_train, y_train)
    importances = pd.Series(selector.feature_importances_, index=X_train.columns)
    return list(importances.nlargest(top_k).index)
