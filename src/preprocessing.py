from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

def build_pipeline(categorical_cols, numeric_cols, model_params):
    """Create preprocessing + model pipeline"""
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("numeric", StandardScaler(), numeric_cols),
        ]
    )
    # Currently only supports SVC, but extendable
    if model_params["type"] == "SVC":
        classifier = SVC(C=model_params["C"], kernel=model_params["kernel"])
    else:
        raise ValueError(f"Unsupported model type: {model_params['type']}")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])
    return pipeline
