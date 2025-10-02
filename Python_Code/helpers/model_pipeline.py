# Model pipeline utilities

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

def create_modeling_pipeline(X, model, target_type='classification'):
    # Identify feature types
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    
    print(f"Pipeline configured:")
    print(f"- Numeric features: {len(numeric_features)}")
    print(f"- Categorical features: {len(categorical_features)}")
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    return pipeline

# Get feature names after preprocessing
def get_feature_names(pipeline, X_original):
    preprocessor = pipeline.named_steps['preprocessor']
    
    numeric_features = X_original.select_dtypes(include=[np.number]).columns.tolist()
    feature_names = numeric_features.copy()
    
    categorical_features = X_original.select_dtypes(include=['object']).columns.tolist()

    if categorical_features:
        ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
        cat_feature_names = ohe.get_feature_names_out(categorical_features)
        feature_names.extend(cat_feature_names)
    
    return feature_names