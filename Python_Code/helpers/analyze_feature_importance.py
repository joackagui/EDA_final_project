# Analyze and print feature importances from trained models

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_feature_importance(model, X_train, target_name, top_n=10):
    if hasattr(model.named_steps['model'], 'feature_importances_'):
        importances = model.named_steps['model'].feature_importances_
        
        from helpers.model_pipeline import get_feature_names
        feature_names = get_feature_names(model, X_train)
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop {top_n} Most Important Features for {target_name}:")
        print(importance_df.head(top_n))
        
        return importance_df

