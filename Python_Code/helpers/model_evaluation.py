# Comprehensive evaluation for classification models

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

def evaluate_classification_model(model, X_train, X_test, y_train, y_test, model_name=""):   
    print(f"\nEVALUATING: {model_name}")
    
    results = {}
    
    # Cross-validation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores_auc = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
    cv_scores_acc = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    
    print("\nCross-Validation Results:")
    print(f"- AUC-ROC: {cv_scores_auc.mean():.4f} (+/- {cv_scores_auc.std()*2:.4f})")
    print(f"- Accuracy: {cv_scores_acc.mean():.4f} (+/- {cv_scores_acc.std()*2:.4f})")
    
    # Train final model
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Test metrics
    test_accuracy = model.score(X_test, y_test)
    test_auc = roc_auc_score(y_test, y_pred_proba)
    
    print("\nTest Set Results:")
    print(f"- Accuracy: {test_accuracy:.4f}")
    print(f"- AUC-ROC: {test_auc:.4f}")
    
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))
    
    # Store results
    results = {
        'cv_auc_mean': cv_scores_auc.mean(),
        'cv_auc_std': cv_scores_auc.std(),
        'cv_acc_mean': cv_scores_acc.mean(),
        'test_accuracy': test_accuracy,
        'test_auc': test_auc,
        'model': model
    }
    
    plot_confusion_matrix(y_test, y_pred, model_name, test_auc)
    
    return results

def plot_confusion_matrix(y_true, y_pred, model_name, auc_score):
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=['No', 'Yes'], 
               yticklabels=['No', 'Yes'])
    plt.title(f'{model_name}\nAUC: {auc_score:.3f}', fontweight='bold')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.show()

# Compare multiple models
def compare_models(results_dict):
    print("\nMODEL COMPARISON")
    
    comparison_data = {}
    for model_name, results in results_dict.items():
        if 'error' not in results:
            comparison_data[model_name] = {
                'CV AUC': results['cv_auc_mean'],
                'CV AUC Std': results['cv_auc_std'],
                'Test Accuracy': results['test_accuracy'],
                'Test AUC': results['test_auc']
            }
    
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data).T
        print(comparison_df.round(4))
        return comparison_df
    else:
        print("No valid results to compare")
        return None