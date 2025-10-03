import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_ride_completion_distribution(df):
    completion_stats = df['Booking Status'].value_counts()
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#4CAF50' if 'Completed' in str(s) else '#F44336' for s in completion_stats.index]
    bars = ax.bar(completion_stats.index, completion_stats.values, color=colors, alpha=0.8)
    ax.set_title('Ride Completion Status Distribution', fontsize=14, fontweight='bold')
    ax.set_ylabel('Number of Rides')

    total = completion_stats.sum()
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + total*0.01,
                f'{height/total*100:.1f}%', ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.show()

def plot_revenue_impact(df):
    revenue_data = df.groupby('Booking Status')['Booking Value'].sum()
    completed_revenue = revenue_data.get('Completed', 0)
    lost_revenue = revenue_data.sum() - completed_revenue

    fig, ax = plt.subplots(figsize=(6, 6))
    values = [completed_revenue, lost_revenue]
    labels = ['Completed Rides', 'Cancelled Rides']
    colors = ['#4CAF50', '#F44336']

    §, texts, autotexts = ax.pie(values, labels=labels, autopct='%1.1f%%',
                                     colors=colors, startangle=90)
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    ax.set_title('Completed - Cancelled Rides', fontsize=14, fontweight='bold')
    plt.show()

def plot_feature_importance(feature_importance: dict):
    features = list(feature_importance.keys())
    importances = list(feature_importance.values())
    fig, ax = plt.subplots(figsize=(8, 6))
    y_pos = np.arange(len(features))
    bars = ax.barh(y_pos, importances, color='steelblue')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.set_xlabel('Importance')
    ax.set_title('Key Features for Prediction')
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2.,
                f'{width:.2f}', ha='left', va='center', fontsize=9)
    plt.tight_layout()
    plt.show()

