import matplotlib.pyplot as plt
import seaborn as sns

def plot_peak_hours(df):
    if 'hour' not in df:
        df['hour'] = df['pickup_datetime'].astype('datetime64[ns]').dt.hour
    
    hourly_data = df.groupby('hour').agg({
        'Booking Value': 'count',
        'Booking Status': lambda x: (x == 'Completed').mean() * 100
    }).rename(columns={'Booking Value': 'ride_count', 'Booking Status': 'completion_rate'})

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    ax1.plot(hourly_data.index, hourly_data['completion_rate'],
             color='darkred', marker='o', linewidth=2, label='Completion Rate')
    ax2.bar(hourly_data.index, hourly_data['ride_count'],
            color='skyblue', alpha=0.4, label='Ride Count')

    ax1.set_ylabel('Completion Rate (%)', color='darkred')
    ax2.set_ylabel('Number of Rides', color='steelblue')
    ax1.set_xlabel('Hour of Day')
    ax1.set_title('Peak Hours: Demand vs Completion Rates', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_cancellation_reasons(df):
    cancellation_reasons = df[df['Booking Status'] != 'Completed']['Booking Status'].value_counts()
    fig, ax = plt.subplots(figsize=(6, 6))
    if len(cancellation_reasons) > 0:
        wedges, texts, autotexts = ax.pie(
            cancellation_reasons.values,
            labels=cancellation_reasons.index,
            autopct='%1.1f%%',
            startangle=90,
            colors=sns.color_palette("pastel", len(cancellation_reasons))
        )
        ax.set_title('Cancellation Reasons Breakdown', fontsize=14, fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'No Cancellation Data', ha='center', va='center', transform=ax.transAxes)
    plt.show()

def plot_model_performance():
    models = ['Logistic Regression', 'Random Forest']
    auc_scores = [0.9507, 0.9561]
    acc_scores = [0.9392, 0.9403]

    fig, ax = plt.subplots(figsize=(8, 6))
    x = range(len(models))
    ax.bar([i-0.2 for i in x], auc_scores, width=0.4, label='AUC', color='seagreen')
    ax.bar([i+0.2 for i in x], acc_scores, width=0.4, label='Accuracy', color='dodgerblue')

    ax.set_ylim(0.9, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_title('Model Performance Metrics', fontsize=14, fontweight='bold')
    ax.legend()
    plt.show()
