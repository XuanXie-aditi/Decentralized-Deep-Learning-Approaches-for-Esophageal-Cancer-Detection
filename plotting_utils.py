import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import numpy as np
from pathlib import Path
import pandas as pd

def create_output_directories():
    """Create directories for storing outputs."""
    base_dir = Path('training_outputs')
    
    # Create directories for different output types
    dirs = {
        'plots': base_dir / 'plots',
        'metrics': base_dir / 'metrics',
        'models': base_dir / 'models',
        'dataset_stats': base_dir / 'dataset_stats'
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return dirs

def plot_confusion_matrix(y_true, y_pred, model_name, output_dir):
    """Plot and save enhanced confusion matrix."""
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, (y_pred > 0.5).astype(int))
    
    # Calculate percentages for annotations
    cm_percentages = cm / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Create annotations with both count and percentage
    annotations = np.array([[f'{count}\n({percentage:.1f}%)' 
                           for count, percentage in zip(row, row_percentages)] 
                           for row, row_percentages in zip(cm, cm_percentages)])
    
    # Plot heatmap with custom colormap
    sns.heatmap(cm, annot=annotations, fmt='', cmap='Blues', 
                xticklabels=['No Esophagus', 'Esophagus'],
                yticklabels=['No Esophagus', 'Esophagus'])
    
    plt.title(f'{model_name}\nConfusion Matrix', pad=20)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Add metrics text box
    accuracy = np.sum(np.diag(cm)) / np.sum(cm)
    sensitivity = cm[1,1] / (cm[1,1] + cm[1,0])
    specificity = cm[0,0] / (cm[0,0] + cm[0,1])
    
    metrics_text = f'Accuracy: {accuracy:.3f}\nSensitivity: {sensitivity:.3f}\nSpecificity: {specificity:.3f}'
    plt.text(1.5, 0.5, metrics_text, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{model_name}_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_roc_curve(y_true, y_pred, model_name, output_dir):
    """Plot and save ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='red', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='blue', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(False)
    
    plt.savefig(output_dir / f'{model_name}_roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_training_history(history, model_name, output_dir):
    """Plot and save enhanced training history."""
    plt.style.use('seaborn')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy plot
    ax1.plot(history['classification_accuracy'], label='Training Accuracy')
    ax1.plot(history['val_classification_accuracy'], label='Validation Accuracy')
    ax1.set_title(f'{model_name} Classification Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend(loc="lower right")
    ax1.grid(False)
    
    # Add final accuracy values
    final_train_acc = history['classification_accuracy'][-1]
    final_val_acc = history['val_classification_accuracy'][-1]
    ax1.text(0.02, 0.98, f'Final Train Acc: {final_train_acc:.3f}\nFinal Val Acc: {final_val_acc:.3f}',
             transform=ax1.transAxes, bbox=dict(facecolor='white', alpha=0.8), verticalalignment='top')
    
    # Loss plot
    ax2.plot(history['loss'], label='Training Loss')
    ax2.plot(history['val_loss'], label='Validation Loss')
    ax2.set_title(f'{model_name} Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend(loc="upper right")
    ax2.grid(False)
    
    # Add final loss values
    final_train_loss = history['loss'][-1]
    final_val_loss = history['val_loss'][-1]
    ax2.text(0.02, 0.98, f'Final Train Loss: {final_train_loss:.3f}\nFinal Val Loss: {final_val_loss:.3f}',
             transform=ax2.transAxes, bbox=dict(facecolor='white', alpha=0.8), verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{model_name}_training_history.png', dpi=300, bbox_inches='tight')
    plt.close()

def save_classification_metrics(y_true, y_pred, model_name, output_dir):
    """Save detailed classification metrics with additional statistics."""
    # Generate classification report
    report = classification_report(y_true, (y_pred > 0.5).astype(int))
    
    # Calculate additional metrics
    cm = confusion_matrix(y_true, (y_pred > 0.5).astype(int))
    accuracy = np.sum(np.diag(cm)) / np.sum(cm)
    sensitivity = cm[1,1] / (cm[1,1] + cm[1,0])
    specificity = cm[0,0] / (cm[0,0] + cm[0,1])
    precision = cm[1,1] / (cm[1,1] + cm[0,1])
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity)
    
    # Save detailed metrics
    with open(output_dir / f'{model_name}_classification_report.txt', 'w') as f:
        f.write(f"Model: {model_name}\n")
        f.write("=" * 50 + "\n\n")
        f.write("Classification Report:\n")
        f.write("-" * 50 + "\n")
        f.write(report)
        f.write("\n\nAdditional Metrics:\n")
        f.write("-" * 50 + "\n")
        f.write(f"Accuracy: {accuracy:.3f}\n")
        f.write(f"Sensitivity (Recall): {sensitivity:.3f}\n")
        f.write(f"Specificity: {specificity:.3f}\n")
        f.write(f"Precision: {precision:.3f}\n")
        f.write(f"F1 Score: {f1_score:.3f}\n")