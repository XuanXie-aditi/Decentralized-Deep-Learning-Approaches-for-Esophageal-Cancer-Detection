from data_loader import load_and_preprocess_data
from trainer import train_and_evaluate_models
from federated_learning import federated_learning_simulation
from plotting_utils import (
    create_output_directories, plot_confusion_matrix, 
    save_classification_metrics, plot_roc_curve
)
import os
import json
from pathlib import Path
import numpy as np

def save_experiment_config(output_dir):
    """Save experiment configuration for reproducibility."""
    config = {
        'image_size': (224, 224),
        'batch_size': 32,
        'epochs': 10,
        'models_tested': ['ResNet50', 'VGG16', 'DenseNet121'],
        'optimizer': 'Adam',
        'learning_rate': 0.0001,
        'loss_functions': {
            'classification': 'binary_crossentropy',
            'segmentation': 'binary_crossentropy'
        },
        'loss_weights': {
            'classification': 1.0,
            'segmentation': 0.5
        }
    }
    
    with open(output_dir / 'experiment_config.json', 'w') as f:
        json.dump(config, f, indent=4)

def main():
    # Create output directories
    output_dirs = create_output_directories()
    
    # Save experiment configuration
    save_experiment_config(output_dirs['metrics'])
    
    # Load and analyze data
    print("\nLoading and preprocessing data...")
    X, y_class, y_seg, stats = load_and_preprocess_data('Endoscopy-esophagus')
    
    # Save detailed dataset statistics
    dataset_stats_path = output_dirs['dataset_stats'] / 'dataset_statistics.txt'
    with open(dataset_stats_path, 'w') as f:
        f.write("Dataset Statistics and Analysis\n")
        f.write("=" * 50 + "\n\n")
        
        # Basic statistics
        f.write("1. Basic Statistics\n")
        f.write("-" * 50 + "\n")
        f.write(f"Total Images: {len(X)}\n")
        f.write(f"Image Shape: {X[0].shape}\n")
        f.write(f"Data Type: {X.dtype}\n")
        f.write(f"Value Range: [{X.min():.3f}, {X.max():.3f}]\n\n")
        
        # Class distribution
        f.write("2. Class Distribution\n")
        f.write("-" * 50 + "\n")
        f.write(f"Esophagus Images: {stats['total_esophagus_images']}\n")
        f.write(f"No-Esophagus Images: {stats['total_no_esophagus_images']}\n")
        #f.write(f"Mask Images Available: {stats['total_mask_images']}\n\n")
        f.write(f"\n")
        
        # Data balance
        total_samples = len(X)
        class_ratio = stats['total_esophagus_images'] / total_samples
        f.write("3. Class Balance\n")
        f.write("-" * 50 + "\n")
        f.write(f"Class Ratio (Esophagus:Total): {class_ratio:.3f}\n")
        f.write(f"Class Balance Score: {min(class_ratio, 1-class_ratio)/max(class_ratio, 1-class_ratio):.3f}\n\n")
    
    print(f"Dataset statistics saved to {dataset_stats_path}")
    print("Data preprocessing completed!")
    
    # Train and evaluate models
    print("\nTraining and evaluating models...")
    results, output_dirs = train_and_evaluate_models(X, y_class, y_seg)
    
    # Find and report best model
    best_model_name = max(results.keys(),
                         key=lambda k: results[k]['evaluation'][1])
    
    print(f"\nBest performing model: {best_model_name}")
    print(f"Best model metrics saved in: {output_dirs['metrics']}")
    print(f"Visualization plots saved in: {output_dirs['plots']}")
    
    # Perform federated learning
    print("\nPerforming federated learning simulation...")
    final_model = federated_learning_simulation(
        best_model=results[best_model_name]['model'],
        best_model_name=best_model_name,
        X=X,
        y_class=y_class,
        y_seg=y_seg
    )
    
    # Save final model
    final_model_path = output_dirs['models'] / 'final_federated_model.h5'
    final_model.save(final_model_path)
    print(f"\nFinal model saved in : {final_model_path}")
    
    # Generate final evaluation metrics and plots
    final_predictions = final_model.predict(X)
    
    # Plot confusion matrix
    plot_confusion_matrix(
        y_class,
        final_predictions[0],
        'Final_Federated_Model',
        output_dirs['plots']
    )
    
    # Plot ROC curve
    plot_roc_curve(
        y_class,
        final_predictions[0],
        'Final_Federated_Model',
        output_dirs['plots']
    )
    
    # Save detailed metrics
    save_classification_metrics(
        y_class,
        final_predictions[0],
        'Final_Federated_Model',
        output_dirs['metrics']
    )
    
    print("\nTraining pipeline completed successfully!")
    print(f"All results saved in: {output_dirs['plots'].parent}")

if __name__ == "__main__":
    main()