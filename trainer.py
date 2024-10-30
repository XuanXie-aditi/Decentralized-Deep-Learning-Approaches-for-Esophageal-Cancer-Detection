from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from model_builder import create_dual_output_model
from plotting_utils import (
    create_output_directories,
    plot_confusion_matrix,
    save_classification_metrics,
    plot_training_history
)

def train_and_evaluate_models(X, y_class, y_seg):
    """Train and evaluate different models with enhanced metrics."""
    # Create output directories
    output_dirs = create_output_directories()
    
    # Split data
    X_train, X_test, y_class_train, y_class_test, y_seg_train, y_seg_test = train_test_split(
        X, y_class, y_seg, test_size=0.2, random_state=42
    )
    
    model_names = ['ResNet50', 'VGG16', 'DenseNet121']
    results = {}
    
    for model_name in model_names:
        print(f"\nTraining {model_name}")
        model = create_dual_output_model(model_name)
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss={
                'classification': 'binary_crossentropy',
                'segmentation': 'binary_crossentropy'
            },
            loss_weights={
                'classification': 1.0,
                'segmentation': 0.5
            },
            metrics={
                'classification': ['accuracy'],
                'segmentation': ['accuracy']
            }
        )
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ModelCheckpoint(
                output_dirs['models'] / f'best_{model_name}.keras',
                save_best_only=True
            )
        ]
        
        # Train model
        history = model.fit(
            X_train,
            {'classification': y_class_train, 'segmentation': y_seg_train},
            validation_split=0.2,
            epochs=10,
            batch_size=32,
            callbacks=callbacks
        )
        
        # Get predictions
        predictions = model.predict(X_test)
        y_pred_class = predictions[0]  # Classification predictions
        
        # Plot and save confusion matrix
        plot_confusion_matrix(
            y_class_test,
            y_pred_class,
            model_name,
            output_dirs['plots']
        )
        
        # Save classification metrics
        save_classification_metrics(
            y_class_test,
            y_pred_class,
            model_name,
            output_dirs['metrics']
        )
        
        # Plot and save training history
        plot_training_history(
            history.history,
            model_name,
            output_dirs['plots']
        )
        
        # Evaluate model
        evaluation = model.evaluate(
            X_test,
            {'classification': y_class_test, 'segmentation': y_seg_test}
        )
        
        results[model_name] = {
            'model': model,
            'history': history.history,
            'evaluation': evaluation,
            'predictions': predictions
        }
    
    return results, output_dirs