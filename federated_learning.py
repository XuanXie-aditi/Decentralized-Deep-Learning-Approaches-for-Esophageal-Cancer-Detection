import numpy as np
from tensorflow.keras.optimizers import Adam
from model_builder import create_dual_output_model

def federated_learning_simulation(best_model, best_model_name, X, y_class, y_seg, num_clients=3):
    """Simulate federated learning with the best model."""
    # Split data into clients
    client_data = []
    data_per_client = len(X) // num_clients
    
    for i in range(num_clients):
        start_idx = i * data_per_client
        end_idx = (i + 1) * data_per_client if i < num_clients - 1 else len(X)
        # Reshape the segmentation masks to ensure correct dimensions
        y_seg_reshaped = y_seg[start_idx:end_idx].reshape((-1, 224, 224, 1))
        client_data.append({
            'X': X[start_idx:end_idx],
            'y_class': y_class[start_idx:end_idx],
            'y_seg': y_seg_reshaped
        })
    
    # Federated learning rounds
    num_rounds = 5
    for round_num in range(num_rounds):
        print(f"\nFederated Learning Round {round_num + 1}")
        
        # Train on each client
        client_weights = []
        for client_id, client in enumerate(client_data):
            print(f"Training on client {client_id + 1}")
            
            # Create and train client model with the same architecture as best_model
            client_model = create_dual_output_model(base_model_name=best_model_name)
            
            # Compile the model before setting weights
            client_model.compile(
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
            
            # Set weights after compilation
            client_model.set_weights(best_model.get_weights())
            
            client_model.fit(
                client['X'],
                {'classification': client['y_class'], 'segmentation': client['y_seg']},
                epochs=2,  # Reduced epochs for each client
                batch_size=32,
                verbose=1
            )
            
            client_weights.append(client_model.get_weights())
        
        # Average weights
        avg_weights = [np.mean([client_weight[i] for client_weight in client_weights], axis=0)
                      for i in range(len(client_weights[0]))]
        
        # Update global model
        best_model.set_weights(avg_weights)
    
    return best_model