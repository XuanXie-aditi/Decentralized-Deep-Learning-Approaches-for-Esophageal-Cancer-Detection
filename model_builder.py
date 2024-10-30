import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16, ResNet50, DenseNet121
from tensorflow.keras.optimizers import Adam

def create_dual_output_model(base_model_name):
    """
    Create a dual-output model for simultaneous classification and segmentation.
    
    Args:
        base_model_name (str): Name of the base model to use ('ResNet50', 'VGG16', or 'DenseNet121')
    
    Returns:
        tensorflow.keras.Model: Compiled dual-output model
    """
    # Input layer
    input_tensor = layers.Input(shape=(224, 224, 3))
    
    # Select and configure base model
    if base_model_name == 'ResNet50':
        base_model = ResNet50(include_top=False, weights='imagenet', input_tensor=input_tensor)
    elif base_model_name == 'VGG16':
        base_model = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)
    elif base_model_name == 'DenseNet121':
        base_model = DenseNet121(include_top=False, weights='imagenet', input_tensor=input_tensor)
    else:
        raise ValueError("Unsupported base model name")
    
    # Freeze the base model layers
    base_model.trainable = False
    
    # Get the output from the base model
    x = base_model.output
    
    # Classification branch
    classification_branch = layers.GlobalAveragePooling2D()(x)
    classification_branch = layers.Dense(512, activation='relu')(classification_branch)
    classification_branch = layers.Dropout(0.3)(classification_branch)
    classification_branch = layers.Dense(256, activation='relu')(classification_branch)
    classification_output = layers.Dense(1, activation='sigmoid', name='classification')(classification_branch)
    
    # Segmentation branch
    segmentation_branch = x
    
    # First upsampling block (2x)
    segmentation_branch = layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same')(segmentation_branch)
    segmentation_branch = layers.BatchNormalization()(segmentation_branch)
    segmentation_branch = layers.Activation('relu')(segmentation_branch)
    
    # Second upsampling block (4x)
    segmentation_branch = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(segmentation_branch)
    segmentation_branch = layers.BatchNormalization()(segmentation_branch)
    segmentation_branch = layers.Activation('relu')(segmentation_branch)
    
    # Third upsampling block (8x)
    segmentation_branch = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(segmentation_branch)
    segmentation_branch = layers.BatchNormalization()(segmentation_branch)
    segmentation_branch = layers.Activation('relu')(segmentation_branch)
    
    # Fourth upsampling block (16x)
    segmentation_branch = layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(segmentation_branch)
    segmentation_branch = layers.BatchNormalization()(segmentation_branch)
    segmentation_branch = layers.Activation('relu')(segmentation_branch)
    
    # Fifth upsampling block (32x) to match input size
    segmentation_branch = layers.Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same')(segmentation_branch)
    segmentation_branch = layers.BatchNormalization()(segmentation_branch)
    segmentation_branch = layers.Activation('relu')(segmentation_branch)
    
    # Final segmentation output
    segmentation_output = layers.Conv2D(1, (1, 1), activation='sigmoid', name='segmentation')(segmentation_branch)
    
    # Create model with both outputs
    model = models.Model(
        inputs=input_tensor,
        outputs=[classification_output, segmentation_output]
    )
    
    return model