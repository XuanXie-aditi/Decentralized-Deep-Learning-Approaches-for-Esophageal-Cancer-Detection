# import os
# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd

# def analyze_image_features(image):
#     """Extract basic image features for analysis."""
#     # Convert normalized image back to 0-255 range for feature calculation
#     img_255 = (image * 255).astype(np.uint8)
    
#     features = {
#         'mean_intensity': np.mean(img_255),
#         'std_intensity': np.std(img_255),
#         'contrast': np.std(cv2.cvtColor(img_255, cv2.COLOR_BGR2LAB)[:,:,0]),
#         'brightness': np.mean(cv2.cvtColor(img_255, cv2.COLOR_BGR2LAB)[:,:,0]),
#         'blur_score': cv2.Laplacian(cv2.cvtColor(img_255, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var(),
#         'sharpness': cv2.Laplacian(img_255, cv2.CV_64F).var()
#     }
#     return features


# def plot_dataset_analysis(images, labels, output_dir):
#     """Create comprehensive dataset analysis plots."""
#     os.makedirs(output_dir, exist_ok=True)
    
#     # 1. Class Distribution Plot
#     plt.figure(figsize=(10, 6))
#     palette = ['#86a68b','#8c92ac']
#     sns.countplot(x=labels, palette=palette)
#     plt.title('Class Distribution')
#     plt.xlabel('Class (0: No-Esophagus, 1: Esophagus)')
#     plt.ylabel('Count')
#     for i in range(2):
#         count = np.sum(labels == i)
#         plt.text(i, count, f'{count}\n({count/len(labels):.1%})', 
#                 ha='center', va='bottom')
#     plt.savefig(os.path.join(output_dir, 'class_distribution.png'))
#     plt.close()
    
#     # 2. Feature Analysis
#     features_list = [analyze_image_features(img) for img in images]
#     features_df = pd.DataFrame(features_list)
#     features_df['class'] = labels
    
#     # Feature distributions by class
#     plt.figure(figsize=(15, 10))
#     for i, feature in enumerate(features_df.columns[:-1], 1):
#         plt.subplot(2, 3, i)
#         plt.title(f'{feature.replace("_", " ").title()} Distribution')
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_dir, 'feature_distributions.png'))
#     plt.close()
    
#     # 3. Sample Images Grid
#     plt.figure(figsize=(15, 6))
#     for class_label in [0, 1]:
#         class_indices = np.where(labels == class_label)[0]
#         samples = np.random.choice(class_indices, min(5, len(class_indices)), replace=False)
#         for i, idx in enumerate(samples):
#             plt.subplot(2, 5, i + 1 + class_label * 5)
#             plt.imshow(images[idx])
#             plt.axis('off')
#             plt.title(f'{"Esophagus" if class_label == 1 else "No-Esophagus"}')
#     plt.savefig(os.path.join(output_dir, 'sample_images.png'))
#     plt.close()
    
#     return features_df

# def load_and_preprocess_data(data_dir):
#     """Load and preprocess images with enhanced analysis."""
#     images = []
#     labels = []
#     masks = []
#     image_stats = {
#         'filenames': [],
#         'shapes': [],
#         'sizes_mb': [],
#         'total_esophagus_images': 0,
#         'total_mask_images': 0,
#         'total_no_esophagus_images': 0
#     }
    
#     # Load esophagus images and masks
#     cancer_dir = os.path.join(data_dir, 'esophagus')
#     mask_dir = os.path.join(cancer_dir, 'masks')
    
#     esophagus_images = [f for f in os.listdir(cancer_dir) 
#                        if f.endswith(('.jpg', '.png')) and os.path.isfile(os.path.join(cancer_dir, f))]
#     image_stats['total_esophagus_images'] = len(esophagus_images)
    
#     # Process esophagus images
#     for img_name in esophagus_images:
#         img_path = os.path.join(cancer_dir, img_name)
#         img = cv2.imread(img_path)
#         if img is not None:
#             image_stats['filenames'].append(img_name)
#             image_stats['shapes'].append(img.shape)
#             image_stats['sizes_mb'].append(os.path.getsize(img_path) / (1024 * 1024))
            
#             img = cv2.resize(img, (224, 224))
#             img = img / 255.0
#             images.append(img)
#             labels.append(1)
            
#             # Handle mask
#             mask_path = os.path.join(mask_dir, img_name)
#             if os.path.exists(mask_path):
#                 mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
#                 mask = cv2.resize(mask, (224, 224))
#                 mask = mask / 255.0
#                 image_stats['total_mask_images'] += 1
#             else:
#                 mask = np.zeros((224, 224))
#             masks.append(mask)
    
#     # Load no-esophagus images
#     non_cancer_dir = os.path.join(data_dir, 'no-esophagus')
#     non_cancer_images = [f for f in os.listdir(non_cancer_dir) 
#                         if f.endswith(('.jpg', '.png'))]
#     image_stats['total_no_esophagus_images'] = len(non_cancer_images)
    
#     for img_name in non_cancer_images:
#         img_path = os.path.join(non_cancer_dir, img_name)
#         img = cv2.imread(img_path)
#         if img is not None:
#             image_stats['filenames'].append(img_name)
#             image_stats['shapes'].append(img.shape)
#             image_stats['sizes_mb'].append(os.path.getsize(img_path) / (1024 * 1024))
            
#             img = cv2.resize(img, (224, 224))
#             img = img / 255.0
#             images.append(img)
#             labels.append(0)
#             masks.append(np.zeros((224, 224)))

#     # Convert to numpy arrays
#     images = np.array(images)
#     labels = np.array(labels)
#     masks = np.array(masks)
    
#     # Create output directory
#     output_dir = 'training_outputs/dataset_stats'
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Generate and save detailed analysis
#     features_df = plot_dataset_analysis(images, labels, output_dir)
    
#     # Save detailed statistics
#     stats_df = pd.DataFrame({
#         'Filename': image_stats['filenames'],
#         'Original Shape': image_stats['shapes'],
#         'Size (MB)': image_stats['sizes_mb']
#     })
#     #stats_df.to_csv(os.path.join(output_dir, 'image_statistics.csv'), index=False)
    
#     # Save summary statistics
#     with open(os.path.join(output_dir, 'dataset_summary.txt'), 'w') as f:
#         f.write("Dataset Summary\n")
#         f.write("-" * 50 + "\n")
#         f.write(f"Total Images: {len(images)}\n")
#         f.write(f"Esophagus Images: {image_stats['total_esophagus_images']}\n")
#         f.write(f"No-Esophagus Images: {image_stats['total_no_esophagus_images']}\n")
#         #f.write(f"Mask Images: {image_stats['total_mask_images']}\n")
#         f.write("\nImage Statistics:\n")
#         f.write(f"Average Size: {np.mean(image_stats['sizes_mb']):.2f} MB\n")
#         f.write(f"Size Range: {min(image_stats['sizes_mb']):.2f} - {max(image_stats['sizes_mb']):.2f} MB\n")
        
#     return images, labels, masks, image_stats




import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def analyze_image_features(image):
    """Extract basic image features for analysis."""
    # Convert normalized image back to 0-255 range for feature calculation
    img_255 = (image * 255).astype(np.uint8)
    
    features = {
        'mean_intensity': np.mean(img_255),
        'std_intensity': np.std(img_255),
        'contrast': np.std(cv2.cvtColor(img_255, cv2.COLOR_BGR2LAB)[:,:,0]),
        'brightness': np.mean(cv2.cvtColor(img_255, cv2.COLOR_BGR2LAB)[:,:,0]),
        'blur_score': cv2.Laplacian(cv2.cvtColor(img_255, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var(),
        'sharpness': cv2.Laplacian(img_255, cv2.CV_64F).var()
    }
    return features

def plot_dataset_analysis(images, labels, output_dir):
    """Create comprehensive dataset analysis plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Class Distribution Plot
    plt.figure(figsize=(10, 6))
    palette = ['#86a68b','#8c92ac']
    sns.countplot(x=labels, palette=palette)
    plt.title('Class Distribution')
    plt.xlabel('Class (0: No-Esophagus, 1: Esophagus)')
    plt.ylabel('Count')
    for i in range(2):
        count = np.sum(labels == i)
        plt.text(i, count, f'{count}\n({count/len(labels):.1%})', 
                ha='center', va='bottom')
    plt.savefig(os.path.join(output_dir, 'class_distribution.png'))
    plt.close()
    
    # 2. Feature Analysis
    features_list = [analyze_image_features(img) for img in images]
    features_df = pd.DataFrame(features_list)
    features_df['class'] = labels
    
    # Feature distributions by class
    plt.figure(figsize=(15, 10))
    for i, feature in enumerate(features_df.columns[:-1], 1):
        plt.subplot(2, 3, i)
        plt.title(f'{feature.replace("_", " ").title()} Distribution')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_distributions.png'))
    plt.close()
    
    # 3. Sample Images Grid
    plt.figure(figsize=(15, 6))
    for class_label in [0, 1]:
        class_indices = np.where(labels == class_label)[0]
        samples = np.random.choice(class_indices, min(5, len(class_indices)), replace=False)
        for i, idx in enumerate(samples):
            plt.subplot(2, 5, i + 1 + class_label * 5)
            plt.imshow(images[idx])
            plt.axis('off')
            plt.title(f'{"Esophagus" if class_label == 1 else "No-Esophagus"}')
    plt.savefig(os.path.join(output_dir, 'sample_images.png'))
    plt.close()
    
    return features_df

def create_feature_distribution_plots(features_df, output_dir):
    """Create and save the feature distribution plots in a single figure."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Mean Intensity Distribution
    sns.histplot(features_df['mean_intensity'], kde=True, ax=axes[0, 0])
    axes[0, 0].set_title('Mean Intensity Distribution')

    # Std Intensity Distribution
    sns.histplot(features_df['std_intensity'], kde=True, ax=axes[0, 1])
    axes[0, 1].set_title('Std Intensity Distribution')

    # Contrast Distribution
    sns.histplot(features_df['contrast'], kde=True, ax=axes[0, 2])
    axes[0, 2].set_title('Contrast Distribution')

    # Brightness Distribution
    sns.histplot(features_df['brightness'], kde=True, ax=axes[1, 0])
    axes[1, 0].set_title('Brightness Distribution')

    # Blur Score Distribution
    sns.histplot(features_df['blur_score'], kde=True, ax=axes[1, 1])
    axes[1, 1].set_title('Blur Score Distribution')

    # Sharpness Distribution
    sns.histplot(features_df['sharpness'], kde=True, ax=axes[1, 2])
    axes[1, 2].set_title('Sharpness Distribution')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_distributions.png'))
    plt.close()

def load_and_preprocess_data(data_dir):
    """Load and preprocess images with enhanced analysis."""
    images = []
    labels = []
    masks = []
    image_stats = {
        'filenames': [],
        'shapes': [],
        'sizes_mb': [],
        'total_esophagus_images': 0,
        'total_mask_images': 0,
        'total_no_esophagus_images': 0
    }
    
    # Load esophagus images and masks
    cancer_dir = os.path.join(data_dir, 'esophagus')
    mask_dir = os.path.join(cancer_dir, 'masks')
    
    esophagus_images = [f for f in os.listdir(cancer_dir) 
                       if f.endswith(('.jpg', '.png')) and os.path.isfile(os.path.join(cancer_dir, f))]
    image_stats['total_esophagus_images'] = len(esophagus_images)
    
    # Process esophagus images
    for img_name in esophagus_images:
        img_path = os.path.join(cancer_dir, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            image_stats['filenames'].append(img_name)
            image_stats['shapes'].append(img.shape)
            image_stats['sizes_mb'].append(os.path.getsize(img_path) / (1024 * 1024))
            
            img = cv2.resize(img, (224, 224))
            img = img / 255.0
            images.append(img)
            labels.append(1)
            
            # Handle mask
            mask_path = os.path.join(mask_dir, img_name)
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                mask = cv2.resize(mask, (224, 224))
                mask = mask / 255.0
                image_stats['total_mask_images'] += 1
            else:
                mask = np.zeros((224, 224))
            masks.append(mask)
    
    # Load no-esophagus images
    non_cancer_dir = os.path.join(data_dir, 'no-esophagus')
    non_cancer_images = [f for f in os.listdir(non_cancer_dir) 
                        if f.endswith(('.jpg', '.png'))]
    image_stats['total_no_esophagus_images'] = len(non_cancer_images)
    
    for img_name in non_cancer_images:
        img_path = os.path.join(non_cancer_dir, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            image_stats['filenames'].append(img_name)
            image_stats['shapes'].append(img.shape)
            image_stats['sizes_mb'].append(os.path.getsize(img_path) / (1024 * 1024))
            
            img = cv2.resize(img, (224, 224))
            img = img / 255.0
            images.append(img)
            labels.append(0)
            masks.append(np.zeros((224, 224)))

    # Convert to numpy arrays
    images = np.array(images)
    labels = np.array(labels)
    masks = np.array(masks)
    
    # Create output directory
    output_dir = 'training_outputs/dataset_stats'
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate and save detailed analysis
    features_df = plot_dataset_analysis(images, labels, output_dir)
    
    # Compute and save the feature distribution plots
    create_feature_distribution_plots(features_df, output_dir)
    
    # Save detailed statistics
    stats_df = pd.DataFrame({
        'Filename': image_stats['filenames'],
        'Original Shape': image_stats['shapes'],
        'Size (MB)': image_stats['sizes_mb']
    })
    #stats_df.to_csv(os.path.join(output_dir, 'image_statistics.csv'), index=False)
    
    # Save summary statistics
    with open(os.path.join(output_dir, 'dataset_summary.txt'), 'w') as f:
        f.write("Dataset Summary\n")
        f.write("-" * 50 + "\n")
        f.write(f"Total Images: {len(images)}\n")
        f.write(f"Esophagus Images: {image_stats['total_esophagus_images']}\n")
        f.write(f"No-Esophagus Images: {image_stats['total_no_esophagus_images']}\n")
        #f.write(f"Mask Images: {image_stats['total_mask_images']}\n")
        f.write("\nImage Statistics:\n")
        f.write(f"Average Size: {np.mean(image_stats['sizes_mb']):.2f} MB\n")
        f.write(f"Size Range: {min(image_stats['sizes_mb']):.2f} - {max(image_stats['sizes_mb']):.2f} MB\n")
        
    return images, labels, masks, image_stats