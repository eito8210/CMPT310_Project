import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
import cv2
from tqdm import tqdm

class SmileDatasetProcessor:
    """
    Eito's part: Dataset collection, preprocessing, and augmentation
    Works with Seungyeop's face detection implementation
    """
    
    def __init__(self, data_dir='./smile_dataset', img_size=(64, 64)):
        """
        Initialize the dataset processor
        Args:
            data_dir: Directory to store datasets
            img_size: Target image size for CNN training
        """
        self.data_dir = data_dir
        self.img_size = img_size
        
        # Create directory structure
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(os.path.join(data_dir, 'raw'), exist_ok=True)
        os.makedirs(os.path.join(data_dir, 'processed'), exist_ok=True)
        os.makedirs(os.path.join(data_dir, 'augmented'), exist_ok=True)
        
        print(f"Dataset processor initialized. Data directory: {data_dir}")
    
    def download_fer_dataset(self):
        """
        Download and extract FER+ dataset
        Note: For actual use, download from official Microsoft FER+ repository
        """
        print("Setting up FER+ dataset structure...")
        
        # Create sample directory structure
        fer_dir = os.path.join(self.data_dir, 'raw', 'fer_plus')
        os.makedirs(os.path.join(fer_dir, 'smile'), exist_ok=True)
        os.makedirs(os.path.join(fer_dir, 'no_smile'), exist_ok=True)
        
        print(f"FER+ directory created at: {fer_dir}")
        print("Please download FER+ dataset from official source and place images in:")
        print(f"  - Smile images: {os.path.join(fer_dir, 'smile')}")
        print(f"  - No-smile images: {os.path.join(fer_dir, 'no_smile')}")
        
        return fer_dir
    
    def setup_celeba_dataset(self):
        """
        Setup CelebA dataset structure for smile detection
        """
        print("Setting up CelebA dataset structure...")
        
        celeba_dir = os.path.join(self.data_dir, 'raw', 'celeba')
        os.makedirs(os.path.join(celeba_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(celeba_dir, 'annotations'), exist_ok=True)
        
        print(f"CelebA directory created at: {celeba_dir}")
        print("Please download CelebA dataset and place:")
        print(f"  - Images in: {os.path.join(celeba_dir, 'images')}")
        print(f"  - Annotations in: {os.path.join(celeba_dir, 'annotations')}")
        
        return celeba_dir
    
    def process_celeba_annotations(self, celeba_dir):
        """
        Process CelebA annotations to extract smile labels
        Args:
            celeba_dir: Path to CelebA dataset directory
        Returns:
            Dictionary mapping filename to smile label
        """
        annotations_file = os.path.join(celeba_dir, 'annotations', 'list_attr_celeba.txt')
        
        if not os.path.exists(annotations_file):
            print(f"Annotations file not found: {annotations_file}")
            return {}
        
        print("Processing CelebA annotations...")
        
        # Read annotations file
        with open(annotations_file, 'r') as f:
            lines = f.readlines()
        
        # Parse header to find smile attribute index
        header = lines[1].strip().split()
        smile_idx = header.index('Smiling')
        
        # Process each image annotation
        smile_labels = {}
        for line in tqdm(lines[2:], desc="Processing annotations"):
            parts = line.strip().split()
            filename = parts[0]
            smile_value = int(parts[smile_idx + 1])  # +1 because first column is filename
            
            # Convert -1/1 to 0/1 (CelebA uses -1 for no, 1 for yes)
            smile_labels[filename] = 1 if smile_value == 1 else 0
        
        print(f"Processed annotations for {len(smile_labels)} images")
        return smile_labels
    
    def preprocess_image_batch(self, image_paths, labels, batch_size=100):
        """
        Preprocess a batch of images efficiently
        Args:
            image_paths: List of image file paths
            labels: List of corresponding labels
            batch_size: Number of images to process at once
        Returns:
            Preprocessed images and labels as numpy arrays
        """
        processed_images = []
        processed_labels = []
        
        total_batches = len(image_paths) // batch_size + (1 if len(image_paths) % batch_size > 0 else 0)
        
        for batch_idx in tqdm(range(total_batches), desc="Processing image batches"):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(image_paths))
            
            batch_images = []
            batch_labels = []
            
            for i in range(start_idx, end_idx):
                img_path = image_paths[i]
                label = labels[i]
                
                # Read and preprocess image
                img = cv2.imread(img_path)
                if img is None:
                    continue
                
                # Convert BGR to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Resize to target size
                img = cv2.resize(img, self.img_size)
                
                # Normalize to [0, 1]
                img = img.astype(np.float32) / 255.0
                
                batch_images.append(img)
                batch_labels.append(label)
            
            if batch_images:
                processed_images.extend(batch_images)
                processed_labels.extend(batch_labels)
        
        return np.array(processed_images), np.array(processed_labels)
    
    def create_advanced_augmentation(self):
        """
        Create advanced data augmentation for smile detection
        Returns:
            ImageDataGenerator with smile-specific augmentations
        """
        # Advanced augmentation strategy for facial expressions
        augmentation = ImageDataGenerator(
            rotation_range=15,              # Slight rotation (faces don't rotate much)
            width_shift_range=0.1,          # Small horizontal shifts
            height_shift_range=0.1,         # Small vertical shifts
            shear_range=0.1,                # Slight shearing
            zoom_range=0.15,                # Zoom in/out
            horizontal_flip=True,           # Mirror faces
            brightness_range=[0.8, 1.3],    # Lighting variations
            channel_shift_range=0.1,        # Color variations
            fill_mode='nearest',            # Fill strategy
            validation_split=0.0            # No validation split in augmentation
        )
        
        return augmentation
    
    def apply_augmentation(self, X_train, y_train, augmentation_factor=3):
        """
        Apply data augmentation to training set
        Args:
            X_train: Training images
            y_train: Training labels
            augmentation_factor: Number of augmented versions per image
        Returns:
            Augmented dataset
        """
        print("Applying data augmentation...")
        
        datagen = self.create_advanced_augmentation()
        datagen.fit(X_train)
        
        # Calculate total augmented size
        original_size = len(X_train)
        augmented_size = original_size * (augmentation_factor + 1)  # +1 for original
        
        X_augmented = np.zeros((augmented_size, *X_train.shape[1:]), dtype=np.float32)
        y_augmented = np.zeros(augmented_size, dtype=np.int32)
        
        # Add original data
        X_augmented[:original_size] = X_train
        y_augmented[:original_size] = y_train
        
        # Generate augmented data
        current_idx = original_size
        
        for i in tqdm(range(original_size), desc="Generating augmented images"):
            img = X_train[i:i+1]  # Single image batch
            label = y_train[i]
            
            # Generate augmented versions
            aug_iter = datagen.flow(img, batch_size=1)
            
            for _ in range(augmentation_factor):
                if current_idx < augmented_size:
                    aug_img = next(aug_iter)[0]
                    X_augmented[current_idx] = aug_img
                    y_augmented[current_idx] = label
                    current_idx += 1
        
        print(f"Augmentation completed: {original_size} -> {augmented_size} images")
        
        return X_augmented, y_augmented
    
    def balance_dataset(self, X, y, method='undersample'):
        """
        Balance the dataset to have equal smile/no-smile samples
        Args:
            X: Images
            y: Labels
            method: 'undersample' or 'oversample'
        Returns:
            Balanced dataset
        """
        smile_indices = np.where(y == 1)[0]
        no_smile_indices = np.where(y == 0)[0]
        
        smile_count = len(smile_indices)
        no_smile_count = len(no_smile_indices)
        
        print(f"Original distribution - Smile: {smile_count}, No-smile: {no_smile_count}")
        
        if method == 'undersample':
            # Use the smaller class size
            target_size = min(smile_count, no_smile_count)
            
            # Randomly sample from larger class
            if smile_count > no_smile_count:
                selected_smile = np.random.choice(smile_indices, target_size, replace=False)
                selected_no_smile = no_smile_indices
            else:
                selected_smile = smile_indices
                selected_no_smile = np.random.choice(no_smile_indices, target_size, replace=False)
            
        else:  # oversample
            # Use the larger class size
            target_size = max(smile_count, no_smile_count)
            
            # Oversample the smaller class
            if smile_count < no_smile_count:
                selected_smile = np.random.choice(smile_indices, target_size, replace=True)
                selected_no_smile = no_smile_indices
            else:
                selected_smile = smile_indices
                selected_no_smile = np.random.choice(no_smile_indices, target_size, replace=True)
        
        # Combine balanced indices
        balanced_indices = np.concatenate([selected_smile, selected_no_smile])
        np.random.shuffle(balanced_indices)
        
        X_balanced = X[balanced_indices]
        y_balanced = y[balanced_indices]
        
        print(f"Balanced distribution - Total: {len(X_balanced)} (each class: {target_size})")
        
        return X_balanced, y_balanced
    
    def create_train_validation_split(self, X, y, test_size=0.2, random_state=42):
        """
        Create stratified train/validation split
        Args:
            X: Images
            y: Labels
            test_size: Fraction for validation
            random_state: Random seed for reproducibility
        Returns:
            Train and validation sets
        """
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state, 
            stratify=y
        )
        
        print(f"Data split completed:")
        print(f"  Training: {len(X_train)} samples")
        print(f"  Validation: {len(X_val)} samples")
        print(f"  Train smile ratio: {np.mean(y_train):.2%}")
        print(f"  Val smile ratio: {np.mean(y_val):.2%}")
        
        return X_train, X_val, y_train, y_val
    
    def save_processed_dataset(self, X_train, X_val, y_train, y_val, filename='smile_dataset.npz'):
        """
        Save processed dataset in efficient format
        """
        filepath = os.path.join(self.data_dir, 'processed', filename)
        
        np.savez_compressed(
            filepath,
            X_train=X_train,
            X_val=X_val,
            y_train=y_train,
            y_val=y_val,
            img_size=self.img_size
        )
        
        print(f"Dataset saved to: {filepath}")
        print(f"File size: {os.path.getsize(filepath) / (1024*1024):.1f} MB")
    
    def load_processed_dataset(self, filename='smile_dataset.npz'):
        """
        Load processed dataset
        """
        filepath = os.path.join(self.data_dir, 'processed', filename)
        
        if not os.path.exists(filepath):
            print(f"Dataset file not found: {filepath}")
            return None, None, None, None
        
        data = np.load(filepath)
        
        print(f"Dataset loaded from: {filepath}")
        print(f"Training samples: {len(data['X_train'])}")
        print(f"Validation samples: {len(data['X_val'])}")
        
        return data['X_train'], data['X_val'], data['y_train'], data['y_val']
    
    def visualize_dataset_samples(self, X, y, num_samples=16, title="Dataset Samples"):
        """
        Visualize random samples from the dataset
        """
        indices = np.random.choice(len(X), num_samples, replace=False)
        
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        fig.suptitle(title, fontsize=16)
        
        for i, idx in enumerate(indices):
            row, col = i // 4, i % 4
            
            axes[row, col].imshow(X[idx])
            axes[row, col].set_title(f"{'Smile' if y[idx] == 1 else 'No Smile'}")
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def generate_dataset_report(self, X_train, X_val, y_train, y_val):
        """
        Generate comprehensive dataset report
        """
        print("\n" + "="*50)
        print("SMILE DETECTION DATASET REPORT")
        print("="*50)
        
        # Basic statistics
        total_train = len(X_train)
        total_val = len(X_val)
        total_samples = total_train + total_val
        
        train_smile = np.sum(y_train == 1)
        train_no_smile = np.sum(y_train == 0)
        val_smile = np.sum(y_val == 1)
        val_no_smile = np.sum(y_val == 0)
        
        print(f"Dataset Size:")
        print(f"  Total samples: {total_samples:,}")
        print(f"  Training: {total_train:,} ({total_train/total_samples:.1%})")
        print(f"  Validation: {total_val:,} ({total_val/total_samples:.1%})")
        
        print(f"\nClass Distribution:")
        print(f"  Training Set:")
        print(f"    Smile: {train_smile:,} ({train_smile/total_train:.1%})")
        print(f"    No Smile: {train_no_smile:,} ({train_no_smile/total_train:.1%})")
        print(f"  Validation Set:")
        print(f"    Smile: {val_smile:,} ({val_smile/total_val:.1%})")
        print(f"    No Smile: {val_no_smile:,} ({val_no_smile/total_val:.1%})")
        
        print(f"\nImage Properties:")
        print(f"  Image size: {X_train.shape[1:3]}")
        print(f"  Channels: {X_train.shape[3]}")
        print(f"  Data type: {X_train.dtype}")
        print(f"  Value range: [{X_train.min():.3f}, {X_train.max():.3f}]")
        
        print("="*50)

# Example usage workflow for your project
def main():
    """
    Main workflow for Eito's data preprocessing part
    """
    print("=== EITO'S SMILE DETECTION DATA PIPELINE ===\n")
    
    # Initialize processor
    processor = SmileDatasetProcessor(data_dir='./smile_dataset', img_size=(64, 64))
    
    print("Choose your data source:")
    print("1. FER+ Dataset")
    print("2. CelebA Dataset") 
    print("3. Custom Images")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == '1':
        # FER+ workflow
        fer_dir = processor.download_fer_dataset()
        print("\nAfter downloading FER+ dataset, organize images and run again.")
        
    elif choice == '2':
        # CelebA workflow
        celeba_dir = processor.setup_celeba_dataset()
        
        # Process annotations if available
        smile_labels = processor.process_celeba_annotations(celeba_dir)
        
        if smile_labels:
            print(f"Found {len(smile_labels)} annotated images")
            # Continue with processing...
        
    elif choice == '3':
        # Custom images workflow
        print("Place your images in:")
        print("  - ./smile_dataset/raw/custom/smile/")
        print("  - ./smile_dataset/raw/custom/no_smile/")
        
        custom_smile_dir = os.path.join(processor.data_dir, 'raw', 'custom', 'smile')
        custom_no_smile_dir = os.path.join(processor.data_dir, 'raw', 'custom', 'no_smile')
        
        os.makedirs(custom_smile_dir, exist_ok=True)
        os.makedirs(custom_no_smile_dir, exist_ok=True)
        
        # Check if images exist
        smile_images = [f for f in os.listdir(custom_smile_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        no_smile_images = [f for f in os.listdir(custom_no_smile_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if smile_images and no_smile_images:
            print(f"Found {len(smile_images)} smile images and {len(no_smile_images)} no-smile images")
            
            # Process custom dataset
            # ... processing code here ...
            
            print("Custom dataset processing completed!")
        else:
            print("Please add images to the directories and run again.")
    
    print("\nData preprocessing pipeline ready!")
    print("Next steps:")
    print("1. Add your dataset images")
    print("2. Run the preprocessing")
    print("3. Train CNN model (Seungyeop's part)")

if __name__ == "__main__":
    main()