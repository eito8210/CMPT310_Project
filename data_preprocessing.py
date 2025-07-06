import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import cv2
from tqdm import tqdm
import zipfile
import tempfile
import shutil

# TensorFlow/Kerasのインポートをオプションにする
TENSORFLOW_AVAILABLE = False
KERAS_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    TENSORFLOW_AVAILABLE = True
    print("✅ Using TensorFlow Keras for data augmentation")
except ImportError:
    try:
        from keras.preprocessing.image import ImageDataGenerator
        KERAS_AVAILABLE = True
        print("✅ Using standalone Keras for data augmentation")
    except ImportError:
        print("⚠️  Neither TensorFlow nor Keras available. Using manual augmentation only.")

# 使用可能フラグ
DATA_AUGMENTATION_AVAILABLE = TENSORFLOW_AVAILABLE or KERAS_AVAILABLE

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
    
    def extract_zip_datasets(self, positives_zip="positives.zip", negatives_zip="negatives.zip"):
        """
        Extract zip files containing positive (smile) and negative (no-smile) images
        Args:
            positives_zip: Path to zip file with smile images
            negatives_zip: Path to zip file with no-smile images
        Returns:
            Paths to extracted directories
        """
        print("Extracting team-provided zip datasets...")
        
        # Create extraction directories
        extract_dir = os.path.join(self.data_dir, 'raw', 'team_data')
        smile_dir = os.path.join(extract_dir, 'smile')
        no_smile_dir = os.path.join(extract_dir, 'no_smile')
        
        os.makedirs(smile_dir, exist_ok=True)
        os.makedirs(no_smile_dir, exist_ok=True)
        
        # Extract positives (smile images)
        if os.path.exists(positives_zip):
            print(f"Extracting {positives_zip} to {smile_dir}")
            with zipfile.ZipFile(positives_zip, 'r') as zip_ref:
                zip_ref.extractall(smile_dir)
            print(f"✅ Extracted smile images to {smile_dir}")
        else:
            print(f"⚠️  {positives_zip} not found")
        
        # Extract negatives (no-smile images)
        if os.path.exists(negatives_zip):
            print(f"Extracting {negatives_zip} to {no_smile_dir}")
            with zipfile.ZipFile(negatives_zip, 'r') as zip_ref:
                zip_ref.extractall(no_smile_dir)
            print(f"✅ Extracted no-smile images to {no_smile_dir}")
        else:
            print(f"⚠️  {negatives_zip} not found")
        
        # Count extracted images
        smile_count = self._count_images_in_directory(smile_dir)
        no_smile_count = self._count_images_in_directory(no_smile_dir)
        
        print(f"Dataset extracted:")
        print(f"  Smile images: {smile_count}")
        print(f"  No-smile images: {no_smile_count}")
        print(f"  Total: {smile_count + no_smile_count}")
        
        return smile_dir, no_smile_dir
    
    def _count_images_in_directory(self, directory):
        """
        Recursively count image files in directory (including subdirectories)
        """
        count = 0
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    count += 1
        return count
    
    def _collect_image_paths_and_labels(self, smile_dir, no_smile_dir):
        """
        Collect all image paths and create corresponding labels
        Handles nested directory structures from zip extraction
        """
        image_paths = []
        labels = []
        
        # Collect smile images (label = 1)
        print("Collecting smile image paths...")
        for root, dirs, files in os.walk(smile_dir):
            for file in tqdm(files, desc="Smile images"):
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    image_paths.append(os.path.join(root, file))
                    labels.append(1)  # smile = 1
        
        # Collect no-smile images (label = 0)
        print("Collecting no-smile image paths...")
        for root, dirs, files in os.walk(no_smile_dir):
            for file in tqdm(files, desc="No-smile images"):
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    image_paths.append(os.path.join(root, file))
                    labels.append(0)  # no-smile = 0
        
        print(f"Collected {len(image_paths)} total image paths")
        return image_paths, labels
    
    def process_team_dataset(self, positives_zip="positives.zip", negatives_zip="negatives.zip"):
        """
        Complete pipeline to process team-provided zip datasets
        Args:
            positives_zip: Path to zip file with smile images
            negatives_zip: Path to zip file with no-smile images
        Returns:
            Processed train and validation sets
        """
        print("=== PROCESSING TEAM DATASET ===")
        
        # Step 1: Extract zip files
        smile_dir, no_smile_dir = self.extract_zip_datasets(positives_zip, negatives_zip)
        
        # Step 2: Collect all image paths and labels
        image_paths, labels = self._collect_image_paths_and_labels(smile_dir, no_smile_dir)
        
        if len(image_paths) == 0:
            print("❌ No images found in zip files!")
            return None, None, None, None
        
        # Step 3: Preprocess all images
        print("Preprocessing images...")
        X, y = self.preprocess_image_batch(image_paths, labels, batch_size=50)
        
        if len(X) == 0:
            print("❌ No valid images after preprocessing!")
            return None, None, None, None
        
        # Step 4: Balance dataset if needed
        print("Balancing dataset...")
        X_balanced, y_balanced = self.balance_dataset(X, y, method='undersample')
        
        # Step 5: Create train/validation split
        print("Creating train/validation split...")
        X_train, X_val, y_train, y_val = self.create_train_validation_split(
            X_balanced, y_balanced, test_size=0.2, random_state=42
        )
        
        # Step 6: Apply data augmentation to training set
        if DATA_AUGMENTATION_AVAILABLE:
            print("Applying Keras/TensorFlow data augmentation...")
            X_train_aug, y_train_aug = self.apply_augmentation(X_train, y_train, augmentation_factor=2)
        else:
            print("Using manual OpenCV data augmentation...")
            X_train_aug, y_train_aug = self._manual_augmentation(X_train, y_train, augmentation_factor=2)
        
        # Step 7: Generate report
        self.generate_dataset_report(X_train_aug, X_val, y_train_aug, y_val)
        
        # Step 8: Save processed dataset
        self.save_processed_dataset(X_train_aug, X_val, y_train_aug, y_val, 'team_dataset.npz')
        
        return X_train_aug, X_val, y_train_aug, y_val
    
    def download_fer_dataset(self):
        """
        Download and extract FER+ dataset
        Note: For actual use, download from official Microsoft FER+ repository
        """
        print("Setting up FER+ dataset structure...")
        
        # Check if FERPlus directory exists (from git clone)
        ferplus_repo = "./FERPlus"
        if os.path.exists(ferplus_repo):
            print(f"✅ Found FERPlus repository at: {ferplus_repo}")
            
            # Check for FER+ annotation file
            fer_plus_csv = os.path.join(ferplus_repo, "fer2013new.csv")
            if os.path.exists(fer_plus_csv):
                print(f"✅ Found FER+ annotations: {fer_plus_csv}")
                
                # Check for original FER2013 data
                fer2013_csv = os.path.join(ferplus_repo, "fer2013.csv")
                if os.path.exists(fer2013_csv):
                    print("✅ Found FER2013 data - ready for processing!")
                    
                    # Process FER+ data automatically
                    return self._process_fer_plus_data(fer2013_csv, fer_plus_csv)
                else:
                    print(f"⚠️  Missing: {fer2013_csv}")
                    print("Note: CSV processing not available, but image processing may work.")
            else:
                print(f"⚠️  Missing FER+ annotations in {ferplus_repo}")
        else:
            print("⚠️  FERPlus repository not found")
            print("Note: CSV processing not available, but image processing may work.")
        
        # Create standard directory structure as fallback
        fer_dir = os.path.join(self.data_dir, 'raw', 'fer_plus')
        os.makedirs(os.path.join(fer_dir, 'smile'), exist_ok=True)
        os.makedirs(os.path.join(fer_dir, 'no_smile'), exist_ok=True)
        
        print(f"FER+ directory created at: {fer_dir}")
        
        # Check if images are already there and process
        if self._check_and_process_fer_dataset(fer_dir):
            return True  # Successfully processed images
        
        print("\nFER+ processing options:")
        print("1. Download fer2013.csv from Kaggle and place in ./FERPlus/")
        print("2. Or place pre-classified images in:")
        print(f"   - Smile images: {os.path.join(fer_dir, 'smile')}")
        print(f"   - No-smile images: {os.path.join(fer_dir, 'no_smile')}")
        
        return False  # No processing completed
    
    def _process_fer_plus_data(self, fer2013_csv, fer_plus_csv):
        """
        Process FER2013 + FER+ data automatically
        """
        print("🔄 Processing FER+ data for smile detection...")
        
        try:
            import pandas as pd
            
            # Read data
            print("Loading FER2013 and FER+ data...")
            fer2013 = pd.read_csv(fer2013_csv)
            fer_plus = pd.read_csv(fer_plus_csv)
            
            print(f"FER2013: {len(fer2013)} samples")
            print(f"FER+: {len(fer_plus)} samples")
            
            # Create output directory
            output_dir = os.path.join(self.data_dir, 'raw', 'fer_plus')
            smile_dir = os.path.join(output_dir, 'smile')
            no_smile_dir = os.path.join(output_dir, 'no_smile')
            
            os.makedirs(smile_dir, exist_ok=True)
            os.makedirs(no_smile_dir, exist_ok=True)
            
            smile_count = 0
            no_smile_count = 0
            
            print("Extracting smile/no-smile images...")
            
            for idx, row in tqdm(fer_plus.iterrows(), total=min(len(fer_plus), len(fer2013)), desc="Processing FER+"):
                try:
                    if idx >= len(fer2013):
                        break
                        
                    # Get image pixels
                    pixels = fer2013.iloc[idx]['pixels']
                    pixel_array = np.array([int(x) for x in pixels.split()], dtype=np.uint8)
                    img = pixel_array.reshape(48, 48)
                    
                    # Convert to RGB
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                    
                    # Get happiness score from FER+
                    happiness = float(row.get('happiness', 0)) if pd.notna(row.get('happiness', 0)) else 0
                    
                    # Sum of other emotions
                    other_emotions = sum([
                        float(row.get('neutral', 0)) if pd.notna(row.get('neutral', 0)) else 0,
                        float(row.get('surprise', 0)) if pd.notna(row.get('surprise', 0)) else 0,
                        float(row.get('sadness', 0)) if pd.notna(row.get('sadness', 0)) else 0,
                        float(row.get('anger', 0)) if pd.notna(row.get('anger', 0)) else 0,
                        float(row.get('disgust', 0)) if pd.notna(row.get('disgust', 0)) else 0,
                        float(row.get('fear', 0)) if pd.notna(row.get('fear', 0)) else 0,
                        float(row.get('contempt', 0)) if pd.notna(row.get('contempt', 0)) else 0
                    ])
                    
                    # Classify as smile if happiness dominates
                    if happiness > other_emotions and happiness > 0:
                        filename = f"fer_smile_{smile_count:06d}.png"
                        cv2.imwrite(os.path.join(smile_dir, filename), img_rgb)
                        smile_count += 1
                    else:
                        filename = f"fer_no_smile_{no_smile_count:06d}.png"
                        cv2.imwrite(os.path.join(no_smile_dir, filename), img_rgb)
                        no_smile_count += 1
                        
                except Exception as e:
                    continue
            
            print(f"✅ FER+ processing completed!")
            print(f"   Smile images: {smile_count}")
            print(f"   No-smile images: {no_smile_count}")
            print(f"   Total: {smile_count + no_smile_count}")
            
            # Now process the images using existing pipeline
            if smile_count > 0 and no_smile_count > 0:
                return self._check_and_process_fer_dataset(output_dir)
            
        except ImportError:
            print("❌ pandas required for FER+ processing: pip install pandas")
        except Exception as e:
            print(f"❌ Error processing FER+ data: {e}")
        
        return False
    
    def _check_and_process_fer_dataset(self, fer_dir):
        """
        Check if FER+ data exists and process automatically
        """
        smile_dir = os.path.join(fer_dir, 'smile')
        no_smile_dir = os.path.join(fer_dir, 'no_smile')
        
        smile_count = self._count_images_in_directory(smile_dir)
        no_smile_count = self._count_images_in_directory(no_smile_dir)
        
        if smile_count > 0 and no_smile_count > 0:
            print(f"Found FER+ images: {smile_count} smile, {no_smile_count} no-smile")
            
            # Process the FER+ dataset
            image_paths, labels = self._collect_image_paths_and_labels(smile_dir, no_smile_dir)
            
            if len(image_paths) > 0:
                print("Processing FER+ dataset...")
                X, y = self.preprocess_image_batch(image_paths, labels)
                
                if len(X) > 0:
                    X_balanced, y_balanced = self.balance_dataset(X, y)
                    X_train, X_val, y_train, y_val = self.create_train_validation_split(X_balanced, y_balanced)
                    
                    if DATA_AUGMENTATION_AVAILABLE:
                        X_train_aug, y_train_aug = self.apply_augmentation(X_train, y_train)
                    else:
                        X_train_aug, y_train_aug = self._manual_augmentation(X_train, y_train)
                    
                    self.save_processed_dataset(X_train_aug, X_val, y_train_aug, y_val, 'fer_dataset.npz')
                    self.generate_dataset_report(X_train_aug, X_val, y_train_aug, y_val)
                    
                    print("✅ FER+ dataset processing completed!")
                    return True
        
        return False
    
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
            ImageDataGenerator with smile-specific augmentations or None
        """
        if not DATA_AUGMENTATION_AVAILABLE:
            return None
            
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
            Augmented dataset (with zero‐padding images removed)
        """
        print("Applying data augmentation...")
        
        datagen = self.create_advanced_augmentation()
        if datagen is None:
            return self._manual_augmentation(X_train, y_train, augmentation_factor)
        
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
            
            aug_img = next(aug_iter)[0]
            X_augmented[current_idx] = aug_img
            y_augmented[current_idx] = label
            current_idx += 1

                
        
        print(f"Augmentation completed: {original_size} -> {current_idx} images")

    # trim off any zero-padded entries before returning
        X_final = X_augmented[:current_idx]
        y_final = y_augmented[:current_idx]
        return X_final, y_final
    
    def _manual_augmentation(self, X_train, y_train, augmentation_factor=3):
        """
        Manual data augmentation using OpenCV when TensorFlow is not available
        """
        print("Using manual augmentation (OpenCV-based)...")
        
        original_size = len(X_train)
        augmented_size = original_size * (augmentation_factor + 1)
        
        X_augmented = np.zeros((augmented_size, *X_train.shape[1:]), dtype=np.float32)
        y_augmented = np.zeros(augmented_size, dtype=np.int32)
        
        # Add original data
        X_augmented[:original_size] = X_train
        y_augmented[:original_size] = y_train
        
        current_idx = original_size
        
        for i in tqdm(range(original_size), desc="Manual augmentation"):
            img = X_train[i]
            label = y_train[i]
            
            for _ in range(augmentation_factor):
                if current_idx < augmented_size:
                    # Apply random transformations
                    aug_img = self._apply_manual_transforms(img)
                    X_augmented[current_idx] = aug_img
                    y_augmented[current_idx] = label
                    current_idx += 1
        
        print(f"Manual augmentation completed: {original_size} -> {augmented_size} images")
        return X_augmented, y_augmented
    
    def _apply_manual_transforms(self, img):
        """
        Apply manual transformations to an image
        """
        # Convert to uint8 for OpenCV operations
        img_uint8 = (img * 255).astype(np.uint8)
        h, w = img_uint8.shape[:2]
        
        # Random rotation (-15 to 15 degrees)
        angle = np.random.uniform(-15, 15)
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        img_uint8 = cv2.warpAffine(img_uint8, rotation_matrix, (w, h))
        
        # Random horizontal flip
        if np.random.random() > 0.5:
            img_uint8 = cv2.flip(img_uint8, 1)
        
        # Random brightness adjustment
        brightness_factor = np.random.uniform(0.8, 1.3)
        img_uint8 = np.clip(img_uint8 * brightness_factor, 0, 255).astype(np.uint8)
        
        # Convert back to float32 and normalize
        return img_uint8.astype(np.float32) / 255.0
    
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
        Visualize random samples from the dataset, skipping zero images
        """
        # Filter out zero (blank) images
        valid_indices = [i for i in range(len(X)) if X[i].mean() > 0]
        if len(valid_indices) < num_samples:
            print(f"Warning: only {len(valid_indices)} valid images available, reducing sample count accordingly.")
            num_samples = len(valid_indices)

        indices = np.random.choice(valid_indices, num_samples, replace=False)







        
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
    print("1. Team Zip Files (positives.zip & negatives.zip)")
    print("2. FER+ Dataset")
    print("3. CelebA Dataset") 
    print("4. Custom Images")
    
    choice = input("Enter choice (1-4): ").strip()
    
    if choice == '1':
        # Team zip files workflow
        print("\n=== PROCESSING TEAM ZIP FILES ===")
        
        # Check if zip files exist
        pos_zip = "positives.zip"
        neg_zip = "negatives.zip"
        
        if os.path.exists(pos_zip) and os.path.exists(neg_zip):
            print(f"Found: {pos_zip} and {neg_zip}")
            
            # Process the team dataset
            X_train, X_val, y_train, y_val = processor.process_team_dataset(pos_zip, neg_zip)
            
            if X_train is not None:
                # Visualize samples
                processor.visualize_dataset_samples(X_train, y_train, num_samples=16, title="Team Dataset Samples")
                
                print("\n✅ Team dataset processing completed successfully!")
                print("📊 Dataset ready for CNN training (Seungyeop's part)")
                print("💾 Processed data saved as 'team_dataset.npz'")
            else:
                print("❌ Failed to process team dataset")
        else:
            print(f"❌ Zip files not found:")
            print(f"  - {pos_zip}: {'✅' if os.path.exists(pos_zip) else '❌'}")
            print(f"  - {neg_zip}: {'✅' if os.path.exists(neg_zip) else '❌'}")
            print("Please ensure both zip files are in the current directory.")
        
    elif choice == '2':
        # FER+ workflow
        print("\n=== PROCESSING FER+ DATASET ===")
        success = processor.download_fer_dataset()
        
        if success:
            print("\n✅ FER+ dataset processing completed successfully!")
            print("📊 Dataset ready for CNN training (Seungyeop's part)")
            print("💾 Processed data saved as 'fer_dataset.npz'")
            print(f"🎯 Expected accuracy: >90% (high-quality dataset)")

            # visualize 16 random training samples from FER+
            X_train, _, y_train, _ = processor.load_processed_dataset('fer_dataset.npz')
            processor.visualize_dataset_samples(
                X_train, y_train,
                num_samples=16,
                title="FER+ Dataset Samples"
            )
            
        else:
            print("\n🔄 FER+ dataset setup completed, but no data processed yet.")
            print("Please ensure images are placed in the correct directories and run again.")
        
    elif choice == '3':
    # CelebA workflow with optimized settings
        print("\n=== PROCESSING CELEBA DATASET (OPTIMIZED) ===")
        
        # Create optimized processor for CelebA (smaller image size)
        celeba_processor = SmileDatasetProcessor(data_dir='./smile_dataset', img_size=(32, 32))
        
        celeba_dir = celeba_processor.setup_celeba_dataset()
        
        # Process annotations if available
        smile_labels = celeba_processor.process_celeba_annotations(celeba_dir)
        
        if smile_labels:
            print(f"Found {len(smile_labels)} annotated images")
            
            # Create image paths and labels
            image_paths = []
            labels = []
            
            images_dir = os.path.join(celeba_dir, 'images')
            
            # Check for img_align_celeba folder
            actual_images_dir = os.path.join(images_dir, 'img_align_celeba')
            if os.path.exists(actual_images_dir):
                images_dir = actual_images_dir
            
            print("Collecting CelebA image paths...")
            for filename, smile_label in tqdm(smile_labels.items(), desc="Processing images"):
                img_path = os.path.join(images_dir, filename)
                if os.path.exists(img_path):
                    image_paths.append(img_path)
                    labels.append(smile_label)
            
            if image_paths:
                print(f"Found {len(image_paths)} valid images")
                
                # Preprocess images with smaller batch size for CelebA
                print("Preprocessing CelebA images (optimized settings)...")
                X, y = celeba_processor.preprocess_image_batch(image_paths, labels, batch_size=50)
                
                if len(X) > 0:
                    print("Balancing CelebA dataset...")
                    X_balanced, y_balanced = celeba_processor.balance_dataset(X, y)
                    
                    print("Creating train/validation split...")
                    X_train, X_val, y_train, y_val = celeba_processor.create_train_validation_split(X_balanced, y_balanced)
                    
                    print("Applying data augmentation...")
                    if DATA_AUGMENTATION_AVAILABLE:
                        X_train_aug, y_train_aug = celeba_processor.apply_augmentation(X_train, y_train, augmentation_factor=2)
                    else:
                        X_train_aug, y_train_aug = celeba_processor._manual_augmentation(X_train, y_train, augmentation_factor=2)
                    
                    # Save processed dataset
                    celeba_processor.save_processed_dataset(X_train_aug, X_val, y_train_aug, y_val, 'celeba_dataset.npz')
                    
                    # Generate report
                    celeba_processor.generate_dataset_report(X_train_aug, X_val, y_train_aug, y_val)
                    
                    # Visualize samples
                    celeba_processor.visualize_dataset_samples(X_train_aug, y_train_aug, num_samples=16, title="CelebA Dataset Samples")
                    
                    print("\n✅ CelebA dataset processing completed successfully!")
                    print("📊 Dataset ready for CNN training (Seungyeop's part)")
                    print("💾 Processed data saved as 'celeba_dataset.npz'")
                else:
                    print("❌ No valid images after preprocessing")
            else:
                print("❌ No valid images found")
        else:
            print("❌ No annotations found")
        
    elif choice == '4':
        # Custom images workflow (enhanced)
        print("\n=== CUSTOM IMAGES SETUP ===")
        print("Place your images in:")
        print("  - ./smile_dataset/raw/custom/smile/")
        print("  - ./smile_dataset/raw/custom/no_smile/")
        
        custom_smile_dir = os.path.join(processor.data_dir, 'raw', 'custom', 'smile')
        custom_no_smile_dir = os.path.join(processor.data_dir, 'raw', 'custom', 'no_smile')
        
        os.makedirs(custom_smile_dir, exist_ok=True)
        os.makedirs(custom_no_smile_dir, exist_ok=True)
        
        # Check if images exist
        smile_images = [f for f in os.listdir(custom_smile_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        no_smile_images = [f for f in os.listdir(custom_no_smile_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        
        if smile_images and no_smile_images:
            print(f"Found {len(smile_images)} smile images and {len(no_smile_images)} no-smile images")
            
            # Process custom dataset using existing pipeline
            print("\n=== PROCESSING CUSTOM DATASET ===")
            
            # Create image paths and labels
            image_paths = []
            labels = []
            
            print("Collecting custom image paths...")
            for img in smile_images:
                image_paths.append(os.path.join(custom_smile_dir, img))
                labels.append(1)  # smile = 1
            
            for img in no_smile_images:
                image_paths.append(os.path.join(custom_no_smile_dir, img))
                labels.append(0)  # no-smile = 0
            
            # Process using existing pipeline
            print("Preprocessing custom images...")
            X, y = processor.preprocess_image_batch(image_paths, labels)
            
            if len(X) > 0:
                print("Balancing custom dataset...")
                X_balanced, y_balanced = processor.balance_dataset(X, y)
                
                print("Creating train/validation split...")
                X_train, X_val, y_train, y_val = processor.create_train_validation_split(X_balanced, y_balanced)
                
                print("Applying data augmentation...")
                if DATA_AUGMENTATION_AVAILABLE:
                    X_train_aug, y_train_aug = processor.apply_augmentation(X_train, y_train, augmentation_factor=2)
                else:
                    X_train_aug, y_train_aug = processor._manual_augmentation(X_train, y_train, augmentation_factor=2)
                
                # Save processed dataset
                processor.save_processed_dataset(X_train_aug, X_val, y_train_aug, y_val, 'custom_dataset.npz')
                
                # Generate report
                processor.generate_dataset_report(X_train_aug, X_val, y_train_aug, y_val)
                
                # Visualize samples
                processor.visualize_dataset_samples(X_train_aug, y_train_aug, num_samples=16, title="Custom Dataset Samples")
                
                print("✅ Custom dataset processing completed!")
                print("📊 Dataset ready for CNN training")
                print("💾 Processed data saved as 'custom_dataset.npz'")
            else:
                print("❌ No valid images after preprocessing")
        else:
            print("❌ No images found in custom directories")
            print("\nTo add custom images:")
            print("1. Copy smile images to: ./smile_dataset/raw/custom/smile/")
            print("2. Copy no-smile images to: ./smile_dataset/raw/custom/no_smile/")
            print("3. Supported formats: .png, .jpg, .jpeg, .bmp, .tiff")
            print("4. Run this script again")
            
            # Provide sample commands
            print("\nExample commands:")
            print("cp /path/to/smile/images/*.jpg ./smile_dataset/raw/custom/smile/")
            print("cp /path/to/no_smile/images/*.jpg ./smile_dataset/raw/custom/no_smile/")
        
    else:
        print("Invalid choice. Please run again and select 1-4.")
        
    print("\nNext steps based on your choice:")
    if choice == '1':
        print("✅ Option 1 (Team Zip): Ready for CNN training")
    elif choice == '2':
        print("✅ Option 2 (FER+): Ready for CNN training") 
    else:
        print("🔄 Option 3 (CelebA): Download dataset and rerun") 
        print("🔄 Option 4 (Custom): Add images and rerun")
    
    print("\nFinal step: Train CNN model with Seungyeop's code")
    print("Use fer_dataset.npz or team_dataset.npz for training")

if __name__ == "__main__":
    main()