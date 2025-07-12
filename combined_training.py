import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models
import numpy as np
import os

# Import your data preprocessing class
from data_preprocessing import SmileDatasetProcessor

# === Config ===
BATCH_SIZE = 32
EPOCHS = 20 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def convert_to_torch_dataset(X, y):
    """
    Convert NumPy arrays to PyTorch tensors
    """
    X_tensor = torch.from_numpy(X).float()
    y_tensor = torch.from_numpy(y).long()
    
    # Reshape dimensions: (N, H, W, C) → (N, C, H, W)
    X_tensor = X_tensor.permute(0, 3, 1, 2)
    
    return TensorDataset(X_tensor, y_tensor)

def safe_load_dataset(processor, filename):
    """
    Safely load dataset with error handling
    """
    try:
        return processor.load_processed_dataset(filename)
    except Exception as e:
        print(f"❌ Error loading {filename}: {e}")
        return None, None, None, None

def combine_64x64_datasets(datasets):
    """
    Combine only 64x64 datasets for memory safety
    """
    all_X_train = []
    all_X_val = []
    all_y_train = []
    all_y_val = []
    
    for dataset_name, (X_train, X_val, y_train, y_val) in datasets.items():
        if X_train is not None:
            current_size = X_train.shape[1:3]
            print(f"📊 {dataset_name}: Train {len(X_train)}, Val {len(X_val)} samples (Size: {current_size})")
            
            # Only use 64x64 datasets
            if current_size == (64, 64):
                print(f"✅ Including {dataset_name} (64x64)")
                all_X_train.append(X_train)
                all_X_val.append(X_val)
                all_y_train.append(y_train)
                all_y_val.append(y_val)
            else:
                print(f"⚠️  Skipping {dataset_name} (Size: {current_size}) - Memory optimization")
        else:
            print(f"❌ {dataset_name}: Failed to load")
    
    if not all_X_train:
        return None, None, None, None
    
    # Combine 64x64 datasets
    print("🔄 Combining 64x64 datasets...")
    combined_X_train = np.concatenate(all_X_train, axis=0)
    combined_X_val = np.concatenate(all_X_val, axis=0)
    combined_y_train = np.concatenate(all_y_train, axis=0)
    combined_y_val = np.concatenate(all_y_val, axis=0)
    
    print(f"🔥 Combined: Train {len(combined_X_train)}, Val {len(combined_X_val)} samples (64x64)")
    
    return combined_X_train, combined_X_val, combined_y_train, combined_y_val

def main():
    print("=== MEMORY SAFE TRAINING (64x64 DATASETS) ===")
    print("Using Team ZIP + FER+ datasets for optimal memory usage")
    
    # Step 1: Load preprocessed data
    processor = SmileDatasetProcessor()
    
    datasets = {}
    
    # Team ZIP Files (64x64)
    print("\n🔍 Loading Team ZIP dataset...")
    team_data = safe_load_dataset(processor, 'team_dataset.npz')
    datasets['Team ZIP Files'] = team_data
    
    # FER+ Dataset (64x64)
    print("\n🔍 Loading FER+ dataset...")
    fer_data = safe_load_dataset(processor, 'fer_dataset.npz')
    datasets['FER+ Dataset'] = fer_data
    
    # CelebA Dataset (32x32) - Check but don't use for memory safety
    print("\n🔍 Checking CelebA dataset...")
    celeba_data = safe_load_dataset(processor, 'celeba_dataset.npz')
    datasets['CelebA Dataset'] = celeba_data
    
    # Step 2: Combine only 64x64 datasets
    print("\n=== COMBINING 64x64 DATASETS ===")
    X_train, X_val, y_train, y_val = combine_64x64_datasets(datasets)
    
    if X_train is None:
        print("❌ No compatible 64x64 datasets found")
        print("Please ensure team_dataset.npz and fer_dataset.npz are available")
        return
    
    # Dataset statistics
    total_samples = len(X_train) + len(X_val)
    smile_ratio = (np.sum(y_train == 1) + np.sum(y_val == 1)) / total_samples * 100
    used_datasets = len([d for name, d in datasets.items() if d[0] is not None and name != 'CelebA Dataset'])
    
    print(f"\n📊 Final Dataset Statistics:")
    print(f"  Used datasets: {used_datasets}/2 (64x64 only)")
    print(f"  Total samples: {total_samples:,}")
    print(f"  Training: {len(X_train):,}")
    print(f"  Validation: {len(X_val):,}")
    print(f"  Image size: 64x64")
    print(f"  Smile ratio: {smile_ratio:.1f}%")
    
    # Step 3: Convert to PyTorch datasets
    print("\n=== CONVERTING TO PYTORCH DATASETS ===")
    train_dataset = convert_to_torch_dataset(X_train, y_train)
    val_dataset = convert_to_torch_dataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Step 4: Setup ResNet18 model for 64x64
    print("=== SETTING UP RESNET18 MODEL (64x64 OPTIMIZED) ===")
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 2)  # Binary classification
    model = model.to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    
    # Step 5: Training loop
    print(f"\n=== STARTING CNN TRAINING ({EPOCHS} epochs) ===")
    best_val_acc = 0
    
    for epoch in range(EPOCHS):
        # Training phase
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Progress update every 50 batches
            if batch_idx % 50 == 0:
                print(f"  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        train_acc = 100 * correct / total
        val_acc = 100 * val_correct / val_total
        
        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"  Train Loss: {total_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'smile_resnet18_best.pt')
            print(f"  🏆 New best model saved! Accuracy: {val_acc:.2f}%")
        
        print("-" * 50)
    
    # Save final model
    torch.save(model.state_dict(), 'smile_resnet18_final.pt')
    
    print("\n" + "="*60)
    print("🎉 MEMORY SAFE TRAINING COMPLETED!")
    print("="*60)
    print(f"Used datasets: {used_datasets}/2 (Team ZIP + FER+)")
    print(f"Total training samples: {len(X_train):,}")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Saved models:")
    print(f"  - smile_resnet18_best.pt (highest accuracy)")
    print(f"  - smile_resnet18_final.pt (final model)")
    print("="*60)
    
    # Performance note
    if total_samples > 30000:
        print("🚀 Excellent! Large dataset should provide high accuracy!")
    print("💡 Memory optimized: Using 64x64 datasets only for stability")

if __name__ == "__main__":
    main()