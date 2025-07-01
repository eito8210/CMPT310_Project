import torch
from torch import nn, optim
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Import your data preprocessing class
from data_preprocessing import SmileDatasetProcessor

# === Config ===
BATCH_SIZE = 32
EPOCHS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def convert_to_torch_dataset(X, y):
    """
    Convert numpy arrays from your preprocessing to PyTorch tensors
    """
    # Convert to PyTorch tensors
    X_tensor = torch.from_numpy(X).float()
    y_tensor = torch.from_numpy(y).long()
    
    # Rearrange dimensions: (N, H, W, C) -> (N, C, H, W)
    X_tensor = X_tensor.permute(0, 3, 1, 2)
    
    # Create TensorDataset
    dataset = TensorDataset(X_tensor, y_tensor)
    return dataset

def main():
    """
    Integrated training using your data preprocessing
    """
    print("=== INTEGRATED SMILE DETECTION TRAINING ===")
    
    # Step 1: Use your preprocessing pipeline
    processor = SmileDatasetProcessor(data_dir='./smile_dataset', img_size=(224, 224))  # ResNet needs 224x224
    
    # Load your processed data
    X_train, X_val, y_train, y_val = processor.load_processed_dataset()
    
    if X_train is None:
        print("No processed dataset found. Please run data preprocessing first.")
        print("Example:")
        print("1. Run data_preprocessing.py")
        print("2. Process your dataset")
        print("3. Save processed data")
        print("4. Run this training script")
        return
    
    print(f"Loaded dataset: {len(X_train)} train, {len(X_val)} validation samples")
    
    # Step 2: Convert to PyTorch datasets
    train_dataset = convert_to_torch_dataset(X_train, y_train)
    val_dataset = convert_to_torch_dataset(X_val, y_val)
    
    # Step 3: Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Step 4: Setup model (same as original)
    from torchvision import models
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 2)  # binary classification
    model = model.to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    
    # Step 5: Training loop
    print("Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
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
        
        # Validation
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
    
    # Step 6: Save model
    torch.save(model.state_dict(), 'smile_resnet18.pt')
    print("✅ Model saved as smile_resnet18.pt")
    
    # Step 7: Generate final report using your processor
    processor.generate_dataset_report(X_train, X_val, y_train, y_val)

if __name__ == "__main__":
    main()