import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import glob
import pickle
import matplotlib.pyplot as plt


DATASET_ROOT = "./geometry_dataset" 
MODEL_SAVE_PATH = "0502-667225454-Zhao1.ZZZ"
SPLIT_FILES = {
    "train": "training.file",
    "test": "testing.file"
}

CLASSES = [
    'Circle', 'Square', 'Octagon', 'Heptagon', 'Nonagon', 
    'Star', 'Hexagon', 'Pentagon', 'Triangle'
]
CLASS_TO_IDX = {cls_name: i for i, cls_name in enumerate(CLASSES)}
IDX_TO_CLASS = {i: cls_name for i, cls_name in enumerate(CLASSES)}

IMG_SIZE = 200
NUM_CLASSES = 9
BATCH_SIZE = 64
NUM_EPOCHS = 10
LEARNING_RATE = 0.01


def create_data_splits():
    train_files = []
    test_files = []

    # Get ALL .png files in the root folder
    all_files_path = os.path.join(DATASET_ROOT, "*.png")
    all_files = sorted(glob.glob(all_files_path))

    if not all_files:
        return [], []

    for class_name in CLASSES:
        class_files = [f for f in all_files if os.path.basename(f).lower().startswith(class_name.lower())]
        
        if len(class_files) != 10000:
            print(f"Warning: Found {len(class_files)} for class {class_name}, expected 10000.")
        
        train_files.extend(class_files[:8000])
        test_files.extend(class_files[8000:10000])

    # Save
    with open(SPLIT_FILES["train"], 'wb') as f:
        pickle.dump(train_files, f)
    with open(SPLIT_FILES["test"], 'wb') as f:
        pickle.dump(test_files, f)
        
    return train_files, test_files

def get_data_splits():
    if os.path.exists(SPLIT_FILES["train"]) and os.path.exists(SPLIT_FILES["test"]):
        with open(SPLIT_FILES["train"], 'rb') as f:
            train_files = pickle.load(f)
        with open(SPLIT_FILES["test"], 'rb') as f:
            test_files = pickle.load(f)
    else:
        train_files, test_files = create_data_splits()
    
    if not train_files or not test_files:
        exit()
        
    return train_files, test_files

class GeometryDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]

        image = Image.open(img_path).convert('RGB')
        
        filename = os.path.basename(img_path)
        label = -1
        
        for class_name in CLASSES:
            if filename.lower().startswith(class_name.lower()):
                label = CLASS_TO_IDX[class_name]
                break
        
        if label == -1:
            label = 0

        if self.transform:
            image = self.transform(image)
            
        return image, label

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(64 * 25 * 25, 128)
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))

        x = x.view(x.size(0), -1) 
        
        x = self.relu4(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def plot_metrics(train_loss, test_loss, train_acc, test_acc):
    epochs = range(1, len(train_loss) + 1)
    
    # Loss
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_loss, 'b-', label='Training Loss')
    plt.plot(epochs, test_loss, 'r-', label='Test Set Loss')
    plt.title('Epochs vs. Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig("epoch_vs_loss.png")

    # Accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_acc, 'b-', label='Training Accuracy')
    plt.plot(epochs, test_acc, 'r-', label='Test Set Accuracy')
    plt.title('Epochs vs. Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig("epoch_vs_accuracy.png")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    train_files, test_files = get_data_splits()
    
    train_dataset = GeometryDataset(train_files, transform=data_transform)
    test_dataset = GeometryDataset(test_files, transform=data_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    print("DataLoaders created.")
    
    # Initialization
    model = SimpleCNN(num_classes=NUM_CLASSES).to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("Starting model training...")

    # Training Loop
    history = {
        'train_loss': [], 'test_loss': [],
        'train_acc': [], 'test_acc': []
    }

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Calculate training accuracy
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        # Calculate epoch stats for training
        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_acc = 100 * correct_train / total_train
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)

        # Validation (Test Set) Loop
        model.eval()
        running_loss_test = 0.0
        correct_test = 0
        total_test = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss_test += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()

        # Calculate epoch stats for testing
        epoch_test_loss = running_loss_test / len(test_loader)
        epoch_test_acc = 100 * correct_test / total_test
        history['test_loss'].append(epoch_test_loss)
        history['test_acc'].append(epoch_test_acc)
        
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], "
              f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}%, "
              f"Test Loss: {epoch_test_loss:.4f}, Test Acc: {epoch_test_acc:.2f}%")

    print("Finished Training.")

    # Save Model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")
    
    # Check model size
    model_size_mb = os.path.getsize(MODEL_SAVE_PATH) / (1024 * 1024)
    print(f"Model size: {model_size_mb:.2f} MB")
    if model_size_mb > 50:
        print("Warning: Model size exceeds 50MB limit! You may need a simpler architecture.")

    # Generate Plots
    plot_metrics(
        history['train_loss'], history['test_loss'],
        history['train_acc'], history['test_acc']
    )

if __name__ == "__main__":
    main()