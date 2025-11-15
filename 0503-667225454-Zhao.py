import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import glob


MODEL_LOAD_PATH = "0502-667225454-Zhao.ZZZ"
CLASSES = [
    'Circle', 'Square', 'Octagon', 'Heptagon', 'Nonagon', 
    'Star', 'Hexagon', 'Pentagon', 'Triangle'
]
IDX_TO_CLASS = {i: cls_name for i, cls_name in enumerate(CLASSES)}
IMG_SIZE = 200


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=len(CLASSES)):
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


def run_inference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN(num_classes=len(CLASSES))
    
    try:
        model.load_state_dict(torch.load(MODEL_LOAD_PATH, map_location=device))
    except FileNotFoundError:
        return
        
    model.to(device)
    model.eval()
    
    data_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    image_files = glob.glob("*.png") + glob.glob("*.jpg") + glob.glob("*.jpeg")
    
    if not image_files:
        return

    # Process each image
    with torch.no_grad():
        for img_path in image_files:
            try:
                image = Image.open(img_path).convert('RGB')
                image_tensor = data_transform(image)
                image_tensor = image_tensor.unsqueeze(0).to(device)

                output = model(image_tensor)
                _, predicted_idx = torch.max(output.data, 1)

                predicted_class = IDX_TO_CLASS[predicted_idx.item()]

                print(f"{os.path.basename(img_path)}: {predicted_class}")
                
            except Exception as e:
                print(f"Could not process {img_path}: {e}")

if __name__ == "__main__":
    run_inference()