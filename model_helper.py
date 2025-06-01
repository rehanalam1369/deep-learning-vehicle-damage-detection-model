import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image

# Initialize model and class names
class_names = ['Front Breakage', 'Front Crushed', 'Front Normal',
               'Rear Breakage', 'Rear Crushed', 'Rear Normal']


# Load model class (modified for better compatibility)
class CarClassifierResNet(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.model = models.resnet50(weights=None)  # Removed 'DEFAULT' for compatibility
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.layer4.parameters():
            param.requires_grad = True
        self.model.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.model.fc.in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)


# Load model weights (modified for reliability)
def load_model():
    device = torch.device('cpu')
    model = CarClassifierResNet().to(device)
    try:
        model.load_state_dict(torch.load("model/saved_model.pth", map_location=device))
    except:
        # Handle case where model file structure is different
        model.load_state_dict(torch.load("model/saved_model.pth", map_location=device)['state_dict'])
    model.eval()
    return model


# Prediction function (simplified and robust)
def predict(image_path):
    # Initialize model if not already loaded
    if not hasattr(predict, 'model'):
        predict.model = load_model()

    # Load and transform image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    try:
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0)

        # Make prediction
        with torch.no_grad():
            output = predict.model(image_tensor)
            _, predicted = torch.max(output, 1)
            return class_names[predicted.item()]

    except Exception as e:
        return f"Prediction error: {str(e)}"