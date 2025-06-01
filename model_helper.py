def predict(image_path):
    # Add device detection at start
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)  # Add device here

    global trained_model

    if trained_model is None:
        trained_model = CarClassifierResNet().to(device)  # Add device here
        # Use forward slashes for path compatibility
        trained_model.load_state_dict(torch.load("model/saved_model.pth", map_location=device))  # Critical map_location
        trained_model.eval()

    with torch.no_grad():
        output = trained_model(image_tensor)
        _, predicted_class = torch.max(output, 1)
        return class_names[predicted_class.item()]