def predict(image_path):
    # Force CPU usage (Streamlit Cloud doesn't have GPU)
    device = torch.device('cpu')

    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)

    global trained_model

    if trained_model is None:
        trained_model = CarClassifierResNet().to(device)
        # Use forward slashes and map_location
        trained_model.load_state_dict(
            torch.load("model/saved_model.pth", map_location=device)
        )
        trained_model.eval()

    with torch.no_grad():
        output = trained_model(image_tensor)
        _, predicted_class = torch.max(output, 1)
        return class_names[predicted_class.item()]