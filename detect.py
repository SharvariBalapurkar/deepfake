import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
from timm import create_model
import torchvision.transforms as transforms

# **1️⃣ Load Xception Model**
class DeepfakeDetector(nn.Module):
    def __init__(self):
        super(DeepfakeDetector, self).__init__()
        self.model = create_model("legacy_xception", pretrained=True)

        num_ftrs = self.model.fc.in_features  # Get the input features of last layer
        self.model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# **2️⃣ Load Trained Model**
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DeepfakeDetector().to(device)
model.load_state_dict(torch.load("deepfake_model.pth", map_location=device))
model.eval()

# **3️⃣ Image Preprocessing**
transform = transforms.Compose([
    transforms.Resize((299, 299)),  # Xception needs 299x299 images
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# **4️⃣ Real-Time Deepfake Detection**
def live_deepfake_detection():
    cap = cv2.VideoCapture(0)  # Use default webcam
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image)
            prediction = torch.sigmoid(output).item()

        label = "FAKE" if prediction > 0.5 else "REAL"
        confidence = prediction if prediction > 0.5 else 1 - prediction

        text = f"{label} ({confidence:.2%})"
        color = (0, 0, 255) if label == "FAKE" else (0, 255, 0)
        cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imshow("Live Deepfake Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# **5️⃣ Run Real-Time Detection**
live_deepfake_detection()
