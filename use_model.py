# use_model.py

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from mp_hack_imageprocessing import CellClassifierCNN  # Import the model class

# Check if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data transformations for testing
data_transforms = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Path to the dataset
test_dataset_dir = 'path_to_your_test_dataset'

# Load the test dataset
test_dataset = datasets.ImageFolder(test_dataset_dir, transform=data_transforms)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load the model
model = CellClassifierCNN().to(device)

# Load trained weights (optional, if already trained)
model.load_state_dict(torch.load('cell_classification_model.pth'))

# Set the model to evaluation mode
model.eval()

# Inference on test dataset
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# Print accuracy on the test set
print(f'Test Accuracy: {100 * correct / total:.2f}%')

# Optionally, visualize some predictions
import numpy as np
import matplotlib.pyplot as plt

def imshow(img, title=None):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    if title is not None:
        plt.title(title)
    plt.show()

# Get a batch of test data
dataiter = iter(test_loader)
images, labels = dataiter.next()

# Get predictions
outputs = model(images.to(device))
_, preds = torch.max(outputs, 1)

# Show images with predicted labels
imshow(torchvision.utils.make_grid(images.cpu()), title=[test_dataset.classes[p] for p in preds])
