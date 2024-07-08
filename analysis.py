import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.manifold import TSNE
import torch
from PIL import Image

# Limit the number of threads used by OpenMP
torch.set_num_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"

# Function to load and preprocess images
def load_and_preprocess_image(img_path):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(img_path).convert('RGB')
    img_data = preprocess(img)
    return img_data

# Custom dataset for loading images
class ImageDataset(Dataset):
    def __init__(self, base_dir, classes):
        self.base_dir = base_dir
        self.classes = classes
        self.image_paths = []
        self.labels = []
        for idx, category in enumerate(classes):
            category_dir = os.path.join(base_dir, category)
            for root, _, files in os.walk(category_dir):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.image_paths.append(os.path.join(root, file))
                        self.labels.append(idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        img_data = load_and_preprocess_image(img_path)
        return img_data, label

def tsne(base_dir, classes):
    # Set the device
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize the pre-trained model for feature extraction
    base_model = models.vgg16(pretrained=True).to(device)
    model = torch.nn.Sequential(*list(base_model.children())[:-1])  # Remove the classification layer
    model.eval()

    # Load images and labels
    dataset = ImageDataset(base_dir, classes)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    features = []
    labels = []

    with torch.no_grad():
        for img_data, label in dataloader:
            img_data = img_data.to(device)
            output = model(img_data).view(img_data.size(0), -1)  # Flatten the output
            features.append(output.cpu().numpy())
            labels.extend(label.numpy())

    # Convert lists to arrays
    features = np.vstack(features)
    labels = np.array(labels)

    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    features_tsne = tsne.fit_transform(features)

    # Plot the t-SNE results
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(features_tsne[:, 0], features_tsne[:, 1], c=labels, cmap='viridis', marker='o')
    plt.title('t-SNE plot of Images')
    plt.xlabel('t-SNE feature 1')
    plt.ylabel('t-SNE feature 2')

    # Add legend
    legend = plt.legend(handles=scatter.legend_elements()[0], labels=classes, title="Classes")
    plt.gca().add_artist(legend)

    # Save the plot
    os.makedirs('Results/Analysis', exist_ok=True)
    plt.savefig('Results/Analysis/tsne_fake_images.png')
    plt.show()