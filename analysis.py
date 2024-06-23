import os
import numpy as np
import matplotlib.pyplot as plt
from keras._tf_keras.keras.preprocessing import image
from keras._tf_keras.keras.applications.vgg16 import VGG16, preprocess_input
from keras._tf_keras.keras.models import Model
from sklearn.manifold import TSNE

# Function to load and preprocess images
def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    return img_data

def tsne(base_dir, classes):

    # Initialize the pre-trained model for feature extraction
    base_model = VGG16(weights='imagenet')
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)

    # Load images and labels
    features = []
    labels = []

    for idx, category in enumerate(classes):
        category_dir = os.path.join(base_dir, category)
        for root, dirs, files in os.walk(category_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(root, file)
                    img_data = load_and_preprocess_image(img_path)
                    feature = model.predict(img_data)
                    features.append(feature.flatten())
                    labels.append(idx)

    # Convert lists to arrays
    features = np.array(features)
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
    plt.savefig('Results/Analysis/tsne_fake_images.png')