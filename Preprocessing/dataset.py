from torchvision.io import read_image

class Entities_Dataset:
    def __init__(self, dataset, transform=None):
        self.data = dataset
        self.class_map = {"aneurysmatic bone cyst": 0, "chondroblastoma": 1, "chondrosarcoma": 2,
                          "enchondroma": 3, "ewing sarcoma": 4, "fibruous dysplasia": 5,
                          "giant cell tumour": 6, "non-ossifying fibroma": 7, "osteochondroma": 8,
                          "osteosarcoma": 9}
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        img_path, class_name = self.data[item]
        image = read_image(img_path)
        label = self.class_map[class_name]
        if self.transform:
            image = self.transform(image)
        return image, label

class Categories_Dataset:
    def __init__(self, dataset, transform=None):
        self.data = dataset
        self.class_map = {"benign": 0, "intermediate": 1, "malignant": 2}
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        img_path, class_name = self.data[item]
        image = read_image(img_path)
        label = self.class_map[class_name]
        if self.transform:
            image = self.transform(image)
        return image, label
