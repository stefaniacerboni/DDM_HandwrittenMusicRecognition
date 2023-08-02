import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageOps


class CustomDataset(Dataset):
    def __init__(self, root_dir, thresh_file, output_height=100):
        self.root_dir = root_dir
        self.thresh_file = thresh_file
        self.output_height = output_height
        self.file_names = self.read_file_names_from_thresh()
        self.rhythm_labels, self.pitch_labels = self.read_labels_from_thresh()
        self.transform = ResizeAndPad(output_height)
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.file_names)

    def binarize_image(self, img):
        threshold = 128
        return img.point(lambda x: 0 if x < threshold else 1)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.file_names[idx])
        # Read the image
        image = Image.open(img_path).convert("L")  # Convert to grayscale

        # Resize the image using the custom transform (ResizeAndPad)
        image = self.transform(image)

        # Convert to tensor and binarize
        image = self.binarize_image(image)

        image = self.to_tensor(image)

        rhythm_label = self.rhythm_labels[idx]
        pitch_label = self.pitch_labels[idx]

        return image, rhythm_label, pitch_label

    def read_file_names_from_thresh(self):
        with open(self.thresh_file, 'r') as f:
            file_names = [line.strip().split("$")[0] + ".png" for line in f]
        return file_names

    def read_labels_from_thresh(self):
        rhythm = []
        pitch = []
        with open(self.thresh_file, 'r') as f:
            labels = [line.strip().split("|")[1] for line in f]
            for row in labels:
                rhythm_row = [elem.split(".")[0] if elem != "epsilon" else elem for elem in row.split("~")]
                pitch_row = [elem.split(".")[1] if elem != "epsilon" else elem for elem in row.split("~")]
                rhythm.append(rhythm_row)
                pitch.append(pitch_row)
        return rhythm, pitch


# Custom transformation to resize and pad images
class ResizeAndPad:
    def __init__(self, output_height):
        self.output_height = output_height

    def __call__(self, image):
        aspect_ratio = image.size[0] / image.size[1]
        output_width = int(aspect_ratio * self.output_height)

        # Resize the image while maintaining the aspect ratio using torchvision's functional resize
        image = transforms.functional.resize(image, (self.output_height, output_width))

        return image
