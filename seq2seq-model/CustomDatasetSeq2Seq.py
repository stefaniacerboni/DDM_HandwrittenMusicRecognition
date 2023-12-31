import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class CustomDatasetSeq2Seq(Dataset):
    def __init__(self, root_dir, thresh_file, output_width=224):
        self.root_dir = root_dir
        self.thresh_file = thresh_file
        self.output_width = output_width
        self.file_names = self.read_file_names_from_thresh()
        self.labels = self.read_labels_from_thresh()
        self.transform = ResizeAndPad(output_width)
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.file_names[idx])
        # Read the image
        image = Image.open(img_path)

        # Resize the image using the custom transform (ResizeAndPad)
        image = self.transform(image)

        image = self.to_tensor(image)

        label = self.labels[idx]

        return image, label

    def read_file_names_from_thresh(self):
        with open(self.thresh_file, 'r') as f:
            file_names = [line.strip().split("$")[0] + ".png" for line in f]
        return file_names

    def read_labels_from_thresh(self):
        res = []
        with open(self.thresh_file, 'r') as f:
            labels = [line.strip().split("|")[1] for line in f]
            for row in labels:
                rhythm_row = [elem for elem in row.split("~")]
                res.append(rhythm_row)
        return res


# Custom transformation to resize and pad images
class ResizeAndPad:
    def __init__(self, output_width):
        self.output_width = output_width

    def __call__(self, image):
        width, height = image.size
        new_height = int((self.output_width / width) * height)
        # Resize the image to the new dimensions
        resized_img = image.resize((self.output_width, new_height), Image.ANTIALIAS)
        # Create a new blank canvas
        canvas = Image.new("RGB", (self.output_width, self.output_width), (218, 202, 177))

        # Calculate padding to center the resized image
        top_padding = (self.output_width - new_height) // 2
        bottom_padding = self.output_width - new_height - top_padding

        # Paste the resized image onto the canvas
        canvas.paste(resized_img, (0, top_padding))

        return canvas
