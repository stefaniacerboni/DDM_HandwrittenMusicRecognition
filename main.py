# Standard imports
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from CustomDataset import CustomDataset
from OMRModel import OMRModel
from mapping import rhythm_mapping, pitch_mapping, inverse_rhythm_mapping, inverse_pitch_mapping

plt.style.use('dark_background')


def show_images(num_images, dataloader, images_per_row=4):
    num_rows = (num_images + images_per_row - 1) // images_per_row

    for i in range(num_images):
        plt.subplot(num_rows, images_per_row, i + 1)
        image = np.transpose(dataloader.dataset[i][0].numpy(), (1, 2, 0))
        plt.axis('off')
        plt.title(dataloader.dataset.file_names[i])
        plt.imshow(image)

    plt.tight_layout()
    plt.show()


def collate_fn(batch):
    # Get the maximum width within the batch
    max_width = max(item[0].size(2) for item in batch)

    # Pad each image individually to the width of the widest image in the batch
    padded_images = [F.pad(item[0], (0, max_width - item[0].size(2)), value=255) for item in batch]

    # Stack the tensors along the batch dimension
    images = torch.stack(padded_images)

    rhythm_labels_list = [item[1] for item in batch]
    pitch_labels_list = [item[2] for item in batch]

    # Encode string labels to numerical values using mapping
    batch_rhythm_labels_encoded = [[inverse_rhythm_mapping[labels] for labels in row] for row in rhythm_labels_list]
    batch_pitch_labels_encoded = [[inverse_pitch_mapping[labels] for labels in row] for row in pitch_labels_list]

    def expand_labels(batch_labels_encoded, label_mapping):
        # Calculate the desired length (length of the longest sequence)
        desired_length = max(len(seq) for seq in batch_labels_encoded)

        # Create an empty list to store the expanded sequences
        expanded_batch_labels = []

        # Expand each sequence in the batch to the desired length
        for sequence in batch_labels_encoded:
            # Calculate the number of repetitions for each element in the sequence
            repeat_factors = [
                desired_length // len(sequence) + 1 if i < desired_length % len(sequence) else desired_length // len(
                    sequence) for i in range(len(sequence))]

            # Repeat each element in the sequence based on the calculated repeat factors
            expanded_sequence = sum([[element] * repeat_factors[i] for i, element in enumerate(sequence)], [])

            expanded_batch_labels.append(expanded_sequence)
        # Convert the expanded batch to a PyTorch tensor and one-hot encode
        expanded_batch_labels_tensor = torch.tensor(expanded_batch_labels)
        batch_labels_onehot = F.one_hot(expanded_batch_labels_tensor, num_classes=len(label_mapping)).float()
        return batch_labels_onehot
    # Convert the labels to one-hot matrices
    batch_rhythm_labels_onehot = expand_labels(batch_rhythm_labels_encoded, rhythm_mapping)
    batch_pitch_labels_onehot = expand_labels(batch_pitch_labels_encoded, pitch_mapping)
    return images, batch_rhythm_labels_onehot, batch_pitch_labels_onehot


def process_output(output_matrix, label_mapping=None):
    # Apply thresholding
    binary_matrix = (output_matrix > 0.5).float()

    # Create an empty list to store the symbols and pitches for each pixel column
    symbol_list = []
    pitch_list = []

    for batch in range(binary_matrix.size(0)):
        # Iterate through each pixel column
        for col in range(binary_matrix.size(1)):
            # Get the binary vector representing symbols and pitches at the current column
            symbol_vector = binary_matrix[batch, col, :]

            # Get the indices where symbols are present
            symbol_indices = (symbol_vector == 1).nonzero().squeeze()

            # Get the corresponding labels using the index-to-label mapping
            if symbol_indices.dim() == 0:
                symbols = [label_mapping[symbol_indices.item()]]
            else:
                symbols = [label_mapping[i.item()] for i in symbol_indices]
            # Join the symbols for each pixel column into a string
            symbol_list.append(symbols)
    symbol_strings = ['~'.join(sym) for sym in symbol_list]
    # Return the list of symbol strings
    return symbol_strings


# Create the custom dataset
root_dir = "words"
thresh_file_train = "gt_final.train.thresh"
train_dataset = CustomDataset(root_dir, thresh_file_train)
# Use DataLoader to load data in parallel and move to GPU
batch_size = 1
train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True, num_workers=4,
                          pin_memory=True)

# show_images(10, train_loader)

# Initialize the model and move it to the GPU
model = OMRModel(input_channels=1, num_pitch_classes=14, num_rhythm_classes=32)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load the saved model's state dictionary
#model.load_state_dict(torch.load('trained_model40epoch.pth'))
model.to(device)

# Define the loss function
criterion = nn.SmoothL1Loss()
# Define the optimizer (e.g., SGD)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-5)

num_epochs = 10
if __name__ == '__main__':
    '''
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        total_loss = 0.0
        with tqdm(train_loader, unit="batch") as tepoch:  # Wrap the train_loader with tqdm for the progress bar
            for batch_images, batch_rhythm_labels, batch_pitch_labels in tepoch:
                # Transfer data to the device (GPU if available)
                batch_images = batch_images.to(device)
                batch_rhythm_labels = batch_rhythm_labels.to(device)
                batch_pitch_labels = batch_pitch_labels.to(device)

                # Forward pass
                rhythm_output, pitch_output = model(batch_images)

                #process_output(rhythm_output, rhythm_mapping)

                # Expand the ground truth labels to match the output size
                downsampled_rhythm_probs = F.interpolate(rhythm_output.permute(0, 2, 1),
                                                       size=batch_rhythm_labels.size(1), mode='linear').permute(0, 2, 1)
                downsampled_pitch_probs = F.interpolate(pitch_output.permute(0, 2, 1),
                                                      size=batch_pitch_labels.size(1), mode='linear').permute(0, 2, 1)
                # Compute the loss
                loss_rhythm = criterion(downsampled_rhythm_probs, batch_rhythm_labels)
                loss_pitch = criterion(downsampled_pitch_probs, batch_pitch_labels)
                loss = loss_rhythm + loss_pitch

                # Backpropagation and parameter update
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Memory management (detach gradients and clear optimizer state)
                loss_rhythm.detach()
                loss_pitch.detach()
                loss.detach()
                optimizer.zero_grad()

                total_loss += loss.item()

                # Update the progress bar
                tepoch.set_postfix(loss=total_loss / len(tepoch))  # Display average loss in the progress bar
        # Print the average loss for the epoch
        average_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {average_loss:.4f}")
        torch.cuda.empty_cache()
    torch.save(model.state_dict(), 'trained_model10epoch.pth')
    # Clear GPU cache at the end of the training
    torch.cuda.empty_cache()
    '''
    # Load the saved model's state dictionary
    model.load_state_dict(torch.load('trained_model10epoch.pth'))
    # Validation Dataset
    thresh_file_valid = "gt_final.valid.thresh"
    valid_dataset = CustomDataset(root_dir, thresh_file_valid)
    # Create data loaders for validation and test sets
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False,
                              num_workers=4,
                              pin_memory=True)
    # Load the saved model's state dictionary (utile quando vogliamo solo fare valid e commentiamo il training)
    #model.load_state_dict(torch.load('trained_model15epoch.pth'))

    model.to(device)
    # Set the model to evaluation mode
    model.eval()

    # Validation loop
    with torch.no_grad():
        total_loss_valid = 0.0
        for batch_images_valid, batch_rhythm_labels_valid, batch_pitch_labels_valid in valid_loader:
            # Transfer data to the device (GPU if available)
            batch_images_valid = batch_images_valid.to(device)
            batch_rhythm_labels_valid = batch_rhythm_labels_valid.to(device)
            batch_pitch_labels_valid = batch_pitch_labels_valid.to(device)

            # Forward pass (no need to compute gradients)
            rhythm_output_valid, pitch_output_valid = model(batch_images_valid)

            # Expand the ground truth labels to match the output size
            downsampled_rhythm_probs = F.interpolate(rhythm_output_valid.permute(0, 2, 1),
                                                     size=batch_rhythm_labels_valid.size(1), mode='linear').permute(0, 2, 1)
            downsampled_pitch_probs = F.interpolate(pitch_output_valid.permute(0, 2, 1),
                                                    size=batch_pitch_labels_valid.size(1), mode='linear').permute(0, 2, 1)

            # Compute the loss for the validation set
            loss_rhythm_valid = criterion(downsampled_rhythm_probs, batch_rhythm_labels_valid)
            loss_pitch_valid = criterion(downsampled_pitch_probs, batch_pitch_labels_valid)
            loss_valid = loss_rhythm_valid + loss_pitch_valid

            total_loss_valid += loss_valid.item()

        # Print the average validation loss
        average_loss_valid = total_loss_valid / len(valid_loader)
        print(f"Validation Loss: {average_loss_valid:.4f}")

    # Test Dataset
    thresh_file_test = "gt_final.test.thresh"
    test_dataset = CustomDataset(root_dir, thresh_file_test)

    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False,
                             num_workers=4,
                             pin_memory=True)
    # Test loop (similar to the validation loop)
    with torch.no_grad():
        total_loss_test = 0.0
        for batch_images_test, batch_rhythm_labels_test, batch_pitch_labels_test in test_loader:
            # Transfer data to the device (GPU if available)
            batch_images_test = batch_images_test.to(device)
            batch_rhythm_labels_test = batch_rhythm_labels_test.to(device)
            batch_pitch_labels_test = batch_pitch_labels_test.to(device)
            # Forward pass (no need to compute gradients)
            rhythm_output_test, pitch_output_test = model(batch_images_test)

            # Expand the ground truth labels to match the output size
            downsampled_rhythm_probs = F.interpolate(rhythm_output_test.permute(0, 2, 1),
                                                     size=batch_rhythm_labels_test.size(1), mode='linear').permute(0, 2, 1)
            downsampled_pitch_probs = F.interpolate(pitch_output_test.permute(0, 2, 1),
                                                    size=batch_pitch_labels_test.size(1), mode='linear').permute(0, 2, 1)
            print(process_output(rhythm_output_test, rhythm_mapping))
            #print(process_output(downsampled_pitch_probs, rhythm_mapping))

            # Compute the loss for the test set
            loss_rhythm_test = criterion(downsampled_rhythm_probs, batch_rhythm_labels_test)
            loss_pitch_test = criterion(downsampled_pitch_probs, batch_pitch_labels_test)
            loss_test = loss_rhythm_test + loss_pitch_test

            total_loss_test += loss_test.item()

        # Print the average test loss
        average_loss_test = total_loss_test / len(test_loader)
        print(f"Test Loss: {average_loss_test:.4f}")
