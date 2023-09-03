# Standard imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from CustomDatasetResnetCombined import CustomDatasetResnetCombined
from OMRModelResNet18Combined import OMRModelResNet18Combined
from mappingCombined import combined_mapping, inverse_mapping


def collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    batch_labels = [item[1] for item in batch]

    batch_labels_encoded = [[inverse_mapping[labels] for labels in row] for row in batch_labels]

    # Calculate the maximum sequence length
    max_sequence_length = max(len(row) for row in batch_labels)

    # Pad sequences with -1 values
    padded_labels = [row + [-1] * (max_sequence_length - len(row)) for row in batch_labels_encoded]

    return images, torch.tensor(padded_labels)


def process_output(notes_probs):
    # 1. Estrazione dei Simboli Massimi
    indices = torch.argmax(notes_probs, dim=1)

    # 2. Mappatura degli Indici ai Simboli
    symbols = [combined_mapping[idx.item()] for idx in indices]
    
    # 3. Fusione dei Simboli Consecutivi Uguali
    def merge_consecutive(symbols):
        merged = [symbols[0]]
        for s in symbols[1:]:
            if s != merged[-1]:
                merged.append(s)
        return merged

    # 4. Fusione dei Simboli Consecutivi Uguali
    combined_symbols = merge_consecutive(symbols)

    # 5. Concatenazione dei Simboli
    sequence = "~".join(combined_symbols)

    return sequence


def cer_wer(decoded_sequence, ground_truth_sequence):
    S = sum(1 for x, y in zip(decoded_sequence, ground_truth_sequence) if x != y)
    D = abs(len(decoded_sequence) - len(ground_truth_sequence))
    I = abs(len(decoded_sequence) - len(ground_truth_sequence))
    N = len(ground_truth_sequence)
    cer = (S + D + I) / N
    return cer


def validate():
    model.eval()
    # Validation loop
    with torch.no_grad():
        total_loss_valid = 0.0
        for batch_images_valid, batch_notes_valid in valid_loader:
            # Transfer data to the device (GPU if available)
            batch_images_valid = batch_images_valid.to(device)
            batch_notes_valid = batch_notes_valid.to(device)

            optimizer.zero_grad()

            # Forward pass (no need to compute gradients)
            notes_output_valid = model(batch_images_valid)

            for i in range(batch_notes_valid.size(0)):

                notes_labels_unpadded = batch_notes_valid[i][batch_notes_valid[i] >= 0]

                batch_labels_recombined = []
                for j in range(len(notes_labels_unpadded)):
                    batch_labels_recombined.append(combined_mapping[batch_notes_valid[i][j].item()])

                loss = cer_wer(process_output(notes_output_valid[i]),
                               "~".join(batch_labels_recombined))

                total_loss_valid += loss
            # Return the average validation loss
        average_loss_valid = total_loss_valid / len(valid_loader.dataset)
        return average_loss_valid


# Create the custom dataset
root_dir = "words"
thresh_file_train = "gt_final.train.thresh"
train_dataset = CustomDatasetResnetCombined(root_dir, thresh_file_train)
# Use DataLoader to load data in parallel and move to GPU
batch_size = 1
train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True, num_workers=4,
                          pin_memory=True)

# Validation Dataset
thresh_file_valid = "gt_final.valid.thresh"
valid_dataset = CustomDatasetResnetCombined(root_dir, thresh_file_valid)
# Create data loaders for validation and test sets
valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False,
                          num_workers=4,
                          pin_memory=True)

# Initialize the model and move it to the GPU
model = OMRModelResNet18Combined(num_combined_classes=99)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load the saved model's state dictionary
model.load_state_dict(torch.load('saveModels/model_checkpoint_epoch_150.pt'))
model.to(device)
# Define the optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
num_epochs = 250
best_val_loss = float('inf')
patience = 6  # Number of epochs with increasing validation loss to tolerate
current_patience = 0
if __name__ == '__main__':
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        total_loss = 0.0
        with tqdm(train_loader, unit="batch") as tepoch:  # Wrap the train_loader with tqdm for the progress bar
            for batch_images, batch_notes_labels in tepoch:
                # Transfer data to the device (GPU if available)
                batch_images = batch_images.to(device)
                batch_notes_labels = batch_notes_labels.to(device)

                optimizer.zero_grad()

                # Forward pass
                log_probs_notes = model(batch_images)

                print(process_output(log_probs_notes[0]))

                # Calculate CTC Loss
                ctc_loss = nn.CTCLoss(blank=98, reduction='mean',
                                      zero_infinity=True)  # Set blank=0 as a special symbol in the sequences
                current_batch_size = batch_notes_labels.size(0)
                pred_size = torch.tensor([batch_notes_labels.size(1)] * current_batch_size)
                # Calculate sequence lengths for each element in the batch
                sequence_lengths = [len([label for label in row if label >= 0]) for row in batch_notes_labels]
                # Convert the sequence lengths list to a tensor
                seq_lengths = torch.tensor(sequence_lengths)
                loss = ctc_loss(log_probs_notes.permute(1, 0, 2),
                                       batch_notes_labels[batch_notes_labels >= 0].view(-1), pred_size, seq_lengths)
                # Backpropagation and parameter update
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                # Update the progress bar
                tepoch.set_postfix(loss=total_loss / len(tepoch))  # Display average loss in the progress bar
        # Print the average loss for the epoch
        average_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {average_loss:.4f}")
        if (epoch+1) % 10 == 0:
            save_path = f"saveModels/model_checkpoint_epoch_{epoch+1}.pt"
            torch.save(model.state_dict(), save_path)
            print(f"Model saved at epoch {epoch+1} - Checkpoint: {save_path}")
            avg_val_loss = validate()
            print(f"Epoch [{epoch + 1}/{num_epochs}] - Train Loss: {average_loss:.4f} - Val Loss: {avg_val_loss:.4f}")
            # Check for early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                current_patience = 0
            else:
                current_patience += 1
                if current_patience >= patience:
                    print("Early stopping triggered.")
                    break
        torch.cuda.empty_cache()
    #torch.save(model.state_dict(), 'saveModels/trained_model005lr.pth')
    # Clear GPU cache at the end of the training
    torch.cuda.empty_cache()
    # Test Dataset
    thresh_file_test = "gt_final.test.thresh"
    test_dataset = CustomDatasetResnetCombined(root_dir, thresh_file_test)

    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False,
                             num_workers=4,
                             pin_memory=True)
    # Test loop (similar to the validation loop)
    with torch.no_grad():
        total_loss_test = 0.0
        for batch_images_test, batch_notes_test in test_loader:
            # Transfer data to the device (GPU if available)
            batch_images_test = batch_images_test.to(device)
            batch_notes_test = batch_notes_test.to(device)

            # Forward pass (no need to compute gradients)
            notes_output_test = model(batch_images_test)

            for i in range(notes_output_test.size(0)):

                notes_labels_unpadded = batch_notes_test[i][batch_notes_test[i] >= 0]

                batch_labels_recombined = []
                for j in range(len(notes_labels_unpadded)):
                    batch_labels_recombined.append(combined_mapping[notes_labels_unpadded[j].item()])

                loss = cer_wer(process_output(notes_output_test[i]),
                               "~".join(batch_labels_recombined))

                total_loss_test += loss
        # Print the average Test loss
        average_loss_test = total_loss_test / len(test_loader.dataset)
        print(f"Test Loss: {average_loss_test:.4f}")
