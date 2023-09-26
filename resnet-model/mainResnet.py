# Standard imports
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from torch.utils.data import DataLoader, random_split, ConcatDataset
from tqdm import tqdm

from CustomDatasetResnet import CustomDatasetResnet
from OMRModelResNet18 import OMRModelResNet18
#from mapping import rhythm_mapping, pitch_mapping, inverse_rhythm_mapping, inverse_pitch_mapping
from new_mapping import rhythm_mapping, pitch_mapping, inverse_rhythm_mapping, inverse_pitch_mapping

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


# Preprocess dell'input nel modello originale. Le immagini non hanno la stessa dimensione,  vanno adattate
# alla dimensione massima del batch
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

    # Calculate the maximum sequence length
    max_sequence_length = max(len(row) for row in batch_rhythm_labels_encoded)

    # Pad sequences with -1 values
    padded_rhythm_labels = [row + [-1] * (max_sequence_length - len(row)) for row in batch_rhythm_labels_encoded]
    padded_pitch_labels = [row + [-1] * (max_sequence_length - len(row)) for row in batch_pitch_labels_encoded]

    return images, torch.tensor(padded_rhythm_labels), torch.tensor(padded_pitch_labels)


# Preprocess dell'input nel modello con la Resnet. Le immagini hanno già tutte la stessa dimensione, non vanno adattate
# alla dimensione massima del batch (le immagini di una resnet devono avere dimensione 224x224). Si paddano solo le
# sequenze, con "-1". Nel calcolo della CTC vengono considerati solo i valori >0 nelle sequenze, in modo da non
# "sporcare" l'apprendimento
'''
def collate_fn2(batch):
    images = torch.stack([item[0] for item in batch])
    rhythm_labels_list = [item[1] for item in batch]
    pitch_labels_list = [item[2] for item in batch]

    # Encode string labels to numerical values using mapping
    batch_rhythm_labels_encoded = [[inverse_rhythm_mapping[labels] for labels in row] for row in rhythm_labels_list]
    batch_pitch_labels_encoded = [[inverse_pitch_mapping[labels] for labels in row] for row in pitch_labels_list]

    # Calculate the maximum sequence length
    max_sequence_length = max(len(row) for row in batch_rhythm_labels_encoded)

    # Pad sequences with -1 values
    padded_rhythm_labels = [row + [-1] * (max_sequence_length - len(row)) for row in batch_rhythm_labels_encoded]
    padded_pitch_labels = [row + [-1] * (max_sequence_length - len(row)) for row in batch_pitch_labels_encoded]

    return images, torch.tensor(padded_rhythm_labels), torch.tensor(padded_pitch_labels)
'''

def process_output(rhythm_probs, pitch_probs):
    # 1. Estrazione dei Simboli Massimi
    rhythm_indices = torch.argmax(rhythm_probs, dim=1)
    pitch_indices = torch.argmax(pitch_probs, dim=1)

    # 2. Mappatura degli Indici ai Simboli
    rhythm_symbols = [rhythm_mapping[idx.item()] for idx in rhythm_indices]
    pitch_symbols = [pitch_mapping[idx.item()] for idx in pitch_indices]

    # 3. Combinazione di Ritmo e Intonazione
    combined_symbols = []
    for r, p in zip(rhythm_symbols, pitch_symbols):
        if r != "blank" and p != "blank":
            if r == "epsilon":
                combined_symbols.append(f"{r}")
            else:
                combined_symbols.append(f"{r}.{p}")

    # 4. Fusione dei Simboli Consecutivi Uguali
    def merge_consecutive(symbols):
        merged = [symbols[0]]
        for s in symbols[1:]:
            if s != merged[-1]:
                merged.append(s)
        return merged

    if len(combined_symbols) > 0:
        combined_symbols = merge_consecutive(combined_symbols)

    # 5. Concatenazione dei Simboli
    sequence = "~".join(combined_symbols)

    return sequence


def validate():
    model.eval()
    # Validation loop
    with torch.no_grad():
        total_loss_valid = 0.0
        for batch_images_valid, batch_rhythm_labels_valid, batch_pitch_labels_valid in current_valid_data_loader:
            # Transfer data to the device (GPU if available)
            batch_images_valid = batch_images_valid.to(device)
            batch_rhythm_labels_valid = batch_rhythm_labels_valid.to(device)
            batch_pitch_labels_valid = batch_pitch_labels_valid.to(device)

            # Forward pass (no need to compute gradients)
            rhythm_output_valid, pitch_output_valid = model(batch_images_valid)

            for i in range(batch_rhythm_labels_valid.size(0)):

                rhythm_labels_unpadded = batch_rhythm_labels_valid[i][batch_rhythm_labels_valid[i] >= 0]
                pitch_labels_unpadded = batch_pitch_labels_valid[i][batch_pitch_labels_valid[i] >= 0]

                batch_labels_recombined = []
                for j in range(len(rhythm_labels_unpadded)):
                    if rhythm_labels_unpadded[j].item() != 1:
                        batch_labels_recombined.append(
                            rhythm_mapping[rhythm_labels_unpadded[j].item()] + "." + pitch_mapping[
                                pitch_labels_unpadded[j].item()])
                    else:
                        batch_labels_recombined.append(rhythm_mapping[rhythm_labels_unpadded[j].item()])

                loss = cer_wer(process_output(rhythm_output_valid[i], pitch_output_valid[i]),
                               "~".join(batch_labels_recombined))

                total_loss_valid += loss
        # Return the average validation loss
        average_loss_valid = total_loss_valid / len(current_valid_data_loader.dataset)
        return average_loss_valid

'''
# Create the custom dataset
root_dir = "historical-dataset/words"
#thresh_file_train = "gt_final.train.thresh"
thresh_file_train = "lilypond-dataset/lilypond.train.thresh"
train_dataset = CustomDatasetResnet(root_dir, thresh_file_train)
# Use DataLoader to load data in parallel and move to GPU
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn2, shuffle=True, num_workers=4,
                          pin_memory=True)

# Validation Dataset
#thresh_file_valid = "gt_final.valid.thresh"
thresh_file_valid = "lilypond-dataset/lilypond.valid.thresh"
valid_dataset = CustomDatasetResnet(root_dir, thresh_file_valid)
# Create data loaders for validation and test sets
valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, collate_fn=collate_fn2, shuffle=False,
                          num_workers=4,
                          pin_memory=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn2, shuffle=True, num_workers=4,
                          pin_memory=True)
'''
historical_root_dir = "../historical-dataset/words"
thresh_file_historical_train = "../historical-dataset/newdef_gt_final.train.thresh"
historical_dataset_train = CustomDatasetResnet(historical_root_dir, thresh_file_historical_train)
synthetic_root_dir = "../lilypond-dataset/words"
thresh_file_synthetic_train = "../lilypond-dataset/lilypond.train.thresh"
synthetic_dataset_train = CustomDatasetResnet(synthetic_root_dir, thresh_file_synthetic_train)

thresh_file_historical_valid = "../historical-dataset/newdef_gt_final.valid.thresh"
historical_dataset_valid = CustomDatasetResnet(historical_root_dir, thresh_file_historical_valid)
thresh_file_synthetic_valid = "../lilypond-dataset/lilypond.valid.thresh"
synthetic_dataset_valid = CustomDatasetResnet(synthetic_root_dir, thresh_file_synthetic_valid)
# Use DataLoader to load data in parallel and move to GPU
batch_size = 16


# Initialize the model and move it to the GPU
#model = OMRModelResNet18(num_rhythm_classes=33, num_pitch_classes=15)
model = OMRModelResNet18(num_rhythm_classes=63, num_pitch_classes=17)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load the saved model's state dictionary
#model.load_state_dict(torch.load('saveModels/lilypond/model_checkpoint_epoch_100.pt'))
model.to(device)
# Define the optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
num_epochs = 100
best_val_loss = float('inf')
patience = 1  # Number of epochs with increasing validation loss to tolerate
current_patience = 0

# Calculate the dataset sizes based on proportions
total_samples_train = len(historical_dataset_train)
initial_historical_size_train = int(0.1 * total_samples_train)

# Create data loaders for the initial proportions
initial_synthetic_data_train, _ = random_split(
    synthetic_dataset_train,
    [len(historical_dataset_train) - initial_historical_size_train, len(synthetic_dataset_train) - (len(historical_dataset_train) - initial_historical_size_train)]
)
initial_historical_data_train, _ = random_split(
    historical_dataset_train,
    [initial_historical_size_train, len(historical_dataset_train) - initial_historical_size_train ]
)

current_train_data_loader = DataLoader(ConcatDataset([initial_synthetic_data_train, initial_historical_data_train]), batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
current_valid_data_loader = DataLoader(synthetic_dataset_valid, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)

if __name__ == '__main__':
    for epoch in range(num_epochs):
        if epoch + 1 % 10 == 0:
            current_proportion = 0.1 + (epoch // 10) * 0.1
            current_historical_size = int(current_proportion * total_samples_train)

            # Create data loaders for the initial proportions
            current_synthetic_data_train, _ = random_split(
                synthetic_dataset_train,
                [len(historical_dataset_train) - current_historical_size,
                 len(synthetic_dataset_train) - (len(historical_dataset_train) - current_historical_size)]
            )
            current_historical_data_train, _ = random_split(
                historical_dataset_train,
                [current_historical_size, len(historical_dataset_train) - current_historical_size]
            )

            # Create a new data loader with the current proportions
            current_train_data_loader = DataLoader(ConcatDataset([current_synthetic_data_train, current_historical_data_train]), batch_size=batch_size, collate_fn=collate_fn, shuffle=True)

            current_proportion_valid = 1.0 - (epoch // 10) * 0.1
            current_historical_size_valid = int(current_proportion * len(historical_dataset_valid))

            # Create data loaders for the initial proportions
            current_synthetic_data_valid, _ = random_split(
                synthetic_dataset_valid,
                [len(historical_dataset_valid) - current_historical_size_valid,
                 len(synthetic_dataset_valid) - (len(historical_dataset_valid) - current_historical_size_valid)]
            )
            current_historical_data_valid, _ = random_split(
                historical_dataset_valid,
                [current_historical_size_valid, len(historical_dataset_valid) - current_historical_size_valid]
            )
            # Create a new data loader with the current proportions
            current_valid_data_loader = DataLoader(
                ConcatDataset([current_synthetic_data_valid, current_historical_data_valid]), batch_size=batch_size, collate_fn=collate_fn,
                shuffle=True)

        # Training loop
        model.train()  # Set the model to training mode
        total_loss = 0.0
        with tqdm(current_train_data_loader, unit="batch") as tepoch:  # Wrap the train_loader with tqdm for the progress bar
            for batch_images, batch_rhythm_labels, batch_pitch_labels in tepoch:
                # Transfer data to the device (GPU if available)
                batch_images = batch_images.to(device)
                batch_rhythm_labels = batch_rhythm_labels.to(device)
                batch_pitch_labels = batch_pitch_labels.to(device)

                optimizer.zero_grad()

                # Forward pass
                log_probs_rhythm, log_probs_pitch = model(batch_images)

                #print(process_output(log_probs_rhythm[0], log_probs_pitch[0]))

                # Calculate CTC Loss
                ctc_loss = nn.CTCLoss(blank=0, reduction='mean',
                                      zero_infinity=True)  # Set blank=0 as a special symbol in the sequences
                current_batch_size = batch_rhythm_labels.size(0)
                pred_size = torch.tensor([log_probs_rhythm.size(1)] * current_batch_size)
                # Calculate sequence lengths for each element in the batch
                sequence_lengths = [len([label for label in row if label >= 0]) for row in batch_rhythm_labels]
                # Convert the sequence lengths list to a tensor
                seq_lengths = torch.tensor(sequence_lengths)
                loss_rhythm = ctc_loss(log_probs_rhythm.permute(1, 0, 2),
                                       batch_rhythm_labels[batch_rhythm_labels >= 0].view(-1), pred_size, seq_lengths)
                loss_pitch = ctc_loss(log_probs_pitch.permute(1, 0, 2),
                                      batch_pitch_labels[batch_pitch_labels >= 0].view(-1), pred_size, seq_lengths)
                loss = loss_rhythm + loss_pitch

                # Backpropagation and parameter update
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                # Update the progress bar
                tepoch.set_postfix(loss=total_loss / len(tepoch))  # Display average loss in the progress bar
        # Print the average loss for the epoch
        average_loss = total_loss / len(current_train_data_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {average_loss:.4f}")
        if (epoch+1) % 10 == 0:
            save_path = f"saveModels/combinedTraining/model_checkpoint_epoch_{epoch+1}.pt"
            torch.save(model.state_dict(), save_path)
            print(f"Model saved at epoch {epoch+1} - Checkpoint: {save_path}")
            avg_val_loss = validate()
            print(f"Epoch [{epoch + 1}/{num_epochs}] - Train Loss: {average_loss:.4f} - Val Loss: {avg_val_loss:.4f}")
            '''
            # Check for early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                current_patience = 0
            else:
                current_patience += 1
                if current_patience >= patience:
                    print("Early stopping triggered.")
                    break
                    '''
        torch.cuda.empty_cache()
    #torch.save(model.state_dict(), 'saveModels/trained_model005lr.pth')
    # Clear GPU cache at the end of the training
    torch.cuda.empty_cache()
    # Test Dataset
    thresh_file_test = "../historical-dataset/newdef_gt_final.test.thresh"
    test_dataset = CustomDatasetResnet(historical_root_dir, thresh_file_test)

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

            for i in range(batch_rhythm_labels_test.size(0)):


                rhythm_labels_unpadded = batch_rhythm_labels_test[i][batch_rhythm_labels_test[i] >= 0]
                pitch_labels_unpadded = batch_pitch_labels_test[i][batch_pitch_labels_test[i] >= 0]

                batch_labels_recombined = []
                for j in range(len(rhythm_labels_unpadded)):
                    if rhythm_labels_unpadded[j].item() != 1:
                        batch_labels_recombined.append(
                            rhythm_mapping[rhythm_labels_unpadded[j].item()] + "." + pitch_mapping[
                                pitch_labels_unpadded[j].item()])
                    else:
                        batch_labels_recombined.append(rhythm_mapping[rhythm_labels_unpadded[j].item()])
                loss = cer_wer(process_output(rhythm_output_test[i], pitch_output_test[i]), "~".join(batch_labels_recombined))

                total_loss_test += loss
        # Print the average Test loss
        average_loss_test = total_loss_test / len(test_loader.dataset)
        print(f"Test SER: {average_loss_test:.4f}")
