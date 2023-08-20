# Standard imports
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from CustomDataset import CustomDataset
from CustomDatasetResnet import CustomDatasetResnet
from OMRModel import OMRModel
from OMRModelResNet18 import OMRModelResNet18
from Seq2Seq import Seq2Seq
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

#Preprocess dell'input nel modello originale. Le immagini non hanno la stessa dimensione,  vanno adattate
#alla dimensione massima del batch
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

#Preprocess dell'input nel modello con la Resnet. Le immagini hanno già tutte la stessa dimensione, non vanno adattate
#alla dimensione massima del batch (le immagini di una resnet devono avere dimensione 224x224). Si paddano solo le
#sequenze, con "-1". Nel calcolo della CTC vengono considerati solo i valori >0 nelle sequenze, in modo da non
#"sporcare" l'apprendimento
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

#TODO: probabilmente da togliere, era l'algoritmo che avevo implementato per ottenere la sequenza in uscita a partire dalle matrici di output
def process_output(rhythm_matrix, pitch_matrix):
    # Apply thresholding
    binary_rhythm_matrix = (rhythm_matrix > 0.5).float()
    binary_pitch_matrix = (pitch_matrix > 0.5).float()

    # Create an empty list to store the symbols and pitches for each pixel column
    rhythm_list = []
    pitch_list = []
    symbols_combined_list = []
    for col in range(binary_rhythm_matrix.size(0)):
        # Get the binary vector representing symbols and pitches at the current column
        symbol_rhythm_vector = binary_rhythm_matrix[col, 1:]
        symbol_pitch_vector = binary_pitch_matrix[col, 1:]
        # Get the indices where symbols are present
        symbol_rhythm_indices = (symbol_rhythm_vector == 1).nonzero().squeeze()
        symbol_pitch_indices = (symbol_pitch_vector == 1).nonzero().squeeze()
        # Get the corresponding labels using the index-to-label mapping
        if symbol_rhythm_indices.dim() == 0:
            symbols_rhythm = [rhythm_mapping[symbol_rhythm_indices.item()+1]]
        else:
            symbols_rhythm = [rhythm_mapping[i.item()+1] for i in symbol_rhythm_indices]
        # Get the corresponding labels using the index-to-label mapping
        if symbol_pitch_indices.dim() == 0:
            symbols_pitch = [pitch_mapping[symbol_pitch_indices.item()+1]]
        else:
            symbols_pitch = [pitch_mapping[i.item()+1] for i in symbol_pitch_indices]
        # Join the symbols for each pixel column into a string
        rhythm_list.append(symbols_rhythm)
        pitch_list.append(symbols_pitch)
    for i in range(len(rhythm_list)):
        for j in range(len(rhythm_list[i])):
            if rhythm_list[i][j] != 'epsilon':
                for k in range(len(pitch_list[i])):
                    if pitch_list[i][k] != 'epsilon':
                        symbols_combined_list.append(rhythm_list[i][j] + "." + pitch_list[i][k])
            else:
                symbols_combined_list.append(rhythm_list[i][j])
    symbol_strings = '~'.join(symbols_combined_list)
    # Return the list of symbol strings
    return symbol_strings

#TODO: da rivedere, teoricamente è l'algoritmo che viene utilizzato per decifrare l'output di una rete trainata con CTC
def beam_search(probs, label_map, beam_width):
    T, C = probs.shape  # T: time steps, C: number of classes

    # Apply thresholding
    binary_matrix = (probs > 0.5).float()

    # Initialize the beam with the blank symbol (index 0)
    initial_beam = [{'labels': [0], 'probability': 1.0}]
    current_beam = initial_beam

    # Iterate over time steps
    for t in range(T):
        new_beam = []

        # Expand each hypothesis in the current beam
        for hypothesis in current_beam:
            prev_label = hypothesis['labels'][-1]

            # Iterate over possible labels (including blank)
            for c in range(C):
                # Skip repeating labels
                if c == prev_label:
                    continue

                # Calculate probability for the new hypothesis
                if c != 0 and c != prev_label:
                    new_prob = hypothesis['probability'] * binary_matrix[t, c] * binary_matrix[t, prev_label]

                    #new_prob = hypothesis['probability'] * binary_matrix[t, c]
                #else:
                    #new_prob = hypothesis['probability'] * binary_matrix[t, c] * binary_matrix[t, prev_label]

                    new_labels = hypothesis['labels'] + [c]
                    new_beam.append({'labels': new_labels, 'probability': new_prob})

        # Select the top `beam_width` hypotheses
        current_beam = sorted(new_beam, key=lambda x: x['probability'], reverse=True)[:beam_width]

    # Filter out hypotheses with repeated labels
    final_hypotheses = []
    for hypothesis in current_beam:
        if all(hypothesis['labels'][i] != hypothesis['labels'][i + 1] for i in range(len(hypothesis['labels']) - 1)):
            final_hypotheses.append(hypothesis)

    # Convert label indices to actual labels using label_map
    decoded_sequences = []
    for hypothesis in final_hypotheses:
        decoded_seq = [label_map[label] for label in hypothesis['labels']]
        decoded_sequences.append(decoded_seq)

    return decoded_sequences

#TODO: DA RAFFINARE: prima implementazione del SER da utilizzare in validation

def cer_wer(decoded_sequence, ground_truth_sequence):
    S = sum(1 for x, y in zip(decoded_sequence, ground_truth_sequence) if x != y)
    D = abs(len(decoded_sequence) - len(ground_truth_sequence))
    I = abs(len(decoded_sequence) - len(ground_truth_sequence))
    N = len(ground_truth_sequence)
    cer = (S + D + I) / N
    return cer


# Create the custom dataset
root_dir = "words"
thresh_file_train = "gt_final.train.thresh"
train_dataset = CustomDatasetResnet(root_dir, thresh_file_train)
# Use DataLoader to load data in parallel and move to GPU
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn2, shuffle=True, num_workers=4,
                          pin_memory=True)

#show_images(10, train_loader)

# Initialize the model and move it to the GPU
model = OMRModelResNet18(num_pitch_classes=15, num_rhythm_classes=33)
#model = Seq2Seq(vocab_size=100)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load the saved model's state dictionary
#model.load_state_dict(torch.load('saveModels/trained_model10epoch001lr.pth'))
model.to(device)

# Define the optimizer (e.g., SGD)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
num_epochs = 10
if __name__ == '__main__':
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        total_loss = 0.0
        with tqdm(train_loader, unit="batch") as tepoch:  # Wrap the train_loader with tqdm for the progress bar
            for batch_images, batch_rhythm_labels, batch_pitch_labels in tepoch:
                # Transfer data to the device (GPU if available)
                batch_images = batch_images.to(device)
                batch_rhythm_labels = batch_rhythm_labels.to(device)
                batch_pitch_labels = batch_pitch_labels.to(device)

                optimizer.zero_grad()

                # Forward pass
                rhythm_output, pitch_output = model(batch_images)
                output_seq = model(batch_images, batch_rhythm_labels[:, :-1])

                log_softmax = nn.LogSoftmax(dim=2)

                #print(beam_search(rhythm_output[0], rhythm_mapping, 10))

                #process_output(rhythm_output[0], pitch_output[0])

                # log softmax
                log_probs_rhythm = log_softmax(rhythm_output)
                log_probs_pitch = log_softmax(pitch_output)

                # Calculate CTC Loss
                ctc_loss = nn.CTCLoss(blank=0, reduction='mean',
                                      zero_infinity=True)  # Set blank=0 as a special symbol in the sequences
                current_batch_size = batch_rhythm_labels.size(0)
                pred_size = torch.tensor([log_probs_rhythm.size(1)] * current_batch_size)
                # Calculate sequence lengths for each element in the batch
                sequence_lengths = [len([label for label in row if label >= 0]) for row in batch_rhythm_labels]
                # Convert the sequence lengths list to a tensor
                seq_lengths = torch.tensor(sequence_lengths)
                loss_rhythm = ctc_loss(log_probs_rhythm.permute(1, 0, 2), batch_rhythm_labels[batch_rhythm_labels >= 0].view(-1), pred_size, seq_lengths)
                loss_pitch = ctc_loss(log_probs_pitch.permute(1, 0, 2), batch_pitch_labels[batch_pitch_labels >= 0].view(-1), pred_size, seq_lengths)
                loss = loss_rhythm + loss_pitch

                # Backpropagation and parameter update
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                # Update the progress bar
                tepoch.set_postfix(loss=total_loss / len(tepoch))  # Display average loss in the progress bar
        # Print the average loss for the epoch
        average_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {average_loss:.4f}")
        if epoch % 10 == 0:
            save_path = f"saveModels/model_checkpoint_epoch_{epoch}.pt"
            torch.save(model.state_dict(), save_path)
            print(f"Model saved at epoch {epoch} - Checkpoint: {save_path}")
        torch.cuda.empty_cache()
    torch.save(model.state_dict(), 'saveModels/trained_model10epoch001lr.pth')
    # Clear GPU cache at the end of the training
    torch.cuda.empty_cache()
    # Load the saved model's state dictionary
    model.load_state_dict(torch.load('saveModels/trained_model10epoch.pth'))
    # Validation Dataset
    thresh_file_valid = "gt_final.valid.thresh"
    valid_dataset = CustomDatasetResnet(root_dir, thresh_file_valid)
    # Create data loaders for validation and test sets
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, collate_fn=collate_fn2, shuffle=False,
                              num_workers=4,
                              pin_memory=True)
    # Load the saved model's state dictionary (utile quando vogliamo solo fare valid e commentiamo il training)
    # model.load_state_dict(torch.load('trained_model15epoch.pth'))

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

            for i in range(batch_rhythm_labels_valid.size(0)):

                rhythm_labels_unpadded = batch_rhythm_labels_valid[i][batch_rhythm_labels_valid[i] >= 0]
                pitch_labels_unpadded = batch_pitch_labels_valid[i][batch_pitch_labels_valid[i] >= 0]

                #batch_labels_recombined = [rhythm_mapping[s1.item()] + "." + pitch_mapping[s2.item()] if s1.item() != 0 else 'epsilon'
                 #                          for s1, s2 in zip(rhythm_labels_unpadded, batch_pitch_labels_valid)]
                batch_labels_recombined = []
                for j in range(len(rhythm_labels_unpadded)):
                    if rhythm_labels_unpadded[j].item() != 0:
                        batch_labels_recombined.append(rhythm_mapping[rhythm_labels_unpadded[j].item()]+ "." + pitch_mapping[pitch_labels_unpadded[j].item()])
                    else:
                        batch_labels_recombined.append(rhythm_mapping[rhythm_labels_unpadded[j].item()])

                loss = cer_wer(process_output(rhythm_output_valid[i], pitch_output_valid[i]), batch_labels_recombined)

                total_loss_valid += loss
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

            # log softmax
            log_probs_rhythm = log_softmax(rhythm_output_test)
            log_probs_pitch = log_softmax(pitch_output_test)

            del rhythm_output_test, pitch_output_test
            # Calculate CTC Loss
            ctc_loss = nn.CTCLoss(blank=0)  # Set blank=0 as a special symbol in the sequences
            current_batch_size = batch_rhythm_labels_test.size(0)
            pred_size = torch.tensor([log_probs_rhythm.size(1)] * current_batch_size)
            seq_length = torch.tensor([batch_rhythm_labels_test.size(1)] * current_batch_size)
            loss_rhythm = ctc_loss(log_probs_rhythm.permute(1, 0, 2), batch_rhythm_labels_test, pred_size,
                                   seq_length) / current_batch_size
            loss_pitch = ctc_loss(log_probs_pitch.permute(1, 0, 2), batch_pitch_labels_test, pred_size,
                                  seq_length) / current_batch_size
            loss = loss_rhythm + loss_pitch

            total_loss_test += loss.item()

        # Print the average test loss
        average_loss_test = total_loss_test / len(test_loader)
        print(f"Test Loss: {average_loss_test:.4f}")
