import torch
from torch.utils.data import DataLoader

from CustomDatasetSeq2Seq import CustomDatasetSeq2Seq
from Seq2Seq import Seq2Seq
from mappingCombined import combined_mapping, inverse_mapping


def process_output(output_seq):
    # 1. Estrazione dei Simboli Massimi
    indices = torch.argmax(output_seq, dim=1)

    # 2. Mappatura degli Indici ai Simboli
    symbols = [combined_mapping[idx.item()] for idx in indices]

    # 3. Fusione dei Simboli Consecutivi Uguali
    def merge_consecutive(symbols):
        merged = [symbols[0]]
        for s in symbols[1:]:
            if s != merged[-1]:
                merged.append(s)
        return merged

    combined_symbols = merge_consecutive(symbols)

    # 5. Concatenazione dei Simboli
    sequence = "~".join(combined_symbols)

    return sequence
def collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    batch_labels = [item[1] for item in batch]

    batch_labels_encoded = [[inverse_mapping[labels] for labels in row] for row in batch_labels]

    # Calculate the maximum sequence length
    max_sequence_length = max(len(row) for row in batch_labels)

    # Pad sequences with -1 values
    padded_labels = [row + [0] * (max_sequence_length - len(row)) for row in batch_labels_encoded]

    return images, torch.tensor(padded_labels)

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
        for batch_images, batch_labels in valid_loader:
            # Transfer data to the device (GPU if available)
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            output_seq = model(batch_images, batch_labels)
            for i in range(batch_labels.size(0)):
                batch_labels_remapped = []
                for j in range(len(batch_labels[i])):
                    if j == 0:
                        batch_labels_remapped.append(combined_mapping[batch_labels[i][j].item()])
                    else:
                        if batch_labels_remapped[len(batch_labels_remapped) - 1] != combined_mapping[batch_labels[i][j].item()]:
                            batch_labels_remapped.append(combined_mapping[batch_labels[i][j].item()])
                loss = cer_wer(process_output(output_seq[i]), "~".join(batch_labels_remapped))
                total_loss_valid += loss
        # Return the average validation loss
        average_loss_valid = total_loss_valid / len(valid_loader.dataset)
        return average_loss_valid

root_dir = "words"
thresh_file_train = "gt_final.train.thresh"
train_dataset = CustomDatasetSeq2Seq(root_dir, thresh_file_train)
# Use DataLoader to load data in parallel and move to GPU
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True, num_workers=4,
                          pin_memory=True)
thresh_file_valid = "gt_final.valid.thresh"
valid_dataset = CustomDatasetSeq2Seq(root_dir, thresh_file_valid)
# Create data loaders for validation and test sets
valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True,
                          num_workers=4,
                          pin_memory=True)
vocab_size = 98
model = Seq2Seq(vocab_size)
model.load_state_dict(torch.load('saveModels/seq2seq/model_checkpoint_epoch_70.pt'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=1e-4)
num_epochs = 250
best_val_loss = float('inf')
patience = 5  # Number of epochs with increasing validation loss to tolerate
current_patience = 0

if __name__ == '__main__':
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        total_loss = 0.0
        with tqdm(train_loader, unit="batch") as tepoch:  # Wrap the train_loader with tqdm for the progress bar
            for batch_images, batch_labels in tepoch:
                # Transfer data to the device (GPU if available)
                batch_images = batch_images.to(device)
                batch_labels = batch_labels.to(device)

                optimizer.zero_grad()

                # Forward pass
                output_seq = model(batch_images, batch_labels)
                #print(process_output(output_seq[0]))
                # Compute loss
                loss = F.cross_entropy(output_seq.view(-1, vocab_size),
                                       batch_labels.flatten())  # Exclude the first token in target

                # Backpropagation and optimization
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                # Update the progress bar
                tepoch.set_postfix(loss=total_loss / len(tepoch))  # Display average loss in the progress bar
        # Print the average loss for the epoch
        average_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {average_loss:.4f}")
        if (epoch + 1) % 5 == 0:
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
    # Test Dataset
    thresh_file_test = "gt_final.test.thresh"
    test_dataset = CustomDatasetSeq2Seq(root_dir, thresh_file_test)

    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True,
                                 num_workers=4,
                                 pin_memory=True)
    # Test loop
    with torch.no_grad():
        total_loss_test = 0.0
        for batch_images, batch_labels in test_loader:
            # Transfer data to the device (GPU if available)
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            output_seq = model(batch_images, batch_labels)
            for i in range(batch_labels.size(0)):
                batch_labels_remapped = []
                for j in range(len(batch_labels[i])):
                    if j == 0:
                        batch_labels_remapped.append(combined_mapping[batch_labels[i][j].item()])
                    else:
                        if batch_labels_remapped[len(batch_labels_remapped) - 1] != combined_mapping[batch_labels[i][j].item()]:
                            batch_labels_remapped.append(combined_mapping[batch_labels[i][j].item()])
                print(process_output(output_seq[i]))
                loss = cer_wer(process_output(output_seq[i]), "~".join(batch_labels_remapped))
                total_loss_test += loss
        # Print the average Test loss
        average_loss_test = total_loss_test / len(test_loader.dataset)
        print(f"Test SER: {average_loss_test:.4f}")


