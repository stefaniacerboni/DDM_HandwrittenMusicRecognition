import torch
import torch.nn.functional as F
from utils import *
from torch.utils.data import DataLoader, random_split, ConcatDataset
from tqdm import tqdm

from CustomDatasetSeq2Seq import CustomDatasetSeq2Seq
from Seq2Seq import Seq2Seq
from new_mapping_combined import new_mapping_combined, inverse_new_mapping_combined


def process_output(output_seq):
    # 1. Estrazione dei Simboli Massimi
    indices = torch.argmax(output_seq, dim=1)

    # 2. Mappatura degli Indici ai Simboli
    symbols = [new_mapping_combined[idx.item()] for idx in indices]

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

    batch_labels_encoded = [[inverse_new_mapping_combined[labels] for labels in row] for row in batch_labels]

    # Calculate the maximum sequence length
    max_sequence_length = max(len(row) for row in batch_labels)

    # Pad sequences with -1 values
    padded_labels = [row + [0] * (max_sequence_length - len(row)) for row in batch_labels_encoded]

    return images, torch.tensor(padded_labels)

def validate():
    model.eval()

    # Validation loop
    with torch.no_grad():
        total_loss_valid = 0.0
        for batch_images, batch_labels in current_valid_data_loader:
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
                        batch_labels_remapped.append(new_mapping_combined[batch_labels[i][j].item()])
                    else:
                        if batch_labels_remapped[len(batch_labels_remapped) - 1] != new_mapping_combined[batch_labels[i][j].item()]:
                            batch_labels_remapped.append(new_mapping_combined[batch_labels[i][j].item()])
                loss = cer_wer(process_output(output_seq[i]), "~".join(batch_labels_remapped))
                total_loss_valid += loss
        # Return the average validation loss
        average_loss_valid = total_loss_valid / len(current_valid_data_loader.dataset)
        return average_loss_valid

def test():
    # Test Dataset
    thresh_file_test = "../historical-dataset/newdef_gt_final.test.thresh"
    test_dataset = CustomDatasetSeq2Seq(historical_root_dir, thresh_file_test)

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
                        batch_labels_remapped.append(new_mapping_combined[batch_labels[i][j].item()])
                    else:
                        if batch_labels_remapped[len(batch_labels_remapped) - 1] != new_mapping_combined[
                            batch_labels[i][j].item()]:
                            batch_labels_remapped.append(new_mapping_combined[batch_labels[i][j].item()])
                #print(process_output(output_seq[i]))
                loss = cer_wer(process_output(output_seq[i]), "~".join(batch_labels_remapped))
                total_loss_test += loss
        # Print the average Test loss
        average_loss_test = total_loss_test / len(test_loader.dataset)
        return average_loss_test
'''
root_dir = "historical-dataset/words"
thresh_file_train = "lilypond-dataset/newdef_gt_final.train.thresh"
train_dataset = CustomDatasetSeq2Seq(root_dir, thresh_file_train)
# Use DataLoader to load data in parallel and move to GPU

train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True, num_workers=4,
                          pin_memory=True)
thresh_file_valid = "gt_final.valid.thresh"
valid_dataset = CustomDatasetSeq2Seq(root_dir, thresh_file_valid)
# Create data loaders for validation and test sets
valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True,
                          num_workers=4,
                          pin_memory=True)
#vocab_size = 98
'''
historical_root_dir = "../historical-dataset/words"
thresh_file_historical_train = "../historical-dataset/newdef_gt_final.train.thresh"
historical_dataset_train = CustomDatasetSeq2Seq(historical_root_dir, thresh_file_historical_train)
synthetic_root_dir = "../lilypond-dataset/words"
thresh_file_synthetic_train = "../lilypond-dataset/lilypond.train.thresh"
synthetic_dataset_train = CustomDatasetSeq2Seq(synthetic_root_dir, thresh_file_synthetic_train)

thresh_file_historical_valid = "../historical-dataset/newdef_gt_final.valid.thresh"
historical_dataset_valid = CustomDatasetSeq2Seq(historical_root_dir, thresh_file_historical_valid)
thresh_file_synthetic_valid = "../lilypond-dataset/lilypond.valid.thresh"
synthetic_dataset_valid = CustomDatasetSeq2Seq(synthetic_root_dir, thresh_file_synthetic_valid)

vocab_size = 690
model = Seq2Seq(vocab_size)
#model.load_state_dict(torch.load('saveModels/seq2seq/model_checkpoint_epoch_70.pt'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4)
num_epochs = 100
best_val_ser = float('inf')
patience = 2  # Number of epochs with increasing validation loss to tolerate
current_patience = 0
batch_size = 16
#REGULAR TRAINING
'''
root_dir = "historical-dataset/words"
thresh_file_train = "lilypond-dataset/newdef_gt_final.train.thresh"
train_dataset = CustomDatasetSeq2Seq(root_dir, thresh_file_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True, num_workers=4,
                          pin_memory=True)
thresh_file_valid = "gt_final.valid.thresh"
valid_dataset = CustomDatasetSeq2Seq(root_dir, thresh_file_valid)
# Create data loaders for validation and test sets
valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True,
                          num_workers=4,
                          pin_memory=True)
'''
curriculum_training = True
current_train_data_loader = get_curriculum_learning_loader(synthetic_dataset_train, historical_dataset_train,
                                                           collate_fn, batch_size, 0.1)
current_valid_data_loader = get_curriculum_learning_loader(synthetic_dataset_valid, historical_dataset_valid,
                                                           collate_fn, batch_size, 1.0)

if __name__ == '__main__':
    for epoch in range(num_epochs):
        if epoch + 1 % 10 == 0 and curriculum_training is True:
            current_proportion_train = 0.1 + (epoch // 10) * 0.1
            current_train_data_loader = get_curriculum_learning_loader(synthetic_dataset_train,
                                                                       historical_dataset_train,
                                                                       collate_fn, batch_size, current_proportion_train)
            current_proportion_valid = 1.0 - (epoch // 10) * 0.1
            current_valid_data_loader = get_curriculum_learning_loader(synthetic_dataset_valid,
                                                                       historical_dataset_valid,
                                                                       collate_fn, batch_size, current_proportion_valid)

        model.train()  # Set the model to training mode
        total_loss = 0.0
        with tqdm(current_train_data_loader, unit="batch") as tepoch:  # Wrap the train_loader with tqdm for the progress bar
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
        average_loss = total_loss / len(current_train_data_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {average_loss:.4f}")
        if (epoch + 1) % 10 == 0:
            save_path = f"saveModels/model_checkpoint_epoch_{epoch+1}.pt"
            torch.save(model.state_dict(), save_path)
            print(f"Model saved at epoch {epoch+1} - Checkpoint: {save_path}")
            avg_val_ser = validate()
            avg_test_ser = test()
            print(f"Epoch [{epoch + 1}/{num_epochs}] - Train Loss: {average_loss:.4f} - Val SER: {avg_val_ser:.4f} - "
                  f"Test SER: {avg_test_ser:.4f}")
            # Check for early stopping
            if avg_val_ser < best_val_ser:
                best_val_ser = avg_val_ser
                current_patience = 0
            else:
                current_patience += 1
                if current_patience >= patience:
                    print("Early stopping triggered.")
                    break
        torch.cuda.empty_cache()
    '''
    # Test Dataset
    thresh_file_test = "lilypond-dataset/newdef_gt_final.test.thresh"
    test_dataset = CustomDatasetSeq2Seq(historical_root_dir, thresh_file_test)

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
                        batch_labels_remapped.append(new_mapping_combined[batch_labels[i][j].item()])
                    else:
                        if batch_labels_remapped[len(batch_labels_remapped) - 1] != new_mapping_combined[batch_labels[i][j].item()]:
                            batch_labels_remapped.append(new_mapping_combined[batch_labels[i][j].item()])
                print(process_output(output_seq[i]))
                loss = cer_wer(process_output(output_seq[i]), "~".join(batch_labels_remapped))
                total_loss_test += loss
        # Print the average Test loss
        average_loss_test = total_loss_test / len(test_loader.dataset)
        print(f"Test SER: {average_loss_test:.4f}")
        '''


