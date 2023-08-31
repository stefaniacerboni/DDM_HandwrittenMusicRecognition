import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from CustomDatasetSeq2Seq import CustomDatasetSeq2Seq
from Seq2Seq import Seq2Seq
from combined_mapping import combined_mapping, inverse_mapping


def collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    batch_labels = [item[1] for item in batch]

    batch_labels_encoded = [[inverse_mapping[labels] for labels in row] for row in batch_labels]

    # Calculate the maximum sequence length
    max_sequence_length = max(len(row) for row in batch_labels)

    # Pad sequences with -1 values
    padded_labels = [row + [-1] * (max_sequence_length - len(row)) for row in batch_labels_encoded]

    return images, torch.tensor(padded_labels)



root_dir = "words"
thresh_file_train = "gt_final.train.thresh"
train_dataset = CustomDatasetSeq2Seq(root_dir, thresh_file_train)
# Use DataLoader to load data in parallel and move to GPU
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True, num_workers=4,
                          pin_memory=True)
model = Seq2Seq(vocab_size=98)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
num_epochs = 250
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

                # Update the progress bar
                tepoch.set_postfix(loss=total_loss / len(tepoch))  # Display average loss in the progress bar
        # Print the average loss for the epoch
        average_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {average_loss:.4f}")
        if epoch % 10 == 0 and epoch != 0:
            save_path = f"saveModels/model_checkpoint_epoch_{epoch}.pt"
            torch.save(model.state_dict(), save_path)
            print(f"Model saved at epoch {epoch} - Checkpoint: {save_path}")
        torch.cuda.empty_cache()