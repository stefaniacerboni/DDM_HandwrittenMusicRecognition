import torch
import torch.nn as nn
import torch.nn.functional as F


class OMRModel(nn.Module):
    def __init__(self, input_channels, num_rhythm_classes, num_pitch_classes):
        super(OMRModel, self).__init__()

        # Convolutional Block
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 1))

        # Recurrent Block with BLSTM layers
        self.recurrent_block = nn.LSTM(input_size=64, hidden_size=128, num_layers=4, batch_first=True,
                                       bidirectional=True)

        # Dense Layers for rhythm and pitch
        self.rhythm_output = nn.Linear(2 * 128, num_rhythm_classes)
        self.pitch_output = nn.Linear(2 * 128, num_pitch_classes)

    def forward(self, x):
        # Pass the input through the Convolutional Block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        # Reshape the output to (batch_size, sequence_length (widthxheight), num_features)
        batch_size, num_features, height, width = x.size()
        # Assuming x is the tensor after the Convolutional Block with shape (batch_size, num_features, height, width)
        # Reshape the tensor to (batch_size, width * height, num_features)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(batch_size, -1, num_features)

        # Pass the output through the Recurrent Block
        x, _ = self.recurrent_block(x)

        x = x.view(batch_size, width, height, 256)

        # max-pooling along the height = 25 to get 1xwidthxnum_features
        x = torch.max(x, dim=2)[0]

        # Forward pass
        rhythm_logits = self.rhythm_output(x)
        pitch_logits = self.pitch_output(x)

        # Apply sigmoid activation to get probabilities
        rhythm_probs = torch.sigmoid(rhythm_logits)
        pitch_probs = torch.sigmoid(pitch_logits)

        # Return the probability matrices for rhythm and pitch predictions
        return rhythm_probs, pitch_probs
