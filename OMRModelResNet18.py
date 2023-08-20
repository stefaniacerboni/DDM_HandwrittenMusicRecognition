import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights


class OMRModelResNet18(nn.Module):
    def __init__(self, num_rhythm_classes, num_pitch_classes):
        super(OMRModelResNet18, self).__init__()

        # Convolutional Block
        resnet18 = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.conv1 = resnet18.conv1
        self.bn1 = resnet18.bn1
        self.relu = resnet18.relu

        # Recurrent Block with BLSTM layers
        self.recurrent_block = nn.LSTM(input_size=64 * 112, hidden_size=512, num_layers=4, batch_first=True,
                                       bidirectional=True)

        # Dense Layers for rhythm and pitch
        self.rhythm_output = nn.Linear(2 * 512, num_rhythm_classes)
        self.pitch_output = nn.Linear(2 * 512, num_pitch_classes)

    def forward(self, x):
        # Pass the input through the Convolutional Block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # Reshape the output to (batch_size, width (sequence_length), height * num_features)
        batch_size, num_features, height, width = x.size()
        # Reshape the tensor to (batch_size, width, height, num_features)
        x = x.permute(0, 3, 2, 1).contiguous()
        # Reshape the tensor to (batch_size, width, height=112 * num_features=64)
        x = x.view(batch_size, width, height * num_features)

        # Pass the output through the Recurrent Block
        x, _ = self.recurrent_block(x)

        # Forward pass
        rhythm_logits = self.rhythm_output(x)
        pitch_logits = self.pitch_output(x)

        rhythm_probs = torch.softmax(rhythm_logits, dim=2)
        pitch_probs = torch.softmax(pitch_logits, dim=2)
        # Return the probability matrices for rhythm and pitch predictions
        return rhythm_probs, pitch_probs
