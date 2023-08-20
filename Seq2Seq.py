import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import VGG19_BN_Weights


class Encoder(nn.Module):
    def __init__(self, encoder_hidden_size):
        super(Encoder, self).__init__()
        # Load pre-trained VGG-19-BN model without the last max-pooling
        vgg19_bn = models.vgg19_bn(weights=VGG19_BN_Weights.DEFAULT)
        self.vgg_features = vgg19_bn.features[:-1]

        # Freeze the weights of the VGG-19-BN layers
        for param in self.vgg_features.parameters():
            param.requires_grad = False

        # Multi-layered Bidirectional GRU
        self.bidirectional_rnn = nn.GRU(
            input_size=512,  # VGG-19 output channels
            hidden_size=encoder_hidden_size,
            batch_first=True,
            bidirectional=True
        )

    def forward(self, x):
        # Extract features using VGG-19-BN
        features = self.vgg_features(x)

        # Reshape VGG features into a two-dimensional feature map
        batch_size, channels, height, width = features.size()
        features = features.view(batch_size, channels, -1).transpose(1, 2)

        # Pass through Bidirectional GRU
        output, _ = self.bidirectional_rnn(features)

        return output


class LocationBasedAttention(nn.Module):
    def __init__(self, encoder_hidden_size, decoder_hidden_size):
        super(LocationBasedAttention, self).__init__()

        self.W = nn.Linear(encoder_hidden_size + decoder_hidden_size, 1)

    def forward(self, encoder_outputs, decoder_hidden):
        # encoder_outputs: Batch x Seq_len x Encoder_hidden
        # decoder_hidden: Batch x Decoder_hidden

        seq_len = encoder_outputs.size(1)
        decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, seq_len, 1)

        combined = torch.cat((encoder_outputs, decoder_hidden), dim=2)
        attention_weights = torch.softmax(self.W(combined), dim=1)

        context_vector = torch.sum(attention_weights * encoder_outputs, dim=1)

        return context_vector, attention_weights


class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, decoder_hidden_size, num_layers):
        super(Decoder, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(
            input_size=embedding_dim + decoder_hidden_size * 2,
            hidden_size=decoder_hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.output_layer = nn.Linear(decoder_hidden_size, vocab_size)

    def forward(self, inputs, hidden, context):
        # inputs: Batch x 1 (sequence length is 1 for each time step)
        # hidden: num_layers x Batch x Decoder_hidden
        # context: Batch x Encoder_hidden

        embedded = self.embedding(inputs)
        combined = torch.cat((embedded, context.unsqueeze(1)), dim=2)
        output, hidden = self.gru(combined, hidden)
        output = self.output_layer(output.squeeze(1))

        return output, hidden


class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, encoder_hidden_size=256, decoder_hidden_size=256, num_layers=2):
        super(Seq2Seq, self).__init__()

        self.encoder = Encoder(encoder_hidden_size)
        self.attention = LocationBasedAttention(encoder_hidden_size * 2, decoder_hidden_size)
        self.decoder = Decoder(vocab_size, embedding_dim, decoder_hidden_size, num_layers)

    def forward(self, source, target):
        encoder_output = self.encoder(source)

        decoder_hidden = torch.zeros(2, source.size(0), self.decoder.gru.hidden_size, device=source.device)
        output_seq = torch.zeros(target.size(0), target.size(1), self.decoder.output_layer.out_features,
                                 device=source.device)

        for t in range(target.size(1)):
            context, _ = self.attention(encoder_output, decoder_hidden[-1])
            output, decoder_hidden = self.decoder(target[:, t:t + 1], decoder_hidden, context)
            output_seq[:, t, :] = output

        return output_seq
