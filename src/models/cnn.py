import torch
import torch.nn as nn
import torch.nn.functional as F


class ShallowCNN(nn.Module):
    """
    Shallow CNN for genomic sequence binary classification.
    Takes one-hot encoded DNA sequences of shape (batch_size, 4, sequence_length).
    """

    def __init__(
        self,
        sequence_length: int = 1000,
        n_filters_1: int = 64,
        n_filters_2: int = 128,
        kernel_size_1: int = 15,
        kernel_size_2: int = 9,
        pool_size: int = 4,
        dropout_rate: float = 0.3,
        num_classes: int = 1,
    ):
        super(ShallowCNN, self).__init__()
        self.num_classes = num_classes

        # Conv Layer 1: capture TF motifs
        self.conv1 = nn.Conv1d(
            in_channels=4,
            out_channels=n_filters_1,
            kernel_size=kernel_size_1,
            padding=kernel_size_1 // 2,
        )
        self.bn1 = nn.BatchNorm1d(n_filters_1)
        self.pool1 = nn.MaxPool1d(kernel_size=pool_size)

        # Conv Layer 2: capture combination features
        self.conv2 = nn.Conv1d(
            in_channels=n_filters_1,
            out_channels=n_filters_2,
            kernel_size=kernel_size_2,
            padding=kernel_size_2 // 2,
        )
        self.bn2 = nn.BatchNorm1d(n_filters_2)

        # Dropout
        self.dropout = nn.Dropout(p=dropout_rate)

        # Fully connected layer
        # Output shape after pool1: sequence_length / pool_size
        # Global Max Pooling is applied after conv2, reducing length dimension to 1
        self.fc1 = nn.Linear(n_filters_2, 128)
        self.bn_fc = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # x shape: (batch_size, 4, seq_len)
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = F.relu(self.bn2(self.conv2(x)))

        # Global Max Pooling along the sequence length dimension (dim 2)
        x = torch.max(x, dim=2)[0]

        # Classification head
        x = self.dropout(x)
        x = F.relu(self.bn_fc(self.fc1(x)))
        x = self.dropout(x)
        logits = self.fc2(x)

        if self.num_classes == 1:
            return logits.squeeze(-1)  # Output shape: (batch_size,)
        return logits


class DeepCNN(nn.Module):
    """
    Deeper CNN architecture with 4 conv layers.
    """

    def __init__(
        self,
        sequence_length: int = 1000,
        dropout_rate: float = 0.4,
        num_classes: int = 1,
    ):
        super(DeepCNN, self).__init__()
        self.num_classes = num_classes

        self.conv1 = nn.Conv1d(4, 64, kernel_size=15, padding=7)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=9, padding=4)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(128, 128, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(2)

        self.conv4 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(256)

        self.dropout = nn.Dropout(p=dropout_rate)

        self.fc1 = nn.Linear(256, 128)
        self.bn_fc = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = F.relu(self.bn4(self.conv4(x)))

        # Global Max Pooling
        x = torch.max(x, dim=2)[0]

        x = self.dropout(x)
        x = F.relu(self.bn_fc(self.fc1(x)))
        x = self.dropout(x)
        logits = self.fc2(x)

        if self.num_classes == 1:
            return logits.squeeze(-1)
        return logits


class AttentionCNN(nn.Module):
    """
    CNN layers + Single Self-Attention layer architecture.
    """

    def __init__(
        self,
        sequence_length: int = 1000,
        n_heads: int = 4,
        dropout_rate: float = 0.3,
        num_classes: int = 1,
    ):
        super(AttentionCNN, self).__init__()
        self.num_classes = num_classes

        self.conv1 = nn.Conv1d(4, 64, kernel_size=15, padding=7)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(4)  # Shape -> (batch, 64, seq_len/4)

        # Dimension of attention query/key/value is n_filters (64)
        # Sequence length dimension becomes input length to Transformer Encoder Layer
        self.attention = nn.MultiheadAttention(
            embed_dim=64, num_heads=n_heads, dropout=dropout_rate, batch_first=True
        )
        self.bn_att = nn.LayerNorm(64)

        self.dropout = nn.Dropout(p=dropout_rate)

        self.fc1 = nn.Linear(64, 32)
        self.bn_fc = nn.BatchNorm1d(32)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))  # Shape: (batch, 64, L)

        # Permute for MultiheadAttention: (batch, L, 64)
        x = x.permute(0, 2, 1)

        # Self-attention forward
        attn_out, _ = self.attention(x, x, x)
        x = x + attn_out  # Residual connection
        x = self.bn_att(x)

        # Global pooling across sequence dimension (dim 1)
        x = torch.max(x, dim=1)[0]  # Shape: (batch, 64)

        x = self.dropout(x)
        x = F.relu(self.bn_fc(self.fc1(x)))
        x = self.dropout(x)
        logits = self.fc2(x)

        if self.num_classes == 1:
            return logits.squeeze(-1)
        return logits
