"""
Usage example in: tests/unit/models/test_class_connectors.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BahdanauSelfAttention(nn.Module):
    def __init__(self, embedding_size):
        super(BahdanauSelfAttention, self).__init__()
        self.embedding_size = embedding_size
        # Weight matrix for alignment model
        self.W = nn.Linear(embedding_size, embedding_size, bias=False)
        # Parameter vector for alignment model
        self.v = nn.Parameter(torch.rand(embedding_size))

    def forward(self, B):
        batch_size, num_classes, _ = B.size()

        # Step 1: Compute alignment scores
        # a(i, j) = v^T * tanh(W * h_i + W * h_j)
        Wi = self.W(B)  # Shape: (batch_size, num_classes, embedding_size)
        # Shape: (batch_size, num_classes, 1, embedding_size)
        Wj = self.W(B).unsqueeze(2)
        alignment_scores = torch.tanh(
            Wi.unsqueeze(1) + Wj)  # Shape: (batch_size, num_classes, num_classes, embedding_size)
        # Shape: (batch_size, num_classes, num_classes)
        alignment_scores = torch.sum(self.v * alignment_scores, dim=-1)

        # Step 2: Compute softmax over alignment scores
        # Shape: (batch_size, num_classes, num_classes)
        attention_weights = F.softmax(alignment_scores, dim=-1)

        # Step 3: Compute context vectors
        # c_i = sum_j( alpha(i, j) * h_j )
        # Shape: (batch_size, num_classes, embedding_size)
        C = torch.bmm(attention_weights, B)

        return C


class TransformerSelfAttention(nn.Module):
    def __init__(self, embedding_size, num_heads):
        super(TransformerSelfAttention, self).__init__()
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.head_dim = embedding_size // num_heads

        assert (self.head_dim * num_heads == embedding_size), \
            "Embedding size needs to be divisible by num_heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(embedding_size, embedding_size)

    def forward(self, B):
        batch_size, num_classes, _ = B.size()
        B = B.view(batch_size, num_classes, self.num_heads, self.head_dim)

        values = self.values(B)
        keys = self.keys(B)
        queries = self.queries(B)

        # Step 1: Compute dot product of queries and keys, and scale by square root of head dimension
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys]) / (self.head_dim ** (1 / 2))

        # Step 2: Compute softmax over the scaled dot-product energies to obtain attention scores
        attention = torch.softmax(energy, dim=3)

        # Step 3: Compute weighted sum of values using the attention scores
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values])
        out = out.flatten(2)

        # Step 4: Apply a linear layer to the concatenated output
        out = self.fc_out(out)

        return out
