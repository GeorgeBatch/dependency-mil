import torch

from source.models.class_connectors import (BahdanauSelfAttention,
                                            TransformerSelfAttention)

batch_size = 4
num_classes = 2
embedding_size = 16


def test_bahdanau_self_attention():
    # Assume B is your input tensor
    B = torch.rand((batch_size, num_classes, embedding_size))

    # Instantiate and apply the self-attention module
    bahdanau_attention_module = BahdanauSelfAttention(embedding_size)
    C_bahdanau = bahdanau_attention_module(B)
    assert C_bahdanau.shape == B.shape
    # print("C_bahdanau.shape:", C_bahdanau.shape)


def test_transformer_self_attention():
    transformer_num_heads = 1
    # Assume B is your input tensor
    B = torch.rand((batch_size, num_classes, embedding_size))

    transformer_attention_module = TransformerSelfAttention(embedding_size, transformer_num_heads)
    C_transformer = transformer_attention_module(B)
    # print("C_transformer.shape:", C_transformer.shape)
    assert C_transformer.shape == B.shape
