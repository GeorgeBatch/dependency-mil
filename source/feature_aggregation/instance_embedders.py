import torch.nn as nn

class IdentityEmbedder(nn.Module):
    def __init__(self):
        super(IdentityEmbedder, self).__init__()

    def forward(self, x):
        return x

class AdaptiveAvgPoolingEmbedder(nn.Module):
    def __init__(self, output_size):
        super(AdaptiveAvgPoolingEmbedder, self).__init__()
        self.adaptive_avg_pool = nn.AdaptiveAvgPool1d(output_size)

    def forward(self, x):
        return self.adaptive_avg_pool(x)

class LinearEmbedder(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearEmbedder, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)
    
class SliceEmbedder(nn.Module):
    def __init__(self, output_size):
        super(SliceEmbedder, self).__init__()
        self.output_size = output_size

    def forward(self, x):
        # slide over the last dimension of x
        return x[..., :self.output_size]