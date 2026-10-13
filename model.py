import torch
import torch.nn as nn

from utils.GA import GraphAttention
from utils.GCN import GCN
from utils.loss_function import loss

# Model hyperparameters
GCN_input_dim = 1280
GCN_hidden_dim = 256
GCN_output_dim = 63
Attention_output_dim = 64
Num_classes = 1


class MAMBA_pro(nn.Module):
    def __init__(self, num_heads=8):
        super(MAMBA_pro, self).__init__()

        self.gcn = GCN(GCN_input_dim, GCN_hidden_dim, GCN_output_dim)
        self.graphormer = GraphAttention()

        self.GRP_linear = nn.Linear(1280, 64)

        self.LN1 = nn.LayerNorm(Attention_output_dim)
        self.LN2 = nn.LayerNorm(Attention_output_dim)

        self.fc1 = nn.Linear(128, 16)
        self.fc2 = nn.Linear(16, 1)

        self.criterion = loss(sigma=100).spearman_loss

    def PSSR_module(self, x, adj, attn_bias, centrality):
        padding_mask = x[:, :, 0].eq(0)

        x = self.gcn(x, adj)
        x = torch.cat((x, centrality.unsqueeze(-1)), dim=-1)
        x = x.masked_fill(padding_mask.unsqueeze(2), float(0))
        x, attn_weight = self.graphormer(x, attn_bias, padding_mask)
        return x, attn_weight

    def forward(self, x1, x2, x3, x4, adj, attn_bias, centrality):
        x1, _ = self.PSSR_module(x1, adj, attn_bias, centrality)
        x2, _ = self.PSSR_module(x2, adj, attn_bias, centrality)

        GRP = x4 - x3
        GRP = self.GRP_linear(GRP)

        x1 = torch.mean(x1, dim=1)
        x1 = self.LN1(x1)
        x2 = torch.mean(x2, dim=1)
        x2 = self.LN2(x2)

        LRP = x2 - x1
        x = torch.cat((LRP, GRP), dim=-1)

        x = torch.relu(self.fc1(x))
        output = torch.sigmoid(self.fc2(x))
        return output.squeeze(0)
