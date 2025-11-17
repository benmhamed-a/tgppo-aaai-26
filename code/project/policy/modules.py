import torch
import torch.nn as nn
from torch.nn import functional as F
import functools


def get_norm_layer(norm_type='none'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm1d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm1d, affine=False, track_running_stats=False)
    elif norm_type == 'layer':
        norm_layer = functools.partial(nn.LayerNorm)
    elif norm_type == 'none':
        norm_layer = functools.partial(nn.Identity)
    else:
        raise NotImplementedError(f'normalization layer [{norm_type}] is not found')
    return norm_layer


class TreeGateBranchingNet(nn.Module):
    def __init__(self, branch_size, tree_state_size, dim_reduce_factor, infimum=8, norm='none', depth=2, hidden_size=128):
        super().__init__()
        norm_layer = get_norm_layer(norm)
        self.branch_size = branch_size
        self.tree_state_size = tree_state_size
        self.dim_reduce_factor = dim_reduce_factor
        self.infimum = infimum
        self.depth = depth
        self.hidden_size = hidden_size

        self.n_layers = 0
        unit_count = infimum
        while unit_count < branch_size:
            unit_count *= dim_reduce_factor
            self.n_layers += 1
        self.n_units_dict = {}

        self.BranchingNet = nn.ModuleList()
        input_dim = hidden_size
        for i in range(self.n_layers):
            output_dim = max(int(input_dim / dim_reduce_factor), 1)
            self.n_units_dict[i] = input_dim
            if i < self.n_layers - 1:
                layer = [nn.Linear(input_dim, output_dim), norm_layer(output_dim), nn.ReLU(True)]
            else:
                layer = [nn.Linear(input_dim, output_dim)]
            input_dim = output_dim
            self.BranchingNet.append(nn.Sequential(*layer))

        self.GatingNet = nn.Sequential()
        self.n_attentional_units = sum(self.n_units_dict.values())
        if depth == 1:
            self.GatingNet.add_module('gate_linear', nn.Linear(tree_state_size, self.n_attentional_units))
            self.GatingNet.add_module('gate_sig', nn.Sigmoid())
        else:
            self.GatingNet.add_module('gate_linear1', nn.Linear(tree_state_size, hidden_size))
            self.GatingNet.add_module('gate_relu1', nn.ReLU(True))
            for i in range(depth - 2):
                self.GatingNet.add_module(f'gate_linear{i+2}', nn.Linear(hidden_size, hidden_size))
                self.GatingNet.add_module(f'gate_relu{i+2}', nn.ReLU(True))
            self.GatingNet.add_module('gate_linear_last', nn.Linear(hidden_size, self.n_attentional_units))
            self.GatingNet.add_module('gate_sig', nn.Sigmoid())

    def forward(self, cands_state_mat, tree_state):
        attn_weights = self.GatingNet(tree_state)
        start_slice_idx = 0
        for index, layer in enumerate(self.BranchingNet):
            end_slice_idx = start_slice_idx + self.n_units_dict[index]
            attn_slice = attn_weights[:, start_slice_idx:end_slice_idx]
            if cands_state_mat.dim() == 3:
                cands_state_mat = cands_state_mat * attn_slice.unsqueeze(1)
            else:
                cands_state_mat = cands_state_mat * attn_slice
            cands_state_mat = layer(cands_state_mat)
            start_slice_idx = end_slice_idx
        if cands_state_mat.size(-1) == 1:
            return cands_state_mat.squeeze(-1)
        else:
            # if 3D -> mean over candidates; if 2D -> mean over features
            dim = 1 if cands_state_mat.dim() == 3 else -1
            return cands_state_mat.mean(dim=dim)


class BiMatchingNet(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear1_1 = nn.Linear(hidden_size, hidden_size)
        self.linear1_2 = nn.Linear(hidden_size, hidden_size)
        self.linear2_1 = nn.Linear(hidden_size, hidden_size)
        self.linear2_2 = nn.Linear(hidden_size, hidden_size)
        self.linear3_1 = nn.Linear(hidden_size, hidden_size)
        self.linear3_2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, tree_feat, var_feat, padding_mask):
        """
        tree_feat: [B, E]
        var_feat:  [B, L, E]
        padding_mask: [B, L] True for pads; may be None
        """
        B, L, E = var_feat.shape
        if padding_mask is None:
            padding_mask = torch.zeros((B, L), dtype=torch.bool, device=var_feat.device)

        tree_feat = tree_feat.unsqueeze(1)  # [B,1,E]
        G_tc = torch.bmm(self.linear1_1(tree_feat), var_feat.transpose(1, 2)).squeeze(1)  # [B, L]
        G_tc = G_tc.masked_fill(padding_mask, float('-inf'))
        G_tc = F.softmax(G_tc, dim=1).unsqueeze(1)  # [B,1,L]

        G_ct = torch.bmm(self.linear1_2(var_feat), tree_feat.transpose(1, 2)).squeeze(2)  # [B,L]
        G_ct = G_ct.masked_fill(padding_mask, float('-inf'))
        G_ct = F.softmax(G_ct, dim=1).unsqueeze(2)  # [B,L,1]

        E_t = torch.bmm(G_tc, var_feat)              # [B,1,E]
        E_c = torch.bmm(G_ct, tree_feat)             # [B,L,1] x [B,1,E] -> [B,L,E]

        S_tc = F.relu(self.linear2_1(E_t))           # [B,1,E]
        S_ct = F.relu(self.linear2_2(E_c))           # [B,L,E]

        attn_weight = torch.sigmoid(self.linear3_1(S_tc) + self.linear3_2(S_ct))  # [B,L,E]
        M_tc = attn_weight * S_tc + (1 - attn_weight) * S_ct
        return M_tc