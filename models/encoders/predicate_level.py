import torch
import torch.nn as nn
from torch import Tensor


class PredicateLevelEncoder(nn.Module):
    def __init__(self, feature3d_dim, hidden_dim, semantics_dim, useless_objects):
        super(PredicateLevelEncoder, self).__init__()
        self.linear_layer = nn.Linear(feature3d_dim, hidden_dim)
        self.W = nn.Linear(hidden_dim, hidden_dim)
        self.U = nn.Linear(hidden_dim, hidden_dim)
        self.b = nn.Parameter(torch.ones(hidden_dim), requires_grad=True)
        self.w = nn.Linear(hidden_dim, 1)
        self.inf = 1e5
        self.useless_objects = useless_objects

        self.bilstm = nn.LSTM(input_size=hidden_dim + hidden_dim,
                            hidden_size=hidden_dim // 2,
                            num_layers=1, bidirectional=True, batch_first=True)
        self.fc_layer = nn.Linear(hidden_dim, semantics_dim)

    def forward(self, features3d: Tensor, objects: Tensor, objects_mask: Tensor):
        """

        Args:
            features3d: (bsz, sample_numb, 3d_dim)
            objects: (bsz, max_objects, hidden_dim)
            objects_mask: (bsz, max_objects_per_video)

        Returns:
            action_features: (bsz, sample_numb, hidden_dim * 2)
            action_pending: (bsz, semantics_dim)
        """
        sample_numb = features3d.shape[1]
        features3d = self.linear_layer(features3d)  # (bsz, sample_numb, hidden_dim)
        Wf3d = self.W(features3d)  # (bsz, sample_numb, hidden_dim)
        Uobjs = self.U(objects)  # (bsz, max_objects, hidden_dim)

        attn_feat = Wf3d.unsqueeze(2) + Uobjs.unsqueeze(1) + self.b  # (bsz, sample_numb, max_objects, hidden_dim)
        attn_weights = self.w(torch.tanh(attn_feat))  # (bsz, sample_numb, max_objects, 1)
        objects_mask = objects_mask[:, None, :, None].repeat(1, sample_numb, 1, 1)  # (bsz, sample_numb, max_objects_per_video, 1)
        if self.useless_objects:
            attn_weights = attn_weights - objects_mask.float() * self.inf
        attn_weights = attn_weights.softmax(dim=-2)  # (bsz, sample_numb, max_objects, 1)
        attn_objects = attn_weights * attn_feat
        attn_objects = attn_objects.sum(dim=-2)  # (bsz, sample_numb, hidden_dim)

        features = torch.cat([features3d, attn_objects], dim=-1)  # (bsz, sample_numb, hidden_dim * 2)
        output, states = self.bilstm(features)  # (bsz, sample_numb, hidden_dim)
        action = torch.max(output, dim=1)[0]  # (bsz, hidden_dim)
        action_pending = self.fc_layer(action)  # (bsz, semantics_dim)
        action_features = output  # (bsz, sample_numb, hidden_dim)

        return action_features, action_pending



