import torch
from torch import nn, Tensor


class SentenceLevelEncoder(nn.Module):
    def __init__(self, feature2d_dim, hidden_dim, semantics_dim, useless_objects):
        super(SentenceLevelEncoder, self).__init__()
        self.inf = 1e5
        self.useless_objects = useless_objects
        self.linear_2d = nn.Linear(feature2d_dim, hidden_dim)

        self.W = nn.Linear(hidden_dim, hidden_dim)

        self.Uo = nn.Linear(hidden_dim, hidden_dim)
        self.Um = nn.Linear(hidden_dim, hidden_dim)

        self.bo = nn.Parameter(torch.ones(hidden_dim), requires_grad=True)
        self.bm = nn.Parameter(torch.ones(hidden_dim), requires_grad=True)

        self.wo = nn.Linear(hidden_dim, 1)
        self.wm = nn.Linear(hidden_dim, 1)

        self.lstm = nn.LSTM(input_size=hidden_dim + hidden_dim + hidden_dim,
                            hidden_size=hidden_dim // 2,
                            num_layers=1, bidirectional=True, batch_first=True)
        self.fc_layer = nn.Linear(hidden_dim, semantics_dim)

    def forward(self, feature2ds: Tensor, vp_features: Tensor, object_features: Tensor, objects_mask: Tensor):
        """

        Args:
            feature2ds: (bsz, sample_numb, hidden_dim)
            vp_features: (bsz, sample_numb, hidden_dim)
            object_features: (bsz, max_objects, hidden_dim)
            objects_mask: (bsz, max_objects_per_video)

        Returns:
            video_features: (bsz, sample_numb, hidden_dim)
            video_pending: (bsz, semantics_dim)
        """
        sample_numb = feature2ds.shape[1]
        feature2ds = self.linear_2d(feature2ds)
        W_f2d = self.W(feature2ds)
        U_objs = self.Uo(object_features)
        U_motion = self.Um(vp_features)

        attn_feat = W_f2d.unsqueeze(2) + U_objs.unsqueeze(1) + self.bo  # (bsz, sample_numb, max_objects, hidden_dim)
        attn_weights = self.wo(torch.tanh(attn_feat))  # (bsz, sample_numb, max_objects, 1)
        objects_mask = objects_mask[:, None, :, None].repeat(1, sample_numb, 1, 1)  # (bsz, sample, max_objects_per_video, 1)
        if self.useless_objects:
            attn_weights = attn_weights - objects_mask.float() * self.inf
        attn_weights = attn_weights.softmax(dim=-2)  # (bsz, sample_numb, max_objects, 1)
        attn_objects = attn_weights * attn_feat
        attn_objects = attn_objects.sum(dim=-2)  # (bsz, sample_numb, hidden_dim)

        attn_feat = W_f2d.unsqueeze(2) + U_motion.unsqueeze(1) + self.bm  # (bsz, sample_numb, sample_numb, hidden_dim)
        attn_weights = self.wm(torch.tanh(attn_feat))  # (bsz, sample_numb, sample_numb, 1)
        attn_weights = attn_weights.softmax(dim=-2)  # (bsz, sample_numb, sample_numb, 1)
        attn_motion = attn_weights * attn_feat
        attn_motion = attn_motion.sum(dim=-2)  # (bsz, sample_numb, hidden_dim)

        features = torch.cat([feature2ds, attn_motion, attn_objects], dim=-1)  # (bsz, sample_numb, hidden_dim * 3)
        output, states = self.lstm(features)  # (bsz, sample_numb, hidden_dim)
        video = torch.max(output, dim=1)[0]  # (bsz, hidden_dim)
        video_pending = self.fc_layer(video)  # (bsz, semantics_dim)
        video_features = output  # (bsz, sample_numb, hidden_dim)

        return video_features, video_pending


