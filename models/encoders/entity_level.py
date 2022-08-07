import torch
import torch.nn as nn
import numpy as np
from torch import Tensor
from torch.nn.modules.linear import Linear


class EntityLevelEncoder(nn.Module):
    def __init__(self, transformer, max_objects, object_dim, feature2d_dim, feature3d_dim, hidden_dim, word_dim):
        super(EntityLevelEncoder, self).__init__()
        self.max_objects = max_objects

        self.query_embed = nn.Embedding(max_objects, hidden_dim)
        self.input_proj = nn.Linear(object_dim, hidden_dim)
        self.feature2d_proj = nn.Linear(feature2d_dim, hidden_dim)
        self.feature3d_proj = nn.Linear(feature3d_dim, hidden_dim)

        self.bilstm = nn.LSTM(input_size=hidden_dim * 2, hidden_size=hidden_dim//2, 
                            batch_first=True, bidirectional=True)
        self.transformer = transformer
        self.fc_layer = nn.Linear(hidden_dim, word_dim)
        
    def forward(self, features_2d: Tensor, features_3d: Tensor, objects: Tensor, objects_mask: Tensor):
        """
        
        Args:
            features_2d: (bsz, sample_numb, feature2d_dim)
            features_3d: (bsz, sample_numb, feature3d_dim)
            objects: (bsz, max_objects_per_video, object_dim)
            objects_mask: (bsz, max_objects_per_video)

        Returns:
            salient_info: (bsz, max_objects, hidden_dim)
            object_pending: (bsz, max_objects, word_dim)
        """
        device = objects.device
        bsz, sample_numb, max_objects_per_video = features_2d.shape[0], features_3d.shape[1], objects.shape[1]
        features_2d = self.feature2d_proj(features_2d.view(-1, features_2d.shape[-1]))
        features_2d = features_2d.view(bsz, sample_numb, -1).contiguous()  # (bsz, sample_numb, hidden_dim)
        features_3d = self.feature3d_proj(features_3d.view(-1, features_3d.shape[-1]))
        features_3d = features_3d.view(bsz, sample_numb, -1).contiguous()  # (bsz, sample_numb, hidden_dim)
        content_vectors = torch.cat([features_2d, features_3d], dim=-1)  # (bsz, sample_numb, hidden_dim * 2)
        
        content_vectors, _ = self.bilstm(content_vectors)  # (bsz, sample_numb, hidden_dim)
        content_vectors = torch.max(content_vectors, dim=1)[0]  # (bsz, hidden_dim)
        
        tgt = content_vectors[None, ...].repeat(self.max_objects, 1, 1)  # (max_objects, bsz, hidden_dim)
        objects = self.input_proj(objects.view(-1, objects.shape[-1]))
        objects = objects.view(bsz, max_objects_per_video, -1).contiguous()  # (bsz, max_objects_per_video, hidden_dim)
        
        mask = objects_mask.to(device).bool()  # (bsz, max_objects_per_video)
        salient_objects = self.transformer(objects, tgt, mask, self.query_embed.weight)[0][0]  # (bsz, max_objects, hidden_dim)
        object_pending = self.fc_layer(salient_objects)
        return salient_objects, object_pending



