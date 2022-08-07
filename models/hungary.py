import torch
from torch import nn, Tensor
import numpy as np
from scipy.optimize import linear_sum_assignment

class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self):
        super(HungarianMatcher, self).__init__()
        self.eps = 1e-6

    @torch.no_grad()
    def forward(self, salient_objects: Tensor, nouns_dict_list: list):
        """ Performs the matching
        Args:
            salient_objects: (bsz, max_objects, word_dim)
            nouns_dict_list: List[{'vec': nouns_vec, 'nouns': nouns}, ...]
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bsz, max_objects = salient_objects.shape[:2]
        device = salient_objects.device
        sizes = [len(item['nouns']) for item in nouns_dict_list]
        nouns_semantics = torch.cat([item['vec'][:len(item['nouns'])] for item in nouns_dict_list]).to(device)  # (\sigma nouns, word_dim)
        nouns_length = torch.norm(nouns_semantics, dim=-1, keepdim=True)  # (\sigma nouns, 1)
        salient_objects = salient_objects.flatten(0, 1)  # (bsz * max_objects, word_dim)
        salient_length = torch.norm(salient_objects, dim=-1, keepdim=True)  # (bsz * max_objects, 1)
        matrix_length = salient_length * nouns_length.permute([1, 0]) + self.eps  # (bsz * max_objects, \sigma nouns)


        cos_matrix = torch.mm(salient_objects, nouns_semantics.permute([1, 0]))  # (bsz * max_objects, \sigma nouns)
        cos_matrix = -cos_matrix / matrix_length  # (bsz * max_objects, \sigma nouns)
        cos_matrix = cos_matrix.view([bsz, max_objects, -1])  # (bsz, max_objects, \sigma nouns)
        indices = [linear_sum_assignment(c[i].detach().cpu().numpy()) for i, c in enumerate(cos_matrix.split(sizes, -1))]

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

