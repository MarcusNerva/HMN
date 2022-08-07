import torch
import torch.nn as nn
from models.caption_models.caption_module import CaptionModule


class HierarchicalModel(CaptionModule):
    def __init__(self, entity_level: nn.Module, predicate_level: nn.Module, sentence_level: nn.Module,
                 decoder: nn.Module, word_embedding_weights, max_caption_len: int, beam_size: int, pad_idx=0, 
                 temperature=1, eos_idx=0, sos_idx=-1, unk_idx=-1):
        """
        Args:
            entity_level: for encoding objects information.
            predicate_level: for encoding action information.
            sentence_level: for encoding the whole video information.
            decoder:  for generating words.
            word_embedding_weights: pretrained word embedding weight.
            max_caption_len: generated sentences are no longer than max_caption_len.
            pad_idx: corresponding index of '<PAD>'.
        """
        super(HierarchicalModel, self).__init__(beam_size=beam_size)
        self.entity_level = entity_level
        self.predicate_level = predicate_level
        self.sentence_level = sentence_level
        self.decoder = decoder
        self.max_caption_len = max_caption_len
        self.temperature = temperature
        self.eos_idx = eos_idx
        self.sos_idx = sos_idx
        self.unk_idx = unk_idx
        self.num_layers = decoder.num_layers
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(word_embedding_weights),
                                                      freeze=False, padding_idx=pad_idx)

    def get_rnn_init_hidden(self, bsz, hidden_size, device):
        # (hidden_state, cell_state)
        return (torch.zeros(self.num_layers, bsz, hidden_size).to(device),
                torch.zeros(self.num_layers, bsz, hidden_size).to(device))

    def forward_encoder(self, objects, objects_mask, feature2ds, feature3ds):
        """

        Args:
            objects: (bsz, max_objects_per_video, object_dim)
            objects_mask: (bsz, max_objects_per_video)
            feature2ds: (bsz, sample_numb, feature2d_dim)
            feature3ds: (bsz, sample_numb, feature3d_dim)

        Returns:
            objects_feats: (bsz, max_objects, hidden_dim)
            action_feats: (bsz, sample_numb, hidden_dim)
            video_feats: (bsz, sample_numb, hidden_dim)

            objects_semantics: (bsz, max_objects, word_dim)
            action_semantics: (bsz, semantics_dim)
            video_semantics: (bsz, semantics_dim)
        """
        objects_feats, objects_semantics = self.entity_level(feature2ds, feature3ds, objects, objects_mask)
        action_feats, action_semantics = self.predicate_level(feature3ds, objects_feats, objects_mask)
        video_feats, video_semantics = self.sentence_level(feature2ds, action_feats, objects_feats, objects_mask)

        return objects_feats, action_feats, video_feats, objects_semantics, action_semantics, video_semantics

    def forward_decoder(self, objects_feats, action_feats, video_feats, objects_semantics, action_semantics, video_semantics, pre_embedding, pre_state):
        """

        Args:
            objects_feats: (bsz, max_objects, hidden_dim)
            action_feats: (bsz, sample_numb, hidden_dim)
            video_feats: (bsz, sample_numb, hidden_dim)
            objects_semantics: (bsz, max_objects, word_dim)
            action_semantics: (bsz, semantics_dim)
            video_semantics: (bsz, semantics_dim)

            pre_embedding: (bsz, word_embed_dim)
            pre_state: (hidden_state, cell_state)

        Returns:
            output_prob: (bsz, n_vocab)
            current_state: (hidden_state, cell_state)
        """
        output_prob, current_state = self.decoder(objects_feats, action_feats, video_feats, objects_semantics, action_semantics, video_semantics, pre_embedding, pre_state)
        return output_prob, current_state

    def forward(self, objects_feats, objects_mask, feature2ds, feature3ds, numberic_captions):
        """

        Args:
            numberic_captions: (bsz, max_caption_len)

        Returns:
            ret_seq: (bsz, max_caption_len, n_vocab)
        """
        bsz, n_vocab = feature2ds.shape[0], self.decoder.n_vocab
        device = objects_feats.device
        objects_feats, action_feats, video_feats, objects_semantics, action_semantics, video_semantics = self.forward_encoder(objects_feats, objects_mask, feature2ds, feature3ds)
        state = self.get_rnn_init_hidden(bsz=bsz, hidden_size=self.decoder.hidden_dim, device=device)
        outputs = []

        for i in range(self.max_caption_len):
            if i > 0 and numberic_captions[:, i].sum() == 0:
                output_word = torch.zeros([bsz, n_vocab]).cuda()
                outputs.append(output_word)
                continue

            it = numberic_captions[:, i].clone()
            it_embeded = self.embedding(it)
            output_word, state = self.forward_decoder(objects_feats, action_feats, video_feats,
                                                      objects_semantics, action_semantics,
                                                      video_semantics, it_embeded, state)
            outputs.append(output_word)

        ret_seq = torch.stack(outputs, dim=1)
        return ret_seq, objects_semantics, action_semantics, video_semantics

