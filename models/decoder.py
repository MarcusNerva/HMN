import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, semantics_dim, hidden_dim, num_layers, embed_dim, n_vocab, with_objects, with_action, with_video, with_objects_semantics, with_action_semantics, with_video_semantics):
        super(Decoder, self).__init__()
        self.n_vocab = n_vocab
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.with_objects_semantics = with_objects_semantics
        self.with_action_semantics = with_action_semantics
        self.with_video_semantics = with_video_semantics

        total_visual_dim = 0
        total_semantics_dim = 0

        # objects visual features and corresponding semantics
        if with_objects:
            setattr(self, 'Uo', nn.Linear(hidden_dim, hidden_dim, bias=False))
            setattr(self, 'bo', nn.Parameter(torch.ones(hidden_dim), requires_grad=True))
            setattr(self, 'wo', nn.Linear(hidden_dim, 1, bias=False))
            total_visual_dim += hidden_dim
        if with_objects_semantics:
            setattr(self, 'Uos', nn.Linear(semantics_dim, hidden_dim, bias=False))
            setattr(self, 'bos', nn.Parameter(torch.ones(hidden_dim), requires_grad=True))
            setattr(self, 'wos', nn.Linear(hidden_dim, 1, bias=False))            
            total_semantics_dim += semantics_dim

        # action visual features and corresponding semantics
        if with_action:
            setattr(self, 'Um', nn.Linear(hidden_dim, hidden_dim, bias=False))
            setattr(self, 'bm', nn.Parameter(torch.ones(hidden_dim), requires_grad=True))
            setattr(self, 'wm', nn.Linear(hidden_dim, 1, bias=False))
            total_visual_dim += hidden_dim
        if with_action_semantics:
            total_semantics_dim += semantics_dim

        # video visual features and corresponding semantics
        if with_video:
            setattr(self, 'Uv', nn.Linear(hidden_dim, hidden_dim, bias=False))
            setattr(self, 'bv', nn.Parameter(torch.ones(hidden_dim), requires_grad=True))
            setattr(self, 'wv', nn.Linear(hidden_dim, 1, bias=False))
            total_visual_dim += hidden_dim
        if with_video_semantics:
            total_semantics_dim += semantics_dim

        # fuse visual features together
        if total_visual_dim != hidden_dim:
            setattr(self, 'linear_visual_layer', nn.Linear(total_visual_dim, hidden_dim))

        # fuse semantics features together
        if with_objects_semantics or with_action_semantics or with_video_semantics:
            setattr(self, 'linear_semantics_layer', nn.Linear(total_semantics_dim, hidden_dim))

        with_semantics = with_objects_semantics or with_action_semantics or with_video_semantics
        self.lstm = nn.LSTM(input_size=hidden_dim * 2 + embed_dim if with_semantics else hidden_dim + embed_dim,
                            hidden_size=hidden_dim, 
                            num_layers=num_layers)
        self.to_word = nn.Linear(hidden_dim, embed_dim)
        self.logit = nn.Linear(embed_dim, n_vocab)
        self.__init_weight()

    def __init_weight(self):
        init_range = 0.1
        self.logit.bias.data.fill_(0)
        self.logit.weight.data.uniform_(-init_range, init_range)

    def forward(self, objects, action, video, object_semantics, action_semantics, video_semantics, embed, last_states):
        last_hidden = last_states[0][0]  # (bsz, hidden_dim)
        Wh = self.W(last_hidden)  # (bsz, hidden_dim)
        U_obj = self.Uo(objects) if hasattr(self, 'Uo') else None  # (bsz, max_objects, hidden_dim)
        U_objs = self.Uos(object_semantics) if hasattr(self, 'Uos') else None  # (bsz, max_objects, emb_dim)
        U_action = self.Um(action) if hasattr(self, 'Um') else None  # (bsz, sample_numb, hidden_dim)
        U_video = self.Uv(video) if hasattr(self, 'Uv') else None  # (bsz, sample_numb, hidden_dim)

        # for visual features
        if U_obj is not None:
            attn_weights = self.wo(torch.tanh(Wh[:, None, :] + U_obj + self.bo))
            attn_weights = attn_weights.softmax(dim=1)  # (bsz, max_objects, 1)
            attn_objects = attn_weights * objects  # (bsz, max_objects, hidden_dim)
            attn_objects = attn_objects.sum(dim=1)  # (bsz, hidden_dim)
        else:
            attn_objects = None

        if U_action is not None:
            attn_weights = self.wm(torch.tanh(Wh[:, None, :] + U_action + self.bm))
            attn_weights = attn_weights.softmax(dim=1)  # (bsz, sample_numb, 1)
            attn_motion = attn_weights * action  # (bsz, sample_numb, hidden_dim)
            attn_motion = attn_motion.sum(dim=1)  # (bsz, hidden_dim)
        else:
            attn_motion = None

        if U_video is not None:
            attn_weights = self.wv(torch.tanh(Wh[:, None, :] + U_video + self.bv))
            attn_weights = attn_weights.softmax(dim=1)  # (bsz, sample_numb, 1)
            attn_video = attn_weights * video  # (bsz, sample_numb, hidden_dim)
            attn_video = attn_video.sum(dim=1)  # (bsz, hidden_dim)
        else:
            attn_video = None

        feats_list = []
        if attn_video is not None:
            feats_list.append(attn_video)
        if attn_motion is not None: 
            feats_list.append(attn_motion)
        if attn_objects is not None: 
            feats_list.append(attn_objects)
        visual_feats = torch.cat(feats_list, dim=-1)
        visual_feats = self.linear_visual_layer(visual_feats) if hasattr(self, 'linear_visual_layer') else visual_feats

        # for semantic features
        semantics_list = []
        if self.with_objects_semantics:
            attn_weights = self.wos(torch.tanh(Wh[:, None, :] + U_objs + self.bos))
            attn_weights = attn_weights.softmax(dim=1)  # (bsz, max_objects, 1)
            attn_objs = attn_weights * object_semantics  # (bsz, max_objects, emb_dim)
            attn_objs = attn_objs.sum(dim=1)  # (bsz, emb_dim)
            semantics_list.append(attn_objs)
        if self.with_action_semantics: semantics_list.append(action_semantics)
        if self.with_video_semantics: semantics_list.append(video_semantics)
        semantics_feats = torch.cat(semantics_list, dim=-1) if len(semantics_list) > 0 else None
        semantics_feats = self.linear_semantics_layer(semantics_feats) if semantics_feats is not None else None

        # in addition to the lastest generated word, fuse visual features and semantic features together
        if semantics_feats is not None:
            input_feats = torch.cat([visual_feats, semantics_feats, embed], dim=-1)  # (bsz, hidden_dim * 2 + embed_dim)
        else:
            input_feats = torch.cat([visual_feats, embed], dim=-1)  # (bsz, hidden_dim + embed_dim)
        output, states = self.lstm(input_feats[None, ...], last_states)
        output = output.squeeze(0)  # (bsz, hidden_dim)
        output = self.to_word(output)  # (bsz, embed_dim)
        output_prob = self.logit(output)  # (bsz, n_vocab)
        output_prob = torch.log_softmax(output_prob, dim=1)  # (bsz, n_vocab)

        return output_prob, states

