import pickle

from models.caption_models.hierarchical_model import HierarchicalModel
from models.decoder import Decoder

from models.encoders.transformer import Transformer
from models.encoders.entity_level import EntityLevelEncoder
from models.encoders.predicate_level import PredicateLevelEncoder
from models.encoders.sentence_level import SentenceLevelEncoder

from configs.settings import TotalConfigs


def build_model(cfgs: TotalConfigs):
    model_name = cfgs.model_name
    embedding_weights_path = cfgs.data.embedding_weights_path
    max_caption_len = cfgs.test.max_caption_len
    temperature = cfgs.test.temperature
    beam_size = cfgs.test.beam_size
    pad_idx = cfgs.dict.eos_idx
    eos_idx = cfgs.dict.eos_idx
    sos_idx = cfgs.dict.sos_idx
    unk_idx = cfgs.dict.unk_idx
    with open(embedding_weights_path, 'rb') as f:
        embedding_weights = pickle.load(f)
    n_vocab = embedding_weights.shape[0]
    cfgs.decoder.n_vocab = n_vocab

    feature2d_dim = cfgs.encoder.backbone_2d_dim
    feature3d_dim = cfgs.encoder.backbone_3d_dim
    object_dim = cfgs.encoder.object_dim
    semantics_dim = cfgs.encoder.semantics_dim
    hidden_dim = cfgs.decoder.hidden_dim
    decoder_num_layers = cfgs.decoder.num_layers
    embed_dim = cfgs.data.word_dim
    max_objects = cfgs.encoder.max_objects

    nheads = cfgs.encoder.nheads
    trans_num_encoder_layers = cfgs.encoder.num_encoder_layer
    trans_num_decoder_layers = cfgs.encoder.num_decoder_layer
    dim_feedforward = cfgs.encoder.dim_feedforward
    transformer_activation = cfgs.encoder.transformer_activation
    d_model = cfgs.encoder.d_model
    trans_dropout = cfgs.encoder.trans_dropout

    if model_name == 'HMN':
        # encoders
        transformer = Transformer(d_model=d_model, nhead=nheads,
                                  num_encoder_layers=trans_num_encoder_layers,
                                  num_decoder_layers=trans_num_decoder_layers,
                                  dim_feedforward=dim_feedforward,
                                  dropout=trans_dropout,
                                  activation=transformer_activation)
        entity_level_encoder = EntityLevelEncoder(transformer=transformer,
                                                         max_objects=max_objects,
                                                         object_dim=object_dim,
                                                         feature2d_dim=feature2d_dim,
                                                         feature3d_dim=feature3d_dim,
                                                         hidden_dim=hidden_dim,
                                                         word_dim=semantics_dim)
        predicate_level_encoder = PredicateLevelEncoder(feature3d_dim=feature3d_dim,
                                                      hidden_dim=hidden_dim,
                                                      semantics_dim=semantics_dim,
                                                      useless_objects=False)
        sentence_level_encoder = SentenceLevelEncoder(feature2d_dim=feature2d_dim,
                                                      hidden_dim=hidden_dim,
                                                      semantics_dim=semantics_dim,
                                                      useless_objects=False)
        
        # decoder
        decoder = Decoder(semantics_dim=semantics_dim, hidden_dim=hidden_dim,
                          num_layers=decoder_num_layers, embed_dim=embed_dim, n_vocab=n_vocab,
                          with_objects=True, with_action=True, with_video=True,
                          with_objects_semantics=True,
                          with_action_semantics=True,
                          with_video_semantics=True)
    
    else:
        raise NotImplementedError

    # HMN
    model = HierarchicalModel(entity_level=entity_level_encoder,
                                predicate_level=predicate_level_encoder,
                                sentence_level=sentence_level_encoder,
                                decoder=decoder,
                                word_embedding_weights=embedding_weights,
                                max_caption_len=max_caption_len,
                                beam_size=beam_size, pad_idx=pad_idx,
                                temperature=temperature,
                                eos_idx=eos_idx,
                                sos_idx=sos_idx,
                                unk_idx=unk_idx)

    return model


