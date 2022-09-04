import os
import argparse
__all__ = ['TotalConfigs', 'get_settings']


def _settings():
    parser = argparse.ArgumentParser()

    """
    =========================General Settings===========================
    """
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--drop_prob', type=float, default=0.5)
    parser.add_argument('--bsz', type=int, default=64, help='batch size')
    parser.add_argument('--sample_numb', type=int, default=15, help='how many frames would you like to sample from a given video')
    parser.add_argument('--model_name', type=str, default='HMN', help='which model you would like to train/test?')

    """
    =========================Data Settings===========================
    """
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints')
    parser.add_argument('--result_dir', type=str, default='-1')
    parser.add_argument('--dataset_name', type=str, default='-1')
    parser.add_argument('--backbone_2d_name', type=str, default='-1', help='2d backbone name (InceptionResNetV2)')
    parser.add_argument('--backbone_3d_name', type=str, default='-1', help='3d backbone name (C3D)')
    parser.add_argument('--object_name', type=str, default='-1', help='object features name (vg_objects)')
    parser.add_argument('--semantics_dim', type=int, default=768, help='semantics embedding dim')

    """
    =========================Encoder Settings===========================
    """
    parser.add_argument('--backbone_2d_dim', type=int, default=2048, help='dimention for inceptionresnetv2')
    parser.add_argument('--backbone_3d_dim', type=int, default=2048, help='dimention for C3D')
    parser.add_argument('--object_dim', type=int, default=2048, help='dimention for vg_objects')
    parser.add_argument('--max_objects', type=int, default=8)

    parser.add_argument('--nheads', type=int, default=8)
    parser.add_argument('--entity_encoder_layer', type=int, default=2)
    parser.add_argument('--entity_decoder_layer', type=int, default=2)
    parser.add_argument('--dim_feedforward', type=int, default=2048)
    parser.add_argument('--transformer_activation', type=str, default='relu')
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--transformer_dropout', type=float, default=0.1)

    """
    =========================Decoder Settings===========================
    """
    parser.add_argument('--word_embedding_dim', type=int, default=300)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=1)


    """
    =========================Word Dict Settings===========================
    """
    parser.add_argument('--eos_idx', type=int, default=0)
    parser.add_argument('--sos_idx', type=int, default=1)
    parser.add_argument('--unk_idx', type=int, default=2)
    parser.add_argument('--n_vocab', type=int, default=-1, help='how many different words are there in the dataset')

    """
    =========================Training Settings===========================
    """
    parser.add_argument('--grad_clip', type=float, default=5.0)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--lambda_entity', type=float, default=1.0)
    parser.add_argument('--lambda_predicate', type=float, default=1.0)
    parser.add_argument('--lambda_sentence', type=float, default=1.0)
    parser.add_argument('--lambda_soft', type=float, default=0.1)
    parser.add_argument('--max_epochs', type=int, default=20)
    parser.add_argument('--visualize_every', type=int, default=10)
    parser.add_argument('--save_checkpoints_every', type=int, default=200)
    parser.add_argument('--save_checkpoints_path', type=str, default='-1')

    """
    =========================Testing Settings===========================
    """
    parser.add_argument('--beam_size', type=int, default=5)
    parser.add_argument('--max_caption_len', type=int, default=20 + 2)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--result_path', type=str, default='-1')

    args = parser.parse_args()
    return args


class TotalConfigs:
    def __init__(self, args):
        self.data = DataConfigs(args)
        self.dict = DictConfigs(args)
        self.encoder = EncoderConfigs(args)
        self.decoder = DecoderConfigs(args)
        self.train = TrainingConfigs(args)
        self.test = TestConfigs(args)

        self.seed = args.seed
        self.bsz = args.bsz
        self.drop_prob = args.drop_prob
        self.model_name = args.model_name
        self.sample_numb = args.sample_numb


class DataConfigs:
    def __init__(self, args):
        self.data_dir = args.data_dir
        self.checkpoints_dir = args.checkpoints_dir
        self.dataset_name = args.dataset_name
        self.backbone_2d_name = args.backbone_2d_name
        self.backbone_3d_name = args.backbone_3d_name
        self.object_name = args.object_name
        self.word_dim = args.word_embedding_dim

        assert self.dataset_name != '-1', 'Please set argument dataset_name'
        assert self.backbone_2d_name != '-1', 'Please set argument backbone_2d_name'
        assert self.backbone_3d_name != '-1', 'Please set argument backbone_3d_name'
        assert self.object_name != '-1', 'Please set argument object_name'

        # data dir
        self.data_dir = os.path.join(self.data_dir, self.dataset_name)

        # language part
        self.language_dir = os.path.join(self.data_dir, 'language')
        self.vid2language_path = os.path.join(self.language_dir, 'vid2language.pkl')
        self.vid2fillmask_path = os.path.join(self.data_dir, 'vid2fillmask_{}.pkl'.format(self.dataset_name))
        self.word2idx_path = os.path.join(self.language_dir, 'word2idx.pkl')
        self.idx2word_path = os.path.join(self.language_dir, 'idx2word.pkl')
        self.embedding_weights_path = os.path.join(self.language_dir, 'embedding_weights.pkl')
        self.vid2groundtruth_path = os.path.join(self.language_dir, 'vid2groundtruth.pkl')

        # visual part
        self.visual_dir = os.path.join(self.data_dir, 'visual')
        self.backbone2d_path_tpl = os.path.join(self.visual_dir, '{}_{}_{}.hdf5'.format(args.dataset_name, args.backbone_2d_name, '{}'))
        self.backbone3d_path_tpl = os.path.join(self.visual_dir, '{}_{}_{}.hdf5'.format(args.dataset_name, args.backbone_3d_name, '{}'))
        self.objects_path_tpl = os.path.join(self.visual_dir, '{}_{}_{}.hdf5'.format(args.dataset_name, args.object_name, '{}'))

        # dataset split part
        self.split_dir = os.path.join(self.data_dir, '{dataset_name}_splits'.format(dataset_name=self.dataset_name))
        self.videos_split_path_tpl = os.path.join(self.split_dir, '{}_{}_list.pkl'.format(self.dataset_name, '{}'))


class DictConfigs:
    def __init__(self, args):
        self.eos_idx = args.eos_idx
        self.sos_idx = args.sos_idx
        self.unk_idx = args.unk_idx
        self.n_vocab = args.n_vocab


class EncoderConfigs:
    def __init__(self, args):
        self.backbone_2d_dim = args.backbone_2d_dim
        self.backbone_3d_dim = args.backbone_3d_dim
        self.semantics_dim = args.semantics_dim
        self.object_dim = args.object_dim
        self.max_objects = args.max_objects

        self.nheads = args.nheads
        self.entity_encoder_layer = args.entity_encoder_layer
        self.entity_decoder_layer = args.entity_decoder_layer
        self.dim_feedforward = args.dim_feedforward
        self.transformer_activation = args.transformer_activation
        self.d_model = args.d_model
        self.trans_dropout = args.transformer_dropout


class DecoderConfigs:
    def __init__(self, args):
        self.hidden_dim = args.hidden_dim
        self.num_layers = args.num_layers
        self.n_vocab = -1


class TrainingConfigs:
    def __init__(self, args):
        self.grad_clip = args.grad_clip
        self.learning_rate = args.learning_rate
        self.lambda_entity = args.lambda_entity
        self.lambda_predicate = args.lambda_predicate
        self.lambda_sentence = args.lambda_sentence
        self.lambda_soft = args.lambda_soft
        self.max_epochs = args.max_epochs
        self.visualize_every = args.visualize_every
        self.checkpoints_dir = os.path.join(args.checkpoints_dir, args.dataset_name)
        self.save_checkpoints_every = args.save_checkpoints_every
        self.save_checkpoints_path = os.path.join(self.checkpoints_dir, 
        '{model_name}_epochs_{max_epochs}_lr_{lr}_entity_{obj}_predicate_{act}_sentence_{v}_soft{s}_ne_{ne}_nd_{nd}_max_objects_{mo}.ckpt'.format(
            model_name=args.model_name,
                max_epochs=args.max_epochs,
                lr=self.learning_rate,
                obj=self.lambda_entity,
                act=self.lambda_predicate,
                v=self.lambda_sentence,
                s=self.lambda_soft,
                ne=args.entity_encoder_layer,
                nd=args.entity_decoder_layer,
                mo=args.max_objects))
        if not os.path.exists(self.checkpoints_dir):
            os.makedirs(self.checkpoints_dir)
        if args.save_checkpoints_path != '-1':
            self.save_checkpoints_path = args.save_checkpoints_path


class TestConfigs:
    def __init__(self, args):
        self.beam_size = args.beam_size
        self.max_caption_len = args.max_caption_len
        self.temperature = args.temperature
        self.result_dir = os.path.join('./results/{dataset_name}'.format(dataset_name=args.dataset_name))
        if args.result_dir != '-1':
            self.result_dir = args.result_dir
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)
        self.result_path = os.path.join(
            self.result_dir, 
        '{model_name}_epochs_{max_epochs}_lr_{lr}_entity_{obj}_predicate_{act}_sentence_{v}_soft_{s}_ne_{ne}_nd_{nd}_max_objects_{mo}.pkl'.format(
            model_name=args.model_name,
            max_epochs=args.max_epochs,
            lr=args.learning_rate,
            obj=args.lambda_entity,
            act=args.lambda_predicate,
            v=args.lambda_sentence,
            s=args.lambda_soft,
            ne=args.entity_encoder_layer,
            nd=args.entity_decoder_layer,
            mo=args.max_objects)
        )
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)
        if args.result_path != '-1':
            self.result_path = args.result_path


def get_settings():
    args = _settings()
    configs = TotalConfigs(args=args)
    return configs

