import torch
import pickle
from collections import OrderedDict, defaultdict

from utils.coco_caption.pycocoevalcap.bleu.bleu import Bleu
from utils.coco_caption.pycocoevalcap.cider.cider import Cider
from utils.coco_caption.pycocoevalcap.meteor.meteor import Meteor
from utils.coco_caption.pycocoevalcap.rouge.rouge import Rouge
from configs.settings import TotalConfigs

def language_eval(sample_seqs, groundtruth_seqs):
    assert len(sample_seqs) == len(groundtruth_seqs), 'length of sampled seqs is different from that of groundtruth seqs!'

    references, predictions = OrderedDict(), OrderedDict()
    for i in range(len(groundtruth_seqs)):
        references[i] = [groundtruth_seqs[i][j] for j in range(len(groundtruth_seqs[i]))]
    for i in range(len(sample_seqs)):
        predictions[i] = [sample_seqs[i]]

    predictions = {i: predictions[i] for i in range(len(sample_seqs))}
    references = {i: references[i] for i in range(len(groundtruth_seqs))}

    avg_bleu_score, bleu_score = Bleu(4).compute_score(references, predictions)
    print('avg_bleu_score == ', avg_bleu_score)
    avg_cider_score, cider_score = Cider().compute_score(references, predictions)
    print('avg_cider_score == ', avg_cider_score)
    avg_meteor_score, meteor_score = Meteor().compute_score(references, predictions)
    print('avg_meteor_score == ', avg_meteor_score)
    avg_rouge_score, rouge_score = Rouge().compute_score(references, predictions)
    print('avg_rouge_score == ', avg_rouge_score)

    return {'BLEU': avg_bleu_score, 'CIDEr': avg_cider_score, 'METEOR': avg_meteor_score, 'ROUGE': avg_rouge_score}


def decode_idx(seq, itow, eos_idx):
    ret = ''
    length = seq.shape[0]
    for i in range(length):
        if seq[i] == eos_idx: break
        if i > 0: ret += ' '
        ret += itow[seq[i]]
    return ret


@torch.no_grad()
def eval_fn(model, loader, device, idx2word, save_on_disk, cfgs: TotalConfigs, vid2groundtruth)->dict:
    model.eval()
    if save_on_disk:
        result_dict = {}
    predictions, gts = [], []

    for i, (feature2ds, feature3ds, object_feats, object_masks, \
            vp_semantics, caption_semantics, numberic_caps, masks, \
            captions, nouns_dict_list, vids, vocab_ids, vocab_probs, fillmasks) \
            in enumerate(loader):
        feature2ds = feature2ds.to(device)
        feature3ds = feature3ds.to(device)
        object_feats = object_feats.to(device)
        object_masks = object_masks.to(device)
        vp_semantics = vp_semantics.to(device)
        caption_semantics = caption_semantics.to(device)
        numberic_caps = numberic_caps.to(device)
        masks = masks.to(device)

        pred, seq_probabilities = model.sample(object_feats, object_masks, feature2ds, feature3ds)
        pred = pred.cpu().numpy()
        batch_pred = [decode_idx(single_seq, idx2word, cfgs.dict.eos_idx) for single_seq in pred]
        predictions += batch_pred
        batch_gts = [vid2groundtruth[id] for id in vids] if save_on_disk else [item for item in captions]
        gts += batch_gts

        if save_on_disk:
            assert len(batch_pred) == len(vids), \
                'expect len(batch_pred) == len(vids), ' \
                'but got len(batch_pred) == {} and len(vids) == {}'.format(len(batch_pred), len(vids))
            for vid, pred in zip(vids, batch_pred):
                result_dict[vid] = pred

    model.train()
    score_states = language_eval(sample_seqs=predictions, groundtruth_seqs=gts)

    if save_on_disk:
        with open(cfgs.test.result_path, 'wb') as f:
            pickle.dump(result_dict, f)

    return score_states

