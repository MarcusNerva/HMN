import torch
from torch.utils.data import Dataset
import numpy as np
import os
import h5py
import pickle
from collections import defaultdict
import sys
sys.path.append('../../')
from configs.settings import TotalConfigs


def get_ids_and_probs(fillmask_steps, max_caption_len):
    if fillmask_steps is None:
        return None, None, None

    ret_ids, ret_probs = [], []
    ret_mask = torch.zeros(max_caption_len)

    for step in fillmask_steps:
        step_ids, step_probs = zip(*step)
        step_ids, step_probs = torch.Tensor(step_ids).long(), torch.Tensor(step_probs).float()
        ret_ids.append(step_ids)
        ret_probs.append(step_probs)
    gap = max_caption_len - len(fillmask_steps)
    for i in range(gap):
        zero_ids, zero_probs = torch.zeros(50).long(), torch.zeros(50).float()
        ret_ids.append(zero_ids)
        ret_probs.append(zero_probs)

    ret_ids = torch.cat([item[None, ...] for item in ret_ids], dim=0)
    ret_probs = torch.cat([item[None, ...] for item in ret_probs], dim=0)
    ret_mask[:len(fillmask_steps)] = 1
    
    return ret_ids, ret_probs, ret_mask


class CaptionDataset(Dataset):
    def __init__(self, cfgs: TotalConfigs, mode, save_on_disk=False, is_total=False):
        """
        Args:
            args: configurations.
            mode: train/valid/test.
            save_on_disk: whether save the prediction on disk or not.
                        True->Each video only appears once.
                        False->The number of times each video appears depends on
                                the number of its corresponding captions.
        """
        super(CaptionDataset, self).__init__()
        self.mode = mode
        self.save_on_disk = save_on_disk
        self.is_total = is_total
        sample_numb = cfgs.sample_numb  # how many frames are sampled to perform video captioning?
        max_caption_len = cfgs.test.max_caption_len

        # language part
        vid2language_path = cfgs.data.vid2language_path
        vid2fillmask_path = cfgs.data.vid2fillmask_path

        # visual part
        backbone2d_path = cfgs.data.backbone2d_path_tpl.format(mode)
        backbone3d_path = cfgs.data.backbone3d_path_tpl.format(mode)
        objects_path = cfgs.data.objects_path_tpl.format(mode)

        # dataset split part
        videos_split_path = cfgs.data.videos_split_path_tpl.format(mode)

        with open(videos_split_path, 'rb') as f:
            video_ids = pickle.load(f)

        self.video_ids = video_ids
        self.corresponding_vid = []

        self.backbone_2d_dict = {}
        self.backbone_3d_dict = {}
        self.objects_dict = {}
        self.total_entries = []  # (numberic words, original caption)
        self.vid2captions = defaultdict(list)

        # feature 2d dict
        with h5py.File(backbone2d_path, 'r') as f:
            for vid in video_ids:
                temp_feat = f[vid][()]
                sampled_idxs = np.linspace(0, len(temp_feat) - 1, sample_numb, dtype=int)
                self.backbone_2d_dict[vid] = temp_feat[sampled_idxs]

        # feature 3d dict
        with h5py.File(backbone3d_path, 'r') as f:
            for vid in video_ids:
                temp_feat = f[vid][()]
                sampled_idxs = np.linspace(0, len(temp_feat) - 1, sample_numb, dtype=int)
                self.backbone_3d_dict[vid] = temp_feat[sampled_idxs]

        # feature object dict
        with h5py.File(objects_path, 'r') as f:
            for vid in video_ids:
                temp_feat = f[vid]['feats'][()]
                self.objects_dict[vid] = temp_feat

        with open(vid2language_path, 'rb') as f:
            self.vid2language = pickle.load(f)
        
        if cfgs.train.lambda_soft > 0 and not save_on_disk:
            with open(vid2fillmask_path, 'rb') as f:
                self.vid2fillmask = pickle.load(f)
        
        for vid in video_ids:
            fillmask_dict = self.vid2fillmask[vid] if cfgs.train.lambda_soft > 0 and not save_on_disk and vid in self.vid2fillmask else None
            for item in self.vid2language[vid]:
                caption, numberic_cap, vp_semantics, caption_semantics, nouns, nouns_vec = item
                current_mask = fillmask_dict[caption] if fillmask_dict is not None else None
                vocab_ids, vocab_probs, fillmasks = get_ids_and_probs(current_mask, max_caption_len)
                self.total_entries.append((numberic_cap, vp_semantics, caption_semantics, nouns, nouns_vec, vocab_ids, vocab_probs, fillmasks))
                self.corresponding_vid.append(vid)
                self.vid2captions[vid].append(caption)
        
    def __getitem__(self, idx):
        """
        Returns:
            feature2d: (sample_numb, dim2d)
            feature3d: (sample_numb, dim3d)
            objects: (sample_numb * object_num, dim_obj) or (object_num_per_video, dim_obj)
            numberic: (max_caption_len, )
            captions: List[str]
            vid: str
        """
        vid = self.corresponding_vid[idx] if (self.mode == 'train' and not self.save_on_disk) or self.is_total else self.video_ids[idx]
        choose_idx = 0

        feature2d = self.backbone_2d_dict[vid]
        feature3d = self.backbone_3d_dict[vid]
        objects = self.objects_dict[vid]

        if (self.mode == 'train' and not self.save_on_disk) or self.is_total:
            numberic_cap, vp_semantics, caption_semantics, nouns, nouns_vec, vocab_ids, vocab_probs, fillmasks = self.total_entries[idx]
        else:
            numberic_cap, vp_semantics, caption_semantics, nouns, nouns_vec = self.vid2language[vid][choose_idx][1:]
            vocab_ids, vocab_probs, fillmasks = None, None, None

        captions = self.vid2captions[vid]
        nouns_dict = {'nouns': nouns, 'vec': torch.FloatTensor(nouns_vec)}

        return torch.FloatTensor(feature2d), torch.FloatTensor(feature3d), torch.FloatTensor(objects), \
               torch.LongTensor(numberic_cap), \
               torch.FloatTensor(vp_semantics), \
               torch.FloatTensor(caption_semantics), captions, nouns_dict, vid, \
                   vocab_ids, vocab_probs, fillmasks

    def __len__(self):
        if (self.mode == 'train' and not self.save_on_disk) or self.is_total:
            return len(self.total_entries)
        else:
            return len(self.video_ids)


def collate_fn_caption(batch):
    feature2ds, feature3ds, objects, numberic_caps, \
    vp_semantics, caption_semantics, captions, nouns_dict_list, vids, \
    vocab_ids, vocab_probs, fillmasks = zip(*batch)

    bsz, obj_dim = len(feature2ds), objects[0].shape[-1]
    longest_objects_num = max([item.shape[0] for item in objects])
    ret_objects = torch.zeros([bsz, longest_objects_num, obj_dim])
    ret_objects_mask = torch.ones([bsz, longest_objects_num])
    for i in range(bsz):
        ret_objects[i, :objects[i].shape[0], :] = objects[i]
        ret_objects_mask[i, :objects[i].shape[0]] = 0.0

    feature2ds = torch.cat([item[None, ...] for item in feature2ds], dim=0)  # (bsz, sample_numb, dim_2d)
    feature3ds = torch.cat([item[None, ...] for item in feature3ds], dim=0)  # (bsz, sample_numb, dim_3d)
    
    vp_semantics = torch.cat([item[None, ...] for item in vp_semantics], dim=0)  # (bsz, dim_sem)
    caption_semantics = torch.cat([item[None, ...] for item in caption_semantics], dim=0)  # (bsz, dim_sem)

    numberic_caps = torch.cat([item[None, ...] for item in numberic_caps], dim=0)  # (bsz, seq_len)
    masks = numberic_caps > 0
    
    captions = [item for item in captions]
    nouns = list(nouns_dict_list)
    vids = list(vids)
    vocab_ids = torch.cat([item[None, ...] for item in vocab_ids], dim=0).long() if vocab_ids[0] is not None else None  # (bsz, seq_len, 50)
    vocab_probs = torch.cat([item[None, ...] for item in vocab_probs], dim=0).float() if vocab_probs[0] is not None else None  # (bsz, seq_len, 50)
    fillmasks = torch.cat([item[None, ...] for item in fillmasks], dim=0).float() if fillmasks[0] is not None else None  # (bsz, seq_len)

    return feature2ds.float(), feature3ds.float(), ret_objects.float(), ret_objects_mask.float(), \
           vp_semantics.float(), caption_semantics.float(), \
           numberic_caps.long(), masks.float(), captions, nouns, vids, \
               vocab_ids, vocab_probs, fillmasks


