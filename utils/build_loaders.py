from torch.utils.data import DataLoader
from data.loader.data_loader import CaptionDataset, collate_fn_caption
from configs.settings import TotalConfigs


def build_loaders(cfgs: TotalConfigs, is_total=False):
    train_dataset = CaptionDataset(cfgs=cfgs, mode='train', save_on_disk=False, is_total=is_total)
    valid_dataset = CaptionDataset(cfgs=cfgs, mode='valid', save_on_disk=False, is_total=is_total)
    test_dataset = CaptionDataset(cfgs=cfgs, mode='test', save_on_disk=True, is_total=is_total)

    train_loader = DataLoader(dataset=train_dataset, batch_size=cfgs.bsz, shuffle=True,
                              collate_fn=collate_fn_caption, num_workers=0)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=cfgs.bsz, shuffle=True,
                              collate_fn=collate_fn_caption, num_workers=0)
    test_loader = DataLoader(dataset=test_dataset, batch_size=cfgs.bsz, shuffle=False,
                             collate_fn=collate_fn_caption, num_workers=0)

    return train_loader, valid_loader, test_loader


def get_test_loader(cfgs: TotalConfigs, is_total=False):
    test_dataset = CaptionDataset(cfgs=cfgs, mode='test', save_on_disk=True, is_total=is_total)
    test_loader = DataLoader(dataset=test_dataset, batch_size=cfgs.bsz, 
                            shuffle=False, collate_fn=collate_fn_caption, 
                            num_workers=0)
    return test_loader


def get_train_loader(cfgs: TotalConfigs, save_on_disk=True):
    train_dataset = CaptionDataset(cfgs=cfgs, mode='train', save_on_disk=save_on_disk, is_total=False)
    train_loader = DataLoader(dataset=train_dataset, batch_size=cfgs.bsz, 
                                shuffle=False, collate_fn=collate_fn_caption,
                                num_workers=0)
    return train_loader

def get_valid_loader(cfgs: TotalConfigs, is_total=False):
    valid_dataset = CaptionDataset(cfgs=cfgs, mode='valid', save_on_disk=True, is_total=is_total)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=cfgs.bsz, 
                            shuffle=False, collate_fn=collate_fn_caption, 
                            num_workers=0)
    return valid_loader

