import os
import pickle
import h5py
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='-1', help='dataset name (MSVD | MSRVTT)')
    parser.add_argument('--split_dir', type=str, default='-1', help='dir contains split_list')
    parser.add_argument('--data_path', type=str, default='-1', help='the path of unsplited data')
    parser.add_argument('--data_name', type=str, default='-1', help='name of unsplited data')
    parser.add_argument('--target_dir', type=str, default='-1', help='where do you want to place your data')
    parser.add_argument('--split_objects', action='store_true')

    args = parser.parse_args()
    dataset_name = args.dataset_name
    split_dir = args.split_dir
    data_path = args.data_path
    data_name = args.data_name
    target_dir = args.target_dir

    assert dataset_name != '-1', 'Please set dataset_name!'
    assert split_dir != '-1', 'Please set split_dir!'
    assert data_path != '-1', 'Please set data_path!'
    assert data_name != '-1', 'Please set data_name!'
    assert target_dir != '-1', 'Please set target_dir!'

    splits_list_path_tpl = '{dataset_name}_{part}_list.pkl'.format(dataset_name=dataset_name, part='{}')
    dataset_split_path_tpl = '{}_{}_{}.hdf5'.format(dataset_name, data_name, '{}')
    split_part_list = ['train', 'valid', 'test']
    print('[split begin]', '=' * 20)
    with h5py.File(data_path, 'r') as f:
        for split in split_part_list:
            cur_vid_list_path = os.path.join(split_dir, splits_list_path_tpl.format(split))
            dataset_split_save_path = os.path.join(target_dir, dataset_split_path_tpl.format(split))
            with open(cur_vid_list_path, 'rb') as v:
                vid_list = pickle.load(v)
            with h5py.File(dataset_split_save_path, 'w') as t:
                for vid in vid_list:
                    if args.split_objects:
                        t[vid] = f[vid][()]
                    else:
                        temp_group = t.create_group(vid)
                        temp_group['feats'] = f[vid]['feats'][()]
                    # temp_group['bboxes'] = f[vid]['bboxes'][()]
                    # temp_group['kinds'] = f[vid]['kinds'][()]
    print('[split end]', '=' * 20)

