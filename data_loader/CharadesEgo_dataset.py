import os
import sys
import csv
import pandas as pd
sys.path.append('/apdcephfs/private_qinghonglin/video_codebase/EgoVLP/')

from base.base_dataset import TextVideoDataset
try:
    from transforms import init_transform_dict, init_video_transform_dict
except:
    pass

import torch
from PIL import Image
from torchvision import transforms

class CharadesEgo(TextVideoDataset):
    def _load_metadata(self):
        split_files = {
            'train': 'metadata_train.csv',
            'val': 'CharadesEgo_v1_test_only1st.csv',
            'test': 'CharadesEgo_v1_test_only1st.csv'
        }
        target_split_fp = split_files[self.split]
        if self.split == 'train':
            metadata = pd.read_csv(os.path.join(self.meta_dir, target_split_fp), delimiter='\t')
        else:
            metadata = pd.read_csv(os.path.join(self.meta_dir, target_split_fp))
        if self.subsample < 1:
            metadata = metadata.sample(frac=self.subsample)

        self.metadata = metadata
        if not self.split == 'train':
            self.label = self._parse_charades_csv(os.path.join(self.meta_dir, target_split_fp))

    def _parse_charades_csv(self, filename):
        labels = {}
        with open(filename) as f:
            reader = csv.DictReader(f)
            for row in reader:
                vid = row['id']
                actions = row['actions']
                if actions == '':
                    actions = []
                else:
                    actions = [a.split(' ') for a in actions.split(';')]
                    actions = [{'class': x, 'start': float(
                        y), 'end': float(z)} for x, y, z in actions]
                labels[vid] = actions
        return labels

    def _get_video_path(self, sample):
        rel_video_fp = sample['id'] + '.mp4'
        full_video_fp = os.path.join(self.data_dir, rel_video_fp)

        return full_video_fp, rel_video_fp

    def _get_caption(self, sample):
        if self.split in ['val', 'test']:
            return sample[6]
        else:
            return sample['narration']

    def _cls2int(self, x):
        return int(x[1:])

    def __getitem__(self, item):
        if self.split in ['val', 'test']:
            return self._get_val(item)
        else:
            return self._get_train(item)

    def _get_train(self, item):
        item = item % len(self.metadata)
        sample = self.metadata.iloc[item]
        video_fp, rel_fp = self._get_video_path(sample)
        caption = self._get_caption(sample)

        start_sec, end_sec = sample['t_start'],  sample['t_end']

        video_loading = self.video_params.get('loading', 'non_strict')
        frame_sample = 'rand'
        if self.split in ['test', 'val']:
            frame_sample = 'uniform'

        try:
            if os.path.isfile(video_fp):
                imgs, idxs = self.video_reader(video_path=video_fp, num_frames=self.video_params['num_frames'], sample=frame_sample,
                                               start_sec=start_sec, end_sec=end_sec)
            else:
                print(f"Warning: missing video file {video_fp}.")
                assert False
        except Exception as e:
            if video_loading == 'strict':
                raise ValueError(
                    f'Video loading failed for {video_fp}, video loading for this dataset is strict.') from e
            else:
                imgs = Image.new('RGB', (self.video_params['input_res'], self.video_params['input_res']), (0, 0, 0))
                imgs = transforms.ToTensor()(imgs).unsqueeze(0)

        if self.transforms is not None:
            if self.video_params['num_frames'] > 1:
                imgs = imgs.transpose(0, 1)  # [T, C, H, W] ---> [C, T, H, W]
                imgs = self.transforms(imgs)
                imgs = imgs.transpose(0, 1)  # recover
            else:
                imgs = self.transforms(imgs)

        final = torch.zeros([self.video_params['num_frames'], 3, self.video_params['input_res'],
                             self.video_params['input_res']])
        final[:imgs.shape[0]] = imgs

        meta_arr = {'raw_captions': caption, 'paths': rel_fp, 'dataset': self.dataset_name}
        data = {'video': final, 'text': caption, 'meta': meta_arr, 'target': sample['cls']}
        return data

    def _get_val(self, item):
        item = item % len(self.metadata)
        sample = self.metadata.iloc[item]
        video_fp, rel_fp = self._get_video_path(sample)
        caption = self._get_caption(sample)

        # construct label
        label = self.label[sample['id']]
        target = torch.IntTensor(157).zero_()
        for x in label:
            target[self._cls2int(x['class'])] = 1

        video_loading = self.video_params.get('loading', 'strict')
        frame_sample = 'rand'
        if self.split in ['val', 'test']:
            frame_sample = 'uniform'

        try:
            if os.path.isfile(video_fp):
                # only supported for read_frames_decord_online
                imgs, idxs = self.video_reader(video_path=video_fp, num_frames=self.video_params['num_frames'], sample=frame_sample)
            else:
                print(f"Warning: missing video file {video_fp}.")
                assert False
        except Exception as e:
            if video_loading == 'strict':
                raise ValueError(
                    f'Video loading failed for {video_fp}, video loading for this dataset is strict.') from e
            else:
                imgs = Image.new('RGB', (self.video_params['input_res'], self.video_params['input_res']), (0, 0, 0))
                imgs = transforms.ToTensor()(imgs).unsqueeze(0)

        if self.transforms is not None:
            if self.video_params['num_frames'] > 1:
                imgs = imgs.transpose(0, 1)  # [T, C, H, W] ---> [C, T, H, W]
                imgs = self.transforms(imgs)
                imgs = imgs.transpose(0, 1)  # recover
            else:
                imgs = self.transforms(imgs)

        final = torch.zeros([self.video_params['num_frames'], 3, self.video_params['input_res'],
                             self.video_params['input_res']])
        final[:imgs.shape[0]] = imgs

        meta_arr = {'raw_captions': caption, 'paths': rel_fp, 'dataset': self.dataset_name}
        data = {'video': final, 'text': caption, 'meta': meta_arr, 'target': target}
        return data

if __name__ == "__main__":
    kwargs = dict(
        dataset_name="CharadesEgo",
        text_params={
            "input": "text"
        },
        video_params={
        "input_res": 224,
        "num_frames": 4,
        "loading": "lax"
        },
        data_dir="/apdcephfs/private_qinghonglin/video_dataset/charades/CharadesEgo_v1_480",
        meta_dir="/apdcephfs/private_qinghonglin/video_dataset/charades/CharadesEgo",
        tsfms=init_video_transform_dict()['test'],
        reader='cv2_charades',
        split='val'
    )
    dataset = CharadesEgo(**kwargs)
    for i in range(100):
        item = dataset[i]
        print(item.keys())