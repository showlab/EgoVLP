import os
import sys
import json
import pandas as pd
sys.path.append('/apdcephfs/private_qinghonglin/video_codebase/EgoVLP/')

from base.base_dataset import TextVideoDataset
from data_loader.transforms import init_transform_dict, init_video_transform_dict

import torch
from PIL import Image
from torchvision import transforms

class EgoClip_EgoMCQ(TextVideoDataset):
    def _load_metadata(self):
        split_files = {
            'train': 'egoclip.csv',
            'val': 'egomcq.json',
            'test': 'egomcq.json'
        }
        target_split_fp = split_files[self.split]

        self.chunk_sec = 600  # Each segment is up to 600s
        self.noun_dim = 582  # num of nouns of ego4d taxonomy dictionary
        self.verb_dim = 118  # num of verbs of ego4d taxonomy dictionary

        if self.split == 'train':
            self.metadata = pd.read_csv(os.path.join(self.meta_dir, target_split_fp), sep='\t',error_bad_lines=False)
            self.frame_sample = 'rand'

            if self.neg_param:
                self.metadata['chunk_id'] = self.metadata['narration_time'] // self.neg_param
                self.metadata['chunk_id'] = self.metadata['chunk_id'].astype(str)
                self.metadata['segment_id'] = self.metadata['video_uid'] + '_' + self.metadata['chunk_id']

        elif self.split in ['val', 'test']:
            self.frame_sample = 'uniform'
            with open(os.path.join(self.meta_dir, target_split_fp), 'r') as load_f:
                self.metadata = json.load(load_f)

    def _get_video_path(self, sample):
        video_uid = sample['video_uid']
        video_start_sec = max(float(sample['clip_start']), 0)
        video_end_sec   = max(float(sample['clip_end']), 0)

        chunk_start_id = int(video_start_sec // self.chunk_sec)
        chunk_end_id = int(video_end_sec // self.chunk_sec)

        full_video_start_fp = os.path.join(self.data_dir, video_uid, str(chunk_start_id) + ".mp4")
        full_video_end_fp = os.path.join(self.data_dir, video_uid, str(chunk_end_id) + ".mp4")

        video_fp = [full_video_start_fp, full_video_end_fp]
        video_sec = [video_start_sec, video_end_sec]
        bound_sec = (chunk_start_id + 1) * self.chunk_sec
        return video_fp, video_sec, bound_sec

    def _get_video_frames(self, video_fp, video_sec, bound_sec):
        video_loading = self.video_params.get('loading', 'strict')
        try:
            if os.path.isfile(video_fp[0]) and os.path.isfile(video_fp[1]):
                imgs, idxs = self.video_reader(video_fp[0], video_fp[1], self.video_params['num_frames'], self.frame_sample,
                                               start_sec=video_sec[0], end_sec=video_sec[1], bound_sec=bound_sec)
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
        return final

    def _get_caption(self, sample):
        noun_vec = torch.zeros(self.noun_dim)
        verb_vec = torch.zeros(self.verb_dim)
        noun_idx = eval(sample['tag_noun'])
        verb_idx = eval(sample['tag_verb'])
        for i in noun_idx:
            noun_vec[i] = 1
        for i in verb_idx:
            verb_vec[i] = 1

        return sample['clip_text'], noun_vec, verb_vec

    def _get_train_item(self, item):
        item = item % len(self.metadata)
        sample = self.metadata.iloc[item]
        video_fp, video_sec, bound_sec = self._get_video_path(sample)
        caption, noun_vec, verb_vec = self._get_caption(sample)
        final = self._get_video_frames(video_fp, video_sec, bound_sec)

        # Scene-aware negative sampling
        if self.neg_param:
            # sample_neg = self.metadata[(self.metadata.video_uid==sample.video_uid)].sample(1).iloc[0] # variant of negative sample from same video
            sample_neg = self.metadata[self.metadata.segment_id==sample.segment_id].sample(1).iloc[0]
            video_fp_neg, video_sec_neg, bound_sec_neg = self._get_video_path(sample_neg)
            caption_neg, noun_vec_neg, verb_vec_neg = self._get_caption(sample_neg)
            final_neg = self._get_video_frames(video_fp_neg, video_sec_neg, bound_sec_neg)

        meta_arr = {'raw_captions': caption, 'paths': video_fp, 'dataset': self.dataset_name}
        if self.neg_param:
            return {'video': final, 'text': caption,
                    'video_neg': final_neg, 'text_neg': caption_neg,
                    'meta': meta_arr,
                    'noun_vec': noun_vec, 'verb_vec': verb_vec,
                    'noun_vec_neg': noun_vec_neg, 'verb_vec_neg': verb_vec_neg}
        else:
            return {'video': final, 'text': caption,
                'meta': meta_arr,
                'noun_vec': noun_vec, 'verb_vec': verb_vec}

    def _get_val_item(self, item):
        item = item % len(self.metadata)
        itemMCQ = self.metadata[str(item)]

        answerIndex = itemMCQ['answer']
        sampleQuery = itemMCQ['query']
        textQuery, _, _ = self._get_caption(sampleQuery)

        sampleOptions = itemMCQ['choices']
        num_options = len(sampleOptions)
        textOptions = []
        videoOptions = torch.zeros([num_options, self.video_params['num_frames'], 3, self.video_params['input_res'],
                             self.video_params['input_res']])

        for id, option in enumerate(sampleOptions):
            sampleOptioni = sampleOptions[option]
            video_fp, video_sec, bound_sec = self._get_video_path(sampleOptioni)
            caption, _, _ = self._get_caption(sampleOptioni)
            textOptions.append(caption)

            imgs = self._get_video_frames(video_fp, video_sec, bound_sec)
            videoOptions[id] = imgs

        type =  itemMCQ['types']    # 1 for inter; 2 for intra
        data = {'video': videoOptions, 'text': textQuery, 'text_ops':textOptions, 'correct': answerIndex, 'type': type}
        return data

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, item):
        if self.split == 'train':
            return self._get_train_item(item)
        elif self.split in ['val', 'test']:
            return self._get_val_item(item)

if __name__ == "__main__":
    kwargs = dict(
        dataset_name="EgoClip_dataset",
        text_params={
            "input": "text"
        },
        video_params={
        "input_res": 224,
        "num_frames": 4,
        "loading": "lax"
        },
        data_dir="/apdcephfs/private_qinghonglin/video_dataset/ego4d_256/data_chunked",
        meta_dir="/apdcephfs/private_qinghonglin/video_dataset/ego4d_toolbox/0_metadata/egovlp",
        tsfms=init_video_transform_dict()['test'],
        reader='cv2_egoclip',
        split='val',
        neg_param=60
    )
    dataset = EgoClip_EgoMCQ(**kwargs)
    for i in range(100):
        item = dataset[i]
        print(item.keys())