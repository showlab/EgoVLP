import os
import pdb
import sys
import json
import pandas as pd

from base.base_dataset import TextVideoDataset
from data_loader.transforms import init_transform_dict, init_video_transform_dict

class NaturalLanguageQueries(TextVideoDataset):
    def _load_metadata(self):
        split_files = {
            'train': 'nlq_train.json',
            'val': 'nlq_val.json',            # there is no test
            'test': 'nlq_test_unannotated.json'
        }
        # target_split_fp = split_files[self.split]
        assert self.subsample in ['video', 'text']

        self.metadata = pd.DataFrame(columns=['video_uid', 'clip_uid',
                                              'video_start_sec', 'video_end_sec',
                                              'query'])

        # for split in ['train', 'val']:
        target_split_fp = split_files[split]
        ann_file = os.path.join(self.meta_dir, target_split_fp)
        with open(ann_file) as f:
            anno_json = json.load(f)

        # forward clip features
        if self.subsample == 'video':
            for anno_video in anno_json["videos"]:
                for anno_clip in anno_video["clips"]:
                    clip_times = float(anno_clip["video_start_sec"]), float(anno_clip["video_end_sec"])
                    clip_duration = clip_times[1] - clip_times[0]
                    new = pd.DataFrame({
                        'video_uid': anno_video['video_uid'],
                        'clip_uid': anno_clip['clip_uid'],
                        'video_start_sec': clip_times[0],
                        'video_end_sec': clip_times[1]}, index=[1])
                    self.metadata = self.metadata.append(new, ignore_index=True)

        # forward text features
        elif self.subsample == 'text':
            for anno_video in anno_json["videos"]:
                for anno_clip in anno_video["clips"]:
                    clip_times = float(anno_clip["video_start_sec"]), float(anno_clip["video_end_sec"])
                    for anno in anno_clip['annotations']:
                        for query in anno["language_queries"]:
                            clip_duration = clip_times[1] - clip_times[0]
                            if 'query' not in query.keys():
                                continue
                            if query['query'] is None:
                                continue
                            new = pd.DataFrame({
                                'video_uid': anno_video['video_uid'],
                                'clip_uid': anno_clip['clip_uid'],
                                'video_start_sec': clip_times[0],
                                'video_end_sec': clip_times[1],
                                'query': query["query"],}, index=[1])
                            self.metadata = self.metadata.append(new, ignore_index=True)

        self.transforms = init_video_transform_dict()['test']

    def _get_video_path(self, sample):
        rel_video_fp = sample[0]
        full_video_fp = os.path.join(self.data_dir, rel_video_fp + '.mp4')
        return full_video_fp, rel_video_fp

    def _get_caption(self, sample):
        caption = sample['query']
        return caption

    def _get_video_feats(self, item):
        sample = self.metadata.iloc [item]
        video_fp, rel_fp = self._get_video_path(sample)

        fps = 1.87
        try:
            imgs, idxs = self.video_reader(video_fp, sample[2]*30, sample[3]*30,
                                               (sample[3]-sample[2]) * fps * self.video_params['num_frames'])
        except:
            print(f"Warning: missing video file {video_fp}.")

        if self.transforms is not None:
            imgs = imgs.transpose(0, 1)  # [T, C, H, W] ---> [C, T, H, W]
            imgs = self.transforms(imgs)
            imgs = imgs.transpose(0, 1)  # recover

        meta_arr = {'video_uid': sample[0], 'clip_uid': sample[1], 'data': video_fp}
        data = {'video': imgs, 'meta' : meta_arr}
        return data

    def _get_text_feats(self, item):
        sample = self.metadata.iloc [item]
        text = self._get_caption(sample)
        meta_arr = {'video_uid': sample[0], 'clip_uid': sample[1], 'dataset': self.dataset_name}
        data = {'text': text, 'meta' : meta_arr}
        return data

    def __getitem__(self, item):
        if self.subsample == 'video':
            return self._get_video_feats(item)
        if self.subsample == 'text':
            return self._get_text_feats(item)

if __name__ == "__main__":
    split = 'val'
    kwargs = dict(
        dataset_name="Ego4d_NLQ",
        text_params={
            "input": "text"
        },
        video_params={
        "input_res": 224,
        "num_frames": 4,
        "loading": "lax"
        },
        data_dir="dataset/ego4d_256/data",
        meta_dir="dataset/ego4d/benchmark_splits/nlq/",
        tsfms=init_video_transform_dict()['test'],
        reader='decord_start_end',
        subsample='text',
        split=split,
    )
    dataset = NaturalLanguageQueries(**kwargs)
    print(len(dataset))
    # for i in range(1000):
    #     item = dataset[i]
    #     # print(item.keys())
    #     print(item)