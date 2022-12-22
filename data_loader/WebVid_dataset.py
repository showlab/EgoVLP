import os

import pandas as pd

from base.base_dataset import TextVideoDataset


class WebVid(TextVideoDataset):
    """
    WebVid Dataset.
    Assumes webvid data is structured as follows.
    Webvid/
        videos/
            000001_000050/      ($page_dir)
                1.mp4           (videoid.mp4)
                ...
                5000.mp4
            ...
    """
    def _load_metadata(self):
        metadata_dir = 'dataset/meta_data'
        split_files = {
            'train': 'webvid_training.csv',
            'val': 'webvid_validation.csv',            # there is no test
            'test': 'webvid_validation.csv',  # there is no test
        }
        target_split_fp = split_files[self.split]
        metadata = pd.read_csv(os.path.join(metadata_dir, target_split_fp), sep='\t')
        if self.subsample < 1:
            metadata = metadata.sample(frac=self.subsample)
        # elif self.split == 'val':
        #     metadata = metadata.sample(1000, random_state=0)  # 15k val is unnecessarily large, downsample.

        # metadata = metadata.sample(3000, random_state=0)
        self.metadata = metadata

    def _get_video_path(self, sample):
        # print(sample)
        rel_video_fp = sample[1] + '.mp4'
        # print(rel_video_fp)
        if self.split in ['train', 'val']:
            full_video_fp = os.path.join(self.data_dir, self.split, rel_video_fp)
        elif self.split in ['test']:
            full_video_fp = os.path.join(self.data_dir, 'val', rel_video_fp)
        return full_video_fp, rel_video_fp

    def _get_caption(self, sample):
        # print(sample[0])
        return sample[0]