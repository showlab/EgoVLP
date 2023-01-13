import os
import pdb

import tqdm
import random
from abc import abstractmethod

import av
import cv2
import decord
import ffmpeg
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, get_worker_info
from torchvision import transforms


class TextVideoDataset(Dataset):
    def __init__(self,
                 dataset_name,
                 text_params,
                 video_params,
                 data_dir,
                 meta_dir=None,
                 split='train',
                 tsfms=None,
                 cut=None,
                 subsample=1,
                 sliding_window_stride=-1,
                 reader='decord',
                 neg_param=None,
                 ):
        self.dataset_name = dataset_name
        self.text_params = text_params
        self.video_params = video_params
        # check for environment variables
        self.data_dir = os.path.expandvars(data_dir)
        if meta_dir is not None:
            self.meta_dir = os.path.expandvars(meta_dir)
        else:
            self.meta_dir = self.data_dir
        self.split = split
        self.transforms = tsfms
        self.cut = cut
        self.subsample = subsample
        self.sliding_window_stride = sliding_window_stride
        self.video_reader = video_reader[reader]
        self.label_type = 'caption'
        self.neg_param = neg_param
        self._load_metadata()
        if self.sliding_window_stride != -1:
            if self.split != 'test':
                raise ValueError('Fixing frame sampling is for test time only. can remove but...')
            self._fix_temporal_samples()

    @abstractmethod
    def _load_metadata(self):
        raise NotImplementedError("Metadata loading must be implemented by subclass")

    @abstractmethod
    def _get_video_path(self, sample):
        raise NotImplementedError("Get video path function must be implemented by subclass")

    def _get_caption(self, sample):
        raise NotImplementedError("Get caption function must be implemented by subclass")

    def _get_video_lens(self):
        vlen_li = []
        for idx, row in self.metadata.iterrows():
            video_path = self._get_video_path(row)[0]
            vlen_li.append(get_video_len(video_path))

        return vlen_li

    def _fix_temporal_samples(self):
        self.metadata['vlen'] = self._get_video_lens()
        self.metadata['frame_intervals'] = self.metadata['vlen'].apply(
            lambda x: np.linspace(start=0, stop=x, num=min(x, self.video_params['num_frames']) + 1).astype(int))
        self.metadata['fix_start'] = self.metadata['frame_intervals'].apply(
            lambda x: np.arange(0, int(x[-1] / len(x - 1)), self.sliding_window_stride)
        )
        self.metadata = self.metadata.explode('fix_start')

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, item):
        item = item % len(self.metadata)
        sample = self.metadata.iloc[item]
        video_fp, rel_fp = self._get_video_path(sample)
        caption = self._get_caption(sample)

        video_loading = self.video_params.get('loading', 'strict')
        frame_sample = 'rand'
        fix_start = None
        if self.split == 'test':
            frame_sample = 'uniform'
        if self.sliding_window_stride != -1:
            fix_start = sample['fix_start']

        try:
            if os.path.isfile(video_fp):
                imgs, idxs = self.video_reader(video_fp, self.video_params['num_frames'], frame_sample,
                                               fix_start=fix_start)
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

        # if self.transforms is not None:
        #     imgs = self.transforms(imgs)

        if self.transforms is not None:
            if self.video_params['num_frames'] > 1:
                imgs = imgs.transpose(0, 1)  # [T, C, H, W] ---> [C, T, H, W]
                imgs = self.transforms(imgs)
                imgs = imgs.transpose(0, 1)  # recover
            else:
                imgs = self.transforms(imgs)

        # if self.transforms is not None:
        #     # for video aug if T>1 else just do image aug
        #     if imgs.size(0) > 1:
        #         # added by Mr. YAN
        #         imgs = imgs.transpose(0, 1) # [T, C, H, W] ---> [C, T, H, W]
        #         imgs = self.transforms(imgs)
        #         imgs = imgs.transpose(0, 1) # recover
        #     else:
        #         imgs = self.transforms(imgs)    # t

        final = torch.zeros([self.video_params['num_frames'], 3, self.video_params['input_res'],
                             self.video_params['input_res']])
        final[:imgs.shape[0]] = imgs

        meta_arr = {'raw_captions': caption, 'paths': rel_fp, 'dataset': self.dataset_name}
        data = {'video': final, 'text': caption, 'meta': meta_arr}
        return data


class TextImageDataset(TextVideoDataset):

    def __getitem__(self, item):
        item = item % len(self.metadata)
        sample = self.metadata.iloc[item]
        video_fp, rel_fp = self._get_video_path(sample)
        caption = self._get_caption(sample)

        video_loading = self.video_params.get('loading', 'strict')

        try:
            img = Image.open(video_fp).convert("RGB")
        except:
            if video_loading == 'strict':
                raise ValueError(f'Image loading failed for {video_fp}, image loading for this dataset is strict.')
            else:
                img = Image.new('RGB', (self.video_params['input_res'], self.video_params['input_res']), (0, 0, 0))

        # convert to tensor because video transforms don't, expand such that its a 1-frame video.
        img = transforms.ToTensor()(img).unsqueeze(0)
        if self.transforms is not None:
            img = self.transforms(img)
        meta_arr = {'raw_captions': caption, 'paths': rel_fp, 'dataset': self.dataset_name}
        data = {'video': img, 'text': caption, 'meta': meta_arr}
        return data


def sample_frames(num_frames, vlen, sample='rand', fix_start=None):
    acc_samples = min(num_frames, vlen)
    intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
    ranges = []
    for idx, interv in enumerate(intervals[:-1]):
        ranges.append((interv, intervals[idx + 1] - 1))
    if sample == 'rand':
        frame_idxs = [random.choice(range(x[0], x[1])) for x in ranges]
    elif fix_start is not None:
        frame_idxs = [x[0] + fix_start for x in ranges]
    elif sample == 'uniform':
        frame_idxs = [(x[0] + x[1]) // 2 for x in ranges]
    else:
        raise NotImplementedError

    return frame_idxs

def sample_frames_clips(start, end, vlen, acc_samples):
    start = max(0, start)
    end = min(vlen, end)

    intervals = np.linspace(start=start, stop=end, num=int(acc_samples) + 1).astype(int)
    ranges = []
    for idx, interv in enumerate(intervals[:-1]):
        ranges.append((interv, intervals[idx + 1] - 1))
        frame_idxs = [(x[0] + x[1]) // 2 for x in ranges
                      ]
    return frame_idxs

def sample_frames_start_end(num_frames, start, end, sample='rand', fix_start=None):
    acc_samples = min(num_frames, end)
    intervals = np.linspace(start=start, stop=end, num=acc_samples + 1).astype(int)
    ranges = []
    for idx, interv in enumerate(intervals[:-1]):
        ranges.append((interv, intervals[idx + 1] - 1))
    if sample == 'rand':
        frame_idxs = [random.choice(range(x[0], x[1])) for x in ranges]
    elif fix_start is not None:
        frame_idxs = [x[0] + fix_start for x in ranges]
    elif sample == 'uniform':
        frame_idxs = [(x[0] + x[1]) // 2 for x in ranges]
    else:
        raise NotImplementedError

    return frame_idxs

def read_frames_cv2(video_path, num_frames, sample='rand', fix_start=None):
    cap = cv2.VideoCapture(video_path)
    assert (cap.isOpened())
    vlen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # get indexes of sampled frames
    frame_idxs = sample_frames(num_frames, vlen, sample=sample, fix_start=fix_start)
    frames = []
    success_idxs = []
    for index in frame_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index - 1)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = torch.from_numpy(frame)
            # (H x W x C) to (C x H x W)
            frame = frame.permute(2, 0, 1)
            frames.append(frame)
            success_idxs.append(index)
        else:
            pass
            # print(frame_idxs, ' fail ', index, f'  (vlen {vlen})')

    frames = torch.stack(frames).float() / 255
    cap.release()
    return frames, success_idxs

def read_frames_cv2_egoclip(video_path_1, video_path_2, num_frames, sample,
                            start_sec, end_sec, bound_sec):
    if video_path_1 == video_path_2:
        cap1 = cv2.VideoCapture(video_path_1)
        cap2 = cap1
        vlen1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
        vlen2 = vlen1
        assert (cap1.isOpened())
    else:   # some clips may span two segments.
        cap1 = cv2.VideoCapture(video_path_1)
        cap2 = cv2.VideoCapture(video_path_2)
        vlen1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
        vlen2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
        assert (cap1.isOpened())
        assert (cap2.isOpened())

    # get indexes of sampled frames
    start_f = max(0, int(start_sec * 30))
    end_f = max(0, int(end_sec * 30))
    bound_f = int(bound_sec * 30)
    frame_idxs = sample_frames_start_end(num_frames, start_f, end_f, sample=sample)

    frames = []
    success_idxs = []
    for index in frame_idxs:
        _index = index % (600 * 30)
        if index > bound_f: # frame from the last video
            _index = min(_index, vlen2)
            cap2.set(cv2.CAP_PROP_POS_FRAMES, _index - 1)
            ret, frame = cap2.read()
        else:   # frame from the first video
            _index = min(_index, vlen1)
            cap1.set(cv2.CAP_PROP_POS_FRAMES, _index - 1)
            ret, frame = cap1.read()

        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = torch.from_numpy(frame)
            # (H x W x C) to (C x H x W)
            frame = frame.permute(2, 0, 1)
            frames.append(frame)
            success_idxs.append(index)
        else:
            pass

    while len(frames) < num_frames: # complete the frame
        frames.append(frames[-1])

    frames = torch.stack(frames).float() / 255
    cap1.release()
    cap2.release()
    return frames, success_idxs

def read_frames_cv2_epic(video_path, start_frame, stop_frame, num_frames, sample='rand', fix_start=None):
    # get indexes of sampled frames
    frame_idxs = sample_frames_start_end(num_frames, start_frame, stop_frame, sample=sample, fix_start=fix_start)
    frames = []
    success_idxs = []
    for index in frame_idxs:
        img_name = 'frame_' + str(index).zfill(10) + '.jpg'
        frame = cv2.imread(os.path.join(video_path, img_name))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame = torch.from_numpy(frame)
        # (H x W x C) to (C x H x W)
        frame = frame.permute(2, 0, 1)
        frames.append(frame)
        success_idxs.append(index)

    frames = torch.stack(frames).float() / 255
    return frames, success_idxs

def read_frames_cv2_charades(video_path, num_frames, sample, start_sec=None, end_sec=None):
    cap = cv2.VideoCapture(video_path)
    assert (cap.isOpened())
    vlen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(5)

    # get indexes of sampled frames
    if not start_sec and not end_sec:
        frame_idxs = sample_frames(num_frames, vlen, sample=sample)
    else:
        start_f = max(0, int(start_sec * fps))
        end_f = min(int(end_sec * fps), vlen)
        frame_idxs = sample_frames_start_end(num_frames, start_f, end_f, sample=sample)

    frames = []
    success_idxs = []
    for index in frame_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index - 1)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = torch.from_numpy(frame)
            # (H x W x C) to (C x H x W)
            frame = frame.permute(2, 0, 1)
            frames.append(frame)
            success_idxs.append(index)
        else:
            pass

    frames = torch.stack(frames).float() / 255
    cap.release()
    return frames, success_idxs

def read_frames_av(video_path, num_frames, sample='rand', fix_start=None):
    reader = av.open(video_path)
    try:
        frames = []
        frames = [torch.from_numpy(f.to_rgb().to_ndarray()) for f in reader.decode(video=0)]
    except (RuntimeError, ZeroDivisionError) as exception:
        print('{}: WEBM reader cannot open {}. Empty '
              'list returned.'.format(type(exception).__name__, video_path))
    vlen = len(frames)
    frame_idxs = sample_frames(num_frames, vlen, sample=sample, fix_start=fix_start)
    frames = torch.stack([frames[idx] for idx in frame_idxs]).float() / 255
    frames = frames.permute(0, 3, 1, 2)
    return frames, frame_idxs

decord.bridge.set_bridge("torch")

def read_frames_decord(video_path, num_frames, sample='rand', fix_start=None):
    video_reader = decord.VideoReader(video_path, num_threads=1)
    vlen = len(video_reader)
    frame_idxs = sample_frames(num_frames, vlen, sample=sample, fix_start=fix_start)
    video_reader.skip_frames(1)
    frames = video_reader.get_batch(frame_idxs)

    frames = frames.float() / 255
    frames = frames.permute(0, 3, 1, 2)
    return frames, frame_idxs

def read_frames_decord_start_end(video_path, start, end, num_frames):
    video_reader = decord.VideoReader(video_path, num_threads=1)
    vlen = len(video_reader)
    frame_idxs = sample_frames_clips(start, end, vlen, num_frames + 1)
    video_reader.skip_frames(1)
    frames = video_reader.get_batch(frame_idxs)

    frames = frames.float() / 255
    frames = frames.permute(0, 3, 1, 2)
    return frames, frame_idxs

def get_video_len(video_path):
    cap = cv2.VideoCapture(video_path)
    if not (cap.isOpened()):
        return False
    vlen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return vlen

video_reader = {
    'av': read_frames_av,
    'cv2': read_frames_cv2,
    'cv2_epic': read_frames_cv2_epic,
    'cv2_charades': read_frames_cv2_charades,
    'cv2_egoclip': read_frames_cv2_egoclip,
    'decord': read_frames_decord,
    'decord_start_end': read_frames_decord_start_end,
}
