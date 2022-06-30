import pdb

from base import BaseDataLoaderExplicitSplit, BaseMultiDataLoader, \
    DistBaseDataLoaderExplicitSplit, MultiDistBaseDataLoaderExplicitSplit
from data_loader.ConceptualCaptions_dataset import ConceptualCaptions3M
from data_loader.WebVid_dataset import WebVid
from data_loader.EgoClip_EgoMCQ_dataset import EgoClip_EgoMCQ
from data_loader.EpicKitchens_MIR_dataset import MultiInstanceRetrieval
from data_loader.CharadesEgo_dataset import CharadesEgo
from data_loader.Ego4D_NLQ_dataset import NaturalLanguageQueries
from data_loader.Ego4D_MQ_dataset import MomentQueries
from data_loader.Ego4D_OSCC_dataset import ObjectStateChangeClassification
from data_loader.Ego4D_PNR_dataset import PNRTemporalLocalization
from data_loader.transforms import init_transform_dict, init_video_transform_dict

def dataset_loader(dataset_name,
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
                   neg_param=None):
    kwargs = dict(
        dataset_name=dataset_name,
        text_params=text_params,
        video_params=video_params,
        data_dir=data_dir,
        meta_dir=meta_dir,
        split=split,
        tsfms=tsfms,
        cut=cut,
        subsample=subsample,
        sliding_window_stride=sliding_window_stride,
        reader=reader,
        neg_param=neg_param,
    )

    # TODO: change to...
    #  dataset = globals()[dataset_name]
    #  ...is this safe / or just lazy?
    if dataset_name == "WebVid":
        dataset = WebVid(**kwargs)
    elif dataset_name == "ConceptualCaptions3M":
        dataset = ConceptualCaptions3M(**kwargs)
    elif dataset_name == "EgoClip":
        dataset = EgoClip_EgoMCQ(**kwargs)

    elif dataset_name == "EpicKitchens_MIR":
        dataset = MultiInstanceRetrieval(**kwargs)
    elif dataset_name == "CharadesEgo":
        dataset = CharadesEgo(**kwargs)
    elif dataset_name == "Ego4D_OSCC":
        dataset = ObjectStateChangeClassification(**kwargs)
    elif dataset_name == "Ego4D_PNR":
        dataset = PNRTemporalLocalization(**kwargs)
    elif dataset_name == "Ego4D_NLQ":
        dataset = NaturalLanguageQueries(**kwargs)
    elif dataset_name == "Ego4D_MQ":
        dataset = MomentQueries(**kwargs)
    else:
        raise NotImplementedError(f"Dataset: {dataset_name} not found.")

    return dataset


class TextVideoDataLoader(BaseDataLoaderExplicitSplit):
    def __init__(self,
                 dataset_name,
                 text_params,
                 video_params,
                 data_dir,
                 meta_dir=None,
                 split='train',
                 tsfm_params=None,
                 tsfm_split=None,
                 cut=None,
                 subsample=1,
                 sliding_window_stride=-1,
                 reader='decord',
                 neg_param=None,
                 batch_size=1,
                 num_workers=1,
                 shuffle=True):
        if tsfm_params is None:
            tsfm_params = {}
        if video_params['num_frames'] > 1:
            # video data can not do flip, crop aug
            tsfm_dict = init_video_transform_dict(**tsfm_params)
        else:
            tsfm_dict = init_transform_dict(**tsfm_params)
        if tsfm_split is None:
            tsfm_split = split
        tsfm = tsfm_dict[tsfm_split]
        dataset = dataset_loader(dataset_name, text_params, video_params, data_dir, meta_dir, split, tsfm, cut,
                                 subsample, sliding_window_stride, reader, neg_param)

        super().__init__(dataset, batch_size, shuffle, num_workers)
        self.dataset_name = dataset_name

class DistTextVideoDataLoader(DistBaseDataLoaderExplicitSplit):
    def __init__(self,
                 dataset_name,
                 text_params,
                 video_params,
                 data_dir,
                 meta_dir=None,
                 split='train',
                 tsfm_params=None,
                 tsfm_split=None,
                 cut=None,
                 subsample=1,
                 sliding_window_stride=-1,
                 reader='cv2',
                 neg_param=None,
                 batch_size=1,
                 num_workers=1,
                 shuffle=True):
        if tsfm_params is None:
            tsfm_params = {}
        if video_params['num_frames'] > 1:
            # video data can not do flip, crop aug
            tsfm_dict = init_video_transform_dict(**tsfm_params)
        else:
            tsfm_dict = init_transform_dict(**tsfm_params)

        # Updated
        if tsfm_split is None:
            tsfm_split = split
        tsfm = tsfm_dict[tsfm_split]

        dataset = dataset_loader(dataset_name, text_params, video_params, data_dir, meta_dir, split, tsfm, cut,
                                 subsample, sliding_window_stride, reader, neg_param)
        super().__init__(dataset, batch_size, shuffle, num_workers)
        self.dataset_name = dataset_name

class MultiDistTextVideoDataLoader(MultiDistBaseDataLoaderExplicitSplit):
    def __init__(self,
                 args,
                 dataset_name,
                 text_params,
                 video_params,
                 data_dir,
                 meta_dir=None,
                 split='train',
                 tsfm_params=None,
                 tsfm_split=None,
                 cut=None,
                 subsample=1,
                 sliding_window_stride=-1,
                 reader='cv2',
                 neg_param=None,
                 batch_size=1,
                 num_workers=1,
                 shuffle=True):
        if tsfm_params is None:
            tsfm_params = {}
        if video_params['num_frames'] > 1:
            # video data can not do flip, crop aug
            tsfm_dict = init_video_transform_dict(**tsfm_params)
        else:
            tsfm_dict = init_transform_dict(**tsfm_params)

        if tsfm_split is None:
            tsfm_split = split
        tsfm = tsfm_dict[tsfm_split]

        dataset = dataset_loader(dataset_name, text_params, video_params, data_dir, meta_dir, split, tsfm, cut,
                                 subsample, sliding_window_stride, reader, neg_param)
        super().__init__(args, dataset, batch_size, shuffle, num_workers)
        self.dataset_name = dataset_name

class TextVideoMultiDataLoader(BaseMultiDataLoader):
    # TODO: figure out neat way to have N data_loaders
    # TODO: also add N weighted sampler
    def __init__(self, data_loader1, data_loader2):
        # get class from "type" in dict
        dls_cfg = [data_loader1, data_loader2]
        dls = []
        for dcfg in dls_cfg:
            dl = globals()[dcfg['type']](**dcfg['args'])
            dls.append(dl)
        super().__init__(dls)