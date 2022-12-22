import os
import sys
import tqdm
import pickle
import argparse
import numpy as np
import pandas as pd
import transformers
from csv import reader
from sacred import Experiment

import torch
from model.model import sim_matrix
import model.metric as module_metric
import data_loader.data_loader as module_data
from utils import state_dict_data_parallel_fix
from parse_config import ConfigParser

ex = Experiment('test')

def charades_map(submission_array, gt_array):
    """
    Approximate version of the charades evaluation function
    For precise numbers, use the submission file with the official matlab script
    """
    fix = submission_array.copy()
    empty = np.sum(gt_array, axis=1) == 0
    fix[empty, :] = np.NINF
    return map(fix, gt_array)

def map(submission_array, gt_array):
    """ Returns mAP, weighted mAP, and AP array """
    m_aps = []
    n_classes = submission_array.shape[1]
    for oc_i in range(n_classes):
        sorted_idxs = np.argsort(-submission_array[:, oc_i])
        tp = gt_array[:, oc_i][sorted_idxs] == 1
        fp = np.invert(tp)
        n_pos = tp.sum()
        if n_pos < 0.1:
            m_aps.append(float('nan'))
            continue
        fp.sum()
        f_pcs = np.cumsum(fp)
        t_pcs = np.cumsum(tp)
        prec = t_pcs / (f_pcs+t_pcs).astype(float)
        avg_prec = 0
        for i in range(submission_array.shape[0]):
            if tp[i]:
                avg_prec += prec[i]
        m_aps.append(avg_prec / n_pos.astype(float))
    m_aps = np.array(m_aps)
    m_ap = np.mean(m_aps)
    w_ap = (m_aps * gt_array.sum(axis=0) / gt_array.sum().sum().astype(float))
    return m_ap, w_ap, m_aps

@ex.main
def run():
    # setup data_loader instances
    config._config['data_loader']['type'] = 'TextVideoDataLoader'
    config._config['data_loader']['args']['split'] = args.split
    config._config['data_loader']['args']['tsfm_split'] = 'test'  # set transform to test split to remove augmentations
    config._config['data_loader']['args']['shuffle'] = False
    config._config['data_loader']['args']['batch_size'] = args.batch_size
    config._config['data_loader']['args']['sliding_window_stride'] = args.sliding_window_stride

    data_loader = config.initialize('data_loader', module_data)

    tokenizer = transformers.AutoTokenizer.from_pretrained(config['arch']['args']['text_params']['model'])

    # build model architecture
    import model.model as module_arch
    model = config.initialize('arch', module_arch)

    # get function handles of loss and metrics
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    # logger.info('Loading checkpoint: {} ...'.format(config.resume))

    if config.resume is not None:
        checkpoint = torch.load(config.resume)
        state_dict = checkpoint['state_dict']
        new_state_dict = state_dict_data_parallel_fix(state_dict, model.state_dict())
        model.load_state_dict(new_state_dict, strict=True)
    else:
        print('Using random weights')

    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    # construct set of sentences.
    cls_arr = []
    with open('dataset/charades/CharadesEgo/Charades_v1_classes.txt', 'r') as charades:
        csv_reader = list(reader(charades))
    for line in csv_reader:
        cls_arr.append(line[0][5:])

    meta_arr, gt_arr = [], []
    text_embed_arr = []
    vid_embed_arr = []
    print(len(data_loader))
    with torch.no_grad():
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        data_cls = tokenizer(cls_arr, return_tensors='pt', padding=True, truncation=True)
        data_cls = {key: val.cuda() for key, val in data_cls.items()}

        dict_cls = {'text':data_cls, 'video': torch.Tensor(1,4,3,224,224)}
        text_embed, _ = model(dict_cls, return_embeds=True)
        text_embeds = text_embed.cpu().detach()

        for i, data in tqdm.tqdm(enumerate(data_loader)):
            meta_arr.append(data['meta'])

            data['text'] = tokenizer(data['text'], return_tensors='pt', padding=True, truncation=True)
            data['text'] = {key: val.cuda() for key, val in data['text'].items()}

            if isinstance(data['video'], list):
                data['video'] = [x.to(device) for x in data['video']]
            else:
                data['video'] = data['video'].to(device)

            try:    # for temporal
                _, vid_embed, _ = model(data, return_embeds=True)
            except:
                _, vid_embed = model(data, return_embeds=True)

            vid_embed_arr.append(vid_embed.cpu().detach())
            gt_arr.append(data['target'].cpu().detach())

    vid_embeds = torch.cat(vid_embed_arr)
    gt_embeds = torch.cat(gt_arr)

    sims = sim_matrix(text_embeds, vid_embeds)
    sims = sims.numpy().T
    gt_embeds = gt_embeds.numpy()
    m_ap, w_ap, m_aps = charades_map(np.vstack(sims), np.vstack(gt_embeds))
    print('mAP: {:.3f}'.format(m_ap * 100))

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')

    args.add_argument('-r', '--resume',
                      default = 'results/EgoClip_Charades_egonce_16f/models/0513_00/checkpoint-epoch1.pth',
                      # default='results_egoclip/EgoClip_M_EgoNCE_N_V_Neg_Seg_60/models/0509_00/checkpoint-epoch1.pth',
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-gpu', '--gpu', default=0, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-s', '--sliding_window_stride', default=-1, type=int,
                      help='test time temporal augmentation, repeat samples with different start times.')
    args.add_argument('--save_feats', default=None,
                      help='path to store text & video feats, this is for saving embeddings if you want to do offline retrieval.')
    args.add_argument('--split', default='test', choices=['train', 'val', 'test'],
                      help='split to evaluate on.')
    args.add_argument('--batch_size', default=1, type=int,
                      help='size of batch')
    config = ConfigParser(args, test=True, eval_mode='charades')

    # hack to get sliding into config
    args = args.parse_args()
    config._config['sliding_window_stride'] = args.sliding_window_stride
    ex.add_config(config.config)

    os.environ["CUDA_VISIBLE_DEVICES"] =  ""+str(args.gpu)


    ex.run()
