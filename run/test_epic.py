import os
import sys
import tqdm
import pickle
import argparse
import numpy as np
import pandas as pd
import transformers
from sacred import Experiment

import torch
import torch.nn.functional as F
from utils import nDCG, mAP
import model.metric as module_metric
import data_loader.data_loader as module_data
from utils import state_dict_data_parallel_fix
from parse_config import ConfigParser

ex = Experiment('test')

def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt.detach().numpy()

def sim_matrix_mm(a, b):
    sim_mt = torch.mm(a, b.transpose(0, 1))
    return sim_mt.detach().numpy()

def softmax_numpy(sim, dim=0):
    sim = torch.Tensor(sim)
    sim = F.softmax(sim, dim=dim)
    return sim.numpy()

def initialise_nDCG_values(relevancy_matrix):
    vis_k_counts = nDCG.calculate_k_counts(relevancy_matrix)
    txt_k_counts = nDCG.calculate_k_counts(relevancy_matrix.T)

    vis_IDCG = nDCG.calculate_IDCG(relevancy_matrix, vis_k_counts)
    txt_IDCG = nDCG.calculate_IDCG(relevancy_matrix.T, txt_k_counts)

    k_counts_dict = {'v': vis_k_counts, 't': txt_k_counts}
    IDCG_dict = {'v': vis_IDCG, 't': txt_IDCG}

    return IDCG_dict, k_counts_dict

def initialise_jpose_nDCG_values(relevancy_matrix):
    action_IDCG, action_k_values = initialise_nDCG_values(relevancy_matrix)

    dataset = {}
    dataset['action'] = {}
    dataset['action']['IDCG'] = action_IDCG
    dataset['action']['k_values'] = action_k_values
    return dataset

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

    # build tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(config['arch']['args']['text_params']['model'],
                                                               TOKENIZERS_PARALLELISM=False)
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

    meta_arr = []
    text_embed_arr = []
    vid_embed_arr = []
    print(len(data_loader))
    with torch.no_grad():
        # for i, data in enumerate(data_loader):
        for i, data in tqdm.tqdm(enumerate(data_loader)):
            # leave this for now since not doing anything on the gpu
            if tokenizer is not None:
                data['text'] = tokenizer(data['text'], return_tensors='pt', padding=True, truncation=True)
            data['text'] = {key: val.cuda() for key, val in data['text'].items()}
            if isinstance(data['video'], list):
                data['video'] = [x.to(device) for x in data['video']]
            else:
                data['video'] = data['video'].to(device)

            text_embed, vid_embed = model(data, return_embeds=True)
            vid_embed_arr.append(vid_embed.cpu().detach())
            text_embed_arr.append(text_embed.cpu().detach())

    vid_embeds = torch.cat(vid_embed_arr)
    text_embeds = torch.cat(text_embed_arr)

    # considered unique narrations for evaluation of EPIC
    path_dataframes = 'dataset/epic-kitchens/epic-kitchens-100-annotations-master/retrieval_annotations'
    video_id=pd.read_csv(os.path.join(path_dataframes , "EPIC_100_retrieval_test.csv")).values[:,0]
    text_id=pd.read_csv(os.path.join(path_dataframes , "EPIC_100_retrieval_test_sentence.csv")).values[:,0]

    indexes=[]
    for elem in text_id:
        indexes.append(video_id.tolist().index(elem))

    path_relevancy = "dataset/epic-kitchens/epic-kitchens-100-annotations-master/retrieval_annotations/relevancy/caption_relevancy_EPIC_100_retrieval_test.pkl"
    pkl_file = open(path_relevancy, 'rb')
    relevancy = pickle.load(pkl_file)

    if not args.dual_softmax:
        similarity_matrix = (sim_matrix(text_embeds, vid_embeds) + 1) / 2
    else:
        # dual-softmax for better similarity scale
        similarity_matrix = sim_matrix_mm(text_embeds, vid_embeds)
        similarity_matrix = softmax_numpy(similarity_matrix / 500, dim=1) * similarity_matrix
        similarity_matrix = softmax_numpy(similarity_matrix, dim=0)
    similarity_matrix = similarity_matrix.T[:, indexes]

    dataset = initialise_jpose_nDCG_values(relevancy)
    vis_nDCG = nDCG.calculate_nDCG(similarity_matrix,
                                   relevancy, dataset['action']['k_values']['v'],
                                   IDCG=dataset['action']['IDCG']['v'])
    txt_nDCG = nDCG.calculate_nDCG(similarity_matrix.T,
                                   relevancy.T, dataset['action']['k_values']['t'],
                                   IDCG=dataset['action']['IDCG']['t'])
    print('nDCG: VT:{:.3f} TV:{:.3f} AVG:{:.3f}'.format(vis_nDCG, txt_nDCG, (vis_nDCG + txt_nDCG) / 2))

    vis_mAP = mAP.calculate_mAP(similarity_matrix, relevancy)
    txt_mAP = mAP.calculate_mAP(similarity_matrix.T, relevancy.T)
    print('mAP: VT:{:.3f} TV:{:.3f} AVG:{:.3f}'.format(vis_mAP, txt_mAP, (vis_mAP + txt_mAP) / 2))

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')

    args.add_argument('-r', '--resume',
                      # default='results_egoclip/EgoClip_M_EgoNCE_N_V_Neg_Seg_60/models/0510_10/checkpoint-epoch1.pth'
                      # default='results/EgoClip_EPIC_16f_best_rel_01_margin_02/models/0512_01/checkpoint-epoch100.pth',
                      default='results/EgoClip_EPIC_16f_best_rel_01_margin_04_weight/models/0512_02/checkpoint-epoch100.pth',
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-gpu', '--gpu', default=0, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-s', '--sliding_window_stride', default=-1, type=int,
                      help='test time temporal augmentation, repeat samples with different start times.')
    args.add_argument('--save_feats', default='',
                      help='path to store text & video feats, this is for saving embeddings if you want to do offline retrieval.')
    args.add_argument('--split', default='test', choices=['train', 'val', 'test'],
                      help='split to evaluate on.')
    args.add_argument('--batch_size', default=1, type=int,
                      help='size of batch')
    args.add_argument('--dual_softmax', default=True, type=bool,
                      help='whether adopt dual-softmax for inference')

    config = ConfigParser(args, test=True, eval_mode='epic')

    # hack to get sliding into config
    args = args.parse_args()
    config._config['sliding_window_stride'] = args.sliding_window_stride
    ex.add_config(config.config)

    os.environ["CUDA_VISIBLE_DEVICES"] =  ""+str(args.gpu)

    ex.run()