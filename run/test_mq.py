import os
import sys
import tqdm
import argparse
import numpy as np
import transformers
from sacred import Experiment

import torch
import model.metric as module_metric
import data_loader.data_loader as module_data
from utils import state_dict_data_parallel_fix
from parse_config import ConfigParser

ex = Experiment('test')

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

    if str(config.resume) is not 'None':
        checkpoint = torch.load(config.resume)
        state_dict = checkpoint['state_dict']
        new_state_dict = state_dict_data_parallel_fix(state_dict, model.state_dict())
        model.load_state_dict(new_state_dict, strict=True)
    else:
        print('Using random weights')

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    print(len(data_loader))

    if not os.path.exists(args.save_feats):
        os.mkdir(args.save_feats)

    # extract clip features
    num_frame = config.config['data_loader']['args']['video_params']['num_frames']
    dim = config.config['arch']['args']['projection_dim']
    with torch.no_grad():
        for i, data in enumerate(tqdm.tqdm(data_loader)):
            # leave this for now since not doing anything on the gpu
            # pdb.set_trace()
            if os.path.exists(os.path.join(args.save_feats, data['meta']['clip_uid'][0]+'.pt')):
                print(f"{data['meta']['clip_uid']} is already.")
                continue
            # this implementation is cautious, we use 4f video-encoder to extract featurs of whole clip.
            f, c, h, w = data['video'].shape[1], data['video'].shape[2], data['video'].shape[3], data['video'].shape[4]
            data['video'] = data['video'][0][:(f // num_frame * num_frame), ]
            data['video'] = data['video'].reshape(-1, num_frame, c, h, w)

            data['video'] = data['video'].to(device)
            outs = torch.zeros(data['video'].shape[0], dim)

            batch = 4
            times = data['video'].shape[0] // batch
            for j in range(times):
                start = j*batch
                if (j+1) * batch > data['video'].shape[0]:
                    end = data['video'].shape[0]
                else:
                    end = (j+1)*batch

                outs[start:end,] = \
                    model.compute_video(data['video'][start:end,])

            torch.save(outs, os.path.join(args.save_feats, data['meta']['clip_uid'][0]+'.pt'))
            print(f"Saved {data['meta']['clip_uid']}. ")

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')

    args.add_argument('-r', '--resume',
                      default='results_egoclip/EgoClip_SE_1_mid_scale_th_v2/models/0502_01/checkpoint-epoch4.pth',
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-s', '--sliding_window_stride', default=-1, type=int,
                      help='test time temporal augmentation, repeat samples with different start times.')
    args.add_argument('--save_feats', default='dataset/ego4d/benchmark_splits/mq/egovlp',
                      help='path to store text & video feats, this is for saving embeddings if you want to do offline retrieval.')
    args.add_argument('--split', default='test', choices=['train', 'val', 'test'],
                      help='split to evaluate on.')
    args.add_argument('--batch_size', default=1, type=int,
                      help='size of batch')
    args.add_argument('-gpu', '--gpu', default=0, type=str,
                      help='indices of GPUs to enable (default: all)')
    config = ConfigParser(args, test=True, eval_mode='mq')
    # hack to get sliding into config
    args = args.parse_args()
    config._config['sliding_window_stride'] = args.sliding_window_stride
    ex.add_config(config.config)

    os.environ["CUDA_VISIBLE_DEVICES"] =  ""+str(args.gpu)
    ex.run()