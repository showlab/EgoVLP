import os
import sys
import argparse
import collections
import transformers
from sacred import Experiment
sys.path.append("/apdcephfs/share_1367250/qinghonglin/video_codebase/EgoVLP")

import torch
import data_loader.data_loader as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
import utils.visualizer as module_vis
from parse_config import ConfigParser
from trainer import Multi_Trainer_dist_PNR
from utils.util import replace_nested_dict_item
from tensorboardX import SummaryWriter

ex = Experiment('train')

@ex.main
def run():
    logger = config.get_logger('train')
    os.environ['TOKENIZERS_PARALLELISM'] = "false"
    # TODO: improve Create identity (do nothing) visualiser?
    if config['visualizer']['type'] != "":
        visualizer = config.initialize(
            name='visualizer',
            module=module_vis,
            exp_name=config['name'],
            web_dir=config._web_log_dir
        )
    else:
        visualizer = None

    torch.cuda.set_device(args.local_rank)

    # if args.world_size > 1:
    if args.master_address != 9339:
        print("DistributedDataParallel")
        # DistributedDataParallel
        torch.distributed.init_process_group(backend='nccl',
                                                 init_method='tcp://{}:{}'.format(
                                                 args.master_address, args.master_port),
                                             rank=args.rank, world_size=args.world_size)
    device = torch.device(f'cuda:{args.local_rank}')

    if args.rank == 0:
        print('world_size', args.world_size, flush=True)
        print('local_rank: ', args.local_rank, flush=True)

    # build tokenizer
    # import pdb;
    # pdb.set_trace()
    tokenizer = None

    # setup data_loader instances
    data_loader, valid_data_loader = init_dataloaders(config, module_data)
    if args.rank == 0:
        print('Train dataset: ', [x.n_samples for x in data_loader], ' samples')
        print('Val dataset: ', [x.n_samples for x in valid_data_loader], ' samples')
    # build model architecture, then print to console

    model = config.initialize('arch', module_arch)

    if args.local_rank == 0:
        logger.info(model)

    # get function handles of loss and metrics
    loss = config.initialize(name="loss", module=module_loss)
    metrics = [getattr(module_metric, met) for met in config['metrics']]
    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.initialize('optimizer', transformers, trainable_params)
    lr_scheduler = None
    if 'lr_scheduler' in config._config:
        if hasattr(transformers, config._config['lr_scheduler']['type']):
            lr_scheduler = config.initialize('lr_scheduler', transformers, optimizer)
        else:
            print('lr scheduler not found')
    if config['trainer']['neptune']:
        writer = ex
    else:
        writer = None

    if args.rank == 0:
        writer = SummaryWriter(log_dir=str(config.tf_dir))

    trainer = Multi_Trainer_dist_PNR(args, model, loss, metrics, optimizer,
                      config=config,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler,
                      visualizer=visualizer,
                      writer=writer,
                      tokenizer=tokenizer,
                      max_samples_per_epoch=config['trainer']['max_samples_per_epoch'])

    trainer.train()


def init_dataloaders(config, module_data):
    """
    We need a way to change split from 'train' to 'val'.
    """
    if "type" in config["data_loader"] and "args" in config["data_loader"]:
        # then its a single dataloader
        data_loader = [config.initialize("data_loader", module_data)]
        config['data_loader']['args'] = replace_nested_dict_item(config['data_loader']['args'], 'split', 'val')
        valid_data_loader = [config.initialize("data_loader", module_data)]
    elif isinstance(config["data_loader"], list):
        data_loader = [config.initialize('data_loader', module_data, index=idx) for idx in
                       range(len(config['data_loader']))]
        new_cfg_li = []
        for dl_cfg in config['data_loader']:
            dl_cfg['args'] = replace_nested_dict_item(dl_cfg['args'], 'split', 'val')
            new_cfg_li.append(dl_cfg)
        config._config['data_loader'] = new_cfg_li
        valid_data_loader = [config.initialize('data_loader', module_data, index=idx) for idx in
                             range(len(config['data_loader']))]
    else:
        raise ValueError("Check data_loader config, not correct format.")

    return data_loader, valid_data_loader


if __name__ == '__main__':
    try:    # with ddp
        master_address = os.environ['MASTER_ADDR']
        master_port = int(os.environ['MASTER_PORT'])
        world_size = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ['RANK'])
        # rank = int(os.environ['LOCAL_RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
    except:  # for debug only
        master_address = 9339
        master_port = 1
        world_size = 1
        rank = 0
        local_rank = 0

    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default='/apdcephfs/private_qinghonglin/video_codebase/frozen-in-time-main/configs/pt/ego4d_4f_o.json', type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-o', '--observe', action='store_true',
                      help='Whether to observe (neptune)')
    args.add_argument('-l', '--launcher', choices=['none', 'pytorch'], default='none',help='job launcher')
    args.add_argument('-k', '--local_rank', type=int, default=local_rank)

    args.add_argument('-ma', '--master_address', default=master_address)
    args.add_argument('-mp', '--master_port', type=int, default=master_port)
    args.add_argument('-ws', '--world_size', type=int, default=world_size)
    args.add_argument('-rk', '--rank', type=int, default=rank)
    args.add_argument('-lr1', '--learning_rate1', type=float, default=2e-4)
    args.add_argument('-sc', '--schedule', default=[60, 80])

    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target=('optimizer', 'args', 'lr')),
        CustomArgs(['--bs', '--batch_size'], type=int, target=('data_loader', 'args', 'batch_size')),
    ]
    config = ConfigParser(args, options)
    args = args.parse_args()
    ex.add_config(config._config)

    if args.rank == 0:
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
    print("The rank(local) of this node is {}({})".format(args.rank, args.local_rank))

    if config['trainer']['neptune']:
        # delete this error if you have added your own neptune credentials neptune.ai
        raise ValueError('Neptune credentials not set up yet.')
        ex.observers.append(NeptuneObserver(
            api_token='INSERT TOKEN',
            project_name='INSERT PROJECT NAME'))
        ex.run()
    else:
        run()