[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/egocentric-video-language-pretraining/multi-instance-retrieval-on-epic-kitchens-100)](https://paperswithcode.com/sota/multi-instance-retrieval-on-epic-kitchens-100?p=egocentric-video-language-pretraining)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/egocentric-video-language-pretraining/action-recognition-on-charades-ego)](https://paperswithcode.com/sota/action-recognition-on-charades-ego?p=egocentric-video-language-pretraining)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/egocentric-video-language-pretraining/natural-language-queries-on-ego4d)](https://paperswithcode.com/sota/natural-language-queries-on-ego4d?p=egocentric-video-language-pretraining)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/egocentric-video-language-pretraining/moment-queries-on-ego4d)](https://paperswithcode.com/sota/moment-queries-on-ego4d?p=egocentric-video-language-pretraining)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/egocentric-video-language-pretraining/object-state-change-classification-on-ego4d)](https://paperswithcode.com/sota/object-state-change-classification-on-ego4d?p=egocentric-video-language-pretraining)

# EgoVLP: Egocentric Video-Language Pretraining

[Project page](https://qinghonglin.github.io/EgoVLP/) | [arXiv](https://arxiv.org/pdf/2206.01670.pdf)

> **TL;DR:** We pioneer Egocentric Video-Language Pretraining from pretraining dataset, model and development benchmark; the resulted pretrained model exhibits strong performance on five downstream tasks across three egocentric datasets.

<img src="/figures/egovlp_framework.jpg" alt="EgoVLP" style="zoom:67%;" />

## 📢 News

- [2022.9.15] EgoVLP got accepted by [**NeurIPS 2022**](https://nips.cc/)! We will further complete the code and guidelines in the next phase.
- [2022.6.30] We release the first version of the EgoVLP codebase.
- [2022.6.20] Our EgoVLP won [**1st place** in OSCC](https://eval.ai/web/challenges/challenge-page/1627/overview) & [**2nd place** in NLQ](https://eval.ai/web/challenges/challenge-page/1629/overview) & [**3rd place** in PNR](https://eval.ai/web/challenges/challenge-page/1622/overview) @ [Ego4D  Challenge 2022](https://ego4d-data.org/docs/challenge/), and [**1st place** in Multi-Instance Retrieval](https://codalab.lisn.upsaclay.fr/competitions/617#learn_the_details) @ [EPIC-Kitchens Challenge 2022](https://epic-kitchens.github.io/2022), hosted by CVPR 2022.
- [2022.6.10] We release the EgoClip pretraining dataset.
- [2022.6.3] We release the arXiv paper.

## 📝 Preparation
### Install dependencies 
```bash
conda create -n egovlp python=3.6
source activate egovlp
cd [Path_To_This_Code]
pip install -r requirements.txt
```

### Ego4D videos and metadata
> You may skip the source video download if pretraining is not required.
1. Follow the guideline [here](https://ego4d-data.org/docs/start-here/#cli-download), download the following to  `{PATH_TO_EGO4D}`
   - Ego4D source videos (nearly 7 TB).
   - Ego4D videos metadata `manifest.csv` and benchmark metadata, e.g., `nlq_train.json` for NLQ.
   - Create the dir `./dataset` and add a soft link by `ln -s {PATH_TO_EGO4D} ./dataset/ego4d`.

2. For effectively pretraining, we compress videos in the following way:
   - Resize the source videos with a short size equal to 256 by script  `./utils/video_resize.py`.
   - Chunk the resized videos to multiple segments (up to 600 sec) by script `./utils/video_chunk.py`.

### EgoClip
- Download the EgoClip metadata from [here](https://drive.google.com/file/d/1-aaDu_Gi-Y2sQI_2rsI2D1zvQBJnHpXl/view?usp=sharing) and put it to `./dataset/egoclip.csv`.

- For the usage of EgoClip, please refer to `./data_loader/EgoClip_EgoMCQ_dataset.py`. The data format of EgoClip is:
  ```python
  import pandas as pd
  
  metadata = pd.read_csv('./dataset/egoclip_metadata.csv', sep='\t', error_bad_lines=False)
  print(metadata.shape[0])
  print(metadata.iloc[0])
  
  # Out:
  3847723                                                         # Num of clips for EgoClip
  
  clip_idx                                                     0  # the idx of clip
  video_uid                 001e3e4e-2743-47fc-8564-d5efd11f9e90  # the uid of source video
  video_dur                                           128.033333  # the duration of source video
  narration_source                              narration_pass_1  # the source of annotator
  narration_ind                                                0  # the idx of narration
  narration_time                                          3.3445  # the narration timestamp
  clip_start                                            2.967651  # the start timestamp of clip
  clip_end                                              3.721266  # the end timestamp of clip
  clip_text           #C C picks a bag of clothes from the floor  # the narration of clip
  tag_verb                                                  [93]  # the verb idx of the narration
  tag_noun                                        [192, 115, 12]  # the noun idx of the narration
  ```
  
  For the usage of `tag_verb` and `tag_noun`, please refer to [here](https://drive.google.com/drive/folders/16fUv5rrZmt06Ty3QAEweDpveC-84RI9Z?usp=sharing).

### EgoMCQ

- Download the EgoMCQ metadata from [here](https://drive.google.com/file/d/1-5iRYf4BCHmj4MYQYFRMY4bhsWJUN3rW/view?usp=sharing) and put it to `./dataset/egomcq.json`.
- For the usage of EgoMCQ, please refer to `./data_loader/EgoClip_EgoMCQ_dataset.py`.

## 🏋️‍️ Pretraining
This code is built on PyTorch with DistributedDataParallel (DDP). We pretrain EgoVLP on 4 nodes, each with 8 A100 GPUs (10 epochs in about two days).

- Train on EgoClip:  `python3 -m torch.distributed.launch 
  --nnodes=$HOST_NUM 
  --node_rank=$INDEX 
  --master_addr $CHIEF_IP 
  --nproc_per_node $HOST_GPU_NUM 
  --master_port 8081 
  ./run/train_egoclip.py --config ./configs/pt/egoclip.json`
  
- Test on EgoMCQ:  `python3 -m torch.distributed.launch 
  --nnodes=$HOST_NUM 
  --node_rank=$INDEX 
  --master_addr $CHIEF_IP 
  --nproc_per_node $HOST_GPU_NUM 
  --master_port 8081 
  ./run/train_egoclip.py --config ./configs/eval/egomcq.json`
  
- Monitor the EgoMCQ curve during pretraining: `tensorboard --logdir ./results  --bind_all`

## 🗄 Pretrained Weights
- We have released our pretrained EgoVLP model (EgoClip w/ EgoNCE) with best performance on EgoMCQ (90.7% inter-video & 57.2% intra-video) in [EgoVLP_PT_BEST](https://drive.google.com/file/d/1-cP3Gcg0NGDcMZalgJ_615BQdbFIbcj7/view?usp=sharing).


*This checkpoint is used for EPIC-Kitchens, NLQ, MQ, OSSC, and PNR tasks, except for Charades-Ego. Since we found that VLP (CC3M+WebVid2M, EgoClip) alway degrades significantly on Charades-Ego after the first epoch, we evaluate Charades-Ego using the first pretraining epoch weights of EgoVLP in [EgoVLP_PT_EPO1](https://drive.google.com/file/d/10lRA4Fldt-c5Azh5D2Zvjwi-_YR5ve5e/view?usp=sharing)*.

## 🔧 Downstream Tasks
### EPIC-Kitchens MIR

- **Preparation:**

1. Follow the instruction [here](https://epic-kitchens.github.io/2022), download the EPIC-Kitchens dataset (RGB frames) and annotation.
2. Follow the instruction [here -> How do I create the relevance matrix?](https://github.com/mwray/Joint-Part-of-Speech-Embeddings) to construct a relevance matrix for evaluation.

- **Results:**

| Model   | Mode                                              | # Frames | Video-Text PT     | Weights                                                 | mAP (V2T) | mAP (T2V) | mAP (Avg) | nDCG (V2T) | nDCG (T2V) | nDCG (Avg) |
| ------- | ------------------------------------------------- | ------ | ----------------- | ------------------------------------------------------------ | --------- | --------- | --------- | ---------- | ---------- | ---------- |
| EgoVLP  | Zero-shot                                         | 4      | EgoClip w/ EgoNCE | [EgoVLP_PT_BEST](https://drive.google.com/file/d/1-cP3Gcg0NGDcMZalgJ_615BQdbFIbcj7/view?usp=sharing) | 19.4      | 13.9      | 16.6      | 24.1       | 22.0       | 23.1       |
| EgoVLP  | Fine-tuning w/<br /> MI-MM                        | 16     | EgoClip w/ EgoNCE | [EgoVLP_FT_EPIC](https://drive.google.com/file/d/1-YEHZ-WBCnO-LZEsDF14jo-pLSJKTp2G/view?usp=sharing) | 49.9      | 40.5      | 45.0      | 60.9       | 57.9       | 59.4       |
| EgoVLP* | Fine-tuning w/ Adaptive-MI-MM + Dual-softmax | 16     | EgoClip w/ EgoNCE | [EgoVLP_FT_EPIC*](https://drive.google.com/file/d/1-SOQeXc-xSn544sJzgFLhC95hkQsm0BR/view?usp=sharing) | **53.8**  | **40.9**  | **47.4**  | **63.3**   | **59.6**   | **61.4**   |

*EgoVLP\* means our submission for [Multi-Instance Retrieval@EPIC-Kitchens Challenge 2022](https://codalab.lisn.upsaclay.fr/competitions/617#learn_the_details)*

- Train: `python3 -m torch.distributed.launch --nnodes=$HOST_NUM  --node_rank=$INDEX  --nproc_per_node $HOST_GPU_NUM --master_port 8081 ./run/train_epic.py --config ./configs/ft/epic.json`

- Test: `python3 ./run/test_epic.py`

### Charades-Ego
- **Preparation:**

1. Follow the instruction [here](https://prior.allenai.org/projects/charades-ego), download the Charades-Ego dataset (480p) and annotation.
2. Create a training metadata via `./utils/charades_meta.py` 

- **Results:**

| Model  | Mode        | # Frames | Video-Text PT     | Weights | mAP  |
| ------ | ----------- | -------- | ----------------- | ----------------- | ---- |
| EgoVLP | Zero-shot   | 16       | EgoClip w/ EgoNCE | [EgoVLP_PT_EPO1](https://drive.google.com/file/d/10lRA4Fldt-c5Azh5D2Zvjwi-_YR5ve5e/view?usp=sharing)                  | 25.0 |
| EgoVLP | Fine-tuning w/ InfoNCE| 16       | EgoClip w/ EgoNCE | [EgoVLP_FT_CHARADES](https://drive.google.com/file/d/1-xWVDH7XO4pi6Hj5QRpKVz6y-QkqcFlQ/view?usp=sharing)                  | **32.1** |

- Train: `python3 -m torch.distributed.launch --nnodes=$HOST_NUM  --node_rank=$INDEX  --nproc_per_node $HOST_GPU_NUM --master_port 8081 ./run/train_epic.py --config ./configs/ft/charades.json`

- Test: `python3 ./run/test_charades.py`


### NLQ
- **Preparation:** Make sure you have prepared the NLQ videos and metadata.
- Extract video features: `python3 ./run/test_nlq.py --subsample 'text'`.
- Extract text features: `python3 ./run/test_nlq.py --subsample 'video'`.
- Fine-tune the [VSLNet](https://github.com/EGO4D/episodic-memory/tree/main/NLQ/VSLNet) by replacing its input video-text features.

### MQ
- **Preparation:** Make sure you have prepared the MQ videos and metadata.
- Extract video features: `python3 ./run/test_mq.py`.
- Fine-tune the [VSGN](https://github.com/EGO4D/episodic-memory/tree/main/MQ) by replacing its input video features.

### OSCC
- **Preparation:**
 
1. Make sure you have prepared the OSCC videos and metadata.
2. Extract the clip frame follow the instruction [here -> Data Preparation](https://github.com/EGO4D/hands-and-objects/tree/main/state-change-localization-classification/i3d-resnet50).

- Train: `python3 -m torch.distributed.launch --nnodes=$HOST_NUM  --node_rank=$INDEX  --nproc_per_node $HOST_GPU_NUM --master_port 8081 ./run/train_oscc.py --config ./configs/ft/oscc.json`

### PNR
- **Preparation:** Same as OSCC.
- Train: `python3 -m torch.distributed.launch --nnodes=$HOST_NUM  --node_rank=$INDEX  --nproc_per_node $HOST_GPU_NUM --master_port 8081 ./run/train_pnr.py --config ./configs/ft/pnr.json`

## 🎓 Citation

If you find our work helps, please cite our paper.

```bibtex
@article{kevin2022egovlp,
	title={Egocentric Video-Language Pretraining},
	author={Kevin Qinghong Lin and Alex Jinpeng Wang and Mattia Soldan and Michael Wray and Rui Yan and Eric Zhongcong Xu and Difei Gao and Rongcheng Tu and Wenzhe Zhao and Weijie Kong and Chengfei Cai and Hongfa Wang and Dima Damen and Bernard Ghanem and Wei Liu and Mike Zheng Shou},
	journal={arXiv preprint arXiv:2206.01670},
	year={2022}
}
```

## ✉️ Contact

This repo is maintained by [Kevin](https://github.com/QinghongLin). Questions and discussions are welcome via kevin.qh.lin@gmail.com.

We are willing to merge results and codes if transfer our EgoVLP to other egocentric tasks or datasets.

## 🙏 Acknowledgements

This codebase is based on [Frozen](https://github.com/m-bain/frozen-in-time). 

Thanks to [Alex](https://github.com/fingerrec) for the help with DDP and [Mattia](https://github.com/Soldelli) for the help with NLQ and MQ benchmarks.

## LICENSE

MIT
