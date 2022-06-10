# EgoVLP: Egocentric Video-Language Pretraining
https://arxiv.org/pdf/2206.01670.pdf

<img src="/figures/egovlp_framework.jpg" alt="EgoVLP" style="zoom:67%;" />

Dataset and code are coming soon :)

## Preparation for EgoClip
1. Follow the guideline [here](https://ego4d-data.org/docs/start-here/#cli-download) to download the `manifest.csv` and source Ego4D videos to `[path_to_ego4d]`
2. Create the dir `./dataset` and add soft link by `ln -s [path_to_ego4d] ./dataset/ego4d`
4. Use the script `./utils/video_resize.py` to resize the source videos with a short size equal to 256.
5. Use the script `./utils/video_chunk.py` to chunk the resized video to multiple segments, up to 600 sec.
