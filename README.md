# EgoVLP: Egocentric Video-Language Pretraining
https://arxiv.org/pdf/2206.01670.pdf

<img src="/figures/egovlp_framework.jpg" alt="EgoVLP" style="zoom:67%;" />

Dataset and code are coming soon :)

## Preparation for EgoClip
1. Follow the guideline [here](https://ego4d-data.org/docs/start-here/#cli-download) to download the Ego4D `manifest.csv` and source videos to `{PATH_TO_EGO4D}`.
2. Create the dir `./dataset` and add a soft link by `ln -s {PATH_TO_EGO4D} ./dataset/ego4d`.
4. Use the script `./utils/video_resize.py` to resize the source videos with a short size equal to 256.
5. Use the script `./utils/video_chunk.py` to chunk the resized videos to multiple segments, up to 600 sec.
6. Download the EgoClip metadata from [here](https://drive.google.com/file/d/1dPxnfUklqTjxrwIoapKZGqlthWs0_UbG/view?usp=sharing) and put it to `./dataset/egoclip_metadata.csv`. The usage and data formats are:

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
