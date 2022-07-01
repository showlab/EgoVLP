import os
from csv import reader
import csv
import pandas as pd

metadata_dir = '/apdcephfs/private_qinghonglin/video_dataset/charades/CharadesEgo/'
split_files = {
    'train': 'CharadesEgo_v1_train_only1st.csv',
    'val': 'CharadesEgo_v1_test_only1st.csv',  # there is no test
    'test': 'CharadesEgo_v1_test_only1st.csv'
}

split = 'test'
target_split_fp = split_files[split]

with open(os.path.join(metadata_dir, target_split_fp), 'r') as charades:
    csv_reader = list(reader(charades))[1:]

path_metadata = "/apdcephfs/private_qinghonglin/video_dataset/charades/CharadesEgo/metadata_" + split + ".csv"
csv_metadata = open(path_metadata, 'w')

metadata_reader = csv.writer(csv_metadata, delimiter='¥', lineterminator='\n',
                             quoting=csv.QUOTE_NONE, escapechar='¥')

header = 'id' + '\t' + 'cls' + '\t' + 't_start' + '\t' + 't_end' + '\t' + 'narration'

metadata_reader.writerow([header])

mapping = pd.read_csv('/apdcephfs/private_qinghonglin/video_dataset/charades/CharadesEgo/Charades_v1_classes.txt', header=None)

count = 0
for row in csv_reader:
    id = row[0]
    actions = row[9].split(';')
    if actions == ['']:
        continue
    for action in actions:
        action_seg = action.split(' ')
        cls, t_start, t_end = int(action_seg[0][1:]), float(action_seg[1]), float(action_seg[2])

        narration = mapping.iloc[cls][0][5:]

        clip_info = id + '\t' + str(cls) + '\t' + \
                    str(t_start) + '\t' + str(t_end) + '\t' + str(narration)
        count += 1
        print(count)
        metadata_reader.writerow([clip_info])