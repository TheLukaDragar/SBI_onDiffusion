import pickle
import os
from sklearn.metrics import roc_auc_score


# Path to the directory containing the pickle files
pickle_path = '/ceph/hpc/data/st2207-pgp-users/ldragar/Marijaproject/SelfBlendedImages/preds'

# List all files in the directory
pickle_files = os.listdir(pickle_path)

# Dictionary to hold combined predictions
all_preds = []

for file in pickle_files:
    file_path = os.path.join(pickle_path, file)
    with open(file_path, 'rb') as f:
        preds = pickle.load(f)
        all_preds.append(preds)
        
        

#concat dataframes
import pandas as pd

df = pd.concat(all_preds)

print(df.head())

print(df.shape)

df["vid_basename"] = df["video"].apply(lambda x: os.path.basename(x))

print(df.head())

#load josns
import json

train = json.load(open('/ceph/hpc/data/st2207-pgp-users/ldragar/Marijaproject/train.json'))
val = json.load(open('/ceph/hpc/data/st2207-pgp-users/ldragar/Marijaproject/val.json'))
test = json.load(open('/ceph/hpc/data/st2207-pgp-users/ldragar/Marijaproject/test.json'))

train_files = ["_".join(x) for x in train]
val_files = ["_".join(x) for x in val]
test_files = ["_".join(x) for x in test]

#add reverse pairs
train_files += ["_".join(x[::-1]) for x in train]
val_files += ["_".join(x[::-1]) for x in val]
test_files += ["_".join(x[::-1]) for x in test]

print(train_files[:5])
print(val_files[:5])
print(test_files[:5])

#add mp4 extension
train_files = [x+'.mp4' for x in train_files]
val_files = [x+'.mp4' for x in val_files]
test_files = [x+'.mp4' for x in test_files]

#set split in df map to vid_basename

df['split'] = 'train'
df.loc[df['vid_basename'].isin(val_files), 'split'] = 'val'
df.loc[df['vid_basename'].isin(test_files), 'split'] = 'test'

print(df.head())


#get counts
print(df['split'].value_counts())

#calc auc on val

val_scores = df[df['split']=='val']['mean'].values
print(val_scores.shape)






