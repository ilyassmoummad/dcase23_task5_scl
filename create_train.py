import glob
import os
import pandas as pd
import torchaudio
import torch
import torch.nn as nn
import numpy as np
from torchaudio import transforms as T
import h5py
import argparse
from args import args

traindir = args.traindir
csv_files = [f for f in glob.glob(os.path.join(traindir, '*/*.csv'))]

TARGET_SR = args.sr
N_FFT = args.nfft
N_MELS = args.nmels
HOP_MEL = args.hoplen
FMIN = args.fmin
FMAX = args.fmax
N_SHOT = args.nshot
fps = TARGET_SR/HOP_MEL
WIN_LEN = args.len
SEG_LEN = WIN_LEN//2
win_len = int(round((WIN_LEN/1000) * fps))
seg_hop = int(round((SEG_LEN/1000) * fps))

mel = T.MelSpectrogram(sample_rate=TARGET_SR, n_fft=N_FFT, hop_length=HOP_MEL, f_min=FMIN, f_max=FMAX, n_mels=N_MELS)
power_to_db = T.AmplitudeToDB()
transform = nn.Sequential(mel, power_to_db)


# class labels
class_names = []
ds_names = []
for csv_file in csv_files:
    ds_name = csv_file.split('/')[-2]

    #wav_file = csv_file.replace('csv','wav')
    df = pd.read_csv(csv_file)
    col_names = df.columns.tolist()
    for col_name in col_names:
        if (col_name not in ['Audiofilename', 'Starttime', 'Endtime']) and (col_name not in class_names) and (len(df[df[col_name]=='POS'])>0):
            class_names.append(col_name)

map_cls_2_int={}
for i in range(len(class_names)):
    map_cls_2_int[class_names[i]] = i

def cls2int(cls_name):
    return map_cls_2_int[cls_name]

hdf_tr = os.path.join(traindir,'train.h5')
hf = h5py.File(hdf_tr,'w')
hf.create_dataset('data', shape=(0, N_MELS, win_len), maxshape=(None, N_MELS, win_len))
hf.create_dataset('label', shape=(0, 1), maxshape=(None, 1))

if len(hf['data'][:]) == 0:
    file_index = 0
else:
    file_index = len(hf['data'][:])

for csv_file in csv_files:
    counter = 0
    ds_name = csv_file.split('/')[-2]
    wav_file = csv_file.replace('csv','wav')
    print(f"creating data for {wav_file}")
    df = pd.read_csv(csv_file)
    df.loc[:,'Starttime'] = df['Starttime'] 
    df.loc[:,'Endtime'] = df['Endtime']
    wav, sr = torchaudio.load(wav_file)
    resample = T.Resample(sr, TARGET_SR)
    wav = resample(wav)
    if wav.shape[0] != 1:
        wav = torch.mean(wav, dim=0, keepdim=True)
    melspec = transform(wav)

    df_cols = df.columns.tolist()

    dfs = []
    for df_col in df_cols:
        if len(df[df[df_col] == 'POS']) > 0:
            dfs.append(df[df[df_col] == 'POS'])   
        
    dfs = pd.concat(dfs)

    for i in range(len(dfs)):
        ith_row = dfs.iloc[i]

        for df_col in df_cols:

            if ith_row[df_col] == 'POS':
                
                label = df_col
                break    
        onset = int(round(ith_row['Starttime'] * fps))
        offset = int(round(ith_row['Endtime'] * fps))

        start_idx = onset

        if offset - start_idx > win_len:
            while offset - start_idx > win_len:
                spec = melspec[...,start_idx:start_idx+win_len]
                
                
                if spec.sum() == 0:
                    counter += 1
                    start_idx += seg_hop
                    continue
                
                spec = (spec - spec.min()) / (spec.max() - spec.min())

                hf['data'].resize((file_index + 1, spec.shape[-2], spec.shape[-1]))
                hf['data'][file_index] = spec
                hf['label'].resize((file_index + 1, 1))
                hf['label'][file_index] = map_cls_2_int[label]
                file_index += 1
                start_idx += seg_hop

            if offset - start_idx > win_len//8 :
                spec = melspec[...,start_idx:offset]         
                repeat_num = int(win_len / (spec.shape[-1])) + 1
                spec = spec.repeat(1,1,repeat_num)
                spec = spec[...,:int(win_len)]
                spec = (spec - spec.min()) / (spec.max() - spec.min())

                hf['data'].resize((file_index + 1, spec.shape[-2], spec.shape[-1]))
                hf['data'][file_index] = spec
                hf['label'].resize((file_index + 1, 1))
                hf['label'][file_index] = map_cls_2_int[label]
                file_index += 1
        else:
            if offset - start_idx > win_len//8: #0
                spec = melspec[...,start_idx:offset] 
                if spec.sum() == 0:
                    counter += 1
                    continue
                
                repeat_num = int(win_len / (spec.shape[-1])) + 1
                spec = spec.repeat(1,1,repeat_num)
                spec = spec[...,:int(win_len)]
                spec = (spec - spec.min()) / (spec.max() - spec.min())
                
                hf['data'].resize((file_index + 1, spec.shape[-2], spec.shape[-1]))
                hf['data'][file_index] = spec
                hf['label'].resize((file_index + 1, 1))
                hf['label'][file_index] = map_cls_2_int[label]

                file_index += 1
    if counter > 0:
        print(f"{counter} patches are null in {wav_file}")
print("Total files created : {}".format(file_index))
hf.close()

hdf_train = h5py.File(hdf_tr, 'r+')
x = hdf_train['data'][:]
y = hdf_train['label'][:]

print(f"data shape : {x.shape}\tlabel shape : {y.shape}\t unique labels : {np.unique(y).shape}")