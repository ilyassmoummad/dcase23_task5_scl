import torch
import torch.nn as nn
import torchaudio
from torchaudio import transforms as T
from torch.utils.data import DataLoader
from models import ResNet
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from util import finetune_ce
import glob
import os
import pickle
from da import RandomCrop, Resize, Compander
from args import args
if args.wandb:
    import wandb

# General params
val_dir = args.valdir
out_dir = os.path.join(args.traindir, '../../outputs')
os.makedirs(out_dir, exist_ok=True)
csv_path = os.path.join(out_dir, 'eval.csv')
ckpt_dir = os.path.join(args.traindir, '../model/')
TARGET_SR = args.sr
N_MELS = args.nmels
HOP_LEN = args.hoplen
N_SHOT = args.nshot
BATCH_SIZE = args.qbs
NUM_WORKERS = args.workers
fps = TARGET_SR/HOP_LEN

# Spectrogram
mel = T.MelSpectrogram(sample_rate=args.sr, n_fft=args.nfft, hop_length=args.hoplen, f_min=args.fmin, f_max=args.fmax, n_mels=args.nmels)
power_to_db = T.AmplitudeToDB()
transform = nn.Sequential(mel, power_to_db)

name_arr = np.array([])
onset_arr = np.array([])
offset_arr = np.array([])
filenames = [file for file in glob.glob(os.path.join(val_dir,'*/*.csv'))]
for filename in filenames:
    print(f"processing file {filename}")
    feat_name = filename.split('/')[-1]
    audio_name = feat_name.replace('csv','wav')
    csv_file = os.path.join(val_dir, filename)
    wav_file = csv_file.replace('csv','wav')
    df = pd.read_csv(csv_file)
    df.loc[:,'Starttime'] = df['Starttime']
    df.loc[:,'Endtime'] = df['Endtime']
    Q_list = df['Q'].to_numpy()
    index_sup = np.where(Q_list == 'POS')[0][:N_SHOT]
    start_time = [int(np.floor(start * fps)) for start in df['Starttime']]
    end_time = [int(np.floor(end * fps)) for end in df['Endtime']]
    difference = []
    for index in index_sup:
        difference.append(end_time[index] - start_time[index])
    """
    max_len = max(difference)

    if max_len < 100:
        win_len = max_len
    elif max_len > 100 and max_len < 500 :
        win_len = max_len//4
    else:
        win_len = max_len//8    
    seg_hop = win_len//2
    """
    max_len = int(round(np.mean(difference)))

    if max_len <= 17:
        win_len = 17
    elif max_len > 17 and max_len <= 100 :
        win_len = max_len
    elif max_len > 100 and max_len <= 200 :
        win_len = max_len//2
    elif max_len > 200 and max_len <= 400 :
        win_len = max_len//4
    else:
        win_len = max_len//8
    seg_hop = win_len//2
    
    wav, sr = torchaudio.load(wav_file)

    resample = T.Resample(sr, TARGET_SR)
    wav = resample(wav)
    melspec = transform(wav)

    features_pos = []
    features_neg = []

    for index in range(len(index_sup)):
        # pos features
        start_idx = start_time[index_sup[index]] 
        if end_time[index_sup[index]] - start_idx > win_len:
            while end_time[index_sup[index]] - start_idx > win_len:
                spec = melspec[...,start_idx:start_idx+win_len]
                start_idx += seg_hop
                features_pos.append(spec)  
            if end_time[index_sup[index]] - start_idx > win_len//2:
                spec = melspec[...,start_idx:end_time[index_sup[index]]]
                repeat_num = int(win_len / (spec.shape[-1])) + 1
                spec = spec.repeat(1,1,repeat_num)
                spec = spec[...,:int(win_len)]
                features_pos.append(spec)  
        else:
            if end_time[index_sup[index]] - start_idx > 0:
                spec = melspec[...,start_idx:end_time[index_sup[index]]]
                repeat_num = int(win_len / (spec.shape[-1])) + 1
                spec = spec.repeat(1,1,repeat_num)
                spec = spec[...,:int(win_len)]
                features_pos.append(spec)  

        # neg features
        start_idx = end_time[index_sup[index]]
        if index < len(index_sup)-1:
            if start_time[index_sup[index+1]] - start_idx > win_len:
                while start_time[index_sup[index+1]] - start_idx > win_len:
                    spec = melspec[...,start_idx:start_idx+win_len]
                    start_idx += seg_hop
                    features_neg.append(spec)
                if start_time[index_sup[index+1]] - start_idx > win_len//2:
                    spec = melspec[...,start_idx:start_time[index_sup[index+1]]]
                    repeat_num = int(win_len / (spec.shape[-1])) + 1
                    spec = spec.repeat(1,1,repeat_num)
                    spec = spec[...,:int(win_len)]
                    features_neg.append(spec)
            else:
                if start_time[index_sup[index+1]] - start_idx > 0:
                    spec = melspec[...,start_idx:start_time[index_sup[index+1]]]
                    repeat_num = int(win_len / (spec.shape[-1])) + 1
                    spec = spec.repeat(1,1,repeat_num)
                    spec = spec[...,:int(win_len)]
                    features_neg.append(spec)   

        # add event before first pos to neg proto
        end_t0 = start_time[index_sup[0]] 
        start_t0 = 0
        curr_t0 = start_t0
        if end_t0 - curr_t0 > win_len : 
            while end_t0 - curr_t0 > win_len :
                spec = melspec[...,curr_t0:curr_t0+win_len]
                curr_t0 += seg_hop
                features_neg.append(spec)
            if end_t0 - curr_t0 > win_len//2 :
                spec = melspec[...,curr_t0:curr_t0+win_len]
                repeat_num = int(win_len / (spec.shape[-1])) + 1
                spec = spec.repeat(1,1,repeat_num)
                spec = spec[...,:int(win_len)]
                features_neg.append(spec)
        else:
            if end_t0 - curr_t0 > 0 :
                spec = melspec[...,curr_t0:curr_t0+win_len]
                repeat_num = int(win_len / (spec.shape[-1])) + 1
                spec = spec.repeat(1,1,repeat_num)
                spec = spec[...,:int(win_len)]
                features_neg.append(spec)

    features_pos = torch.stack(features_pos)
    features_neg = torch.stack(features_neg)
    nb_positives = features_pos.shape[0]
    nb_negatives = features_neg.shape[0]

    # query features
    labels_q = []
    features_q = []

    last_frame = melspec.shape[-1]
    curr_frame = end_time[index_sup[-1]]
    if last_frame - curr_frame > win_len:
        while last_frame - curr_frame > win_len:
            spec = melspec[...,curr_frame:curr_frame+win_len]
            features_q.append(spec)
            curr_frame += seg_hop
        if last_frame - curr_frame > win_len//2:
            spec = melspec[...,curr_frame:last_frame]
            repeat_num = int(win_len / (spec.shape[-1])) + 1
            spec = spec.repeat(1,1,repeat_num)
            spec = spec[...,:int(win_len)]
            features_q.append(spec)   
    else:
        if last_frame - curr_frame > win_len//2:
            spec = melspec[...,curr_frame:last_frame]
            repeat_num = int(win_len / (spec.shape[-1])) + 1
            spec = spec.repeat(1,1,repeat_num)
            spec = spec[...,:int(win_len)]
            features_q.append(spec)

    features_q = torch.stack(features_q)


    # light DA for CE fitting
    rc = RandomCrop(n_mels=128, time_steps=features_q.shape[-1], tcrop_ratio=0.9)
    resize = Resize(n_mels=128, time_steps=features_q.shape[-1])
    comp = Compander(comp_alpha=0.9)
    if args.ft == 0:
        makeview = nn.Identity()
    else:
        #makeview = nn.Sequential(rc, resize, comp)
        makeview = nn.Sequential(rc, resize)

    # Loading model
    encoder = ResNet(method='ce', num_classes=2)
    ckpt = torch.load(os.path.join(ckpt_dir, 'ckpt.pth'), map_location=torch.device('cpu'))
    encoder.load_state_dict(ckpt['encoder'], strict=False)
    encoder = encoder.to(args.device)

    # finetuning
    data_finetune = torch.cat([features_pos, features_neg], dim=0)
    label_finetune = torch.cat([torch.ones(features_pos.shape[0], dtype=torch.long), torch.zeros(features_neg.shape[0], dtype=torch.long)], dim=0)

    bs_finetune = args.ftbs

    ds_finetune = TensorDataset(data_finetune, label_finetune)
    dl_finetune = DataLoader(ds_finetune, batch_size=bs_finetune, shuffle=True)
    encoder = finetune_ce(encoder, dl_finetune, makeview, args)
    
    # prepare query for prediction
    ds_pos = TensorDataset(features_pos)
    ds_neg = TensorDataset(features_neg)
    ds_q = TensorDataset(features_q)

    loader_q = DataLoader(ds_q, batch_size=BATCH_SIZE)
    
    labels_pred = []
    labels_comb = []

    encoder.eval()

    # predict class label for queries
    for x_q in tqdm(loader_q):
        _, z_q = encoder(x_q[0].to(args.device))
        label_pred = torch.argmax(z_q.detach().cpu(), dim=-1)
        labels_pred.extend(label_pred.tolist())

    labels_comb.append(labels_pred)

    labels_comb = torch.tensor(labels_comb)
  
    labs_pred, _ = torch.mode(labels_comb, axis=0)

    labs_pred = np.array(labs_pred)

    krn = np.array([1, -1])

    changes = np.convolve(krn, labs_pred)

    onset_frames = np.where(changes == 1)[0]
    offset_frames = np.where(changes == -1)[0]

    str_time_query = end_time[index_sup[-1]] * HOP_LEN / TARGET_SR

    onset = (onset_frames ) * (seg_hop) * HOP_LEN / TARGET_SR
    onset = onset + str_time_query

    offset = (offset_frames ) * (seg_hop) * HOP_LEN / TARGET_SR
    offset = offset + str_time_query

    assert len(onset) == len(offset)

    name = np.repeat(audio_name,len(onset))
    name_arr = np.append(name_arr,name)
    onset_arr = np.append(onset_arr,onset)
    offset_arr = np.append(offset_arr,offset)

df_out = pd.DataFrame({'Audiofilename':name_arr,'Starttime':onset_arr,'Endtime':offset_arr})
df_out.to_csv(csv_path,index=False)
