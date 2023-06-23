import torch
from torch import nn
from tqdm import tqdm
from losses import SupConLoss
from da import RandomCrop, Resize, Compander, GaussNoise, FreqShift, MixRandom
from models import ResNet
from torchinfo import summary
from args import args
import math
import h5py
import os

def train_scl(encoder, train_loader, transform1, transform2, args):

    print(f"Training starting on {args.device}")
    
    loss_fn = SupConLoss(temperature=args.tau, device=args.device)
    
    optim = torch.optim.SGD(encoder.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
    num_epochs = args.epochs

    ckpt_dir = os.path.join(args.traindir, '../model/')
    os.makedirs(ckpt_dir, exist_ok=True) 
    last_model_path = os.path.join(ckpt_dir, 'ckpt.pth')

    encoder = encoder.to(args.device)
    
    for epoch in range(1, num_epochs+1):
        tr_loss = 0.
        print("Epoch {}".format(epoch))
        adjust_learning_rate(optim, args.lr, epoch, num_epochs+1)
        train_iterator = iter(train_loader)
        for batch in tqdm(train_iterator):
            optim.zero_grad()
            
            x, y = batch
            x = x.to(args.device)
            y = y.to(args.device)

            x1 = transform1(x); x2 = transform2(x)

            _, x_out1 = encoder(x1); _, x_out2 = encoder(x2)

            if args.method == 'ssl':
                loss = loss_fn(x_out1, x_out2)
            elif args.method == 'scl':
                loss = loss_fn(x_out1, x_out2, y)
            tr_loss += loss.item()

            loss.backward()
            optim.step()

        tr_loss = tr_loss/len(train_iterator)
        print('Average train loss: {}'.format(tr_loss))

    torch.save({'encoder':encoder.state_dict()},last_model_path)

    return encoder

def adjust_learning_rate(optimizer, init_lr, epoch, tot_epochs):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / tot_epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr

if __name__ == "__main__":

    # Load data
    hdf_tr = os.path.join(args.traindir,'train.h5')
    hdf_train = h5py.File(hdf_tr, 'r+')
    X = hdf_train['data'][:]
    Y = hdf_train['label'][:]

    # Create dataset
    train_dataset = torch.utils.data.TensorDataset(torch.tensor(X).unsqueeze(1), torch.tensor(Y.squeeze(), dtype=torch.long))
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.bs, num_workers=args.workers, pin_memory=True, shuffle=True, drop_last=True)

    # Data augmentation
    time_steps = int(args.sr / (1000/args.len) / args.hoplen)
    rc = RandomCrop(n_mels=args.nmels, time_steps=time_steps, tcrop_ratio=args.tratio)
    resize = Resize(n_mels=args.nmels, time_steps=time_steps)
    awgn = GaussNoise(stdev_gen=args.noise, device=args.device)
    comp = Compander(comp_alpha=args.comp)
    mix = MixRandom(device=args.device)
    fshift = FreqShift(Fshift=args.fshift)

    # Prepare views
    transform1 = nn.Sequential(mix, fshift, rc, resize, comp, awgn) # only one branch has mixing with a background sound
    transform2 = nn.Sequential(fshift, rc, resize, comp, awgn)

    # Prepare model
    encoder = ResNet(method=args.method)
    print(summary(encoder))

    # Launch training
    model = train_scl(encoder, train_loader, transform1, transform2, args)