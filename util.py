import torch
from train import adjust_learning_rate
from tqdm import tqdm

def finetune_ce(encoder, train_loader, transform,  args):

    print(f"Training finetuning on {args.device}")
    
    loss_fn = torch.nn.CrossEntropyLoss()
    
    non_trainable_parameters = []
    if args.ft <= 2 :
        for param in encoder.layer1.parameters():
            non_trainable_parameters.append(param)
        if args.ft <= 1:
            for param in encoder.layer2.parameters():
                non_trainable_parameters.append(param)
            if args.ft <= 0:
                for param in encoder.layer3.parameters():
                    non_trainable_parameters.append(param)

    trainable_parameters = list( set(encoder.parameters()) - set(non_trainable_parameters) )

    if args.adam:
        optim = torch.optim.Adam([{'params':trainable_parameters}] ,lr=args.ftlr)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optim, gamma=args.gamma, step_size=args.step)
    else:
        optim = torch.optim.SGD(trainable_parameters, lr=args.ftlr, momentum=args.momentum, weight_decay=args.wd)
    
    num_epochs = args.ftepochs

    best_loss = None
    encoder = encoder.to(args.device)
    
    for epoch in range(1, num_epochs+1):
        tr_loss = 0.
        print("Epoch {}".format(epoch))
        if not args.adam:
            adjust_learning_rate(optim, args.ftlr, epoch, num_epochs+1)
        train_iterator = iter(train_loader)

        for batch in tqdm(train_iterator):
            optim.zero_grad()
            
            x, y = batch
            x = x.to(args.device)
            y = y.to(args.device)

            x = transform(x)

            _, x_out = encoder(x)

            loss = loss_fn(x_out, y)

            tr_loss += loss.item()

            loss.backward()
            optim.step()

            if best_loss is None:
                best_loss = loss

            if best_loss > loss:
                best_loss = loss
                best_state_dict = encoder.state_dict()
                
        tr_loss = tr_loss/len(train_iterator)
        print('Average train loss: {}'.format(tr_loss))
        if args.adam:
            lr_scheduler.step()

    encoder.load_state_dict(best_state_dict)
    return encoder
