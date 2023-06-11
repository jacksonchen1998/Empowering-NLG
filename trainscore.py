import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from model import scoremodel
from transformers import RobertaTokenizer
from dataprepare import collectscore, read
from tqdm import tqdm

device='cuda:0' if torch.cuda.is_available() else 'cpu'
@torch.no_grad()
def valid(model:torch.nn.Module, bar:tqdm,):
    model.eval()

    L1s=[]
    for tokens, masks, scores in bar:
        '''
        tokens:(bs, len)
        masks:(bs, len)
        scores:(bs, 1)
        '''
        tokens=tokens.to(device)
        masks=masks.to(device)
        scores=scores.to(device)

        out = model(tokens, masks) #(bs, 1)

        #log
        L1s.append((out-scores).abs().mean().item())

    return np.mean(L1s)

def train(model:torch.nn.Module, bar:tqdm, optim:torch.optim.AdamW, loss_func:torch.nn):
    '''
    bar: tqdm(DataLoader)
    '''

    model.train()
    losses=1
    mv_L1=0.5
    for i,(tokens, masks, scores) in enumerate(bar):
        '''
        tokens:(bs, len)
        masks:(bs, len)
        scores:(bs, 1)
        '''
        tokens=tokens.to(device)
        masks=masks.to(device)
        scores=scores.to(device)

        out = model(tokens, masks) #(bs, 1)

        #update
        loss = loss_func(out, scores)
        optim.zero_grad()
        loss.backward()
        optim.step()

        #logging
        L1=(out.detach()-scores).abs()
        losses = losses*(1-0.01)+loss.item()*0.01
        mv_L1 = mv_L1*(1-0.01)+torch.mean(L1).item()*0.01
        bar.set_postfix_str(f'loss: {losses:5.04f}, L1: {mv_L1: 4.03f}')

    return losses, mv_L1


if __name__=='__main__':
    x_train, x_test, y_train, y_test=read('tweet_reply.json')

    model=scoremodel().to(device)
    for param in model.net.parameters():
        param.requires_grad_(False)

    c_func=collectscore(max_len=512, tokenizer=RobertaTokenizer.from_pretrained("roberta-base"))
    train_cfg={'batch_size':    32,
            'shuffle':       True,
            'num_workers':   4,
            'collate_fn':    c_func,
            }

    train_loader=torch.utils.data.DataLoader(list(zip(x_train,y_train)), **train_cfg, persistent_workers=True )
    test_loader=torch.utils.data.DataLoader(list(zip(x_test,y_test)), **train_cfg, persistent_workers=True)

    optim=torch.optim.AdamW(model.parameters(), lr= 1e-3, betas=[0.9,0.98],weight_decay=0.01)
    loss_func=torch.nn.BCELoss()
    reward_file=open('error.log','w')
    start=0
    num_epoch=30
    for epoch in range(start+1,start+num_epoch+1):

        test_bar=tqdm(test_loader, ncols=0)
        valid_L1=valid(model, test_bar)

        bar=tqdm(train_loader, ncols=100)
        bar.set_description_str(f'epoch:{epoch:03d}')
        loss, L1=train(model, bar, optim, loss_func)

        if epoch>0:
            train_cfg['batch_size']=16
            train_loader=torch.utils.data.DataLoader(list(zip(x_train,y_train)), **train_cfg)
            optim.param_groups[0]['lr']=1e-5
            for param in model.net.parameters():
                param.requires_grad_(True)
        if epoch%5==0:
            torch.save(model.state_dict(), f'scoremodel_{epoch:03d}.pt')
        reward_file.write(f'{L1:4.3f}, {valid_L1:4.3f}\n')
        reward_file.flush()

