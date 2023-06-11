import numpy as np
import torch
import torch.nn as nn
from model import gptmodel
from dataprepare import collectgpt, read
from tqdm import tqdm
from transformers import GPT2Tokenizer


device='cuda:0' if torch.cuda.is_available() else 'cpu'
def train(model:torch.nn.Module, bar:tqdm, optim:torch.optim.AdamW, warmup:torch.optim.lr_scheduler.LambdaLR):
    '''
    bar: tqdm(DataLoader)
    '''

    model.train()
    losses=3
    acc=[]
    for tokens, masks, target in bar:
        '''
        tokens:(bs, len)
        masks:(bs, len)
        target:(bs, len)
        '''
        tokens=tokens.to(device)
        masks=masks.to(device)
        target=target.to(device)

        out, loss = model(tokens, masks, target) #(bs, len, vocab), (bs, 1)

        #update
        optim.zero_grad()
        loss.backward()
        if warmup.last_epoch<1000:
            warmup.step()
        optim.step()

        #log
        losses = losses*(1-0.01)+loss.item()*0.01
        bar.set_postfix_str(f'loss: {losses:5.4f}')

    return np.mean(losses)


if __name__=='__main__':
    x_train, x_test, y_train, y_test=read('tweet_reply.json')


    model=gptmodel().to(device)
    # model.load_state_dict(torch.load('gptmodel_025.pt', map_location='cpu'))

    c_func=collectgpt(max_len=512, tokenizer = GPT2Tokenizer.from_pretrained('gpt2'))
    train_cfg={'batch_size':    8,
            'shuffle':       True,
            'num_workers':   4,
            'collate_fn':    c_func,
            }

    train_loader=torch.utils.data.DataLoader(list(zip(x_train,y_train))[:], **train_cfg)

    optim=torch.optim.AdamW(model.parameters(), lr= 1e-5, betas=[0.9,0.98],weight_decay=0.01)
    warmup=torch.optim.lr_scheduler.LambdaLR(optim, lambda i: i/1000 if i<1000 else 1)
    start=0
    num_epoch=30
    for epoch in range(start+1,start+num_epoch+1):
        bar=tqdm(train_loader, ncols=100)
        bar.set_description_str(f'epoch:{epoch:03d}')
        loss=train(model, bar, optim, warmup)

        if epoch%1==0:
            torch.save(model.state_dict(), f'gptmodel_{epoch:03d}.pt')


