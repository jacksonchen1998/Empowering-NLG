import numpy as np 
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from model import scoremodel
from dataprepare import collectscore, read
from tqdm import tqdm
from transformers import RobertaTokenizer

    
device='cuda:0' if torch.cuda.is_available() else 'cpu'

if __name__=='__main__':
    model=scoremodel()
    model.load_state_dict(torch.load('/home/DL/Final/scoremodel_010.pt', map_location='cpu'))
    model.to(device)
    model.eval()
    c_func=collectscore(max_len=512, tokenizer=RobertaTokenizer.from_pretrained("roberta-base"))
    train_cfg={'batch_size':    32,
           'shuffle':       True,
           'num_workers':   4,
           'collate_fn':    c_func,
           }
    
    X_train, X_test, y_train, y_test=read()
    test_loader=torch.utils.data.DataLoader(list(zip(X_test,y_test)), **train_cfg)
    accs=[]
    with torch.no_grad():
        for x,masks, y in tqdm(test_loader, ncols=0):
            x=x.to(device)
            masks=masks.to(device)
            y=y.to(device)
            out=model(x, masks)
            accs.append(torch.mean(((out>0.5)==(y>0.5)).float()).item())
        print(np.mean(accs)) #010:0.959     015:0.964
        