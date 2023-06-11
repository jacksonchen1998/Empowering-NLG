import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from transformers import RobertaTokenizer, GPT2Tokenizer
from datasets import load_dataset
import json
import numpy as np

class collectscore():
    def __init__(self, max_len=512, tokenizer=None):
        assert tokenizer is not None

        self.tokenizer = tokenizer

        self.max_p_len=int(max_len*0.7)
        self.max_c_len=max_len-self.max_p_len

        self.bos_id=self.tokenizer(self.tokenizer.bos_token)['input_ids'][1]#0
        self.eos_id=self.tokenizer(self.tokenizer.eos_token)['input_ids'][1]#2
        self.sep_id=self.tokenizer(self.tokenizer.sep_token)['input_ids'][1]#2
        self.cls_id=self.tokenizer(self.tokenizer.cls_token)['input_ids'][1]#0
        self.pad_id=self.tokenizer(self.tokenizer.pad_token)['input_ids'][1]#1

    def __call__(self, batch):
        '''
        input:(bs,( (2,(len,len)),1) )
        output: <bos>parent<eos><bos>child<eos>

        '''

        parents,texts, scores=[],[],[]

        for [parent, text], score in batch:
            parents.append(parent)
            texts.append(text)
            scores.append(score)
        scores=torch.tensor(np.array(scores), dtype=torch.float)
        #scores rescale


        tokens=[]
        masks=[]
        for p, c in zip(parents,texts):
            p_out=self.tokenizer(p, return_tensors="pt", truncation=True, max_length=self.max_p_len)
            #print(p_out.input_ids.shape)#(1, len)
            p_ids =p_out['input_ids'][0]
            if p_ids[-1] != self.eos_id:
                p_ids[-1] = self.eos_id
            p_mask=p_out['attention_mask'][0]

            c_out=self.tokenizer(c, return_tensors="pt", truncation=True, max_length=self.max_c_len)
            c_ids =c_out['input_ids'][0]
            if c_ids[-1] != self.eos_id:
                c_ids[-1] = self.eos_id
            c_mask=c_out['attention_mask'][0]

            ids=torch.cat([p_ids,c_ids])

            tokens.append(ids)
            masks.append(torch.cat([p_mask,c_mask]))

        tokens=pad_sequence(tokens, batch_first=True, padding_value=self.pad_id)
        masks=pad_sequence(masks, batch_first=True)

        return tokens, masks, scores


class collectgpt():
    def __init__(self, max_len=512, tokenizer=None):
        assert tokenizer is not None

        self.tokenizer = tokenizer

        self.max_p_len=int(max_len*0.7)
        self.max_c_len=max_len-self.max_p_len

        self.bos_id=self.tokenizer(self.tokenizer.bos_token)['input_ids'][0]
        self.eos_id=self.tokenizer(self.tokenizer.eos_token)['input_ids'][0]
        self.pad_id=self.eos_id


    def __call__(self, batch):
        '''
        input:(bs,( (2,(len,len)),1) )
        output: <bos>parent<eos>child<eos>
        '''

        parents,texts=[],[]

        for [parent, text], score in batch:
            parents.append(parent)
            texts.append(text)
        #scores rescale


        tokens=[]
        masks=[]
        targets=[]
        for p, c in zip(parents,texts):
            p_out=self.tokenizer(p, return_tensors="pt", truncation=True, max_length=self.max_p_len-2)
            c_out=self.tokenizer(c, return_tensors="pt", truncation=True, max_length=self.max_c_len-1)
            #print(p_out.input_ids.shape)#(1, len)

            p_ids =torch.cat([torch.tensor([self.bos_id]),p_out['input_ids'][0],torch.tensor([self.eos_id])])
            if p_ids[-1] != self.eos_id:
                p_ids[-1] = self.eos_id
            p_mask=torch.ones_like(p_ids)

            c_ids =torch.cat([c_out['input_ids'][0],torch.tensor([self.eos_id])])
            if c_ids[-1] != self.eos_id:
                c_ids[-1] = self.eos_id
            c_mask=torch.ones_like(c_ids)

            ids=torch.cat([p_ids,c_ids])

            tokens.append(ids)
            masks.append(torch.cat([p_mask,c_mask]))
            targets.append(torch.cat([torch.ones_like(p_ids)*-100, c_ids]))

        tokens=pad_sequence(tokens, batch_first=True, padding_value=self.pad_id)
        masks=pad_sequence(masks, batch_first=True)
        targets=pad_sequence(targets, batch_first=True, padding_value=-100)
        # print( tokens.shape, masks.shape, targets.shape)#check OK
        return tokens, masks, targets


class collectRL():
    def __init__(self, max_len=512):

        self.gpt_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.bert_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

        self.max_p_len=int(max_len*0.7)
        self.max_c_len=max_len-self.max_p_len
        self.bos_id=self.bert_tokenizer(self.bert_tokenizer.bos_token)['input_ids'][1]#0
        self.eos_id=self.bert_tokenizer(self.bert_tokenizer.eos_token)['input_ids'][1]#2
        self.pad_id=self.bert_tokenizer(self.bert_tokenizer.pad_token)['input_ids'][1]#1



    def __call__(self, batch):
        '''
        input:(bs,( (2,(len,len)),1) )
        output:
            parent raw text
            parent RL text <bos>parent<eos>
            parent bert id <bos>parent<eos>
            mask bert

        while get gpt output
        gpt tokenizer -> <bos>parent<eos>child<eos>
        bert tokenizer ->  <bos>parent<eos><bos>child<eos>

        '''

        parents,texts=[],[]

        for [parent, text], score in batch:
            parents.append(parent)


        out=self.bert_tokenizer(parents, return_tensors="pt", padding=True, truncation=True, max_length=self.max_p_len)
        texts=[self.bert_tokenizer.decode(text, skip_special_tokens=True, clean_up_tokenization_spaces=False) for text in out['input_ids']]
        bert_tokens =out['input_ids']
        bert_masks=out['attention_mask']

        return texts



def read(path):
    dataset= load_dataset('json', data_files=path)
    cols = ['main_tweet','reply','reply_likes']
    dataset=list(zip(*[dataset['train'][c] for c in cols]))
    dataset=np.unique(dataset,axis=0)
    x, y=dataset[:,[0,1]], dataset[:,2]
    y=np.array([int(y) for y in y])
    # y= np.log(y+1)[:,None]
    print('ratio of like>=5:',np.sum(y>=5)/len(y))
    y= (y>=5)[:,None]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.01,random_state=43)
    return x_train, x_test, y_train, y_test


if __name__=='__main__':
    read('/home/DS/final/tweets.json')
