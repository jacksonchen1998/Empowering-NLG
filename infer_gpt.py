import numpy as np
import torch
import torch.nn as nn
from model import gptactor,scoremodel
from dataprepare import collectRL, read
from tqdm import tqdm
from transformers import GPT2Tokenizer, pipeline, RobertaTokenizer
from transformers.utils import logging
from torch.distributions import Categorical

logging.set_verbosity_error()


device='cuda:0' if torch.cuda.is_available() else 'cpu'
# device='cpu'
class mypipeline(nn.Module):
    def __init__(self, model, tokenizer, k=100):
        super().__init__()
        self.model=model
        self.tokenizer=tokenizer
        self.eos_id=tokenizer(tokenizer.eos_token)['input_ids'][0]
        self.k=k

    @torch.no_grad()
    def forward(self, ids):
        '''
        ids:(len)
        out:(len+response)
        '''

        while(True):
            out,value=model(ids, torch.ones_like(ids))
            val, id=torch.topk(out[-1], k=self.k)
            m=Categorical(logits=val)
            act=torch.tensor([id[m.sample()]]).to(out.device)

            ids=torch.cat([ids,act])
            if act[0]==self.eos_id:
                return ids




if __name__=='__main__':
    x_train, x_test, y_train, y_test=read('tweet_reply.json')

    gpt_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    bert_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    bert_eos_id=bert_tokenizer(bert_tokenizer.eos_token)['input_ids'][1]

    gpt_tokenizer.pad_token=gpt_tokenizer.eos_token
    model=gptactor().to(device)
    model.load_state_dict(torch.load('gpt_RL_003.pt', map_location='cpu'), strict=False)
    model.eval()
    scorefunc=scoremodel()
    scorefunc.load_state_dict(torch.load('scoremodel_020.pt', map_location='cpu'))
    scorefunc.to(device)
    scorefunc.eval()


    pipe=pipeline('text-generation', model=model.net, tokenizer=gpt_tokenizer, device=device,
                max_length=512, num_return_sequences=5, top_k=50, top_p=0.95,
                temperature=2., num_beams=5, no_repeat_ngram_size=2,
                batch_size=8, early_stopping=False, pad_token_id = 50256, max_new_tokens=150 ,return_full_text=True)
    # pipe=mypipeline(model, gpt_tokenizer)
    c_func=collectRL(max_len=512)
    train_cfg={'batch_size':    1,
            'shuffle':       False,
            'num_workers':   0,
            'collate_fn':    c_func,
            }

    train_loader=torch.utils.data.DataLoader(list(zip(x_train,y_train))[:20], **train_cfg)
    # train_loader=[["Anyone not love machine learning?"],
    #               ["Macbook is bad."]]
    rewards=[]
    for texts in train_loader:
        RL_texts=[gpt_tokenizer.bos_token + text + gpt_tokenizer.eos_token for text in texts]
        length=[len(text) for text in RL_texts]
        token_length=[len(gpt_tokenizer(RL_text, return_tensors="pt")['input_ids'][0]) for RL_text in RL_texts]

        #get response, this is a bottle neck, very very slow
        with torch.no_grad():
            response_s=pipe(RL_texts)#tem 0 is greedy

        #for all input texts
        for i in range(len(texts)):
            actions=[]#all response for one texts. (num, len), and feed into bert
            for action in response_s[i]:#num_return
                actions.append(texts[i]+bert_tokenizer.eos_token+bert_tokenizer.bos_token+action['generated_text'][length[i]:])

            c_out_id=bert_tokenizer(actions, return_tensors="pt", padding=True, truncation=True, max_length=512)
            c_ids =c_out_id['input_ids']
            c_mask=c_out_id['attention_mask']
            with torch.no_grad():
                reward=scorefunc(c_ids.to(device), c_mask.to(device))#num_return reward
            rewards.extend(reward.squeeze().cpu())
            print('<article input>',texts[i],'<article end>')
            for text, r in zip(response_s[i],reward):
                print('<response>',text['generated_text'][length[i]:],'<response end>',f'{r[0].item():4.2f}')

    print(np.mean(rewards))
