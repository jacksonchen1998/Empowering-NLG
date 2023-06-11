import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import gptactor, scoremodel
from dataprepare import collectRL, read
from tqdm import tqdm
from transformers import GPT2Tokenizer, pipeline, RobertaTokenizer
from transformers.utils import logging
from torch.nn.utils.rnn import pad_sequence

logging.set_verbosity_error()
'''
1. pipeline sampling. text = LM(article)
2. get reward r = S([article:text])
3. policy gradient. -r*log(LM([article:text]))
'''

device='cuda:0' if torch.cuda.is_available() else 'cpu'
def train(epoch, model:torch.nn.Module, oldmodel:torch.nn.Module, orimodel:torch.nn.Module, scorefunc:torch.nn.Module, bar:tqdm, optim:torch.optim.AdamW, warmup:torch.optim.lr_scheduler.LambdaLR):
    '''
    bar: tqdm(DataLoader)
    '''
    global mv_reward, alpha
    model.eval()
    losses=0
    count=0
    clip_range=0.1
    replay=[]
    length=[]
    token_length=[]
    for texts in bar:
        #update
        count+=1
        '''
        tokens:(bs, len)
        masks:(bs, len)
        target:(bs, len)
        '''
        optim.zero_grad()

        #ready for gpt text
        RL_texts=[gpt_tokenizer.bos_token + text + gpt_tokenizer.eos_token for text in texts]

        #get response, this is a bottle neck, very very slow
        with torch.no_grad():
            response_s=pipe(RL_texts, max_new_tokens=new_len)
        replay.extend([[t, r_n[0]['generated_text']] for t,r_n in zip(texts, response_s)])
        length.extend([len(text) for text in RL_texts])
        token_length.extend([len(gpt_tokenizer(RL_text, return_tensors="pt")['input_ids'][0]) for RL_text in RL_texts])
        assert len(token_length)==len(replay) and len(replay)==len(length)
        while len(replay)>128:
            token_length.pop(0)
            replay.pop(0)
            length.pop(0)
        shuffle_item=list(zip(replay,length,token_length))
        minibatch=torch.utils.data.DataLoader(shuffle_item, batch_size=8, shuffle=True)
        for batch,(minireplay, minilength, minitoken_length) in enumerate(minibatch):
            if batch==8:
                break
            actions=[]#all response for one texts. (num, len), and feed into bert
            for i,(text, action) in enumerate(zip(*minireplay)):
                actions.append(text+bert_tokenizer.eos_token+bert_tokenizer.bos_token+action[minilength[i]:])

            c_out_id=bert_tokenizer(actions, return_tensors="pt", padding=True, truncation=True, max_length=512)
            c_ids =c_out_id['input_ids'].to(device)
            c_mask=c_out_id['attention_mask'].to(device)
            with torch.no_grad():
                reward=scorefunc(c_ids, c_mask)#num_return reward
                reward=torch.sigmoid(torch.log(reward+1e-8))*2

            actions=[]#all response for one texts. (num, len), and feed into gpt
            for i,(text, action) in enumerate(zip(*minireplay)):
                actions.append(action+gpt_tokenizer.eos_token)
            out=gpt_tokenizer(actions, return_tensors="pt", padding=True, truncation=True, max_length=512)
            ids =out['input_ids'].to(device)
            masks=out['attention_mask'].to(device)
            # reward = (ids==gpt_tokenizer(text_target=' the', return_tensors="pt")['input_ids'].squeeze().to(device)).float().sum(1)
            #get distribution
            prob, value = model(ids , masks) #(num, len, vocab), (num, len, 1)

            with torch.no_grad():
                oldprob, oldvalue=oldmodel(ids , masks)
                oriprob, orivalue=orimodel(ids , masks)

            log_prob=F.log_softmax(prob-prob.max(dim=-1,keepdim=True).values, dim=-1)#(num_return, len, vocab)
            log_oldprob=F.log_softmax(oldprob-oldprob.max(dim=-1,keepdim=True).values, dim=-1)#(num_return, len, vocab)
            log_oriprob=F.log_softmax(oriprob-oriprob.max(dim=-1,keepdim=True).values, dim=-1)



            loss=0
            for i in range(len(minireplay)):
                #for num_return
                mask=masks[i,minitoken_length[i]-1:]
                log_pi= log_prob[i,minitoken_length[i]-1:][mask==1]
                log_oldpi=log_oldprob[i,minitoken_length[i]-1:][mask==1]
                log_oripi=log_oriprob[i,minitoken_length[i]-1:][mask==1]
                v=value[i,minitoken_length[i]-1:][:,0][mask==1]
                oldv=oldvalue[i,minitoken_length[i]-1:][:,0][mask==1]
                act=ids[i,minitoken_length[i]-1:][mask==1]
                mask=mask[mask==1]
                # print(log_pi.shape)#[282, 50257]
                # print(log_oldpi.shape)#[282, 50257]
                # print(v.shape)#[282]
                # print(act.shape)#[282]
                # print(mask.shape)#[282]

                # loss+=100*torch.nn.functional.cross_entropy(torch.exp(log_pi),torch.ones([len(log_pi)],device=log_pi.device,dtype=torch.long)*gpt_tokenizer('The', return_tensors="pt")['input_ids'].squeeze())
                KL_loss=torch.nn.functional.kl_div(log_pi, log_oripi, log_target=True, reduction='none')
                gamma=1
                R=torch.ones_like(mask, device=device, dtype=torch.float)*reward[i].squeeze()
                # R=R-KL_loss.sum(1)
                ratio=torch.exp(log_pi[:-1][torch.arange(len(log_pi)-1), act[1:]]-log_oldpi[:-1][torch.arange(len(log_pi)-1), act[1:]])
                ratio_diff=(ratio.detach()).mean().item()
                adv=(mask*(R - oldv))[:-1]
                # adv=mask[:-1]*(0+gamma*oldv[1:]-oldv[:-1])
                # adv=(adv-adv.mean())/(adv.std()+1e-8)
                adv_update=mask*(R - v)
                gae=[]
                lastadv=0
                for a in adv.flip(0):
                    gae.append(a+0.95*gamma*lastadv)
                    lastadv=gae[-1]
                gae=torch.stack(gae).flip(0)
                # print(gae[act[1:]==262])
                # print(adv.shape)#(len)
                # print(gae.shape)#(len)
                # print(ratio.shape)#(len)
                # norm_adv=(gae-gae.mean())/(gae.std()+1e-8)
                policy_loss_1 = gae.detach() * ratio
                policy_loss_2 = gae.detach() * \
                    torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -torch.minimum(policy_loss_1, policy_loss_2)
                loss += policy_loss.mean()

                explo_loss=torch.mean(log_pi[torch.arange(len(log_pi)), act])
                loss+=0.01*explo_loss

                value_loss=F.smooth_l1_loss(adv_update, torch.zeros_like(adv_update, device=adv.device), beta=0.2, reduction='mean')
                loss+=0.01*value_loss

                KL_loss=KL_loss.sum(1).mean()
                loss+=alpha*KL_loss
                if KL_loss.item()>1:
                    alpha*=1.01
                elif KL_loss.item()<1:
                    alpha/=1.001
                alpha=min(10, alpha)

            loss.backward()
            losses = losses*(1-0.05)+policy_loss.mean().item()*0.05
            if mv_reward is None:
                mv_reward=torch.mean(reward).item()
            mv_reward = mv_reward*(1-0.05)+torch.mean(reward).item()*0.05

            if warmup.last_epoch<1000:
                warmup.step()
            optim.step()

        #log
        bar.set_postfix_str(f'loss: {losses:7.1e},'+
                    f'v_err:{(adv_update.abs().sum()/mask.sum()).item():5.3f},'+
                    f'ratio:{ratio_diff:3.2f},'+
                    f'KL:{KL_loss.item():3.2f},'+
                    f'reward: {mv_reward:3.2f}')
        reward_file.write(f'{mv_reward:4.3f}\n')
        reward_file.flush()
        loss_file.write(f'{np.mean(losses):4.3f}\n')
        loss_file.flush()
        if count>(8000/train_cfg['batch_size']):
            return np.mean(losses)
        if count%int(1000/train_cfg['batch_size'])==0:
            torch.save(model.state_dict(), 'current_RL.pt')
        if count%20==0:
            oldmodel.load_state_dict(model.state_dict())

    return np.mean(losses)


if __name__=='__main__':
    x_train, x_test, y_train, y_test=read('tweet_reply.json')
    # print(X_train[0], X_test[0])
    # print(y_train[0], y_test[0])
    reward_file=open('reward.log','a')
    loss_file=open('loss.log','a')
    c_func=collectRL(max_len=512)
    train_cfg={'batch_size':    8,
            'shuffle':       True,
            'num_workers':   4,
            'collate_fn':    c_func,
            }

    gpt_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    gpt_tokenizer.pad_token = gpt_tokenizer.eos_token
    bert_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    bert_eos_id=bert_tokenizer(bert_tokenizer.eos_token)['input_ids'][1]


    model=gptactor().to(device)
    model.load_state_dict(torch.load('gpt_RL_005.pt', map_location='cpu'), strict=False)
    oldmodel=gptactor().to(device)
    oldmodel.load_state_dict(model.state_dict())
    oldmodel.eval()

    orimodel=gptactor().to(device)
    orimodel.load_state_dict(torch.load('gptmodel_030.pt', map_location='cpu'), strict=False)
    orimodel.eval()
    pipe=pipeline('text-generation', model=oldmodel.net, tokenizer=gpt_tokenizer, device=device,
                max_length=512, num_return_sequences=1, top_k=0, top_p=0.9,
                temperature=1, num_beams=1, no_repeat_ngram_size=2,
                batch_size=4, early_stopping=False, pad_token_id = 50256 )

    assert id(pipe.model)==id(oldmodel.net)
    scorefunc=scoremodel()
    scorefunc.load_state_dict(torch.load('scoremodel_020.pt', map_location='cpu'))
    scorefunc.to(device)
    scorefunc.eval()

    max_c_len=512-int(512*0.7)

    train_loader=torch.utils.data.DataLoader(list(zip(x_train,y_train)), **train_cfg, persistent_workers =True)

    optim=torch.optim.AdamW([{'params':model.net.parameters(), 'lr':1e-6},
                             {'params':model.critic.parameters(),'lr':3e-4}], betas=[0.5, 0.98], weight_decay=0.01)
    warmup=torch.optim.lr_scheduler.LambdaLR(optim, lambda i: i/1000 if i<1000 else 1)
    lrsch=torch.optim.lr_scheduler.LinearLR(optim, start_factor=1, end_factor=1, total_iters=20)
    start=5
    num_epoch=20
    mv_reward=0
    alpha=0.01
    new_len=50
    for epoch in range(start+1,start+num_epoch+1):
        bar=tqdm(train_loader, ncols=0)
        bar.set_description_str(f'epoch:{epoch:03d}')
        loss=train(epoch,model, oldmodel, orimodel, scorefunc, bar, optim, warmup)
        lrsch.step()
        if epoch%1==0:
            torch.save(model.state_dict(), f'gpt_RL_{epoch:03d}.pt')


