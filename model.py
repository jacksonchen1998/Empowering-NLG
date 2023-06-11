
from transformers import RobertaModel
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import torch.nn as nn

class scoremodel(nn.Module):
    def __init__(self,) -> None:
        super().__init__()
        self.net=RobertaModel.from_pretrained("roberta-base")
        self.outlayer=nn.Sequential(nn.Dropout(),
                                    nn.Linear(768, 128),
                                    nn.LeakyReLU(inplace=True),
                                    nn.Linear(128, 1))
    def forward(self, ids, masks):
        '''
        x: (batch, len)

        output:(batch, 1)
        '''
        x=self.net(input_ids=ids, attention_mask=masks)

        x=self.outlayer(x['last_hidden_state'][...,0,:])
        return torch.sigmoid(x)


class gptmodel(nn.Module):
    def __init__(self,) -> None:
        super().__init__()
        self.net  = GPT2LMHeadModel.from_pretrained('gpt2')
    def forward(self, ids, masks, target):

        output = self.net(input_ids=ids, attention_mask=masks, labels=target)

        return output.logits, output.loss

class gptactor(nn.Module):
    def __init__(self,) -> None:
        super().__init__()
        self.net  = GPT2LMHeadModel.from_pretrained('gpt2')
        self.critic= nn.Sequential(nn.Linear(self.net.get_output_embeddings().in_features, 128),
                                   nn.LeakyReLU(inplace=True),
                                   nn.Linear(128, 1))
    def forward(self, ids, masks, target=None):
        '''
        input: id, masks:(bs, len)

        output: prob:(bs, len, vocab)
        '''
        out=[]
        output = self.net(input_ids=ids, attention_mask=masks, output_hidden_states =True, labels=target)
        value=self.critic(output.hidden_states[-1])
        out.append(output.logits)
        out.append(value)
        if target is not None:
            out.append(out.loss)
        return out
