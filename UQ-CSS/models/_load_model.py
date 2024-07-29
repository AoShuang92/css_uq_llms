# This script exists just to load models faster
import functools
import os

import torch
from transformers import (AutoModelForCausalLM,
                          AutoModelForSequenceClassification, AutoTokenizer,
                          OPTForCausalLM)

from _settings import LLAMA_PATH

from transformers import CLIPProcessor, CLIPModel
import torch.nn as nn

from sentence_transformers import SentenceTransformer

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CLIPModel_Text(nn.Module):
    def __init__(self, device):
        super(CLIPModel_Text, self).__init__()
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.config = model.config
        self.text_model = model.text_model
        self.text_projection = model.text_projection
        self.logit_scale = model.logit_scale
        self.device= device

    def forward(
        self,
        input_ids = None,
        attention_mask = None,
        position_ids = None,
        return_loss = None,
        output_attentions = None,
        output_hidden_states = None,
        return_dict = None,
    ):

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states)

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        text_embeds = text_outputs[1]
        text_embeds = self.text_projection(text_embeds)

        # normalized features
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
        # print('text_embeds',text_embeds.shape)

        # cosine similarity as logits
        # prob_per_pair1 = text_embeds[0] * text_embeds[1] #torch.mm(text_embeds[0], text_embeds[1]) #* logit_scale\
        # prob_per_pair2 = text_embeds[2] * text_embeds[3]

        all_prob_pairs = []
        for i in range(int(text_embeds.shape[0]/2)):
            i = 2*i
            prob_per_pair_ = text_embeds[i] * text_embeds[i+1]
            all_prob_pairs.append(prob_per_pair_)

        return torch.stack(all_prob_pairs, dim = 0)




@functools.lru_cache()
def _load_pretrained_model(model_name, device, torch_dtype=torch.float16):
    if model_name.startswith('facebook/opt-'):
        model = OPTForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype)
    elif model_name == "microsoft/deberta-large-mnli":
        model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-large-mnli")#, torch_dtype=torch_dtype)
    elif model_name == 'llama-7b-hf' or model_name == 'llama-13b-hf':
        model = AutoModelForCausalLM.from_pretrained(os.path.join(LLAMA_PATH, model_name), cache_dir=None, torch_dtype=torch_dtype)
#         model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", device_map="auto")
    elif model_name == 'roberta-large-mnli':
         model = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli")#, torch_dtype=torch_dtype)
    
    elif model_name == "openai/clip-vit-base-patch32":
        model = CLIPModel_Text(device)
    
    elif model_name == "openai/all-MiniLM-L6-v2":
        model = SentenceTransformer("all-MiniLM-L6-v2")
        
    model.to(device)
    return model


@functools.lru_cache()
def _load_pretrained_tokenizer(model_name, use_fast=False):
    if model_name.startswith('facebook/opt-'):
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=use_fast)
    elif model_name == "microsoft/deberta-large-mnli":
        tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-large-mnli")
    elif model_name == "roberta-large-mnli":
        tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")
        
    elif model_name == "openai/clip-vit-base-patch32":
        tokenizer = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    elif model_name == "openai/all-MiniLM-L6-v2":
        tokenizer = model.encode(sentences)

    elif model_name == 'llama-7b-hf' or model_name == 'llama-13b-hf':
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(LLAMA_PATH, model_name), cache_dir=None, use_fast=use_fast)
#         tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
        tokenizer.eos_token_id = 2
        tokenizer.bos_token_id = 1
        tokenizer.eos_token = tokenizer.decode(tokenizer.eos_token_id)
        tokenizer.bos_token = tokenizer.decode(tokenizer.bos_token_id)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer