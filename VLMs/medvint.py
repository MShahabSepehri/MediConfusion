import os
import csv
import json
import math
import torch
import difflib 
import argparse
import numpy as np
import transformers
from torch import nn
import tqdm.auto as tqdm
from typing import Optional
from transformers import Trainer
from torchvision import transforms
from torch.nn import functional as F 
from torch.utils.data import DataLoader 
from transformers import LlamaTokenizer
from dataclasses import dataclass, field
from .MedVInT.llama.vqa_model import Binary_VQA_Model
from .MedVInT.dataset.randaugment import RandomAugment

# EMBED_DIM = 768
# IMAGE_RES = 512
# PRETRAINED_TOKENIZER = "../../LLAMA_Model/tokenizer"
# PRETRAINED_MODEL = "../../LLAMA_Model/llama-7b-hf"
# IMAGE_ENCODER = "CLIP"
# PMCCLIP_PRETRAINED = "./models/pmc_clip/checkpoint.pt"
# CLIP_PRETRAINED = "openai/clip-vit-base-patch32"
# CKP = "./Results/VQA_lora_noclip/vqa/checkpoint-6500"

@dataclass
class ModelArguments:
    embed_dim: Optional[int] = field(default=768)
    pretrained_tokenizer:  Optional[str] = field(default="../../LLAMA_Model/tokenizer")
    pretrained_model: Optional[str] = field(default="../../LLAMA_Model/llama-7b-hf")
    image_encoder: Optional[str] = field(default="CLIP")
    pmcclip_pretrained: Optional[str] = field(default="./models/pmc_clip/checkpoint.pt")
    clip_pretrained: Optional[str] = field(default="openai/clip-vit-base-patch32")
    ckp: Optional[str] = field(default="./Results/VQA_lora_noclip/vqa/checkpoint-6500")

def str_similarity(str1, str2):
    seq = difflib.SequenceMatcher(None, str1, str2)
    return seq.ratio()
 
def find_most_similar_index(str_list, target_str):
    """
    Given a list of strings and a target string, returns the index of the most similar string in the list.
    """
    # Initialize variables to keep track of the most similar string and its index
    most_similar_str = None
    most_similar_index = None
    highest_similarity = 0
    
    # Iterate through each string in the list
    for i, str in enumerate(str_list):
        # Calculate the similarity between the current string and the target string
        similarity = str_similarity(str, target_str)
        
        # If the current string is more similar than the previous most similar string, update the variables
        if similarity > highest_similarity:
            most_similar_str = str
            most_similar_index = i
            highest_similarity = similarity
    
    # Return the index of the most similar string
    return most_similar_index
  
def get_generated_texts(label, outputs, tokenizer):
    #1,256
    outputs = outputs[label!=0][1:-1]
    generated_text = tokenizer.decode(outputs)
    return generated_text

def load_model(model_args):

    ckp = model_args.ckp + '/pytorch_model.bin'
    model = Binary_VQA_Model(model_args)
    model.load_state_dict(torch.load(ckp, map_location='cpu'))
    model = model.to('cuda')
    model.eval()

    tokenizer = LlamaTokenizer.from_pretrained('../../LLAMA_Model/tokenizer')
    special_tokens_dict = {'mask_token': "</s>",
                            'eos_token': "</s>",
                            'bos_token': "<s>",
                            'unk_token': "<unk>"}
    tokenizer.add_special_tokens(special_tokens_dict)
    tokenizer.pad_token_id=0

    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    image_transform = transforms.Compose([                        
            transforms.RandomResizedCrop([IMAGE_RES, IMAGE_RES],scale=(0.2, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            RandomAugment(2, 7, isPIL=True,augs=['Identity','Equalize','Brightness','Sharpness',#AutoContrast',
                                                 'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
            transforms.ToTensor(),
            normalize,
        ])   
    return model, tokenizer, image_transform


def encode_mlm(tokenizer, question_text, question_text_with_answer, mask_token= '</s>', pad_token='<unk>', eos_token = '</s>'):
        def measure_word_len(word):
            token_ids = tokenizer.encode(word)
            # tokens = [tokenizer.decode(x) for x in token_ids]
            return len(token_ids) - 1
        
        question_text_with_answer_tokens = question_text_with_answer.split()
        question_text_tokens = question_text.split()
        bert_input_tokens = []
        output_mask = []
        bert_label_tokens = []  # 被 mask 的保留原词, 否则用 [PAD] 代替
        
        for i, token in enumerate(question_text_with_answer_tokens):
            if i < len(question_text_tokens):
                word_len = measure_word_len(token)
                bert_input_tokens += [token]
                bert_label_tokens += [pad_token] * word_len
                output_mask += [0] * word_len
            else:
                word_len = measure_word_len(token)
                bert_input_tokens += [mask_token] * word_len
                bert_label_tokens += [token]
                output_mask += [1] * word_len
        bert_input_tokens += [eos_token]
        bert_label_tokens += [eos_token]
        bert_input = ' '.join(bert_input_tokens)
        bert_label = ' '.join(bert_label_tokens)
        return bert_input, bert_label


def ask_question(model, tokenizer, question, image, max_length=256):
    image = image.to('cuda')
    answer = '-'
    answer_label = '-'
    question_text = question
    question_text_with_answer = question_text + answer_label
    bert_input, bert_label = encode_mlm(tokenizer, question_text, question_text_with_answer)

    encoded_input = tokenizer(bert_input, add_special_tokens=True, padding='max_length', truncation=True, max_length=max_length,return_tensors="pt")
    encoded_label = tokenizer(bert_label, add_special_tokens=True, padding='max_length', truncation=True, max_length=max_length,return_tensors="pt")
        
    label = encoded_label['input_ids'].to('cuda')[:,0,:]
    question_inputids = encoded_input['input_ids'].to('cuda')[:,0,:]
    question_attenmask = encoded_input['attention_mask'].to('cuda')[:,0,:]
    with torch.no_grad():
        outputs = model(image, question_inputids, question_attenmask)
    raise ValueError (outputs)
    loss = F.nll_loss(outputs.transpose(1, 2), label, ignore_index=0)
    
    generated_texts = get_generated_texts(label, outputs.argmax(-1), tokenizer)
    Choice_A = sample['Choice_A'][0]
    Choice_B = sample['Choice_B'][0]
    Choice_C = sample['Choice_C'][0]
    Choice_D = sample['Choice_D'][0] 
    Answer_label = sample['Answer_label'][0] 
    # print(loss,Answer_label,generated_texts)
    Choice_list = [Choice_A, Choice_B, Choice_C, Choice_D]
    index_pred = find_most_similar_index(['A','B','C','D'], generated_texts)
    index_label  = find_most_similar_index(['A','B','C','D'], Answer_label)
    corret = 0
    if index_pred == index_label:
        ACC = ACC +1
        corret = 1 
    writer.writerow([img_path,Answer_label,generated_texts,corret])
    cc = cc + 1