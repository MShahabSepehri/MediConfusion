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
from dataclasses import dataclass, field
from MedVINT.llama.vqa_model import Binary_VQA_Model

@dataclass
class ModelArguments:
    embed_dim: Optional[int] = field(default=768)
    pretrained_tokenizer:  Optional[str] = field(default="../../LLAMA_Model/tokenizer")
    pretrained_model: Optional[str] = field(default="../../LLAMA_Model/llama-7b-hf")
    image_encoder: Optional[str] = field(default="CLIP")
    pmcclip_pretrained: Optional[str] = field(default="./models/pmc_clip/checkpoint.pt")
    clip_pretrained: Optional[str] = field(default="openai/clip-vit-base-patch32")
    ckp: Optional[str] = field(default="./Results/VQA_lora_noclip/vqa/checkpoint-6500")
    

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    output_dir: Optional[str] = field(default="./Results")
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    logging_dir: Optional[str] = field(default="./logs")
    logging_steps: Optional[int] = field(default=50)
    

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
  
def get_generated_texts(label,outputs,tokenizer):
    #1,256
    outputs = outputs[label!=0][1:-1]
    generated_text = tokenizer.decode(outputs)
    return generated_text

def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    print("Setup Model")
    ckp = model_args.ckp + '/pytorch_model.bin'
    model = Binary_VQA_Model(model_args)
    model.load_state_dict(torch.load(ckp, map_location='cpu'))
    
    ACC = 0
    cc = 0
    
    print("Start Testing")
    
    model = model.to('cuda')
    model.eval()
    with open(os.path.join(training_args.output_dir,'result.csv'), mode='w') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['Figure_path','Pred','Label','Correct'])
        for sample in tqdm.tqdm(Test_dataloader):
            img_path = sample['image_path']
            image = sample['image'].to('cuda')
            label = sample['label'].to('cuda')[:,0,:]
            question_inputids = sample['encoded_input_ids'].to('cuda')[:,0,:]
            question_attenmask = sample['encoded_attention_mask'].to('cuda')[:,0,:]
            with torch.no_grad():
                outputs = model(image,question_inputids,question_attenmask)# 
            loss = F.nll_loss(outputs.transpose(1, 2), label, ignore_index=0)
            
            generated_texts = get_generated_texts(label,outputs.argmax(-1),Test_dataset.tokenizer)
            Choice_A = sample['Choice_A'][0]
            Choice_B = sample['Choice_B'][0]
            Choice_C = sample['Choice_C'][0]
            Choice_D = sample['Choice_D'][0] 
            Answer_label = sample['Answer_label'][0] 
            # print(loss,Answer_label,generated_texts)
            Choice_list = [Choice_A, Choice_B, Choice_C, Choice_D]
            index_pred = find_most_similar_index(['A','B','C','D'],generated_texts)
            index_label  = find_most_similar_index(['A','B','C','D'],Answer_label)
            corret = 0
            if index_pred == index_label:
                ACC = ACC +1
                corret = 1 
            writer.writerow([img_path,Answer_label,generated_texts,corret])
            cc = cc + 1
        print(ACC/cc)  
        writer.writerow([ACC/cc])


if __name__ == "__main__":
    main()
    