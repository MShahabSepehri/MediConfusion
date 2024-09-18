import torch
from PIL import Image
from utils import io_tools
from torchvision import transforms
from transformers import LlamaTokenizer
from .RadFM.multimodality_model import MultiLLaMAForCausalLM

ROOT = io_tools.get_root(__file__, 2)


def get_tokenizer(tokenizer_path, max_img_size=100, image_num=32):
    '''
    Initialize the image special tokens
    max_img_size denotes the max image put length and image_num denotes how many patch embeddings the image will be encoded to 
    '''
    if isinstance(tokenizer_path, str):
        image_padding_tokens = []
        text_tokenizer = LlamaTokenizer.from_pretrained(
            tokenizer_path,
        )
        special_token = {"additional_special_tokens": ["<image>","</image>"]}
        for i in range(max_img_size):
            image_padding_token = ""
            
            for j in range(image_num):
                image_token = "<image"+str(i*image_num+j)+">"
                image_padding_token = image_padding_token + image_token
                special_token["additional_special_tokens"].append("<image"+str(i*image_num+j)+">")
            image_padding_tokens.append(image_padding_token)
            text_tokenizer.add_special_tokens(
                special_token
            )
            ## make sure the bos eos pad tokens are correct for LLaMA-like models
            text_tokenizer.pad_token_id = 0
            text_tokenizer.bos_token_id = 1
            text_tokenizer.eos_token_id = 2    
    
    return  text_tokenizer,image_padding_tokens    

def combine_and_preprocess(question,image_list,image_padding_tokens):
    
    transform = transforms.Compose([                        
                transforms.RandomResizedCrop([512,512], scale=(0.8, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
            ])
    images  = []
    new_qestions = [_ for _ in question]
    padding_index = 0
    for img in image_list:
        img_path = img['img_path']
        position = img['position']
        
        image = Image.open(img_path).convert('RGB')   
        image = transform(image)
        image = image.unsqueeze(0).unsqueeze(-1) # c,w,h,d
        
        ## pre-process the img first
        target_H = 512 
        target_W = 512 
        target_D = 4 
        # This can be different for 3D and 2D images. For demonstration we here set this as the default sizes for 2D images. 
        images.append(torch.nn.functional.interpolate(image, size=(target_H,target_W,target_D)))
        
        ## add img placeholder to text
        new_qestions[position] = "<image>" + image_padding_tokens[padding_index] + "</image>" + new_qestions[position]
        padding_index += 1
    
    vision_x = torch.cat(images,dim = 1).unsqueeze(0) #cat tensors and expand the batch_size dim
    text = ''.join(new_qestions) 
    return text, vision_x, 
    
def load_model(model_path, device='cuda'):
    language_files_path = f'{ROOT}/Models/RadFM/Language_files'
    text_tokenizer, image_padding_tokens = get_tokenizer(language_files_path)
    model = MultiLLaMAForCausalLM(lang_model_path=language_files_path)
    ckpt = torch.load(model_path, map_location='cpu')
    model.load_state_dict(ckpt)
    model = model.to(device)
    model.eval()
    return model, text_tokenizer, image_padding_tokens
    
def ask_question(model, question, image_path, text_tokenizer, image_padding_tokens, mode, device):
    image =[
            {
                'img_path': image_path,
                'position': 0, #indicate where to put the images in the text string, range from [0,len(question)-1]
            }, # can add abitrary number of imgs
        ] 
    if mode == 'prefix':
        return do_prefix_forward(model, question, text_tokenizer, image_padding_tokens, image, device)
        
    text, vision_x = combine_and_preprocess(question, image, image_padding_tokens)
    with torch.no_grad():
        lang_x = text_tokenizer(text, max_length=2048, truncation=True, return_tensors="pt")['input_ids'].to(device)
        vision_x = vision_x.to(device)
    if mode == 'greedy':
        return do_forward(model, text_tokenizer, lang_x, vision_x)
    elif mode in ['mc', 'gpt4']:
        return do_generation(model, text_tokenizer, lang_x, vision_x)

@torch.no_grad()
def do_generation(model, text_tokenizer, lang_x, vision_x):
    generation = model.generate(lang_x, vision_x)
    generated_texts = text_tokenizer.batch_decode(generation, skip_special_tokens=True) 
    return generated_texts[0]

@torch.no_grad()
def do_forward(model, text_tokenizer, lang_x, vision_x):
    VALID_ANSWERS = ['A', 'B']
    TOKEN_IDs = [text_tokenizer.encode(x, return_tensors="pt", add_special_tokens=False) for x in VALID_ANSWERS]
    input_embedding, _= model.embedding_layer(lang_x, vision_x, key_words_query=None) 
    out = model.lang_model(inputs_embeds=input_embedding, attention_mask=None, labels=None)
    logits = out['logits'][0, -1, :]
    soft_max = torch.nn.Softmax(dim=0)
    probs = soft_max(torch.cat([logits[x] for x in TOKEN_IDs]))
    outputs = VALID_ANSWERS[probs.argmax().item()]
    return outputs

@torch.no_grad()
def do_prefix_forward(model, problem, text_tokenizer, image_padding_tokens, image, device):
    # PREFIX_PROMPT_TEMPLATE = "{} {}"
    PREFIX_PROMPT_TEMPLATE = problem.get('format')
    scores = []
    questions = []
    qs = problem["question"]

    for option in [problem["option_A"], problem["option_B"]]:
        prompt = PREFIX_PROMPT_TEMPLATE.format(qs, option)
        questions.append(prompt)
        text, vision_x = combine_and_preprocess(prompt, image, image_padding_tokens)
        with torch.no_grad():
            lang_x = text_tokenizer(text, max_length=2048, truncation=True, return_tensors="pt")['input_ids'].to(device)
            vision_x = vision_x.to(device=device)
        answer_tokens = text_tokenizer.encode(" " + option, add_special_tokens=False)[1:]
        num_answer_tokens = len(answer_tokens)

        # try to find the answer tokens in input ids
        start_indices = []
        for i in range(lang_x.size(1) - num_answer_tokens + 1):
            if torch.equal(lang_x[0, i:i+num_answer_tokens], torch.tensor(answer_tokens).to(device=device)):
                start_indices.append(i)
        
        if len(start_indices) == 0:
            raise ValueError("Answer tokens not found in input_ids")
        answer_start = start_indices[-1]
        answer_start_from_back = answer_start - lang_x.size(1)
        with torch.inference_mode():
            input_embedding, _= model.embedding_layer(lang_x, vision_x, key_words_query=None) 
            output = model.lang_model(inputs_embeds=input_embedding, attention_mask=None, labels=None)
            # shift by 1 compared to input
            logits = output['logits'][0, answer_start_from_back-1:answer_start_from_back-1+num_answer_tokens]
            probs = torch.nn.functional.softmax(logits, dim=-1)

            # Pick the probabilities corresponding to each of the answer tokens
            probs = torch.gather(probs, 1, torch.tensor(answer_tokens).to(device=device).unsqueeze(0))
            prefix_score = torch.prod(probs.pow(1/num_answer_tokens))
            scores.append(prefix_score.item())
    outputs = "A" if scores[0] > scores[1] else "B"
    return outputs