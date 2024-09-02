import torch
from PIL import Image
from einops import repeat
from .Med_Flamingo.src.utils import FlamingoProcessor
from open_flamingo import create_model_and_transforms

LLaMa_PATH = '/data/models/llama'
CHECKPOINT_PATH = '/data/models/med_flamingo/model.pt' 
IMAGE_PATH = '/data/datasets/pmc/images/'
FEW_SHOT_IMAGES = [
    'PMC1064097_F2.jpg',
    'PMC1065025_F1.jpg',
    'PMC1087855_F3.jpg',
]
FEW_SHOT_IMAGES = [(IMAGE_PATH + IM) for IM in FEW_SHOT_IMAGES]
FEW_SHOT_QUESTIONS = [
    'What radiological technique was used to confirm the diagnosis?',
    'What did the CT scan show?',
    'What is the purpose of the asterisk shown in the figure?',
]
FEW_SHOW_ANSWERS = [
    [1, 'Mammography'],
    [0, 'Cerebral edema'],
    [1, 'To indicate the normal lentoid shape of hypocotyl nuclei.']
]
FEW_SHOT_OPTIONS = [
    ['A: CT Scan', 'B: Mammography'],
    ['A: Cerebral edema', 'B: Intracranial hemorrhage'],
    ['A: To indicate the formation of lobes around the contracting nucleus.', 'B: To indicate the normal lentoid shape of hypocotyl nuclei.']
]

def load_model():
    model, image_processor, tokenizer = create_model_and_transforms(
        clip_vision_encoder_path="ViT-L-14",
        clip_vision_encoder_pretrained="openai",
        lang_encoder_path=LLaMa_PATH,
        tokenizer_path=LLaMa_PATH,
        cross_attn_every_n_layers=4
    )
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location='cuda'), strict=False)
    model.cuda()
    model.eval()
    processor = FlamingoProcessor(tokenizer, image_processor)
    return model, processor

def get_few_shot_sample(num, use_option):
    question = FEW_SHOT_QUESTIONS[num]
    answer = FEW_SHOW_ANSWERS[num]
    options = FEW_SHOT_OPTIONS[num]
    if use_option:
        return f'{question}\n{options[0]}\n{options[1]}\nAnswer: {options[answer[0]]}'
    return f'{question} Answer: {answer[1]}'

def process_prompt(prompt, use_option):
    for q in range(len(FEW_SHOT_QUESTIONS)):
        prompt = prompt.replace(f'**Q{q+1}**', get_few_shot_sample(q, use_option))
    return prompt

def ask_question(model, processor, image_path, question, max_new_tokens, use_forward, use_option):
    tmp = FEW_SHOT_IMAGES.copy()
    tmp.append(image_path)
    images = [Image.open(image_path) for image_path in tmp]
    pixels = processor.preprocess_images(images)
    pixels = repeat(pixels, 'N c h w -> b N T c h w', b=1, T=1)
    question = process_prompt(question, use_option)
    tokenized_data = processor.encode_text(question)
    if use_forward:
        return do_forward(model, processor, pixels, tokenized_data)
    return do_generation(model, processor, pixels, tokenized_data, max_new_tokens)


@torch.no_grad()
def do_forward(model, processor, pixels, tokenized_data):
    VALID_ANSWERS = ['A', 'B']
    TOKEN_ID_A = processor.tokenizer("A", return_tensors="pt", add_special_tokens=False).get('input_ids')
    TOKEN_ID_B = processor.tokenizer("B", return_tensors="pt", add_special_tokens=False).get('input_ids')
    device = 'cuda'
    outputs = model.forward(vision_x=pixels.to(device),
                            lang_x=tokenized_data["input_ids"].to(device),
                            attention_mask=tokenized_data["attention_mask"].to(device))
    logits = outputs.logits[0, -1, :]
    logits = logits.reshape(-1, 1)
    soft_max = torch.nn.Softmax(dim=0)
    probs = soft_max(torch.cat([logits[TOKEN_ID_A], logits[TOKEN_ID_B]][:len(VALID_ANSWERS)]))
    outputs = VALID_ANSWERS[probs.argmax().item()]
    return outputs

@torch.no_grad()
def do_generation(model, processor, pixels, tokenized_data, max_new_tokens):
    device = 'cuda'
    generated_text = model.generate(
        vision_x=pixels.to(device),
        lang_x=tokenized_data["input_ids"].to(device),
        attention_mask=tokenized_data["attention_mask"].to(device),
        max_new_tokens=max_new_tokens,
    )
    response = processor.tokenizer.decode(generated_text[0]).replace('<unk> ', '').strip()
    tmp = processor.tokenizer.decode(tokenized_data.get('input_ids')[0])
    response = response.replace(f'{tmp} ', '')
    while response[0] == ' ':
        response = response[1: ]
    return response