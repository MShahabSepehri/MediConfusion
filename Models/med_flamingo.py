import torch
from PIL import Image
from einops import repeat
from .Med_Flamingo.src.utils import FlamingoProcessor
from open_flamingo import create_model_and_transforms

FEW_SHOT_IMAGES = [
    'PMC1064097_F2.jpg',
    'PMC1065025_F1.jpg',
    'PMC1087855_F3.jpg',
]
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

def load_model(LLaMa_PATH, CHECKPOINT_PATH, device='cuda'):
    model, image_processor, tokenizer = create_model_and_transforms(
        clip_vision_encoder_path="ViT-L-14",
        clip_vision_encoder_pretrained="openai",
        lang_encoder_path=LLaMa_PATH,
        tokenizer_path=LLaMa_PATH,
        cross_attn_every_n_layers=4
    )
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location='cpu'), strict=False)
    model.to(device=device)
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

def ask_question(model, processor, image_path, question, max_new_tokens, mode, IMAGE_DIR):
    tmp = [(f'{IMAGE_DIR}/{IM}') for IM in FEW_SHOT_IMAGES]
    tmp.append(image_path)
    images = [Image.open(image_path) for image_path in tmp]
    pixels = processor.preprocess_images(images)
    pixels = repeat(pixels, 'N c h w -> b N T c h w', b=1, T=1)

    if mode == 'prefix':
        return do_prefix_forward(model, question, pixels, processor)

    question = process_prompt(question, use_option=(mode in ['mc', 'greedy']))
    tokenized_data = processor.encode_text(question)
    if mode == 'greedy':
        return do_forward(model, processor, pixels, tokenized_data)
    elif mode in ['mc', 'gpt4']:
        return do_generation(model, processor, pixels, tokenized_data, max_new_tokens)


@torch.no_grad()
def do_forward(model, processor, pixels, tokenized_data):
    device = model.lang_encoder.device
    VALID_ANSWERS = ['A', 'B']
    TOKEN_IDs = [processor.tokenizer(x, return_tensors="pt", add_special_tokens=False).get('input_ids') for x in VALID_ANSWERS]
    outputs = model.forward(vision_x=pixels.to(device),
                            lang_x=tokenized_data["input_ids"].to(device),
                            attention_mask=tokenized_data["attention_mask"].to(device))
    logits = outputs.logits[0, -1, :].reshape(-1, 1)
    soft_max = torch.nn.Softmax(dim=0)
    probs = soft_max(torch.cat([logits[x] for x in TOKEN_IDs]))
    outputs = VALID_ANSWERS[probs.argmax().item()]
    return outputs

@torch.no_grad()
def do_generation(model, processor, pixels, tokenized_data, max_new_tokens):
    device = model.lang_encoder.device
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

@torch.no_grad()
def do_prefix_forward(model, problem, pixels, processor):
    PREFIX_PROMPT_TEMPLATE = process_prompt(problem.get('format'), use_option=False)
    scores = []
    questions = []
    qs = problem["question"]
    device = model.lang_encoder.device
    for option in [problem["option_A"], problem["option_B"]]:
        prompt = PREFIX_PROMPT_TEMPLATE.format(qs, option)
        prompt = process_prompt(prompt, use_option=False)
        tokenized_data = processor.encode_text(prompt)
        questions.append(prompt)
        answer_tokens = processor.tokenizer.encode(" " + option, add_special_tokens=False)[1:]
        num_answer_tokens = len(answer_tokens)
        input_ids = tokenized_data['input_ids']
        # try to find the answer tokens in input ids
        start_indices = []
        for i in range(input_ids.size(1) - num_answer_tokens + 1):
            if torch.equal(input_ids[0, i:i+num_answer_tokens], torch.tensor(answer_tokens)):
                start_indices.append(i)
        
        if len(start_indices) == 0:
            raise ValueError("Answer tokens not found in input_ids")
        answer_start = start_indices[-1]
        answer_start_from_back = answer_start - input_ids.size(1)
        with torch.inference_mode():
            outputs = model.forward(vision_x=pixels.to(device),
                            lang_x=tokenized_data["input_ids"].to(device),
                            attention_mask=tokenized_data["attention_mask"].to(device))
            # shift by 1 compared to input
            logits = outputs.logits[0, answer_start_from_back-1:answer_start_from_back-1+num_answer_tokens]
            probs = torch.nn.functional.softmax(logits, dim=-1)

            # Pick the probabilities corresponding to each of the answer tokens
            probs = torch.gather(probs, 1, torch.tensor(answer_tokens).to(device=device).unsqueeze(0))
            prefix_score = torch.prod(probs.pow(1/num_answer_tokens))
            scores.append(prefix_score.item())
    outputs = "A" if scores[0] > scores[1] else "B"
    return outputs