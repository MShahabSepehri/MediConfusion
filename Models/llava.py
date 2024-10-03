import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX


def load_model(model_id, device='cuda'):
    processor = LlavaNextProcessor.from_pretrained(model_id, device_map=device)
    model = LlavaNextForConditionalGeneration.from_pretrained(model_id) 
    model.to(device)
    return model, processor

def ask_question(model, processor, question, image, mode, temperature=0.2, top_p=None, num_beams=1, max_new_tokens=100):
    if mode == 'prefix':
        return do_prefix_forward(model, question, image, processor)
    
    conversation = [
        {
        "role": "user",
        "content": [
            {"type": "text", "text": question},
            {"type": "image"},
            ],
        },
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)

    if mode == 'greedy':
        outputs = do_forward(model, inputs, processor)
    elif mode in ['mc', 'gpt4']:
        outputs = do_generation(model, inputs, processor, temperature, top_p, num_beams, max_new_tokens)
    return outputs


@torch.no_grad()
def do_forward(model, inputs, processor):
    VALID_ANSWERS = ['A', 'B']
    TOKEN_IDs = [processor.tokenizer.encode(x, return_tensors="pt", add_special_tokens=False) for x in VALID_ANSWERS]

    with torch.inference_mode():
        out = model.forward(**inputs)
        
    logits = out.logits[0, -1, :]
    soft_max = torch.nn.Softmax(dim=0)
    probs = soft_max(torch.cat([logits[x] for x in TOKEN_IDs]))
    outputs = VALID_ANSWERS[probs.argmax().item()]
    return outputs

@torch.no_grad()
def do_generation(model, inputs, processor, temperature, top_p, num_beams, max_new_tokens):
    output = model.generate(**inputs, 
                            temperature=temperature, 
                            do_sample=(temperature > 0),
                            top_p=top_p, 
                            num_beams=num_beams, 
                            max_new_tokens=max_new_tokens)
    tmp = processor.decode(inputs.get('input_ids')[0], skip_special_tokens=True)
    txt = processor.decode(output[0], skip_special_tokens=True).replace(tmp, '')
    return txt

@torch.no_grad()
def do_prefix_forward(model, problem, image, processor):
    # python scripts/answering.py --vlm_name llava --mode prefix
    device = model.device
    scores = []
    questions = []
    qs = problem["question"]
    for option in [problem["option_A"], problem["option_B"]]:
        conv_template = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": f"{qs}"},
                ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": f"{option}"},
                ],
        }
        ]
        prompt = processor.apply_chat_template(conv_template, add_generation_prompt=True)
        questions.append(prompt)
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
        answer_tokens = processor.tokenizer.encode(" " + option, add_special_tokens=False)[1:]
        num_answer_tokens = len(answer_tokens)
        input_ids = inputs["input_ids"]
        # try to find the answer tokens in input ids
        start_indices = []
        for i in range(input_ids.size(1) - num_answer_tokens + 1):
            if torch.equal(input_ids[0, i:i+num_answer_tokens], torch.tensor(answer_tokens).to(device=device)):
                start_indices.append(i)
        
        if len(start_indices) == 0:
            raise ValueError("Answer tokens not found in input_ids")
        answer_start = start_indices[-1]
        answer_start_from_back = answer_start - input_ids.size(1)

        with torch.inference_mode():
            out = model(**inputs)
            # shift by 1 compared to input
            raise ValueError(out.logits.shape, answer_start, input_ids.size(1), answer_start_from_back)
            logits = out.logits[0, answer_start_from_back-1:answer_start_from_back-1+num_answer_tokens]
            probs = torch.nn.functional.softmax(logits, dim=-1)

            # Pick the probabilities corresponding to each of the answer tokens
            probs = torch.gather(probs, 1, torch.tensor(answer_tokens).to(device=device).unsqueeze(0))
            prefix_score = torch.prod(probs.pow(1/num_answer_tokens))
            scores.append(prefix_score.item())
    outputs = "A" if scores[0] > scores[1] else "B"
    return outputs