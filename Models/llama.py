import torch
import requests
from PIL import Image
from huggingface_hub import login
# from accelerate import Accelerator
from transformers import MllamaForConditionalGeneration, AutoProcessor

def load_model(token, model_id, device='cuda'):
    if token is not None:
        login(token)
    else:
        login()
    model = MllamaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    processor = AutoProcessor.from_pretrained(model_id)
    # model.to(device)
    return model, processor


def ask_question(model, processor, question, image, mode, temperature=0.2, top_p=None, num_beams=1, max_new_tokens=100):
    if mode == 'prefix':
        return do_prefix_forward(model, question, image, processor)
    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": question}
        ]}
    ]
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(image, input_text, add_special_tokens=False, return_tensors="pt").to(model.device)

    if mode == 'greedy':
        outputs = do_forward(model, inputs, processor)
    elif mode in ['mc', 'gpt4']:
        outputs = do_generation(model, inputs, processor, temperature, top_p, num_beams, max_new_tokens)
    return outputs


@torch.no_grad()
def do_generation(model, inputs, processor, temperature, top_p, num_beams, max_new_tokens):
    output = model.generate(**inputs, 
                            temperature=temperature, 
                            do_sample=(temperature > 0),
                            top_p=top_p, 
                            num_beams=num_beams, 
                            max_new_tokens=max_new_tokens)
    # raise ValueError(inputs.get('input_ids'), output[0])
    tmp = processor.decode(inputs.get('input_ids')[0], skip_special_tokens=True)
    txt = processor.decode(output[0], skip_special_tokens=True).replace(tmp, '')
    return txt

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
def do_prefix_forward(model, problem, image, processor):
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
                {"type": "text", "text": f"Question: {qs}"},
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
        answer_tokens = processor.tokenizer.encode(option, add_special_tokens=False)
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
            logits = out.logits[0, answer_start_from_back-1:answer_start_from_back-1+num_answer_tokens]
            probs = torch.nn.functional.softmax(logits, dim=-1)

            # Pick the probabilities corresponding to each of the answer tokens
            probs = torch.gather(probs, 1, torch.tensor(answer_tokens).to(device=device).unsqueeze(0))
            prefix_score = torch.prod(probs.pow(1/num_answer_tokens))
            scores.append(prefix_score.item())
    outputs = "A" if scores[0] > scores[1] else "B"
    return outputs