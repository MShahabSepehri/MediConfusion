import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig


def load_model(model_id, device='cuda'):
    processor = AutoProcessor.from_pretrained(
        'allenai/Molmo-7B-D-0924',
        trust_remote_code=True,
        torch_dtype='auto',
        device_map=device
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype='auto',
        device_map=device
    )
    return model, processor

def ask_question(model, question, image_path, processor, num_beams, max_length, top_p, temperature, mode):
    image = [Image.open(image_path).convert("RGB")]
    if mode == 'prefix':
        return do_prefix_forward(model, question, image, processor)
    inputs = processor.process(images=image, text=question)
    inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}
    if mode == 'greedy':
        return do_forward(model, processor, inputs)
    elif mode in ['mc', 'gpt4']:
        return do_generation(model, 
                            processor, 
                            inputs,
                            num_beams=num_beams,
                            top_p=top_p,
                            temperature=temperature,
                            max_new_tokens=max_length)

@torch.no_grad()
def do_generation(model, 
                  processor, 
                  inputs,
                  num_beams, 
                  top_p,
                  temperature, 
                  max_new_tokens):
    gc = GenerationConfig(
        do_sample=(temperature > 0),
        num_beams=num_beams, 
        top_p=top_p, 
        temperature=temperature, 
        max_new_tokens=max_new_tokens,
        stop_strings="<|endoftext|>",
    )
    output = model.generate_from_batch(inputs, gc, tokenizer=processor.tokenizer)
    generated_tokens = output[0, inputs['input_ids'].size(1):]
    generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return generated_text

@torch.no_grad()
def do_forward(model, processor, inputs):
    VALID_ANSWERS = ['A', 'B']
    TOKEN_IDs = [processor.tokenizer(x, return_tensors="pt", add_special_tokens=False).get('input_ids') for x in VALID_ANSWERS]
    logits = model.forward(**inputs).logits
    logits = logits[0, -1, :]
    logits = logits.reshape(-1, 1)
    soft_max = torch.nn.Softmax(dim=0)
    probs = soft_max(torch.cat([logits[x] for x in TOKEN_IDs][:len(VALID_ANSWERS)]))
    outputs = VALID_ANSWERS[probs.argmax().item()]
    return outputs

@torch.no_grad()
def do_prefix_forward(model, problem, image, processor):
    # PREFIX_PROMPT_TEMPLATE = "Question: {} Answer: {}"
    device = model.device
    PREFIX_PROMPT_TEMPLATE = problem.get('format')
    scores = []

    qs = problem["question"]

    for option in [problem["option_A"], problem["option_B"]]:
        prompt = PREFIX_PROMPT_TEMPLATE.format(qs, option, return_tensors="pt")
        inputs = processor.process(images=image, text=prompt)
        inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}
        answer_tokens = processor.tokenizer.encode(' ' + option, add_special_tokens=False)
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
            out = model(**inputs
                )
            # shift by 1 compared to input
            logits = out.logits[0, answer_start_from_back-1:answer_start_from_back-1+num_answer_tokens]
            probs = torch.nn.functional.softmax(logits, dim=-1)

            # Pick the probabilities corresponding to each of the answer tokens
            probs = torch.gather(probs, 1, torch.tensor(answer_tokens).to(device=device).unsqueeze(0))
            prefix_score = torch.prod(probs.pow(1/num_answer_tokens))
            scores.append(prefix_score.item())

    outputs = "A" if scores[0] > scores[1] else "B"
    return outputs