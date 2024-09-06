import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX


def load_model():
    processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
    model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True) 
    model.to("cuda:0")
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
        outputs = do_generation(model, inputs, temperature, top_p, num_beams, max_new_tokens)
    return outputs


@torch.no_grad()
def do_forward(model, inputs, processor):
    VALID_ANSWERS = ['A', 'B']
    TOKEN_ID_A = processor.tokenizer.encode("A", add_special_tokens=False)
    TOKEN_ID_B = processor.tokenizer.encode("B", add_special_tokens=False)

    with torch.inference_mode():
        out = model.forward(**inputs)
        
    logits = out.logits[0, -1, :]
    soft_max = torch.nn.Softmax(dim=0)
    probs = soft_max(torch.cat([logits[TOKEN_ID_A], logits[TOKEN_ID_B]][:len(VALID_ANSWERS)]))
    outputs = VALID_ANSWERS[probs.argmax().item()]
    return outputs

@torch.no_grad()
def do_generation(model, inputs, processor, max_new_tokens):
    output = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return processor.decode(output[0], skip_special_tokens=True)

@torch.no_grad()
def do_prefix_forward(model, problem, image, processor):
    # python scripts/answering.py --vlm_name llava --mode prefix
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
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
        answer_tokens = processor.tokenizer.encode(" " + option, add_special_tokens=False)[1:]
        num_answer_tokens = len(answer_tokens)
        input_ids = inputs["input_ids"]
        # try to find the answer tokens in input ids
        start_indices = []
        for i in range(input_ids.size(1) - num_answer_tokens + 1):
            if torch.equal(input_ids[0, i:i+num_answer_tokens], torch.tensor(answer_tokens).cuda()):
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
            probs = torch.gather(probs, 1, torch.tensor(answer_tokens).cuda().unsqueeze(0))
            prefix_score = torch.prod(probs.pow(1/num_answer_tokens))
            scores.append(prefix_score.item())
    outputs = "A" if scores[0] > scores[1] else "B"
    return outputs