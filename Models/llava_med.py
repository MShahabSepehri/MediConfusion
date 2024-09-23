import os
import torch
from utils import io_tools
from llava.utils import disable_torch_init
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, tokenizer_image_token, process_images
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from torch.nn import CrossEntropyLoss
from torch.nn.functional import softmax

def load_model(model_path, model_base):
    disable_torch_init()
    model_path = os.path.expanduser(model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base, model_name)
    return tokenizer, model, image_processor, context_len

def get_input_id(tokenizer, question, conv_mode):
    # qs = convert_question(question, mm_use_im_start_end, use_options)
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    return input_ids

@torch.no_grad()
def do_forward(model, input_ids, image_tensor, image_size, tokenizer):
    VALID_ANSWERS = ['A', 'B']
    TOKEN_IDs = [tokenizer.encode(x, return_tensors="pt", add_special_tokens=False) for x in VALID_ANSWERS]

    with torch.inference_mode():
        out = model(input_ids,
                    images=image_tensor.unsqueeze(0).half().cuda(),
                    image_sizes=[image_size],
                    )
        
        logits = out.logits[0, -1, :]
        soft_max = torch.nn.Softmax(dim=0)
        probs = soft_max(torch.cat([logits[x] for x in TOKEN_IDs]))
        outputs = VALID_ANSWERS[probs.argmax().item()]
    return outputs

@torch.no_grad()
def do_generation(model, input_ids, image_tensor, tokenizer, temperature, top_p, num_beams, max_new_tokens):
    with torch.inference_mode():
        output_ids = model.generate(input_ids,
                                    images=image_tensor.unsqueeze(0).half().cuda(),
                                    do_sample=True if temperature > 0 else False,
                                    temperature=temperature,
                                    top_p=top_p,
                                    num_beams=num_beams,
                                    max_new_tokens=max_new_tokens,
                                    use_cache=True)

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return outputs

def ask_question(model, question, image, image_processor, tokenizer, mode, conv_mode, temperature=0.2, top_p=None, num_beams=1, max_new_tokens=100):
    image_tensor = process_images([image], image_processor, model.config)[0]
    
    if mode == 'greedy':
        input_ids = get_input_id(tokenizer, question, conv_mode)
        outputs = do_forward(model, input_ids, image_tensor, image.size, tokenizer)
    elif mode in ['mc', 'gpt4']:
        input_ids = get_input_id(tokenizer, question, conv_mode)
        outputs = do_generation(model, input_ids, image_tensor, tokenizer, temperature, top_p, num_beams, max_new_tokens)
    elif mode == 'prefix':
        outputs = do_prefix_forward(model, question, image_tensor, image.size, tokenizer, conv_mode)
    return outputs

@torch.no_grad()
def do_prefix_forward(model, problem, image_tensor, image_size, tokenizer, conv_mode):
    scores = []
    questions = []
    qs = problem["question"]
    for option in [problem["option_A"], problem["option_B"]]:
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], option)
        prompt = conv.get_prompt()
        questions.append(prompt)
        answer_tokens = tokenizer.encode(" " + option, add_special_tokens=False)[1:]
        num_answer_tokens = len(answer_tokens)
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
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
            out = model(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                use_cache=True
                )
            logits = out.logits[0, answer_start_from_back-1:answer_start_from_back-1+num_answer_tokens]
            probs = torch.nn.functional.softmax(logits, dim=-1)

            # Pick the probabilities corresponding to each of the answer tokens
            probs = torch.gather(probs, 1, torch.tensor(answer_tokens).cuda().unsqueeze(0))
            prefix_score = torch.prod(probs.pow(1/num_answer_tokens))
            scores.append(prefix_score.item())
    outputs = "A" if scores[0] > scores[1] else "B"
    return outputs