import os
import torch
from utils import io_tools
from llava.utils import disable_torch_init
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, tokenizer_image_token, process_images
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

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
    TOKEN_ID_A = tokenizer.encode("A", add_special_tokens=False)
    TOKEN_ID_B = tokenizer.encode("B", add_special_tokens=False)

    with torch.inference_mode():
        out = model(input_ids,
                    images=image_tensor.unsqueeze(0).half().cuda(),
                    image_sizes=[image_size],
                    )
        
        logits = out.logits[0, -1, :]
        soft_max = torch.nn.Softmax(dim=0)
        probs = soft_max(torch.cat([logits[TOKEN_ID_A], logits[TOKEN_ID_B]][:len(VALID_ANSWERS)]))
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

def ask_question(model, input_ids, image, image_processor, tokenizer, mode, temperature=0.2, top_p=None, num_beams=1, max_new_tokens=100):
    image_tensor = process_images([image], image_processor, model.config)[0]

    if mode == 'greedy':
        outputs = do_forward(model, input_ids, image_tensor, image.size, tokenizer)
    elif mode in ['mc', 'gpt4']:
        outputs = do_generation(model, input_ids, image_tensor, tokenizer, temperature, top_p, num_beams, max_new_tokens)
    return outputs