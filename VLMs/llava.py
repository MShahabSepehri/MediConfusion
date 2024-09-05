import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration


def load_model():
    processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
    model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True) 
    model.to("cuda:0")
    return model, processor

def ask_question(model, processor, question, image, mode, temperature=0.2, top_p=None, num_beams=1, max_new_tokens=100):
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
    inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda:0")

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
    raise ValueError (logits)
    soft_max = torch.nn.Softmax(dim=0)
    probs = soft_max(torch.cat([logits[TOKEN_ID_A], logits[TOKEN_ID_B]][:len(VALID_ANSWERS)]))
    outputs = VALID_ANSWERS[probs.argmax().item()]
    return outputs

@torch.no_grad()
def do_generation(model, inputs, processor, max_new_tokens):
    output = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return processor.decode(output[0], skip_special_tokens=True)