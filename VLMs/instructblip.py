import torch
from PIL import Image
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration

def load_model():
    processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")
    model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-7b", load_in_4bit=True, torch_dtype=torch.float16)
    return model, processor


def ask_question(model, question, image_path, processor, num_beams, max_length, top_p, temperature, use_forward):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, text=question, return_tensors="pt").to(device="cuda", dtype=torch.float16)
    if use_forward:
        return do_forward(model, processor, inputs)
    return do_generation(model, 
                         processor, 
                         inputs,
                         num_beams=num_beams,
                         top_p=top_p,
                         repetition_penalty=1.5,
                         length_penalty=1,
                         temperature=temperature,
                         max_new_tokens=max_length)

@torch.no_grad()
def do_generation(model, 
                  processor, 
                  inputs, 
                  num_beams, 
                  top_p, 
                  repetition_penalty, 
                  length_penalty, 
                  temperature, 
                  max_new_tokens):

    outputs = model.generate(
            **inputs,
            num_beams=num_beams,
            do_sample=True,
            max_new_tokens=max_new_tokens,
            min_length=1,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=temperature,
    )
    generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
    return generated_text

@torch.no_grad()
def do_forward(model, processor, inputs):
    VALID_ANSWERS = ['A', 'B']
    TOKEN_ID_A = processor.tokenizer("A", return_tensors="pt", add_special_tokens=False).get('input_ids')
    TOKEN_ID_B = processor.tokenizer("B", return_tensors="pt", add_special_tokens=False).get('input_ids')
    logits = model.forward(**inputs).logits[0, -1, :]
    logits = logits.reshape(-1, 1)
    soft_max = torch.nn.Softmax(dim=0)
    probs = soft_max(torch.cat([logits[TOKEN_ID_A], logits[TOKEN_ID_B]][:len(VALID_ANSWERS)]))
    outputs = VALID_ANSWERS[probs.argmax().item()]
    return outputs
