import torch
from PIL import Image
from transformers import AutoProcessor, Blip2ForConditionalGeneration


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def load_model():
    processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
    model.to(get_device())
    return model, processor

def ask_question(model, question, image_path, processor, num_beams, max_length, top_p, temperature, use_forward):
    if use_forward:
        return do_forward(model, processor, image_path, question)
    return do_generation(model, 
                         processor, 
                         image_path, 
                         question,
                         num_beams=num_beams,
                         top_p=top_p,
                         temperature=temperature,
                         max_new_tokens=max_length)

@torch.no_grad()
def do_generation(model, 
                  processor, 
                  image_path, 
                  question, 
                  num_beams, 
                  top_p,
                  temperature, 
                  max_new_tokens):
    image = Image.open(image_path).convert("RGB")
    # question = "this is a picture of"
    inputs = processor(image, text=question, return_tensors="pt").to(device="cuda", dtype=torch.float16)
    outputs = model.generate(
            **inputs,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            min_length=1,
            top_p=top_p,
            temperature=temperature,
    )
    generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
    raise ValueError(generated_text, temperature, top_p, question, inputs.get('input_ids').shape)
    return generated_text

@torch.no_grad()
def do_forward(model, processor, image_path, question):
    VALID_ANSWERS = ['A', 'B']
    question = 'The first letter of alphabet: '
    TOKEN_ID_A = processor.tokenizer("A", return_tensors="pt", add_special_tokens=False).get('input_ids')
    TOKEN_ID_B = processor.tokenizer("B", return_tensors="pt", add_special_tokens=False).get('input_ids')
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, text=question, return_tensors="pt").to(device=get_device(), dtype=torch.float16)

    ss = model.forward(**inputs).logits
    raise ValueError(ss.shape)
    ss = ss[0, -1, :]
    logits = logits.reshape(-1, 1)
    soft_max = torch.nn.Softmax(dim=0)
    probs = soft_max(torch.cat([logits[TOKEN_ID_A], logits[TOKEN_ID_B]][:len(VALID_ANSWERS)]))
    outputs = VALID_ANSWERS[probs.argmax().item()]
    raise ValueError(probs)
    return outputs
