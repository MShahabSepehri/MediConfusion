import os
import time
import pandas as pd
from tqdm import tqdm
from PIL import Image
from utils import io_tools
from transformers import set_seed, logging


os.environ["TOKENIZERS_PARALLELISM"] = "true"
logging.set_verbosity_error()

ROOT = io_tools.get_root(__file__, 2)
PROMPTS_LOC = f'{ROOT}/configs/prompts/answering.json'
DATA_PATH = f'{ROOT}/data/test_dataset.json'
DATA = io_tools.load_json(DATA_PATH)
PROMPTS = io_tools.load_json(PROMPTS_LOC)


class BaseAnsweringModel():
    def __init__(self, model_args_path, mode, data_path, tr=3, max_samples=-1):
        self.key = None
        self.model_args_path = model_args_path
        self.conversion = io_tools.load_json(PROMPTS_LOC).get('conversion')
        self.mode = mode
        self.tr = tr
        self.max_samples = max_samples
        self.data_path = data_path
        self.prompt_key = 'prompt'
        self.set_model_params()

    def set_model_params(self):
        args = io_tools.load_json(self.model_args_path)
        self.set_init_prompt(args.get('init_prompt_id'))
        self.temperature = args.get("temperature")
        self.num_beams = args.get('num_beams')
        self.max_new_tokens = args.get('max_new_tokens')
        self.top_p = args.get('top_p')
        if self.mode == 'mc':
            self.temperature = 0
            self.top_p = None
            self.max_new_tokens = 32
        if self.mode == 'gpt4':
            self.clean_up = self.clean_up_no_option
            global gpt
            from VLMs import gpt
        else:
            self.clean_up = self.clean_up_with_option
        return args

    def ask_question(self, question, options, image_list):
        return self.convert_question(question, options)

    def set_init_prompt(self, init_prompt_id):
        self.init_prompt = None
        if init_prompt_id is None:
            return
        tmp = PROMPTS.get('init_prompts').get(self.key)
        if tmp is not None:
            self.init_prompt = tmp.get(init_prompt_id)
        else:
            self.init_prompt = PROMPTS.get('init_prompts').get('default')


    def evaluate(self, resume_path, save_dir):
        results = io_tools.load_resume_dict(resume_path)
        score = self.create_score_table(0, 0)
        save_path = self.check_folder(save_dir)
        key_list = list(DATA.keys())
        if self.max_samples != -1:
            key_list = key_list[: self.max_samples]
        for id in tqdm(key_list):
            if id in results.keys():
                sample_score = results.get(id).get('score')
            else:
                sample = DATA.get(id)
                ans_dict, sample_score = self.sample_eval(sample)
                results[id] = {'answer': ans_dict, 'score': sample_score}
            
            self.update_score_table(score, sample_score)
            if save_path is not None:
                io_tools.save_json(results, f'{save_path}/{self.key}_{self.mode}_test.json')
        self.print_score(score, num_samples=self.max_samples)
        if save_path is not None:
            io_tools.save_json(score, f'{save_path}/{self.key}_{self.mode}_test_score.json')
        return results, score
    
    def sample_eval(self, sample):
        image_list = [f"{self.data_path}/{sample.get('im')}"]
        question = sample.get('question')
        # options = [sample.get('option_A'), sample.get('option_B')]
        options = [sample.get('option_A'), sample.get('option_B'), sample.get('option_C'), sample.get('option_D')]
        ans = sample.get('correct')
        responses = self.ask_question(question, options, image_list)
        ans_dict = self.clean_up(question, options, responses[0])
        correct, invalid = self.get_score(ans_dict, ans)
        scores = self.create_score_table(correct, invalid)
        
        return ans_dict, scores
        
    def get_clean_up_prompt(self, question, options, response):
        role = self.conversion.get('role')
        return (f'[Question]\n{question}\n\n'
                f'[Answer A]\n{options[0]}\n\n'
                f'[Answer B]\n{options[1]}\n\n'
                f'[Answer C]\n{options[2]}\n\n'
                f'[Answer D]\n{options[3]}\n\n'
                f'[{role}]\n{response}\n\n[End of {role}]\n\n'
                f'[System]\n{self.conversion.get("instruct_prompt")}\n\n')

    def get_score(self, ans_dict, ans):
        correct, c = self.check_answer(ans, 
                                        ans_dict.get('A'), 
                                        ans_dict.get('B'), 
                                        ans_dict.get('C'), 
                                        ans_dict.get('D'), 
                                        self.tr)
        invalid = 1 * (c == '-')
        return correct, invalid
    
    def clean_up_no_option(self, question, options, answer):
        client = gpt.get_client()
        prompt = self.get_clean_up_prompt(question, options, answer)
        response = gpt.get_response(client=client,
                                    deployment_name=self.conversion.get('gpt_deployment_name'),
                                    init_prompt=self.conversion.get('init_prompt'),
                                    prompt=prompt,
                                    temperature=float(self.conversion.get('temperature')),
                                    )
        ans = self.process_gpt_response(response)
        ans['full_answer'] = answer
        return ans
    
    def clean_up_with_option(self, question, options, answer):
        labels = ['A', 'B', 'C', 'D']
        scores = {'full_answer': answer}
        for key in labels:
            scores[key] = 0
        if answer is not None:
            tmp = answer.split(' ')
            for la in labels:
                valid_list = [f'{la}', f'{la}:', f'.{la}', f'.{la}:', f'{la}.', f'{la}\")', f'{la}\n', f'\n{la}']
                correct = any([x in tmp for x in valid_list])
                if correct:
                    scores[la] = 10
        tmp = [1 for x in scores.values() if x==10]
        if sum(tmp) > 1:
            for key in labels:
                scores[key] = 0
        return scores
    
    def convert_question(self, question, options):
        prompt_dict = PROMPTS.get(self.prompt_key).get(self.mode)
        if self.key in prompt_dict.keys():
            key = self.key
        else:
            key = 'default'
        
        tmp = prompt_dict.get(key)
        if self.mode == 'gpt4':
            output = tmp.format(question)
        elif self.mode == 'greedy':
            output = tmp.format(question, options[0], options[1], options[2], options[3])
        elif self.mode == 'mc':
            output = tmp.format(question, options[0], options[1], options[2], options[3])
        elif self.mode == 'prefix':
            output = {"question": question, 
                      "option_A": options[0],
                      "option_B": options[1], 
                      "option_C": options[2], 
                      "option_D": options[3],
                      "format": tmp}
        return output
    
    def check_folder(self, save_dir):
        if save_dir is None:
            return None
        save_path = f'{save_dir}/{self.key}'
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        return save_path
    
    @staticmethod
    def update_score_table(score, sample_score):
        for key in score:
            score[key] += sample_score.get(key)

    @staticmethod
    def create_score_table(correct, invalid):
        score_table = {
            'individual_score': correct,
            'invalid': invalid,
        }
        return score_table

    @staticmethod
    def print_score(score, num_samples=None, precision=2):
        print('\n')
        print_format = "{:<10} {:<17} {:<15}"
        print(print_format.format('Total', 
                                  'Individual acc.', 
                                  'Invalid acc.',
                                  ))

        if (num_samples is None) or (num_samples == -1):
            total = len(DATA)
        else:
            total = num_samples
        num = total / 100
        individual_acc = round(score.get('individual_score') / num, precision)
        invalid = round(score.get('invalid') / num, precision)
        print(print_format.format(total, individual_acc, invalid))
            
    
    @staticmethod
    def process_gpt_response(response):
        if response is None:
            return {
            'A': 0,
            'B': 0,
            'C': 0,
            'D': 0,
            'gpt_reason': '',
        }
        tmp = response.replace('\n\n', '\n').split('\n')

        ans = {
            'A': int(tmp[0].replace('A: ', '')),
            'B': int(tmp[1].replace('B: ', '')),
            'C': int(tmp[2].replace('C: ', '')),
            'D': int(tmp[3].replace('D: ', '')),
            'gpt_reason': tmp[4].replace('Your explanation: ', ''),
        }
        return ans
    
    @staticmethod
    def check_answer(answer, a_score, b_score, c_score, d_score, tr):
        chosen = '-'
        labels = ['A', 'B', 'C', 'D']
        score_list = [a_score, b_score, c_score, d_score]
        max_score = max(score_list)
        tmp = min([max_score - score for score in score_list])
        if tmp >= tr:
            chosen = labels[score_list.index(max_score)]
        if chosen == answer:
            return 1, chosen
        return 0, chosen

    
class GPTAnswering(BaseAnsweringModel):

    def set_model_params(self):
        global gpt
        from VLMs import gpt
        self.key = 'gpt'
        args = super().set_model_params()
        self.deployment_name = args.get("deployment_name")
        self.client = gpt.get_client()
        if self.mode in ['greedy', 'prefix']:
            raise ValueError(f'Cannot use forward for GPT!')

    def ask_question(self, question, options, image_list):
        qs = super().ask_question(question, options, image_list)
        response_list = []
        for image in image_list:
            response = gpt.ask_question(self.client, image, qs, self.init_prompt, self.deployment_name, self.temperature)
            response_list.append(response)
        return response_list
    
class ClaudeAnswering(BaseAnsweringModel):

    def set_model_params(self):
        global claude
        from VLMs import claude
        self.key = 'claude'
        args = super().set_model_params()
        # self.deployment_name = args.get("deployment_name")
        self.client = claude.get_client()
        if self.mode in ['greedy', 'prefix']:
            raise ValueError(f'Cannot use forward for Claude!')

    def ask_question(self, question, options, image_list, max_retry=3):
        qs = super().ask_question(question, options, image_list)
        response_list = []
        for image in image_list:
            counter = 0
            response = None
            while counter < max_retry:
                try:
                    response = claude.ask_question(self.client, image, qs, self.init_prompt, self.temperature)
                    break
                except Exception as e:
                    counter += 1
                    print(counter, e)
            response_list.append(response)
        return response_list
    

class GeminiAnswering(BaseAnsweringModel):
    def set_model_params(self):
        global gemini
        from VLMs import gemini
        self.key = 'gemini'
        args = super().set_model_params()
        # self.deployment_name = args.get("deployment_name")
        self.model = gemini.load_model(self.init_prompt, self.temperature)
        if self.mode in ['greedy', 'prefix']:
            raise ValueError(f'Cannot use forward for Claude!')

    def ask_question(self, question, options, image_list, max_retry=3):
        qs = super().ask_question(question, options, image_list)
        response_list = []
        for image in image_list:
            response = None
            counter = 0
            while counter < max_retry:
                try:
                    response = gemini.ask_question(self.model, image, qs)
                    break
                except Exception as e:
                    counter += 1
                    print(counter, e)
                # time.sleep(10)
            response_list.append(response)
        return response_list


class LLAVAMedAnswering(BaseAnsweringModel):

    def set_model_params(self):
        global llava_med
        from VLMs import llava_med
        self.key = 'llava_med'
        args = super().set_model_params()
        
        set_seed(0)
        tokenizer, model, image_processor, context_len = \
            llava_med.load_model(args.get("model_path"), args.get("model_base"))

        self.model = model
        self.tokenizer = tokenizer
        self.image_processor = image_processor

        self.conv_mode = args.get("conv_mode")
        self.use_im_start_end = args.get('use_im_start_end')

    def convert_question(self, question, options):
        tmp = super().convert_question(question, options)
        if self.mode == 'prefix':
            to_process = tmp["question"]
        else:
            to_process = tmp

        if self.prompt_key == 'with_image':
            to_process = '<image>\n' + to_process
            qs = to_process.replace(llava_med.DEFAULT_IMAGE_TOKEN, '').strip()
            if self.use_im_start_end:
                qs = llava_med.DEFAULT_IM_START_TOKEN + llava_med.DEFAULT_IMAGE_TOKEN + llava_med.DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = llava_med.DEFAULT_IMAGE_TOKEN + '\n' + qs
        
        if self.mode == 'prefix':
            tmp["question"] = qs
            return tmp
        else:
            return qs

    def ask_question(self, question, options, image_list):
        question = super().ask_question(question, options, image_list)
        response_list = []
        image_list = [Image.open(x) for x in image_list]
        for image in image_list:
            outputs = llava_med.ask_question(self.model, 
                                             question, 
                                             image, 
                                             self.image_processor, 
                                             self.tokenizer, 
                                             self.mode,
                                             conv_mode=self.conv_mode,
                                             temperature=self.temperature,
                                             top_p=self.top_p, 
                                             num_beams=self.num_beams,
                                             max_new_tokens=self.max_new_tokens)
            response_list.append(outputs)

        return response_list


class LLAVAAnswering(BaseAnsweringModel):

    def set_model_params(self):
        global llava
        from VLMs import llava
        self.key = 'llava'
        args = super().set_model_params()
        
        set_seed(0)
        model, processor = llava.load_model()

        self.model = model
        self.processor = processor
        self.conv_mode = args.get("conv_mode")

    def ask_question(self, question, options, image_list):
        question = super().ask_question(question, options, image_list)
        response_list = []
        image_list = [Image.open(x) for x in image_list]
        for image in image_list:
            outputs = llava.ask_question(self.model, 
                                         self.processor, 
                                         question, 
                                         image, 
                                         self.mode,
                                         temperature=self.temperature,
                                         top_p=self.top_p, 
                                         num_beams=self.num_beams)
            response_list.append(outputs)

        return response_list



class RadFMAnswering(BaseAnsweringModel):

    def set_model_params(self):
        global radfm
        from VLMs import radfm
        self.key = 'radfm'
        args = super().set_model_params()
        self.language_files_path = args.get("language_files_path")
        self.model_path = args.get("model_path")
        model, text_tokenizer, image_padding_tokens = radfm.load_model(self.language_files_path, self.model_path)
        self.model = model
        self.text_tokenizer = text_tokenizer
        self.image_padding_tokens = image_padding_tokens

    def ask_question(self, question, options, image_list):
        question = super().ask_question(question, options, image_list)
        response_list = []
        for image_path in image_list:
            outputs = radfm.ask_question(self.model, 
                                         question, 
                                         image_path, 
                                         self.text_tokenizer, 
                                         self.image_padding_tokens,
                                         self.mode)
            response_list.append(outputs)
        return response_list

class BLIP2Answering(BaseAnsweringModel):

    def set_model_params(self):
        global blip2
        from VLMs import blip2
        self.key = 'blip2'
        args = super().set_model_params()
        model, processor = blip2.load_model()
        self.model = model
        self.processor = processor

    def ask_question(self, question, options, image_list):
        question = super().ask_question(question, options, image_list)
        response_list = []
        for image_path in image_list:
            outputs = blip2.ask_question(self.model, 
                                         question, 
                                         image_path, 
                                         self.processor,
                                         self.num_beams,
                                         self.max_new_tokens,
                                         self.top_p,
                                         self.temperature,
                                         self.mode)
            response_list.append(outputs)
        return response_list
    
class InstructBLIPAnswering(BaseAnsweringModel):

    def set_model_params(self):
        global instructblip
        from VLMs import instructblip
        self.key = 'instructblip'
        args = super().set_model_params()
        model, processor = instructblip.load_model()
        self.model = model
        self.processor = processor
        if self.temperature == 0:
            self.temperature = 0

    def ask_question(self, question, options, image_list):
        question = super().ask_question(question, options, image_list)
        response_list = []
        for image_path in image_list:
            outputs = instructblip.ask_question(self.model, 
                                                question, 
                                                image_path, 
                                                self.processor,
                                                self.num_beams,
                                                self.max_new_tokens,
                                                self.top_p,
                                                self.temperature,
                                                self.mode)
            response_list.append(outputs)
        return response_list
    

class MedFlamingoAnswering(BaseAnsweringModel):

    def set_model_params(self):
        global med_flamingo
        from VLMs import med_flamingo
        self.key = 'med_flamingo'
        args = super().set_model_params()
        self.LLaMa_PATH = args.get('LLaMa_PATH')
        self.CHECKPOINT_PATH = args.get('CHECKPOINT_PATH')
        self.IMAGE_PATH = args.get('IMAGE_PATH')
        model, processor = med_flamingo.load_model(self.LLaMa_PATH, self.CHECKPOINT_PATH)
        self.model = model
        self.processor = processor

    def ask_question(self, question, options, image_list):
        question = super().ask_question(question, options, image_list)
        response_list = []
        for image_path in image_list:
            outputs = med_flamingo.ask_question(self.model, 
                                                self.processor,
                                                image_path, 
                                                question, 
                                                self.max_new_tokens,
                                                self.mode,
                                                self.IMAGE_PATH,
                                                )
            response_list.append(outputs)
        return response_list


class MedVInTAnswering(BaseAnsweringModel):

    def set_model_params(self):
        global med_flamingo
        from VLMs import medvint
        self.key = 'medvint'
        args = super().set_model_params()
        self.model_args = medvint.ModelArguments()
        self.model_args.embed_dim = args.get("EMBED_DIM")
        self.model_args.pretrained_tokenizer = args.get("PRETRAINED_TOKENIZER")
        self.model_args.pretrained_model = args.get("PRETRAINED_MODEL")
        self.model_args.image_encoder = args.get("IMAGE_ENCODER")
        self.model_args.pmcclip_pretrained = args.get("PMCCLIP_PRETRAINED")
        self.model_args.clip_pretrained = args.get("CLIP_PRETRAINED")
        self.model_args.ckp = args.get("CKP")
        model, image_transform, tokenizer = medvint.load_model(self.model_args)
        self.model = model
        self.image_transform = image_transform
        self.tokenizer = tokenizer

    def ask_question(self, question, options, image_list):
        question = super().ask_question(question, options, image_list)
        image_list = [Image.open(x).convert('RGB') for x in image_list]
        response_list = []
        for image in image_list:
            image = self.image_transform(image)
            outputs = med_flamingo.ask_question(self.model, 
                                                self.tokenizer, 
                                                question, 
                                                image,
                                                )
            response_list.append(outputs)
        return response_list
ANSWERING_CLASS_DICT = {
    'gpt': GPTAnswering,
    'claude': ClaudeAnswering,
    'gemini': GeminiAnswering,
    'llava_med': LLAVAMedAnswering,
    'llava': LLAVAAnswering,
    'radfm': RadFMAnswering,
    'blip2': BLIP2Answering,
    'instructblip': InstructBLIPAnswering,
    'med_flamingo': MedFlamingoAnswering,
    'medvint': MedVInTAnswering,
}

DEFAULT_MODEL_CONFIGS = {
    'gpt': f'{ROOT}/configs/VLM/gpt/vanilla.json',
    'claude': f'{ROOT}/configs/VLM/claude/vanilla.json',
    'gemini': f'{ROOT}/configs/VLM/gemini/vanilla.json',
    'llava_med': f'{ROOT}/configs/VLM/llava_med/vanilla.json',
    'llava': f'{ROOT}/configs/VLM/llava/vanilla.json',
    'radfm': f'{ROOT}/configs/VLM/radfm/vanilla.json',
    'blip2': f'{ROOT}/configs/VLM/blip2/vanilla.json',
    'instructblip': f'{ROOT}/configs/VLM/instructblip/vanilla.json',
    'med_flamingo': f'{ROOT}/configs/VLM/med_flamingo/vanilla.json',
    'medvint': f'{ROOT}/configs/VLM/medvint/vanilla.json',
}
