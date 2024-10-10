import os
import time
import warnings
from tqdm import tqdm
from PIL import Image
from utils import io_tools
from transformers import set_seed, logging


os.environ["TOKENIZERS_PARALLELISM"] = "true"
logging.set_verbosity_error()

ROOT = io_tools.get_root(__file__, 2)
PROMPTS_LOC = f'{ROOT}/configs/prompts/answering.json'
DATA_PATH = f'{ROOT}/data/dataset.json'
STATS_PATH = f'{ROOT}/data/stats/statistics.json'
DATA = io_tools.load_json(DATA_PATH)
STATS = io_tools.load_json(STATS_PATH)
PROMPTS = io_tools.load_json(PROMPTS_LOC)


class BaseAnsweringModel():
    def __init__(self, model_args_path, mode, data_path, local_image_address=True, tr=3, device='cuda'):
        self.key = None
        self.model_args_path = model_args_path
        self.conversion = io_tools.load_json(PROMPTS_LOC).get('conversion')
        self.mode = mode
        self.tr = tr
        self.data_path = data_path
        self.prompt_key = 'prompts'
        self.local_image_address = local_image_address
        self.device = device
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
            self.num_beams = 1
            self.top_p = None
            self.max_new_tokens = 32
        if self.mode == 'gpt4':
            self.clean_up = self.clean_up_gpt
            global gpt
            from Models import gpt
        else:
            self.clean_up = self.clean_up_manual
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
        score = self.create_score_table([], [], -1, -1, -1, -1, -1)
        save_path = self.check_folder(save_dir)
        for id in tqdm(DATA.keys()):
            if id in results.keys():
                sample_score = results.get(id).get('score')
            else:
                sample = DATA.get(id)
                ans_dict, sample_score = self.sample_eval(sample)
                results[id] = {'answer': ans_dict, 'score': sample_score}
            
            self.update_score_table(score, sample_score)
            if save_path is not None:
                io_tools.save_json(results, f'{save_path}/{self.key}_{self.mode}.json')
        self.print_score(score)
        if save_path is not None:
            io_tools.save_json(score, f'{save_path}/{self.key}_{self.mode}_score.json')
        return results, score
    
    def sample_eval(self, sample):
        if self.local_image_address:
            image_list = [f"{self.data_path}/{sample.get(x)}.jpg" for x in ['im_1_local', 'im_2_local']]
            image_list = [f"{self.data_path}/{sample.get('im_1_local')}.jpg",  f"{self.data_path}/{sample.get('im_2_local')}.jpg"]
        else:
            image_list = [f"{self.data_path}/roco-dataset/data/{sample.get(x)}" for x in ['im_1', 'im_2']]
        question = sample.get('question')
        options = [sample.get('option_A'), sample.get('option_B')]
        im1_ans = sample.get('im_1_correct')
        im2_ans = sample.get('im_2_correct')
        responses = self.ask_question(question, options, image_list)
        ans_dict = {'im1': self.clean_up(question, options, responses[0]), 
                    'im2': self.clean_up(question, options, responses[1])}
        im1_correct, im1_invalid, im2_correct, im2_invalid, confused = self.get_score(ans_dict, im1_ans, im2_ans)
        scores = self.create_score_table(sample.get('category_1'), 
                                         sample.get('category_2'), 
                                         im1_correct, 
                                         im2_correct, 
                                         im1_invalid,
                                         im2_invalid,
                                         confused,
                                         )
        

        return ans_dict, scores
        
    def get_clean_up_prompt(self, question, options, response):
        role = self.conversion.get('role')
        return (f'[Question]\n{question}\n\n'
                f'[Answer A]\n{options[0]}\n\n'
                f'[Answer B]\n{options[1]}\n\n'
                f'[{role}]\n{response}\n\n[End of {role}]\n\n'
                f'[System]\n{self.conversion.get("instruct_prompt")}\n\n')

    def get_score(self, ans_dict, im1_ans, im2_ans):
        im1_correct, c1 = self.check_answer(im1_ans, 
                                            ans_dict.get('im1').get('A'), 
                                            ans_dict.get('im1').get('B'), 
                                            self.tr)
        invalid1 = (c1 == '-')
        im2_correct, c2 = self.check_answer(im2_ans, 
                                            ans_dict.get('im2').get('A'), 
                                            ans_dict.get('im2').get('B'), 
                                            self.tr)
        invalid2 = (c2 == '-')
        confused = 1 * ((c1 == c2) and (c1 != '-'))
        ans_dict
        return im1_correct, invalid1, im2_correct, invalid2, confused
    
    def clean_up_gpt(self, question, options, answer):
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
    
    def clean_up_manual(self, question, options, answer):
        labels = ['A', 'B']
        scores = {'full_answer': answer}
        for key in labels:
            scores[key] = 0
        if answer is not None:
            answer = answer.replace('\n', ' ')
            tmp = answer.split(' ')
            for la in labels:
                valid_list = [f'{la}', f'{la}:', f'.{la}', f'.{la}:', f'{la}.', 
                              f'{la}\")', f'{la}\n', f'\n{la}', f'(\"{la}\":', 
                              f'(\"{la}\")', f'(\"{la}\").']
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
            output = tmp.format(question, options[0], options[1])
        elif self.mode == 'mc':
            output = tmp.format(question, options[0], options[1])
        elif self.mode == 'prefix':
            output = {
                "question": question, 
                "option_A": options[0], 
                "option_B": options[1],
                "format": tmp
                }
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
            tmp = score.get(key)
            for cat in tmp.keys():
                tmp[cat] += sample_score.get(key).get(cat)

    @staticmethod
    def create_score_table(cat_1, cat_2, im1_correct, im2_correct, im1_invalid, im2_invalid, confused):
        scores = {'set_score': {}, 'individual_score': {}, 'confused': {}, 'invalid': {}, 'valids': {}, 'valid_pairs': {}}
        for v in scores.values():
            for key in STATS.keys():
                v[key] = 0
            v['total'] = 0
        if confused == 1:
            scores.get('confused')['total'] += 1
            for c in (cat_1 + cat_2):
                scores.get('confused')[c] += 1

        if im1_correct == 1:
            scores.get('individual_score')['total'] += 1
            for c in cat_1:
                scores.get('individual_score')[c] += 1
        if im2_correct == 1:
            scores.get('individual_score')['total'] += 1
            for c in cat_2:
                scores.get('individual_score')[c] += 1

        if im1_invalid == 1:
            scores.get('invalid')['total'] += 1
            for c in cat_1:
                scores.get('invalid')[c] += 1
        if im2_invalid == 1:
            scores.get('invalid')['total'] += 1
            for c in cat_2:
                scores.get('invalid')[c] += 1

        if (im1_invalid == 0) and (im2_invalid == 0):
            scores.get('valids')['total'] += 1
            for c in (cat_1 + cat_2):
                scores.get('valids')[c] += 1
            if confused == 0:
                scores.get('valid_pairs')['total'] += 1
                for c in (cat_1 + cat_2):
                    scores.get('valid_pairs')[c] += 1

        if (im1_correct == 1) and (im2_correct == 1):
            scores.get('set_score')['total'] += 1
            for c in (cat_1 + cat_2):
                scores.get('set_score')[c] += 1
        return scores

    @staticmethod
    def print_score(score, precision=2):
        print('\n')
        # print_format = "{:<17} {:<10} {:<10} {:<10} {:<12} {:<17} {:<10}"
        print_format = "{:<17} {:<10} {:<10} {:<17} {:<15} {:<15} {:<15} {:<15} {:<15}"
        print(print_format.format('Category', 
                                  'Total', 
                                  'Set acc.', 
                                  'Individual acc.', 
                                  'Confused acc.',
                                  'Valid pairs',
                                  'Invalid acc.',
                                  'Precision',
                                  'Precision total',
                                  ))
        key_list = list(STATS.keys()) + ['total']
        for cat in key_list:
            if cat == 'total':
                total = len(DATA)
                num = total / 100
                individual_acc = round(score.get('individual_score').get(cat) / num / 2, precision)
                invalid = round(score.get('invalid').get(cat) / num / 2, precision)
                txt = 'All'
            else:
                total = STATS.get(cat)
                num = total / 100
                individual_acc = round(score.get('individual_score').get(cat) / num, precision)
                invalid = round(score.get('invalid').get(cat) / num, precision)
                txt = cat

            set_acc = round(score.get('set_score').get(cat) / num, precision)
            valid_pairs = score.get('valids').get(cat)
            precision_total = score.get('valid_pairs').get(cat)

            confused = 0
            if score.get('valids').get(cat) > 0:
                confused = round(score.get('confused').get(cat) / score.get('valids').get(cat) * 100, precision)
            
            pr = 0
            if score.get('valid_pairs').get(cat) > 0:
                pr = round(score.get('set_score').get(cat) / score.get('valid_pairs').get(cat) * 100, precision)

            print(print_format.format(txt, total, set_acc, individual_acc, confused, valid_pairs, invalid, pr, precision_total))
            
    @staticmethod
    def process_gpt_response(response):
        if response is None:
            return {
            'A': 0,
            'B': 0,
            'gpt_reason': '',
        }
        tmp = response.replace('\n\n', '\n').split('\n')
        ans = {
            'A': int(tmp[0].replace('A: ', '')),
            'B': int(tmp[1].replace('B: ', '')),
            'gpt_reason': tmp[2].replace('Your explanation: ', ''),
        }
        return ans
    
    @staticmethod
    def check_answer(answer, a_score, b_score, tr):
        chosen = '-'
        if a_score >= b_score + tr:
            chosen = 'A'
        elif b_score >= a_score + tr:
            chosen = 'B'
        if chosen == answer:
            return 1, chosen
        return 0, chosen

    
class GPTAnswering(BaseAnsweringModel):

    def set_model_params(self):
        global gpt
        from Models import gpt
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
        from Models import claude
        self.key = 'claude'
        args = super().set_model_params()
        self.deployment_name = args.get("deployment_name")
        self.client = claude.get_client()
        if self.mode in ['greedy', 'prefix']:
            raise ValueError(f'Cannot use forward for Claude!')

    def ask_question(self, question, options, image_list):
        qs = super().ask_question(question, options, image_list)
        response_list = []
        for image in image_list:
            response = claude.ask_question(self.client, image, qs, self.init_prompt, self.temperature, self.deployment_name)
            response_list.append(response)
        return response_list
    

class GeminiAnswering(BaseAnsweringModel):
    def set_model_params(self):
        global gemini
        from Models import gemini
        self.key = 'gemini'
        args = super().set_model_params()
        self.deployment_name = args.get("deployment_name")
        self.model = gemini.load_model(self.init_prompt, self.temperature, self.deployment_name)
        if self.mode in ['greedy', 'prefix']:
            raise ValueError(f'Cannot use forward for Claude!')

    def ask_question(self, question, options, image_list):
        qs = super().ask_question(question, options, image_list)
        response_list = []
        for image in image_list:
            flag = True
            counter = 0
            while flag:
                try:
                    response = gemini.ask_question(self.model, image, qs)
                    flag = False
                except Exception as e:
                    counter += 1
                    print(counter, e)
                time.sleep(10)
            response_list.append(response)
        return response_list


class LLaVAMedAnswering(BaseAnsweringModel):

    def set_model_params(self):
        global llava_med
        from Models import llava_med
        self.key = 'llava_med'
        args = super().set_model_params()
        
        set_seed(0)
        tokenizer, model, image_processor, context_len = \
            llava_med.load_model(args.get("model_path"), args.get("model_base"))

        self.model = model
        self.tokenizer = tokenizer
        self.image_processor = image_processor

        if self.device == 'cpu':
            warnings.warn('LLaVA-Med implepenation does not support CPU! Switching to CUDA.')
            self.device = 'cuda'

        self.conv_mode = args.get("conv_mode")
        self.use_im_start_end = args.get('use_im_start_end')

    def convert_question(self, question, options):
        tmp = super().convert_question(question, options)
        if self.mode == 'prefix':
            to_process = tmp["question"]
        else:
            to_process = tmp

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


class LLaVAAnswering(BaseAnsweringModel):

    def set_model_params(self):
        global llava
        from Models import llava
        self.key = 'llava'
        args = super().set_model_params()
        
        set_seed(0)
        model_id = args.get('model_id')
        model, processor = llava.load_model(model_id, self.device)

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
        from Models import radfm
        self.key = 'radfm'
        args = super().set_model_params()
        self.model_path = args.get("model_path")
        model, text_tokenizer, image_padding_tokens = radfm.load_model(self.model_path, self.device)
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
                                         self.mode,
                                         self.device)
            response_list.append(outputs)
        return response_list

class BLIP2Answering(BaseAnsweringModel):

    def set_model_params(self):
        global blip2
        from Models import blip2
        self.key = 'blip2'
        args = super().set_model_params()
        model_id = args.get('model_id')
        model, processor = blip2.load_model(model_id, self.device)
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
        from Models import instructblip
        self.key = 'instructblip'
        args = super().set_model_params()
        model_id = args.get('model_id')
        model, processor = instructblip.load_model(model_id, self.device)
        self.model = model
        self.processor = processor
        if self.temperature == 0:
            self.temperature = 0.1

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


class MolmoAnswering(BaseAnsweringModel):

    def set_model_params(self):
        global molmo
        from Models import molmo
        self.key = 'molmo'
        args = super().set_model_params()
        model_id = args.get('model_id')
        model, processor = molmo.load_model(model_id, self.device)
        self.model = model
        self.processor = processor

    def ask_question(self, question, options, image_list):
        question = super().ask_question(question, options, image_list)
        response_list = []
        for image_path in image_list:
            outputs = molmo.ask_question(self.model, 
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
        from Models import med_flamingo
        self.key = 'med_flamingo'
        args = super().set_model_params()
        self.LLaMa_PATH = args.get('LLaMa_PATH')
        self.CHECKPOINT_PATH = args.get('CHECKPOINT_PATH')
        self.IMAGE_PATH = args.get('IMAGE_PATH')
        model, processor = med_flamingo.load_model(self.LLaMa_PATH, self.CHECKPOINT_PATH, self.device)
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


class LlamaAnswering(BaseAnsweringModel):

    def set_model_params(self):
        global llama
        from Models import llama
        self.key = 'llama'
        args = super().set_model_params()
        
        token = args.get('token')
        model_id = args.get('model_id')
        model, processor = llama.load_model(token, model_id, self.device)

        self.model = model
        self.processor = processor

    def ask_question(self, question, options, image_list):
        question = super().ask_question(question, options, image_list)
        response_list = []
        image_list = [Image.open(x) for x in image_list]
        for image in image_list:
            outputs = llama.ask_question(self.model, 
                                         self.processor, 
                                         question, 
                                         image, 
                                         self.mode,
                                         temperature=self.temperature,
                                         top_p=self.top_p, 
                                         num_beams=self.num_beams)
            response_list.append(outputs)

        return response_list


class MedVInTAnswering(BaseAnsweringModel):

    def set_model_params(self):
        global med_flamingo
        from Models import medvint
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
    'llava_med': LLaVAMedAnswering,
    'llava': LLaVAAnswering,
    'radfm': RadFMAnswering,
    'blip2': BLIP2Answering,
    'instructblip': InstructBLIPAnswering,
    'med_flamingo': MedFlamingoAnswering,
    'medvint': MedVInTAnswering,
    'llama': LlamaAnswering,
    'molmo': MolmoAnswering,
}

DEFAULT_MODEL_CONFIGS = {
    'gpt': f'{ROOT}/configs/Models/gpt/vanilla.json',
    'claude': f'{ROOT}/configs/Models/claude/vanilla.json',
    'gemini': f'{ROOT}/configs/Models/gemini/vanilla.json',
    'llava_med': f'{ROOT}/configs/Models/llava_med/vanilla.json',
    'llava': f'{ROOT}/configs/Models/llava/vanilla.json',
    'radfm': f'{ROOT}/configs/Models/radfm/vanilla.json',
    'blip2': f'{ROOT}/configs/Models/blip2/vanilla.json',
    'instructblip': f'{ROOT}/configs/Models/instructblip/vanilla.json',
    'med_flamingo': f'{ROOT}/configs/Models/med_flamingo/vanilla.json',
    'medvint': f'{ROOT}/configs/Models/medvint/vanilla.json',
    'llama': f'{ROOT}/configs/Models/llama/vanilla.json',
    'molmo': f'{ROOT}/configs/Models/molmo/vanilla.json',
}
