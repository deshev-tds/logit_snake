import sys
import time
import torch
import numpy as np
from argparse import Namespace
from typing import Tuple, List, Any, Optional, Callable, Union, Dict
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig
from transformers.utils.generic import ModelOutput
from utils.wrapper import ModelConfiguration
from utils.cost import CostCounter
from openai import OpenAI


class LLM:
    is_hf_model = True

    def __init__(self, args):
        super().__init__()

        self.verbose = args.get('verbose', False)
        self.max_attempts = args.get('max_attempts', 5)

        if self.is_hf_model:
            self.tokenizer = AutoTokenizer.from_pretrained(args["model_path"],
                                                           trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(args["model_path"],
                                                              trust_remote_code=True,
                                                              device_map='auto')
        else:  # API Model
            self.generation_config = {}

        self.counter = CostCounter()

    def get_generation_config(self, key: str):
        if self.is_hf_model:
            return getattr(self.model.generation_config, key, None)
        else:
            return getattr(self, key, None)

    def set_generation_config(self, key: str, value: Any):
        if value is None:
            return

        if self.is_hf_model:
            setattr(self.model.generation_config, key, value)
        else:
            self.generation_config[key] = value

    def get_token_probabilities(self, logits: Union[torch.Tensor, Dict], tokens: Optional[List[str]], offset: int = 0):
        if self.is_hf_model and isinstance(logits, torch.Tensor) and tokens is not None:
            token_ids = [self.tokenizer(token, return_tensors='pt').input_ids[0, offset] for token in tokens]
            output = logits[offset, token_ids]

        elif not self.is_hf_model and isinstance(logits, Dict) and tokens is not None:
            output = torch.ones(len(tokens)) * -torch.inf
            for i, target in enumerate(tokens):
                for keyword, prob in logits.items():
                    if target in keyword:
                        output[i] = prob
                        break

        else:
            output = None

        return output

    def inner_understand(self,
                         query: str,
                         history: List[Tuple[str, str]],
                         targets: Optional[List[str]],
                         **kwargs) -> Tuple[torch.Tensor, str, Any]:
        raise NotImplementedError()

    def understand(self,
                   query: str,
                   history: List[Tuple[str, str]] = None,
                   targets: Optional[List[str]] = None,
                   post_process: Optional[Callable[[torch.Tensor, str, ModelOutput], bool]] = None,
                   **kwargs) -> Tuple[torch.Tensor, str, ModelOutput]:
        """
        :param query: string
        :param history: list of tuple (query, answer)
        :param targets: desired output tokens, used to compute logits
        :param post_process: optional function that takes outputs and returns a post-processed output
        :param kwargs:
        :return:
        """
        if history is None:
            history = []

        if self.verbose:
            print()
            print("###Query")
            print(query.replace("\n\n", "\n"))

        # retry if raise errors
        attempts = 0
        while attempts < self.max_attempts:
            try:
                # enable sampling if attempts > 0
                with ModelConfiguration(self, config_name='do_sample', value=True, enable=attempts > 0):
                    prob, response, outputs = self.inner_understand(query, history, targets, **kwargs)

                if self.verbose:
                    print("###Response")
                    print(response.replace("\n\n", "\n"))

                # output check
                if post_process:
                    prob, response, outputs = post_process(prob, response, outputs)
                break

            except Exception as e:
                attempts += 1
                time.sleep(5)
                if attempts == self.max_attempts:
                    raise e

        # counter
        if self.counter:
            self.counter.step(len(outputs.scores) if outputs else None)

        return prob, response, outputs


class QWenModel(LLM):
    def __init__(self, args):
        super().__init__(args)
        sys.path.append(args['model_path'])

    def make_context(self, query: str, history: List[Tuple[str, str]] = None, max_window_size: int = 6144):
        system = "You are a helpful assistant."

        if history is None:
            history = []

        im_start, im_end = "<|im_start|>", "<|im_end|>"
        im_start_tokens = [self.tokenizer.im_start_id]
        im_end_tokens = [self.tokenizer.im_end_id]
        nl_tokens = self.tokenizer.encode("\n")

        def _tokenize_str(role, content):
            return f"{role}\n{content}", self.tokenizer.encode(
                role, allowed_special=set()
            ) + nl_tokens + self.tokenizer.encode(content, allowed_special=set())

        system_text, system_tokens_part = _tokenize_str("system", system)
        system_tokens = im_start_tokens + system_tokens_part + im_end_tokens

        raw_text = ""
        context_tokens = []

        for turn_query, turn_response in reversed(history):
            query_text, query_tokens_part = _tokenize_str("user", turn_query)
            query_tokens = im_start_tokens + query_tokens_part + im_end_tokens
            response_text, response_tokens_part = _tokenize_str("assistant", turn_response)
            response_tokens = im_start_tokens + response_tokens_part + im_end_tokens

            next_context_tokens = nl_tokens + query_tokens + nl_tokens + response_tokens
            prev_chat = f"\n{im_start}{query_text}{im_end}\n{im_start}{response_text}{im_end}"

            current_context_size = (len(system_tokens) + len(next_context_tokens) + len(context_tokens))
            if current_context_size < max_window_size:
                context_tokens = next_context_tokens + context_tokens
                raw_text = prev_chat + raw_text
            else:
                break

        context_tokens = system_tokens + context_tokens
        raw_text = f"{im_start}{system_text}{im_end}{raw_text}"
        context_tokens += (
                nl_tokens
                + im_start_tokens
                + _tokenize_str("user", query)[1]
                + im_end_tokens
                + nl_tokens
                + im_start_tokens
                + self.tokenizer.encode("assistant")
                + nl_tokens
        )
        raw_text += f"\n{im_start}user\n{query}{im_end}\n{im_start}assistant\n"

        return raw_text, context_tokens

    def decode_chatml(self, tokens: List[int], raw_text_len: int, context_length: int):

        eod_token_ids = [self.tokenizer.im_start_id, self.tokenizer.im_end_id],

        eod_token_idx = context_length
        for eod_token_idx in range(context_length, len(tokens)):
            if tokens[eod_token_idx] in eod_token_ids:
                break

        trim_decode_tokens = self.tokenizer.decode(tokens[:eod_token_idx])[raw_text_len:]
        trim_decode_tokens = trim_decode_tokens.strip()
        return trim_decode_tokens

    def inner_understand(self, query, history, targets, **kwargs):
        raw_text, context_tokens = self.make_context(query,
                                                     history=history,
                                                     max_window_size=6144)

        input_ids = torch.tensor([context_tokens]).to(self.model.device)

        self.model.generation_config.pad_token_id = self.model.generation_config.eos_token_id
        outputs = self.model.generate(input_ids,
                                      return_dict_in_generate=True,
                                      generation_config=self.model.generation_config,
                                      output_scores=True,
                                      **kwargs)

        # phase outputs
        logits = torch.cat([logit for logit in outputs.scores], dim=0)
        logits = self.get_token_probabilities(logits, targets)
        response = self.decode_chatml(outputs.sequences[0],
                                      raw_text_len=len(raw_text),
                                      context_length=len(context_tokens))

        return logits, response, outputs


class QWen2Model(LLM):
    def inner_understand(self, query, history, targets, **kwargs):
        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        for (his_q, his_r) in history:
            messages.append({"role": "user", "content": his_q})
            messages.append({"role": "assistant", "content": his_r})
        messages.append({"role": "user", "content": query})

        text = self.tokenizer.apply_chat_template(messages,
                                                  tokenize=False,
                                                  add_generation_prompt=True)
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(**model_inputs,
                                      return_dict_in_generate=True,
                                      generation_config=self.model.generation_config,
                                      output_scores=True,
                                      **kwargs)

        # phase outputs
        logits = torch.cat([logit for logit in outputs.scores], dim=0)
        logits = self.get_token_probabilities(logits, targets)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, outputs.sequences)]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return logits, response, outputs


class LlamaModel(LLM):
    def inner_understand(self, query, history, targets, **kwargs):
        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        for (his_q, his_r) in history:
            messages.append({"role": "user", "content": his_q})
            messages.append({"role": "assistant", "content": his_r})
        messages.append({"role": "user", "content": query})

        text = self.tokenizer.apply_chat_template(messages,
                                                  tokenize=False,
                                                  add_generation_prompt=True)
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(**model_inputs,
                                      return_dict_in_generate=True,
                                      generation_config=self.model.generation_config,
                                      output_scores=True,
                                      **kwargs)

        # phase outputs
        logits = torch.cat([logit for logit in outputs.scores], dim=0)
        logits = self.get_token_probabilities(logits, targets, offset=1)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, outputs.sequences)]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return logits, response, outputs


class BaichuanModel(LLM):
    def build_chat_input(self, model, tokenizer, messages: List[dict], max_new_tokens: int = 0):
        def _parse_messages(messages, split_role="user"):
            system, rounds = "", []
            round = []
            for i, message in enumerate(messages):
                if message["role"] == "system":
                    assert i == 0
                    system = message["content"]
                    continue
                if message["role"] == split_role and round:
                    rounds.append(round)
                    round = []
                round.append(message)
            if round:
                rounds.append(round)
            return system, rounds

        max_new_tokens = max_new_tokens or model.generation_config.max_new_tokens
        max_input_tokens = model.config.model_max_length - max_new_tokens
        system, rounds = _parse_messages(messages, split_role="user")
        system_tokens = tokenizer.encode(system)
        max_history_tokens = max_input_tokens - len(system_tokens)

        history_tokens = []
        for round in rounds[::-1]:
            round_tokens = []
            for message in round:
                if message["role"] == "user":
                    round_tokens.append(model.generation_config.user_token_id)
                else:
                    round_tokens.append(model.generation_config.assistant_token_id)
                round_tokens.extend(tokenizer.encode(message["content"]))
            if len(history_tokens) == 0 or len(history_tokens) + len(round_tokens) <= max_history_tokens:
                history_tokens = round_tokens + history_tokens  # concat left
                if len(history_tokens) < max_history_tokens:
                    continue
            break

        input_tokens = system_tokens + history_tokens
        if messages[-1]["role"] != "assistant":
            input_tokens.append(model.generation_config.assistant_token_id)
        input_tokens = input_tokens[-max_input_tokens:]  # truncate left
        return torch.LongTensor([input_tokens]).to(model.device)

    def inner_understand(self, query, history, targets, **kwargs):
        messages = []
        for (his_q, his_r) in history:
            messages.append({"role": "user", "content": his_q})
            messages.append({"role": "assistant", "content": his_r})
        messages.append({"role": "user", "content": query})
        input_ids = self.build_chat_input(self.model, self.tokenizer, messages, self.model.generation_config.max_new_tokens)
        outputs = self.model.generate(input_ids,
                                      return_dict_in_generate=True,
                                      generation_config=self.model.generation_config,
                                      output_scores=True,
                                      **kwargs)

        # phase outputs
        logits = torch.cat([logit for logit in outputs.scores], dim=0)
        logits = self.get_token_probabilities(logits, targets)
        response = self.tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)

        return logits, response, outputs


class ChatGLMModel(LLM):
    def __init__(self, args):
        super().__init__(args)

        self.eos_token_id = [self.tokenizer.eos_token_id, self.tokenizer.get_command("<|user|>"),
                             self.tokenizer.get_command("<|observation|>")]

    def inner_understand(self, query, history, targets, **kwargs):
        messages = []
        for (his_q, his_r) in history:
            messages.append({"role": "user", "content": his_q})
            messages.append({"role": "assistant", "content": his_r})

        inputs = self.tokenizer.build_chat_input(query, messages, role='user')
        input_ids = inputs.input_ids.to(self.model.device)
        outputs = self.model.generate(input_ids,
                                      generation_config=self.model.generation_config,
                                      eos_token_id=self.eos_token_id,
                                      return_dict_in_generate=True,
                                      output_scores=True)

        # phase outputs
        logits = torch.cat([logit for logit in outputs.scores], dim=0)
        logits = self.get_token_probabilities(logits, targets, offset=2)
        response = self.tokenizer.decode(outputs.sequences[0][len(input_ids[0]):], skip_special_tokens=True)

        return logits, response, outputs


class APIModel(LLM):
    is_hf_model = False

    def __init__(self, args: Dict):
        super().__init__(args)

        self.model_name = args['model_name']
        self.server = args['server']

        if 'aliyun' == self.server:
            key_file = './utils/apis/aliyun.key'
            base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        elif 'deepseek' == self.server:
            key_file = './utils/apis/deepseek.key'
            base_url = "https://api.deepseek.com"
        elif 'atalk' == self.server:
            key_file = './utils/apis/atalk.key'
            base_url = "https://api.atalk-ai.com"
        elif 'anywhere' == self.server:
            key_file = './utils/apis/anywhere.key'
            base_url = "https://api.chatanywhere.tech/v1"
        else:
            raise ValueError()

        with open(key_file, 'r') as f:
            api_key = f.read().strip()

        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def inner_understand(self, query, history, targets, **kwargs):
        messages = []
        for (his_q, his_r) in history:
            messages.append({"role": "user", "content": his_q})
            messages.append({"role": "assistant", "content": his_r})
        messages.append({"role": "user", "content": query})

        kwargs['temperature'] = self.generation_config['temperature'] if self.generation_config['do_sample'] else 0.0
        kwargs['max_tokens'] = self.generation_config.get('max_new_tokens', None)
        kwargs['logprobs'] = True if targets else False
        kwargs['top_logprobs'] = 20 if targets else None

        output = self.client.chat.completions.create(model=self.model_name,
                                                     messages=messages,
                                                     **kwargs)

        # phrase outputs
        if self.server == 'atalk':
            return None, output, None

        response = output.choices[0].message.content
        logits = {ele.token: ele.logprob for ele in output.choices[0].logprobs.content[0].top_logprobs} if targets else None
        logits = self.get_token_probabilities(logits, targets)
        return logits, response, Namespace(scores=[1] * output.usage.completion_tokens)


class NLIModel:
    is_hf_model = False
    is_modelscope_model = False

    def __init__(self, model_path):
        if 'deberta' in model_path:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path,
                                                                            device_map="cuda:0")
            self.is_hf_model = True
        elif 'nlp_structbert_nli_chinese' in model_path:
            from modelscope.pipelines import pipeline

            self.pipeline = pipeline('nli', model_path, device='gpu')
            self.is_modelscope_model = True
        else:
            raise ValueError()

    def understand(self, sentence, other):
        if self.is_hf_model:
            inputs = f"[CLS] {sentence} [SEP] {other} [SEP]"
            inputs = self.tokenizer(inputs, return_tensors='pt').to(self.model.device)
            output = self.model(**inputs)
            logits = output.logits.cpu()
            response = ["contradict", 'neutral', 'entail'][logits.argmax().item()]

        elif self.is_modelscope_model:
            output = self.pipeline(input=(sentence, other))
            logits = output["scores"]
            response = output['labels'][np.argmax(logits).item()]
            if response == '蕴涵':
                response = '相同'

        else:
            raise ValueError()

        print("NLI result:", response)
        return logits, response, output


def load_model(args: Dict):
    if args.get("model", None):
        return args["model"]

    model_name = args['model_name']
    if 'baichuan' in model_name:
        model = BaichuanModel(args)
    elif 'llama' in model_name:
        model = LlamaModel(args)
    elif 'qwen2' in model_name or 'qwen1_5' in model_name:
        model = QWen2Model(args)
    elif 'qwen' in model_name:
        model = QWenModel(args)
    elif 'chatglm' in model_name:
        model = ChatGLMModel(args)
    else:
        model = APIModel(args)

    # generation config
    model.set_generation_config('do_sample', args.get("do_sample", False))
    model.set_generation_config('max_new_tokens', args.get("max_new_tokens"))
    model.set_generation_config('temperature', args.get('temperature', 1.0))

    args["model"] = model
    return model
