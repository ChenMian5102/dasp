from pickle import FALSE
from torch.utils.data.sampler import RandomSampler
import random
from transformers.configuration_utils import PretrainedConfig
from transformers.generation_utils import GenerationMixin
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import *
from openprompt.data_utils import InputExample, InputFeatures
from torch.utils.data._utils.collate import default_collate
from tqdm.std import tqdm
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.utils.dummy_pt_objects import PreTrainedModel
from openprompt.plms.utils import TokenizerWrapper
from openprompt.prompt_base import Template, Verbalizer
from collections import defaultdict
from openprompt.utils import round_list, signature
import numpy as np
from torch.utils.data import DataLoader
from yacs.config import CfgNode
from openprompt.utils.logging import logger
from transformers import  AdamW, get_linear_schedule_with_warmup

def collate_fct(tasks):
    r'''
    This function is used to collate the input_features.

    Args:
        batch (:obj:`List[Union[Dict, InputFeatures]]`): A batch of the current data.

    Returns:
        :obj:`InputFeatures`: Return the :py:class:`~openprompt.data_utils.data_utils.InputFeatures of the current batch of data.
    '''
    return_tasks = []
    
    for t in tasks:
        task = []
        for data in t:
            elem = data[0]
            return_dict = {}
            for key in elem:
                if key == "encoded_tgt_text":
                    return_dict[key] = [d[key] for d in data]
                else:
                    try:
                        return_dict[key] = default_collate([d[key] for d in data])
                    except:
                        print(f"key{key}\n d {[data[i][key] for i in range(len(data))]} ")
            return_features = InputFeatures(**return_dict)
            task.append(return_features)
        return_tasks.append(task)
    return return_tasks

class MetaTaskDataLoader(object):
    r"""
    PromptDataLoader wraps the orginal dataset. The input data is firstly wrapped with the
    prompt's template, and then is tokenized by a wrapperd-tokenizer.

    Args:
        dataset (:obj:`Dataset` or :obj:`List`): Either a DatasetObject or a list containing the input examples.
        template (:obj:`Template`): A derived class of of :obj:`Template`
        tokenizer (:obj:`PretrainedTokenizer`): The pretrained tokenizer.
        tokenizer_wrapper_class (:cls:`TokenizerWrapper`): The class of tokenizer wrapper.
        max_seq_length (:obj:`int`, optional): The max sequence length of the input ids. It's used to trucate sentences.
        batch_size (:obj:`int`, optional): The batch_size of data loader
        teacher_forcing (:obj:`bool`, optional): Whether to fill the mask with target text. Set to true in training generation model.
        decoder_max_length (:obj:`int`, optional): the decoder maximum length of an encoder-decoder model.
        predict_eos_token (:obj:`bool`, optional): Whether to predict the <eos> token. Suggest to set to true in generation.
        truncate_method (:obj:`bool`, optional): the truncate method to use. select from `head`, `tail`, `balanced`.
        kwargs  :Other kwargs that might be passed into a tokenizer wrapper.
    """
    def __init__(self,
                 dataset: Union[Dataset, List],
                 trainDomains,
                 template: Template,
                 tokenizer_wrapper: Optional[TokenizerWrapper] = None,
                 tokenizer: PreTrainedTokenizer = None,
                 tokenizer_wrapper_class = None,
                 verbalizer: Optional[Verbalizer] = None,
                 max_seq_length: Optional[str] = 512,
                 batch_size: Optional[int] = 1,
                 num_tasks: Optional[int] = 200,
                 k_spt: Optional[int] = 4,
                 k_qry: Optional[int] = 8,
                 shuffle: Optional[bool] = False,
                 teacher_forcing: Optional[bool] = False,
                 decoder_max_length: Optional[int] = -1,
                 predict_eos_token: Optional[bool] = False,
                 truncate_method: Optional[str] = "tail",
                 drop_last: Optional[bool] = False,
                 **kwargs,
                ):

        assert hasattr(dataset, "__iter__"), f"The dataset must have __iter__ method. dataset is {dataset}"
        assert hasattr(dataset, "__len__"), f"The dataset must have __len__ method. dataset is {dataset}"
        self.raw_dataset = dataset
        self.trainDomains = trainDomains

        self.num_tasks = num_tasks
        self.k_spt = k_spt
        self.k_qry = k_qry
        self.tasks = []

        self.wrapped_dataset = []
        self.tensor_dataset = []
        self.template = template
        self.verbalizer = verbalizer
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.teacher_forcing = teacher_forcing

        if tokenizer_wrapper is None:
            if tokenizer_wrapper_class is None:
                raise RuntimeError("Either wrapped_tokenizer or tokenizer_wrapper_class should be specified.")
            if tokenizer is None:
                raise RuntimeError("No tokenizer specified to instantiate tokenizer_wrapper.")

            tokenizer_wrapper_init_keys = signature(tokenizer_wrapper_class.__init__).args
            prepare_kwargs = {
                "max_seq_length" : max_seq_length,
                "truncate_method" : truncate_method,
                "decoder_max_length" : decoder_max_length,
                "predict_eos_token" : predict_eos_token,
                "tokenizer" : tokenizer,
                **kwargs,
            }

            to_pass_kwargs = {key: prepare_kwargs[key] for key in prepare_kwargs if key in tokenizer_wrapper_init_keys}
            self.tokenizer_wrapper = tokenizer_wrapper_class(**to_pass_kwargs)
        else:
            self.tokenizer_wrapper = tokenizer_wrapper

        # check the satisfiability of each component
        assert hasattr(self.template, 'wrap_one_example'), "Your prompt has no function variable \
                                                         named wrap_one_example"

        # processs
        self.wrap()
        self.tokenize()
        self.creat_tasks(self.num_tasks)

        if self.shuffle:
            sampler = RandomSampler(self.tasks)
        else:
            sampler = None

        # self.dataloader = DataLoader(
        #     self.tasks,
        #     batch_size = self.batch_size,
        #     sampler= sampler,
        #     collate_fn = InputFeatures.collate_fct,
        #     drop_last = drop_last,
        # )
        self.dataloader = DataLoader(
            self.tasks,
            batch_size = self.batch_size,
            sampler= sampler,
            collate_fn = collate_fct,
            drop_last = drop_last,
        )


    def wrap(self):
        r"""A simple interface to pass the examples to prompt, and wrap the text with template.
        """
        if isinstance(self.raw_dataset, Dataset) or isinstance(self.raw_dataset, List) or isinstance(self.raw_dataset, Dict):
            assert len(self.raw_dataset) > 0, 'The dataset to be wrapped is empty.'
            # for idx, example in tqdm(enumerate(self.raw_dataset),desc='Wrapping'):
            if isinstance(self.raw_dataset, Dict):
                self.wrapped_dataset = {}
                for domain in self.trainDomains:
                    self.wrapped_dataset[domain] = []
                    for idx, example in enumerate(self.raw_dataset[domain]):
                        if self.verbalizer is not None and hasattr(self.verbalizer, 'wrap_one_example'):
                            example = self.verbalizer.wrap_one_example(example)
                        wrapped_example = self.template.wrap_one_example(example)
                        self.wrapped_dataset[domain].append(wrapped_example)
            else:
                for idx, example in enumerate(self.raw_dataset):
                    if self.verbalizer is not None and hasattr(self.verbalizer, 'wrap_one_example'): # some verbalizer may also process the example.
                        example = self.verbalizer.wrap_one_example(example)
                    wrapped_example = self.template.wrap_one_example(example)
                    self.wrapped_dataset.append(wrapped_example)
        else:
            raise NotImplementedError

    def tokenize(self) -> None:
        r"""Pass the wraped text into a prompt-specialized tokenizer,
           the true PretrainedTokenizer inside the tokenizer is flexible, e.g. AlBert, Bert, T5,...
        """
        if isinstance(self.wrapped_dataset, Dict):
            self.tensor_dataset = {}
            for domain in self.trainDomains:
                self.tensor_dataset[domain] = []
                for idx, wrapped_example in tqdm(enumerate(self.wrapped_dataset[domain]), desc=f'tokenizing on domain {domain}'):
                    inputfeatures = InputFeatures(**self.tokenizer_wrapper.tokenize_one_example(
                        wrapped_example, self.teacher_forcing), **wrapped_example[1]).to_tensor()
                    self.tensor_dataset[domain].append(inputfeatures)

        else:
            for idx, wrapped_example in tqdm(enumerate(self.wrapped_dataset),desc='tokenizing'):
            # for idx, wrapped_example in enumerate(self.wrapped_dataset):
                inputfeatures = InputFeatures(**self.tokenizer_wrapper.tokenize_one_example(wrapped_example, self.teacher_forcing), **wrapped_example[1]).to_tensor()
                self.tensor_dataset.append(inputfeatures)

    def creat_tasks(self, num_tasks):
        for i in tqdm(range(num_tasks), desc='creating task'):
            selected_domains = random.sample(self.trainDomains, 3)
            spt_domain = selected_domains[0:2]
            qry_domain = selected_domains[2]
            spt_features = random.sample(
                self.tensor_dataset[spt_domain[0]], int(self.k_spt/2))
            spt_features.extend(random.sample(
                self.tensor_dataset[spt_domain[1]], int(self.k_spt/2)))
            # spt_features.extend(random.sample(
            #     self.tensor_dataset[spt_domain[2]], int(self.k_spt/2)))
            qry_features = random.sample(self.tensor_dataset[qry_domain], self.k_qry)

            # selected_features = random.sample(self.tensor_dataset, self.k_spt+self.k_qry)
            # random.shuffle(selected_features)
            # spt_features = selected_features[:self.k_spt]
            # qry_features = selected_features[self.k_spt:]

            self.tasks.append([spt_features, qry_features])

        print(len(self.tasks))

    def __len__(self):
        return  len(self.dataloader)

    def __iter__(self,):
        return self.dataloader.__iter__()