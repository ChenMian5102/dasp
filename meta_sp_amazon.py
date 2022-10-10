from openprompt.plms import load_plm
from openprompt.prompts import SoftTemplate
from openprompt.prompts import ManualVerbalizer, KnowledgeableVerbalizer
from openprompt.data_utils import InputExample, InputFeatures
from openprompt import PromptForClassification
from openprompt import PromptDataLoader
from openprompt.data_utils.data_sampler import FewShotSampler
from MetaTask import MetaTaskDataLoader
from MetaLearner import Reptile, MAML
from utils import *
import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
import numpy as np
import random
import os
import tqdm
import wandb

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

dataPath = '/home/thesis/amazon/mtl-dataset/'
#dataPath = '/home/thesis/amazon/ziser17/'
dataPath_16 = '/home/thesis/amazon/16-shot'
splits = ['train', 'test']
domains = ['apparel', 'baby', 'books', 'camera_photo',
           'dvd', 'electronics', 'health_personal_care', 'imdb', 'kitchen_housewares', 'magazines', 'MR', 'music', 'software', 'sports_outdoors', 'toys_games', 'video']
test_domains = ['dvd', 'books', 'electronics', 'kitchen_housewares']
test_domains = domains
#domains = ['books', 'dvd', 'electronics', 'kitchen_housewares']
seeds = [7, 10, 21, 26, 85]

model = 'bert'
model_path = 'bert-base-uncased'
tune_plm = False

dataset = constructDataset(dataPath, domains, splits, seeds=None)
#fewShotDataset = constructDataset(dataPath_16, domains, splits, seeds)

for n in range(20,21):
    for testDomain in test_domains:
        #testDomain = 'kitchen_housewares'
        train_dataset = {}
        trainDomains = [domain for domain in domains if domain != testDomain]
        train_num = 0
        for trainDomain in domains:
            if trainDomain != testDomain:
                train_dataset[trainDomain] = []
                train_dataset[trainDomain].extend(dataset[trainDomain]['train'] + dataset[trainDomain]['test'])
                train_num += len(train_dataset[trainDomain])
        
        #print(train_num)
        valid_dataset = dataset[testDomain]['train'] + dataset[testDomain]['test']
        #test_dataset = dataset[testDomain]['train'] + dataset[testDomain]['test']
        #print(len(test_dataset))
        
        model = 'bert'
        model_path = 'bert-base-uncased'
        plm, tokenizer, model_config, WrapperClass = load_plm(model, model_path)

        
        #template = '{"placeholder": "text_a"} {"soft": "It was"} {"mask"} .'
        #template = '{"placeholder": "text_a"} This review is {"mask"} .'
        # template = '{"placeholder": "text_a"} This is a {"mask"} review .'
        # template = '{"placeholder": "text_a"} All in all , it was {"mask"} .'
        #template = '{"placeholder": "text_a"} In summary , it was {"mask"} .'
        template = '{"placeholder": "text_a"} {"mask"} .'
        soft_tokens = n

        mytemplate = SoftTemplate(model=plm, text=template, tokenizer=tokenizer,
                                num_tokens=soft_tokens, initialize_from_vocab=True)
        
        label_words = [['bad', 'negative'], ['good', 'positive']]
        myverbalizer = ManualVerbalizer(tokenizer=tokenizer, num_classes=2, label_words=label_words)
        #myverbalizer = KnowledgeableVerbalizer(tokenizer, num_classes=2).from_file(
        #    "/home/thesis/OpenPrompt/scripts/TextClassification/amazon/knowledgeable_verbalizer.txt")

        prompt_model = PromptForClassification(
            plm=plm, template=mytemplate, verbalizer=myverbalizer, freeze_plm=(not tune_plm), plm_eval_mode=False)
        #prompt_model = torch.load(
        #    "PromptLearning/model/kitchen_housewares/meta_sp_{'batch_size': 8, 'epoch': 10, 'inner_learning_rate': 0.01, 'outer_learning_rate': 0.001, 'inner_update_step': 4, 'task_bz': 4, 'num_tasks': 500, 'soft_tokens': 2, 'lw': [['bad', 'negative'], ['good', 'positive']], 'max_seq_length': 256}_e3.pt")
        #sp_param = [p for name, p in prompt_model.template.named_parameters() if 'raw_embedding' not in name]
        #print(sp_param[0].shape)
        #exit()
        

        
        #prompt_model.to(device)
        #print(promp_model)

        batch_size = 8
        num_tasks = 2000
        k_spt = 8
        k_qry = 8
        task_bz = 8
        epochs = 10
        inner_learning_rate = 0.005
        outer_learning_rate = 0.001
        inner_update_step = 4
        #max_seq_length = 512-soft_tokens
        max_seq_length = 256

        ####### wandb section
        run_name = 'Meta test on FDU-MTL ' + testDomain
        job_type = 'amazon meta sp'
        config = dict(batch_size=batch_size,
                        epoch=epochs,
                        inner_learning_rate=inner_learning_rate,
                        outer_learning_rate=outer_learning_rate,
                        inner_update_step= inner_update_step,
                        task_bz = task_bz,
                        num_tasks=num_tasks,
                        soft_tokens = soft_tokens,
                        lw = label_words,
                        max_seq_length=max_seq_length,
                        tune_plm = tune_plm)
        config_str = str(config)
        print(len(config_str))
        wandb.init(project='prompt_learning', name=run_name, config=config,
                notes=model_path, job_type=job_type, reinit=True)
        wandb.watch(prompt_model)
        #######

        trainLoader = MetaTaskDataLoader(
            dataset=train_dataset,
            trainDomains=trainDomains,
            tokenizer=tokenizer,
            template=mytemplate,
            tokenizer_wrapper_class=WrapperClass,
            num_tasks = num_tasks,
            k_spt = k_spt,
            k_qry = k_qry,
            shuffle=True,
            batch_size=task_bz,
            max_seq_length=max_seq_length,
        )
        validLoader = PromptDataLoader(
            dataset=valid_dataset,
            tokenizer=tokenizer,
            template=mytemplate,
            tokenizer_wrapper_class=WrapperClass,
            shuffle=False,
            batch_size=batch_size,
            max_seq_length=max_seq_length,
        )

        print('Run at train domain: {}, test domain: {}'.format(
            trainDomains, testDomain), flush=True)


        learner = Reptile(
            model = prompt_model, outer_update_lr=outer_learning_rate, inner_update_lr=inner_learning_rate, inner_update_step = inner_update_step, tune_plm = tune_plm, device = device)
        #learner = MAML(
        #    model = prompt_model, outer_update_lr=outer_learning_rate, inner_update_lr=inner_learning_rate, inner_update_step = inner_update_step, tune_plm = tune_plm, device = device)

        loss_func = torch.nn.CrossEntropyLoss()

        global_step = 0
        save_acc = 0
        for epoch in range(epochs):

            for step, task_batch in enumerate(trainLoader):
                loss, acc = learner(task_batch)
                if step % 5 == 0:
                    print(f'Epoch: {epoch}\tStep: {step}\ttraining loss: {loss}\ttraining acc: {acc}')
                wandb.log({'train loss': loss, 'train acc': acc})

                if global_step % 20 == 0:
                    print("------------Testing Mode------------")
                    test_acc, test_loss = learner.zero_shot_eval(validLoader)
                    print(f'Epoch: {epoch}\tStep: {step}\ttest loss: {test_loss}\ttest acc: {test_acc}\n')
                    wandb.log({'test acc': test_acc, 'test loss': test_loss})

                global_step+=1
            if test_acc > 0.87:
                model_save_dir = f'/home/thesis/PromptLearning/model/{testDomain}/'
                model_save_name = f'meta_sp_{test_acc}_t5.pt'
                save_model(prompt_model, model_save_dir, model_save_name)
            
                    
                    


        