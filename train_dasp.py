from openprompt.plms import load_plm
from openprompt.prompts import SoftTemplate
from openprompt.prompts import ManualVerbalizer, KnowledgeableVerbalizer
from openprompt.data_utils import InputExample, InputFeatures
from openprompt import PromptForClassification
from openprompt import PromptDataLoader
from openprompt.data_utils.data_sampler import FewShotSampler
from MetaTask_dasp import MetaTaskDataLoader
from MetaLearner_dasp import Reptile, MAML
from utils import *
import argparse
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default='mtl-dataset/', type=str,
                        help="Path to dataset folder")
    
    parser.add_argument("--dataset", default = 'amazon', type=str,
                        help="amazon or mtl")
    
    parser.add_argument("--test_domain", default='all', type=str,
                        help="Test all domains or single domain")
    
    parser.add_argument("--seed", default = 777, type=int, 
                        help="Random seed")
    
    parser.add_argument("--plm_type", default='bert', type=str,
                        help='Type of Pre-trained language model')
    
    parser.add_argument("--plm", default='bert-base-uncased', type=str,
                        help = 'Pre-trained language model')
    
    parser.add_argument('--tune_plm', default = False, type=bool,
                        help = "Whether tune plm")
    
    parser.add_argument("--num_soft", default = 10, type = int,
                        help="Number of soft token in prompt template")
    
    parser.add_argument("--max_seq", default = 256, type = int,
                        help = "Maximum length of sequence")
    
    parser.add_argument("--epoch", default=10, type=int,
                        help="Training epoch")
    
    parser.add_argument("--bz", default=8, type=int,
                        help="Batch size")
    
    parser.add_argument("--k_spt", default=8, type=int,
                        help="Number of support samples per task")

    parser.add_argument("--k_qry", default=8, type=int,
                        help="Number of query samples per task")
    
    parser.add_argument("--num_task", default=1000, type=int,
                        help="Number of tasks")
    
    parser.add_argument("--task_bz", default=8, type=int,
                        help="Number of tasks in one batch")

    parser.add_argument("--inner_lr", default = 5e-3, type=float,
                        help = "Inner loop learning rate")
    
    parser.add_argument("--outer_lr", default=1e-3, type=float,
                        help="Outer loop learning rate")
    
    parser.add_argument("--inner_step", default=4, type=int,
                        help="Number of iteration in the inner loop")
    
    args = parser.parse_args()

    
    dataPath = args.data_path
    if args.dataset == "amazon":
        domains = ['books', 'dvd', 'electronics', 'kitchen_housewares']
    else:
        domains = ['apparel', 'baby', 'books', 'camera_photo',
                'dvd', 'electronics', 'health_personal_care', 'imdb', 'kitchen_housewares', 'magazines', 'MR', 'music', 'software', 'sports_outdoors', 'toys_games', 'video']
    
    if args.test_domain == 'all':
        test_domains = domains
    else:
        test_domains = [args.test_domain]
    seeds = args.seed

    model = args.plm_type
    model_path = args.plm
    tune_plm = args.tune_plm

    dataset = constructDataset(
        dataPath, domains, splits=['train', 'test'], seeds=None)

    for testDomain in test_domains:
        train_dataset = {}  # For construct meta-training task
        trainDomains = [domain for domain in domains if domain != testDomain]
        train_num = 0
        for trainDomain in domains:
            if trainDomain != testDomain:
                train_dataset[trainDomain] = []
                train_dataset[trainDomain].extend(
                    dataset[trainDomain]['train'] + dataset[trainDomain]['test'])
                train_num += len(train_dataset[trainDomain])

        #print(train_num)
        valid_dataset = dataset[testDomain]['train'] + \
            dataset[testDomain]['test']
        #test_dataset = dataset[testDomain]['train'] + dataset[testDomain]['test']
        #print(len(test_dataset))

        model = args.plm_type
        model_path = args.plm
        plm, tokenizer, model_config, WrapperClass = load_plm(
            model, model_path)

        #template = '{"placeholder": "text_a"} {"soft": "It was"} {"mask"} .'
        #template = '{"placeholder": "text_a"} This review is {"mask"} .'
        #template = '{"placeholder": "text_a"} This is a {"mask"} review .'
        #template = '{"placeholder": "text_a"} All in all , it was {"mask"} .'
        #template = '{"placeholder": "text_a"} In summary , it was {"mask"} .'
        template = '{"placeholder": "text_a"} {"mask"} .'
        soft_tokens = args.num_soft

        mytemplate = SoftTemplate(model=plm, text=template, tokenizer=tokenizer,
                                    num_tokens=soft_tokens, initialize_from_vocab=True)

        label_words = [['bad', 'negative'], ['good', 'positive']]
        myverbalizer = ManualVerbalizer(
            tokenizer=tokenizer, num_classes=2, label_words=label_words)


        prompt_model = PromptForClassification(
            plm=plm, template=mytemplate, verbalizer=myverbalizer, freeze_plm=(not tune_plm), plm_eval_mode=False)
        print(promt_model)

        ####### wandb section
        run_name = f'Meta test on {args.dataset} {testDomain}'
        job_type = f'{args.dataset} meta sp'
        config = dict(batch_size=args.bz,
                        epoch=args.epoch,
                        inner_learning_rate=args.inner_lr,
                        outer_learning_rate=args.outer_lr,
                        inner_update_step=args.inner_step,
                        task_bz=args.task_bz,
                        num_tasks=args.num_task,
                        soft_tokens=args.num_soft,
                        lw=label_words,
                        max_seq_length=args.max_seq,
                        tune_plm=args.tune_plm)
        config_str = str(config)
        
        wandb.init(project='DASP', name=run_name, config=config,
                    notes=model_path, job_type=job_type, reinit=True)
        wandb.watch(prompt_model)
        #######

        trainLoader = MetaTaskDataLoader(
            dataset=train_dataset,
            trainDomains=trainDomains,
            tokenizer=tokenizer,
            template=mytemplate,
            tokenizer_wrapper_class=WrapperClass,
            num_tasks=args.num_task,
            k_spt=args.k_spy,
            k_qry=args.k_qry,
            shuffle=True,
            batch_size=args.task_bz,
            max_seq_length=args.max_seq,
        )
        validLoader = PromptDataLoader(
            dataset=valid_dataset,
            tokenizer=tokenizer,
            template=mytemplate,
            tokenizer_wrapper_class=WrapperClass,
            shuffle=False,
            batch_size=args.bz,
            max_seq_length=args.max_seq,
        )

        print('Run at train domain: {}, test domain: {}'.format(
            trainDomains, testDomain), flush=True)

        learner = Reptile(
            model=prompt_model, outer_update_lr=args.outer_lr, inner_update_lr=args.inner_lr, inner_update_step=args.inner_step, tune_plm=args.tune_plm, device=device)
        #learner = MAML(
        #    model = prompt_model, outer_update_lr=outer_learning_rate, inner_update_lr=inner_learning_rate, inner_update_step = inner_update_step, tune_plm = tune_plm, device = device)

        loss_func = torch.nn.CrossEntropyLoss()

        global_step = 0
        save_acc = 0
        best_acc = -1
        for epoch in range(epochs):

            for step, task_batch in enumerate(trainLoader):
                loss, acc = learner(task_batch)
                if step % 10 == 0:
                    print(
                        f'Epoch: {epoch}\tStep: {step}\ttraining loss: {loss}\ttraining acc: {acc}')
                wandb.log({'train loss': loss, 'train acc': acc})

                if global_step % 50 == 0:
                    print("------------Testing Mode------------")
                    test_acc, test_loss = learner.zero_shot_eval(
                        validLoader)
                    print(
                        f'Epoch: {epoch}\tStep: {step}\ttest loss: {test_loss}\ttest acc: {test_acc}\n')
                    wandb.log(
                        {'test acc': test_acc, 'test loss': test_loss})

                global_step += 1
            if test_acc > best_acc:
                model_save_dir = f'/home/thesis/PromptLearning/model/{testDomain}/'
                model_save_name = f'meta_sp_epoch{epoch}_acc{round(test_acc,2)}.pt'
                save_model(prompt_model, model_save_dir, model_save_name)
                best_acc = test_acc


if __name__ == '__main__':
    main()