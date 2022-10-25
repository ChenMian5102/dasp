from openprompt.plms import load_plm
from openprompt.prompts import SoftTemplate
from openprompt.prompts import ManualVerbalizer, KnowledgeableVerbalizer
from openprompt.data_utils import InputExample
from openprompt import PromptForClassification
from openprompt import PromptDataLoader
from openprompt.data_utils.data_sampler import FewShotSampler
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
dataPath_16 = '/home/thesis/amazon/16-shot'
splits = ['train', 'test']
domains = ['apparel', 'baby', 'books', 'camera_photo',
           'dvd', 'electronics', 'health_personal_care', 'imdb', 'kitchen_housewares', 'magazines', 'MR', 'music', 'software', 'sports_outdoors', 'toys_games', 'video']seeds = [7, 10, 21, 26, 85]


tune_plm = False

dataset = constructDataset(dataPath, domains, splits, seeds=None)
fewShotDataset = constructDataset(dataPath_16, domains, splits, seeds)

for testDomain in domains[::-1]:
    testDomain = 'video'
    train_dataset = []
    for trainDomain in domains:
        if trainDomain != testDomain:
            train_dataset.extend(dataset[trainDomain]['train'] + dataset[trainDomain]['test'])
        
    valid_dataset = dataset[testDomain]['train'] + dataset[testDomain]['test']
    test_dataset = dataset[testDomain]['train'] + dataset[testDomain]['test']

    for seed in seeds[1:2]:
        
        model = 'bert'
        model_path = 'bert-base-uncased'

        plm, tokenizer, model_config, WrapperClass = load_plm(model, model_path)

        #template = '{"placeholder": "text_a"} It was {"mask"}.'
        template = '{"placeholder": "text_a"} {"mask"}.'

        soft_tokens = 10
        mytemplate = SoftTemplate(model=plm, text=template, tokenizer=tokenizer,
                                  num_tokens=soft_tokens, initialize_from_vocab=True)

        label_words = [["negative"], ['positive']]
        myverbalizer = ManualVerbalizer(
            tokenizer=tokenizer, num_classes=2, label_words=label_words)
        #myverbalizer = KnowledgeableVerbalizer(tokenizer, num_classes=2).from_file(
        #    "/home/thesis/OpenPrompt/scripts/TextClassification/amazon/knowledgeable_verbalizer.txt")

        #wrapped_example = mytemplate.wrap_one_example(dataset['books']['train'][0])
        #print(wrapped_example)
        prompt_model = PromptForClassification(
            plm=plm, template=mytemplate, verbalizer=myverbalizer, freeze_plm=(not tune_plm), plm_eval_mode=False)
        prompt_model.to(device)


        batch_size = 8
        epochs = 30
        learning_rate = 1e-4
        #max_seq_length = 512-soft_tokens
        max_seq_length = 256

        run_name = 'MultiSource test on ' + testDomain
        job_type = 'amazon sp'

        config = dict(batch_size=batch_size,
                      epoch=epochs,
                      learning_rate=learning_rate,
                      soft_tokens=soft_tokens,
                      label_words=label_words,
                      max_seq_length=max_seq_length)
        config_str = str(config)
        wandb.init(project='prompt_learning', name=run_name, config=config,
                   notes=model_path, job_type=job_type, reinit=True)
        wandb.watch(prompt_model)

        trainLoader = PromptDataLoader(
            dataset=train_dataset,
            tokenizer=tokenizer,
            template=mytemplate,
            tokenizer_wrapper_class=WrapperClass,
            shuffle=True,
            batch_size=batch_size,
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


        # Now the training is standard
        loss_func = torch.nn.CrossEntropyLoss()

        if tune_plm:  # normally we freeze the model when using soft_template. However, we keep the option to tune plm
            # it's always good practice to set no decay to biase and LayerNorm parameters
            no_decay = ['bias', 'LayerNorm.weight']
            optimizer_grouped_parameters1 = [
                {'params': [p for n, p in prompt_model.plm.named_parameters() if (
                    not any(nd in n for nd in no_decay))], 'weight_decay': 0.01},
                {'params': [p for n, p in prompt_model.plm.named_parameters() if any(
                    nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
            optimizer1 = AdamW(optimizer_grouped_parameters1, lr=learning_rate)
            scheduler1 = get_linear_schedule_with_warmup(
                optimizer1,
                num_warmup_steps=500, num_training_steps=5000)
        else:
            optimizer1 = None
            scheduler1 = None

        optimizer_grouped_parameters2 = [{'params': [p for name, p in prompt_model.template.named_parameters(
        ) if 'raw_embedding' not in name]}]  # note that you have to remove the raw_embedding manually from the optimization
        #for name, p in prompt_model.template.named_parameters():
        #    print(name, p.shape)

        optimizer = AdamW(optimizer_grouped_parameters2, lr=learning_rate)  # usually lr = 0.5
        # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=250, num_training_steps=epochs*len(trainLoader))

        print('Run at train domain: {}, test domain: {}'.format(
            [domain for domain in domains if domain != testDomain], testDomain), flush=True)

        for epoch in range(epochs):
            tot_loss = 0
            prompt_model.train()
            for step, inputs in enumerate(trainLoader):
                inputs = inputs.to(device)

                logits = prompt_model(inputs)
                labels = inputs['label']

                loss = loss_func(logits, labels)
                loss.backward()
                tot_loss += loss.item()

                optimizer.step()
                #scheduler.step()
                optimizer.zero_grad()

                if step % 100 == 1:
                    print(f"Epoch {epoch}, Step {step}, average loss: {tot_loss/(step+1)}", flush=True)
                wandb.log({'train loss': tot_loss/(step+1)})

            #wandb.log({'train loss': tot_loss/len(trainLoader)})


            validPred = []
            validTrue = []
            tot_valid_loss = 0
            prompt_model.eval()
            for step, inputs in enumerate(validLoader):
                inputs = inputs.to(device)
                logits = prompt_model(inputs)
                labels = inputs['label']
                loss = loss_func(logits, labels)
                tot_valid_loss += loss.item()
                validTrue.extend(labels.cpu().tolist())
                validPred.extend(torch.argmax(logits, dim=-1).cpu().tolist())
                wandb.log({'test loss': loss.item})

            valid_loss = tot_valid_loss/len(validLoader)
            valid_acc = sum([int(i == j) for i, j in zip(
                validPred, validTrue)])/len(validPred)
            print(f"Epoch {epoch}, test loss: {valid_loss}, test acc: {valid_acc}\n", flush=True)
            wandb.log({'test loss': valid_loss, 'test acc': valid_acc})


            model_save_dir = f'/home/thesis/PromptLearning/model/{testDomain}/'
            model_save_name = f'sp_{config_str}.pt'
            save_model(prompt_model, model_save_dir, model_save_name)
        

        allpreds = []
        alllabels = []
        prompt_model.eval()
        for step, inputs in enumerate(validLoader):
            inputs = inputs.to(device)
            logits = prompt_model(inputs)
            labels = inputs['label']
            alllabels.extend(labels.cpu().tolist())
            allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
            if step % 10 == 1:
                acc = sum([int(i == j)
                            for i, j in zip(allpreds, alllabels)])/len(allpreds)
                print("acc: {}".format(acc), flush=True)
        wandb.log({'test acc': acc})
        test_acc = sum([int(i == j) for i, j in zip(allpreds, alllabels)])/len(allpreds)
        print(f"training domain: {[domain for domain in domains if domain != testDomain]}\ntestint domain: {testDomain}\nacc: {test_acc}\n", flush=True)
            
        prompt_model.to('cpu')
        del prompt_model