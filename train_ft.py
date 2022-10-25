from utils import *
from transformers import *
from datasets import load_metric
from torch.utils.data import DataLoader
from torch.optim import AdamW
import numpy as np
import torch
import random
import os
from tqdm.auto import tqdm
import wandb

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
print(device)

dataPath = '/home/thesis/amazon/mtl-dataset/'
splits = ['train', 'test']
domains = ['books', 'dvd', 'electronics', 'kitchen_housewares']
seeds = [7, 10, 21, 26, 85]

trainDomain = 'books'
testDomain = 'dvd'

def read_amazon_split(dataPath, domain, split, seed):
    texts = []
    labels = []
    if seed == None:
        with open(os.path.join(dataPath, domain + '.task.' + split), encoding='latin-1') as f:
            for idx, line in enumerate(f):
                content = line.strip().split('\t')
                texts.append(content[1])
                labels.append(int(content[0]))
    else:
        with open('/home/thesis/amazon/16-shot/' + str(seed) + '/' + domain + '.txt') as f:
            for idx, line in enumerate(f):
                content = line.strip().split('\t')
                texts.append(content[1])
                labels.append(int(content[0]))

    return texts, labels

class AmazonDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx])
                for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


#def compute_metrics(eval_pred):
#    logits, labels = eval_pred
#    predictions = np.argmax(logits, axis=-1)
#    return metric.compute(predictions=predictions, references=labels)


for testDomain in domains[:1]:
    # testDomain = 'kitchen_housewares'
    

    train_texts = []
    train_labels= []
    src_domain_labels = []
    for trainDomain in domains:
        if trainDomain == testDomain:
            continue
        train_text, train_label = read_amazon_split(dataPath, trainDomain, 'train', seed = None)
        dl = [domains.index(trainDomain)]*len(train_text)
        
        src_domain_labels.extend(dl)
        train_texts.extend(train_text)
        train_labels.extend(train_label)

        train_text, train_label = read_amazon_split(dataPath, trainDomain, 'test', seed = None)
        dl = [domains.index(trainDomain)]*len(train_text)
        src_domain_labels.extend(dl)
        train_texts.extend(train_text)
        train_labels.extend(train_label)
        
        #exit()

    for seed in seeds[:1]:

        test_texts, test_labels = read_amazon_split(dataPath, testDomain, 'train', seed = None)
        t_t, t_l = read_amazon_split(dataPath, testDomain, 'test', seed = None)
        test_texts.extend(t_t)
        test_labels.extend(t_l)

        ####
        domain_labels = [domains.index(testDomain)]*len(test_texts)
        domain_labels.extend(src_domain_labels)
        test_texts.extend(train_texts)
        test_labels.extend(train_labels)
        ####

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        max_seq_length = 256
        batch_size = 8
        epochs = 25
        learning_rate = 1e-4
        max_seq_length = 256

        train_encodings = tokenizer(
            train_texts, truncation=True, padding=True, max_length=max_seq_length)
        # valid_encodings = tokenizer(valid_texts, truncation=True,
        #                         padding=True, max_length=256)
        test_encodings = tokenizer(test_texts, truncation=True,
                                   padding=True, max_length=max_seq_length)


        train_dataset = AmazonDataset(train_encodings, train_labels)
        test_dataset = AmazonDataset(test_encodings, test_labels)

        train_dataloader = DataLoader(
            train_dataset, shuffle=True, batch_size=batch_size)
        eval_dataloader = DataLoader(test_dataset, batch_size=batch_size)
        
        
        model_type = 'bert-base-uncased'
        # run_name = trainDomain + ' to ' + testDomain
        run_name = 'Multisource test on ' + testDomain
        job_type = 'amazon ft'

        config = dict(batch_size=batch_size,
                      epoch=epochs,
                      learning_rate=learning_rate,
                      max_seq_length=max_seq_length)
        config_str = str(config)
        #print(config_str)
        #wandb.init(project='prompt_learning', name=run_name, config=config,
        #           notes=model_type, job_type=job_type, reinit=True)


        model = BertForSequenceClassification.from_pretrained(
            model_type, num_labels=2)
        model.to(device)
        #wandb.watch(model)
        #print(model)
        #exit()
        
        optimizer_grouped_parameters = [{'params': [p for n, p in model.named_parameters() if 'classifier' in n]}]
        optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
        #optimizer = AdamW(model.parameters(), lr=learning_rate)

        num_training_steps = epochs * len(train_dataloader)
        progress_bar = tqdm(range(num_training_steps))

        ## Save latents
        model.eval()
        latents = []
        for batch in tqdm(eval_dataloader):
            #print(batch)
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            hidden_outs = hidden_states[-2]
            latent = hidden_outs[torch.arange(len(hidden_outs)), 0]
            latents.extend(latent.tolist())
            #print(latent[0])
            #exit()
            #print(len(latent))
            #exit()
        latents = np.array(latents)
        print(latents.shape)
        np.savez(f'embeddings/ft.npz', e=latents, d=domain_labels, l = test_labels)
        exit()


        model.train()
        train_tot_loss = 0
        
        global_step = 0
        for epoch in range(epochs):
            train_tot_loss = 0
            for step, batch in enumerate(train_dataloader):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch, output_hidden_states=True)
                
                loss = outputs.loss
                train_tot_loss += loss.item()
                loss.backward()

                optimizer.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                
                #if global_step % 50 == 0:
            wandb.log({'train loss': train_tot_loss/(step+1)})

            metric = load_metric("accuracy")
            model.eval()
            test_tot_loss = 0
            test_tot_acc = 0
            for step, batch in enumerate(eval_dataloader):
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.no_grad():
                    outputs = model(**batch)
                loss = outputs.loss
                test_tot_loss += loss.item()
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                metric.add_batch(predictions=predictions, references=batch["labels"])
                test_tot_acc += metric.compute()['accuracy']
            test_acc = test_tot_acc / len(eval_dataloader)
            test_loss = test_tot_loss / len(eval_dataloader)
            wandb.log({'test loss': test_loss, 'test acc': test_acc})

                    #acc = metric.compute()['accuracy']
                #wandb.log({'test acc': test_acc})
                #global_step += 1

            print("training domain: {}\ntestint domain: {}\nacc: {}\n".format(
                [trainDomain for trainDomain in domains if trainDomain != testDomain], testDomain, test_acc))

        model_save_dir = f'/home/thesis/PromptLearning/model/{testDomain}/'
        model_save_name = f'ft_{config_str}.pt'
        save_model(model, model_save_dir, model_save_name)
