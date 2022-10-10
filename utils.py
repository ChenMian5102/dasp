from openprompt.data_utils import InputExample, InputFeatures
from openprompt.plms import MLMTokenizerWrapper, load_plm
from transformers import BertTokenizer
from openprompt import PromptDataLoader
import torch
import numpy as np
from typing import *
import random
import json
from tqdm import tqdm
import os
import pickle
import sklearn
from sklearn import manifold
import matplotlib.pyplot as plt
import seaborn as sns
import wandb



def constructDataset(dataPath, domains, splits, seeds):
    dataset = {}
    #domains = ['books', 'dvd', 'electronics', 'kitchen_housewares']
    if seeds is None:
        for domain in domains:
            print(f'processing {domain}')
            dataset[domain] = {}
            for split in splits:
                #print(f'processing {split}')
                dataset[domain][split] = []
                with open(os.path.join(dataPath, domain + '.task.' + split), encoding='latin-1') as f:
                    for idx, line in enumerate(f):
                        content = line.strip().split('\t')
                        try:
                            input_example = InputExample(
                                text_a=content[1], label=int(content[0]), guid=idx)
                        except:
                            #print(idx, content)
                            continue
                        dataset[domain][split].append(input_example)

            dataset[domain]['unlabel'] = []
            with open(os.path.join(dataPath, domain + '.task.unlabel'), encoding='latin-1') as f:
                for idx, line in enumerate(f):
                    content = line.strip()
                    #input_example = InputExample(
                    #    text_a=content, label=int(1), guid=idx)
                    #dataset[domain]['unlabel'].append(input_example)
                    dataset[domain]['unlabel'].append(content)
            
    else:
        for seed in seeds:
            dataset[seed] = {}
            for domain in domains:
                dataset[seed][domain] = []
                with open(os.path.join(dataPath, str(seed), domain + '.txt'), encoding='utf-8') as f:
                    for idx, line in enumerate(f):
                        content = line.strip().split('\t')
                        input_example = InputExample(
                            text_a=content[1], label=int(content[0]), guid=idx)
                        dataset[seed][domain].append(input_example)
    return dataset

def constructAmazon(dataPath, domains):
    dataset = {}
    domains = ['books', 'dvd', 'electronics', 'kitchen_housewares']
    
    for domain in domains:
        dataset[domain] = {}
        dataset[domain]['labeled'] = []

        with open(os.path.join(dataPath, domain + 'all.txt'), encoding='latin-1') as f:
            for idx, line in enumerate(f):
                content = line.strip().split('\t')
                input_example = InputExample(
                    text_a=content[1], label=int(content[0]), guid=idx)
                dataset[domain]['unlabel'].append(input_example)

        dataset[domain]['unlabel'] = []
        with open(os.path.join(dataPath, domain + 'unl.txt'), encoding='latin-1') as f:
            for idx, line in enumerate(f):
                content = line.strip().split('\t')
                #input_example = InputExample(
                #    text_a=content, label=int(1), guid=idx)
                #dataset[domain]['unlabel'].append(input_example)
                dataset[domain]['unlabel'].append(content[1])
    return dataset

def constructMNLI(dataPath):
    dataset = {}
    domains = ['government', 'telephone', 'fiction', 'travel', 'slate']
    splits = ['train', 'test']
    for domain in domains:
        dataset[domain] = {}
        for split in splits:
            dataset[domain][split] = []
            with open(os.path.join(dataPath, domain, domain + '.' + split + '.txt'), encoding='utf-8') as f:
                for idx, line in enumerate(f):
                    content = line.strip().split('\t')
                    input_example = InputExample(
                        text_a=content[1], text_b = content[2], label=int(content[0]), guid=idx)
                    
                    dataset[domain][split].append(input_example)
                    
    return dataset

    
def zero_shot_eval(model, dataloader, device):
    allpreds = []
    alllabels = []
    model.eval()
    model.to(device)
    for step, inputs in enumerate(dataloader):
        inputs = inputs.to(device)
        logits = model(inputs)
        labels = inputs['label']
        alllabels.extend(labels.cpu().tolist())
        allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
        if step % 50 == 1:
            acc = sum([int(i == j)
                        for i, j in zip(allpreds, alllabels)])/len(allpreds)
            print("acc: {}".format(acc), flush=True)
    test_acc = sum([int(i == j)
                    for i, j in zip(allpreds, alllabels)])/len(allpreds)

    return test_acc

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
    return InputFeatures(**return_dict)

def extract_unlabel(dataPath, domain):
    extract = False
    review = ""
    reviews = []
    with open(dataPath, encoding='latin-1') as f:
        for line in f:
            line = line.strip()
            if line == "<review_text>":
                extract = True
                continue
            if line == "</review_text>":
                reviews.append(review)
                extract = False
                review = ""
                continue
            if extract:
                review += (line+" ")
    random.shuffle(reviews)
    with open(os.path.join("/home/thesis/amazon/unlabeled", domain+".txt"), 'w') as f:
        for review in reviews:
            f.write(review+"\n")
    print(domain, f": {len(reviews)}, {len(set(reviews))}")
    #for review in reviews[:5]:
    #    print(review+"\n")
        

def processMLNI(dataPath, targetPath, train=True):
    if train:
        datapath = "/home/thesis/multinli/multinli_1.0/multinli_1.0_train.jsonl"
    else:
        datapath = "/home/thesis/multinli/multinli_1.0/multinli_1.0_dev_matched.jsonl"
    targetpath = "/home/thesis/multinli/processed/"
    if not os.path.isdir(targetpath):
        os.mkdir(targetpath)
    dataDict = {}
    labelDict = {'entailment': '0', 'neutral': '1', 'contradiction': '2'}

    with open(datapath, encoding='utf-8') as f:
        json_list = list(f)
    for json_str in tqdm(json_list):
        result = json.loads(json_str)
        domain = result['genre']
        if not os.path.isdir(targetpath+domain):
            os.mkdir(targetpath+domain)
        if domain not in dataDict.keys():
            dataDict[domain] = []

        try:
            text = labelDict[result['gold_label']] + '\t' + \
                result['sentence1'] + '\t' + result['sentence2'] + '\n'
        except:
            continue

        dataDict[domain].append(text)

    if train:
        for domain in dataDict.keys():
            
            dataSize = len(dataDict[domain])
            print(domain, len(dataDict[domain]))
            downsampleSize = int(dataSize/30)
            print(domain, downsampleSize)
            dataset = random.sample(dataDict[domain], downsampleSize)
            print(domain, len(dataset))
            
            with open(os.path.join(targetpath, domain, domain+'.train.txt'), 'w') as f:
                for line in tqdm(dataset):
                    f.write(line)
            print()


def save_model(model, save_dir, model_name):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    torch.save(model, os.path.join(save_dir, model_name))

def latent_visualization(model_path, domain, prompt=True):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = torch.load(model_path, map_location=torch.device('cpu'))
    #model = torch.load(model_path)
    #model.to(device)

    plm, tokenizer, model_config, WrapperClass = load_plm(
        'bert', 'bert-base-uncased')


    dataPath = '/home/thesis/amazon/mtl-dataset/'
    domains = ['books', 'dvd', 'electronics', 'kitchen_housewares']
    splits = ['train', 'test']
    dataset = constructDataset(
        dataPath, domains, splits, seeds=None)


    domain_labels = []
    labels = []
    latents = []

    if not prompt:
        embeddings = np.load(f'embeddings/ft.npz')
        latents = embeddings['e']
        domain_labels = embeddings['d']
        labels = embeddings['l']

    else:
        if not os.path.exists(f'embeddings/{domain}/{model_name}.npz'):
            for idx, doma in enumerate(domains):
                d = dataset[doma]['train']+dataset[doma]['test']
                #d = d[:100]
                d_size = len(d)
                domain_labels.extend([idx]*d_size)

                dataloader = PromptDataLoader(
                    dataset=d,
                    tokenizer=tokenizer,
                    template=model.template,
                    tokenizer_wrapper_class=WrapperClass,
                    shuffle=False,
                    batch_size=8,
                    max_seq_length=256,
                )

                for batch in tqdm(dataloader):
                    #batch.to(device)
                    outputs = model.prompt_model(batch)
                    _, mask_idx = torch.where(batch['loss_ids']>0)

                    mask_idx += model.template.num_tokens

                    hidden_outs = outputs.hidden_states[-2]

                    mask_latent = hidden_outs[torch.arange(len(hidden_outs)), mask_idx]
                    latents.extend(mask_latent.tolist())
                    labels.extend(batch['label'].tolist())
            if not os.path.exists(f'embeddings/{domain}/'):
                os.mkdir(f'embeddings/{domain}/')
            np.savez(f'embeddings/{domain}/{model_name}.npz', e=mask_latents, d=domain_labels, l = labels)
        else:
            embeddings = np.load(f'embeddings/{domain}/{model_name}.npz')
            latents = embeddings['e']
            domain_labels = embeddings['d']
            labels = embeddings['l']
    print(latents.shape)
    print(domain_labels.shape)
    print(labels.shape)
    #exit()
    target_idx = 3
    if isinstance(domain_labels, list):
        domain_labels = np.array(domain_labels)
    if isinstance(latents, list):
        latents = np.array(latents)
    if isinstance(labels, list):
        labels = np.array(labels)

    domain_labels_2 = (domain_labels == target_idx)

    X_tsne = manifold.TSNE(n_components=2, init='random',
                           random_state=5, verbose=1, perplexity=150, n_iter=500).fit_transform(latents)
    #Data Visualization
    label_color = {0: 'red', 1: 'blue'}
    label_map = {0: 'negative', 1: 'positive'}
    domain_color = {0: 'orange', 1: 'green'}
    domain_map = {0: 'source', 1:'target'}
    domain_marker = {0: 'o', 1:'s', 2:'v', 3:'*'}

    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  # Normalize
    fig_type = ['class', 'domain', 'both']
    for t in fig_type[2:]:
        plt.figure(figsize=(8, 8))

        if t == 'both':
            for i in set(labels):
                for d in set(domain_labels):
                    if d == 0:
                        selected_idx = np.logical_and(labels == i, domain_labels == d) 
                        idx = np.where(selected_idx == True)
                        plt.scatter(X_norm[idx, 0], X_norm[idx, 1],
                                edgecolors=label_color[i], s=10, label=f'{domains[d][0]} {label_map[i]}', marker = domain_marker[d], facecolors='none')

        if t == 'class':
            for i in set(labels):
                idx = np.where(labels == i)
                plt.scatter(X_norm[idx, 0], X_norm[idx, 1],
                            c=label_color[i], s=10, label=label_map[i])
        if t == 'domain':
            for i in set(domain_labels):
                idx = np.where(domain_labels_2 == i)
                plt.scatter(X_norm[idx, 0], X_norm[idx, 1],
                            c=domain_color[i], s=10, label=domain_map[i])

        plt.legend()
        if not prompt:
            plt.savefig('fig/ft.pdf')
        else:
            if not os.path.exists(f'fig/{domain}/'):
                os.mkdir(f'fig/{domain}/')
            plt.savefig(f'fig/{domain}/{model_name}_{t}.pdf')

def soft_prompt_interpretation(model_path, domain):
    model = torch.load(model_path, map_location=torch.device('cpu'))
    #print(model)
    plm = model.plm
    sp = model.template.soft_embeds
    emb = model.template.raw_embedding.weight
    print(sp.shape)
    print(emb.shape)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    for s in sp:
        cs = []
        for w in emb:
            cs.append(torch.nn.functional.cosine_similarity(s, w, dim=0).data.numpy())
        cs = np.array(cs)
        #print(cs)
        token_id = np.argmax(cs)
        #print(token_id)
        cs = cs.argsort()[-10:][::-1]
        #print(cs)

        #print(token_id, tokenizer.decode(token_ids=[token_id]))
        print(tokenizer.decode(token_ids=cs))

def attention_visualization(model_path, domain):
    model = torch.load(model_path, map_location=torch.device('cpu'))
    plm, tokenizer, model_config, WrapperClass = load_plm(
        'bert', 'bert-base-uncased')

    raw_embedding = model.plm.get_input_embeddings()
    soft_embedding = model.template.soft_embeds
    num_soft_tokens = model.template.num_tokens
    print(num_soft_tokens)

    batch_size = 1
    #sentence = "excellent movie that shows how families should be and how most african american families really are. [MASK]"
    #sentence = "very nice , warm blanket , but too big for a full size bed [MASK]"
    sentence = "this is an great book for anyone that cooks [MASK]"
    inputs = tokenizer(sentence, return_tensors='pt')
    text_inputs = [tokenizer.decode([i]) for i in inputs['input_ids'][0]]
    print(text_inputs)
    #exit()
    mask_idx = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
    print(mask_idx)
    print(inputs)

    inputs_embeds = raw_embedding(inputs['input_ids'])
    soft_embeds = soft_embedding.repeat(batch_size, 1, 1)
    inputs_embeds = torch.cat([soft_embeds, inputs_embeds], 1)

    inputs['input_ids'] = None
    inputs['inputs_embeds'] = inputs_embeds
    am = inputs['attention_mask']
    inputs['attention_mask'] = torch.cat([torch.ones((batch_size, num_soft_tokens), dtype = am.dtype,device=am.device), am], dim=-1)
    tti = inputs['token_type_ids']
    inputs['token_type_ids'] = torch.cat([torch.zeros((batch_size, num_soft_tokens), dtype = tti.dtype,device=tti.device), tti], dim=-1)
    #print(inputs)
    

    with torch.no_grad():
        outputs = model.plm(**inputs, output_attentions = True)
    attentions = outputs.attentions 
    attentions = torch.cat(attentions).to('cpu') 
    attentions = attentions.permute(2,1,0,3) # (sequence_length, num_heads, layer, sequence_length)
    attentions_no_sp = attentions[num_soft_tokens:, :, :, num_soft_tokens:]
    print(attentions_no_sp.shape)
    print(attentions.shape)
    sp_attentions = attentions[0,:,:,num_soft_tokens:]
    print(sp_attentions.shape)
    mask_attentions = attentions_no_sp[int(mask_idx)]
    print(mask_attentions.shape)
    
    cols = 2
    rows = int(12/2)
    print ('Attention weights for [MASK] token')
    for sp in range(num_soft_tokens):
        sp_attentions = attentions[sp,:,:,num_soft_tokens:]
        avg_attention = sp_attentions.mean(dim = 0)
        fig = sns.heatmap(avg_attention,vmin = 0, vmax = 1, xticklabels= text_inputs)
        fig = fig.get_figure()
        fig.savefig(f'fig/avg_att{sp}.pdf')
        fig.clf()
        #fig, axes = plt.subplots( rows,cols, figsize = (14,30))
        #axes = axes.flat
        #for i,att in enumerate(sp_attentions):
        #    ids = [*range(att.shape[1])][1:-1]
        #    att = att[:,ids]
        #    #im = axes[i].imshow(att, cmap='gray')
        #    sns.heatmap(att, vmin = 0, vmax = 1, ax = axes[i], xticklabels = np.array(text_inputs)[ids])
        #    axes[i].set_title(f'head - {i} ' )
        #    axes[i].set_ylabel('layers')
        #plt.savefig(f'fig/att{sp}.pdf')
    #print(inputs_embeds, inputs_embeds.shape)

def mask_prediction(model_path, sentence, soft=True):
    model = torch.load(model_path, map_location=torch.device('cpu'))
    plm, tokenizer, model_config, WrapperClass = load_plm(
        'bert', 'bert-base-uncased')

    raw_embedding = model.plm.get_input_embeddings()
    soft_embedding = model.template.soft_embeds
    num_soft_tokens = model.template.num_tokens
    print(num_soft_tokens)

    if soft:
        template = '[MASK]'
        prompted_sentence = f'{sentence} {template}.'
        print(prompted_sentence)

        batch_size = 1
        inputs = tokenizer(prompted_sentence, return_tensors='pt')
        inputs_embeds = raw_embedding(inputs['input_ids'])
        soft_embeds = soft_embedding.repeat(batch_size, 1, 1)
        inputs_embeds = torch.cat([soft_embeds, inputs_embeds], 1)
        mask_idx = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
        mask_idx += num_soft_tokens
        print(mask_idx)

        inputs['input_ids'] = None
        inputs['inputs_embeds'] = inputs_embeds
        am = inputs['attention_mask']
        inputs['attention_mask'] = torch.cat([torch.ones((batch_size, num_soft_tokens), dtype = am.dtype,device=am.device), am], dim=-1)
        tti = inputs['token_type_ids']
        inputs['token_type_ids'] = torch.cat([torch.zeros((batch_size, num_soft_tokens), dtype = tti.dtype,device=tti.device), tti], dim=-1)
    else:
        template = 'This review is [MASK]'
        prompted_sentence = f'{sentence} {template}.'
        print(prompted_sentence)
        inputs = tokenizer(prompted_sentence, return_tensors='pt')
        mask_idx = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]

    with torch.no_grad():
        logits = model.plm(**inputs).logits
    
    print(logits.shape)

    #predicted_token_id = logits[0, mask_idx].argmax(axis=-1)
    print(logits[0, mask_idx].shape)
    predicted_token_id = logits[0, mask_idx].argmax(axis=-1)
    print(predicted_token_id)
    predicted_token_id = torch.topk(logits[0, mask_idx].flatten(), 6).indices
    print(predicted_token_id)
    print(tokenizer.decode(predicted_token_id))

if __name__ == '__main__':
    domains = ['books', 'dvd', 'electronics', 'kitchen_housewares']
    domain = 'books'
    #model_name = "sp_{'batch_size': 8, 'epoch': 20, 'learning_rate': 0.0001, 'soft_tokens': 2, 'label_words': [['negative'], ['positive']], 'max_seq_length': 256}.pt"
    #model_name = "meta_sp_0.863_t5.pt"
    model_name = "meta_sp_0.8895_t5.pt"
    #soft_prompt_interpretation(os.path.join('model', domain, model_name), domain)
    #latent_visualization(os.path.join('model', domain, model_name), domain, prompt=True)
    #attention_visualization(os.path.join('model', domain, model_name), domain)
    sentence = "an excellent book for anyone that barbecues"
    #sentence = "the author really just could not hook me . a lot about food but not sure what else"
    #sentence = "this album contains only rap and no rock songs . this was very disappointing to say the least"
    #sentence = "very comfortable and sexy . hanky panky is the standard in comfortable lace"
    #sentence = "i am a huge fan of btnh , but even i will admitt this is not a good album ."
    #sentence = "i still have not received this magazine , what is taking so long ! !"
    #sentence = "sexy and well made . i am a size 12 , but i bought the queen size and it fit great"
    mask_prediction(os.path.join('model', domain, model_name), sentence, soft=False)
    

