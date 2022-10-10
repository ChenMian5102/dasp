from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from copy import deepcopy
import gc
from sklearn.metrics import accuracy_score
import torch
import numpy as np
import wandb



class Reptile(nn.Module):
    """
    Meta Learner
    """

    def __init__(self, model, outer_update_lr, inner_update_lr, inner_update_step, tune_plm, device):
        """
        :param args:
        """
        super(Reptile, self).__init__()

        self.model = model
        self.tune_plm = tune_plm
        self.outer_update_lr = outer_update_lr
        self.inner_update_lr = inner_update_lr
        self.inner_update_step = inner_update_step
        self.device = device
        self.loss_func = torch.nn.CrossEntropyLoss()

        if self.tune_plm:
            outer_grouped_param = [
                {'params': [p for name, p in self.model.plm.named_parameters()]},
                {'params': [p for name, p in self.model.template.named_parameters(
            ) if 'raw_embedding' not in name]}]
            self.outer_optimizer = AdamW(outer_grouped_param, lr=self.outer_update_lr)
        else:
            outer_grouped_param = [{'params': [p for name, p in self.model.template.named_parameters(
            ) if 'raw_embedding' not in name]}]
            self.outer_optimizer = AdamW(outer_grouped_param, lr=self.outer_update_lr)
        self.model.train()

    def forward(self, batch_tasks, training=True, zero_shot = False):
        """
        batch = [(support , query),
                 (support , query),
                 (support , query),
                 (support , query)]
        
        # support = InputFeatures
        """
        task_loss = []
        task_acc = []
        sum_gradients = []
        num_tasks = len(batch_tasks)

        for task_id, task in enumerate(batch_tasks):
            support = task[0]
            query = task[1]

            fast_model = deepcopy(self.model)
            fast_model.to(self.device)

            if self.tune_plm:
                inner_grouped_param  = [
                {'params': [p for name, p in fast_model.plm.named_parameters()]},
                {'params': [p for name, p in fast_model.template.named_parameters(
                ) if 'raw_embedding' not in name]}]
                inner_optimizer = AdamW(inner_grouped_param, lr = self.inner_update_lr)
            else:
                inner_grouped_param  = [{'params': [p for name, p in fast_model.template.named_parameters(
                ) if 'raw_embedding' not in name]}]
                inner_optimizer = AdamW(inner_grouped_param, lr = self.inner_update_lr)
            loss_func = torch.nn.CrossEntropyLoss()

            fast_model.train()
            # support = support.to(self.device)
            

            # print(f'------------Task {task_id}------------')
            inner_loss = 0
            for i in range(self.inner_update_step):
                sup = deepcopy(support)
                sup = sup.to(self.device)
                logits = fast_model(sup)
                labels = sup['label']

                loss = loss_func(logits, labels)
                loss.backward()
                inner_optimizer.step()
                inner_optimizer.zero_grad()

                del sup
                inner_loss += loss.item()
                # print(f'Inner Loss: {loss.item()}')
            task_loss.append(inner_loss/self.inner_update_step)
            
            fast_model.to(torch.device('cpu'))
            self.model.to(torch.device('cpu'))

            if training:
                if self.tune_plm:
                    meta_weights = list(self.model.plm.parameters()) + [p for name, p in self.model.template.named_parameters(
                    ) if 'raw_embedding' not in name]
                    fast_weights = list(fast_model.plm.parameters()) + [p for name, p in fast_model.template.named_parameters(
                    ) if 'raw_embedding' not in name]
                else:
                    meta_weights = [p for name, p in self.model.template.named_parameters(
                    ) if 'raw_embedding' not in name]
                    fast_weights = [p for name, p in fast_model.template.named_parameters(
                    ) if 'raw_embedding' not in name]

                gradients = []
                for i, (meta_params, fast_params) in enumerate(zip(meta_weights, fast_weights)):
                    gradient = meta_params - fast_params
                    if task_id == 0:
                        sum_gradients.append(gradient)
                    else:
                        sum_gradients[i] += gradient

            fast_model.to(self.device)
            fast_model.eval()
            with torch.no_grad():
                query = query.to(self.device)
                
                logits = fast_model(query)
                labels = query['label']

                labels = labels.cpu().tolist()
                pred = torch.argmax(logits, dim=-1).cpu().tolist()
                acc = sum([int(i == j) for i, j in zip(
                    pred, labels)])/len(pred)
                task_acc.append(acc)

            fast_model.to(torch.device('cpu'))
            del fast_model, inner_optimizer
            torch.cuda.empty_cache()

        if training:
            # Average gradient across tasks
            for i in range(0, len(sum_gradients)):
                sum_gradients[i] = sum_gradients[i] / float(num_tasks)

            #Assign gradient for original model, then using optimizer to update its weights
            # self.model.to(torch.device('cpu'))
            # self.model.to(self.device)
            if self.tune_plm:
                meta_params = list(self.model.plm.parameters()) + [p for name, p in self.model.template.named_parameters(
                    ) if 'raw_embedding' not in name]
            else:
                meta_params = [p for name, p in self.model.template.named_parameters(
                ) if 'raw_embedding' not in name]

            for i, params in enumerate(meta_params):
                # print(params)
                # print(params.grad)
                # print(sum_gradients[i])
                params.grad = sum_gradients[i]

            self.outer_optimizer.step()
            self.outer_optimizer.zero_grad()

            del sum_gradients
            gc.collect()

        return np.mean(task_loss), np.mean(task_acc)

    def zero_shot_eval(self, dataloader):
        allpreds = []
        alllabels = []
        totalloss = 0
        self.model.eval()
        self.model.to(self.device)
        for step, inputs in enumerate(dataloader):
            inputs = inputs.to(self.device)
            logits = self.model(inputs)
            labels = inputs['label']
            loss = self.loss_func(logits, labels)
            totalloss += loss.item()
            alllabels.extend(labels.cpu().tolist())
            allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
            if step % 50 == 1:
                acc = sum([int(i == j)
                           for i, j in zip(allpreds, alllabels)])/len(allpreds)
                print("acc: {}".format(acc), flush=True)
        test_acc = sum([int(i == j) for i, j in zip(allpreds, alllabels)])/len(allpreds)
        test_loss = totalloss / len(dataloader)

        return test_acc, test_loss

class MAML(nn.Module):
    """
    Meta Learner
    """

    def __init__(self, model, outer_update_lr, inner_update_lr, inner_update_step, tune_plm, device):
        """
        :param args:
        """
        super(MAML, self).__init__()

        self.model = model
        self.tune_plm = tune_plm
        self.outer_update_lr = outer_update_lr
        self.inner_update_lr = inner_update_lr
        self.inner_update_step = inner_update_step
        self.device = device
        self.loss_func = torch.nn.CrossEntropyLoss()

        outer_grouped_param = [{'params': [p for name, p in self.model.template.named_parameters(
        ) if 'raw_embedding' not in name]}]
        self.outer_optimizer = AdamW(outer_grouped_param, lr=self.outer_update_lr)
        #self.outer_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=250, num_training_steps=)
        self.model.train()

    def forward(self, batch_tasks, training=True, zero_shot = False):
        """
        batch = [(support , query),
                 (support , query),
                 (support , query),
                 (support , query)]
        
        # support = InputFeatures
        """
        task_loss = []
        task_acc = []
        sum_gradients = []
        num_tasks = len(batch_tasks)

        for task_id, task in enumerate(batch_tasks):
            support = task[0]
            query = task[1]

            fast_model = deepcopy(self.model)
            fast_model.to(self.device)
            
            inner_grouped_param  = [{'params': [p for name, p in fast_model.template.named_parameters(
            ) if 'raw_embedding' not in name]}]
            inner_optimizer = AdamW(inner_grouped_param, lr = self.inner_update_lr)
            loss_func = torch.nn.CrossEntropyLoss()

            fast_model.train()
            # support = support.to(self.device)
            

            # print(f'------------Task {task_id}------------')
            inner_loss = 0
            for i in range(self.inner_update_step):
                sup = deepcopy(support)
                sup = sup.to(self.device)
                logits = fast_model(sup)
                labels = sup['label']

                loss = loss_func(logits, labels)
                loss.backward()
                inner_optimizer.step()
                inner_optimizer.zero_grad()

                del sup
                inner_loss += loss.item()
                # print(f'Inner Loss: {loss.item()}')
            task_loss.append(inner_loss/self.inner_update_step)
            
            query = query.to(self.device)
            logits = fast_model(query)
            labels = query['label']
            loss = self.loss_func(logits, labels)

            if training:
                loss.backward()
                fast_model.to(torch.device('cpu'))
                for i, params in enumerate([p for name, p in fast_model.template.named_parameters(
            ) if 'raw_embedding' not in name]):
                    if task_id == 0:
                        sum_gradients.append(deepcopy(params.grad))
                    else:
                        sum_gradients[i] += deepcopy(params.grad)


            labels = labels.cpu().tolist()
            pred = torch.argmax(logits, dim=-1).cpu().tolist()
            acc = sum([int(i == j) for i, j in zip(
                pred, labels)])/len(pred)
            task_acc.append(acc)

            fast_model.to(torch.device('cpu'))
            del fast_model, inner_optimizer
            torch.cuda.empty_cache()

        if training:
            # Average gradient across tasks
            for i in range(0, len(sum_gradients)):
                sum_gradients[i] = sum_gradients[i] / float(num_tasks)

            #Assign gradient for original model, then using optimizer to update its weights
            # self.model.to(self.device)
            self.model.to(torch.device('cpu'))
            meta_params = [p for name, p in self.model.template.named_parameters(
            ) if 'raw_embedding' not in name]
            for i, params in enumerate(meta_params):
                params.grad = sum_gradients[i]

            self.outer_optimizer.step()
            self.outer_optimizer.zero_grad()

            del sum_gradients
            gc.collect()

        return np.mean(task_loss), np.mean(task_acc)

    def zero_shot_eval(self, dataloader):
        allpreds = []
        alllabels = []
        totalloss = 0
        self.model.eval()
        self.model.to(self.device)
        for step, inputs in enumerate(dataloader):
            inputs = inputs.to(self.device)
            logits = self.model(inputs)
            labels = inputs['label']
            loss = self.loss_func(logits, labels)
            totalloss += loss.item()
            alllabels.extend(labels.cpu().tolist())
            allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
            if step % 50 == 1:
                acc = sum([int(i == j)
                           for i, j in zip(allpreds, alllabels)])/len(allpreds)
                print("acc: {}".format(acc), flush=True)
        test_acc = sum([int(i == j) for i, j in zip(allpreds, alllabels)])/len(allpreds)
        test_loss = totalloss / len(dataloader)

        return test_acc, test_loss
