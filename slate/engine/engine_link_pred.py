from typing import Dict, List, Any
import logging
import random
import time

import numpy as np
import torch
from omegaconf import DictConfig
from hydra.utils import instantiate
from torcheval.metrics import BinaryAUROC, BinaryAUPRC
from sklearn.metrics import roc_auc_score, average_precision_score

from slate.engine import EngineBase
from slate.datasets import LinkPredData, ListLinkPredDataset 
from slate.lib import feed_dict_to_device

NoneType = type(None)
ArgsType = List[Any]
KwargsType = Dict[str, Any]


class EngineLinkPred(EngineBase):
    def __init__(
        self,
        config: DictConfig,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        dts: LinkPredData,
        logger: logging.Logger,
        *args: ArgsType,
        **kwargs: KwargsType,
    ) -> NoneType: # type: ignore
        super().__init__(*args, **kwargs)
        logger.info("EngineLinkPred initialization.")
        self.config = config
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.dts = dts
        self.device = device
        self.logger = logger
        self.log_train = self.config.task.engine.log_train
    def update_metrics(self, pred: torch.Tensor, labels: torch.LongTensor):
        with torch.no_grad():
            for _,m in self.metrics.items(): 
                m.update(pred.squeeze(),labels.squeeze())
    
    def compute_metrics(self):
        scores = []
        for m in self.metrics.values():
            scores.append(m.compute())
        return scores
    
    def reset_metrics(self):
        for m in self.metrics.values():
            m.reset()
        
    def train(self, datasets, T_train):
        # Train for one epoch 
        if self.config.model.name == 'HTGN':
            self.model.init_hiddens()

        self.model.train()
        losses = []
        t_min = 1 if self.pred_next else 0
        for t in range(t_min, T_train):
            dataset_t = datasets.get_dataset_t(t)
            graphs_t = dataset_t.get_graphs()
            dataloader_t = instantiate(
                self.config.task.engine.train_loader,
                dataset=dataset_t,
                shuffle=self.config.task.engine.shuffle_train_loader,
            )
            for feed_dict in dataloader_t:
                feed_dict = feed_dict_to_device(feed_dict, self.device)
                self.optimizer.zero_grad()
                with torch.autocast(device_type=('cuda'), dtype=torch.float16, enabled=self.use_amp):
                    loss, pos_prob, neg_prob = self.model.get_loss_link_pred(feed_dict, graphs_t)  # [B,C] B: batch size, C: number of classes           
                    losses.append(loss.item())

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                
                if self.config.model.clip_grad:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                if self.log_train:
                    pred = torch.cat((pos_prob,neg_prob))
                    labels = torch.cat((torch.ones_like(pos_prob,device=self.device), torch.zeros_like(neg_prob,device=self.device)))
                    self.update_metrics(pred,labels)
                    
        if self.log_train:
            scores = self.compute_metrics()
            scores = {name:score.item() for name,score in zip(self.metrics.keys(),scores)}
            self.reset_metrics()  
        else:
            scores = None
        mean_loss = sum(losses)/len(losses)
        return mean_loss, scores

    def eval(self,datasets: ListLinkPredDataset, t_min_max : List[int]):
        self.model.eval()
        scores = {}
        t_min,t_max = t_min_max
        all_roc = []
        all_ap = []
        for t in range(t_min,t_max):
            dataset_t = datasets.get_dataset_t(t)
            graphs_t = dataset_t.get_graphs()
            dataloader_t = instantiate(
                self.config.task.engine.train_loader,
                dataset=dataset_t,
                shuffle=False,
            )
            for feed_dict in dataloader_t:
                feed_dict = feed_dict_to_device(feed_dict,self.device)
                with torch.autocast(device_type=('cuda'), dtype=torch.float16, enabled=self.use_amp):
                    pos_prob,neg_prob = self.model.score_eval(feed_dict,graphs_t)
                pred = torch.cat((pos_prob,neg_prob))
                labels = torch.cat((torch.ones_like(pos_prob,device=self.device), torch.zeros_like(neg_prob,device=self.device)))
                if self.use_val:
                    self.update_metrics(pred,labels)
                else:
                    roc, ap = roc_auc_score(labels.cpu().numpy(),pred.cpu().numpy()),\
                        average_precision_score(labels.cpu().numpy(),pred.cpu().numpy())
                    all_roc.append(roc)
                    all_ap.append(ap)
                    
        if not self.use_val: 
            scores['AP'] = np.mean(all_ap)
            scores['ROC-AUC'] = np.mean(all_roc)
        else:       
            scores = self.compute_metrics()
            scores = {name:score.item() for name,score in zip(self.metrics.keys(),scores)}
            self.reset_metrics()     
            
        return scores

    def run(self):
        # info 
        self.logger.info("Model: {}".format(self.model.__class__.__name__))
        self.logger.info("Criterion: {}".format(self.criterion.__class__.__name__))
        self.logger.info("Dataset: {}".format(self.config.dataset.name))
        self.logger.info("Number of nodes: {}".format(self.model.num_nodes))
        self.logger.info("Total number of snapshots: {}".format(self.config.dataset.timestamp))
        self.logger.info("Split protocol : {}".format(self.config.task.split))
        self.logger.info("Edge sampling in evaluation: {}".format(self.config.task.sampling))
        self.logger.info("Evolving graph: {}".format(self.config.model.evolving))
        self.logger.info("Predicting next links: {}".format(self.config.model.pred_next))
        self.logger.info("Undirected graphs: {}".format(self.config.model.link_pred.undirected))
        self.logger.info("One hot features: {}".format(self.config.model.one_hot))
        
        # amp 
        try: 
            self.use_amp = self.model.flash
        except:
            self.use_amp = False
        
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        # Get parameters
        epoch = self.config.task.engine.epoch
        eval_every = self.config.task.engine.eval_every
        n_runs = self.config.task.engine.n_runs
        seeds = np.arange(n_runs)
        all_ap_test = []
        all_roc_auc_test = []
        self.use_val = not(self.dts.T_val == self.dts.T_train)
        
        for i in range(n_runs):
            self.logger.info("Run {}/{}".format(i+1,n_runs))
            
            # Handle Reproducibility 
            torch.manual_seed(seeds[i])
            np.random.seed(seeds[i])
            random.seed(seeds[i])
        
            #  dataset and dataloaders
            self.data_name = self.config.dataset.name
            self.logger.info("Constructing the Discrete Dynamic Graph")
            st = time.time()
            train_dts, val_dts, test_dts = self.dts.get_datasets(seeds[i])
            t_test = self.dts.T_test
            t_val = self.dts.T_val
            t_train = self.dts.T_train
            et = time.time()
            self.logger.info("Time to construct the Discrete Dynamic Graph: {:.4f} s".format(et-st))
            self.logger.info("Number of edges in train: {}".format(len(self.dts.edge_index_train[0])))
            self.logger.info("Number of edges in val: {}".format(len(self.dts.edge_index_val[0])))
            self.logger.info("Number of edges in test: {}".format(len(self.dts.edge_index_test[0])))
            self.logger.info("Constructing the dataloaders")
            
            # model
            self.pred_next = self.config.model.pred_next
            
            # metrics
            self.ap = BinaryAUPRC()
            self.rocauc = BinaryAUROC()
            self.metrics = {
                'AP': self.ap,
                'ROC-AUC':self.rocauc,
            }
            
            # model and graphs to gpu 
            self.model.to(self.device)
            try: 
                self.model.set_device(self.device) # for models that need to know the device
            except:
                pass

            # log model
            self.logger.info("Model on device: {}".format(self.device))

            # edgebank 
            if self.config.model.name == 'EdgeBank':
                tw = t_test - t_val
                self.model.construct_node_history(train_dts.get_dataset_t(t_train-1).get_graphs(),tw)
            
            # Training loop
            self.logger.info("Start training")
            val_ap = []
            train_ap = []
            test_ap = []
            test_roc = []
            max_val_ap = -1
            
            for ep in range(epoch):
                # train
                self.logger.info(f"Epoch {ep}")
                if self.config.model.name != 'EdgeBank': # EdgeBank do not need to be trained
                    et = time.time()
                    mean_loss, scores = self.train(train_dts,t_train)
                    st = time.time()
                    self.logger.info(f"Time to train one epoch: {st-et:.4f} s")
                    if self.log_train:
                        current_train_ap = scores['AP']
                        self.logger.info(f"AP Train: {current_train_ap}")
                        train_ap.append(current_train_ap)
                    self.logger.info(f"Loss: {mean_loss}")
                    #scheduler.step()

                # eval
                if ep % eval_every == 0:
                    if self.use_val:
                        # val
                        scores = self.eval(val_dts,[t_train,t_val])
                        current_val_ap = scores['AP']
                        current_val_auc = scores['ROC-AUC']
                        self.logger.info(f"AUC Val: {current_val_auc}")
                        self.logger.info(f"AP Val: {current_val_ap}")
                        val_ap.append(current_val_ap)
                    
                    # test
                    scores = self.eval(test_dts, [t_val,t_test])
                    self.logger.info(f"AP Test: {scores['AP']}")
                    self.logger.info(f"AUC Test: {scores['ROC-AUC']}")
                    test_ap.append(scores['AP'])
                    test_roc.append(scores['ROC-AUC'])

                    # early stopping
                    if self.use_val:
                        if current_val_ap > max_val_ap:
                            max_val_ap = current_val_ap
                            runs_without_improvement = 0
                        else:
                            runs_without_improvement += 1
                        if runs_without_improvement >= self.config.task.engine.early_stopping: # No early stopping for the first run
                            self.logger.info("Early stopping")
                            break
                    

            self.logger.info("Training finished")
            if self.use_val:
                self.logger.info("Best AP on val: {}".format(max(val_ap))) # 
                self.logger.info("AP on test: {}".format(test_ap[val_ap.index(max(val_ap))])) # 
                all_ap_test.append(test_ap[val_ap.index(max(val_ap))])
                all_roc_auc_test.append(test_roc[val_ap.index(max(val_ap))])
            else:
                self.logger.info("AP on test: {}".format(max(test_ap))) # 
                all_ap_test.append(max(test_ap))
                all_roc_auc_test.append(max(test_roc))

        self.logger.info(f"Maximum memory allocated: {torch.cuda.max_memory_allocated()} bytes")
        final_ap_test = np.mean(all_ap_test)
        final_roc_auc_test = np.mean(all_roc_auc_test)
        final_ap_test_std = np.std(all_ap_test)
        final_roc_auc_test_std = np.std(all_roc_auc_test)
        self.logger.info("Mean AP on test: {}".format(final_ap_test)) #  
        self.logger.info("Std AP on test: {}".format(final_ap_test_std)) # 
        self.logger.info("Mean ROC-AUC on test: {}".format(final_roc_auc_test))
        self.logger.info("Std AP on test: {}".format(final_roc_auc_test_std))
        
    
