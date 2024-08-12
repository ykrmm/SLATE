python dgt/run.py \
 wandb_conf.name=DGT_STA_Best_Rnd_Can \
 dataset=DGB-CanParl \
 model=DGT_STA \
 gpu=1 \
 lr=0.01 \
 task.engine.batch_size=1024 \
 model.evolving=False \
 model.clip_grad=True \
 model.pred_next=False \
 model.link_pred.bias_lin_pe=False \
 model.link_pred.spatial_pe=rwpe \
 model.link_pred.add_temporal_pe=False \
 model.link_pred.window=3 \
 model.link_pred.cat_te=True \
 model.link_pred.lin_te=True \
 model.link_pred.add_lin_pe=True \
 model.link_pred.dim_pe=12 \
 model.link_pred.performer=True \
 model.link_pred.dim_feedforward=1024 \
 model.link_pred.nhead=8 \
 model.link_pred.aggr=last \
 model.link_pred.temp_pe_drop=0.1 \
 model.link_pred.norm_first=True \
 model.link_pred.undirected=True \
 optim.optimizer.weight_decay=0 \
 task.engine.n_runs=1 \


 
 