python dgt/run.py \
 --multirun \
 wandb_conf.name=FASTLink_Search_Enron \
 dataset=DGB-Enron \
 model=DGT_STA \
 gpu=0 \
 lr=0.001\
 task.engine.batch_size=128 \
 task.split=lastk \
 model.evolving=True \
 model.clip_grad=True \
 model.pred_next=False \
 model.link_pred.spatial_pe=rwpe \
 model.link_pred.add_temporal_pe=True \
 model.link_pred.use_cross_attn=True \
 model.link_pred.light_ca=False \
 model.link_pred.nhead_ca=2 \
 model.link_pred.dropout_ca=0.1 \
 model.link_pred.bias_ca=False \
 model.link_pred.add_bias_kv=True \
 model.link_pred.add_zero_attn=True \
 model.link_pred.bias_lin_pe=True \
 model.link_pred.window=5 \
 model.link_pred.cat_te=True \
 model.link_pred.lin_te=True \
 model.link_pred.add_lin_pe=True \
 model.link_pred.dim_pe=12,14,16 \
 model.link_pred.dim_feedforward=1024 \
 model.link_pred.nhead=2 \
 model.link_pred.aggr=last \
 model.link_pred.temp_pe_drop=0.1 \
 model.link_pred.norm_first=False \
 model.link_pred.undirected=True \
 optim.optimizer.weight_decay=0 \
 task.engine.n_runs=1 \
 task.engine.epoch=150 \
 