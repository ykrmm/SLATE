python dgt/run.py \
 --multirun \
 wandb_conf.name=FASTLink_Search_Vote \
 dataset=DGB-UNvote \
 model=DGT_STA \
 gpu=2 \
 lr=0.0001 \
 task.engine.batch_size=1024 \
 model.evolving=False \
 model.clip_grad=False \
 model.pred_next=False \
 model.link_pred.bias_lin_pe=False \
 model.link_pred.spatial_pe=rwpe \
 model.link_pred.add_temporal_pe=True \
 model.link_pred.use_cross_attn=True\
 model.link_pred.light_ca=False \
 model.link_pred.nhead_ca=1,2,4,8 \
 model.link_pred.dropout_ca=0,0.1 \
 model.link_pred.bias_ca=True,False \
 model.link_pred.add_bias_kv=False,True \
 model.link_pred.add_zero_attn=False,True \
 model.link_pred.window=2,3 \
 model.link_pred.cat_te=True \
 model.link_pred.lin_te=True \
 model.link_pred.add_lin_pe=True \
 model.link_pred.dim_pe=12 \
 model.link_pred.dim_feedforward=512 \
 model.link_pred.nhead=8 \
 model.link_pred.aggr=last \
 model.link_pred.temp_pe_drop=0.1 \
 model.link_pred.norm_first=False \
 model.link_pred.undirected=True \
 optim.optimizer.weight_decay=0 \
 task.engine.n_runs=1 \
 