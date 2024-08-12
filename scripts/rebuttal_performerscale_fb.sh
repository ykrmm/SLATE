python dgt/run.py \
 wandb_conf.name=SLATE_SEARCH_FB \
 dataset=DGB-FB \
 model=DGT_STA \
 gpu=1 \
 lr=0.001 \
 task.engine.batch_size=1024 \
 task.split=lastk \
 model.evolving=False \
 model.clip_grad=True \
 model.pred_next=False \
 model.link_pred.bias_lin_pe=False \
 model.link_pred.use_cross_attn=True\
 model.link_pred.light_ca=False \
 model.link_pred.flash=False \
 model.link_pred.nhead_ca=2 \
 model.link_pred.dropout_ca=0.1 \
 model.link_pred.bias_ca=True\
 model.link_pred.add_bias_kv=False \
 model.link_pred.add_zero_attn=True \
 model.link_pred.spatial_pe=lpe \
 model.link_pred.add_temporal_pe=False \
 model.link_pred.window=1 \
 model.link_pred.cat_te=True \
 model.link_pred.lin_te=True \
 model.link_pred.add_lin_pe=True \
 model.link_pred.dim_pe=6 \
 model.link_pred.performer=True \
 model.link_pred.dim_feedforward=1024 \
 model.link_pred.nhead=4 \
 model.link_pred.aggr=last \
 model.link_pred.temp_pe_drop=0.1 \
 model.link_pred.norm_first=True \
 model.link_pred.undirected=True \
 optim.optimizer.weight_decay=5e-7 \
 task.engine.n_runs=1 \
 task.engine.epoch=15 \


 
 