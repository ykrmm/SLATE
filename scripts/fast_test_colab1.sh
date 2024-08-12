python dgt/run.py \
 --multirun \
 wandb_conf.name=FAST_TestSearch_Colab \
 dataset=DGB-Colab \
 model=DGT_STA \
 gpu=1 \
 lr=0.0005\
 task.engine.batch_size=128 \
 task.split=lastk \
 model.evolving=True \
 model.clip_grad=True \
 model.pred_next=False \
 model.link_pred.use_cross_attn=True \
 model.link_pred.light_ca=True \
 model.link_pred.nhead_ca=1,2,4,8 \
 model.link_pred.bias_lin_pe=True \
 model.link_pred.spatial_pe=rwpe \
 model.link_pred.add_temporal_pe=True \
 model.link_pred.window=6 \
 model.link_pred.cat_te=True \
 model.link_pred.lin_te=True \
 model.link_pred.add_lin_pe=True \
 model.link_pred.dim_pe=5 \
 model.link_pred.dim_feedforward=1024 \
 model.link_pred.nhead=2 \
 model.link_pred.aggr=last \
 model.link_pred.temp_pe_drop=0.1 \
 model.link_pred.norm_first=False \
 model.link_pred.undirected=True \
 optim.optimizer.weight_decay=5e-7 \
 task.engine.n_runs=1 \
 task.engine.epoch=200 \
 