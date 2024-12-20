python slate/run.py \
 dataset=DGB-UNvote \
 model=SLATE \
 gpu=1 \
 lr=0.0001 \
 task.engine.batch_size=1024 \
 model.evolving=False \
 model.clip_grad=False \
 model.pred_next=False \
 model.link_pred.window=3 \
 model.link_pred.decision=dot \
 model.link_pred.aggr=last \
 model.link_pred.light_ca=False \
 model.link_pred.use_performer=True \
 model.link_pred.use_cross_attn=True \
 model.link_pred.nhead_ca=1 \
 model.link_pred.nhead=8 \
 model.link_pred.dropout_ca=0.1 \
 model.link_pred.bias_ca=True \
 model.link_pred.add_bias_kv=False \
 model.link_pred.add_zero_attn=True \
 model.link_pred.dropout_dec=0.1 \
 model.link_pred.dim_emb=128 \
 model.link_pred.one_hot=True \
 model.link_pred.norm_first=False \
 model.link_pred.dim_pe=12 \
 model.link_pred.add_lin_pe=True \
 model.link_pred.bias_lin_pe=False \
 model.link_pred.dim_feedforward=512 \
 model.link_pred.norm_lap=rw \
 model.link_pred.add_eig_vals=False \
 model.link_pred.add_vn=True \
 model.link_pred.remove_isolated=True \
 model.link_pred.isolated_in_transformer=True \
 model.link_pred.add_time_connection=True \
 model.link_pred.p_self_time=0.1 \
 model.link_pred.undirected=True \
 optim.optimizer.weight_decay=0 \
 task.engine.n_runs=1 \
 task.engine.epoch=5 \


python slate/run.py \
 dataset=DGB-UNvote \
 model=SLATE \
 gpu=1 \
 lr=0.0001 \
 task.engine.batch_size=1024 \
 model.evolving=False \
 model.clip_grad=False \
 model.pred_next=False \
 model.link_pred.window=3 \
 model.link_pred.decision=dot \
 model.link_pred.aggr=last \
 model.link_pred.light_ca=False \
 model.link_pred.use_performer=False \
 model.link_pred.use_cross_attn=True \
 model.link_pred.nhead_ca=1 \
 model.link_pred.nhead=8 \
 model.link_pred.dropout_ca=0.1 \
 model.link_pred.bias_ca=True \
 model.link_pred.add_bias_kv=False \
 model.link_pred.add_zero_attn=True \
 model.link_pred.dropout_dec=0.1 \
 model.link_pred.dim_emb=128 \
 model.link_pred.one_hot=True \
 model.link_pred.norm_first=False \
 model.link_pred.dim_pe=12 \
 model.link_pred.add_lin_pe=True \
 model.link_pred.bias_lin_pe=False \
 model.link_pred.dim_feedforward=512 \
 model.link_pred.norm_lap=rw \
 model.link_pred.add_eig_vals=False \
 model.link_pred.add_vn=True \
 model.link_pred.remove_isolated=True \
 model.link_pred.isolated_in_transformer=True \
 model.link_pred.add_time_connection=True \
 model.link_pred.p_self_time=0.1 \
 model.link_pred.undirected=True \
 optim.optimizer.weight_decay=0 \
 task.engine.n_runs=1 \
 task.engine.epoch=5 \
