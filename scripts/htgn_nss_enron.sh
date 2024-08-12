python dgt/run.py \
 --multirun \
 wandb_conf.name=HTGN_NSS_Enron \
 dataset=DGB-Enron \
 model=HTGN \
 gpu=0 \
 lr=0.001\
 task.engine.batch_size=64 \
 model.evolving=False \
 model.clip_grad=True \
 model.pred_next=False \
 model.link_pred.nhid=16 \
 model.link_pred.dropout=0.0 \
 model.link_pred.curvature=1.0 \
 model.link_pred.nout=16 \
 model.link_pred.fixed_curvature=1 \
 model.link_pred.nfeat=128 \
 model.link_pred.use_hta=1 \
 model.link_pred.use_gru=True \
 model.link_pred.model_type=HTGN \
 model.link_pred.aggregation='deg' \
 model.link_pred.heads=1 \
 model.link_pred.window=5 \
 model.link_pred.device=cuda:0 \
 optim.optimizer.weight_decay=0 \
 task.sampling=historical,inductive \
 task.engine.n_runs=5 \