python dgt/run.py \
 wandb_conf.name=Rebuttal_GCLSTMVN_Enron \
 dataset=DGB-Enron \
 model=GCLSTM_VN \
 gpu=2 \
 lr=0.0001 \
 task.engine.batch_size=128 \
 task.split=lastk \
 model.evolving=True \
 model.clip_grad=True \
 model.pred_next=False \
 model.link_pred.K=2 \
 model.link_pred.normalization=sym \
 model.link_pred.bias=False \
 model.link_pred.undirected=True \
 optim.optimizer.weight_decay=5e-7 \
 task.engine.n_runs=1 \
 task.engine.epoch=200 \