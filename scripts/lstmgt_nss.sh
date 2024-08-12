# enron 
python dgt/run.py \
 --multirun \
 wandb_conf.name=LSTMGT_RNS_Enron \
 dataset=DGB-Enron \
 model=LSTMGT \
 gpu=0 \
 lr=0.0001 \
 task.engine.batch_size=128 \
 task.split=lastk \
 model.evolving=False \
 model.clip_grad=True \
 model.pred_next=False \
 model.link_pred.window=5 \
 model.link_pred.spatial_pe='rwpe' \
 model.link_pred.dim_emb=128 \
 model.link_pred.dim_pe=12 \
 model.link_pred.dim_feedforward=512 \
 model.link_pred.nhead=2 \
 model.link_pred.num_layers_lstm=1 \
 model.link_pred.norm_first=False \
 model.link_pred.undirected=True \
 optim.optimizer.weight_decay=0 \
 task.engine.n_runs=3 \


# colab 
python dgt/run.py \
 --multirun \
 wandb_conf.name=LSTMGT_RNS_Colab \
 dataset=DGB-Colab \
 model=LSTMGT \
 gpu=1 \
 lr=0.0001 \
 task.engine.batch_size=256 \
 task.split=lastk \
 model.evolving=False \
 model.clip_grad=True \
 model.pred_next=False \
 model.link_pred.window=5 \
 model.link_pred.spatial_pe='rwpe' \
 model.link_pred.dim_emb=128 \
 model.link_pred.dim_pe=12 \
 model.link_pred.dim_feedforward=512 \
 model.link_pred.nhead=2 \
 model.link_pred.num_layers_lstm=1 \
 model.link_pred.norm_first=False \
 model.link_pred.undirected=True \
 optim.optimizer.weight_decay=0 \
 task.engine.n_runs=3 \

# AS733

python dgt/run.py \
 --multirun \
 wandb_conf.name=LSTMGT_RNS_AS733 \
 dataset=DGB-AS733 \
 model=LSTMGT \
 gpu=1 \
 lr=0.0001 \
 task.engine.batch_size=1024 \
 task.split=lastk \
 model.evolving=False \
 model.clip_grad=True \
 model.pred_next=False \
 model.link_pred.window=3 \
 model.link_pred.spatial_pe='rwpe' \
 model.link_pred.dim_emb=128 \
 model.link_pred.dim_pe=12 \
 model.link_pred.dim_feedforward=512 \
 model.link_pred.nhead=8 \
 model.link_pred.num_layers_lstm=1 \
 model.link_pred.norm_first=False \
 model.link_pred.undirected=True \
 optim.optimizer.weight_decay=0 \
 task.engine.n_runs=1 \

# HepPh

python dgt/run.py \
 --multirun \
 wandb_conf.name=LSTMGT_NSS_HepPh \
 dataset=DGB-HepPh \
 model=LSTMGT \
 gpu=1 \
 lr=0.0001 \
 task.engine.batch_size=1024 \
 task.split=lastk \
 model.evolving=False \
 model.clip_grad=True \
 model.pred_next=False \
 model.link_pred.window=1 \
 model.link_pred.spatial_pe='rwpe' \
 model.link_pred.dim_emb=64 \
 model.link_pred.dim_pe=12 \
 model.link_pred.dim_feedforward=512 \
 model.link_pred.nhead=8 \
 model.link_pred.num_layers_lstm=1 \
 model.link_pred.norm_first=False \
 model.link_pred.undirected=True \
 optim.optimizer.weight_decay=0 \
 task.sampling=random,historical,inductive \
 task.engine.n_runs=1 \