import hydra
from omegaconf import DictConfig
from omegaconf import OmegaConf
import torch
import numpy as np
from slate.getter import Getter
import slate.lib as lib

logger = lib.LOGGER
OmegaConf.register_new_resolver("mult", lambda *numbers: float(np.prod([float(x) for x in numbers])))
OmegaConf.register_new_resolver("sum", lambda *numbers: sum(map(float, numbers)))
OmegaConf.register_new_resolver("sub", lambda x, y: float(x) - float(y))
OmegaConf.register_new_resolver("div", lambda x, y: float(x) / float(y))

@hydra.main(config_path="config", config_name="default",version_base="1.3")
def run(config: DictConfig):
    getter = Getter(config)
    torch.set_printoptions(sci_mode=False) # Disable scientific notation
    device = torch.device("cuda:"+str(config.gpu) if torch.cuda.is_available() else "cpu")
    
    
    # """""""""""" Create dataset """"""""""""""
    dataset = getter.get_dataset()
    
    
    # """""""""""" Create model """"""""""""""
    model = getter.get_model()
    model = model[config.task_name] # Get the right model for the task
    # """""""""""" Create loss """"""""""""""
    criterion = getter.get_loss()

    # """""""""""" Create optimizer """"""""""""""
    optimizer = getter.get_optimizer(model)

    # """""""""""" Create engine """"""""""""""
    
    if config.task_name == 'node_reg':
        from slate.engine import EngineReg as Engine
        logger.info("Node Regression task")
        dts = dataset[config.task_name]
        
        
    elif config.task_name == 'link_pred':
        from slate.engine import EngineLinkPred as Engine
        logger.info("Link Prediction task")
        dts = dataset[config.task_name]
    else:
        logger.error(config.task_name+" not implemented")
        
    # """""""""""" Run engine """"""""""""""
    engine_task = Engine(config,
                        model,
                        criterion,
                        optimizer,
                        device,
                        dts,
                        logger)

    return engine_task.run()



if __name__ == "__main__":

    run()