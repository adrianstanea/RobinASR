import logging
import os
import hydra
from hydra.core.config_store import ConfigStore

from deepspeech_pytorch.configs.inference_config import TranscribeConfig
from deepspeech_pytorch.inference import transcribe
from hydra.utils import get_original_cwd

cs = ConfigStore.instance()
cs.store(name="config", node=TranscribeConfig)


@hydra.main(config_name="config")
def hydra_main(cfg: TranscribeConfig):
    os.chdir(get_original_cwd())
    logging.info(os.getcwd())
    
    transcribe(cfg=cfg)


if __name__ == '__main__':
    hydra_main()
