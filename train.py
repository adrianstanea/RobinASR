import os

import hydra
from hydra.core.config_store import ConfigStore

from deepspeech_pytorch.configs.train_config import (
    AdamConfig,
    BiDirectionalConfig,
    DeepSpeechConfig,
    FileCheckpointConfig,
    GCSCheckpointConfig,
    SGDConfig,
    UniDirectionalConfig,
)
from deepspeech_pytorch.training import train

cs = ConfigStore.instance()
cs.store(name="config", node=DeepSpeechConfig)
cs.store(group="optim", name="sgd", node=SGDConfig)
cs.store(group="optim", name="adam", node=AdamConfig)
cs.store(group="checkpointing", name="file", node=FileCheckpointConfig)
# cs.store(group="checkpointing", name="gcs", node=GCSCheckpointConfig)
cs.store(group="model", name="bidirectional", node=BiDirectionalConfig)
# cs.store(group="model", name="unidirectional", node=UniDirectionalConfig)


@hydra.main(config_name="config")
def hydra_main(cfg: DeepSpeechConfig):
    original_working_dir = hydra.utils.get_original_cwd()
    print("Original working directory:", original_working_dir)
    os.chdir(original_working_dir)

    train(cfg=cfg)


if __name__ == '__main__':
    hydra_main()
