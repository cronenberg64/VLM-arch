import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.datamodule import DataModule
from src.models.factory import ModelFactory
from src.engine.trainer import Trainer

@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    
    # Initialize WandB
    if not cfg.debug:
        wandb.init(
            project="vlm-arch-benchmark",
            config=OmegaConf.to_container(cfg, resolve=True),
            name=f"{cfg.model.name}_{cfg.dataset.name}_{cfg.dataset.subset or 'full'}"
        )
    
    # Data
    dm = DataModule(cfg)
    dm.setup()
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    
    # Model
    model = ModelFactory.create_model(cfg)
    
    # Trainer
    trainer = Trainer(model, train_loader, val_loader, cfg)
    trainer.fit()

if __name__ == "__main__":
    main()
