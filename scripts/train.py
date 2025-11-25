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
    
    # DEBUG: Check CUDA availability
    import torch
    print(f"\n{'='*50}")
    print(f"CUDA DEBUG INFO:")
    print(f"Python executable: {sys.executable}")
    print(f"Torch location: {torch.__file__}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}")
    if torch.cuda.is_available():
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    print(f"{'='*50}\n")
    
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
