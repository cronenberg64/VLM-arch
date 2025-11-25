import torch
import torch.nn as nn
import timm
from omegaconf import DictConfig

class ModelFactory:
    @staticmethod
    def create_model(cfg: DictConfig):
        model_name = cfg.model.name
        num_classes = cfg.dataset.num_classes
        pretrained = cfg.model.pretrained
        
        # Mapping for custom names if needed, otherwise rely on timm
        # We can add custom logic here for specific backbones if timm doesn't support them perfectly
        # or if we need specific modifications for the VLM context (though this is backbone benchmarking).
        
        try:
            model = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=num_classes,
                drop_path_rate=cfg.model.get('drop_path_rate', 0.0)
            )
        except Exception as e:
            raise ValueError(f"Failed to create model {model_name}: {e}")
            
        return model

def get_model_info(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"params": params}
