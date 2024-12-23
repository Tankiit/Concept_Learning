from dataclasses import dataclass,field
from typing import List, Optional
import torch

@dataclass
class ModelConfig:
    # Model Architecture
    input_channels: int = 1  # MNIST has 1 channel
    img_size: int = 28      # MNIST image size
    input_dim: int = 784  # 28x28 images
    zy_dim: int = 64     # invariant latent dim
    zd_dim: int = 32     # variant latent dim
    concept_dim: int = 768  # BERT embedding dimension
    n_classes: int = 10   # number of digits
    n_domains: int = 4    # number of rotations
    hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])
    concept_type: str = 'simple'

@dataclass
class TrainingConfig:
    # Basic training parameters
    batch_size: int = 32
    epochs: int = 10
    lr: float = 0.001
    weight_decay: float = 1e-5
    
    # Loss weights - adjusted based on observed metrics
    recon_weight: float = 1.0      # Keep reconstruction loss as base
    kl_weight: float = 0.1         # Reduce KL weight as it's too high (1.4-1.5)
    digit_weight: float = 2.0      # Increase digit classification weight
    domain_weight: float = 1.0     # Moderate domain classification weight
    concept_sep_weight: float = 0.2 # Slightly increase concept separation weight
    
    # Annealing factors for KL divergence
    kl_anneal_start: float = 0.0
    kl_anneal_end: float = 0.1
    kl_anneal_epochs: int = 5
    
    # Training settings
    seed: int = 42
    device: str = 'cpu'
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    
    # Checkpointing and monitoring
    checkpoint_dir: str = 'checkpoints'
    cache_dir: str = './cache'
    save_freq: int = 5
    patience: int = 10
    save_plots: bool = True
    generate_explanations: bool = True
    
    # Learning Rate Schedule
    scheduler_type: str = 'step'
    step_size: int = 10
    gamma: float = 0.1
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    min_delta: float = 1e-4


def get_kl_weight(config, epoch):
    """Get annealed KL weight"""
    if epoch >= config.training.kl_anneal_epochs:
        return config.training.kl_weight
    return config.training.kl_anneal_start + (config.training.kl_anneal_end - config.training.kl_anneal_start) * \
           (epoch / config.training.kl_anneal_epochs)


@dataclass
class DataConfig:
    root_dir: str = './data'
    angles: List[int] = field(default_factory=lambda: [0, 90, 180, 270])
    concept_type: str = 'simple'
    mean: float = 0.1307
    std: float = 0.3081
    augmentation: bool = True
    cache_concepts: bool = True
    concept_batch_size: int = 100
    precompute_concepts: bool = True
    concept_batch_size: int = 100

@dataclass
class WandBConfig:
    project: str = "rotated-mnist-concepts"
    name: str = "concept-learning"
    tags: List[str] = field(default_factory=list)
    notes: str = "Experiment with concept learning on rotated MNIST"

@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    wandb: WandBConfig = field(default_factory=WandBConfig)

    def to_dict(self):
        """Convert config to dictionary for wandb"""
        return {
            'model': vars(self.model),
            'training': vars(self.training),
            'data': vars(self.data),
            'wandb': vars(self.wandb)
        }


def get_config():
    """Get default configuration"""
    return Config() 