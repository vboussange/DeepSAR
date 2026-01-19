import pytorch_lightning as pl
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataclasses import dataclass, field
from pathlib import Path
import git
from deepsar.utils import symmetric_arch

@dataclass
class TrainConfig:
    devices: list = field(default_factory=lambda: [])
    batch_size: int = 1024
    num_workers: int = 0
    n_epochs: int = 100
    lr: float = 1e-3
    lr_scheduler_factor: float = 0.5
    lr_scheduler_patience: int = 5
    weight_decay: float = 1e-4
    seed: int = 1
    hash: str = None
    climate_variables: list = field(default_factory=lambda: ["bio1", "pet_penman_mean", "sfcWind_mean", "bio4", "rsds_1981-2010_range_V.2.1", "bio12", "bio15"])
    layer_sizes: list = field(default_factory=lambda: symmetric_arch(6, base=32, factor=4))
    run_folder: Path = None
    sbcv_path: Path = None

    def __post_init__(self):
        root = Path(__file__).parent
        
        # Set hash from git if not provided
        if self.hash is None:
            try:
                repo = git.Repo(search_parent_directories=True)
                self.hash = repo.git.rev_parse(repo.head, short=True)
            except git.InvalidGitRepositoryError:
                raise ValueError("Could not determine git hash and none was provided")
        
        # if self.sbcv_path is None:
        #     self.sbcv_path = (
        #         root
        #         / "../data"
        #         / "processed"
        #         / "training_samples"
        #         / "cv"
        #         / self.hash
        #     )
        
        if self.run_folder is None:
            self.run_folder = root / ".." / "scripts"/ "results" / "train" / self.hash
            self.run_folder.mkdir(parents=True, exist_ok=True)
            
class DeepSARLitModule(pl.LightningModule):
    def __init__(self, model, config, loss_fn):
        super().__init__()
        self.model = model
        self.config = config
        self.loss_fn = loss_fn
        # self.save_hyperparameters(ignore=['model', 'loss_fn'])

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        scheduler = ReduceLROnPlateau(
            optimizer,
            factor=self.config.lr_scheduler_factor,
            patience=self.config.lr_scheduler_patience,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }


if __name__ == "__main__":
    # Test case for DeepSARLitModule
    from deepsar.mlp import FullyConnectedBatchNormBlock
    from deepsar.utils import MSELogLoss
    
    # Create a simple test model
    class SimpleTestModel(torch.nn.Module):
        def __init__(self, input_dim=10, output_dim=1):
            super().__init__()
            self.block1 = FullyConnectedBatchNormBlock(input_dim, 32)
            self.block2 = FullyConnectedBatchNormBlock(32, 16)
            self.output = torch.nn.Linear(16, output_dim)
        
        def forward(self, x):
            x = self.block1(x)
            x = self.block2(x)
            x = self.output(x)
            return x
    
    # Create test config
    config = TrainConfig(
        hash="test_hash",
        batch_size=32,
        n_epochs=2,
        lr=1e-3,
    )
    
    # Initialize model and module
    model = SimpleTestModel(input_dim=10, output_dim=1)
    loss_fn = MSELogLoss()
    lit_module = DeepSARLitModule(model, config, loss_fn)
    
    # Create dummy data
    x = torch.randn(32, 10)
    y = torch.randn(32, 1).abs() + 1  # Positive values for log loss
    
    # Test forward pass
    y_pred = lit_module(x)
    
    # Test training step
    batch = (x, y)
    loss = lit_module.training_step(batch, 0)
    
    # Test optimizer configuration
    optimizer_config = lit_module.configure_optimizers()
