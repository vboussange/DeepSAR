import pytorch_lightning as pl
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataclasses import dataclass, field
from pathlib import Path
from muscari.utils import symmetric_arch, get_git_hash


@dataclass
class TrainConfig:
    devices: list = field(default_factory=lambda: [])
    seed: int = 1
    batch_size: int = 1024
    num_workers: int = 0
    n_epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-4
    lr_scheduler_factor: float = 0.5
    lr_scheduler_patience: int = 5
    climate_variables: list = field(default_factory=lambda: [])
    run_folder: Path = None
    path_sbcv_data: Path = None
    muscari_batchnorm: bool = False
    layer_sizes: list = field(
        default_factory=lambda: symmetric_arch(6, base=32, factor=4)
    )



class MuScaRiLitModule(pl.LightningModule):
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
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
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
    # Test case for MuScaRiLitModule
    from muscari.ffnn import FFNN
    from muscari.utils import MSELogLoss

    # Create a simple test model
    class SimpleTestModel(torch.nn.Module):
        def __init__(self, input_dim=10, output_dim=1):
            super().__init__()
            self.backbone = FFNN(input_dim=input_dim, layer_sizes=[32, 16], output_dim=output_dim, batchnorm=True)

        def forward(self, x):
            return self.backbone(x)

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
    lit_module = MuScaRiLitModule(model, config, loss_fn)

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
