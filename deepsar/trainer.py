import pytorch_lightning as pl
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

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
    hash_data: str = HASH
    climate_variables: list = field(default_factory=lambda: ["bio1", "pet_penman_mean", "sfcWind_mean", "bio4", "rsds_1981-2010_range_V.2.1", "bio12", "bio15"])
    layer_sizes: list = field(default_factory=lambda: symmetric_arch(6, base=32, factor=4))
    run_folder: Path = None
    cv_data_path: Path = None

    def __post_init__(self):
        root = Path(__file__).parent
        if self.cv_data_path is None:
            self.cv_data_path = (
                root
                / "../data"
                / "processed"
                / "training_samples"
                / "cv"
                / self.hash_data
            )
        if self.run_folder is None:
            self.run_folder = root / "results" / "train" / self.hash_data
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
