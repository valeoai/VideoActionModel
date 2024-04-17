import os
import torch
from torch.nn import Linear, ReLU, Sequential
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPFullyShardedStrategy

class LargeModel(pl.LightningModule):
    def __init__(self, batch_size):
        super().__init__()
        self.layer = Sequential(
            Linear(1000, 10000),
            ReLU(),
            Linear(10000, 1000)
        )
        
        self.batch_size = batch_size

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.mse_loss(y_hat, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.002)

    def train_dataloader(self):
        # Dummy dataset: replace with your actual data loader
        dataset = torch.utils.data.TensorDataset(torch.randn(1024, 1000), torch.randn(1024, 1000))
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, num_workers=8, pin_memory=True)
    
    
if __name__ == "__main__":

    # Initialize the trainer with the FSDP strategy
    trainer = pl.Trainer(
        strategy='fsdp',
        gpus=-1,  # Use all GPUs available
        num_nodes=2,
        accelerator='gpu',
        precision='bf16-true'
    )
    
    # init the model directly on the device and with parameters in half-precision
    with trainer.init_module():
        model = LargeModel()
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Number of parameters: {num_params}')
    
    trainer.fit(model)
    
    trainer.print(torch.cuda.memory_summary())
