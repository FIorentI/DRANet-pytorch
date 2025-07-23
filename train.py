from __future__ import print_function
import torch
import wandb  # <-- NEW
from args import get_args
from trainer import Trainer

if __name__ == '__main__':
    opt = get_args()

    # Detect device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    opt.device = device

    # Initialize wandb
    wandb.init(
        project=opt.project_name if hasattr(opt, 'DRANet') else 'DRANet',
        name=opt.run_name if hasattr(opt, 'run_DRANet_pytorch') else 'run_DRANet_pytorch',
        config=vars(opt)
    )

    trainer = Trainer(opt)
    trainer.train()

    wandb.finish()
