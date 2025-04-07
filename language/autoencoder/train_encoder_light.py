import os
import torch
import torch.nn.functional as F
import lightning.pytorch as pl
from model import AutoencoderLight
import torch.nn as nn
from lightning.pytorch.strategies.ddp import DDPStrategy
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import ModelCheckpoint
import lightning as L
from argparse import ArgumentParser
from dataset import Autoencoder_dataset    

def main(args):
    torch.multiprocessing.set_start_method('spawn', force=True)
    
    dataset_path = args.dataset_paths
    val_dataset_path = args.val_dataset_paths
    train_dataset = Autoencoder_dataset(dataset_path) 
    test_dataset = Autoencoder_dataset(val_dataset_path)
    
    
    model_light = AutoencoderLight(args.encoder_dims, args.decoder_dims, args.in_channel_dim, is_MLP=True)
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=args.output_dir)
    checkpoint_callback = ModelCheckpoint(monitor='val_loss/total_loss')
    
    trainer = L.Trainer(
        max_epochs=args.num_epochs, 
        log_every_n_steps=2,
        logger=tb_logger,
        callbacks=[checkpoint_callback],
        accelerator="gpu",
        devices=[0,1]
        #accelerator="gpu", devices=args.device,
        #strategy=DDPStrategy(find_unused_parameters=False)
    )
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1
    )
    
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1
    )
        
    trainer.fit(model_light, train_dataloader, test_dataloader, ckpt_path=args.ckpt_path)

if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")

    # Modify the argument parser to accept multiple dataset paths for training
    # parser.add_argument('--dataset_paths', nargs='+', type=str, default=["office0/highres_feat",
    #                                                                  "room0/highres_feat",
    #                                                                  "office1/highres_feat",
    #                                                                  "room1/highres_feat",
    #                                                                  "office2/highres_feat",
    #                                                                  "room2/highres_feat"])

    parser.add_argument('--dataset_paths', nargs='+', type=str, default=["low_res_lang"])
    
    parser.add_argument('--val_dataset_paths', nargs='+', type=str, default=["low_res_lang"])
    
    # parser.add_argument('--val_dataset_paths', nargs='+', type=str, default=["office3/highres_feat",
    #                                                                          "office4/highres_feat"])

    parser.add_argument("--device", type=str, default="0,1")
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument('--num_epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=15)
    parser.add_argument('--encoder_dims',
                    nargs = '+',
                    type=int,
                    default=[384, 192, 96, 48, 24, 15],
                    )
    parser.add_argument('--decoder_dims',
                    nargs = '+',
                    type=int,
                    default=[24, 48, 96, 192, 384, 384, 768],
                    )
    parser.add_argument('--in_channel_dim', type=int, default=768)
    parser.add_argument('--output_dir', type=str, default="code15")
    
    args = parser.parse_args()
    
    main(args)