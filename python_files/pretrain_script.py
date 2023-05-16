#!/usr/bin/env python
# *** coding: utf-8 ***

"""train.py: Script to pretrain a ViT model.

   * Author: Delanoe PIRARD
   * Email: delanoe.pirard.pro@gmail.com
   * Licence: MIT

   * Usage: pretrain_script.py \
            --adam_weight_decay=0.03 \
            --batch_size=4096 \
            --checkpoint_iterations=1000 \
            --dataset=./data/ImageNet-21k/images \
            --dropout_value=0.1 \
            --embedding_dim=768 \
            --embedding_dropout_value=0.1 \
            --in_channels=3 \
            --learning_rate=0.001 \
            --load_checkpoint=./your/checkpoint.pt \
            --load_model=./your/model.pt \
            --lr_decay_patience=25 \
            --max_batch_iterations=10000000 \
            --mlp_size=3072 \
            --num_heads=12 \
            --num_transformer_layers=12 \
            --patch_size=16 \
            --resize_img=224 \
            --save_path=./path/to/save/ \
            --warmup_iterations=10000
"""

# IMPORTS -------------------------------------------------
import torch

import utils
import modules
import logging
import matplotlib.pyplot as plt

from torchvision import transforms
from torch.optim import Adam
from torch.nn import MSELoss
from torchinfo import summary
from flags import pretrain_flags
# from pathlib import Path


# FUNCTIONS -----------------------------------------------
def main():
    args = pretrain_flags()
    if args.dataset is None:
        logging.error("Need to set up a dataset with `--dataset`.")

    # Create dataloader
    logging.info("Loading dataset...")
    train_dataloader = utils.create_pretrain_dataloaders(
        train_dir=args.dataset,
        patch_size=args.patch_size,
        transform=transforms.Compose([
            transforms.Resize((args.resize_img, args.resize_img)),
            transforms.ToTensor()
        ]),
        batch_size=args.batch_size,
        num_workers=2
    )
    logging.info("Dataset loaded!")

    logging.info("Initializing model...")

    if args.load_model is not None:
        model = torch.load(args.load_model)
    else:
        model = modules.ViT(
            img_size=args.resize_img,
            in_channels=args.in_channels,
            patch_size=args.patch_size,
            pretraining=True,
            mlp_size=args.mlp_size,
            mlp_dropout=args.dropout_value,
            num_heads=args.num_heads,
            embedding_dim=args.embedding_dim,
            embedding_dropout=args.embedding_dropout_value,
            num_transformer_layers=args.num_transformer_layers
        )
    logging.info("Model initialized!")

    # # Print a summary of our ViT model
    logging.info(f"""\n {summary(model=model,
                                 input_size=(args.batch_size, args.in_channels, args.resize_img, args.resize_img),
                                 col_names=["input_size", "output_size", "num_params", "trainable"],
                                 col_width=20,
                                 row_settings=["var_names"],
                                 verbose=0)}
                 """)

    optimizer = Adam(
        params=model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=args.adam_weight_decay
    )
    loss_fn = MSELoss()

    logging.info("Running pretraining...")
    if args.load_checkpoint is not None:
        loaded_checkpoint = torch.load(args.load_checkpoint)
    else:
        loaded_checkpoint = None

    results = utils.pretrain(
        model=model,
        train_dataloader=train_dataloader,
        loss_fn=loss_fn,
        metrics={},
        optimizer=optimizer,
        save_path=args.save_path,
        optim_scheduler_patience=args.lr_decay_patience,
        warmup_duration=args.warmup_iterations,
        max_batch_iterations=args.max_batch_iterations,
        nb_iterations_checkpoint=args.checkpoint_iterations,
        loaded_checkpoint=loaded_checkpoint
    )
    logging.info("Pretraining finished!")

    train_loss = results["train_loss"]
    epochs = range(len(train_loss))
    plt.figure(figsize=(15, 7))
    plt.plot(epochs, train_loss, label="Train loss")
    plt.title("Loss")
    plt.xlabel("Batch iterations")
    plt.show()


if __name__ == '__main__':
    args = pretrain_flags()
    # save_path = Path(args.save_path)
    logging.basicConfig(
        # filename=f"{save_path / 'pretraining.log'}",
        # filemode='w',
        format="%(asctime)s [%(levelname)s]: %(message)s",
        level=logging.INFO
    )
    # logging.info(f"Register logs in {save_path / 'pretraining.log'}")
    logging.info(f"Pretraining initialization...")
    logging.info("***** Chosen parameters *****\n---------------------------------------------------------------------")
    max_len = max([len(i) for i in vars(args).keys()])
    fmt_string = "\t%" + str(max_len) + "s : %s"

    for key, arg in sorted(vars(args).items()):
        logging.info(fmt_string, key, arg)

    main()
