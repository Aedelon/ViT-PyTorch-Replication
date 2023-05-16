#!/usr/bin/env python
# *** coding: utf-8 ***

"""flags.py: Argument parsers for pretraining script and training script

   * Author: Delanoe PIRARD
   * Email: delanoe.pirard.pro@gmail.com
   * Licence: MIT
"""

import argparse


def pretrain_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default=None,
        type=str,
        help="Use custom dataset for the pretraining. If None, the script will use ImageNet-21k."
    )
    parser.add_argument(
        "--patch_size",
        default=16,
        type=int,
        help="Height and width of the patches."
    )
    parser.add_argument(
        "--resize_img",
        default=224,
        type=int,

    )
    parser.add_argument(
        "--max_batch_iterations",
        default=1000000,
        type=int,
        help="Number of batch iterations for pretraining. Default: 1 000 000."
    )
    parser.add_argument(
        "--adam_weight_decay",
        default=0.03,
        type=float,
        help="Value of weight decay for Adam optimizer."
    )
    parser.add_argument(
        "--batch_size",
        default=4094,
        type=int,
        help="Number of samples by batch for pretraining."
    )
    parser.add_argument(
        "--dropout_value",
        default=0.1,
        type=float,
        help="Dropout value."
    )
    parser.add_argument(
        "--embedding_dropout_value",
        default=0.1,
        type=float,
        help="Embedding dropout value."
    )
    parser.add_argument(
        "--learning_rate",
        default=0.001,
        type=float,
        help="Learning rate value"
    )
    parser.add_argument(
        "--warmup_iterations",
        default=10000,
        type=int,
        help="Number of iterations for warmup."
    )
    parser.add_argument(
        "--lr_decay_patience",
        default=25,
        type=int,
        help="Number of iteration before decaying the learning rate."
    )
    parser.add_argument(
        "--save_path",
        default="./",
        type=str,
        help="Folder to save the checkpoints and the final pretrained model."
    )
    parser.add_argument(
        "--checkpoint_iterations",
        default=1000,
        type=int,
        help="Save a checkpoint each x iterations."
    )
    parser.add_argument(
        "--load_checkpoint",
        default=None,
        type=str,
        help="Path to the checkpoint to load."
    )
    parser.add_argument(
        "--load_model",
        default=None,
        type=str,
        help="Path to a pre-existent ViT model."
    )
    parser.add_argument(
        "--in_channels",
        default=3,
        type=int,
        help="Number of color channels."
    )
    parser.add_argument(
        "--mlp_size",
        default=3072,
        type=int,
        help="Size of the Feedforward layer in Transformer layers."
    )
    parser.add_argument(
        "--embedding_dim",
        default=768,
        type=int,
        help="Size of the embedding of patches."
    )
    parser.add_argument(
        "--num_heads",
        default=12,
        type=int,
        help="Number of heads in transformer layers."
    )
    parser.add_argument(
        "--num_transformer_layers",
        default=12,
        type=int,
        help="Number of transformer layers."
    )

    return parser.parse_args()


def train_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_dataset",
        default=None,
        type=str,
        help="Use custom dataset for the training."
    )
    parser.add_argument(
        "--val_dataset",
        default=None,
        type=str,
        help="Use custom dataset for the validation."
    )
    parser.add_argument(
        "--patch_size",
        default=16,
        type=int,
        help="Height and width of the patches. Default: 16."
    )
    parser.add_argument(
        "--resize_img",
        default=224,
        type=int,
        help="Resize image. Default: 384."
    )
    parser.add_argument(
        "--max_epochs",
        default=40,
        type=int,
        help="Number of epochs for training. Default: 40."
    )
    parser.add_argument(
        "--sgd_weight_decay",
        default=0,
        type=float,
        help="Value of weight decay for SGD optimizer. Default: 0."
    )
    parser.add_argument(
        "--label_smoothing_value",
        default=0.1,
        type=float,
        help="Value of label smoothing."
    )
    parser.add_argument(
        "--batch_size",
        default=512,
        type=int,
        help="Number of samples by batch for pretraining."
    )
    parser.add_argument(
        "--dropout_value",
        default=0.1,
        type=float,
        help="Dropout value. Default: 0.1."
    )
    parser.add_argument(
        "--embedding_dropout_value",
        default=0.1,
        type=float,
        help="Embedding dropout value. Default: 0.1."
    )
    parser.add_argument(
        "--gradient_clipping",
        default=True,
        type=bool,
        help="Enable gradient clipping. Default: True."
    )
    parser.add_argument(
        "--gradient_clipping_thresh",
        default=1,
        type=float,
        help="Value of the clipping threshold of gradient clipping. Default: 1."
    )
    parser.add_argument(
        "--learning_rate",
        default=0.001,
        type=float,
        help="Learning rate value"
    )
    parser.add_argument(
        "--save_path",
        default="./",
        type=str,
        help="Folder to save the checkpoints and the final pretrained model."
    )
    parser.add_argument(
        "--load_checkpoint",
        default=None,
        type=str,
        help="Path to the checkpoint to load."
    )
    parser.add_argument(
        "--load_model",
        default=None,
        type=str,
        help="Path to a pre-existent ViT model."
    )
    parser.add_argument(
        "--in_channels",
        default=3,
        type=int,
        help="Number of color channels."
    )
    parser.add_argument(
        "--mlp_size",
        default=3072,
        type=int,
        help="Size of the Feedforward layer in Transformer layers."
    )
    parser.add_argument(
        "--embedding_dim",
        default=768,
        type=int,
        help="Size of the embedding of patches."
    )
    parser.add_argument(
        "--num_heads",
        default=12,
        type=int,
        help="Number of heads in transformer layers."
    )
    parser.add_argument(
        "--num_transformer_layers",
        default=12,
        type=int,
        help="Number of transformer layers."
    )

    return parser.parse_args()
