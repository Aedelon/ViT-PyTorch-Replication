#!/usr/bin/env python
# *** coding: utf-8 ***

"""train.py: Script to train or fine-tune a ViT model.

   * Author: Delanoe PIRARD
   * Email: delanoe.pirard.pro@gmail.com
   * Licence: MIT

   * Usage: train_script.py \
            --batch_size=512 \
            --dropout_value=0.1 \
            --embedding_dim=768 \
            --embedding_dropout_value=0.1 \
            --gradient_clipping=True \
            --gradient_clipping_thresh=1 \
            --in_channels=3 \
            --label_smoothing_value=0.1 \
            --learning_rate=0.001 \
            --load_checkpoint=./your/checkpoint.pt \
            --load_model=./your/model.pt \
            --max_epochs=40 \
            --mlp_size=3072 \
            --num_heads=12 \
            --num_transformer_layers=12 \
            --patch_size=16 \
            --resize_img=224 \
            --save_path=./path/to/save/ \
            --sgd_weight_decay=0 \
            --train_dataset=./data/your_train_dataset/images/ \
            --val_dataset=./data/your_validation_dataset/images/
"""

import torch
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy, AUROC

import utils
import modules
import logging
import matplotlib.pyplot as plt

from torchvision import transforms, datasets
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from torchinfo import summary
from flags import train_flags


def main():
    args = train_flags()
    if args.train_dataset is None:
        logging.warning("No training dataset indicated with `--train_dataset`.")

    if args.val_dataset is None:
        logging.warning("No validation dataset indicated with `--val_dataset`.")

    if args.train_dataset is None or args.val_dataset is None:
        logging.warning("Use of the OxfordIIITPet dataset for training and validation.")
        # Load training set
        full_train_dataset = datasets.OxfordIIITPet(
            root="./data/Flowers102/train",
            download=True,
            transform=transforms.Compose([
                transforms.Resize((args.resize_img, args.resize_img)),
                transforms.ToTensor()
            ]),
            target_types="category"
        )
        class_names = full_train_dataset.classes

        # Split `full_train_dataset` into a training dataset and a validation dataset
        train_dataset, val_dataset = random_split(full_train_dataset, [0.9, 0.1])

        # Data loader creation
        train_dataloader = DataLoader(
            dataset=train_dataset,
            shuffle=True,
            num_workers=0,
            batch_size=args.batch_size
        )

        val_dataloader = DataLoader(
            dataset=val_dataset,
            num_workers=0,
            batch_size=args.batch_size
        )
    else:
        # Create dataloader
        logging.info("Loading datasets...")
        logging.info(f"Training dataset: {args.train_dataset}")
        logging.info(f"Validation dataset: {args.val_dataset}")
        train_dataloader, val_dataloader, class_names = utils.create_dataloaders(
            train_dir=args.train_dataset,
            test_dir=args.val_dataset,
            transform=transforms.Compose([
                transforms.Resize((args.resize_img, args.resize_img)),
                transforms.ToTensor()
            ]),
            batch_size=args.batch_size,
            num_workers=2
        )
        logging.info("Datasets loaded!")

    logging.info("Initializing model...")
    if args.load_model is not None:
        model = modules.ViT(torch.load(args.load_model))
        model.pretraining = False
        logging.info("Model loaded!")
    else:
        model = modules.ViT(
            img_size=args.resize_img,
            in_channels=args.in_channels,
            patch_size=args.patch_size,
            pretraining=False,
            mlp_size=args.mlp_size,
            mlp_dropout=args.dropout_value,
            num_heads=args.num_heads,
            embedding_dim=args.embedding_dim,
            embedding_dropout=args.embedding_dropout_value,
            num_transformer_layers=args.num_transformer_layers,
            num_classes=len(class_names)
        )
        logging.info("Model initialized!")

    # # Print a summary of model
    logging.info(f"""\n {summary(model=model,
                                 input_size=(args.batch_size, args.in_channels, args.resize_img, args.resize_img),
                                 col_names=["input_size", "output_size", "num_params", "trainable"],
                                 col_width=20,
                                 row_settings=["var_names"],
                                 verbose=0)}
                 """)

    # Initialize optimizer, loss function and metrics
    optimizer = SGD(
        params=model.parameters(),
        lr=args.learning_rate,
        momentum=0.9,
        weight_decay=args.sgd_weight_decay
    )
    loss_fn = CrossEntropyLoss(label_smoothing=args.label_smoothing_value)
    acc_fn = Accuracy(task='multiclass', num_classes=len(class_names))
    # auroc_fn = AUROC(task='multiclass', num_classes=len(class_names))

    # Load Checkpoint ?
    if args.load_checkpoint is not None:
        loaded_checkpoint = torch.load(args.load_checkpoint)
    else:
        loaded_checkpoint = None

    # Training
    logging.info("Running training...")
    results = utils.train(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=val_dataloader,
        loss_fn=loss_fn,
        metrics={"Accuracy": acc_fn},
        optimizer=optimizer,
        save_path=args.save_path,
        epochs=args.max_epochs,
        loaded_checkpoint=loaded_checkpoint
    )
    logging.info("training finished!")

    # Plot the losses and the metrics
    plt.figure(figsize=(15, 7))

    train_loss = results["train_loss"]
    val_loss = results["validation_loss"]
    train_metrics = results["train_metrics"]
    val_metrics = results["validation_metrics"]
    epochs = range(len(train_loss))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label="Train loss")
    plt.plot(epochs, val_loss, label="Test loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_metrics["Accuracy"], label="Train accuracy")
    plt.plot(epochs, val_metrics["Accuracy"], label="Test accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()

    plt.show()


if __name__ == '__main__':
    main_args = train_flags()
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s]: %(message)s",
        level=logging.INFO
    )

    logging.info(f"Pretraining initialization...")
    logging.info("***** Chosen parameters *****\n---------------------------------------------------------------------")
    max_len = max([len(i) for i in vars(main_args).keys()])
    fmt_string = "\t%" + str(max_len) + "s : %s"

    for key, arg in sorted(vars(main_args).items()):
        logging.info(fmt_string, key, arg)

    main()
