# ViT: Low Memory Usage and Fast (Pre-)Training - PyTorch Replication
PyTorch's replication of the model ViT. The repository contains a script for pretraining a model and a script for 
training or fine-tuning a model.

## Pretraining usage:
````
pretrain_script.py \
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
````

## Training or fine-tuning usage
````
train_script.py \
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
````

## Soon(TM)
A **pretrained model** with ImageNet-21k and a notebook with fine-tuning on **CIFAR-10**, **CIFAR-100**, **ImageNet**, 
**ImageNet ReaL**, **Oxford Flowers-102** and **Oxford-IIIT-Pets**.

## Bibliography
[An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)