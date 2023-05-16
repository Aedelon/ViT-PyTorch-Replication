#!/usr/bin/env python
# *** coding: utf-8 ***

"""modules.py: Contains the different modules which will be used to make a ViT model.

   * Author: Delanoe PIRARD
   * Email: delanoe.pirard.pro@gmail.com
   * Licence: MIT

   * Paper: "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
   * Paper's authors: Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn,
    Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold,
    Sylvain Gelly, Jakob Uszkoreit and Neil Houlsby.
   * Paper DOI: https://doi.org/10.48550/arXiv.2010.11929
"""

# IMPORTS -------------------------------------------------
import torch
import torch.nn as nn


# CLASSES -------------------------------------------------
class PatchEmbedding(nn.Module):
    """Create a PatchEmbedding module.

    This module will divide an image into different patches which will embedded.
    The below steps will be followed during the forward pass:
    1. Divide of the image into patches;
    2. Flatten the patch tensor;
    3. Concatenate classification token to the flattened path tensor;
    4. Add the positional embedding to the class embedding;
    5. Pass the tensor to a dropout layer.
    """

    def __init__(self,
                 in_channels: int = 3,
                 img_size: int = 224,
                 patch_size: int = 16,
                 embedding_dim: int = 768,
                 dropout: float = 0.1,
                 pretraining: bool = False,
                 device: torch.device = ('cuda' if torch.cuda.is_available() else 'cpu')):
        """Initialize PatchEmbedding

        :param in_channels: Number of image channels.
        :param img_size: Size of the image (height and width).
        :param patch_size: Number of pixels for the height and the width of the patches.
        :param embedding_dim: Dimension of the embedding for each patch.
        :param dropout: Value of the dropout layer.
        :param device: The device type Pytorch will use ('cuda', 'mps' or 'cpu').
        """
        super().__init__()

        # Assert the image size is compatible with the patch size
        assert img_size % patch_size == 0, f"Image size must be \
            divisible by patch size. Image size: {img_size}. Patch size: {patch_size}"

        # Calculate the number of patches (H * W / P^2)
        self.num_patches = (img_size * img_size) // patch_size ** 2
        self.patcher = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embedding_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
            device=device
        )
        self.class_embedding = nn.Parameter(
            data=torch.randn(1, 1, embedding_dim, device=device),
            requires_grad=True
        )
        self.position_embedding = nn.Parameter(
            data=torch.randn(1, self.num_patches + 1, embedding_dim, device=device),
            requires_grad=True
        )
        self.embedding_dropout = nn.Dropout(p=dropout)
        self.flatten = nn.Flatten(start_dim=2, end_dim=3)

        self.pretraining = pretraining
        self.mask = nn.Parameter(
            data=torch.randn(1, 1, embedding_dim, device=device)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        class_token = self.class_embedding.expand(batch_size, -1, -1)

        if self.pretraining:
            x_embedded = self.flatten(self.patcher(x)).permute(0, 2, 1)
            x_embedded_clone = x_embedded.detach().clone()

            for i in range(batch_size):
                random_array = torch.rand(self.num_patches)
                for j, rand in enumerate(random_array):
                    if rand < .4:
                        x_embedded[i, j] = self.mask
                    elif .4 <= rand < .45:
                        random_patch = torch.randint(0, self.num_patches, (1,))
                        x_embedded[i, j] = x_embedded_clone[i, random_patch[0]]
                    elif .45 <= rand < .5:
                        x_embedded[i, j] = x_embedded[i, j]

            return self.embedding_dropout(
                (torch.cat((class_token, x_embedded), dim=1))
                + self.position_embedding
            )
        else:
            return self.embedding_dropout(
                (torch.cat((class_token, self.flatten(self.patcher(x)).permute(0, 2, 1)), dim=1))
                + self.position_embedding
            )


class ViT(nn.Module):
    """Create the ViT model."""

    def __init__(self,
                 img_size: int = 224,
                 in_channels: int = 3,
                 patch_size: int = 16,
                 num_transformer_layers: int = 12,
                 embedding_dim: int = 768,
                 mlp_size: int = 3072,
                 num_heads: int = 12,
                 mlp_dropout: float = 0.1,
                 embedding_dropout: float = 0.1,
                 num_classes: int = 1000,
                 pretraining: bool = False,
                 device: torch.device = ('cuda' if torch.cuda.is_available() else 'cpu')):
        """Initialize ViT

        :param img_size: Size of the image (height and width).
        :param in_channels: Number of image channels.
        :param patch_size: Number of pixels for the height and the width of the patches.
        :param num_transformer_layers: Number of consecutive transformer layers.
        :param embedding_dim: Dimension of the embedding for each patch.
        :param mlp_size: Number of nodes in the feed-forward layer.
        :param num_heads: Number of heads in the transformer layers.
        :param mlp_dropout: Value of the dropout layer after the feed-forward-layer.
        :param embedding_dropout: Value of the dropout layer in the PatchEmbedding instance.
        :param num_classes: Number of classes to classify.
        :param device: The device type Pytorch will use ('cuda', 'mps' or 'cpu')
        """
        super().__init__()

        # Create patch embedding layer
        self.patch_embedding = PatchEmbedding(
            in_channels=in_channels,
            patch_size=patch_size,
            embedding_dim=embedding_dim,
            img_size=img_size,
            dropout=embedding_dropout,
            pretraining=pretraining,
            device=device
        )

        # Create the Transformer Encoder
        self.transformer_encoder = nn.Sequential(*[
            nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=num_heads,
                dim_feedforward=mlp_size,
                dropout=mlp_dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
                device=device
            ) for _ in range(num_transformer_layers)
        ])

        # Create classifier head
        self.classifier = nn.Sequential(
            nn.LayerNorm(
                normalized_shape=embedding_dim,
                device=device
            ),
            nn.Linear(
                in_features=embedding_dim,
                out_features=num_classes,
                device=device
            )
        )
        self.pretrain_head = nn.Sequential(
            nn.LayerNorm(
                normalized_shape=(embedding_dim, ((img_size * img_size) // patch_size ** 2)+1),
                device=device
            ),
            nn.Conv1d(
                in_channels=embedding_dim,
                out_channels=embedding_dim,
                kernel_size=1
            ),
            nn.Conv1d(
                in_channels=embedding_dim,
                out_channels=3,
                kernel_size=1
            )
        )
        self.pretraining = pretraining

    def forward(self, x):
        if self.pretraining:
            return self.pretrain_head(
                self.transformer_encoder(self.patch_embedding(x)).permute(0, 2, 1)
            ).permute(0, 2, 1)[:, 1:]
        else:
            return self.classifier(self.transformer_encoder(self.patch_embedding(x))[:, 0])
