import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict, Callable

class AutoEncoderMLP(nn.Module):
    """
    Torch implementation of an autoencoder with Multilayer Perceptron encoder and decoder.

    The encoder and decoder architectures are defined by the user through dimension lists.
    Users can also specify prevalence and content covariates, as well as target labels to guide the encoding and decoding.

    Attributes:
        encoder_dims (List[int]): Dimensions of the encoder layers.
        encoder_non_linear_activation (Optional[str]): Activation function for encoder ("relu" or "sigmoid").
        encoder_bias (bool): Whether to use bias in encoder layers.
        decoder_dims (List[int]): Dimensions of the decoder layers.
        decoder_non_linear_activation (Optional[str]): Activation function for decoder ("relu" or "sigmoid").
        decoder_bias (bool): Whether to use bias in decoder layers.
        dropout (nn.Dropout): Dropout layer.
        encoder_nonlin (Optional[Callable]): Encoder activation function.
        decoder_nonlin (Optional[Callable]): Decoder activation function.
        encoder (nn.ModuleDict): Encoder layers.
        decoder (nn.ModuleDict): Decoder layers.
    """

    def __init__(
        self,
        encoder_dims: List[int] = [2000, 1024, 512, 20],
        encoder_non_linear_activation: Optional[str] = "relu",
        encoder_bias: bool = True,
        decoder_dims: List[int] = [20, 1024, 2000],
        decoder_non_linear_activation: Optional[str] = None,
        decoder_bias: bool = False,
        dropout: float = 0.0,
        use_batch_norm: bool = False,
        use_layer_norm: bool = False
    ):
        # super(AutoEncoderMLP, self).__init__()
        super().__init__()

        self.encoder_dims = encoder_dims
        self.encoder_non_linear_activation = encoder_non_linear_activation
        self.encoder_bias = encoder_bias
        self.decoder_dims = decoder_dims
        self.decoder_non_linear_activation = decoder_non_linear_activation
        self.decoder_bias = decoder_bias
        self.dropout = nn.Dropout(p=dropout)
    
        
        self.encoder_nonlin: Optional[Callable] = None
        if encoder_non_linear_activation is not None:
            self.encoder_nonlin = {"relu": F.relu, "sigmoid": torch.sigmoid}[
                encoder_non_linear_activation
            ]
        
        self.decoder_nonlin: Optional[Callable] = None
        if decoder_non_linear_activation is not None:
            self.decoder_nonlin = {"relu": F.relu, "sigmoid": torch.sigmoid}[
                decoder_non_linear_activation
            ]

        self.encoder = nn.ModuleDict()
        self.encoder_norm_layers = nn.ModuleDict()
        for i in range(len(encoder_dims) - 1):
            self.encoder[f"enc_{i}"] = nn.Linear(
                encoder_dims[i], encoder_dims[i + 1], bias=encoder_bias
            )
            if use_batch_norm:
                self.encoder_norm_layers[f"norm_{i}"] = nn.BatchNorm1d(encoder_dims[i + 1])
            elif use_layer_norm:
                self.encoder_norm_layers[f"norm_{i}"] = nn.LayerNorm(encoder_dims[i + 1])

        self.decoder = nn.ModuleDict()
        self.decoder_norm_layers = nn.ModuleDict()
        for i in range(len(decoder_dims) - 1):
            self.decoder[f"dec_{i}"] = nn.Linear(
                decoder_dims[i], decoder_dims[i + 1], bias=decoder_bias
            )
            if use_batch_norm:
                self.decoder_norm_layers[f"norm_{i}"] = nn.BatchNorm1d(decoder_dims[i + 1])
            elif use_layer_norm:
                self.decoder_norm_layers[f"norm_{i}"] = nn.LayerNorm(decoder_dims[i + 1])


    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode the input.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Encoded representation.
        """
        hid = x
        for i, (_, layer) in enumerate(self.encoder.items()):
            hid = layer(hid)
            if f"norm_{i}" in self.encoder_norm_layers:
                hid = self.encoder_norm_layers[f"norm_{i}"](hid)
            if (
                i < len(self.encoder) - 1
                and self.encoder_non_linear_activation is not None
            ):
                hid = self.encoder_nonlin(hid)
            hid = self.dropout(hid)
        return hid

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode the input.

        Args:
            z (torch.Tensor): Encoded representation.

        Returns:
            torch.Tensor: Reconstructed input.
        """
        hid = z
        for i, (_, layer) in enumerate(self.decoder.items()):
            hid = layer(hid)
            if f"norm_{i}" in self.decoder_norm_layers:
                hid = self.decoder_norm_layers[f"norm_{i}"](hid)
            if (
                i < len(self.decoder) - 1
                and self.decoder_non_linear_activation is not None
            ):
                hid = self.decoder_nonlin(hid)
            hid = self.dropout(hid)
        return hid

    def forward(
        self,
        x: torch.Tensor,
        prevalence_covariates: Optional[torch.Tensor],
        content_covariates: Optional[torch.Tensor]
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform forward pass through the autoencoder.

        Args:
            x (torch.Tensor): Input tensor.
            prevalence_covariates (Optional[torch.Tensor]): Prevalence covariates.
            content_covariates (Optional[torch.Tensor]): Content covariates.
            to_simplex (bool): Whether to apply softmax to the encoded representation.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Reconstructed input and encoded representation.
        """
        if prevalence_covariates is not None:
            x = torch.cat((x, prevalence_covariates), 1)
        z = self.encode(x)
        theta = F.softmax(z, dim=1)
        if content_covariates is not None:
            theta_x = torch.cat((theta, content_covariates), 1)
        else:
            theta_x = theta
        x_recon = self.decode(theta_x)
        return x_recon, theta, z
        

class EncoderMLP(nn.Module):
    """
    Torch implementation of an encoder Multilayer Perceptron.

    Attributes:
        encoder_dims (List[int]): Dimensions of the encoder layers.
        encoder_non_linear_activation (Optional[str]): Activation function for encoder ("relu" or "sigmoid").
        encoder_bias (bool): Whether to use bias in encoder layers.
        dropout (nn.Dropout): Dropout layer.
        encoder_nonlin (Optional[Callable]): Encoder activation function.
        encoder (nn.ModuleDict): Encoder layers.
    """
    def __init__(
        self,
        encoder_dims: List[int] = [2000, 1024, 512, 20],
        encoder_non_linear_activation: Optional[str] = "relu",
        encoder_bias: bool = True,
        dropout: float = 0.0,
        use_batch_norm: bool = False,
        use_layer_norm: bool = False,
    ):
        super(EncoderMLP, self).__init__()

        self.encoder_dims = encoder_dims
        self.encoder_non_linear_activation = encoder_non_linear_activation
        self.encoder_bias = encoder_bias
        self.dropout = nn.Dropout(p=dropout)
        
        
        self.encoder_nonlin: Optional[Callable] = None
        if encoder_non_linear_activation is not None:
            self.encoder_nonlin = {"relu": F.relu, "sigmoid": torch.sigmoid}[
                encoder_non_linear_activation
            ]

        self.encoder = nn.ModuleDict()
        self.norm_layers = nn.ModuleDict()

        for i in range(len(encoder_dims) - 1):
            self.encoder[f"enc_{i}"] = nn.Linear(
                encoder_dims[i], encoder_dims[i + 1], bias=encoder_bias
            )
            if use_batch_norm:
                self.norm_layers[f"norm_{i}"] = nn.BatchNorm1d(encoder_dims[i + 1])
            elif use_layer_norm:
                self.norm_layers[f"norm_{i}"] = nn.LayerNorm(encoder_dims[i + 1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode the input.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Encoded representation.
        """
        hid = x
        for i, (_, layer) in enumerate(self.encoder.items()):
            hid = self.dropout(layer(hid))
            if f"norm_{i}" in self.norm_layers:
                hid = self.norm_layers[f"norm_{i}"](hid)
            if (
                i < len(self.encoder) - 1
                and self.encoder_non_linear_activation is not None
            ):
                hid = self.encoder_nonlin(hid)
        return hid


class DecoderMLP(nn.Module):
    """
    Torch implementation of a decoder Multilayer Perceptron.

    Attributes:
        decoder_dims (List[int]): Dimensions of the decoder layers.
        decoder_non_linear_activation (Optional[str]): Activation function for decoder ("relu" or "sigmoid").
        decoder_bias (bool): Whether to use bias in decoder layers.
        dropout (nn.Dropout): Dropout layer.
        decoder_nonlin (Optional[Callable]): Decoder activation function.
        decoder (nn.ModuleDict): Decoder layers.
    """
    def __init__(
        self,
        decoder_dims: List[int] = [20, 1024, 2000],
        decoder_non_linear_activation: Optional[str] = None,
        decoder_bias: bool = False,
        dropout: float = 0.0,
        use_batch_norm: bool = False,
        use_layer_norm: bool = False,
    ):
        super(DecoderMLP, self).__init__()

        self.decoder_dims = decoder_dims
        self.decoder_non_linear_activation = decoder_non_linear_activation
        self.decoder_bias = decoder_bias
        self.dropout = nn.Dropout(p=dropout)

        self.decoder_nonlin: Optional[Callable] = None
        if decoder_non_linear_activation is not None:
            self.decoder_nonlin = {"relu": F.relu, "sigmoid": torch.sigmoid}[
                decoder_non_linear_activation
            ]

        self.decoder = nn.ModuleDict()
        self.norm_layers = nn.ModuleDict()
        
        for i in range(len(decoder_dims) - 1):
            self.decoder[f"dec_{i}"] = nn.Linear(
                decoder_dims[i], decoder_dims[i + 1], bias=decoder_bias
            )
            if use_batch_norm:
                self.norm_layers[f"norm_{i}"] = nn.BatchNorm1d(decoder_dims[i + 1])
            elif use_layer_norm:
                self.norm_layers[f"norm_{i}"] = nn.LayerNorm(decoder_dims[i + 1])

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode the input.

        Args:
            z (torch.Tensor): Encoded representation.

        Returns:
            torch.Tensor: Reconstructed input.
        """
        hid = z
        for i, (_, layer) in enumerate(self.decoder.items()):
            hid = self.dropout(layer(hid))
            if f"norm_{i}" in self.norm_layers:
                hid = self.norm_layers[f"norm_{i}"](hid)
            if (
                i < len(self.decoder) - 1
                and self.decoder_non_linear_activation is not None
            ):
                hid = self.decoder_nonlin(hid)
        return hid