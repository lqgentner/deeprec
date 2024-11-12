import abc
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor

from deepwaters.models.base import BaseModel, matrix2tensor
from deepwaters.models.cropresnet import CropResBlock, ResNetMixin
from deepwaters.utils import ROOT_DIR, wandb_checkpoint_download


class PretrainMixin(nn.Module):
    @abc.abstractmethod
    def forward(self, x) -> None:
        pass

    def _load_weights(self, ckpt_path: str, submodule: str | None = None) -> None:
        """Load model weights from a state dict.

        Parameters
        ----------

        ckpt_file: str
            The path of the checkpoint
            Can be a system path or Weights & Biases artifact.
            In the latter case it must start with `wandb://`,
            e.g. `wandb://user/project/model-12345678:best`.

        submodule: str, optional
            The name of the submodule to load the weights for.
            If not specified, the weights for the complete module will
            be loaded.

        """
        # Path can be an absolute system path, relative system path,
        # or a Weights & Biases artifact (starts with "wandb://")
        if ckpt_path.split("/")[0] == "wandb:":
            # Download W&B artifact
            artifact_path = "/".join(ckpt_path.split("/")[2:])
            ckpt_path = wandb_checkpoint_download(artifact_path)
        elif not Path(ckpt_path).is_absolute():
            # Make relative path absolute
            ckpt_path = ROOT_DIR / ckpt_path
        else:
            # Path is absolute
            ckpt_path = Path(ckpt_path)

        print(f"Loading pre-training checkpoint from: {ckpt_path}")
        state_dict = torch.load(ckpt_path)["state_dict"]

        if submodule:
            # Get submodule instance
            obj = getattr(self, submodule)
            # Remove keys from other submodules,
            # than remove the submodule prefix from the remaining keys
            state_dict = {
                key.removeprefix(submodule + "."): value
                for key, value in state_dict.items()
                if key.startswith(submodule)
            }
        else:
            obj = self

        if not isinstance(obj, nn.Module):
            raise TypeError(f"Specified submodule {submodule} must be a nn.Module.")

        msg = obj.load_state_dict(state_dict)
        print(msg)

    def _freeze_submodule(self, name: str) -> None:
        """Freeze all params of a sub-module for inference."""
        sub: nn.Module = getattr(self, name)
        for param in sub.parameters():
            param.requires_grad = False
        sub.eval()
        print(f"Module '{name}' frozen.")


class CropResNetLSTM(BaseModel, ResNetMixin, PretrainMixin):
    def __init__(
        self,
        n_tensor_inps: int,
        n_matrix_inps: int,
        n_vector_inps: int,
        cnn_channel_mult: float = 1.0,
        rnn_layers: int = 2,
        rnn_hidden_size: int = 24,
        rnn_dropout: float = 0.0,
        pretrain_checkpoint: str | None = None,
        cnn_checkpoint: str | None = None,
        freeze_encoder: bool = False,
        **kwargs,
    ):
        """Combination of CropResNet encoder and a LSTM. Requires spatial input of size 35x35."""
        super(CropResNetLSTM, self).__init__(
            n_tensor_inps=n_tensor_inps,
            n_matrix_inps=n_matrix_inps,
            n_vector_inps=n_vector_inps,
            rnn_layers=rnn_layers,
            **kwargs,
        )

        # -------- ResNet --------

        in_channels = n_matrix_inps + n_tensor_inps

        # Stepwise increase of the CNN channels
        resnet_layers = []
        for _ in range(4):
            out_channels = round(cnn_channel_mult * in_channels)
            resnet_layers.extend(
                [
                    CropResBlock(in_channels, in_channels),
                    CropResBlock(in_channels, out_channels),
                ]
            )
            in_channels = out_channels

        self.cnn = nn.Sequential(
            *resnet_layers,
            nn.Conv2d(out_channels, out_channels, kernel_size=3),
            nn.ReLU(),
        )

        # -------- LSTM --------

        rnn_input_size = out_channels + n_vector_inps

        self.rnn = nn.LSTM(
            rnn_input_size,
            rnn_hidden_size,
            num_layers=rnn_layers,
            batch_first=True,
            dropout=rnn_dropout,
        )

        # -------- Fully Connected --------

        # If the uncertainty is returned, we need two neurons in last layer
        if self.return_uncertainty:
            fc_out = 2
            self.softplus = nn.Softplus()
        else:
            fc_out = 1
        self.fc = nn.Sequential(
            nn.Linear(rnn_hidden_size, 24),
            nn.ReLU(),
            nn.Linear(24, fc_out),
        )
        if pretrain_checkpoint:
            # Load state dict from pre-training
            self._load_weights(pretrain_checkpoint)
            # Freeze encoder
            if freeze_encoder:
                self._freeze_submodule("cnn")
        else:
            self._init_weights()

        # -------- Checkpoint loading --------

        if pretrain_checkpoint:
            # Load state dict of full module from pre-training
            self._load_weights(pretrain_checkpoint)
        if cnn_checkpoint:
            # Load state dict of CNN encoder from pre-training
            self._load_weights(cnn_checkpoint, submodule="cnn")
        if freeze_encoder:
            # Freeze CNN encoder
            if pretrain_checkpoint or cnn_checkpoint:
                self._freeze_submodule("cnn")
                print("Encoder parameters frozen.")
            else:
                print("No checkpoint specified: Ignoring `freeze_encoder=True`.")

    def forward(self, *_, **inputs: Tensor) -> Tensor:
        xt = inputs["tensor_inputs"]
        xm = inputs["matrix_inputs"]
        xv = inputs["vector_inputs"]

        N, _, L, H, W = xt.shape
        # Repeat matrix inputs for every time step
        xm = matrix2tensor(xm, out_depth=L)
        # Combine spatial inputs
        x = torch.cat([xt, xm], dim=1)
        # Shape: (N, C, L, H, W)
        x = x.transpose(1, 2)
        # Shape: (N, L, C, H, W)
        # Compress batch and time dimension
        x = x.reshape(N * L, -1, H, W)
        # Shape: (N * L, C, H, W)
        # Feed into CNN
        x = self.cnn(x).squeeze()
        # Shape: (N * L, C)
        # Undo batch-time squeeze
        x = x.reshape(N, L, -1)
        # Shape: (N, L, C)
        # Combine spatial and vector inputs
        x = torch.cat([x, xv.transpose(1, 2)], dim=2)
        # Feed into RNN
        x, _ = self.rnn(x)
        # Shape: (N, L, C)
        # Only take last prediction
        x = x[:, -1]
        # Shape: (N, C)
        # Feed into FC
        x = self.fc(x)

        if self.return_uncertainty:
            # split into mu and sigma
            mu, sig = x.T
            # ensure sigma is positive
            sig = self.softplus(sig)
            # Recombine
            x = torch.stack([mu, sig], dim=1)
            # Shape: (N, 2)
        else:
            # Only mu is returned
            x = x.squeeze()
            # Shape: (N, )

        return x


class CropResNetLSTM2(BaseModel, ResNetMixin, PretrainMixin):
    def __init__(
        self,
        n_tensor_inps: int,
        n_matrix_inps: int,
        n_vector_inps: int,
        cnn_channel_mult: float = 1.0,
        rnn_layers: int = 2,
        rnn_hidden_size: int = 24,
        rnn_dropout: float = 0.0,
        pretrain_checkpoint: str | None = None,
        cnn_checkpoint: str | None = None,
        freeze_encoder: bool = False,
        **kwargs,
    ):
        """Combination of CropResNet encoder and a LSTM.
        More neurons in hidden fully connected layer (64 vs 24).
        Requires spatial input of size 35x35.
        """
        super(CropResNetLSTM2, self).__init__(
            n_tensor_inps=n_tensor_inps,
            n_matrix_inps=n_matrix_inps,
            n_vector_inps=n_vector_inps,
            rnn_layers=rnn_layers,
            **kwargs,
        )

        # -------- ResNet --------

        in_channels = n_matrix_inps + n_tensor_inps

        # Stepwise increase of the CNN channels
        resnet_layers = []
        for _ in range(4):
            out_channels = round(cnn_channel_mult * in_channels)
            resnet_layers.extend(
                [
                    CropResBlock(in_channels, in_channels),
                    CropResBlock(in_channels, out_channels),
                ]
            )
            in_channels = out_channels

        self.cnn = nn.Sequential(
            *resnet_layers,
            nn.Conv2d(out_channels, out_channels, kernel_size=3),
            nn.ReLU(),
        )

        # -------- LSTM --------

        rnn_input_size = out_channels + n_vector_inps

        self.rnn = nn.LSTM(
            rnn_input_size,
            rnn_hidden_size,
            num_layers=rnn_layers,
            batch_first=True,
            dropout=rnn_dropout,
        )

        # -------- Fully Connected --------

        # If the uncertainty is returned, we need two neurons in last layer
        if self.return_uncertainty:
            fc_out = 2
            self.softplus = nn.Softplus()
        else:
            fc_out = 1
        self.fc = nn.Sequential(
            nn.Linear(rnn_hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, fc_out),
        )
        if pretrain_checkpoint:
            # Load state dict from pre-training
            self._load_weights(pretrain_checkpoint)
            # Freeze encoder
            if freeze_encoder:
                self._freeze_submodule("cnn")
        else:
            self._init_weights()

        # -------- Checkpoint loading --------

        if pretrain_checkpoint:
            # Load state dict of full module from pre-training
            self._load_weights(pretrain_checkpoint)
        if cnn_checkpoint:
            # Load state dict of CNN encoder from pre-training
            self._load_weights(cnn_checkpoint, submodule="cnn")
        if freeze_encoder:
            # Freeze CNN encoder
            if pretrain_checkpoint or cnn_checkpoint:
                self._freeze_submodule("cnn")
                print("Encoder parameters frozen.")
            else:
                print("No checkpoint specified: Ignoring `freeze_encoder=True`.")

    def forward(self, *_, **inputs: Tensor) -> Tensor:
        xt = inputs["tensor_inputs"]
        xm = inputs["matrix_inputs"]
        xv = inputs["vector_inputs"]

        N, _, L, H, W = xt.shape
        # Repeat matrix inputs for every time step
        xm = matrix2tensor(xm, out_depth=L)
        # Combine spatial inputs
        x = torch.cat([xt, xm], dim=1)
        # Shape: (N, C, L, H, W)
        x = x.transpose(1, 2)
        # Shape: (N, L, C, H, W)
        # Compress batch and time dimension
        x = x.reshape(N * L, -1, H, W)
        # Shape: (N * L, C, H, W)
        # Feed into CNN
        x = self.cnn(x).squeeze()
        # Shape: (N * L, C)
        # Undo batch-time squeeze
        x = x.reshape(N, L, -1)
        # Shape: (N, L, C)
        # Combine spatial and vector inputs
        x = torch.cat([x, xv.transpose(1, 2)], dim=2)
        # Feed into RNN
        x, _ = self.rnn(x)
        # Shape: (N, L, C)
        # Only take last prediction
        x = x[:, -1]
        # Shape: (N, C)
        # Feed into FC
        x = self.fc(x)

        if self.return_uncertainty:
            # split into mu and sigma
            mu, sig = x.T
            # ensure sigma is positive
            sig = self.softplus(sig)
            # Recombine
            x = torch.stack([mu, sig], dim=1)
            # Shape: (N, 2)
        else:
            # Only mu is returned
            x = x.squeeze()
            # Shape: (N, )

        return x


class CropResNetEncoder(nn.Module):
    def __init__(self, in_channels: int, channel_mult: float = 1.5):
        """CNN Encoder for the CropResNet-LSTM.
        This can be used for training a single-time-step encoder separately,
        before combining with the RNN.
        Requires spatial input of size 35x35.
        """
        super(CropResNetEncoder, self).__init__()

        self.in_channels = in_channels

        # Create ResNet Blocks
        ch_curr = self.in_channels
        ch_next = ch_curr
        res_blocks = 8

        resnet_layers = []
        for _ in range(res_blocks // 2):
            resnet_layers.extend(
                [
                    CropResBlock(ch_curr, ch_next),
                    CropResBlock(ch_next, ch_next),
                ]
            )
            ch_curr = ch_next
            ch_next = round(channel_mult * ch_curr)

        self.resnet = nn.Sequential(*resnet_layers)
        self.out_channels = ch_curr

        self.output_layer = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3), nn.ReLU()
        )
        self.crop = nn.ZeroPad2d(-2 * res_blocks)

        self.downsample = nn.Sequential(
            # Downsample identity with a 1x1 convolution
            nn.Conv2d(
                self.in_channels,
                self.out_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(self.out_channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        # long skip connection
        shortcut = x
        shortcut = self.downsample(self.crop(shortcut))

        x = self.resnet(x)
        x = x + shortcut
        x = self.output_layer(x)

        return x.squeeze()
