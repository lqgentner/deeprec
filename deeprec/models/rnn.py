from abc import abstractmethod
from pathlib import Path

import torch
from torch import Tensor
import torch.nn as nn

from deeprec.models.base import BaseModel, matrix2tensor
from deeprec.utils import ROOT_DIR, wandb_checkpoint_download


class PretrainMixin(nn.Module):
    @abstractmethod
    def forward(self, x):
        pass

    def _load_weights(
        self, ckpt_path: str | Path, submodule: str | None = None
    ) -> None:
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
        if isinstance(ckpt_path, str):
            if ckpt_path.split("/")[0] == "wandb:":
                # Download W&B artifact
                artifact_path = "/".join(ckpt_path.split("/")[2:])
                ckpt_path = wandb_checkpoint_download(artifact_path)
            else:
                # Not a WandB artifact
                ckpt_path = Path(ckpt_path)
        if not Path(ckpt_path).is_absolute():
            # Make relative path absolute
            ckpt_path = ROOT_DIR / ckpt_path

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


class CropResBlock(nn.Module):
    """Like ResNetV1, but use padding=0 instead of stride=2 to downsample identity."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        padding: int = 0,
        groups: int = 1,
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=padding,
            groups=groups,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=padding,
            groups=groups,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.crop: nn.ZeroPad2d | None = None
        self.downsample: nn.Sequential | None = None

        if padding != 1:
            # Change window size by cropping to the center
            self.crop = nn.ZeroPad2d(2 * (-1 + padding))

        if out_channels != in_channels:
            self.downsample = nn.Sequential(
                # Downsample identity with a 1x1 convolution
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    groups=groups,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: Tensor):
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.crop is not None:
            identity = self.crop(identity)

        if self.downsample is not None:
            identity = self.downsample(identity)

        x += identity
        x = self.relu(x)

        return x


class CropResNetLSTM(BaseModel, PretrainMixin):
    def __init__(
        self,
        n_tensor_inps: int,
        n_matrix_inps: int,
        n_vector_inps: int,
        cnn_channels: tuple[int, int, int, int] = (32, 48, 72, 108),
        rnn_layers: int = 2,
        rnn_hidden_size: int = 128,
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
        super().__init__(
            n_tensor_inps=n_tensor_inps,
            n_matrix_inps=n_matrix_inps,
            n_vector_inps=n_vector_inps,
            rnn_layers=rnn_layers,
            **kwargs,
        )

        # -------- ResNet --------

        CNN_LAYER_NUM = 4

        # Check that cnn_channels has correct length
        if not len(cnn_channels) == CNN_LAYER_NUM:
            raise ValueError("'cnn_channels' must contain 4 elements.")

        resnet_layers = []
        in_channels = n_matrix_inps + n_tensor_inps

        # Create Blocks of ResNet
        for out_channels in cnn_channels:
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
