from dataclasses import dataclass

import torch
from torch import Tensor
from torch.utils.data import Dataset


@dataclass
class DeepRecTensors:
    """
    Container for all tensors required by DeepRecDataset.

    Attributes
    ----------
    lookup : Tensor
        Lookup table mapping from sample index to (time, lat, lon) indices.
    tensor_inputs : Tensor | None
        3D input variables (time, lat, lon) as tensor, or None if not used.
    matrix_inputs : Tensor | None
        2D input variables (lat, lon) as tensor, or None if not used.
    vector_inputs : Tensor | None
        1D input variables (time,) as tensor, or None if not used.
    scalar_inputs : Tensor | None
        0D input variables as tensor, or None if not used.
    target : Tensor | None
        Target variable as tensor, or None if not used (e.g., in prediction).
    weight : Tensor | None
        Latitude weights as tensor, or None if not used.
    """

    lookup: Tensor
    tensor_inputs: Tensor | None
    matrix_inputs: Tensor | None
    vector_inputs: Tensor | None
    scalar_inputs: Tensor | None
    target: Tensor | None
    weight: Tensor | None


class DeepRecDataset(Dataset):
    def __init__(
        self,
        tensors: DeepRecTensors,
        space_window: int,
        time_window: int,
    ) -> None:
        super().__init__()

        # Infer spatial extension
        for inputs in [
            tensors.tensor_inputs,
            tensors.matrix_inputs,
            tensors.vector_inputs,
            tensors.scalar_inputs,
        ]:
            if inputs is not None:
                nlat = inputs.size(2)
                nlon = inputs.size(3)
                break
        else:
            raise ValueError("No inputs provided, at least one is required.")

        # Space window
        if space_window % 2 != 1:
            raise ValueError(f"{space_window = } must be an odd integer.")
        # Half extension of spacial window (without center)
        # e.g. space_window = 5 -> half_patchwidth = 2
        half_patchwidth = space_window // 2

        # Time window
        # Previous time steps excluding current time step
        past_steps = time_window - 1

        # Set attributes
        self.lookup = tensors.lookup
        self.tensor_inputs = tensors.tensor_inputs
        self.matrix_inputs = tensors.matrix_inputs
        self.vector_inputs = tensors.vector_inputs
        self.scalar_inputs = tensors.scalar_inputs
        self.target = tensors.target
        self.weight = tensors.weight
        self.half_patchwidth = half_patchwidth
        self.past_steps = past_steps
        self.nlat = nlat
        self.nlon = nlon

        return

    def __getitem__(self, index) -> dict[str, Tensor | dict[str, Tensor]]:
        """Returns a single sample."""
        time_idx, lat_idx, lon_idx = self.lookup[index]

        # Calculate extend of time and space windows
        time_min = time_idx - self.past_steps
        time_max = time_idx + 1

        lat_min = lat_idx - self.half_patchwidth
        lat_max = lat_idx + self.half_patchwidth + 1
        lon_min = lon_idx - self.half_patchwidth
        lon_max = lon_idx + self.half_patchwidth + 1

        lats: Tensor | slice
        lons: Tensor | slice

        # Check if space window extension is within index ranges
        if lat_min < 0 or lat_max >= self.nlat or lon_min < 0 or lon_max >= self.nlon:
            # Latitude and/or longitude are out-of-range, we have to use advanced indexing
            # with arrays
            lats, lons = torch.arange(lat_min, lat_max), torch.arange(lon_min, lon_max)
            # Repeat lat rows and lon cols
            lats, lons = torch.meshgrid(lats, lons, indexing="ij")
            # Handle wrapping around the poles and the date line
            lats, lons = wrap_coord_indices(lats, lons, nlat=self.nlat, nlon=self.nlon)
        else:
            # Latitude and longitude are continuous, use standard indexing with slices
            lats, lons = slice(lat_min, lat_max), slice(lon_min, lon_max)

        # Build input dictionary
        inputs: dict[str, Tensor] = {}

        # Create scalars
        if self.scalar_inputs is not None:
            inputs["scalar_inputs"] = self.scalar_inputs[:, time_idx, lat_idx, lon_idx]

        # Create vectors, 1D time series (time,)
        if self.vector_inputs is not None:
            inputs["vector_inputs"] = self.vector_inputs[
                :, time_min:time_max, lat_idx, lon_idx
            ]

        # Create matrices, 2D patches (lat, lon)
        if self.matrix_inputs is not None:
            # Flip latitude axis (+90° to -90°)
            inputs["matrix_inputs"] = self.matrix_inputs[:, time_idx, lats, lons].flip(
                1
            )

        # Create tensors, 3D cubes (time, lat, lon)
        if self.tensor_inputs is not None:
            # Flip latitude axis (+90° to -90°)
            inputs["tensor_inputs"] = self.tensor_inputs[
                :, time_min:time_max, lats, lons
            ].flip(2)

        # Build sample dictionary
        sample: dict[str, Tensor | dict[str, Tensor]] = {}

        sample["inputs"] = inputs
        # Return target and additional arguments in train/val/test phase
        if self.target is not None and self.weight is not None:
            sample["target"] = self.target[index]
            sample["weight"] = self.weight[lat_idx]

        return sample

    def __len__(self) -> int:
        return len(self.lookup)


def wrap_coord_indices(
    lat: Tensor, lon: Tensor, nlat: int = 360, nlon: int = 720
) -> tuple[Tensor, Tensor]:
    """Handle extensions of coordinate indices (latitude, longitude) over the poles
    and/or over the date time. The returned indices all lie within the provided bounds.

    Parameters
    ----------
    lat: Tensor
        The latitude indices of the window. If the window has N x M grid cells,
        lat must have shape (N, M).
    lon: Tensor
        The longitude indices of the window. If the window has N x M grid cells,
        lon must have shape (N, M).
    nlat: int, default: 360
        The number of latitude indices of the underlying data.
        For 0.5° x 0.5° gridded data, nlat is 360 (the default).
    nlon: int, default: 720
        The number of longitude indices of the underlying data.
        For 0.5° x 0.5° gridded data, nlon is 720 (the default).


    Returns
    -------

    tuple[Tensor, Tensor]:
        The adjusted coordinate indices.
    """

    if not lat.shape == lon.shape:
        raise ValueError("Latitude and longitdue tensors must have the same shapes.")

    # Handle extension over the poles
    npole_cross = lat < 0
    spole_cross = lat > nlat - 1
    lat = torch.where(npole_cross, -lat, lat)
    lat = torch.where(spole_cross, 2 * nlat - 1 - lat, lat)
    lon = torch.where(npole_cross | spole_cross, lon + nlon // 2, lon)
    # Extend over the date line
    lon = lon % nlon

    return lat, lon
