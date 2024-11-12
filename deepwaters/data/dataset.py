import torch
from torch import Tensor
from torch.utils.data import Dataset


class DeepwatersDataset(Dataset):
    def __init__(
        self,
        lookup: Tensor,
        tensor_inputs: Tensor | None,
        matrix_inputs: Tensor | None,
        vector_inputs: Tensor | None,
        scalar_inputs: Tensor | None,
        target: Tensor | None,
        weight: Tensor | None,
        std: Tensor | None,
        space_window: int,
        time_window: int,
    ) -> None:
        super().__init__()

        # Infer spatial extension
        for inputs in [tensor_inputs, matrix_inputs, vector_inputs, scalar_inputs]:
            if inputs is not None:
                nlat = inputs.size(2)
                nlon = inputs.size(3)
                break
        else:  # No break block
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
        self.inp_scalar = scalar_inputs
        self.inp_vector = vector_inputs
        self.inp_matrix = matrix_inputs
        self.inp_tensor = tensor_inputs
        self.target = target
        self.weight = weight
        self.std = std
        self.lookup = lookup
        self.half_patchwidth = half_patchwidth
        self.past_steps = past_steps
        self.nlat = nlat
        self.nlon = nlon

        return

    def __getitem__(self, index) -> dict[str, Tensor | dict[str, Tensor]]:
        """If a target variable was provided, returns a tuple containing a dictionary of the inputs
        and the target. If no target was provided, only the input dictionary is returned.
        """
        time_idx, lat_idx, lon_idx = self.lookup[index]

        data = {}
        inputs = {}

        # Calculate extend of time and space windows
        time_min = time_idx - self.past_steps
        time_max = time_idx + 1

        lat_min = lat_idx - self.half_patchwidth
        lat_max = lat_idx + self.half_patchwidth + 1
        lon_min = lon_idx - self.half_patchwidth
        lon_max = lon_idx + self.half_patchwidth + 1

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

        # Create scalars
        if self.inp_scalar is not None:
            xs = self.inp_scalar[:, time_idx, lat_idx, lon_idx]
            inputs["scalar_inputs"] = xs

        # Create vectors, 1D time series (time,)
        if self.inp_vector is not None:
            xv = self.inp_vector[:, time_min:time_max, lat_idx, lon_idx]
            inputs["vector_inputs"] = xv

        # Create matrices, 2D patches (lat, lon)
        if self.inp_matrix is not None:
            # Flip latitude axis (+90° to -90°)
            xm = self.inp_matrix[:, time_idx, lats, lons].flip(1)
            inputs["matrix_inputs"] = xm

        # Create tensors, 3D cubes (time, lat, lon)
        if self.inp_tensor is not None:
            # Flip latitude axis (+90° to -90°)
            xt = self.inp_tensor[:, time_min:time_max, lats, lons].flip(2)
            inputs["tensor_inputs"] = xt

        data["inputs"] = inputs
        # Return target and additional arguments in train/val/test phase
        if self.target is not None:
            data["target"] = self.target[index]
            data["weight"] = self.weight[lat_idx]
            data["std"] = self.std[lat_idx, lon_idx]

        return data

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
