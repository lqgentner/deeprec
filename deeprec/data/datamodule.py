from pathlib import Path
from typing import Literal

import dask
import lightning as L
import numpy as np
import pandas as pd
from tabulate import tabulate
import torch
from torch import Tensor
from torch.utils.data import DataLoader
import xarray as xr

import deeprec  # noqa
from deeprec.data.dataset import DeepRecDataset, DeepRecTensors
from deeprec.preprocessing.scalers import AbstractScaler, RobustScaler, StandardScaler
from deeprec.regions import select_basins, select_countries
from deeprec.utils import ROOT_DIR

Region = Literal["basins", "countries"]
Scaler = Literal["standard", "robust"]
Partition = Literal["train", "val", "test", "predict"]


class DeepRecDataModule(L.LightningDataModule):
    """
    A data module for handling, preprocessing, and partitioning of environmental data for
    deep learning models in PyTorch, utilizing the PyTorch Lightning framework. This module
    focuses on the integration of spatial-temporal data from different sources, applying
    data masking based on specified geographic regions, scaling input variables, and providing
    data loaders for training, validation, testing, and prediction stages.

    The module supports multiple types of input variables with the following dimensions:
    - Scalar (0D)
    - Vector (1D: time)
    - Matrix (2D: latitude x longitude)
    - Tensor (3D: time x latitude x longitude)

    The module also allows for the selection of specific geographic regions
    (basins, countries) for analysis and utilizes robust or standard scaling methods
    to normalize the data.
    """

    def __init__(
        self,
        data_dir: str,
        target_var: str,
        coverage: dict[Region, list[str]],
        scalar_input_vars: list[str] | None = None,
        vector_input_vars: list[str] | None = None,
        matrix_input_vars: list[str] | None = None,
        tensor_input_vars: list[str] | None = None,
        train_split: tuple[str, str] | None = None,
        val_split: tuple[str, str] | None = None,
        test_split: tuple[str, str] | None = None,
        scale_method: Scaler | None = None,
        space_window: int = 25,
        time_window: int = 1,
        **dl_kwargs,
    ) -> None:
        """
        Initializes the DeepRecDataModule.

        Parameters
        ----------
        data_dir: str:
            Path to the directory containing the Zarr datasets for inputs and targets.
        train_split: tuple[str, str]
            Start and end dates for the training dataset partition. If not provided,
            train on all time steps which are not in the validation or test partition.
            Defaults to None.
        val_split: tuple[str, str], optional
            Start and end dates for the validation dataset partition. Defaults to None.
        test_split: tuple[str, str], optional
            Start and end dates for the test dataset partition. Defaults to None.
        coverage: dict[Region, list[str]]
            A dictionary specifying the geographic regions to include
            in the analysis. The keys are region types ('basins', 'countries'), and
            the values are lists of region names.
        target_var: str
            The name of the target variable in the dataset.
        scalar_input_vars: list[str], optional
            List of names of 0D input variables to include. Defaults to None.
        vector_input_vars: list[str], optional
            List of names of 1D input variables (time,) to include. Defaults to None
        matrix_input_vars: list[str], optional
            List of names of 2D input variables (lat, lon) to include.
            Defaults to None.
        tensor_input_vars: list[str], optional
            List of names of 3D input variables (time, lat, lon) to include.
            Defaults to None.
        scale_method: Scaler, optional
            The method used for scaling the input variables. Can be either
            'standard' or 'robust'. Defaults to None.
        space_window: int
            The size of the spatial window for data extraction.
        time_window: int
            The size of the temporal window for data extraction. Must be 1 if
            tensor_input_vars is not provided.
        dl_kwargs
            Additional keyword arguments to be passed to the PyTorch DataLoader.
        """
        super().__init__()

        # Load data
        data_path = Path(data_dir)
        if not data_path.is_absolute():
            data_path = ROOT_DIR / data_path
        inputs_orig = xr.open_zarr(data_path / "inputs.zarr")
        targets_orig = xr.open_zarr(data_path / "targets.zarr")

        # Select inputs and target
        input_vars: list[str] = []
        for inp in [
            tensor_input_vars,
            matrix_input_vars,
            vector_input_vars,
            scalar_input_vars,
        ]:
            input_vars = input_vars + inp if inp is not None else input_vars

        if not isinstance(target_var, str):
            raise ValueError(
                "target_var must be a string (only a single target allowed)."
            )

        inputs = inputs_orig[input_vars]
        # During pre-training the target might be part of the input store
        try:
            target = targets_orig[target_var]
        except KeyError:
            target = inputs_orig[target_var]

        # Remove target time steps where it is not available
        # (Mascons differ in available time steps)
        target = target.dropna("time", how="all")

        # Remove input and target time steps where one or more inputs is not available
        time_notnull = inputs.dr.time_notnull()
        inputs = inputs.where(time_notnull, drop=True)
        target = target.where(time_notnull, drop=True)

        # Verify data
        self._verify_dims(target, ["time", "lat", "lon"], "target")
        self._verify_dims(inputs, ["time", "lat", "lon"], "inputs")
        self._verify_time_freq(inputs, "inputs")

        if tensor_input_vars is None and time_window != 1:
            raise ValueError(
                "If no tensor_input_vars are provided (single-time-step mode), time_window must be 1."
            )
        elif tensor_input_vars is not None and time_window == 1:
            raise ValueError(
                "If tensor_input_vars are provided (multi-time-step mode), time_window must not be 1. "
            )

        # Get time steps of partitions
        time = inputs.time

        if train_split is not None:
            time_train = time.sel(time=slice(*train_split))
        else:
            # If training time is not provided, use all overlapping input and target time steps
            # as training steps (minus the time steps required for warmup)
            inp_time = inputs.get_index("time")
            tgt_time = target.get_index("time")
            start = inp_time[0] if inp_time[0] > tgt_time[0] else tgt_time[0]
            end = inp_time[-1] if inp_time[-1] < tgt_time[-1] else tgt_time[-1]
            time_train = time.sel(time=slice(start, end))[(time_window - 1) :]

        if val_split is not None:
            time_val = time.sel(time=slice(*val_split))
            # Ensure val time steps are not part of training set
            time_train = time_train.drop_sel(time=time_val, errors="ignore")
        else:
            time_val = None

        if test_split is not None:
            time_test = time.sel(time=slice(*test_split))
            # Ensure test time steps are not part of training set
            time_train = time_train.drop_sel(time=time_test, errors="ignore")
        else:
            time_test = None

        # Create coverage mask
        # Multiple specified masks will be AND combined
        mask = xr.ones_like(target.isel(time=0))
        for region_type, region_list in coverage.items():
            if isinstance(region_list, str):
                region_list = [region_list]
            match region_type:
                case "countries":
                    mask = select_countries(mask, region_list, return_region=False)
                case "basins":
                    mask = select_basins(mask, region_list, return_region=False)
                case "data_vars":
                    # Use data variable in inputs/targets as mask
                    for data_var in region_list:
                        # Try to locate specified data variable
                        if data_var in inputs_orig.data_vars:
                            dvar_mask = inputs_orig[data_var]
                        elif data_var in targets_orig.data_vars:
                            dvar_mask = targets_orig[data_var]
                        else:
                            raise ValueError(
                                f"Specified mask {data_var} could not be located in data variables."
                            )
                        # Combine with existing mask (where replaces all zeros with NAs)
                        mask *= dvar_mask.where(dvar_mask)
                case _:
                    raise ValueError(f"Unknown region type {region_type} in coverage.")

        # Apply mask to target
        target = target.where(mask == 1)

        # Scale inputs
        scaler: AbstractScaler | None
        match scale_method:
            case "standard":
                scaler = StandardScaler()
            case "robust":
                scaler = RobustScaler()
            case None:
                scaler = None
            case _:
                raise ValueError(f"Unknown scale_method {scale_method}.")

        # Scale according to train set of selected area
        if scaler is not None:
            inputs_train = inputs.where(inputs.time.isin(time_train), drop=True).where(
                mask == 1
            )
            scaler.fit(inputs_train.compute())
            inputs = scaler.transform(inputs)
        # Fill missing values in inputs with zeros (after scaling!)
        inputs = inputs.fillna(0)

        # Assign attributes
        # Xarray/Dask data arrays
        self._inputs = inputs
        self._target = target
        self._mask = mask
        # Config hyperparameters
        self._time_train = time_train
        self._time_val = time_val
        self._time_test = time_test
        self._target_var = target_var
        self._scalar_input_vars = scalar_input_vars
        self._vector_input_vars = vector_input_vars
        self._matrix_input_vars = matrix_input_vars
        self._tensor_input_vars = tensor_input_vars
        self._space_window = space_window
        self._time_window = time_window
        self._dl_kwargs = dl_kwargs
        # Containers of PyTorch tensors, assigned in _prepare_tensors
        self._train_tensors: DeepRecTensors | None = None
        self._val_tensors: DeepRecTensors | None = None
        self._test_tensors: DeepRecTensors | None = None
        self._predict_tensors: DeepRecTensors | None = None

    def setup(self, stage: str) -> None:
        """Creates dictionaries with PyTorch tensors of all inputs,
        targets, and lookup tables required by the DataLoader(s)
        in the provided stage."""
        match stage:
            case "fit":
                self._train_tensors = self._prepare_tensors("train")
                self._val_tensors = self._prepare_tensors("val")
            case "validate":
                self._val_tensors = self._prepare_tensors("val")
            case "test":
                self._test_tensors = self._prepare_tensors("test")
            case "predict":
                self._predict_tensors = self._prepare_tensors("predict")
            case _:
                raise ValueError(f"Unknown stage {stage}.")

    def teardown(self, stage: str) -> None:
        """Releases memories occupied by the tensors at the end
        of the provided stage."""
        match stage:
            case "fit":
                self._train_tensors = None
                self._val_tensors = None
            case "validate":
                self._val_tensors = None
            case "test":
                self._test_tensors = None
            case "predict":
                self._predict_tensors = None
            case _:
                raise ValueError(f"Unknown stage {stage}.")

    def train_dataloader(self) -> DataLoader:
        if self._train_tensors is None:
            raise TypeError(
                "Training tensors have not been prepared. Call setup('fit') first."
            )
        return DataLoader(
            DeepRecDataset(
                self._train_tensors,
                space_window=self._space_window,
                time_window=self._time_window,
            ),
            shuffle=True,
            **self._dl_kwargs,
        )

    def val_dataloader(self) -> DataLoader:
        if self._val_tensors is None:
            raise TypeError(
                "Validation tensors have not been prepared. Call setup('fit') or setup('validate') first."
            )
        return DataLoader(
            DeepRecDataset(
                self._val_tensors,
                space_window=self._space_window,
                time_window=self._time_window,
            ),
            shuffle=False,
            **self._dl_kwargs,
        )

    def test_dataloader(self) -> DataLoader:
        if self._test_tensors is None:
            raise TypeError(
                "Testing tensors have not been prepared. Call setup('test') first."
            )
        return DataLoader(
            DeepRecDataset(
                self._test_tensors,
                space_window=self._space_window,
                time_window=self._time_window,
            ),
            shuffle=False,
            **self._dl_kwargs,
        )

    def predict_dataloader(self) -> DataLoader:
        if self._predict_tensors is None:
            raise TypeError(
                "Prediction tensors have not been prepared. Call setup('test') first."
            )
        return DataLoader(
            DeepRecDataset(
                self._predict_tensors,
                space_window=self._space_window,
                time_window=self._time_window,
            ),
            shuffle=False,
            **self._dl_kwargs,
        )

    def _time_notnull_any(self, obj: xr.Dataset) -> xr.DataArray:
        """Returns a boolean array which indicates all time indices where
        any data variable of the dataset is not available.
        Checks for every time step if all values across lat-lon are NA for
        at least one data variable.
        """
        time_na = (
            obj.to_dataarray()
            .isnull()
            .all(dim=["lat", "lon"])
            .any(dim="variable")
            .compute()
        )
        return ~time_na

    def _verify_dims(
        self, obj: xr.DataArray | xr.Dataset, dims: list, name: str
    ) -> None:
        """Equality check between object dimensions and provided dimensions,
        including order."""
        obj_dims = list(obj.dims)
        if obj_dims != dims:
            raise ValueError(
                f"Dimensions of {name} must be {dims}, but received {obj_dims}."
            )

    def _verify_time_freq(self, obj: xr.DataArray, name: str) -> None:
        """Check that time indices are un-interupted and monothonically increasing."""
        FREQ = "MS"
        if pd.infer_freq(obj.get_index("time")) != FREQ:
            raise ValueError(f"Time index frequency of {name} must be {FREQ}.")

    @property
    def variables(self) -> str:
        """Returns a bullet list of all input and target variables."""
        bullet = "\n- "

        variables = "Target: "
        variables += self._target_var
        if self._tensor_input_vars is not None:
            variables += "\nTensor inputs:" + bullet
            variables += bullet.join(self._tensor_input_vars)
        if self._matrix_input_vars is not None:
            variables += "\nMatrix inputs:" + bullet
            variables += bullet.join(self._matrix_input_vars)
        if self._vector_input_vars is not None:
            variables += "\nVector inputs:" + bullet
            variables += bullet.join(self._vector_input_vars)
        if self._scalar_input_vars is not None:
            variables += "\nScalar inputs:" + bullet
            variables += bullet.join(self._scalar_input_vars)

        return variables

    @property
    def split_stats(self) -> str:
        """Returns a table which indicates the absolute and relative number of time steps
        in inputs and target"""

        predict = self.inputs_predict
        target = self._target
        test = self.target_test
        val = self.target_val
        train = self.target_train

        items = [predict, target, test, val, train]
        names = [
            "Prediction",
            "Target",
            "├─ Test",
            "├─ Val",
            "└─ Train",
        ]
        lengths = [len(item.time) for item in items]
        starts = [
            item.get_index("time")[0].date() if len(item.get_index("time")) > 0 else "-"
            for item in items
        ]
        ends = [
            item.get_index("time")[-1].date()
            if len(item.get_index("time")) > 0
            else "-"
            for item in items
        ]
        # Calc rel size of data sets (inputs/train)
        merged = xr.merge([predict, target.rename("target")])
        share_dsets = [len(dset.time) / len(merged.time) * 100 for dset in items[:2]]
        # Calc rel size of partitions relative to target]
        share_part = [len(part.time) / len(target.time) * 100 for part in items[2:]]
        share = share_dsets + share_part

        table = {
            "Item": names,
            "Start": starts,
            "End": ends,
            "Time steps": lengths,
            "Share (%)": share,
        }
        return tabulate(table, headers="keys", floatfmt=".1f", intfmt=",")

    def predictions_to_xarray(
        self,
        pred: Tensor | np.ndarray | list[Tensor],
        partition: Partition = "predict",
        name: str | None = None,
        restore_global_extend: bool = True,
    ) -> xr.DataArray | xr.Dataset:
        """Returns the model predictions as a Xarray DataArray"""

        # Prediction batches are returned as list
        if isinstance(pred, list):
            pred = torch.cat(pred)
        # Convert PyTorch Tensor to Numpy array
        if isinstance(pred, Tensor):
            pred = pred.cpu().numpy()

        # Create coordinate template
        with dask.config.set({"array.slicing.split_large_chunks": False}):
            if partition == "predict":
                # Extract DataArray from inputs with length of predictions
                ones = np.ones(
                    tuple(self.inputs_predict.sizes.values()), dtype="float32"
                )
                template = (
                    xr.DataArray(ones, coords=self.inputs_predict.coords)
                    .where(self._mask == 1)
                    .dr.stack_spacetime()
                    .chunk({"sample": -1})
                )
            else:
                # Use target as template
                template = (
                    getattr(self, f"target_{partition}")
                    .dr.stack_spacetime()
                    .chunk({"sample": -1})
                )
            # Insert predictions into data array
            match pred.ndim:
                case 1:
                    # Tensor contains only predictions - return data array
                    pred_name = f"pred_{name}" if name else "pred"
                    xr_pred = (
                        template.copy(data=pred)
                        .rename(pred_name)
                        .dr.unstack_spacetime()
                    )
                    xr_pred.attrs = {
                        "long_name": "Predicted Terrestrial Water Storage Anomaly",
                        "standard_name": "twsa_pred",
                        "units": "mm",
                    }
                case 2:
                    # Tensor contains prediction and uncertainty - return dataset
                    pred_name = f"pred_{name}" if name else "pred"
                    uncert_name = f"uncertainty_{name}" if name else "uncertainty"

                    da_pred = template.copy(data=pred[:, 0]).rename(pred_name)
                    da_uncert = template.copy(data=pred[:, 1]).rename(uncert_name)

                    da_pred.attrs = {
                        "long_name": "Predicted Terrestrial Water Storage Anomaly",
                        "standard_name": "twsa",
                        "units": "mm",
                    }
                    da_uncert.attrs = {
                        "long_name": "Predicted Scale Parameter",
                        "standard_name": "scale_parameter",
                        "units": "mm",
                    }

                    xr_pred = xr.merge([da_pred, da_uncert]).dr.unstack_spacetime()
                case d:
                    raise ValueError(f"Tensors with {d} dimensions not supported.")
            # Restore global extend (otherwise, some empty lat/lon col/rows will be missing)
            if restore_global_extend:
                xr_pred = xr_pred.reindex_like(self._inputs.drop_dims("time"))

        return xr_pred

    @property
    def target(self) -> xr.DataArray:
        return self._target

    @property
    def target_train(self) -> xr.DataArray:
        return self._target.where(self._target.time.isin(self._time_train), drop=True)

    @property
    def target_val(self) -> xr.DataArray:
        return self._target.where(self._target.time.isin(self._time_val), drop=True)

    @property
    def target_test(self) -> xr.DataArray:
        return self._target.where(self._target.time.isin(self._time_test), drop=True)

    @property
    def inputs(self) -> xr.Dataset:
        return self._inputs

    @property
    def inputs_train(self) -> xr.Dataset:
        return self._inputs.where(self._inputs.time.isin(self._time_train), drop=True)

    @property
    def inputs_val(self) -> xr.Dataset:
        return self._inputs.where(self._inputs.time.isin(self._time_val), drop=True)

    @property
    def inputs_test(self) -> xr.Dataset:
        return self._inputs.where(self._inputs.time.isin(self._time_test), drop=True)

    @property
    def inputs_predict(self) -> xr.Dataset:
        return self._inputs.isel(time=slice(self._time_window - 1, None))

    def _create_lookup(
        self,
        inputs: xr.DataArray,
        target: xr.DataArray | None = None,
    ) -> np.ndarray:
        """Creates a index lookup table to map from target sample index to integer-based
        input (time, lat, lon) indices."""

        # Extract coordinates from inputs
        # (Create a dummy DataArray with the same shape as inputs)
        inputs = xr.zeros_like(inputs.isel(feature=0).squeeze())

        # Take coordinates from target, if provided
        # (The target has temporal gaps and is shorter)
        # If target is not provided, create a 'fake target' data array
        # which starts 'time_window' steps after the inputs to account for the
        # multi-time-step input
        if target is None:
            target = inputs[(self._time_window - 1) :].where(self._mask == 1)

        # Merge inputs (don't change time steps)
        merged = xr.merge(
            [inputs.rename("inputs"), target.rename("target")], join="left"
        )

        # Assign integer indices
        merged = merged.assign_coords(
            time=np.arange(len(merged.time)),
            lat=np.arange(len(merged.lat)),
            lon=np.arange(len(merged.lon)),
        )
        # Create lookup for all samples where target is not NA
        lookup = (
            merged.target.dr.stack_spacetime().get_index("sample").to_frame().to_numpy()
        )

        # Verify lookup validity: No out of bounds time values allowed
        time_idxs = lookup[:, 0]
        if (time_idxs - (self._time_window - 1) < 0).any():
            raise RuntimeError(
                "Lookup would result in negative time indices. This should not happen by design."
            )

        return lookup

    def _prepare_tensors(self, partition: Partition) -> DeepRecTensors:
        print(f"DataModule: Preparing {partition} tensors...")
        # Partition inputs and target
        match partition:
            case "train" | "val" | "test":
                time = getattr(self, f"_time_{partition}")

                if time is None:
                    raise ValueError(
                        f"Temporal extend of partition {partition} not specified."
                    )

                start = pd.Timestamp(time[0].values)
                end = pd.Timestamp(time[-1].values)
                # Earlier start of inputs in case of multi-time-step mode
                start_ext = start - pd.DateOffset(months=(self._time_window - 1))
                target = self._target.where(self._target.time.isin(time), drop=True)
                # Slicing makes sure inputs are continuous
                inputs = self._inputs.sel(time=slice(start_ext, end))
            case "predict":
                # Take full temporal extend of inputs
                inputs = self._inputs
                target = None
                weight = None
            case _:
                raise ValueError(f"Unknown partiton {partition}.")

        # Convert inputs to DataArray
        inputs = inputs.to_dataarray("feature")

        scalar_inputs: Tensor | None = None
        vector_inputs: Tensor | None = None
        matrix_inputs: Tensor | None = None
        tensor_inputs: Tensor | None = None

        # Create lookup table
        lookup = torch.as_tensor(self._create_lookup(inputs, target), dtype=torch.int16)

        # Scalar inputs
        if self._scalar_input_vars is not None:
            scalar_inputs = torch.as_tensor(
                inputs.sel(feature=self._scalar_input_vars).values, dtype=torch.float32
            )

        # Vector inputs
        if self._vector_input_vars is not None:
            vector_inputs = torch.as_tensor(
                inputs.sel(feature=self._vector_input_vars).values, dtype=torch.float32
            )

        # Matrix inputs
        if self._matrix_input_vars is not None:
            matrix_inputs = torch.as_tensor(
                inputs.sel(feature=self._matrix_input_vars).values, dtype=torch.float32
            )

        # Tensor inputs
        if self._tensor_input_vars is not None:
            tensor_inputs = torch.as_tensor(
                inputs.sel(feature=self._tensor_input_vars).values, dtype=torch.float32
            )

        # Target
        if target is not None:
            # Calculate latitude weights
            weight = np.cos(np.deg2rad(target.lat.values))
            weight = torch.as_tensor(weight, dtype=torch.float32)

            # Reshape target dimensions (time, lat, lon) to 1D
            target = target.dr.stack_spacetime()
            target = torch.as_tensor(target.values, dtype=torch.float32)

            assert len(target) == len(lookup)

        return DeepRecTensors(
            lookup=lookup,
            tensor_inputs=tensor_inputs,
            matrix_inputs=matrix_inputs,
            vector_inputs=vector_inputs,
            scalar_inputs=scalar_inputs,
            target=target,
            weight=weight,
        )
