from datetime import datetime

from xgboost import DMatrix
import dask
from deeprec.data.datamodule import DeepRecDataModule, Partition, Region, Scaler


class XGBoostDataModule(DeepRecDataModule):
    def __init__(
        self,
        data_dir: str,
        train_split: tuple[datetime, datetime],
        val_split: tuple[datetime, datetime],
        test_split: tuple[datetime, datetime],
        coverage: dict[Region, list[str]],
        target_var: str,
        scalar_input_vars: list[str],
        scale_method: Scaler | None = None,
    ) -> None:
        self.dtrain: DMatrix | None = None
        self.dval: DMatrix | None = None
        self.dtest: DMatrix | None = None
        self.dval: DMatrix | None = None
        self.dpredict: DMatrix | None = None

        super().__init__(
            data_dir=data_dir,
            train_split=train_split,
            val_split=val_split,
            test_split=test_split,
            coverage=coverage,
            target_var=target_var,
            scalar_input_vars=scalar_input_vars,
            matrix_input_vars=None,
            tensor_input_vars=None,
            scale_method=scale_method,
            space_window=1,
            time_window=1,
        )

    def setup(self, stage: str) -> None:
        """Creates XGBoost DMatrix objects of all inputs and
        target partitions in the provided stage."""
        match stage:
            case "fit":
                self.dtrain = self._create_dmatrix("train")
                self.dval = self._create_dmatrix("val")
            case "validate":
                self.dval = self._create_dmatrix("val")
            case "test":
                self.dtest = self._create_dmatrix("test")
            case "predict":
                self.dpredict = self._create_dmatrix("predict")
            case _:
                raise ValueError(f"Unknown stage {stage}.")

    def teardown(self, stage: str) -> None:
        """Releases memories occupied by the DMatrix objects at the end
        of the provided stage."""
        match stage:
            case "fit":
                self.dtrain = None
                self.dval = None
            case "validate":
                self.dval = None
            case "test":
                self.dtest = None
            case "predict":
                self.dpredict = None
            case _:
                raise ValueError(f"Unknown stage {stage}.")

    def _create_dmatrix(self, partition: Partition) -> DMatrix:
        inputs = getattr(self, f"inputs_{partition}")[self._scalar_input_vars]

        with dask.config.set(**{"array.slicing.split_large_chunks": False}):
            if not partition == "predict":
                target = (
                    getattr(self, f"target_{partition}")
                    .where(self._mask == 1)
                    .stack(sample=("time", "lat", "lon"))
                    .dropna("sample")
                )
                inputs = inputs.stack(sample=("time", "lat", "lon")).reindex_like(
                    target
                )
            else:
                target = None
                inputs = (
                    inputs.where(self._mask == 1)
                    .stack(sample=("time", "lat", "lon"))
                    .dropna("sample")
                )
            inputs = inputs.to_dataarray("feature").transpose("sample", "feature")

        return DMatrix(inputs, label=target, feature_names=self._scalar_input_vars)
