from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np


class DatasetMetadataBuilder:
    """Collects dataset metadata and writes it as a JSON sidecar."""

    def __init__(self, dataset_dir: str | Path, metadata_filename: str = "metadata.json"):
        self.dataset_dir = Path(dataset_dir)
        self.metadata_path = self.dataset_dir / metadata_filename
        self.metadata: dict[str, Any] = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "initialized",
        }

    def add_section(self, name: str, values: dict[str, Any]) -> None:
        self.metadata[name] = values

    def update_section(self, name: str, values: dict[str, Any]) -> None:
        section = self.metadata.setdefault(name, {})
        section.update(values)

    def add_dataset_info(
        self,
        *,
        dataset_name: str,
        hdf5_file: str,
        n_requested: int,
        n_sampled_parameters: int,
        n_saved: int | None,
        block_size: int,
        seed: int | None,
    ) -> None:
        self.add_section(
            "dataset",
            {
                "name": dataset_name,
                "directory": str(self.dataset_dir),
                "hdf5_file": hdf5_file,
                "metadata_file": self.metadata_path.name,
                "n_requested": n_requested,
                "n_sampled_parameters": n_sampled_parameters,
                "n_saved": n_saved,
                "block_size": block_size,
                "seed": seed,
            },
        )

    def add_generation_settings(
        self,
        *,
        sampler: str,
        filter_with_snr: bool,
        snr_min: float,
        snr_max: float,
        difficulty_factor: float,
        dt: float,
        tobs_years: float,
        tobs_seconds: float,
        f0_center: float,
        n_frequency_bins: int,
    ) -> None:
        self.add_section(
            "generation",
            {
                "sampler": sampler,
                "filter_with_snr": filter_with_snr,
                "snr_min": snr_min if filter_with_snr else None,
                "snr_max": snr_max if filter_with_snr else None,
                "difficulty_factor": difficulty_factor,
                "dt": dt,
                "tobs_years": tobs_years,
                "tobs_seconds": tobs_seconds,
                "f0_center": f0_center,
                "N": n_frequency_bins,
            },
        )

    def add_parameter_config(
        self,
        *,
        fixed_params: dict[str, Any],
        sampled_ranges: dict[str, Any],
        param_order: list[str],
        computed_f0_range: list[float],
    ) -> None:
        effective_ranges = dict(sampled_ranges)
        effective_ranges["f0"] = computed_f0_range
        self.add_section(
            "parameters",
            {
                "fixed_params": fixed_params,
                "sampled_ranges": effective_ranges,
                "param_order": param_order,
            },
        )

    def add_config_snapshot(self, values: dict[str, Any]) -> None:
        self.add_section("config_snapshot", values)

    def add_hdf5_structure(self, hdf5_path: str | Path) -> None:
        hdf5_path = Path(hdf5_path)
        structure: dict[str, Any] = {}
        with h5py.File(hdf5_path, "r") as h5_file:
            h5_file.visititems(
                lambda name, obj: structure.__setitem__(name, self._describe_hdf5_object(obj))
            )
        self.add_section("hdf5_structure", structure)

    def add_stats_from_hdf5(
        self,
        hdf5_path: str | Path,
        *,
        snr_bounds: tuple[float, float] | None = None,
    ) -> None:
        with h5py.File(hdf5_path, "r") as h5_file:
            params_group = h5_file.get("params")
            if params_group is None:
                return

            if "snr" in params_group:
                snr = np.asarray(params_group["snr"][:])
                snr_stats = self._array_stats(snr)
                if snr_bounds is not None:
                    lower, upper = snr_bounds
                    snr_stats[f"fraction_between_{self._label_number(lower)}_{self._label_number(upper)}"] = float(
                        np.mean((snr >= lower) & (snr <= upper))
                    )
                    snr_stats["histogram"] = self._histogram(
                        snr,
                        lower=lower,
                        upper=upper,
                        bin_width=5.0,
                    )
                self.add_section("snr_stats", snr_stats)

            param_stats = {}
            for name, dataset in params_group.items():
                if name == "snr":
                    continue
                param_stats[name] = self._array_stats(np.asarray(dataset[:]))
            self.add_section("param_stats", param_stats)

    def mark_complete(self) -> None:
        self.metadata["status"] = "complete"
        self.metadata["completed_at"] = datetime.now(timezone.utc).isoformat()

    def mark_failed(self, error: BaseException) -> None:
        self.metadata["status"] = "failed"
        self.metadata["failed_at"] = datetime.now(timezone.utc).isoformat()
        self.metadata["error"] = {
            "type": type(error).__name__,
            "message": str(error),
        }

    def generate_metadata(self) -> Path:
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        with self.metadata_path.open("w", encoding="utf-8") as stream:
            json.dump(self._to_jsonable(self.metadata), stream, indent=2, sort_keys=True)
            stream.write("\n")
        return self.metadata_path

    @staticmethod
    def _array_stats(values: np.ndarray) -> dict[str, Any]:
        values = np.asarray(values)
        if values.size == 0:
            return {"count": 0}

        percentiles = np.percentile(values, [1, 5, 25, 50, 75, 95, 99])
        return {
            "count": int(values.size),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "std": float(np.std(values)),
            "percentiles": {
                "p01": float(percentiles[0]),
                "p05": float(percentiles[1]),
                "p25": float(percentiles[2]),
                "p50": float(percentiles[3]),
                "p75": float(percentiles[4]),
                "p95": float(percentiles[5]),
                "p99": float(percentiles[6]),
            },
        }

    @classmethod
    def _histogram(
        cls,
        values: np.ndarray,
        *,
        lower: float,
        upper: float,
        bin_width: float,
    ) -> dict[str, Any]:
        values = np.asarray(values)
        if upper <= lower or bin_width <= 0:
            return {
                "bin_width": float(bin_width),
                "bounds": [float(lower), float(upper)],
                "count": 0,
                "bins": [],
            }

        edges = np.arange(lower, upper, bin_width, dtype=float)
        if edges.size == 0 or edges[0] != lower:
            edges = np.insert(edges, 0, lower)
        if not np.isclose(edges[-1], upper):
            edges = np.append(edges, upper)

        counts, _ = np.histogram(values, bins=edges)
        bins = []
        for index, count in enumerate(counts):
            left = float(edges[index])
            right = float(edges[index + 1])
            is_last = index == len(counts) - 1
            bins.append(
                {
                    "interval": cls._format_interval(left, right, include_upper=is_last),
                    "lower": left,
                    "upper": right,
                    "count": int(count),
                }
            )

        return {
            "bin_width": float(bin_width),
            "bounds": [float(lower), float(upper)],
            "count": int(np.sum(counts)),
            "bins": bins,
        }

    @staticmethod
    def _describe_hdf5_object(obj: h5py.Dataset | h5py.Group) -> dict[str, Any]:
        if isinstance(obj, h5py.Dataset):
            return {
                "type": "dataset",
                "shape": list(obj.shape),
                "dtype": str(obj.dtype),
                "compression": obj.compression,
                "compression_opts": obj.compression_opts,
            }
        return {"type": "group"}

    @staticmethod
    def _label_number(value: float) -> str:
        label = f"{value:g}".replace("-", "m").replace(".", "p")
        return label

    @staticmethod
    def _format_interval(lower: float, upper: float, *, include_upper: bool) -> str:
        closing = "]" if include_upper else ")"
        return f"[{lower:g}, {upper:g}{closing}"

    @classmethod
    def _to_jsonable(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): cls._to_jsonable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._to_jsonable(item) for item in value]
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.ndarray):
            return cls._to_jsonable(value.tolist())
        if isinstance(value, np.generic):
            return value.item()
        return value
