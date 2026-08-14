from __future__ import annotations
import json
import pandas as pd
import pytest
from pathlib import Path
from client.engine.data_source import ProdDataSource, NoNewDataError


def _make_csv(path: Path, rows: int = 3, prefix: str = "filtration_20260814_120000") -> Path:
    df = pd.DataFrame({
        "time_min": list(range(rows)),
        "flux_lmh": [100.0 - i for i in range(rows)],
        "tmp_bar":  [1.0 + 0.01 * i for i in range(rows)],
    })
    p = path / f"{prefix}.csv"
    df.to_csv(p, index=False)
    return p


def test_no_files_raises(tmp_path: Path) -> None:
    ds = ProdDataSource(str(tmp_path))
    with pytest.raises(NoNewDataError):
        ds.get_data()


def test_reads_single_csv(tmp_path: Path) -> None:
    _make_csv(tmp_path)
    ds = ProdDataSource(str(tmp_path))
    time, flux, tmp = ds.get_data()
    assert len(time) == 3
    assert all(flux > 0)


def test_processed_file_skipped(tmp_path: Path) -> None:
    _make_csv(tmp_path)
    ds = ProdDataSource(str(tmp_path))
    ds.get_data()                   # marks file as processed
    with pytest.raises(NoNewDataError):
        ds.get_data()               # same file — no new data


def test_concatenates_multiple_new_csvs(tmp_path: Path) -> None:
    _make_csv(tmp_path, rows=3, prefix="filtration_20260814_120000")
    _make_csv(tmp_path, rows=5, prefix="filtration_20260814_130000")
    ds = ProdDataSource(str(tmp_path))
    time, flux, tmp = ds.get_data()
    assert len(time) == 8           # 3 + 5 rows concatenated


def test_sidecar_persists_across_instances(tmp_path: Path) -> None:
    _make_csv(tmp_path)
    ds1 = ProdDataSource(str(tmp_path))
    ds1.get_data()                  # processes and writes sidecar
    ds2 = ProdDataSource(str(tmp_path))   # new instance reads sidecar
    with pytest.raises(NoNewDataError):
        ds2.get_data()


def test_has_new_data_false_when_empty(tmp_path: Path) -> None:
    ds = ProdDataSource(str(tmp_path))
    assert not ds.has_new_data()


def test_has_new_data_true_after_csv_added(tmp_path: Path) -> None:
    ds = ProdDataSource(str(tmp_path))
    _make_csv(tmp_path)
    assert ds.has_new_data()


def test_has_new_data_false_after_processed(tmp_path: Path) -> None:
    _make_csv(tmp_path)
    ds = ProdDataSource(str(tmp_path))
    ds.get_data()
    assert not ds.has_new_data()


def test_sidecar_written_atomically(tmp_path: Path) -> None:
    _make_csv(tmp_path)
    ds = ProdDataSource(str(tmp_path))
    ds.get_data()
    sidecar = tmp_path / ".processed.json"
    assert sidecar.exists()
    data = json.loads(sidecar.read_text())
    assert isinstance(data, list)
    assert any("filtration_" in name for name in data)


def test_ignores_non_filtration_csvs(tmp_path: Path) -> None:
    df = pd.DataFrame({"time_min": [0], "flux_lmh": [100.0], "tmp_bar": [1.0]})
    df.to_csv(tmp_path / "other_data.csv", index=False)
    ds = ProdDataSource(str(tmp_path))
    with pytest.raises(NoNewDataError):
        ds.get_data()


def test_missing_columns_raises_value_error(tmp_path: Path) -> None:
    # Create a filtration CSV missing required columns
    df = pd.DataFrame({"time_min": [0, 1, 2], "flux_lmh": [100.0, 99.0, 98.0]})
    (tmp_path / "filtration_20260814_140000.csv").write_text(df.to_csv(index=False))
    ds = ProdDataSource(str(tmp_path))
    with pytest.raises(ValueError, match="missing columns"):
        ds.get_data()
