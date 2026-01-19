#!/usr/bin/env python3
"""
Filtra i ticket con Order Status = Paid e salva un CSV "VERIFIED".
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

READ_KWARGS = {
    "sep": ",",
    "quotechar": "\"",
    "encoding": "utf-8",
    "engine": "python",
    "dtype": str,
    "skip_blank_lines": True,
}

DEFAULT_INPUT = Path(__file__).resolve().parents[1] / "Documenti/Tickets/Attendee_List_Paid_19-01-26_12-15.csv"


def build_verified_path(input_path: Path) -> Path:
    stem = input_path.stem
    suffix = input_path.suffix or ".csv"
    prefix = "Attendee_List_Paid_"
    if stem.startswith(prefix):
        tail = stem[len(prefix) :]
        new_name = f"{prefix}VERIFIED__{tail}{suffix}"
    else:
        new_name = f"{stem}_VERIFIED{suffix}"
    return input_path.with_name(new_name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Filtra i ticket con Order Status = Paid")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="CSV dei ticket")
    args = parser.parse_args()

    input_path = args.input
    df = pd.read_csv(input_path, **READ_KWARGS)

    status_col = "Order Status"
    if status_col not in df.columns:
        raise KeyError(f"Colonna mancante: '{status_col}'")

    status_clean = df[status_col].fillna("").astype(str).str.strip().str.lower()
    paid = df[status_clean == "paid"].copy()

    output_path = build_verified_path(input_path)
    paid.to_csv(output_path, index=False, encoding="utf-8")

    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Righe totali: {len(df)}")
    print(f"Righe Paid: {len(paid)}")


if __name__ == "__main__":
    main()
