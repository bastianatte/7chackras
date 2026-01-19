#!/usr/bin/env python3
"""
Crea una versione "flat" di un CSV, sostituendo i newline nei campi con spazi.
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


def build_flat_path(input_path: Path) -> Path:
    stem = input_path.stem
    suffix = input_path.suffix or ".csv"
    return input_path.with_name(f"{stem}_FLAT{suffix}")


def flatten_cell(value: object) -> str:
    if value is None:
        return ""
    text = str(value)
    return " ".join(text.splitlines()).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Crea versione FLAT di un CSV")
    parser.add_argument("--input", type=Path, required=True, help="CSV di input")
    args = parser.parse_args()

    input_path = args.input
    df = pd.read_csv(input_path, **READ_KWARGS)
    flat = df.applymap(flatten_cell)

    output_path = build_flat_path(input_path)
    flat.to_csv(output_path, index=False, encoding="utf-8")

    print(f"Input: {input_path}")
    print(f"Output FLAT: {output_path}")
    print(f"Righe: {len(flat)}")


if __name__ == "__main__":
    main()
