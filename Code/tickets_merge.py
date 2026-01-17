#!/usr/bin/env python3
"""
Unisce i due export Tickera in un unico CSV.
Default:
- early/phase0: Documenti/Libro_Soci_2025-2026/ListaTicketVenduti_27Nov25.csv
- phase1/christmas/ambassador: Documenti/Libro_Soci_2025-2026/Attendee_List_19.12.25_phase1+Chr+ambassPSYKY.csv
- output: Documenti/Libro_Soci_2025-2026/merged_19.12.25.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import pandas as pd

READ_KWARGS: Dict[str, object] = {
    "sep": ",",
    "quotechar": '"',
    "encoding": "utf-8",
    "engine": "python",
    "dtype": str,
    "skip_blank_lines": True,
}

DEFAULT_LEFT = Path(__file__).resolve().parents[1] / "Documenti/Libro_Soci_2025-2026/ListaTicketVenduti_27Nov25.csv"
DEFAULT_RIGHT = Path(__file__).resolve().parents[1] / "Documenti/Libro_Soci_2025-2026/Attendee_List_19.12.25_phase1+Chr+ambassPSYKY.csv"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "Documenti/Libro_Soci_2025-2026/merged_19.12.25.csv"


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, **READ_KWARGS)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge export Tickera in un unico CSV")
    parser.add_argument("--left", type=Path, default=DEFAULT_LEFT, help="CSV early/phase0 (ListaTicketVenduti_27Nov25.csv)")
    parser.add_argument("--right", type=Path, default=DEFAULT_RIGHT, help="CSV phase1+christmas+ambassador (Attendee_List_19.12.25_phase1+Chr+ambassPSYKY.csv)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Percorso CSV di output")
    args = parser.parse_args()

    df_left = load_csv(args.left)
    df_right = load_csv(args.right)

    # Uniforma le colonne: prende l'unione, riordina con le colonne del primo file poi le nuove.
    all_cols = list(dict.fromkeys(list(df_left.columns) + list(df_right.columns)))
    merged = pd.concat(
        [
            df_left.reindex(columns=all_cols),
            df_right.reindex(columns=all_cols),
        ],
        ignore_index=True,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False, encoding="utf-8")

    print(f"Merged rows: {len(merged)}")
    print(f"Colonne: {len(all_cols)}")
    print(f"Salvato in: {args.output}")


if __name__ == "__main__":
    main()
