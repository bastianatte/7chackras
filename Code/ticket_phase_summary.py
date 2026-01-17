#!/usr/bin/env python3
"""
Conteggia i ticket venduti per fase (early, phase 0, phase 1, christmas, ambassador).
Può lavorare con un unico CSV (--single-file) oppure con due export separati
(--early-phase0 e --phase1-other). Filtra solo i ticket con Order Status = Paid.
Salva un CSV di riepilogo e stampa i conteggi a console.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Iterable

import pandas as pd

READ_KWARGS: Dict[str, object] = {
    "sep": ",",
    "quotechar": '"',
    "encoding": "utf-8",
    "engine": "python",
    "dtype": str,
    "skip_blank_lines": True,
}

DEFAULT_EARLY_PHASE0 = Path(__file__).resolve().parents[1] / "Documenti/Libro_Soci_2025-2026/ListaTicketVenduti_27Nov25.csv"
DEFAULT_PHASE1_OTHER = Path(__file__).resolve().parents[1] / "Documenti/Libro_Soci_2025-2026/Attendee_List_19.12.25_phase1+Chr+ambassPSYKY.csv"
DEFAULT_SINGLE = Path(__file__).resolve().parents[1] / "Documenti/Libro_Soci_2025-2026/merged_19.12.25.csv"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "Documenti/ticket_phase_summary.csv"

TICKET_TYPE_COLS = ["Ticket Type"]
ORDER_STATUS_COLS = ["Order Status"]


def find_column(df: pd.DataFrame, candidates: Iterable[str], required: bool = True) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    if required:
        raise KeyError(f"Colonna mancante: una tra {candidates}")
    return ""


def slugify_ticket(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^0-9a-z]+", "_", text)
    text = re.sub(r"__+", "_", text).strip("_")
    return text[:80]


def classify_ticket(text: str) -> tuple[str, str, bool]:
    """
    Ritorna (primary_phase, variant_slug, is_ambassador).
    - primary_phase: early_bird, phase_0, phase_1, christmas_bundle, ambassador, unknown
    - variant_slug: slug completo del Ticket Type (per distinguere sottospecie)
    - is_ambassador: True se nel testo compare 'ambassador'/'ambass'
    """
    if not text:
        return "unknown", "unknown", False
    s = text.lower()
    s_space = re.sub(r"\s+", " ", s)
    is_amb = "ambassador" in s_space or "ambass" in s_space
    if "christmas" in s_space or "xmas" in s_space or "natale" in s_space:
        primary = "christmas_bundle"
    elif "early" in s_space:
        primary = "early_bird"
    elif "phase 0" in s_space or "phase0" in s_space:
        primary = "phase_0"
    elif "phase 1" in s_space or "phase1" in s_space:
        primary = "phase_1"
    else:
        primary = "unknown"
    # Se ambassador ma primary non trovato, assegna a ambassador come primary
    if is_amb and primary == "unknown":
        primary = "ambassador"
    variant = slugify_ticket(text)
    return primary, variant, is_amb


def load_paid(path: Path, phase_hint: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(path, **READ_KWARGS)
    ticket_type_col = find_column(df, TICKET_TYPE_COLS)
    status_col = find_column(df, ORDER_STATUS_COLS)

    df["order_status_clean"] = df[status_col].fillna("").str.strip().str.lower()
    paid = df[df["order_status_clean"] == "paid"].copy()
    classified = paid[ticket_type_col].fillna("").map(classify_ticket)
    paid[["primary_phase", "variant", "is_ambassador"]] = pd.DataFrame(classified.tolist(), index=paid.index)

    if phase_hint:
        paid.loc[paid["primary_phase"] == "unknown", "primary_phase"] = phase_hint

    return paid


def main() -> None:
    parser = argparse.ArgumentParser(description="Conteggio ticket venduti per fase")
    parser.add_argument("--single-file", type=Path, default=None, help="CSV unico con tutti i ticket (default: merged_19.12.25.csv se esiste)")
    parser.add_argument("--early-phase0", type=Path, default=DEFAULT_EARLY_PHASE0, help="CSV early bird + phase 0")
    parser.add_argument("--phase1-other", type=Path, default=DEFAULT_PHASE1_OTHER, help="CSV phase1 + christmas + ambassador")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Percorso CSV di output")
    args = parser.parse_args()

    if args.single_file:
        combined = load_paid(args.single_file)
    else:
        df_early = load_paid(args.early_phase0)
        df_phase1 = load_paid(args.phase1_other)
        combined = pd.concat([df_early, df_phase1], ignore_index=True)

    primary_counts = combined.groupby("primary_phase").size().sort_values(ascending=False)
    amb_counts = combined[combined["is_ambassador"]].groupby("primary_phase").size()
    variant_counts = combined.groupby(["primary_phase", "variant"]).size().sort_values(ascending=False)

    print("== Ticket venduti per fase (solo Paid) ==")
    for phase, cnt in primary_counts.items():
        amb = int(amb_counts.get(phase, 0))
        other = cnt - amb
        print(f"{phase}: {cnt} (ambassador: {amb}, altri: {other})")
        top_variants = variant_counts.loc[phase].sort_values(ascending=False) if phase in variant_counts.index.levels[0] else None
        if top_variants is not None:
            preview = ", ".join(f"{var}={val}" for var, val in top_variants.head(5).items())
            print(f"  varianti: {preview}")
    print(f"Totale: {len(combined)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    summary_df = primary_counts.reset_index()
    summary_df.columns = ["primary_phase", "tickets_paid"]
    summary_df["ambassadors_paid"] = summary_df["primary_phase"].map(amb_counts).fillna(0).astype(int)
    summary_df["others_paid"] = summary_df["tickets_paid"] - summary_df["ambassadors_paid"]
    summary_df.to_csv(args.output, index=False, encoding="utf-8")

    variants_path = args.output.with_name(args.output.stem + "_variants.csv")
    variant_counts.reset_index().rename(columns={0: "tickets_paid"}).to_csv(variants_path, index=False, encoding="utf-8")

    print(f"Riepilogo fasi salvato in: {args.output}")
    print(f"Dettaglio varianti salvato in: {variants_path}")


if __name__ == "__main__":
    main()
