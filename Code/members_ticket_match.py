#!/usr/bin/env python3
"""
Match soci 2025 con i ticket Paid del 2026.
Genera quattro CSV: match, soci senza match, ticket Paid senza socio e duplicati.
"""
from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd

READ_KWARGS = {
    "sep": ",",
    "quotechar": "\"",
    "encoding": "utf-8",
    "engine": "python",
    "dtype": str,
    "skip_blank_lines": True,
}

DEFAULT_MEMBERS = Path(__file__).resolve().parents[1] / "Documenti/Libro_Soci_2025-2026/Libro_soci_2025.csv"
DEFAULT_TICKETS = Path(__file__).resolve().parents[1] / "Documenti/Libro_Soci_2025-2026/ListaTickets_31.12.25_10_50.csv"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "Documenti/match_2026_reports"

ATTENDEE_FIRST_NAMES = ["Attendee First Name", "First Name"]
ATTENDEE_LAST_NAMES = ["Attendee Last Name", "Last Name"]
ATTENDEE_EMAILS = ["Attendee E-mail", "Buyer E-Mail", "Email"]
BUYER_FIRST_NAMES = ["Buyer First Name"]
BUYER_LAST_NAMES = ["Buyer Last Name"]
ORDER_STATUS_COLS = ["Order Status"]
ORDER_NUMBER_COLS = ["Order Number"]
TICKET_TYPE_COLS = ["Ticket Type"]


def normalize_name(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^0-9A-Za-z]+", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def name_key(first: object, last: object) -> str:
    return f"{normalize_name(first)}|{normalize_name(last)}"


def find_column(df: pd.DataFrame, candidates: Iterable[str], required: bool = True) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    if required:
        raise KeyError(f"Colonna mancante: una tra {candidates}")
    return ""


def load_members(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, **READ_KWARGS)
    for col in ("Cognome", "Nome", "Email"):
        if col not in df.columns:
            raise KeyError(f"Nel file soci manca la colonna '{col}'")
    df["norm_key"] = df.apply(lambda r: name_key(r.get("Nome"), r.get("Cognome")), axis=1)
    df["email_norm"] = df["Email"].fillna("").str.strip().str.lower()
    df = df[df["norm_key"] != ""].copy()
    return df[["Cognome", "Nome", "Email", "norm_key", "email_norm"]]


def load_paid_attendees(path: Path) -> Tuple[pd.DataFrame, str, str, str, str, str, str, str]:
    df = pd.read_csv(path, **READ_KWARGS)
    first_col = find_column(df, ATTENDEE_FIRST_NAMES)
    last_col = find_column(df, ATTENDEE_LAST_NAMES)
    email_col = find_column(df, ATTENDEE_EMAILS)
    status_col = find_column(df, ORDER_STATUS_COLS)
    order_col = find_column(df, ORDER_NUMBER_COLS, required=False) or ""
    type_col = find_column(df, TICKET_TYPE_COLS, required=False) or ""
    buyer_first_col = find_column(df, BUYER_FIRST_NAMES, required=False) or ""
    buyer_last_col = find_column(df, BUYER_LAST_NAMES, required=False) or ""

    df["order_status_clean"] = df[status_col].fillna("").str.strip().str.lower()
    paid = df[df["order_status_clean"] == "paid"].copy()

    paid["norm_attendee"] = paid.apply(lambda r: name_key(r.get(first_col), r.get(last_col)), axis=1)
    paid["norm_buyer"] = paid.apply(lambda r: name_key(r.get(buyer_first_col), r.get(buyer_last_col)), axis=1)
    # usa prima il nome/cognome attendee, se vuoto usa buyer
    paid["norm_key"] = paid["norm_attendee"]
    paid.loc[paid["norm_key"] == "", "norm_key"] = paid.loc[paid["norm_key"] == "", "norm_buyer"]
    paid["email_norm"] = paid[email_col].fillna("").str.strip().str.lower()
    paid = paid[paid["norm_key"] != ""].copy()

    return paid, first_col, last_col, email_col, order_col, type_col, buyer_first_col, buyer_last_col


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    print(f"Salvato: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Match soci 2025 con ticket Paid 2026")
    parser.add_argument("--members", type=Path, default=DEFAULT_MEMBERS, help="CSV soci 2025")
    parser.add_argument("--tickets", type=Path, default=DEFAULT_TICKETS, help="CSV ticket 2026")
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Cartella di output per i report"
    )
    args = parser.parse_args()

    members = load_members(args.members)
    paid, first_col, last_col, email_col, order_col, type_col, buyer_first_col, buyer_last_col = load_paid_attendees(args.tickets)

    dup_mask = paid.duplicated(subset=["norm_key"], keep=False)
    duplicates = paid.loc[dup_mask].copy()
    paid_unique = paid.drop_duplicates(subset=["norm_key"], keep="first").copy()

    matches = members.merge(
        paid_unique,
        on="norm_key",
        how="left",
        suffixes=("_member", "_ticket"),
        indicator=False,
    )

    matched = matches[~matches[first_col].isna()].copy()
    soci_no_match = matches[matches[first_col].isna()].copy()

    attendees_no_member = paid_unique.merge(members[["norm_key"]], on="norm_key", how="left", indicator=True)
    attendees_no_member = attendees_no_member[attendees_no_member["_merge"] == "left_only"].copy()
    attendees_no_member = attendees_no_member.drop(columns=["_merge"])

    print("== Riepilogo ==")
    print(f"Soci totali: {len(members)}")
    print(f"Ticket Paid 2026: {len(paid)} (unici per nome: {len(paid_unique)})")
    print(f"Duplicati Paid (stesso nome): {len(duplicates)}")
    print(f"Soci con match: {len(matched)}")
    print(f"Soci senza match: {len(soci_no_match)}")
    print(f"Paid senza socio 2025: {len(attendees_no_member)}")

    out_dir = args.output_dir

    def existing_cols(df: pd.DataFrame, candidates: Iterable[str]) -> list[str]:
        return [c for c in candidates if c and c in df.columns]

    match_cols = ["Cognome", "Nome", "Email"]
    match_cols += existing_cols(matched, [last_col, first_col, buyer_last_col, buyer_first_col, email_col, order_col, type_col])
    save_csv(matched[match_cols], out_dir / "match2026.csv")

    save_csv(soci_no_match[["Cognome", "Nome", "Email"]], out_dir / "soci_senza_match2026.csv")

    no_soci_cols = existing_cols(attendees_no_member, [last_col, first_col, buyer_last_col, buyer_first_col, email_col, order_col, type_col])
    save_csv(attendees_no_member[no_soci_cols], out_dir / "noSoci2025.csv")

    dup_cols = existing_cols(duplicates, [last_col, first_col, buyer_last_col, buyer_first_col, email_col, order_col, type_col, "order_status_clean"])
    save_csv(duplicates[dup_cols], out_dir / "duplicati2026.csv")


if __name__ == "__main__":
    main()
