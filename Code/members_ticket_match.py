#!/usr/bin/env python3
"""
Match soci 2025 con i ticket Paid del 2026.
Calcola un punteggio basato sui parametri disponibili in entrambe le liste.
"""
from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path
from dataclasses import dataclass
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
ATTENDEE_EMAILS = ["Attendee E-mail"]
BUYER_EMAILS = ["Buyer E-Mail"]
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


def first_token(value: object) -> str:
    text = normalize_name(value)
    if not text:
        return ""
    return text.split(" ", maxsplit=1)[0]


def find_column(df: pd.DataFrame, candidates: Iterable[str], required: bool = True) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    if required:
        raise KeyError(f"Colonna mancante: una tra {candidates}")
    return ""


@dataclass(frozen=True)
class ParamDef:
    label: str
    member_col: str
    ticket_col: str
    mode: str


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    return re.sub(r"\s+", " ", text)


def normalize_email(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def normalize_phone(value: object) -> str:
    if value is None:
        return ""
    return re.sub(r"[^0-9]+", "", str(value))


def normalize_tax(value: object) -> str:
    if value is None:
        return ""
    return re.sub(r"[^0-9A-Za-z]+", "", str(value)).upper()


def normalize_date(value: object) -> str:
    if value is None:
        return ""
    return re.sub(r"[^0-9]+", "", str(value))


def normalize_zip(value: object) -> str:
    if value is None:
        return ""
    return re.sub(r"[^0-9]+", "", str(value))


def normalize_by_mode(value: object, mode: str) -> str:
    if mode == "email":
        return normalize_email(value)
    if mode == "phone":
        return normalize_phone(value)
    if mode == "tax":
        return normalize_tax(value)
    if mode == "date":
        return normalize_date(value)
    if mode == "zip":
        return normalize_zip(value)
    return normalize_text(value)


def infer_mode(col_name: str) -> str:
    name = col_name.lower()
    if "mail" in name:
        return "email"
    if "tel" in name or "cell" in name or "phone" in name or "mobile" in name:
        return "phone"
    if "fiscale" in name or "tax" in name or name == "cf":
        return "tax"
    if "data" in name or "date" in name or "birth" in name or "nascita" in name:
        return "date"
    if name == "cap" or "zip" in name:
        return "zip"
    return "text"


def pick_ticket_column(df: pd.DataFrame, candidates: Iterable[str]) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    return ""


def load_members(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, **READ_KWARGS)
    for col in ("Cognome", "Nome", "Email"):
        if col not in df.columns:
            raise KeyError(f"Nel file soci manca la colonna '{col}'")
    df["norm_key"] = df.apply(lambda r: name_key(r.get("Nome"), r.get("Cognome")), axis=1)
    df["first_norm"] = df["Nome"].map(normalize_name)
    df["last_norm"] = df["Cognome"].map(normalize_name)
    df["first_token"] = df["Nome"].map(first_token)
    df["email_norm"] = df["Email"].map(normalize_email)
    df = df[df["norm_key"] != ""].copy()
    df["member_id"] = range(len(df))
    return df


def load_paid_attendees(path: Path) -> Tuple[pd.DataFrame, str, str, str, str, str, str, str, str]:
    df = pd.read_csv(path, **READ_KWARGS)
    first_col = find_column(df, ATTENDEE_FIRST_NAMES)
    last_col = find_column(df, ATTENDEE_LAST_NAMES)
    email_col = find_column(df, ATTENDEE_EMAILS)
    buyer_email_col = find_column(df, BUYER_EMAILS, required=False) or ""
    status_col = find_column(df, ORDER_STATUS_COLS)
    order_col = find_column(df, ORDER_NUMBER_COLS, required=False) or ""
    type_col = find_column(df, TICKET_TYPE_COLS, required=False) or ""
    buyer_first_col = find_column(df, BUYER_FIRST_NAMES, required=False) or ""
    buyer_last_col = find_column(df, BUYER_LAST_NAMES, required=False) or ""

    df["order_status_clean"] = df[status_col].fillna("").str.strip().str.lower()
    paid = df[df["order_status_clean"] == "paid"].copy()

    attendee_first = paid[first_col].fillna("").astype(str).str.strip()
    attendee_last = paid[last_col].fillna("").astype(str).str.strip()
    buyer_first = paid[buyer_first_col].fillna("").astype(str).str.strip() if buyer_first_col else ""
    buyer_last = paid[buyer_last_col].fillna("").astype(str).str.strip() if buyer_last_col else ""
    paid["first_effective"] = attendee_first
    paid.loc[paid["first_effective"] == "", "first_effective"] = buyer_first
    paid["last_effective"] = attendee_last
    paid.loc[paid["last_effective"] == "", "last_effective"] = buyer_last
    paid["first_norm"] = paid["first_effective"].map(normalize_name)
    paid["last_norm"] = paid["last_effective"].map(normalize_name)
    paid["first_token"] = paid["first_effective"].map(first_token)
    paid["norm_key"] = paid.apply(lambda r: name_key(r.get("first_effective"), r.get("last_effective")), axis=1)
    paid["email_norm"] = paid[email_col].map(normalize_email)
    paid["buyer_email_norm"] = paid[buyer_email_col].map(normalize_email) if buyer_email_col else ""
    paid = paid[paid["norm_key"] != ""].copy()
    paid["ticket_id"] = range(len(paid))

    return (
        paid,
        first_col,
        last_col,
        email_col,
        buyer_email_col,
        order_col,
        type_col,
        buyer_first_col,
        buyer_last_col,
    )


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
    (
        paid,
        first_col,
        last_col,
        email_col,
        buyer_email_col,
        order_col,
        type_col,
        buyer_first_col,
        buyer_last_col,
    ) = load_paid_attendees(args.tickets)

    dup_mask = paid.duplicated(subset=["norm_key"], keep=False)
    duplicates = paid.loc[dup_mask].copy()

    matches = members.merge(
        paid,
        on="norm_key",
        how="left",
        suffixes=("_member", "_ticket"),
        indicator=True,
    )
    candidates = matches[matches["_merge"] == "both"].copy()
    soci_no_match = matches[matches["_merge"] == "left_only"].copy()

    params: list[ParamDef] = [ParamDef("email", "email_norm_member", "email_norm_ticket", "email")]

    def add_param(label: str, member_col: str, ticket_col: str, mode: str) -> None:
        if member_col in members.columns and ticket_col in paid.columns:
            params.append(ParamDef(label, f"{member_col}_member", f"{ticket_col}_ticket", mode))

    ticket_birth = pick_ticket_column(
        paid,
        [
            "Date of birth / Data di nascita (Campi ticket holder)",
            "Data di Nascita / Date of Birth (Campi ticket restaurant)",
        ],
    )
    if ticket_birth:
        add_param("data_nascita", "Data di nascita", ticket_birth, "date")

    ticket_birth_place = pick_ticket_column(
        paid,
        ["Place of birth (city) / Luogo di nascita (Campi ticket holder)"],
    )
    if ticket_birth_place:
        add_param("luogo_nascita", "Luogo di nascita", ticket_birth_place, "text")

    ticket_city = pick_ticket_column(
        paid,
        [
            "City of residence / Città di residenza (Campi ticket holder)",
            "Città di Residenza / City of Residence (Campi ticket restaurant)",
        ],
    )
    if ticket_city:
        add_param("residenza", "Residenza", ticket_city, "text")

    ticket_tax = pick_ticket_column(
        paid,
        [
            "Document ID / Codice Fiscale (Campi ticket holder)",
            "Document ID / Codice Fiscale (FREE) (Free Artist Fields)",
            "Codice Fiscale / Document ID (R/S) (Campi ticket restaurant)",
        ],
    )
    if ticket_tax:
        add_param("codice_fiscale", "Codice fiscale", ticket_tax, "tax")

    def score_row(row: pd.Series) -> pd.Series:
        available = 0
        matched = 0
        extra_matched = 0
        matched_labels: list[str] = []
        for param in params:
            left = normalize_by_mode(row.get(param.member_col), param.mode)
            right = normalize_by_mode(row.get(param.ticket_col), param.mode)
            if left == "" or right == "":
                continue
            available += 1
            if left == right:
                matched += 1
                if param.label != "email":
                    extra_matched += 1
                matched_labels.append(param.label)
        percent = round((matched / available) * 100, 2) if available else 0.0
        return pd.Series(
            {
                "params_available": available,
                "params_matched": matched,
                "match_percent": percent,
                "extra_params_matched": extra_matched,
                "matched_params": ", ".join(matched_labels),
            }
        )

    if not candidates.empty:
        scores = candidates.apply(score_row, axis=1)
        candidates = pd.concat([candidates, scores], axis=1)
    else:
        candidates["params_available"] = 0
        candidates["params_matched"] = 0
        candidates["match_percent"] = 0.0
        candidates["extra_params_matched"] = 0
        candidates["matched_params"] = ""

    candidates["email_match"] = (
        (candidates["email_norm_member"] != "")
        & (candidates["email_norm_ticket"] != "")
        & (candidates["email_norm_member"] == candidates["email_norm_ticket"])
    )
    candidates["buyer_email_match"] = (
        (candidates["email_norm_member"] != "")
        & (candidates["buyer_email_norm"] != "")
        & (candidates["email_norm_member"] == candidates["buyer_email_norm"])
    )
    candidates["email_mismatch"] = (
        (candidates["email_norm_member"] != "")
        & (candidates["email_norm_ticket"] != "")
        & (candidates["email_norm_member"] != candidates["email_norm_ticket"])
    )

    best_per_member = (
        candidates.sort_values(
            ["member_id", "params_matched", "match_percent", "params_available"],
            ascending=[True, False, False, False],
        )
        .drop_duplicates(subset=["member_id"], keep="first")
        .copy()
    )

    matched_strong = best_per_member[best_per_member["email_match"]].copy()
    matched_plus = matched_strong[matched_strong["extra_params_matched"] > 0].copy()
    matched_discard = best_per_member[~best_per_member["email_match"]].copy()
    matched_email_diff = best_per_member[best_per_member["email_mismatch"]].copy()

    email_join = members.merge(
        paid,
        on="email_norm",
        how="inner",
        suffixes=("_member", "_ticket"),
    )
    email_join = email_join[email_join["norm_key_member"] != email_join["norm_key_ticket"]].copy()
    email_join["name_swapped"] = (
        (email_join["first_norm_member"] == email_join["last_norm_ticket"])
        & (email_join["last_norm_member"] == email_join["first_norm_ticket"])
    )
    email_join["name_partial"] = (
        (email_join["last_norm_member"] == email_join["last_norm_ticket"])
        & (email_join["first_token_member"] != "")
        & (email_join["first_token_member"] == email_join["first_token_ticket"])
    )
    email_join["name_duplicate_field"] = (
        (email_join["first_norm_member"] == email_join["last_norm_member"])
        & (email_join["first_norm_member"].str.contains(" "))
    ) | (
        (email_join["first_norm_ticket"] == email_join["last_norm_ticket"])
        & (email_join["first_norm_ticket"].str.contains(" "))
    )
    email_join["name_sim_reason"] = ""
    email_join.loc[email_join["name_swapped"], "name_sim_reason"] = "nome_cognome_invertiti"
    email_join.loc[email_join["name_partial"], "name_sim_reason"] = (
        email_join.loc[email_join["name_partial"], "name_sim_reason"].replace("", "nome_parziale")
    )
    email_join.loc[email_join["name_duplicate_field"], "name_sim_reason"] = (
        email_join.loc[email_join["name_duplicate_field"], "name_sim_reason"]
        .replace("", "nome_cognome_in_stesso_campo")
    )
    matched_name_similar = email_join[
        email_join["name_swapped"] | email_join["name_partial"] | email_join["name_duplicate_field"]
    ].copy()

    attendees_no_member = paid.merge(members[["norm_key"]], on="norm_key", how="left", indicator=True)
    attendees_no_member = attendees_no_member[attendees_no_member["_merge"] == "left_only"].copy()
    attendees_no_member = attendees_no_member.drop(columns=["_merge"])

    print("== Riepilogo ==")
    print(f"Soci totali: {len(members)}")
    print(f"Ticket Paid 2026: {len(paid)}")
    print(f"Duplicati Paid (stesso nome): {len(duplicates)}")
    print(f"Soci senza ticket (nome): {len(soci_no_match)}")
    print(f"Match sicuri (nome+email): {len(matched_strong)}")
    print(f"Match sicuri con extra parametri: {len(matched_plus)}")
    print(f"Scarti (email non combaciante): {len(matched_discard)}")
    print(f"Nome combacia ma email diversa: {len(matched_email_diff)}")
    print(f"Nome simile con email uguale: {len(matched_name_similar)}")
    if buyer_email_col:
        print(f"Buyer email presente: colonna '{buyer_email_col}'")
    print(f"Ticket Paid senza socio 2025 (nome): {len(attendees_no_member)}")
    print(f"Parametri usati per punteggio (escluso nome): {len(params)}")
    print(" - " + ", ".join(p.label for p in params))

    out_dir = args.output_dir

    def existing_cols(df: pd.DataFrame, candidates: Iterable[str]) -> list[str]:
        return [c for c in candidates if c and c in df.columns]

    match_cols = [
        "Cognome",
        "Nome",
        "Email",
        "buyer_email_match",
        "params_available",
        "params_matched",
        "extra_params_matched",
        "match_percent",
        "matched_params",
    ]
    match_cols += existing_cols(
        matched_strong,
        [last_col, first_col, buyer_last_col, buyer_first_col, email_col, buyer_email_col, order_col, type_col],
    )
    ticket_name = args.tickets.name
    suffix = ticket_name[ticket_name.find("_") :] if "_" in ticket_name else f"_{ticket_name}"
    save_csv(matched_strong[match_cols], out_dir / f"match2026{suffix}")

    plus_cols = [c for c in match_cols if c in matched_plus.columns]
    save_csv(matched_plus[plus_cols], out_dir / f"match2026_plus{suffix}")

    discard_cols = [c for c in match_cols if c in matched_discard.columns]
    save_csv(matched_discard[discard_cols], out_dir / f"match2026_scarti{suffix}")

    diff_cols = [c for c in match_cols if c in matched_email_diff.columns]
    save_csv(matched_email_diff[diff_cols], out_dir / f"match2026_email_diversa{suffix}")

    sim_cols = [
        "Cognome",
        "Nome",
        "Email",
        "name_sim_reason",
        "buyer_email_match",
        "params_available",
        "params_matched",
        "extra_params_matched",
        "match_percent",
        "matched_params",
    ]
    sim_cols = [c for c in sim_cols if c in matched_name_similar.columns]
    save_csv(matched_name_similar[sim_cols], out_dir / f"match2026_nome_simile{suffix}")

    save_csv(soci_no_match[["Cognome", "Nome", "Email"]], out_dir / f"soci_senza_match2026{suffix}")

    no_soci_cols = existing_cols(
        attendees_no_member,
        [last_col, first_col, buyer_last_col, buyer_first_col, email_col, buyer_email_col, order_col, type_col],
    )
    save_csv(attendees_no_member[no_soci_cols], out_dir / f"noSoci2025{suffix}")

    dup_cols = existing_cols(
        duplicates,
        [last_col, first_col, buyer_last_col, buyer_first_col, email_col, order_col, type_col, "order_status_clean"],
    )
    save_csv(duplicates[dup_cols], out_dir / f"duplicati2026{suffix}")

    reverse_issues = matched_strong[(matched_strong["member_id"].isna()) | (matched_strong["ticket_id"].isna())].copy()
    if not reverse_issues.empty:
        save_csv(reverse_issues, out_dir / f"verifica_inversa_errori{suffix}")
        print(f"Verifica inversa: errori trovati {len(reverse_issues)} (vedi CSV)")
    else:
        print("Verifica inversa: OK")


if __name__ == "__main__":
    main()
