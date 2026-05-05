#!/usr/bin/env python3
"""
Analisi EDA del festival 7 Chakras.

Lo script replica le elaborazioni principali del notebook `7chakras_eda_full.ipynb`
ma le rende eseguibili da linea di comando. Un file di configurazione JSON
(`eda_config.json`) permette di impostare rapidamente il CSV da analizzare,
la cartella di output e i nomi colonna specifici dell'export.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import textwrap
from typing import Dict, Iterable, List, Optional
import unicodedata

import numpy as np
import pandas as pd

# Usa backend "Agg" per salvare i grafici anche su server/headless.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (dipende dal backend impostato)
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FuncFormatter, MultipleLocator
from datetime import datetime

READ_KWARGS: Dict[str, object] = {
    "sep": ",",
    "quotechar": '"',
    "encoding": "utf-8",
    "engine": "python",
    "dtype": str,
    "skip_blank_lines": True,
}

PHASE_MARKER_COLOR = "#00897b"
LOVERS_BUNDLE_COLOR = "#ff6f00"
CHRISTMAS_BUNDLE_KEYWORD = "christmas bundle"
LINEUP_RELEASE_KEYWORD = "lineup release"

NUMERIC_CANDIDATES = [
    "Order Total",
    "Ticket Subtotal",
    "Ticket Discount",
    "Ticket Fee",
    "Ticket Total",
    "Price",
]

DEFAULT_CHECKIN_COLUMNS = ["Checked-in", "Check-ins", "Check-outs"]
PARSED_DATE_COL = "Payment_Date_parsed"
FULL_FESTIVAL_INCLUDE_KEYWORDS = ("full festival", "ambassador", "bundle")
FULL_FESTIVAL_EXCLUDE_KEYWORDS = ("caravan", "membership", "reticketing", "caregiver")
VOLUNTEER_ANALYSIS_TICKET_TYPE_COL = "Ticket Type Analysis"
IS_VOLUNTEER_COL = "Is Volunteer"
VOLUNTEER_MATCH_METHOD_COL = "Volunteer Match Method"
VOLUNTEER_SOURCE_NAME_COL = "Volunteer Source Name"
VOLUNTEER_SOURCE_EMAIL_COL = "Volunteer Source Email"
VOLUNTEER_ORIGINAL_TICKET_TYPE_COL = "Volunteer Original Ticket Type"
VOLUNTEER_ORIGINAL_TICKET_TOTAL_COL = "Volunteer Original Ticket Total"
VOLUNTEER_POTENTIAL_REFUND_COL = "Volunteer Potential Refund"
VOLUNTEER_TICKET_PREFIX = "VOLUNTEER - "


def load_config(path: Path) -> Dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Config non trovato: {path}")
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def json_default(value: object) -> object:
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except TypeError:
            pass
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def save_config(path: Path, config: Dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(config, fh, ensure_ascii=False, indent=2, default=json_default)


def normalize_columns(columns: Iterable[str]) -> List[str]:
    seen: Dict[str, int] = {}
    normalized = []
    for name in columns:
        clean_name = " ".join((name or "").split())
        if clean_name not in seen:
            seen[clean_name] = 0
            normalized.append(clean_name)
        else:
            seen[clean_name] += 1
            normalized.append(f"{clean_name}__{seen[clean_name]}")
    return normalized


def parse_payment_date(value: object) -> pd.Timestamp:
    if pd.isna(value):
        return pd.NaT
    s = str(value).strip().replace("\ufffd", "-")
    for fmt in ("%d/%m/%Y - %H:%M", "%d/%m/%Y %H:%M", "%d/%m/%Y", "%d-%m-%Y %H:%M"):
        try:
            return pd.Timestamp(datetime.strptime(s, fmt))
        except ValueError:
            continue
    return pd.NaT


def to_num(value: object) -> float:
    if value is None:
        return np.nan
    if isinstance(value, float) and np.isnan(value):
        return np.nan
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none"}:
        return np.nan
    primary = (
        s.replace("\u20ac", "")
        .replace("\ufffd", "")
        .replace(",", "")
        .replace(" ", "")
    )
    try:
        return float(primary)
    except ValueError:
        secondary = (
            s.replace("\u20ac", "")
            .replace("\ufffd", "")
            .replace(".", "")
            .replace(",", ".")
            .replace(" ", "")
        )
        try:
            return float(secondary)
        except ValueError:
            return np.nan


def missing_count(series: pd.Series) -> int:
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_datetime64_any_dtype(series):
        return int(series.isna().sum())
    ser = series.astype(str).str.strip()
    mask = ser.eq("") | ser.str.lower().isin({"nan", "none", "null"})
    return int(mask.sum())


def drop_nan_categories(series: pd.Series) -> pd.Series:
    idx = pd.Series(series.index, dtype=object)
    normalized = idx.astype(str).str.strip().str.lower()
    mask = (~idx.isna()) & (normalized != "nan") & (normalized != "")
    return series[mask.to_numpy()]


def slugify(text: str) -> str:
    """Create a filesystem-friendly slug from a column name."""
    safe = "".join(ch.lower() if ch.isalnum() else "_" for ch in text)
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe.strip("_")[:80]


def extract_ambassador_name(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    match = re.search(r"\bambassador\b\s*[:\-]?\s*(.+)", text, flags=re.IGNORECASE)
    if not match:
        return ""
    tail = match.group(1).strip()
    for sep in (" - ", " | ", " / ", "(", "[", "{", "\u2013", "\u2014"):
        if sep in tail:
            tail = tail.split(sep, 1)[0].strip()
    return tail


def extract_phase_label(value: object) -> str:
    if value is None:
        return "unknown"
    text = str(value).strip().lower()
    if not text:
        return "unknown"
    if "early" in text:
        return "early_bird"
    phase_match = re.search(r"\bphase\s*([0-9]+)\b", text)
    if phase_match:
        return f"phase_{phase_match.group(1)}"
    if "christmas" in text:
        return "christmas"
    if "ambassador" in text:
        return "ambassador"
    return "unknown"


def extract_ticket_type_amount(value: object) -> float:
    if value is None:
        return np.nan
    text = str(value)
    match = re.search(r"(\d+(?:[.,]\d+)?)\s*(?:\u20ac|eur|\u0192'\u00aa)", text, flags=re.IGNORECASE)
    if not match:
        return np.nan
    raw = match.group(1).replace(",", ".")
    try:
        return float(raw)
    except ValueError:
        return np.nan


def normalize_phase_token(text: str) -> Optional[str]:
    if not text:
        return None
    match = re.search(r"\bphase[\s_\-]*([a-z0-9]+)\b", text, flags=re.IGNORECASE)
    if not match:
        return None
    token = match.group(1).strip()
    if token.isdigit():
        return f"PHASE {int(token)}"
    return f"PHASE {token.upper()}"


def focused_ticket_category(value: object) -> str:
    if value is None:
        return "UNKNOWN"
    text = " ".join(str(value).strip().split())
    if not text:
        return "UNKNOWN"

    lowered = text.lower()
    phase_label = normalize_phase_token(lowered)
    if phase_label:
        year_match = re.search(r"\b(20\d{2})\b", text)
        year_prefix = f"{year_match.group(1)} – " if year_match else ""
        if "volunteer" in lowered:
            return f"{year_prefix}VOLUNTEER – FULL FESTIVAL – {phase_label}"
        if "full festival" in lowered or "ambassador" in lowered:
            return f"{year_prefix}FULL FESTIVAL – {phase_label}"
        return f"{year_prefix}{phase_label}"

    return text


def is_full_festival_pass_ticket(value: object) -> bool:
    if value is None:
        return False
    normalized = str(value).lower()
    normalized = normalized.replace("\u2013", "-").replace("\u2014", "-")
    normalized = " ".join(normalized.split())
    if not normalized:
        return False
    if "volunteer" in normalized:
        return False
    if any(keyword in normalized for keyword in FULL_FESTIVAL_EXCLUDE_KEYWORDS):
        return False
    return any(keyword in normalized for keyword in FULL_FESTIVAL_INCLUDE_KEYWORDS)


def is_accessory_ticket(value: object) -> bool:
    if value is None:
        return False
    normalized = str(value).lower()
    normalized = normalized.replace("\u2013", "-").replace("\u2014", "-")
    normalized = " ".join(normalized.split())
    if not normalized:
        return False
    accessory_keywords = ("volunteer", "caravan", "membership", "reticketing", "caregiver")
    return any(keyword in normalized for keyword in accessory_keywords)


def normalize_match_email(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip().lower()


def normalize_match_name(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKD", str(value).strip().lower())
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-z ]+", " ", text)
    return " ".join(text.split())


def name_tokens(value: object) -> tuple[str, ...]:
    normalized = normalize_match_name(value)
    return tuple(normalized.split()) if normalized else tuple()


def is_name_token_match(volunteer_tokens: tuple[str, ...], candidate_tokens: tuple[str, ...]) -> bool:
    if len(volunteer_tokens) < 2 or len(candidate_tokens) < 2:
        return False
    return set(volunteer_tokens).issubset(set(candidate_tokens))


def join_name_parts(*values: object) -> str:
    return " ".join(str(v).strip() for v in values if v is not None and not pd.isna(v) and str(v).strip())


def volunteer_enriched_output_path(volunteers_path: Path, suffix: str) -> Path:
    clean_suffix = suffix or "_enriched"
    if not clean_suffix.startswith("_"):
        clean_suffix = f"_{clean_suffix}"
    return volunteers_path.with_name(f"{volunteers_path.stem}{clean_suffix}{volunteers_path.suffix}")


def apply_volunteer_enrichment(
    df: pd.DataFrame,
    volunteers_cfg: Dict[str, object],
    ticket_type_col: Optional[str],
    ticket_total_num: Optional[str],
    payment_date_col: Optional[str],
    attendee_email_col: Optional[str],
    buyer_email_col: Optional[str],
    name_col: Optional[str],
    first_name_col: Optional[str],
    last_name_col: Optional[str],
    order_number_col: Optional[str],
    ticket_code_col: Optional[str],
    ticket_id_col: Optional[str],
    csv_dir: Path,
) -> Dict[str, object]:
    info: Dict[str, object] = {"enabled": False, "matched_count": 0}
    if not volunteers_cfg or not volunteers_cfg.get("enabled", False):
        return info
    if not ticket_type_col or ticket_type_col not in df.columns:
        print("\nVolontari: colonna Ticket Type non disponibile, enrichment saltato.")
        return info

    volunteers_path_raw = volunteers_cfg.get("csv_path")
    if not volunteers_path_raw:
        print("\nVolontari: csv_path non configurato, enrichment saltato.")
        return info
    volunteers_path = Path(str(volunteers_path_raw)).expanduser()
    if not volunteers_path.exists():
        print(f"\nVolontari: file non trovato, enrichment saltato: {volunteers_path}")
        return info

    name_source_col = str(volunteers_cfg.get("name_col", "Name") or "Name").strip()
    email_source_col = str(volunteers_cfg.get("email_col", "Email") or "Email").strip()
    volunteers = pd.read_csv(volunteers_path, dtype=str, keep_default_na=False)
    volunteers.columns = [str(c).strip() for c in volunteers.columns]
    if name_source_col not in volunteers.columns or email_source_col not in volunteers.columns:
        print(
            f"\nVolontari: colonne richieste non trovate in {volunteers_path.name}: "
            f"{name_source_col!r}, {email_source_col!r}."
        )
        return info

    working = df.copy()
    if attendee_email_col and attendee_email_col in working.columns:
        working["__vol_attendee_email_norm"] = working[attendee_email_col].map(normalize_match_email)
    else:
        working["__vol_attendee_email_norm"] = ""
    if buyer_email_col and buyer_email_col in working.columns:
        working["__vol_buyer_email_norm"] = working[buyer_email_col].map(normalize_match_email)
    else:
        working["__vol_buyer_email_norm"] = ""
    if name_col and name_col in working.columns:
        working["__vol_name_tokens"] = working[name_col].map(name_tokens)
    else:
        working["__vol_name_tokens"] = [tuple()] * len(working)
    if first_name_col and last_name_col and first_name_col in working.columns and last_name_col in working.columns:
        full_names = working.apply(lambda row: join_name_parts(row.get(first_name_col), row.get(last_name_col)), axis=1)
        working["__vol_first_last_tokens"] = full_names.map(name_tokens)
    else:
        working["__vol_first_last_tokens"] = [tuple()] * len(working)

    def candidate_name_match(row: pd.Series, volunteer_tokens: tuple[str, ...]) -> bool:
        return is_name_token_match(volunteer_tokens, row["__vol_name_tokens"]) or is_name_token_match(
            volunteer_tokens,
            row["__vol_first_last_tokens"],
        )

    def choose_candidate(candidates: pd.DataFrame, volunteer_tokens: tuple[str, ...]) -> Optional[int]:
        if candidates.empty:
            return None

        scored: List[tuple[int, int, float, int]] = []
        for idx, row in candidates.iterrows():
            full_festival_score = 1 if is_full_festival_pass_ticket(row.get(ticket_type_col)) else 0
            name_score = 1 if candidate_name_match(row, volunteer_tokens) else 0
            amount = to_num(row.get(ticket_total_num)) if ticket_total_num in row.index else 0.0
            amount = 0.0 if pd.isna(amount) else float(amount)
            scored.append((full_festival_score, name_score, amount, idx))
        scored.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        return scored[0][3]

    enriched_rows: List[Dict[str, object]] = []
    matched_indices: Dict[int, Dict[str, object]] = {}

    volunteers_norm = volunteers.copy()
    volunteers_norm["__vol_email_norm"] = volunteers_norm[email_source_col].map(normalize_match_email)
    volunteers_norm["__vol_name_tokens"] = volunteers_norm[name_source_col].map(name_tokens)

    for _, volunteer in volunteers_norm.iterrows():
        volunteer_email = volunteer["__vol_email_norm"]
        volunteer_tokens = volunteer["__vol_name_tokens"]
        email_candidates = working[
            (working["__vol_attendee_email_norm"] == volunteer_email)
            | (working["__vol_buyer_email_norm"] == volunteer_email)
        ]
        selected_idx = choose_candidate(email_candidates, volunteer_tokens)
        match_method = "unmatched"
        if selected_idx is not None:
            selected_row = working.loc[selected_idx]
            if candidate_name_match(selected_row, volunteer_tokens):
                match_method = "email_and_name_match"
            else:
                match_method = "email_only_match"
        else:
            name_mask = working.apply(lambda row: candidate_name_match(row, volunteer_tokens), axis=1)
            name_candidates = working[name_mask]
            selected_idx = choose_candidate(name_candidates, volunteer_tokens)
            if selected_idx is not None:
                match_method = "name_only_match"

        matched_row = working.loc[selected_idx] if selected_idx is not None else pd.Series(dtype=object)
        ticket_total_value = matched_row.get(ticket_total_num, np.nan) if ticket_total_num and ticket_total_num in matched_row.index else np.nan
        matched_ticket_type = matched_row.get(ticket_type_col, "")
        enriched = {
            name_source_col: volunteer.get(name_source_col, ""),
            email_source_col: volunteer.get(email_source_col, ""),
            "Volunteer Match Method": match_method,
            "Matched Ticket Type": matched_ticket_type,
            "Matched Ticket Total": ticket_total_value,
            "Matched Payment Date": matched_row.get(payment_date_col, "") if payment_date_col else "",
            "Matched Order Number": matched_row.get(order_number_col, "") if order_number_col else "",
            "Matched Ticket Code": matched_row.get(ticket_code_col, "") if ticket_code_col else "",
            "Matched Ticket ID": matched_row.get(ticket_id_col, "") if ticket_id_col else "",
            "Matched Attendee E-mail": matched_row.get(attendee_email_col, "") if attendee_email_col else "",
            "Matched Buyer E-Mail": matched_row.get(buyer_email_col, "") if buyer_email_col else "",
            "Matched Ticket Holder Name": matched_row.get(name_col, "") if name_col else "",
            "Matched Row Index": selected_idx if selected_idx is not None else "",
        }
        enriched_rows.append(enriched)

        if selected_idx is not None:
            matched_indices[selected_idx] = {
                "method": match_method,
                "source_name": volunteer.get(name_source_col, ""),
                "source_email": volunteer.get(email_source_col, ""),
            }

    enriched_df = pd.DataFrame(enriched_rows)
    enriched_path = volunteer_enriched_output_path(
        volunteers_path,
        str(volunteers_cfg.get("enriched_suffix", "_enriched") or "_enriched"),
    )
    enriched_df.to_csv(enriched_path, index=False, encoding="utf-8")

    csv_dir.mkdir(parents=True, exist_ok=True)
    output_copy_path = csv_dir / enriched_path.name
    enriched_df.to_csv(output_copy_path, index=False, encoding="utf-8")

    df[IS_VOLUNTEER_COL] = "No"
    df[VOLUNTEER_MATCH_METHOD_COL] = ""
    df[VOLUNTEER_SOURCE_NAME_COL] = ""
    df[VOLUNTEER_SOURCE_EMAIL_COL] = ""
    df[VOLUNTEER_ORIGINAL_TICKET_TYPE_COL] = ""
    df[VOLUNTEER_ORIGINAL_TICKET_TOTAL_COL] = np.nan
    df[VOLUNTEER_POTENTIAL_REFUND_COL] = 0.0
    df[VOLUNTEER_ANALYSIS_TICKET_TYPE_COL] = df[ticket_type_col].fillna("")

    for idx, match in matched_indices.items():
        if idx not in df.index:
            continue
        original_type = df.at[idx, ticket_type_col]
        original_total = df.at[idx, ticket_total_num] if ticket_total_num and ticket_total_num in df.columns else np.nan
        df.at[idx, IS_VOLUNTEER_COL] = "Yes"
        df.at[idx, VOLUNTEER_MATCH_METHOD_COL] = match["method"]
        df.at[idx, VOLUNTEER_SOURCE_NAME_COL] = match["source_name"]
        df.at[idx, VOLUNTEER_SOURCE_EMAIL_COL] = match["source_email"]
        df.at[idx, VOLUNTEER_ORIGINAL_TICKET_TYPE_COL] = original_type
        df.at[idx, VOLUNTEER_ORIGINAL_TICKET_TOTAL_COL] = original_total
        df.at[idx, VOLUNTEER_POTENTIAL_REFUND_COL] = original_total if pd.notna(original_total) else 0.0
        df.at[idx, VOLUNTEER_ANALYSIS_TICKET_TYPE_COL] = f"{VOLUNTEER_TICKET_PREFIX}{original_type}"

    match_counts = enriched_df["Volunteer Match Method"].value_counts(dropna=False).to_dict()
    matched_count = int((enriched_df["Volunteer Match Method"] != "unmatched").sum())
    refund_total = float(pd.to_numeric(df[VOLUNTEER_POTENTIAL_REFUND_COL], errors="coerce").fillna(0).sum())
    summary = pd.DataFrame(
        [
            {"metric": "volunteers_source_rows", "value": int(len(volunteers))},
            {"metric": "volunteers_matched", "value": matched_count},
            {"metric": "volunteers_unmatched", "value": int((enriched_df["Volunteer Match Method"] == "unmatched").sum())},
            {"metric": "volunteer_potential_refund", "value": refund_total},
        ]
    )
    summary.to_csv(csv_dir / "volunteer_economics.csv", index=False, encoding="utf-8")

    print(f"\nVolontari caricati: {len(volunteers):,}")
    print(f"Volontari matchati: {matched_count:,} su {len(volunteers):,}")
    print("Metodo match volontari:")
    for method, count in sorted(match_counts.items()):
        print(f" - {method}: {count}")
    print(f"Potenziale rimborso volontari: {refund_total:,.2f}")
    print(f"File volontari enriched salvato in: {enriched_path}")
    print(f"Copia output volontari enriched salvata in: {output_copy_path}")

    info.update(
        {
            "enabled": True,
            "matched_count": matched_count,
            "source_count": int(len(volunteers)),
            "refund_total": refund_total,
            "analysis_ticket_type_col": VOLUNTEER_ANALYSIS_TICKET_TYPE_COL,
            "enriched_path": str(enriched_path),
            "output_copy_path": str(output_copy_path),
            "summary_path": str(csv_dir / "volunteer_economics.csv"),
        }
    )
    return info


def build_focused_ticket_summary(df: pd.DataFrame, ticket_type_col: str) -> pd.DataFrame:
    focused = df[ticket_type_col].fillna("").map(focused_ticket_category)
    summary = (
        focused.value_counts(dropna=False)
        .rename_axis("ticket_category")
        .reset_index(name="tickets")
        .sort_values(["tickets", "ticket_category"], ascending=[True, True])
        .reset_index(drop=True)
    )
    total_row = pd.DataFrame(
        [{"ticket_category": "TOTAL", "tickets": int(summary["tickets"].sum())}]
    )
    summary = pd.concat([summary, total_row], ignore_index=True)
    return summary


def plot_focused_ticket_summary(
    summary: pd.DataFrame,
    plots_dir: Path,
    plot_stem: str,
    plot_format: str,
    show_total_subtitle: bool = True,
    show_total_note: bool = True,
) -> None:
    if summary.empty:
        print("\nNessun dato disponibile per il grafico focused per Ticket Type.")
        return

    ordered = summary[summary["ticket_category"] != "TOTAL"].copy()
    ordered = ordered.sort_values(["tickets", "ticket_category"], ascending=[True, True]).reset_index(drop=True)
    total_tickets = int(ordered["tickets"].sum()) if not ordered.empty else 0
    caravan_mask = ordered["ticket_category"].astype(str).str.contains("CARAVAN PASS", case=False, na=False)
    caravan_tickets = int(ordered.loc[caravan_mask, "tickets"].sum()) if not ordered.empty else 0
    person_only_tickets = total_tickets - caravan_tickets
    height = max(4.5, 0.45 * len(ordered))
    fig, ax = plt.subplots(figsize=(12, height))
    ordered.plot(kind="barh", x="ticket_category", y="tickets", ax=ax, color="#1565c0", legend=False)
    ax.set_xlabel("Tickets")
    ax.set_ylabel("")
    ax.set_title("Ticket categories focused by phase")
    ax.grid(axis="x", alpha=0.25)
    max_tickets = int(ordered["tickets"].max()) if not ordered.empty else 0
    if max_tickets:
        ax.set_xlim(0, max_tickets * 1.15)
        ax.bar_label(ax.containers[0], padding=3, fmt="%.0f")
    if show_total_note:
        fig.text(
            0.985,
            0.02,
            f"TOTAL = {total_tickets}\npeople_only = {person_only_tickets}",
            ha="right",
            va="bottom",
            fontsize=13,
            bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "#999999", "boxstyle": "round,pad=0.35"},
        )

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    save_plot(fig, plots_dir, plot_stem, plot_format)


def export_focused_ticket_summary(
    df: pd.DataFrame,
    ticket_type_col: Optional[str],
    output_dir: Path,
    plots_dir: Path,
    plot_format: str,
    focused_cfg: Dict[str, object],
) -> None:
    if not focused_cfg.get("enabled", True):
        return
    if not ticket_type_col or ticket_type_col not in df.columns:
        return

    csv_subdir = str(focused_cfg.get("csv_subdir", "csv") or "csv")
    plot_subdir = str(focused_cfg.get("plot_subdir", "plots") or "plots")
    csv_name = str(focused_cfg.get("csv_name", "by_type_focused.csv") or "by_type_focused.csv")
    plot_name = str(focused_cfg.get("plot_name", "by_type_focused.png") or "by_type_focused.png")
    show_total_subtitle = bool(focused_cfg.get("show_total_subtitle", True))
    show_total_note = bool(focused_cfg.get("show_total_note", True))

    csv_dir = output_dir / csv_subdir
    plot_dir = output_dir / plot_subdir
    csv_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    summary = build_focused_ticket_summary(df, ticket_type_col)
    csv_path = csv_dir / csv_name
    summary.to_csv(csv_path, index=False, encoding="utf-8")
    plot_stem = Path(plot_name).stem if plot_name else "by_type_focused"
    plot_focused_ticket_summary(
        summary,
        plot_dir,
        plot_stem,
        plot_format,
        show_total_subtitle=show_total_subtitle,
        show_total_note=show_total_note,
    )
    print(f"\nRiepilogo focused salvato in: {csv_path}")
    print(f"Grafico focused salvato in: {plot_dir / f'{plot_stem}.{plot_format}' }")


def build_full_festival_ticket_type_summary(
    df: pd.DataFrame,
    ticket_type_col: str,
    ticket_total_num: Optional[str],
) -> pd.DataFrame:
    full_mask = df[ticket_type_col].map(is_full_festival_pass_ticket)
    full_df = df.loc[full_mask].copy()
    if full_df.empty:
        return pd.DataFrame()

    full_df["ticket_category"] = full_df[ticket_type_col].fillna("").map(focused_ticket_category)
    aggregations: Dict[str, object] = {"tickets": ("ticket_category", "size")}
    if ticket_total_num and ticket_total_num in full_df.columns:
        aggregations["revenue"] = (ticket_total_num, "sum")
        aggregations["avg_price"] = (ticket_total_num, "mean")

    summary = (
        full_df.groupby("ticket_category", dropna=False)
        .agg(**aggregations)
        .reset_index()
        .sort_values(["tickets", "ticket_category"], ascending=[False, True])
        .reset_index(drop=True)
    )

    total_row: Dict[str, object] = {
        "ticket_category": "TOTAL",
        "tickets": int(summary["tickets"].sum()),
    }
    if "revenue" in summary.columns:
        total_revenue = float(summary["revenue"].sum())
        total_tickets = int(total_row["tickets"])
        total_row["revenue"] = total_revenue
        total_row["avg_price"] = total_revenue / total_tickets if total_tickets else np.nan
        summary["revenue"] = summary["revenue"].round(2)
        summary["avg_price"] = summary["avg_price"].round(2)

    return pd.concat([summary, pd.DataFrame([total_row])], ignore_index=True)


def plot_full_festival_ticket_type_summary(
    summary: pd.DataFrame,
    plots_dir: Path,
    plot_format: str,
) -> None:
    if summary.empty:
        print("\nNessun dato disponibile per il grafico full festival ticket types.")
        return

    ordered = summary[summary["ticket_category"] != "TOTAL"].copy()
    ordered = ordered.sort_values(["tickets", "ticket_category"], ascending=[True, True]).reset_index(drop=True)
    total_tickets = int(ordered["tickets"].sum()) if not ordered.empty else 0

    fig, ax = plt.subplots(figsize=(12, max(4.8, 0.52 * len(ordered))))
    ordered.plot(kind="barh", x="ticket_category", y="tickets", ax=ax, color="#2e7d32", legend=False)
    ax.set_title("Full festival pass by ticket type")
    ax.set_xlabel("Tickets")
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.25)
    if not ordered.empty:
        max_tickets = int(ordered["tickets"].max())
        ax.set_xlim(0, max_tickets * 1.18)
        ax.bar_label(ax.containers[0], padding=3, fmt="%.0f")
    fig.text(
        0.985,
        0.02,
        f"TOTAL = {total_tickets}",
        ha="right",
        va="bottom",
        fontsize=13,
        bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "#999999", "boxstyle": "round,pad=0.35"},
    )
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    save_plot(fig, plots_dir, "full_festival_ticket_types", plot_format)


def export_full_festival_ticket_type_summary(
    df: pd.DataFrame,
    ticket_type_col: Optional[str],
    ticket_total_num: Optional[str],
    csv_dir: Path,
    plots_dir: Path,
    plot_format: str,
    plots_enabled: bool,
) -> None:
    if not ticket_type_col or ticket_type_col not in df.columns:
        return

    summary = build_full_festival_ticket_type_summary(df, ticket_type_col, ticket_total_num)
    if summary.empty:
        print("\nNessun dato full festival disponibile per export dedicato.")
        return

    csv_path = csv_dir / "full_festival_ticket_types.csv"
    summary.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"\nRiepilogo full festival ticket types salvato in: {csv_path}")
    if plots_enabled:
        plot_full_festival_ticket_type_summary(summary, plots_dir, plot_format)


def build_filtered_ticket_type_summary(
    df: pd.DataFrame,
    ticket_type_col: str,
    ticket_total_num: Optional[str],
    mask: pd.Series,
) -> pd.DataFrame:
    filtered = df.loc[mask].copy()
    if filtered.empty:
        return pd.DataFrame()

    filtered["ticket_category"] = filtered[ticket_type_col].fillna("").map(focused_ticket_category)
    aggregations: Dict[str, object] = {"tickets": ("ticket_category", "size")}
    if ticket_total_num and ticket_total_num in filtered.columns:
        aggregations["revenue"] = (ticket_total_num, "sum")
        aggregations["avg_price"] = (ticket_total_num, "mean")

    summary = (
        filtered.groupby("ticket_category", dropna=False)
        .agg(**aggregations)
        .reset_index()
        .sort_values(["tickets", "ticket_category"], ascending=[False, True])
        .reset_index(drop=True)
    )

    total_row: Dict[str, object] = {
        "ticket_category": "TOTAL",
        "tickets": int(summary["tickets"].sum()),
    }
    if "revenue" in summary.columns:
        total_revenue = float(summary["revenue"].sum())
        total_tickets = int(total_row["tickets"])
        total_row["revenue"] = round(total_revenue, 2)
        total_row["avg_price"] = round(total_revenue / total_tickets, 2) if total_tickets else np.nan
        summary["revenue"] = summary["revenue"].round(2)
        summary["avg_price"] = summary["avg_price"].round(2)

    return pd.concat([summary, pd.DataFrame([total_row])], ignore_index=True)


def plot_ticket_type_summary_barh(
    summary: pd.DataFrame,
    plots_dir: Path,
    plot_name: str,
    title: str,
    color: str,
    plot_format: str,
) -> None:
    if summary.empty:
        print(f"\nNessun dato disponibile per il grafico {plot_name}.")
        return

    ordered = summary[summary["ticket_category"] != "TOTAL"].copy()
    ordered = ordered.sort_values(["tickets", "ticket_category"], ascending=[True, True]).reset_index(drop=True)
    total_tickets = int(ordered["tickets"].sum()) if not ordered.empty else 0

    fig, ax = plt.subplots(figsize=(12, max(4.8, 0.52 * len(ordered))))
    ordered.plot(kind="barh", x="ticket_category", y="tickets", ax=ax, color=color, legend=False)
    ax.set_title(title)
    ax.set_xlabel("Tickets")
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.25)
    if not ordered.empty:
        max_tickets = int(ordered["tickets"].max())
        ax.set_xlim(0, max_tickets * 1.18)
        ax.bar_label(ax.containers[0], padding=3, fmt="%.0f")
    fig.text(
        0.985,
        0.02,
        f"TOTAL = {total_tickets}",
        ha="right",
        va="bottom",
        fontsize=13,
        bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "#999999", "boxstyle": "round,pad=0.35"},
    )
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    save_plot(fig, plots_dir, plot_name, plot_format)


def export_accessory_ticket_type_summary(
    df: pd.DataFrame,
    ticket_type_col: Optional[str],
    ticket_total_num: Optional[str],
    csv_dir: Path,
    plots_dir: Path,
    plot_format: str,
    plots_enabled: bool,
) -> None:
    if not ticket_type_col or ticket_type_col not in df.columns:
        return

    accessory_mask = df[ticket_type_col].map(is_accessory_ticket)
    summary = build_filtered_ticket_type_summary(df, ticket_type_col, ticket_total_num, accessory_mask)
    if summary.empty:
        print("\nNessun dato accessori disponibile per export dedicato.")
        return

    csv_path = csv_dir / "accessory_ticket_types.csv"
    summary.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"\nRiepilogo accessory ticket types salvato in: {csv_path}")
    if plots_enabled:
        plot_ticket_type_summary_barh(
            summary,
            plots_dir,
            "accessory_ticket_types",
            "Accessory ticket types",
            "#6d4c41",
            plot_format,
        )


def add_timeline_markers(ax: plt.Axes, markers: List[Dict[str, object]]) -> None:
    """Add vertical lines with labels to a time-based axis."""
    if not markers:
        return
    handles = []
    for marker in markers:
        mdate = marker["date"].date()
        line = ax.axvline(
            mdate,
            color=marker["color"],
            linestyle="--",
            alpha=0.8,
            linewidth=1.2,
            label=marker["label"],
        )
        handles.append(line)
    return handles


def plot_sales_timelines(
    daily_counts: pd.Series,
    markers: List[Dict[str, object]],
    plots_dir: Path,
    fmt: str,
) -> None:
    """Plot vendite giornaliere e cumulative con marker delle fasi."""
    fig, ax = plt.subplots(figsize=(15, 6.2))
    daily_counts.plot(ax=ax, marker="o", color="#388e3c", label="Vendite")
    ax.set_title("Biglietti venduti per giorno")
    ax.set_xlabel("Data")
    ax.set_ylabel("Biglietti")
    ax.grid(True, alpha=0.3)
    if not daily_counts.empty:
        start = daily_counts.index.min() - pd.Timedelta(days=21)
        end = daily_counts.index.max() + pd.Timedelta(days=21)
        ax.set_xlim(start, end)
    handles = add_timeline_markers(ax, markers)
    if not daily_counts.empty:
        daily_max = int(daily_counts.max())
        ax.set_ylim(0, max(1, int(np.ceil(daily_max * 1.35))))
        ax.yaxis.set_major_locator(MultipleLocator(5))
    if handles:
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0.0, fontsize=10)
    fig.tight_layout(rect=[0, 0, 0.8, 1])
    save_plot(fig, plots_dir, "vendite_giornaliere", fmt)

    fig, ax = plt.subplots(figsize=(15, 6.2))
    daily_counts.cumsum().plot(ax=ax, marker="o", markersize=3, color="#d32f2f", label="Cumulato")
    ax.set_title("Biglietti cumulati")
    ax.set_xlabel("Data")
    ax.set_ylabel("Cumulato")
    ax.grid(True, alpha=0.3)
    if not daily_counts.empty:
        start = daily_counts.index.min() - pd.Timedelta(days=21)
        end = daily_counts.index.max() + pd.Timedelta(days=21)
        ax.set_xlim(start, end)
    handles = add_timeline_markers(ax, markers)
    if not daily_counts.empty:
        cumulative_max = int(daily_counts.cumsum().max())
        ax.set_ylim(0, max(1, int(np.ceil(cumulative_max * 1.15))))
        ax.yaxis.set_major_locator(MultipleLocator(50))
    if handles:
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0.0, fontsize=10)
    fig.tight_layout(rect=[0, 0, 0.8, 1])
    save_plot(fig, plots_dir, "vendite_cumulative", fmt)


def build_monthly_ticket_sales_summary(
    df: pd.DataFrame,
    ticket_type_col: Optional[str],
) -> pd.DataFrame:
    """Monthly ticket totals and daily averages for all tickets and full festival passes."""
    if PARSED_DATE_COL not in df.columns:
        return pd.DataFrame()

    ts = df.dropna(subset=[PARSED_DATE_COL]).copy()
    if ts.empty:
        return pd.DataFrame()

    ts["_sale_date"] = ts[PARSED_DATE_COL].dt.normalize()
    ts["_sale_month"] = ts[PARSED_DATE_COL].dt.to_period("M")
    min_date = ts["_sale_date"].min().normalize()
    max_date = ts["_sale_date"].max().normalize()
    months = pd.period_range(min_date.to_period("M"), max_date.to_period("M"), freq="M")

    if ticket_type_col and ticket_type_col in ts.columns:
        full_festival_mask = ts[ticket_type_col].map(is_full_festival_pass_ticket)
    else:
        full_festival_mask = pd.Series(False, index=ts.index)

    all_monthly = ts.groupby("_sale_month").size()
    all_active_days = ts.groupby("_sale_month")["_sale_date"].nunique()
    full_ts = ts.loc[full_festival_mask].copy()
    full_monthly = full_ts.groupby("_sale_month").size() if not full_ts.empty else pd.Series(dtype=int)
    full_active_days = (
        full_ts.groupby("_sale_month")["_sale_date"].nunique()
        if not full_ts.empty
        else pd.Series(dtype=int)
    )

    rows: List[Dict[str, object]] = []
    for month in months:
        month_start = max(month.start_time.normalize(), min_date)
        month_end = min(month.end_time.normalize(), max_date)
        observed_days = int((month_end - month_start).days + 1) if month_end >= month_start else 0

        all_tickets = int(all_monthly.get(month, 0))
        full_tickets = int(full_monthly.get(month, 0))
        all_days = int(all_active_days.get(month, 0))
        full_days = int(full_active_days.get(month, 0))

        rows.append(
            {
                "month": str(month),
                "observed_days": observed_days,
                "all_tickets": all_tickets,
                "all_active_sales_days": all_days,
                "all_avg_per_observed_day": all_tickets / observed_days if observed_days else 0.0,
                "all_avg_per_active_sales_day": all_tickets / all_days if all_days else 0.0,
                "full_festival_tickets": full_tickets,
                "full_festival_active_sales_days": full_days,
                "full_festival_avg_per_observed_day": full_tickets / observed_days if observed_days else 0.0,
                "full_festival_avg_per_active_sales_day": full_tickets / full_days if full_days else 0.0,
                "full_festival_share_pct": (full_tickets / all_tickets * 100) if all_tickets else 0.0,
            }
        )

    summary = pd.DataFrame(rows)
    float_cols = [
        "all_avg_per_observed_day",
        "all_avg_per_active_sales_day",
        "full_festival_avg_per_observed_day",
        "full_festival_avg_per_active_sales_day",
        "full_festival_share_pct",
    ]
    for col in float_cols:
        summary[col] = summary[col].round(2)
    return summary


def _label_monthly_bars(ax: plt.Axes, decimals: int = 1) -> None:
    fmt = f"%.{decimals}f"
    for container in ax.containers:
        ax.bar_label(container, padding=3, fmt=fmt, fontsize=9)


def _plot_single_monthly_average(
    summary: pd.DataFrame,
    value_col: str,
    title: str,
    color: str,
    plots_dir: Path,
    name: str,
    fmt: str,
) -> None:
    fig_width = max(10.0, 0.72 * len(summary) + 3.0)
    fig, ax = plt.subplots(figsize=(fig_width, 5.6))
    x_labels = summary["month"].astype(str)
    values = summary[value_col].astype(float)
    ax.bar(x_labels, values, color=color, width=0.78)
    ax.set_title(title)
    ax.set_xlabel("Mese")
    ax.set_ylabel("Biglietti / giorno osservato")
    ax.grid(axis="y", alpha=0.25)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1f}"))
    if not values.empty and values.max() > 0:
        ax.set_ylim(0, values.max() * 1.22)
    ax.tick_params(axis="x", rotation=45)
    _label_monthly_bars(ax, decimals=1)
    fig.tight_layout()
    save_plot(fig, plots_dir, name, fmt)


def plot_monthly_ticket_sales_summary(
    summary: pd.DataFrame,
    plots_dir: Path,
    fmt: str,
) -> None:
    if summary.empty:
        print("\nNessun dato disponibile per i grafici mensili.")
        return

    _plot_single_monthly_average(
        summary,
        "all_avg_per_observed_day",
        "Media giornaliera biglietti venduti per mese - tutti i ticket",
        "#1565c0",
        plots_dir,
        "vendite_mensili_media_giornaliera_tutti_ticket",
        fmt,
    )
    _plot_single_monthly_average(
        summary,
        "full_festival_avg_per_observed_day",
        "Media giornaliera full festival pass venduti per mese",
        "#2e7d32",
        plots_dir,
        "vendite_mensili_media_giornaliera_full_festival",
        fmt,
    )

    x = np.arange(len(summary))
    width = 0.38
    labels = summary["month"].astype(str).tolist()

    fig_width = max(11.0, 0.8 * len(summary) + 3.0)
    fig, ax = plt.subplots(figsize=(fig_width, 5.8))
    all_avg = summary["all_avg_per_observed_day"].astype(float)
    full_avg = summary["full_festival_avg_per_observed_day"].astype(float)
    ax.bar(x - width / 2, all_avg, width, label="Tutti i ticket", color="#1565c0")
    ax.bar(x + width / 2, full_avg, width, label="Full festival pass", color="#2e7d32")
    ax.set_title("Media giornaliera biglietti venduti per mese")
    ax.set_xlabel("Mese")
    ax.set_ylabel("Biglietti / giorno osservato")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    max_value = max(float(all_avg.max()), float(full_avg.max())) if not summary.empty else 0.0
    if max_value > 0:
        ax.set_ylim(0, max_value * 1.22)
    _label_monthly_bars(ax, decimals=1)
    fig.tight_layout()
    save_plot(fig, plots_dir, "vendite_mensili_media_giornaliera_confronto", fmt)

    fig, ax = plt.subplots(figsize=(fig_width, 5.8))
    all_totals = summary["all_tickets"].astype(int)
    full_totals = summary["full_festival_tickets"].astype(int)
    ax.bar(x - width / 2, all_totals, width, label="Tutti i ticket", color="#5c6bc0")
    ax.bar(x + width / 2, full_totals, width, label="Full festival pass", color="#00897b")
    ax.set_title("Biglietti venduti per mese")
    ax.set_xlabel("Mese")
    ax.set_ylabel("Biglietti")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    max_total = max(int(all_totals.max()), int(full_totals.max())) if not summary.empty else 0
    if max_total > 0:
        ax.set_ylim(0, max_total * 1.18)
    _label_monthly_bars(ax, decimals=0)
    fig.tight_layout()
    save_plot(fig, plots_dir, "vendite_mensili_totali_confronto", fmt)


def analyze_geography(
    df: pd.DataFrame,
    geo_country_cols: List[str],
    geo_city_cols: List[str],
    plots_enabled: bool,
    plots_dir: Path,
    plot_format: str,
) -> None:
    """Analisi e plot per geografia (paesi/citta)."""
    geo_report_cols = list(dict.fromkeys(geo_country_cols + geo_city_cols))
    report_missing(df, geo_report_cols, "Geografia")

    for col in geo_country_cols:
        by_country = df.groupby(col, dropna=False).size().sort_values(ascending=False)
        print(f"\nTop paesi ({col}):")
        print(by_country.head(20))
        if plots_enabled:
            country_plot = drop_nan_categories(by_country).head(15)
            if country_plot.empty:
                print(f" - Nessun dato valido per il grafico paesi ({col}).")
            else:
                fig, ax = plt.subplots(figsize=(10, 6))
                country_plot.plot(kind="barh", ax=ax, color="#6a1b9a")
                ax.set_title("")
                ax.set_xlabel("Biglietti")
                ax.set_ylabel("Paese di provenienza")
                for bar in ax.patches:
                    bar.set_height(0.9)
                ax.tick_params(axis="both", labelsize=12)
                ax.xaxis.label.set_size(13)
                ax.yaxis.label.set_size(13)
                fig.tight_layout()
                save_plot(fig, plots_dir, f"top_paesi_{slugify(col)}", plot_format)
            cleaned = drop_nan_categories(by_country)
            if not cleaned.empty:
                labels_norm = cleaned.index.astype(str).str.strip().str.lower()
                italy_mask = labels_norm.isin({"italia", "italy"})
                italy_count = int(cleaned[italy_mask].sum())
                abroad_count = int(cleaned[~italy_mask].sum())
                if italy_count + abroad_count > 0:
                    fig, ax = plt.subplots(figsize=(5.5, 5))
                    pie_sizes = [italy_count, abroad_count]
                    pie_labels = ["Italy", "Abroad"]
                    pie_colors = ["#00b050", "#1565c0"]

                    def make_autopct(labels: list[str]) -> callable:
                        def _autopct(pct: float) -> str:
                            if not labels:
                                return f"{pct:.1f}%"
                            label = labels.pop(0)
                            return f"{label}\n{pct:.1f}%"

                        return _autopct

                    pie_common = dict(
                        labels=None,
                        autopct=make_autopct(pie_labels.copy()),
                        startangle=90,
                        counterclock=False,
                        colors=pie_colors,
                        explode=(0.02, 0.02),
                        wedgeprops={"edgecolor": "white"},
                        pctdistance=0.55,
                    )

                    fig, ax = plt.subplots(figsize=(5.5, 5))
                    ax.pie(pie_sizes, textprops={"fontsize": 12, "weight": "bold", "color": "#111"}, **pie_common)
                    ax.axis("equal")
                    ax.set_title("Italy vs Abroad")
                    fig.tight_layout()
                    save_plot(fig, plots_dir, f"italy_abroad_{slugify(col)}_black", plot_format)

                    fig, ax = plt.subplots(figsize=(5.5, 5))
                    ax.pie(
                        pie_sizes,
                        textprops={"fontsize": 12, "weight": "bold", "color": "#fff"},
                        **{**pie_common, "autopct": make_autopct(pie_labels.copy())},
                    )
                    ax.axis("equal")
                    ax.set_title("Italy vs Abroad")
                    fig.tight_layout()
                    save_plot(fig, plots_dir, f"italy_abroad_{slugify(col)}_white", plot_format)

            cleaned = drop_nan_categories(by_country)
            if not cleaned.empty:
                labels_norm = cleaned.index.astype(str).str.strip().str.lower()
                italy_mask = labels_norm.isin({"italia", "italy"})
                italy_count = int(cleaned[italy_mask].sum())
                abroad_count = int(cleaned[~italy_mask].sum())
                if italy_count + abroad_count > 0:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.pie(
                        [italy_count, abroad_count],
                        labels=["Italy", "Abroad"],
                        autopct="%1.1f%%",
                        colors=["#2e7d32", "#1565c0"],
                    )
                    ax.set_title("Italy vs Abroad")
                    fig.tight_layout()
                    save_plot(fig, plots_dir, f"italia_estero_{slugify(col)}", plot_format)

            # Pie chart Italy vs abroad (esclude valori NaN/vuoti)
            cleaned = drop_nan_categories(by_country)
            if not cleaned.empty:
                labels_norm = cleaned.index.astype(str).str.strip().str.lower()
                italy_mask = labels_norm.isin({"italia", "italy"})
                italy_count = int(cleaned[italy_mask].sum())
                abroad_count = int(cleaned[~italy_mask].sum())
                if italy_count + abroad_count > 0:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.pie(
                        [italy_count, abroad_count],
                        labels=["Italy", "Abroad"],
                        autopct="%1.1f%%",
                        colors=["#2e7d32", "#1565c0"],
                    )
                    ax.set_title("Italy vs Abroad")
                    fig.tight_layout()
                    save_plot(fig, plots_dir, f"italia_estero_{slugify(col)}", plot_format)

    for col in geo_city_cols:
        by_city = df.groupby(col, dropna=False).size().sort_values(ascending=False)
        print(f"\nTop cittÃ  ({col}):")
        print(by_city.head(20))
        if plots_enabled:
            city_plot = drop_nan_categories(by_city).head(15)
            if city_plot.empty:
                print(f" - Nessun dato valido per il grafico cittÃ  ({col}).")
            else:
                fig, ax = plt.subplots(figsize=(10, 7))
                city_plot.plot(kind="barh", ax=ax, color="#00838f")
                ax.set_title("")
                ax.set_xlabel("Biglietti")
                ax.set_ylabel("CittÃ  di provenienza")
                for bar in ax.patches:
                    bar.set_height(0.9)
                ax.tick_params(axis="both", labelsize=12)
                ax.xaxis.label.set_size(13)
                ax.yaxis.label.set_size(13)
                fig.tight_layout()
                save_plot(fig, plots_dir, f"top_citta_{slugify(col)}", plot_format)


def export_summary_tables(
    df: pd.DataFrame,
    geo_country_cols: List[str],
    geo_city_cols: List[str],
    ticket_type_col: Optional[str],
    ticket_total_num: Optional[str],
    payment_gateway_col: Optional[str],
    csv_dir: Path,
    plots_dir: Path,
    plot_format: str,
) -> None:
    """Esporta CSV di riepilogo."""
    exports: Dict[str, pd.DataFrame] = {}
    if ticket_type_col in df.columns:
        unit_price_col = "_ticket_type_amount_eur"
        row_amount_col = "_row_amount_eur"
        df[unit_price_col] = df[ticket_type_col].map(extract_ticket_type_amount)

        def normalize_label(label: str) -> str:
            normalized = label.lower()
            normalized = normalized.replace("\u2013", "-").replace("\u2014", "-")
            normalized = " ".join(normalized.split())
            return normalized

        def bundle_size_from_label(label: str) -> Optional[int]:
            match = re.search(
                r"christmas bundle\s*-\s*(\d+)\s+full festival pass",
                normalize_label(label),
            )
            if match:
                try:
                    return int(match.group(1))
                except ValueError:
                    return None
            return None

        def adjust_unit_price(label: str, price: float) -> float:
            normalized = normalize_label(label or "")
            if "caravan pass" in normalized and "base" in normalized:
                return 25.0
            size = bundle_size_from_label(label or "")
            if size and price:
                return price / size
            return price

        if ticket_total_num and ticket_total_num in df.columns:
            df[row_amount_col] = df[ticket_total_num].fillna(df[unit_price_col])
        else:
            df[row_amount_col] = df[unit_price_col]

        by_type = (
            df.groupby(ticket_type_col, dropna=False)
            .agg(
                tickets=(ticket_type_col, "size"),
                revenue=(row_amount_col, "sum"),
                unit_price_eur=(
                    unit_price_col,
                    lambda s: s.dropna().iloc[0] if not s.dropna().empty else np.nan,
                ),
            )
            .sort_values(["tickets"], ascending=False)
        )
        by_type["unit_price_eur"] = [
            adjust_unit_price(str(label), price)
            if not pd.isna(price)
            else adjust_unit_price(str(label), np.nan)
            for label, price in zip(by_type.index, by_type["unit_price_eur"])
        ]
        by_type["expected_amount_eur"] = by_type["tickets"] * by_type["unit_price_eur"]
        by_type["diff_amount_eur"] = by_type["revenue"] - by_type["expected_amount_eur"]
        by_type["total_ass_eur"] = [
            0.0 if "caravan pass" in normalize_label(str(label)) else tickets * 25.0
            for label, tickets in zip(by_type.index, by_type["tickets"])
        ]
        by_type["total_festival_eur"] = by_type["revenue"] - by_type["total_ass_eur"]
        total_row = pd.DataFrame(
            {
                "tickets": [by_type["tickets"].sum()],
                "revenue": [by_type["revenue"].sum()],
                "unit_price_eur": [np.nan],
                "expected_amount_eur": [by_type["expected_amount_eur"].sum()],
                "diff_amount_eur": [by_type["diff_amount_eur"].sum()],
                "total_ass_eur": [by_type["total_ass_eur"].sum()],
                "total_festival_eur": [by_type["total_festival_eur"].sum()],
            },
            index=["TOTAL"],
        )
        by_type = pd.concat([by_type, total_row])
        by_type = by_type.rename(
            columns={
                "revenue": "Total Amount (€)",
                "unit_price_eur": "Unit Price (€)",
                "expected_amount_eur": "Expected Amount (€)",
                "diff_amount_eur": "Diff (€)",
                "total_ass_eur": "Total Ass (€)",
                "total_festival_eur": "Total Festival (€)",
            }
        )
        ordered_cols = [
            "tickets",
            "Unit Price (€)",
            "Total Ass (€)",
            "Total Festival (€)",
            "Total Amount (€)",
            "Expected Amount (€)",
            "Diff (€)",
        ]
        by_type = by_type[[c for c in ordered_cols if c in by_type.columns]]
        for col in [
            "Total Amount (€)",
            "Unit Price (€)",
            "Expected Amount (€)",
            "Diff (€)",
            "Total Ass (€)",
            "Total Festival (€)",
        ]:
            by_type[col] = by_type[col].map(format_eur)
        exports["by_type.csv"] = by_type
    for idx, col in enumerate(geo_country_cols):
        fname = "by_country.csv" if idx == 0 else f"by_country_{slugify(col)}.csv"
        exports[fname] = (
            df.groupby(col, dropna=False)
            .size()
            .to_frame("tickets")
            .sort_values("tickets", ascending=False)
        )
    for idx, col in enumerate(geo_city_cols):
        fname = "by_city.csv" if idx == 0 else f"by_city_{slugify(col)}.csv"
        exports[fname] = (
            df.groupby(col, dropna=False)
            .size()
            .to_frame("tickets")
            .sort_values("tickets", ascending=False)
        )
    if payment_gateway_col and payment_gateway_col in df.columns:
        by_gateway = (
            df.groupby(payment_gateway_col, dropna=False)
            .agg(
                tickets=(payment_gateway_col, "size"),
                revenue=(ticket_total_num, "sum") if ticket_total_num in df.columns else (payment_gateway_col, "size"),
            )
            .sort_values(["tickets"], ascending=False)
        )
        total_row = pd.DataFrame(
            {
                "tickets": [by_gateway["tickets"].sum()],
                "revenue": [by_gateway["revenue"].sum()],
            },
            index=["TOTAL"],
        )
        by_gateway = pd.concat([by_gateway, total_row])
        by_gateway = by_gateway.rename(columns={"revenue": "Total Amount (€)"})
        by_gateway["Total Amount (€)"] = by_gateway["Total Amount (€)"].map(format_eur)
        exports["by_payment_gateway.csv"] = by_gateway

    for name, table in exports.items():
        out_path = csv_dir / name
        table.to_csv(out_path, encoding="utf-8")
        print(f"Esportato: {out_path}")
        widen_first = False
        highlight = None
        narrow_numeric = False
        font_size = 8
        col_widths_override = None
        fig_width_override = None
        fig_height_override = None
        scale_x_override = None
        scale_y_override = None
        bbox_override = None
        manual_table = False
        header_font_size = None
        header_labels_override = None
        row_height_override = None
        header_height_override = None
        dpi_override = None
        if name == "by_type.csv":
            widen_first = True
            highlight = "TOTAL"
            narrow_numeric = True
            font_size = 11
        if name == "ambassador_sales.csv":
            highlight = "TOTAL"
            narrow_numeric = True
            font_size = 36
            col_widths_override = [0.78, 0.10, 0.12]
            fig_width_override = 13.5
            fig_height_override = max(14.0, 1.4 * len(table))
            scale_x_override = None
            scale_y_override = None
            bbox_override = None
            header_font_size = 30
            header_labels_override = ["", "tickets", "Total €"]
            row_height_override = 3.6
            header_height_override = 4.0
            manual_table = True
            dpi_override = 300
        if name == "by_payment_gateway.csv":
            highlight = "TOTAL"
        save_table_image(
            table,
            plots_dir,
            f"table_{Path(name).stem}",
            plot_format,
            highlight_value=highlight,
            widen_first_col=widen_first,
            narrow_numeric_cols=narrow_numeric,
            col_widths_override=col_widths_override,
            font_size=font_size,
            fig_width_override=fig_width_override,
            fig_height_override=fig_height_override,
            scale_x_override=scale_x_override,
            scale_y_override=scale_y_override,
            bbox_override=bbox_override,
            manual_table=manual_table,
            header_font_size=header_font_size,
            header_labels_override=header_labels_override,
            row_height_override=row_height_override,
            header_height_override=header_height_override,
            dpi_override=dpi_override,
        )


def normalize_and_prepare_columns(
    df_raw: pd.DataFrame,
    columns_cfg: Dict[str, str],
    extra_country_cfg: Iterable[str],
    extra_city_cfg: Iterable[str],
    checkin_columns: Iterable[str],
) -> Dict[str, object]:
    """Normalizza colonne, risolve i nomi e converte numerici/email."""
    df = df_raw.copy()
    df.columns = normalize_columns(df.columns)

    def col_value(key: str, fallback: Optional[str]) -> Optional[str]:
        value = columns_cfg.get(key)
        return value or fallback

    def ensure_existing(name: Optional[str], *alternatives: Optional[str]) -> Optional[str]:
        candidates = (name,) + alternatives
        for candidate in candidates:
            if candidate and candidate in df.columns:
                return candidate
        return None

    def resolve_existing_list(names: Iterable[Optional[str]]) -> List[str]:
        resolved: List[str] = []
        for name in names:
            if not name:
                continue
            alt = name.replace("\u00e0", "\ufffd") if "\u00e0" in name else None
            found = ensure_existing(name, alt)
            if found and found in df.columns and found not in resolved:
                resolved.append(found)
        return resolved

    def to_list(value: object) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        try:
            return [v for v in value if v is not None]
        except TypeError:
            return []

    payment_date_col = ensure_existing(col_value("payment_date", "Payment Date"))
    ticket_type_col = ensure_existing(col_value("ticket_type", "Ticket Type"))
    ticket_total_col = ensure_existing(col_value("ticket_total", "Ticket Total"))
    order_total_col = ensure_existing(col_value("order_total", "Order Total"))
    payment_gateway_col = ensure_existing(col_value("payment_gateway", "Payment Gateway"))
    discount_col = ensure_existing(col_value("discount_code", "Discount Code"))
    ticket_discount_col = ensure_existing(col_value("ticket_discount", "Ticket Discount"))
    country_col = ensure_existing(
        col_value("country", "Country of residence / Paese di residenza (Campi ticket holder)")
    )
    city_preferred = col_value("city", "City of residence / Citt\u00e0 di residenza (Campi ticket holder)")
    city_col = ensure_existing(
        city_preferred,
        city_preferred.replace("\u00e0", "\ufffd") if city_preferred else None,
    )
    geo_country_cols = resolve_existing_list([country_col] + to_list(extra_country_cfg))
    geo_city_cols = resolve_existing_list([city_col] + to_list(extra_city_cfg))
    attendee_email_col = ensure_existing(col_value("attendee_email", "Attendee E-mail"))
    buyer_email_col = ensure_existing(col_value("buyer_email", "Buyer E-Mail"))
    order_number_col = ensure_existing(col_value("order_number", "Order Number"))
    order_status_col = ensure_existing(col_value("order_status", "Order Status"))
    payment_gateway_col = ensure_existing(col_value("payment_gateway", "Payment Gateway"))
    ticket_id_col = ensure_existing(col_value("ticket_id", "Ticket ID"))
    payment_gateway_col = ensure_existing(col_value("payment_gateway", "Payment Gateway"))
    ticket_id_col = ensure_existing(col_value("ticket_id", "Ticket ID"))
    payment_gateway_col = ensure_existing(col_value("payment_gateway", "Payment Gateway"))
    ticket_id_col = ensure_existing(col_value("ticket_id", "Ticket ID"))
    payment_gateway_col = ensure_existing(col_value("payment_gateway", "Payment Gateway"))
    ticket_id_col = ensure_existing(col_value("ticket_id", "Ticket ID"))
    ticket_id_col = ensure_existing(col_value("ticket_id", "Ticket ID"))
    dob_preferred = col_value("date_of_birth", "Date of birth / Data di nascita (Campi ticket holder)")
    dob_col = ensure_existing(
        dob_preferred,
        dob_preferred.replace("\u00e0", "\ufffd") if dob_preferred else None,
    )

    if payment_date_col and payment_date_col in df.columns:
        df[PARSED_DATE_COL] = df[payment_date_col].map(parse_payment_date)
    else:
        df[PARSED_DATE_COL] = pd.NaT

    if ticket_type_col and payment_date_col:
        lovers_color = marker_color_by_label(parsed_timeline_markers, "lovers bundle", LOVERS_BUNDLE_COLOR)
        parsed_timeline_markers.extend(
            build_dynamic_bundle_markers(
                df,
                PARSED_DATE_COL,
                ticket_type_col,
                CHRISTMAS_BUNDLE_KEYWORD,
                lovers_color,
            )
        )

    numeric_targets = set(NUMERIC_CANDIDATES)
    numeric_targets.update(
        [
            ticket_total_col,
            order_total_col,
            ticket_discount_col,
            col_value("ticket_subtotal", "Ticket Subtotal"),
            col_value("ticket_fee", "Ticket Fee"),
            col_value("price", "Price"),
        ]
    )
    numeric_map: Dict[str, str] = {}
    for col_name in filter(None, numeric_targets):
        if col_name in df.columns:
            num_col = f"{col_name}_num"
            df[num_col] = df[col_name].map(to_num)
            numeric_map[col_name] = num_col

    for email_col in filter(None, [attendee_email_col, buyer_email_col]):
        if email_col in df.columns:
            df[email_col] = df[email_col].astype(str).str.strip().str.lower()

    if not attendee_email_col:
        print("\nALERT: colonna attendee email mancante; i controlli basati su Attendee E-mail verranno saltati.")

    present_checkin_cols = [c for c in checkin_columns if c in df.columns]

    return {
        "df": df,
        "payment_date_col": payment_date_col,
        "ticket_type_col": ticket_type_col,
        "ticket_total_col": ticket_total_col,
        "order_total_col": order_total_col,
        "payment_gateway_col": payment_gateway_col,
        "discount_col": discount_col,
        "ticket_discount_col": ticket_discount_col,
        "country_col": country_col,
        "city_col": city_col,
        "geo_country_cols": geo_country_cols,
        "geo_city_cols": geo_city_cols,
        "attendee_email_col": attendee_email_col,
        "buyer_email_col": buyer_email_col,
        "order_number_col": order_number_col,
        "order_status_col": order_status_col,
        "ticket_id_col": ticket_id_col,
        "dob_col": dob_col,
        "numeric_map": numeric_map,
        "present_checkin_cols": present_checkin_cols,
    }


def parse_birth_date(value: object) -> pd.Timestamp:
    if pd.isna(value):
        return pd.NaT
    s = str(value).strip()
    if not s:
        return pd.NaT
    for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d"):
        try:
            return pd.Timestamp(datetime.strptime(s, fmt))
        except ValueError:
            continue
    return pd.NaT


def parse_timeline_markers(raw_markers: Iterable[object]) -> List[Dict[str, object]]:
    parsed: List[Dict[str, object]] = []
    for item in raw_markers:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", "")).strip()
        date_str = str(item.get("date", "")).strip()
        color = item.get("color") or "#ef6c00"
        if not label or not date_str:
            continue
        label_lower = label.lower()
        if LINEUP_RELEASE_KEYWORD in label_lower:
            continue
        if "early bird" in label_lower or "phase" in label_lower:
            color = PHASE_MARKER_COLOR
        elif "lovers bundle" in label_lower:
            color = LOVERS_BUNDLE_COLOR
        ts = pd.to_datetime(date_str, errors="coerce")
        if pd.isna(ts):
            continue
        parsed.append({"label": label, "date": ts.normalize(), "color": color})
    return parsed


def marker_color_by_label(markers: Iterable[Dict[str, object]], label_fragment: str, default: str) -> str:
    fragment = label_fragment.lower()
    for marker in markers:
        label = str(marker.get("label", "")).lower()
        if fragment in label:
            return str(marker.get("color") or default)
    return default


def build_dynamic_bundle_markers(
    df: pd.DataFrame,
    date_col: str,
    ticket_type_col: str,
    keyword: str,
    base_color: str,
) -> List[Dict[str, object]]:
    if date_col not in df.columns or ticket_type_col not in df.columns:
        return []
    mask = df[ticket_type_col].fillna("").astype(str).str.contains(keyword, case=False, na=False)
    if not bool(mask.any()):
        return []
    dates = pd.to_datetime(df.loc[mask, date_col], errors="coerce").dropna()
    if dates.empty:
        return []
    start = pd.Timestamp(dates.min()).normalize()
    end = pd.Timestamp(dates.max()).normalize()
    return [
        {"label": "Start Christmas Bundle", "date": start.strftime("%Y-%m-%d"), "color": base_color},
        {"label": "End Christmas Bundle", "date": end.strftime("%Y-%m-%d"), "color": base_color},
    ]


def sync_christmas_bundle_markers(
    config: Dict[str, object],
    config_path: Path,
    christmas_markers: List[Dict[str, object]],
) -> List[Dict[str, object]]:
    raw_markers = list(config.get("timeline_markers", []) or [])
    filtered = []
    for item in raw_markers:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", "")).strip().lower()
        if label in {"start christmas bundle", "end christmas bundle"}:
            continue
        filtered.append(item)

    updated_markers = filtered + christmas_markers
    config["timeline_markers"] = updated_markers
    save_config(config_path, config)
    return updated_markers


def write_missing_report(
    df_full: pd.DataFrame,
    destination: Path,
    df_paid: Optional[pd.DataFrame] = None,
) -> None:
    total = len(df_full)
    stats = []
    for col in df_full.columns:
        missing = missing_count(df_full[col])
        filled = total - missing
        percent = (missing / total * 100) if total else 0
        stats.append((col, filled, missing, percent))
    stats.sort(key=lambda item: item[3], reverse=True)
    lines = [
        "Report valori mancanti per colonna",
        f"Totale righe nel CSV: {total}",
        "",
        "Formato: <colonna> | compilati | mancanti | % mancanti",
        "",
    ]
    for col, filled, missing, percent in stats:
        lines.append(f"{col} | {filled} | {missing} | {percent:.1f}%")

    if df_paid is not None:
        total_paid = len(df_paid)
        stats_paid = []
        for col in df_paid.columns:
            missing = missing_count(df_paid[col])
            filled = total_paid - missing
            percent = (missing / total_paid * 100) if total_paid else 0
            stats_paid.append((col, filled, missing, percent))
        stats_paid.sort(key=lambda item: item[3], reverse=True)

        lines.extend(
            [
                "",
                "Report valori mancanti per colonna (solo ordini Paid)",
                f"Totale righe considerate: {total_paid}",
                "",
                "Formato: <colonna> | compilati | mancanti | % mancanti",
                "",
            ]
        )
        for col, filled, missing, percent in stats_paid:
            lines.append(f"{col} | {filled} | {missing} | {percent:.1f}%")

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport dettagliato colonne salvato in: {destination}")


def report_missing(df: pd.DataFrame, columns: Iterable[Optional[str]], label: str) -> None:
    targets = [c for c in columns if c]
    if not targets:
        return
    total = len(df)
    print(f"\nValori mancanti - {label}:")
    for col in targets:
        if col not in df.columns:
            print(f" - {col}: colonna non presente nel file")
            continue
        missing = missing_count(df[col])
        percentage = (missing / total * 100) if total else 0
        print(f" - {col}: {missing} su {total} ({percentage:.1f}%)")


def save_plot(fig: plt.Figure, destination: Path, name: str, fmt: str) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    out_path = destination / f"{name}.{fmt}"
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Grafico salvato: {out_path}")
    plt.close(fig)


def normalize_marker_label(label: object) -> str:
    normalized = str(label or "").lower()
    normalized = normalized.replace("\u2013", "-").replace("\u2014", "-")
    normalized = " ".join(normalized.split())
    return normalized


def is_phase_timeline_marker(label: object) -> bool:
    normalized = normalize_marker_label(label)
    return bool(re.fullmatch(r"(early bird|phase\s*[0-9]+|phase final)", normalized))


def timeline_marker_to_bucket(label: object) -> str:
    normalized = normalize_marker_label(label)
    if normalized == "early bird":
        return "early_bird"
    match = re.fullmatch(r"phase\s*([0-9]+|final)", normalized)
    if match:
        return f"phase_{match.group(1)}"
    return normalized.replace(" ", "_")


def build_phase_window_summary(
    df: pd.DataFrame,
    timeline_markers: List[Dict[str, object]],
    ticket_type_col: Optional[str],
    ticket_total_num: Optional[str],
) -> pd.DataFrame:
    if not ticket_type_col or ticket_type_col not in df.columns:
        return pd.DataFrame()

    phase_markers: List[Dict[str, object]] = []
    for marker in timeline_markers:
        if not isinstance(marker, dict):
            continue
        if is_phase_timeline_marker(marker.get("label")):
            ts = pd.to_datetime(marker.get("date"), errors="coerce")
            if pd.notna(ts):
                phase_markers.append({"label": marker.get("label"), "date": ts.normalize()})
    phase_markers.sort(key=lambda item: item["date"])

    data_end = pd.NaT
    if PARSED_DATE_COL in df.columns:
        valid_dates = df[PARSED_DATE_COL].dropna()
        if not valid_dates.empty:
            data_end = valid_dates.max().normalize()

    phase_labels = df[ticket_type_col].fillna("").map(extract_phase_label)
    rows: List[Dict[str, object]] = []
    for idx, marker in enumerate(phase_markers):
        label = normalize_marker_label(marker["label"])
        bucket = timeline_marker_to_bucket(label)
        start = marker["date"]
        end = phase_markers[idx + 1]["date"] if idx + 1 < len(phase_markers) else pd.NaT
        if pd.notna(end):
            span_days = int((end - start).days)
            end_label = end.strftime("%d/%m/%Y")
        elif pd.notna(data_end) and data_end >= start:
            span_days = int((data_end - start).days) + 1
            end_label = f"ongoing ({data_end.strftime('%d/%m/%Y')})"
        else:
            span_days = np.nan
            end_label = "ongoing"
        mask = phase_labels.eq(bucket)
        tickets = int(mask.sum())
        revenue = (
            float(df.loc[mask, ticket_total_num].sum())
            if ticket_total_num and ticket_total_num in df.columns
            else np.nan
        )
        tickets_per_day = float(tickets / span_days) if pd.notna(span_days) and span_days > 0 else np.nan
        rows.append(
            {
                "phase": bucket,
                "start": start.strftime("%d/%m/%Y"),
                "end": end_label,
                "span_days": span_days,
                "tickets": tickets,
                "tickets/day": tickets_per_day,
                "revenue": revenue,
            }
        )

    phase_df = pd.DataFrame(rows)
    if phase_df.empty:
        return phase_df
    phase_df["revenue"] = phase_df["revenue"].map(format_eur)
    phase_df["tickets/day"] = phase_df["tickets/day"].map(lambda v: "" if pd.isna(v) else f"{v:.2f}")
    phase_df["span_days"] = phase_df["span_days"].map(lambda v: "" if pd.isna(v) else int(v))
    return phase_df


def build_bundle_summary(
    df: pd.DataFrame,
    timeline_markers: List[Dict[str, object]],
    ticket_type_col: Optional[str],
    ticket_total_num: Optional[str],
) -> pd.DataFrame:
    if not ticket_type_col or ticket_type_col not in df.columns:
        return pd.DataFrame()

    marker_map = {
        normalize_marker_label(marker.get("label")): pd.to_datetime(marker.get("date"), errors="coerce")
        for marker in timeline_markers
        if isinstance(marker, dict) and pd.notna(pd.to_datetime(marker.get("date"), errors="coerce"))
    }
    bundle_defs = [
        ("start christmas bundle", "end christmas bundle", "christmas bundle"),
        ("start lovers bundle", "end lovers bundle", "lovers bundle"),
    ]
    rows: List[Dict[str, object]] = []
    text_series = df[ticket_type_col].fillna("").astype(str).str.lower()
    for start_key, end_key, bucket_name in bundle_defs:
        start = marker_map.get(start_key)
        end = marker_map.get(end_key)
        if pd.isna(start) or pd.isna(end):
            continue
        mask = text_series.str.contains(bucket_name, case=False, na=False)
        tickets = int(mask.sum())
        revenue = (
            float(df.loc[mask, ticket_total_num].sum())
            if ticket_total_num and ticket_total_num in df.columns
            else np.nan
        )
        span_days = max(1, int((end - start).days) + 1)
        rows.append(
            {
                "bundle": bucket_name,
                "start": start.strftime("%d/%m/%Y"),
                "end": end.strftime("%d/%m/%Y"),
                "span_days": span_days,
                "tickets": tickets,
                "tickets/day": float(tickets / span_days) if span_days else np.nan,
                "revenue": revenue,
            }
        )
    bundle_df = pd.DataFrame(rows)
    if bundle_df.empty:
        return bundle_df
    bundle_df["revenue"] = bundle_df["revenue"].map(format_eur)
    bundle_df["tickets/day"] = bundle_df["tickets/day"].map(lambda v: "" if pd.isna(v) else f"{v:.2f}")
    return bundle_df


def render_pdf_text_page(
    pdf: PdfPages,
    title: str,
    paragraphs: List[str],
    bullets: Optional[List[str]] = None,
    footer: Optional[str] = None,
    figsize: tuple[float, float] = (11.69, 8.27),
) -> None:
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.text(0.05, 0.96, title, fontsize=20, fontweight="bold", va="top")

    y = 0.90
    for paragraph in paragraphs:
        wrapped = textwrap.fill(paragraph, width=115)
        ax.text(0.05, y, wrapped, fontsize=11.5, va="top")
        y -= 0.075 * max(1, wrapped.count("\n") + 1)
        y -= 0.02

    if bullets:
        y -= 0.01
        for bullet in bullets:
            wrapped = textwrap.fill(f"- {bullet}", width=113, subsequent_indent="  ")
            ax.text(0.06, y, wrapped, fontsize=11.5, va="top")
            y -= 0.075 * max(1, wrapped.count("\n") + 1)

    if footer:
        ax.text(0.05, 0.03, footer, fontsize=9.5, color="#555555", va="bottom")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def render_pdf_table_page(
    pdf: PdfPages,
    title: str,
    table_df: pd.DataFrame,
    note: Optional[str] = None,
    figsize: tuple[float, float] = (11.69, 8.27),
    font_size: int = 9,
) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    ax.set_title(title, fontsize=18, fontweight="bold", pad=18)
    if table_df.empty:
        ax.text(0.05, 0.5, "Nessun dato disponibile.", fontsize=12)
    else:
        table = ax.table(
            cellText=table_df.values,
            colLabels=table_df.columns,
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(font_size)
        table.scale(1.0, 1.35)
        for (r, c), cell in table.get_celld().items():
            cell.set_edgecolor("#333333")
            cell.set_linewidth(0.6)
            if r == 0:
                cell.set_facecolor("#f3f3f3")
                cell.set_text_props(weight="bold")
            elif c == 0:
                cell.set_text_props(ha="left")
    if note:
        ax.text(0.02, 0.03, note, fontsize=9.5, color="#555555", va="bottom")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def render_pdf_image_page(
    pdf: PdfPages,
    title: str,
    image_path: Path,
    note: Optional[str] = None,
    figsize: tuple[float, float] = (11.69, 8.27),
) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    ax.set_title(title, fontsize=18, fontweight="bold", pad=18)
    if image_path.exists():
        img = plt.imread(image_path)
        ax.imshow(img)
    else:
        ax.text(0.5, 0.5, f"Immagine non trovata:\n{image_path}", ha="center", va="center", fontsize=12)
    if note:
        ax.text(0.02, 0.03, note, fontsize=9.5, color="#555555", va="bottom")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def export_detailed_pdf_report(
    output_dir: Path,
    csv_path: Path,
    df_raw: pd.DataFrame,
    df: pd.DataFrame,
    csv_dir: Path,
    plots_dir: Path,
    timeline_markers: List[Dict[str, object]],
    ticket_type_col: Optional[str],
    ticket_total_num: Optional[str],
    order_total_num: Optional[str],
    order_status_col: Optional[str],
    country_col: Optional[str],
    city_col: Optional[str],
    dob_col: Optional[str],
    order_status_counts: Optional[pd.Series],
    by_type: Optional[pd.DataFrame],
    phase_table: Optional[pd.DataFrame],
    amb_table: Optional[pd.DataFrame],
) -> None:
    pdf_dir = output_dir / "pdf"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = pdf_dir / f"{slugify(csv_path.stem)}_detailed_report.pdf"

    total_rows_raw = len(df_raw)
    total_rows = len(df)
    ticket_sum = float(df[ticket_total_num].sum()) if ticket_total_num and ticket_total_num in df.columns else np.nan
    order_sum = float(df[order_total_num].sum()) if order_total_num and order_total_num in df.columns else np.nan
    date_min = df[PARSED_DATE_COL].min() if PARSED_DATE_COL in df.columns else pd.NaT
    date_max = df[PARSED_DATE_COL].max() if PARSED_DATE_COL in df.columns else pd.NaT

    phase_summary = build_phase_window_summary(df, timeline_markers, ticket_type_col, ticket_total_num)
    bundle_summary = build_bundle_summary(df, timeline_markers, ticket_type_col, ticket_total_num)

    phase1_amb_share = np.nan
    phase2_amb_share = np.nan
    phase3_amb_share = np.nan
    if amb_table is not None and not amb_table.empty and phase_table is not None and not phase_table.empty:
        amb_body = amb_table.drop(index="TOTAL", errors="ignore").copy()
        phase1_cols = [c for c in amb_body.columns if "phase_1" in str(c).lower()]
        phase2_cols = [c for c in amb_body.columns if "phase_2" in str(c).lower()]
        phase3_cols = [c for c in amb_body.columns if "phase_3" in str(c).lower()]
        if phase1_cols:
            phase1_total = int(phase_table.loc["phase_1", "tickets"]) if "phase_1" in phase_table.index else np.nan
            phase1_amb = pd.to_numeric(amb_body[phase1_cols].stack(), errors="coerce").sum()
            phase1_amb_share = (phase1_amb / phase1_total * 100) if phase1_total else np.nan
        if phase2_cols:
            phase2_total = int(phase_table.loc["phase_2", "tickets"]) if "phase_2" in phase_table.index else np.nan
            phase2_amb = pd.to_numeric(amb_body[phase2_cols].stack(), errors="coerce").sum()
            phase2_amb_share = (phase2_amb / phase2_total * 100) if phase2_total else np.nan
        if phase3_cols:
            phase3_total = int(phase_table.loc["phase_3", "tickets"]) if "phase_3" in phase_table.index else np.nan
            phase3_amb = pd.to_numeric(amb_body[phase3_cols].stack(), errors="coerce").sum()
            phase3_amb_share = (phase3_amb / phase3_total * 100) if phase3_total else np.nan

    phase1_tickets = int(phase_summary.loc[phase_summary["phase"] == "phase_1", "tickets"].sum()) if not phase_summary.empty and "phase_1" in set(phase_summary.get("phase", [])) else 0
    phase2_tickets = int(phase_summary.loc[phase_summary["phase"] == "phase_2", "tickets"].sum()) if not phase_summary.empty and "phase_2" in set(phase_summary.get("phase", [])) else 0
    phase3_tickets = int(phase_summary.loc[phase_summary["phase"] == "phase_3", "tickets"].sum()) if not phase_summary.empty and "phase_3" in set(phase_summary.get("phase", [])) else 0
    phase1_rate = float(phase_summary.loc[phase_summary["phase"] == "phase_1", "tickets/day"].astype(float).iloc[0]) if not phase_summary.empty and "phase_1" in set(phase_summary.get("phase", [])) else np.nan
    phase2_rate = float(phase_summary.loc[phase_summary["phase"] == "phase_2", "tickets/day"].astype(float).iloc[0]) if not phase_summary.empty and "phase_2" in set(phase_summary.get("phase", [])) else np.nan
    phase3_rate = float(phase_summary.loc[phase_summary["phase"] == "phase_3", "tickets/day"].astype(float).iloc[0]) if not phase_summary.empty and "phase_3" in set(phase_summary.get("phase", [])) else np.nan

    with PdfPages(pdf_path) as pdf:
        render_pdf_text_page(
            pdf,
            "7 Chakras EDA - Detailed Report",
            paragraphs=[
                f"Source CSV: {csv_path}",
                f"Rows in raw CSV: {total_rows_raw:,}. Rows analyzed after quality filter: {total_rows:,}.",
                f"Ticket total (sum of Ticket Total): {ticket_sum:,.2f}." if pd.notna(ticket_sum) else "Ticket total not available.",
                f"Order total (sum of Order Total): {order_sum:,.2f}." if pd.notna(order_sum) else "Order total not available.",
                f"Payment dates span from {date_min.strftime('%d/%m/%Y') if pd.notna(date_min) else 'n/a'} to {date_max.strftime('%d/%m/%Y') if pd.notna(date_max) else 'n/a'}.",
            ],
            bullets=[
                "Early bird and phase 0 should be treated as launch / pre-lineup windows, not as the core benchmark for the commercial engine.",
                f"Phase 1 sells {phase1_tickets:,} tickets at {phase1_rate:.2f} tickets/day; phase 2 sells {phase2_tickets:,} at {phase2_rate:.2f} tickets/day, so demand is denser after the first stable block.",
                f"Phase 3 is now measurable, not just a future marker: {phase3_tickets:,} tickets at {phase3_rate:.2f} tickets/day in the observed window, so the current phase is holding the phase-2 pace so far.",
                f"Ambassador contribution rises from phase 1 to phase 2 ({phase1_amb_share:.1f}% to {phase2_amb_share:.1f}%), then sits at {phase3_amb_share:.1f}% in the partial phase-3 window.",
                "Christmas and Lovers bundles behave as tactical bursts: they are short windows with concentrated demand, not recurring structural phases.",
                "Monthly averages confirm the same trend: April is the strongest full month in both total tickets and full festival passes, while May is still a partial month.",
            ],
            footer="Generated by ticket_eda.py",
        )

        if order_status_counts is not None and not order_status_counts.empty:
            status_df = order_status_counts.rename_axis("status").reset_index(name="tickets").head(12)
            render_pdf_table_page(
                pdf,
                "Data Quality and Order Status",
                status_df,
                note="Top order-status counts from the full dataset. Useful as a quality gate before reading the commercial performance.",
                figsize=(11.69, 8.27),
                font_size=10,
            )

        if phase_summary is not None and not phase_summary.empty:
            render_pdf_table_page(
                pdf,
                "Phase Windows and Sales Density",
                phase_summary.rename(columns={"tickets/day": "tickets/day"}),
                note="Phase windows are computed from the timeline markers in the config. The phase duration is the distance between consecutive phase starts; bundles use the first/last sold ticket dates saved in the same config.",
                figsize=(11.69, 8.27),
                font_size=9,
            )

        if bundle_summary is not None and not bundle_summary.empty:
            render_pdf_table_page(
                pdf,
                "Bundle Windows",
                bundle_summary.rename(columns={"tickets/day": "tickets/day"}),
                note="These are tactical bursts with a short duration and high density. The start and end dates are persisted in the config, so they remain stable across runs.",
                figsize=(11.69, 8.27),
                font_size=10,
            )

        if by_type is not None and not by_type.empty:
            top_types = by_type.head(15).reset_index().rename(columns={"index": "ticket_type"})
            render_pdf_table_page(
                pdf,
                "Top Ticket Types by Volume",
                top_types.head(15),
                note="The distribution is concentrated: a few ticket families account for most of the volume, while the rest form a long tail.",
                figsize=(11.69, 8.27),
                font_size=8,
            )

        if amb_table is not None and not amb_table.empty:
            amb_pdf = amb_table.drop(index="TOTAL", errors="ignore").copy()
            if not amb_pdf.empty:
                amb_pdf = amb_pdf.reset_index().rename(columns={"index": "ambassador"})
                keep_cols = [c for c in amb_pdf.columns if c == "ambassador" or c == "tickets_total" or c == "Total Amount (€)" or "phase_" in str(c).lower()]
                amb_pdf = amb_pdf[keep_cols].head(12)
                note = (
                    f"Ambassador share: phase 1 = {phase1_amb_share:.1f}% of phase-1 tickets; "
                    f"phase 2 = {phase2_amb_share:.1f}% of phase-2 tickets; "
                    f"phase 3 = {phase3_amb_share:.1f}% of phase-3 tickets so far."
                    if pd.notna(phase1_amb_share) and pd.notna(phase2_amb_share) and pd.notna(phase3_amb_share)
                    else "Ambassador sales are concentrated in a small subset of profiles, with a long tail of low-volume contributors."
                )
                render_pdf_table_page(
                    pdf,
                    "Ambassador Sales - Top Contributors",
                    amb_pdf,
                    note=note,
                    figsize=(11.69, 8.27),
                    font_size=8,
                )

        render_pdf_image_page(
            pdf,
            "Sales Timeline - Daily",
            plots_dir / "vendite_giornaliere.png",
            note="Daily sales with the configured phase markers.",
            figsize=(11.69, 8.27),
        )
        render_pdf_image_page(
            pdf,
            "Sales Timeline - Cumulative",
            plots_dir / "vendite_cumulative.png",
            note="Cumulative sales with the same markers and a denser y-axis.",
            figsize=(11.69, 8.27),
        )
        render_pdf_image_page(
            pdf,
            "Ticket Type Concentration",
            plots_dir / "by_type_focused.png",
            note="The focused bar chart groups phase variants and keeps a stable relative scale across its separate PNGs.",
            figsize=(11.69, 8.27),
        )

    print(f"\nPDF dettagliato salvato in: {pdf_path}")


def export_narrative_pdf_report(
    output_dir: Path,
    csv_path: Path,
    df_raw: pd.DataFrame,
    df: pd.DataFrame,
    timeline_markers: List[Dict[str, object]],
    ticket_type_col: Optional[str],
    ticket_total_num: Optional[str],
    order_total_num: Optional[str],
    order_status_counts: Optional[pd.Series],
    by_type: Optional[pd.DataFrame],
    phase_table: Optional[pd.DataFrame],
    amb_table: Optional[pd.DataFrame],
) -> None:
    pdf_dir = output_dir / "pdf"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    stem = slugify(csv_path.stem)
    pdf_path = pdf_dir / f"{stem}_conclusioni_narrative.pdf"
    txt_path = pdf_dir / f"{stem}_conclusioni_narrative.txt"

    total_rows_raw = len(df_raw)
    total_rows = len(df)
    ticket_sum = float(df[ticket_total_num].sum()) if ticket_total_num and ticket_total_num in df.columns else np.nan
    order_sum = float(df[order_total_num].sum()) if order_total_num and order_total_num in df.columns else np.nan
    date_min = df[PARSED_DATE_COL].min() if PARSED_DATE_COL in df.columns else pd.NaT
    date_max = df[PARSED_DATE_COL].max() if PARSED_DATE_COL in df.columns else pd.NaT

    phase_summary = build_phase_window_summary(df, timeline_markers, ticket_type_col, ticket_total_num)
    bundle_summary = build_bundle_summary(df, timeline_markers, ticket_type_col, ticket_total_num)
    monthly_sales = build_monthly_ticket_sales_summary(df, ticket_type_col)

    best_all_month = "n/d"
    best_all_avg = np.nan
    best_full_month = "n/d"
    best_full_avg = np.nan
    latest_month = "n/d"
    latest_observed_days = 0
    latest_all_avg = np.nan
    latest_full_avg = np.nan
    if not monthly_sales.empty:
        monthly_complete = monthly_sales.copy()
        monthly_complete["days_in_month"] = monthly_complete["month"].map(
            lambda value: pd.Period(str(value), freq="M").days_in_month
        )
        monthly_complete = monthly_complete[
            monthly_complete["observed_days"] >= monthly_complete["days_in_month"]
        ]
        if not monthly_complete.empty:
            best_all_row = monthly_complete.loc[monthly_complete["all_avg_per_observed_day"].idxmax()]
            best_full_row = monthly_complete.loc[
                monthly_complete["full_festival_avg_per_observed_day"].idxmax()
            ]
            best_all_month = str(best_all_row["month"])
            best_all_avg = float(best_all_row["all_avg_per_observed_day"])
            best_full_month = str(best_full_row["month"])
            best_full_avg = float(best_full_row["full_festival_avg_per_observed_day"])
        latest_row = monthly_sales.iloc[-1]
        latest_month = str(latest_row["month"])
        latest_observed_days = int(latest_row["observed_days"])
        latest_all_avg = float(latest_row["all_avg_per_observed_day"])
        latest_full_avg = float(latest_row["full_festival_avg_per_observed_day"])

    phase1_tickets = int(phase_summary.loc[phase_summary["phase"] == "phase_1", "tickets"].sum()) if not phase_summary.empty and "phase_1" in set(phase_summary.get("phase", [])) else 0
    phase2_tickets = int(phase_summary.loc[phase_summary["phase"] == "phase_2", "tickets"].sum()) if not phase_summary.empty and "phase_2" in set(phase_summary.get("phase", [])) else 0
    phase3_tickets = int(phase_summary.loc[phase_summary["phase"] == "phase_3", "tickets"].sum()) if not phase_summary.empty and "phase_3" in set(phase_summary.get("phase", [])) else 0
    phase1_rate = float(phase_summary.loc[phase_summary["phase"] == "phase_1", "tickets/day"].astype(float).iloc[0]) if not phase_summary.empty and "phase_1" in set(phase_summary.get("phase", [])) else np.nan
    phase2_rate = float(phase_summary.loc[phase_summary["phase"] == "phase_2", "tickets/day"].astype(float).iloc[0]) if not phase_summary.empty and "phase_2" in set(phase_summary.get("phase", [])) else np.nan
    phase3_rate = float(phase_summary.loc[phase_summary["phase"] == "phase_3", "tickets/day"].astype(float).iloc[0]) if not phase_summary.empty and "phase_3" in set(phase_summary.get("phase", [])) else np.nan

    ambassador_share_phase1 = np.nan
    ambassador_share_phase2 = np.nan
    ambassador_share_phase3 = np.nan
    if amb_table is not None and not amb_table.empty and phase_table is not None and not phase_table.empty:
        amb_body = amb_table.drop(index="TOTAL", errors="ignore").copy()
        phase1_cols = [c for c in amb_body.columns if "phase_1" in str(c).lower()]
        phase2_cols = [c for c in amb_body.columns if "phase_2" in str(c).lower()]
        phase3_cols = [c for c in amb_body.columns if "phase_3" in str(c).lower()]
        if phase1_cols and "phase_1" in phase_table.index:
            ambassador_share_phase1 = pd.to_numeric(amb_body[phase1_cols].stack(), errors="coerce").sum() / float(phase_table.loc["phase_1", "tickets"]) * 100
        if phase2_cols and "phase_2" in phase_table.index:
            ambassador_share_phase2 = pd.to_numeric(amb_body[phase2_cols].stack(), errors="coerce").sum() / float(phase_table.loc["phase_2", "tickets"]) * 100
        if phase3_cols and "phase_3" in phase_table.index:
            ambassador_share_phase3 = pd.to_numeric(amb_body[phase3_cols].stack(), errors="coerce").sum() / float(phase_table.loc["phase_3", "tickets"]) * 100

    ticket_categories = int(len(by_type.drop(index="TOTAL", errors="ignore"))) if by_type is not None and not by_type.empty else 0
    amb_total = int(amb_table.loc["TOTAL", "tickets_total"]) if amb_table is not None and not amb_table.empty and "TOTAL" in amb_table.index else np.nan

    phase1_count = int(phase_summary.loc[phase_summary["phase"] == "phase_1", "tickets"].iloc[0]) if not phase_summary.empty and "phase_1" in set(phase_summary.get("phase", [])) else 0
    phase2_count = int(phase_summary.loc[phase_summary["phase"] == "phase_2", "tickets"].iloc[0]) if not phase_summary.empty and "phase_2" in set(phase_summary.get("phase", [])) else 0
    phase3_count = int(phase_summary.loc[phase_summary["phase"] == "phase_3", "tickets"].iloc[0]) if not phase_summary.empty and "phase_3" in set(phase_summary.get("phase", [])) else 0
    early_count = int(phase_summary.loc[phase_summary["phase"] == "early_bird", "tickets"].iloc[0]) if not phase_summary.empty and "early_bird" in set(phase_summary.get("phase", [])) else 0

    lines: List[str] = []
    lines.append("Conclusioni narrative - 7 Chakras EDA")
    lines.append("")
    lines.append(
        f"Questa analisi parte dal file {csv_path.name} e si chiude con {total_rows:,} ticket analizzati su {total_rows_raw:,} ticket grezzi. Il quadro economico finale non è banale: la somma del Ticket Total è pari a {ticket_sum:,.2f}, mentre la somma dell'Order Total arriva a {order_sum:,.2f}. Le due metriche sono entrambe utili, ma raccontano aspetti diversi del flusso: il Ticket Total misura ciò che entra dal singolo ticket, mentre l'Order Total fotografa il movimento più ampio che passa dal checkout."
    )
    lines.append(
        f"La timeline dei pagamenti va dal {date_min.strftime('%d/%m/%Y') if pd.notna(date_min) else 'n/d'} al {date_max.strftime('%d/%m/%Y') if pd.notna(date_max) else 'n/d'}. Dentro questa finestra la lettura corretta non è quella di un mercato piatto, ma quella di una curva che si apre, si consolida e mantiene intensità anche nella fase più recente."
    )
    lines.append("")
    lines.append("Punti chiave")
    lines.append(f"- Early bird: {early_count:,} ticket. È una fase di lancio e va letta come un test della domanda iniziale, non come il benchmark principale della maturità commerciale.")
    lines.append("- Phase 0: resta una finestra pre-lineup molto corta. Qui il nome del festival lavora quasi da solo, quindi il dato serve più a misurare la forza del brand che la tenuta del piano artistico.")
    lines.append(
        f"- Phase 1 e Phase 2: restano il primo confronto solido. Phase 1 chiude a {phase1_count:,} ticket con {phase1_rate:.2f} ticket/giorno, mentre Phase 2 arriva a {phase2_count:,} ticket con {phase2_rate:.2f} ticket/giorno. La domanda quindi non si appiattisce: si concentra in una finestra più corta."
    )
    lines.append(
        f"- Phase 3: al {date_max.strftime('%d/%m/%Y') if pd.notna(date_max) else 'run corrente'} non è più solo un marker futuro. Ha già {phase3_count:,} ticket e un ritmo osservato di {phase3_rate:.2f} ticket/giorno, quindi per ora sta tenendo una densità simile a phase 2."
    )
    lines.append(
        f"- Ambassador: il canale resta rilevante ma cambia peso. In phase 1 pesa circa {ambassador_share_phase1:.1f}% del volume di fase, in phase 2 sale a circa {ambassador_share_phase2:.1f}%, mentre in phase 3, ancora parziale, è al {ambassador_share_phase3:.1f}%. La leva rimane strutturale, ma l'ultima fase sembra più guidata dai ticket standard."
    )
    lines.append(
        f"- Ritmo mensile: tra i mesi completi, {best_all_month} è il picco recente sui ticket totali ({best_all_avg:.2f} ticket/giorno) e {best_full_month} è il picco sui full festival pass ({best_full_avg:.2f} pass/giorno). {latest_month} va letto a parte perché contiene solo {latest_observed_days} giorni osservati: {latest_all_avg:.2f} ticket/giorno totali e {latest_full_avg:.2f} full festival pass/giorno."
    )
    lines.append(
        "- Bundle: Christmas Bundle e Lovers Bundle sono picchi tattici, non fasi strutturali. La loro logica è quella del burst commerciale: pochi giorni, molta densità, impatto immediato."
    )
    lines.append("")
    lines.append("Lettura estesa")
    lines.append(
        f"Il ticket type count mostra una distribuzione molto sbilanciata: {ticket_categories} categorie effettive e una coda lunga di tipi con peso marginale. Questo è coerente con un modello in cui poche famiglie di ticket trainano la maggior parte del volume, mentre il resto del catalogo si frammenta in segmenti più piccoli."
    )
    lines.append(
        f"La parte ambassador è ancora più istruttiva: nella tabella complessiva compaiono {amb_total:,} ticket ambassador esclusa la riga TOTAL, quindi il canale ha ormai un peso reale sul totale e non può più essere letto come una semplice appendice."
    )
    lines.append(
        "Il passaggio da phase 1 a phase 2 resta il primo segnale forte del run: il volume complessivo è quasi identico, ma il tempo necessario per generarlo si riduce. La novità più recente è che phase 3 non rompe questo andamento: pur essendo ancora una finestra aperta, mantiene un ritmo giornaliero in linea con phase 2."
    )
    lines.append(
        f"Il dato mensile conferma questa lettura: ad aprile il run raggiunge il massimo tra i mesi completi, mentre i primi giorni di maggio sono ancora troppo pochi per essere confrontati come mese intero. Sono però utili come segnale di continuità, perché il ritmo iniziale resta alto."
    )
    lines.append("")
    lines.append("Conclusione finale")
    lines.append(
        "La lettura complessiva resta positiva: il progetto non mostra segni di appiattimento, ma un passaggio progressivo verso una vendita più densa e più organizzata. Phase 2 ha accelerato rispetto a phase 1 e phase 3, nei dati disponibili al momento, sta mantenendo quella soglia di intensità. Gli ambassador restano una leva strutturale, mentre i bundle funzionano come acceleratori tattici."
    )
    lines.append(
        "Per il prossimo run, il punto non sarà solo quante vendite arrivano, ma dove si concentrano e con quale intensità giornaliera. È questa la metrica che rende davvero confrontabili le fasi e permette di capire se il festival sta reggendo la pressione del pricing ladder."
    )
    txt_path.write_text("\n".join(lines), encoding="utf-8")

    with PdfPages(pdf_path) as pdf:
        render_pdf_text_page(
            pdf,
            "7 Chakras EDA - Conclusioni narrative",
            paragraphs=[
                f"Questa analisi parte dal file {csv_path.name} e si chiude con {total_rows:,} ticket analizzati su {total_rows_raw:,} ticket grezzi. Il quadro economico finale non è banale: la somma del Ticket Total è pari a {ticket_sum:,.2f}, mentre la somma dell'Order Total arriva a {order_sum:,.2f}. Le due metriche sono entrambe utili, ma raccontano aspetti diversi del flusso: il Ticket Total misura ciò che entra dal singolo ticket, mentre l'Order Total fotografa il movimento più ampio che passa dal checkout.",
                f"La timeline dei pagamenti va dal {date_min.strftime('%d/%m/%Y') if pd.notna(date_min) else 'n/d'} al {date_max.strftime('%d/%m/%Y') if pd.notna(date_max) else 'n/d'}. Dentro questa finestra la lettura corretta non è quella di un mercato piatto, ma quella di una curva che si apre, si consolida e mantiene intensità anche nella fase più recente.",
            ],
            bullets=[
                f"Early bird: {early_count:,} ticket. È una fase di lancio e va letta come un test della domanda iniziale, non come il benchmark principale della maturità commerciale.",
                "- Phase 0: resta una finestra pre-lineup molto corta. Qui il nome del festival lavora quasi da solo, quindi il dato serve più a misurare la forza del brand che la tenuta del piano artistico.",
                f"Phase 1 e Phase 2: Phase 1 chiude a {phase1_count:,} ticket con {phase1_rate:.2f} ticket/giorno, mentre Phase 2 arriva a {phase2_count:,} ticket con {phase2_rate:.2f} ticket/giorno. La domanda quindi non si appiattisce: si concentra.",
                f"Phase 3: al {date_max.strftime('%d/%m/%Y') if pd.notna(date_max) else 'run corrente'} ha già {phase3_count:,} ticket e un ritmo osservato di {phase3_rate:.2f} ticket/giorno, quindi sta tenendo una densità simile a phase 2.",
                f"Ambassador: in phase 1 pesa circa {ambassador_share_phase1:.1f}%, in phase 2 sale a circa {ambassador_share_phase2:.1f}%, mentre in phase 3, ancora parziale, è al {ambassador_share_phase3:.1f}%. La leva rimane strutturale, ma l'ultima fase sembra più guidata dai ticket standard.",
                f"Ritmo mensile: tra i mesi completi, {best_all_month} è il picco sui ticket totali ({best_all_avg:.2f}/giorno) e {best_full_month} è il picco sui full festival pass ({best_full_avg:.2f}/giorno).",
                "- Bundle: Christmas Bundle e Lovers Bundle sono picchi tattici, non fasi strutturali. La loro logica è quella del burst commerciale: pochi giorni, molta densità, impatto immediato.",
            ],
            footer="Conseguenze narrative generate automaticamente dal run corrente.",
            figsize=(11.69, 8.27),
        )

        render_pdf_text_page(
            pdf,
            "Lettura delle fasi",
            paragraphs=[
                "Il modo più utile per leggere le fasi è smettere di guardare soltanto i totali e cominciare a guardare il ritmo giornaliero. Una fase che vende meno ticket ma in meno tempo può essere più forte di una fase lunga con lo stesso volume, perché mostra che la domanda continua a reagire anche quando il pricing ladder sale.",
                "È esattamente per questo che phase 2 conta così tanto. Se phase 1 e phase 2 chiudono quasi con lo stesso numero di ticket, ma phase 2 comprime quel volume in meno giorni, la lettura commerciale è positiva: il mercato non è stanco, si sta muovendo più velocemente. La novità del run corrente è che phase 3, pur ancora aperta, sta tenendo lo stesso ordine di grandezza nel ritmo.",
            ],
            bullets=[
                "Early bird e phase 0 vanno trattate come finestre di lancio, non come KPI principale della salute commerciale.",
                f"Phase 1 si muove intorno a {phase1_rate:.2f} ticket/giorno, quindi è il primo vero test della capacità del prodotto di continuare a vendere quando l'impulso iniziale si attenua.",
                f"Phase 2 sale a {phase2_rate:.2f} ticket/giorno, che è il segnale più forte dell'intera ladder perché mostra che il mercato non si è raffreddato: si è compattato.",
                f"Phase 3 è da leggere come fase corrente parziale: {phase3_count:,} ticket e {phase3_rate:.2f} ticket/giorno fino al {date_max.strftime('%d/%m/%Y') if pd.notna(date_max) else 'run corrente'}.",
            ],
            footer="Il modello delle fasi va letto come modello di ritmo, non solo come modello di prezzo.",
            figsize=(11.69, 8.27),
        )

        render_pdf_text_page(
            pdf,
            "Lettura dei canali",
            paragraphs=[
                f"Il mix dei ticket è abbastanza ampio da mostrare che la crescita non arriva da una sola leva. Nel run corrente ci sono {ticket_categories} categorie effettive nel focused breakdown, e la coda lunga è reale. Poche categorie portano gran parte del volume, mentre il resto si distribuisce in tasche più piccole e più tattiche.",
                f"La performance ambassador merita un'attenzione specifica perché la tabella ambassador contiene {amb_total:,} ticket totali se escludiamo la riga TOTAL finale. Non è più un canale marginale: è una vera layer di distribuzione.",
            ],
            bullets=[
                "Il canale ambassador è concentrato: un gruppo più piccolo di profili fa la maggior parte del lavoro, cosa tipica di una rete che si sta maturando ma non si è ancora livellata del tutto.",
                "Il canale bundle è bursty per costruzione. Non è un difetto: è una forma commerciale utile perché permette di creare momenti di vendita brevi e ad alta intensità.",
                "Le pagine su geografia e gateway nel report tecnico suggeriscono che il processo di acquisto è abbastanza stabile da sostenere questi pattern; quindi la domanda non è più se la gente può comprare, ma dove conviene spingere meglio l'incremento.",
            ],
            footer="Questa sezione è volutamente descrittiva, così può essere portata direttamente in una slide narrativa.",
            figsize=(11.69, 8.27),
        )

        render_pdf_text_page(
            pdf,
            "Conclusioni aperte",
            paragraphs=[
                "La conclusione generale è che la campagna è ancora viva e strutturata. Non si sta distribuendo in modo uniforme in un modo che farebbe pensare a una domanda debole; si sta invece concentrando nei punti in cui prodotto, pricing ladder e canali di distribuzione si allineano meglio.",
            ],
            bullets=[
                "Le fasi di lancio confermano una domanda iniziale reale.",
                "Le fasi centrali confermano che il festival riesce a continuare a vendere dopo la prima ondata.",
                "La dinamica delle fasi più avanzate parla di accelerazione prima e tenuta poi, non di stanchezza.",
                "Ambassador e bundle stanno diventando leve strategiche, non semplici supporti accessori.",
                "Il prossimo run andrà letto sulle stesse finestre di fase, così il trend resterà confrontabile nel tempo.",
            ],
            footer="Phase 3 è ormai misurabile, ma va ancora letta come finestra aperta.",
            figsize=(11.69, 8.27),
        )

    print(f"\nPDF narrativo salvato in: {pdf_path}")
    print(f"TXT narrativo salvato in: {txt_path}")


def save_chunked_barh_plot(
    df: pd.DataFrame,
    destination: Path,
    name: str,
    fmt: str,
    chunk_size: int = 15,
    fig_width: float = 14.0,
    base_height: float = 5.4,
    row_height_scale: float = 0.42,
    label_font_size: int = 12,
    title_font_size: int = 16,
    dpi_override: Optional[int] = None,
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    if df.empty:
        return

    ordered = df.reset_index()
    if ordered.columns[0] == "index":
        ordered = ordered.rename(columns={"index": "ticket_type"})
    label_col = ordered.columns[0]
    ordered = ordered.sort_values(["tickets", label_col], ascending=[False, True]).reset_index(drop=True)
    chunks = [ordered.iloc[i : i + chunk_size].copy() for i in range(0, len(ordered), chunk_size)]
    fixed_x_max = 0.0
    if chunks:
        first_chunk = chunks[0]
        if not first_chunk.empty:
            fixed_x_max = float(first_chunk["tickets"].max()) * 1.15

    for idx, chunk in enumerate(chunks, start=1):
        if chunk.empty:
            continue

        plot_df = chunk.sort_values(["tickets", label_col], ascending=[True, True]).copy()
        labels = plot_df[label_col].astype(str)
        values = plot_df["tickets"].astype(float)
        fig_height = max(base_height, row_height_scale * len(plot_df) + 1.8)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.barh(labels, values, color="#1976d2", height=0.82)
        ax.set_title(f"Conteggio biglietti per tipo - blocco {idx}", fontsize=title_font_size)
        ax.set_xlabel("Biglietti")
        ax.set_ylabel("")
        ax.tick_params(axis="both", labelsize=label_font_size)
        ax.xaxis.label.set_size(label_font_size + 1)
        if fixed_x_max > 0:
            ax.set_xlim(0, fixed_x_max)
        else:
            ax.set_xlim(0, max(values.max() if not values.empty else 0, 1) * 1.15)
        ax.grid(axis="x", alpha=0.25)
        for container in ax.containers:
            ax.bar_label(container, padding=3, fmt="%.0f", fontsize=label_font_size)
        fig.tight_layout()
        suffix = f"_{idx:02d}" if len(chunks) > 1 else ""
        out_path = destination / f"{name}{suffix}.{fmt}"
        fig.savefig(out_path, bbox_inches="tight", dpi=dpi_override or 250)
        print(f"Grafico salvato: {out_path}")
        plt.close(fig)


def save_table_image(
    df: pd.DataFrame,
    destination: Path,
    name: str,
    fmt: str,
    highlight_value: str | None = None,
    widen_first_col: bool = False,
    narrow_numeric_cols: bool = False,
    col_widths_override: Optional[List[float]] = None,
    fig_width_override: Optional[float] = None,
    fig_height_override: Optional[float] = None,
    scale_x_override: Optional[float] = None,
    scale_y_override: Optional[float] = None,
    bbox_override: Optional[List[float]] = None,
    manual_table: bool = False,
    header_font_size: Optional[int] = None,
    header_labels_override: Optional[List[str]] = None,
    row_height_override: Optional[float] = None,
    header_height_override: Optional[float] = None,
    font_size: int = 8,
    dpi_override: Optional[int] = None,
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    if df.empty:
        return
    shown = df.copy()
    shown = shown.reset_index() if shown.index.name or shown.index.names else shown.reset_index(drop=False)
    shown = shown.head(50)
    if shown.columns[0] == "index":
        shown = shown.rename(columns={"index": ""})
    width = 12.5 + max(0, len(shown.columns) - 3) * 1.6
    if fig_width_override:
        width = fig_width_override
    height = fig_height_override if fig_height_override is not None else max(2.5, 0.35 * len(shown))
    fig, ax = plt.subplots(figsize=(width, height))
    ax.axis("off")
    if manual_table:
        col_widths = col_widths_override or [1.0 / len(shown.columns)] * len(shown.columns)
        total_width = sum(col_widths)
        body_rows = len(shown)
        row_height = row_height_override if row_height_override is not None else 1.2
        header_height = header_height_override if header_height_override is not None else row_height
        total_height = (body_rows * row_height) + header_height
        ax.set_xlim(0, total_width)
        ax.set_ylim(0, total_height)
        ax.axis("off")
        x_positions = [0.0]
        for width in col_widths:
            x_positions.append(x_positions[-1] + width)
        # Body horizontal lines
        for r in range(body_rows + 1):
            y = r * row_height
            ax.hlines(y, 0, total_width, color="black", linewidth=1)
        # Header separator + top line
        ax.hlines(body_rows * row_height, 0, total_width, color="black", linewidth=1)
        ax.hlines(total_height, 0, total_width, color="black", linewidth=1)
        for x in x_positions:
            ax.vlines(x, 0, total_height, color="black", linewidth=1)
        header_size = header_font_size if header_font_size is not None else max(10, int(font_size * 0.9))
        header_labels = header_labels_override or [str(c) for c in shown.columns]
        for col_idx, label in enumerate(header_labels):
            x_center = (x_positions[col_idx] + x_positions[col_idx + 1]) / 2
            ax.text(
                x_center,
                total_height - (header_height / 2),
                str(label),
                ha="center" if col_idx else "left",
                va="center",
                fontsize=header_size,
            )
        for row_idx, row in enumerate(shown.values, start=1):
            y = (body_rows - row_idx) * row_height + (row_height / 2)
            for col_idx, value in enumerate(row):
                if col_idx == 0:
                    x = x_positions[0] + (total_width * 0.01)
                    ha = "left"
                else:
                    x = (x_positions[col_idx] + x_positions[col_idx + 1]) / 2
                    ha = "center"
                weight = "bold" if highlight_value and str(row[0]).strip().lower() == highlight_value.lower() else "normal"
                cell_font_size = font_size if col_idx == 0 else font_size + 3
                ax.text(x, y, str(value), ha=ha, va="center", fontsize=cell_font_size, weight=weight)
        out_path = destination / f"{name}.{fmt}"
        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight", dpi=dpi_override or 200)
        print(f"Tabella salvata: {out_path}")
        plt.close(fig)
        return
    col_widths = None
    if col_widths_override:
        col_widths = col_widths_override
    if widen_first_col and len(shown.columns) > 1:
        if narrow_numeric_cols and len(shown.columns) == 3:
            col_widths = [0.7, 0.15, 0.15]
        else:
            first = 0.42
            rest = (1.0 - first) / (len(shown.columns) - 1)
            col_widths = [first] + [rest] * (len(shown.columns) - 1)
    table_kwargs = dict(
        cellText=shown.values,
        colLabels=shown.columns,
        loc="center",
        cellLoc="center",
        colWidths=col_widths,
    )
    if bbox_override:
        table_kwargs["bbox"] = bbox_override
    elif col_widths_override:
        table_kwargs["bbox"] = [0, 0, 1, 1]
    table = ax.table(**table_kwargs)
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    scale_x = scale_x_override if scale_x_override is not None else 1
    scale_y = scale_y_override if scale_y_override is not None else 1.2
    table.scale(scale_x, scale_y)
    if col_widths_override:
        for col_idx, width in enumerate(col_widths_override):
            for row_idx in range(0, len(shown) + 1):
                table[(row_idx, col_idx)].set_width(width)
    if narrow_numeric_cols:
        for col_idx in range(1, len(shown.columns)):
            table[(0, col_idx)]._loc = "center"
            table[(0, col_idx)].set_text_props(ha="center", va="center")
            table[(0, col_idx)].PAD = 0.02
            for row_idx in range(1, len(shown) + 1):
                table[(row_idx, col_idx)]._loc = "center"
                table[(row_idx, col_idx)].set_text_props(ha="center", va="center")
                table[(row_idx, col_idx)].PAD = 0.02
    # Keep the first column left-aligned (labels).
    for row_idx in range(0, len(shown) + 1):
        table[(row_idx, 0)]._loc = "left"
        table[(row_idx, 0)].set_text_props(ha="left", va="center")
        table[(row_idx, 0)].PAD = 0.04
    if highlight_value:
        for row_idx, row in enumerate(shown.values, start=1):
            if str(row[0]).strip().lower() == highlight_value.lower():
                for col_idx in range(len(shown.columns)):
                    table[(row_idx, col_idx)].set_text_props(weight="bold")
                break
    out_path = destination / f"{name}.{fmt}"
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=dpi_override or 200)
    print(f"Tabella salvata: {out_path}")
    plt.close(fig)


def save_chunked_table_image(
    df: pd.DataFrame,
    destination: Path,
    name: str,
    fmt: str,
    chunk_size: int = 10,
    font_size: int = 22,
    header_font_size: int = 18,
    panel_width: float = 7.8,
    panel_height: float = 7.8,
    row_height_scale: float = 1.9,
    dpi_override: Optional[int] = None,
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    if df.empty:
        return

    shown = df.copy()
    shown = shown.reset_index() if shown.index.name or shown.index.names else shown.reset_index(drop=False)
    if shown.columns[0] == "index":
        shown = shown.rename(columns={"index": ""})

    total_mask = shown.iloc[:, 0].astype(str).str.upper().eq("TOTAL")
    total_row = shown[total_mask].copy()
    shown = shown[~total_mask].copy()

    rows = shown.to_dict(orient="records")
    chunks = [rows[i : i + chunk_size] for i in range(0, len(rows), chunk_size)]
    if total_row is not None and not total_row.empty:
        total_chunk = total_row.to_dict(orient="records")
        if chunks:
            chunks.append(total_chunk)
        else:
            chunks = [total_chunk]

    for panel_idx, chunk in enumerate(chunks, start=1):
        panel_df = pd.DataFrame(chunk)
        if panel_df.empty:
            continue
        panel_df = panel_df[shown.columns]
        panel_df = panel_df.reset_index(drop=True)

        fig_width = max(14.0, panel_width)
        fig_height = max(panel_height, 0.42 * len(panel_df) + 1.8)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.axis("off")

        n_cols = len(panel_df.columns)
        if n_cols > 1:
            first_col_width = 0.36
            remaining_width = max(0.1, 1.0 - first_col_width)
            other_col_width = remaining_width / (n_cols - 1)
            col_widths = [first_col_width] + [other_col_width] * (n_cols - 1)
        else:
            col_widths = [1.0]

        table = ax.table(
            cellText=panel_df.values,
            colLabels=panel_df.columns,
            loc="center",
            cellLoc="center",
            colWidths=col_widths,
        )
        table.auto_set_font_size(False)
        table.set_fontsize(font_size)
        table.scale(1.0, row_height_scale)

        for (r, c), cell in table.get_celld().items():
            cell.set_edgecolor("black")
            cell.set_linewidth(0.8)
            if r == 0:
                cell.set_text_props(weight="bold", fontsize=header_font_size)
                cell.set_facecolor("#f3f3f3")
            elif c == 0:
                cell.set_text_props(ha="left")

        if len(panel_df.columns) > 0:
            table[(0, 0)].set_text_props(ha="left", weight="bold", fontsize=header_font_size)
            for r in range(1, len(panel_df) + 1):
                if (r, 0) in table.get_celld():
                    table[(r, 0)].set_text_props(ha="left", fontsize=font_size)
                    table[(r, 0)].PAD = 0.05

        suffix = f"_{panel_idx:02d}" if len(chunks) > 1 else ""
        out_path = destination / f"{name}{suffix}.{fmt}"
        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight", dpi=dpi_override or 300)
        print(f"Tabella salvata: {out_path}")
        plt.close(fig)


def format_eur(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    try:
        amount = float(value)
    except (TypeError, ValueError):
        return ""
    rounded = round(amount)
    formatted = f"{rounded:,.0f}".replace(",", ".")
    return f"{formatted} \u20ac"


def main() -> None:
    parser = argparse.ArgumentParser(description="EDA 7 Chakras da linea di comando.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Percorso del file di configurazione JSON (default: Code/eda_config.json).",
    )
    args = parser.parse_args()

    default_config = Path(__file__).with_name("eda_config.json")
    config_path = args.config if args.config else default_config
    config = load_config(config_path)

    csv_path = Path(config["csv_path"]).expanduser()
    output_dir = Path(config["output_dir"]).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_dir = output_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)

    plots_cfg = config.get("plots", {})
    plots_enabled = plots_cfg.get("enabled", True)
    plot_format = plots_cfg.get("format", "png")
    plots_dir = output_dir / "plots"
    focused_ticket_cfg = config.get("focused_ticket_type_summary", {}) or {}
    volunteers_cfg = config.get("volunteers", {}) or {}
    pdf_cfg = config.get("pdf_report", {}) or {}
    narrative_pdf_cfg = config.get("narrative_pdf_report", {}) or {}

    columns_cfg: Dict[str, str] = config.get("columns", {})
    extra_country_cfg = config.get("extra_country_columns", []) or []
    extra_city_cfg = config.get("extra_city_columns", []) or []
    checkin_columns = config.get("checkin_columns", DEFAULT_CHECKIN_COLUMNS)

    def col_value(key: str, fallback: Optional[str]) -> Optional[str]:
        value = columns_cfg.get(key)
        return value or fallback

    def to_list(value: object) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        try:
            return [v for v in value if v is not None]
        except TypeError:
            return []

    pd.set_option("display.max_columns", 120)
    pd.set_option("display.width", 160)

    order_status_counts: Optional[pd.Series] = None
    by_type_df: Optional[pd.DataFrame] = None
    phase_table_df: Optional[pd.DataFrame] = None
    amb_table_df: Optional[pd.DataFrame] = None

    print(f"Carico CSV: {csv_path}")
    df_raw = pd.read_csv(csv_path, **READ_KWARGS)
    print(f"Shape iniziale: {df_raw.shape}")

    print("\nElenco colonne (ordine originale):")
    for idx, name in enumerate(df_raw.columns, start=1):
        print(f"{idx:>3}. {name}")

    df = df_raw.copy()
    df.columns = normalize_columns(df.columns)

    def ensure_existing(name: Optional[str], *alternatives: Optional[str]) -> Optional[str]:
        candidates = (name,) + alternatives
        for candidate in candidates:
            if candidate and candidate in df.columns:
                return candidate
        return None

    def resolve_existing_list(names: Iterable[Optional[str]]) -> List[str]:
        resolved: List[str] = []
        for name in names:
            if not name:
                continue
            alt = name.replace("\u00e0", "\ufffd") if "\u00e0" in name else None
            found = ensure_existing(name, alt)
            if found and found in df.columns and found not in resolved:
                resolved.append(found)
        return resolved

    payment_date_col = ensure_existing(col_value("payment_date", "Payment Date"))
    ticket_type_col = ensure_existing(col_value("ticket_type", "Ticket Type"))
    ticket_total_col = ensure_existing(col_value("ticket_total", "Ticket Total"))
    order_total_col = ensure_existing(col_value("order_total", "Order Total"))
    discount_col = ensure_existing(col_value("discount_code", "Discount Code"))
    ticket_discount_col = ensure_existing(col_value("ticket_discount", "Ticket Discount"))
    country_col = ensure_existing(
        col_value("country", "Country of residence / Paese di residenza (Campi ticket holder)")
    )
    city_preferred = col_value("city", "City of residence / Citt\u00e0 di residenza (Campi ticket holder)")
    city_col = ensure_existing(
        city_preferred,
        city_preferred.replace("\u00e0", "\ufffd") if city_preferred else None,
    )
    geo_country_cols = resolve_existing_list([country_col] + to_list(extra_country_cfg))
    geo_city_cols = resolve_existing_list([city_col] + to_list(extra_city_cfg))
    geo_report_cols = list(dict.fromkeys(geo_country_cols + geo_city_cols))
    attendee_email_col = ensure_existing(col_value("attendee_email", "Attendee E-mail"))
    buyer_email_col = ensure_existing(col_value("buyer_email", "Buyer E-Mail"))
    name_col = ensure_existing(col_value("name", "Name"))
    first_name_col = ensure_existing(col_value("first_name", "First Name"))
    last_name_col = ensure_existing(col_value("last_name", "Last Name"))
    order_number_col = ensure_existing(col_value("order_number", "Order Number"))
    order_status_col = ensure_existing(col_value("order_status", "Order Status"))
    payment_gateway_col = ensure_existing(col_value("payment_gateway", "Payment Gateway"))
    ticket_id_col = ensure_existing(col_value("ticket_id", "Ticket ID"))
    ticket_code_col = ensure_existing(col_value("ticket_code", "Ticket Code"))
    dob_preferred = col_value("date_of_birth", "Date of birth / Data di nascita (Campi ticket holder)")
    dob_col = ensure_existing(
        dob_preferred,
        dob_preferred.replace("\u00e0", "\ufffd") if dob_preferred else None,
    )

    if payment_date_col and payment_date_col in df.columns:
        df[PARSED_DATE_COL] = df[payment_date_col].map(parse_payment_date)
    else:
        df[PARSED_DATE_COL] = pd.NaT

    timeline_markers = config.get("timeline_markers", []) or []
    parsed_timeline_markers = parse_timeline_markers(timeline_markers)
    if ticket_type_col and payment_date_col:
        lovers_color = marker_color_by_label(parsed_timeline_markers, "lovers bundle", LOVERS_BUNDLE_COLOR)
        christmas_markers = build_dynamic_bundle_markers(
            df,
            PARSED_DATE_COL,
            ticket_type_col,
            CHRISTMAS_BUNDLE_KEYWORD,
            lovers_color,
        )
        if christmas_markers:
            timeline_markers = sync_christmas_bundle_markers(config, config_path, christmas_markers)
            parsed_timeline_markers = parse_timeline_markers(timeline_markers)

    numeric_targets = set(NUMERIC_CANDIDATES)
    numeric_targets.update(
        [
            ticket_total_col,
            order_total_col,
            ticket_discount_col,
            col_value("ticket_subtotal", "Ticket Subtotal"),
            col_value("ticket_fee", "Ticket Fee"),
            col_value("price", "Price"),
        ]
    )
    numeric_map: Dict[str, str] = {}
    for col_name in filter(None, numeric_targets):
        if col_name in df.columns:
            num_col = f"{col_name}_num"
            df[num_col] = df[col_name].map(to_num)
            numeric_map[col_name] = num_col

    for email_col in filter(None, [attendee_email_col, buyer_email_col]):
        if email_col in df.columns:
            df[email_col] = df[email_col].astype(str).str.strip().str.lower()

    df_full = df.copy()
    total_rows_full = len(df_full)
    if order_status_col and order_status_col in df_full.columns:
        normalized_status = df_full[order_status_col].astype(str).str.strip().str.lower()
        paid_mask = normalized_status == "paid"
        paid_count = int(paid_mask.sum())
        print(f"\nFiltro order status 'paid': considero {paid_count} righe su {total_rows_full}.")
        if paid_count > 0:
            df = df_full.loc[paid_mask].copy()
        else:
            print("Nessuna riga con stato 'paid': analisi su tutte le righe.")
            df = df_full
    else:
        print("\nColonna Order Status non disponibile: analisi su tutte le righe.")
        df = df_full

    original_ticket_type_col = ticket_type_col
    ticket_total_num_for_volunteers = numeric_map.get(ticket_total_col)
    volunteer_info = apply_volunteer_enrichment(
        df=df,
        volunteers_cfg=volunteers_cfg,
        ticket_type_col=original_ticket_type_col,
        ticket_total_num=ticket_total_num_for_volunteers,
        payment_date_col=payment_date_col,
        attendee_email_col=attendee_email_col,
        buyer_email_col=buyer_email_col,
        name_col=name_col,
        first_name_col=first_name_col,
        last_name_col=last_name_col,
        order_number_col=order_number_col,
        ticket_code_col=ticket_code_col,
        ticket_id_col=ticket_id_col,
        csv_dir=csv_dir,
    )
    if volunteer_info.get("analysis_ticket_type_col"):
        ticket_type_col = str(volunteer_info["analysis_ticket_type_col"])

    clean_path = csv_dir / "tickets_clean.csv"
    try:
        df.to_csv(clean_path, index=False, encoding="utf-8")
        print(f"\nFile pulito salvato in: {clean_path}")
    except PermissionError:
        fallback_clean_path = csv_dir / "tickets_clean_recovered.csv"
        df.to_csv(fallback_clean_path, index=False, encoding="utf-8")
        print(f"\nFile pulito salvato in fallback (per file bloccato): {fallback_clean_path}")

    missing_report_path = csv_dir / "column_missing_report.txt"
    write_missing_report(df_raw, missing_report_path, df_paid=df)

    n_rows, n_cols = df.shape
    print(f"\nRighe (record biglietti): {n_rows:,}")
    print(f"Colonne: {n_cols:,}")

    key_fields = [
        col_value("event_name", "Event Name"),
        order_number_col,
        order_status_col,
        payment_date_col,
        PARSED_DATE_COL,
        attendee_email_col,
        buyer_email_col,
        ticket_type_col,
        col_value("ticket_code", "Ticket Code"),
        col_value("ticket_id", "Ticket ID"),
        numeric_map.get(ticket_total_col),
        numeric_map.get(order_total_col),
        numeric_map.get(col_value("price", "Price")),
        discount_col,
        *geo_report_cols,
    ]
    key_fields = [c for c in key_fields if c and c in df.columns]
    report_missing(df, key_fields, "Campi chiave")

    if key_fields:
        sample_cols = key_fields[: min(len(key_fields), 10)]
        print("\nAnteprima dei primi record (solo campi chiave):")
        print(df[sample_cols].head(10).to_string(index=False))

    # === Vendite & ricavi ====================================================
    ticket_total_num = numeric_map.get(ticket_total_col)
    order_total_num = numeric_map.get(order_total_col)
    report_missing(
        df,
        [ticket_total_num, order_total_num, ticket_type_col],
        "Metriche economiche",
    )

    tot_tickets = len(df)
    tot_revenue_ticket = (
        df[ticket_total_num].sum() if ticket_total_num in df.columns else np.nan
    )
    tot_revenue_order = (
        df[order_total_num].sum() if order_total_num in df.columns else np.nan
    )
    avg_ticket_price = (
        df[ticket_total_num].mean() if ticket_total_num in df.columns else np.nan
    )

    print(f"\nTotale biglietti: {tot_tickets:,}")
    if ticket_total_num in df.columns:
        print(f"Somma {ticket_total_num}: {tot_revenue_ticket:,.2f}")
        print(f"Prezzo medio per riga: {avg_ticket_price:,.2f}")
    if order_total_num in df.columns:
        print(f"Somma {order_total_num}: {tot_revenue_order:,.2f}")
    if volunteer_info.get("enabled") and volunteer_info.get("matched_count", 0):
        volunteer_refund = float(volunteer_info.get("refund_total", 0.0) or 0.0)
        print(f"Potenziale rimborso volontari: {volunteer_refund:,.2f}")
        if ticket_total_num in df.columns:
            print(f"Ticket Total netto dopo potenziale rimborso volontari: {tot_revenue_ticket - volunteer_refund:,.2f}")

    # Order Status distribution (usa sempre il dataset completo)
    if order_status_col and order_status_col in df_full.columns:
        order_status_counts = df_full[order_status_col].fillna("NaN").value_counts()
        print("\nDistribuzione Order Status (dataset completo):")
        print(order_status_counts)

        if plots_enabled:
            status_plot = drop_nan_categories(order_status_counts)
            fig, ax = plt.subplots(figsize=(6, 4))
            status_plot.plot(kind="bar", ax=ax, color="#455a64")
            ax.set_title("Distribuzione Order Status")
            ax.set_ylabel("Ordini")
            ax.set_xlabel("Status")
            for container in ax.containers:
                ax.bar_label(container, padding=2)
            fig.tight_layout()
            save_plot(fig, plots_dir, "order_status_distribution", plot_format)

    if ticket_type_col in df.columns:
        by_type = (
            df.groupby(ticket_type_col, dropna=False)
            .agg(
                tickets=(ticket_type_col, "size"),
                revenue=(ticket_total_num, "sum") if ticket_total_num in df.columns else (ticket_type_col, "size"),
                avg_price=(ticket_total_num, "mean")
                if ticket_total_num in df.columns
                else (ticket_type_col, "size"),
            )
            .sort_values(["tickets"], ascending=False)
        )
        print("\nVendite per tipo di ticket:")
        print(by_type.head(20))
        by_type_df = by_type.copy()

        if plots_enabled:
            save_chunked_barh_plot(
                by_type[["tickets"]].copy(),
                plots_dir,
                "ticket_type_counts",
                plot_format,
                chunk_size=15,
                fig_width=14.5,
                base_height=5.4,
                row_height_scale=0.62,
                label_font_size=12,
                title_font_size=16,
                dpi_override=250,
            )

        export_focused_ticket_summary(
            df,
            ticket_type_col,
            output_dir,
            plots_dir,
            plot_format,
            focused_ticket_cfg,
        )
        export_full_festival_ticket_type_summary(
            df,
            ticket_type_col,
            ticket_total_num,
            csv_dir,
            plots_dir,
            plot_format,
            plots_enabled,
        )
        export_accessory_ticket_type_summary(
            df,
            ticket_type_col,
            ticket_total_num,
            csv_dir,
            plots_dir,
            plot_format,
            plots_enabled,
        )

    # === Ricavi per fase (da Ticket Type) ====================================
    if ticket_type_col in df.columns:
        df["phase_label"] = df[ticket_type_col].map(extract_phase_label)
        df["ticket_type_amount_eur"] = df[ticket_type_col].map(extract_ticket_type_amount)
        phase_table = (
            df.groupby("phase_label", dropna=False)
            .agg(
                tickets=(ticket_type_col, "size"),
                amount_eur=("ticket_type_amount_eur", "sum"),
                amount_missing=("ticket_type_amount_eur", lambda s: s.isna().sum()),
            )
            .sort_values(["amount_eur", "tickets"], ascending=False)
        )
        total_row = pd.DataFrame(
            {
                "tickets": [phase_table["tickets"].sum()],
                "amount_eur": [phase_table["amount_eur"].sum()],
                "amount_missing": [phase_table["amount_missing"].sum()],
            },
            index=["TOTAL"],
        )
        phase_table = pd.concat([phase_table, total_row])
        phase_table = phase_table.rename(columns={"amount_eur": "Total Amount (€)"})
        phase_table["Total Amount (€)"] = phase_table["Total Amount (€)"].map(format_eur)
        phase_table_df = phase_table.copy()
        phase_path = csv_dir / "phase_revenue_from_ticket_type.csv"
        phase_table.to_csv(phase_path, encoding="utf-8")
        print(f"\nRicavi per fase (da Ticket Type) salvati in: {phase_path}")
        save_table_image(phase_table, plots_dir, "table_phase_revenue_from_ticket_type", plot_format)
        if "TOTAL" in phase_table.index:
            total_eur = phase_table.loc["TOTAL", "Total Amount (€)"]
            print(f"Totale incassato (da Ticket Type): {total_eur}")

    # === Ambassador summary ==================================================
    if ticket_type_col in df.columns or ticket_id_col in df.columns:
        amb_sources = [c for c in [ticket_type_col, ticket_id_col] if c and c in df.columns]
        if amb_sources:
            def find_ambassador(row: pd.Series) -> str:
                for col in amb_sources:
                    name = extract_ambassador_name(row.get(col))
                    if name:
                        return name
                return ""

            amb_name_col = "__ambassador_name_calc"
            while amb_name_col in df.columns:
                amb_name_col = f"{amb_name_col}_x"
            phase_col = "__phase_label_calc"
            while phase_col in df.columns:
                phase_col = f"{phase_col}_x"
            ticket_type_amount_col = "__ticket_type_amount_eur_calc"
            while ticket_type_amount_col in df.columns:
                ticket_type_amount_col = f"{ticket_type_amount_col}_x"
            row_amount_col = "__row_amount_eur_calc"
            while row_amount_col in df.columns:
                row_amount_col = f"{row_amount_col}_x"

            df[amb_name_col] = df.apply(find_ambassador, axis=1)
            ambassadors = df[df[amb_name_col] != ""].copy()
            if not ambassadors.empty:
                if ticket_type_col and ticket_type_col in ambassadors.columns:
                    ambassadors[phase_col] = ambassadors[ticket_type_col].map(extract_phase_label)
                    ambassadors[ticket_type_amount_col] = ambassadors[ticket_type_col].map(
                        extract_ticket_type_amount
                    )
                else:
                    ambassadors[phase_col] = "unknown"
                    ambassadors[ticket_type_amount_col] = np.nan

                if ticket_total_num and ticket_total_num in ambassadors.columns:
                    ambassadors[row_amount_col] = ambassadors[ticket_total_num]
                    ambassadors[row_amount_col] = ambassadors[row_amount_col].fillna(
                        ambassadors[ticket_type_amount_col]
                    )
                else:
                    ambassadors[row_amount_col] = ambassadors[ticket_type_amount_col]

                phase_counts = (
                    ambassadors.groupby([amb_name_col, phase_col], dropna=False)
                    .size()
                    .unstack(fill_value=0)
                    .astype(int)
                )

                def phase_sort_key(label: object) -> tuple[int, object]:
                    txt = str(label)
                    m = re.match(r"phase_(\d+)$", txt)
                    if m:
                        return (0, int(m.group(1)))
                    if txt == "early_bird":
                        return (1, 0)
                    if txt == "christmas":
                        return (2, 0)
                    if txt == "ambassador":
                        return (3, 0)
                    if txt == "unknown":
                        return (9, 0)
                    return (4, txt)

                phase_cols = sorted(phase_counts.columns.tolist(), key=phase_sort_key)
                phase_counts = phase_counts[phase_cols]

                phase_price_map: dict[str, str] = {}
                phase_price_source = ambassadors[[phase_col, ticket_type_amount_col]].copy()
                phase_price_source = phase_price_source.dropna(subset=[ticket_type_amount_col])
                if not phase_price_source.empty:
                    for phase_name, group in phase_price_source.groupby(phase_col, dropna=False):
                        prices = sorted({int(round(float(v))) for v in group[ticket_type_amount_col].tolist()})
                        if prices:
                            if len(prices) == 1:
                                phase_price_map[str(phase_name)] = f"{prices[0]} €"
                            else:
                                joined = "/".join(str(p) for p in prices)
                                phase_price_map[str(phase_name)] = f"{joined} €"

                amount_by_amb = ambassadors.groupby(amb_name_col, dropna=False)[row_amount_col].sum()
                amount_col = "Total Amount (\u20ac)"
                amb_table = phase_counts.copy()
                amb_table.insert(0, "tickets_total", amb_table.sum(axis=1))
                amb_table[amount_col] = amount_by_amb
                amb_table[amount_col] = amb_table[amount_col].fillna(0.0)
                amb_table = amb_table.sort_values(["tickets_total"], ascending=False)
                amb_table.index.name = "ambassador"

                total_values: dict[str, object] = {"tickets_total": int(amb_table["tickets_total"].sum())}
                for col in phase_cols:
                    total_values[col] = int(amb_table[col].sum())
                total_values[amount_col] = float(amb_table[amount_col].sum())
                total_row = pd.DataFrame([total_values], index=["TOTAL"])
                amb_table = pd.concat([amb_table, total_row])
                phase_renames: dict[str, str] = {}
                for col in phase_cols:
                    price_label = phase_price_map.get(str(col))
                    if price_label:
                        phase_renames[col] = f"{col} ({price_label})"
                if phase_renames:
                    amb_table = amb_table.rename(columns=phase_renames)
                amb_table[amount_col] = amb_table[amount_col].map(format_eur)
                amb_table_df = amb_table.copy()
                amb_path = csv_dir / "ambassador_sales.csv"
                amb_table.to_csv(amb_path, encoding="utf-8")
                print(f"\nReport ambassador salvato in: {amb_path}")
                shown_cols_count = len(amb_table.columns) + 1
                first_col_width = 0.34
                other_col_width = (1.0 - first_col_width) / max(1, shown_cols_count - 1)
                amb_col_widths = [first_col_width] + [other_col_width] * (shown_cols_count - 1)
                table_width = max(14.0, 3.25 * shown_cols_count)
                save_table_image(
                    amb_table,
                    plots_dir,
                    "table_ambassador_sales",
                    plot_format,
                    highlight_value="TOTAL",
                    manual_table=True,
                    col_widths_override=amb_col_widths,
                    font_size=22,
                    fig_width_override=table_width,
                    fig_height_override=max(10.0, 1.05 * len(amb_table)),
                    header_font_size=16,
                    row_height_override=2.3,
                    header_height_override=2.6,
                    dpi_override=250,
                )
                save_chunked_table_image(
                    amb_table,
                    plots_dir,
                    "table_ambassador_sales_readable",
                    plot_format,
                    chunk_size=10,
                    font_size=22,
                    header_font_size=19,
                    panel_width=max(18.0, 3.4 * shown_cols_count),
                    panel_height=6.2,
                    row_height_scale=1.75,
                    dpi_override=350,
                )

    # === Payment Gateway =====================================================
    if payment_gateway_col and payment_gateway_col in df.columns:
        by_gateway = (
            df.groupby(payment_gateway_col, dropna=False)
            .agg(
                tickets=(payment_gateway_col, "size"),
                revenue=(ticket_total_num, "sum") if ticket_total_num in df.columns else (payment_gateway_col, "size"),
            )
            .sort_values(["tickets"], ascending=False)
        )
        print("\nDistribuzione Payment Gateway:")
        print(by_gateway.head(20))

    # Timeline vendite
    if PARSED_DATE_COL in df.columns:
        ts = df.dropna(subset=[PARSED_DATE_COL]).copy()
        if not ts.empty:
            ts["date"] = ts[PARSED_DATE_COL].dt.date
            daily = ts.groupby("date").size().sort_index()
            print("\nTimeline vendite (prime righe):")
            print(daily.head())
            if plots_enabled:
                plot_sales_timelines(daily, parsed_timeline_markers, plots_dir, plot_format)
        else:
            print("\nNessuna data valida per la timeline vendite.")

    monthly_sales = build_monthly_ticket_sales_summary(df, ticket_type_col)
    if not monthly_sales.empty:
        monthly_sales_path = csv_dir / "monthly_ticket_sales_average.csv"
        monthly_sales.to_csv(monthly_sales_path, index=False, encoding="utf-8")
        print(f"\nRiepilogo mensile vendite salvato in: {monthly_sales_path}")
        print("\nMedia giornaliera vendite per mese:")
        print(
            monthly_sales[
                [
                    "month",
                    "observed_days",
                    "all_tickets",
                    "all_avg_per_observed_day",
                    "full_festival_tickets",
                    "full_festival_avg_per_observed_day",
                ]
            ].to_string(index=False)
        )
        if plots_enabled:
            plot_monthly_ticket_sales_summary(monthly_sales, plots_dir, plot_format)

    # === Provenienza geografica ==============================================
    analyze_geography(df, geo_country_cols, geo_city_cols, plots_enabled, plots_dir, plot_format)

    # === Demografia (date di nascita) =========================================
    if dob_col in df.columns:
        dob_parsed_col = "BirthDate_parsed"
        df[dob_parsed_col] = pd.to_datetime(df[dob_col].map(parse_birth_date), errors="coerce")
        dob_valid = df.dropna(subset=[dob_parsed_col]).copy()
        if dob_valid.empty:
            print("\nDate di nascita non disponibili o non parsabili.")
        else:
            today = pd.Timestamp.today().normalize()
            ages = (today - dob_valid[dob_parsed_col]).dt.days / 365.25
            dob_valid["Eta (anni)"] = ages
            print("\nStatistiche Eta basate sulle date di nascita:")
            print(dob_valid["Eta (anni)"].describe().round(1))

            birth_year_counts = (
                dob_valid[dob_parsed_col].dt.year.value_counts().sort_index()
            )
            print("\nPartecipanti per anno di nascita (prime righe):")
            print(birth_year_counts.head(20))

            if plots_enabled:
                age_years = ages.dropna().astype(float).round(0).astype(int)
                age_counts = age_years.value_counts().sort_index()

                fig, ax = plt.subplots(figsize=(10, 9))
                ax.barh(age_counts.index.astype(str), age_counts.values, color="#5d4037", height=0.9)
                ax.set_xlabel("Partecipanti")
                ax.set_ylabel("Eta (anni)")
                ax.xaxis.set_major_locator(MultipleLocator(5))
                ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x)}"))
                ax.tick_params(axis="both", labelsize=12)
                ax.xaxis.label.set_size(13)
                ax.yaxis.label.set_size(13)
                fig.tight_layout()
                save_plot(fig, plots_dir, "eta_partecipanti", plot_format)

                fig, ax = plt.subplots(figsize=(10, 12))
                tail_years = birth_year_counts.tail(40)
                ax.barh(tail_years.index.astype(str), tail_years.values, color="#3949ab", height=0.9)
                ax.set_xlabel("Partecipanti")
                ax.set_ylabel("Anno")
                ax.xaxis.set_major_locator(MultipleLocator(5))
                ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x)}"))
                ax.tick_params(axis="both", labelsize=12)
                ax.xaxis.label.set_size(13)
                ax.yaxis.label.set_size(13)
                fig.tight_layout()
                save_plot(fig, plots_dir, "nascite_per_anno", plot_format)

    # === Sconti ===============================================================
    report_missing(df, [discount_col, numeric_map.get(ticket_discount_col)], "Sconti")
    if discount_col in df.columns:
        has_discount = df[discount_col].fillna("").str.strip() != ""
        n_disc = int(has_discount.sum())
        print(f"\nRighe con codice sconto: {n_disc} ({n_disc / len(df) * 100:.1f}%)")
        disc_counts = df.loc[has_discount, discount_col].value_counts()
        print("\nFrequenza codici sconto:")
        print(disc_counts.head(30))
        if plots_enabled and not disc_counts.empty:
            fig, ax = plt.subplots(figsize=(8, 4))
            disc_counts.plot(kind="bar", ax=ax, color="#ff8f00")
            ax.set_title("Frequenza codici sconto")
            ax.set_ylabel("Occorrenze")
            fig.tight_layout()
            save_plot(fig, plots_dir, "codici_sconto", plot_format)

        ticket_discount_num = numeric_map.get(ticket_discount_col)
        if ticket_discount_num in df.columns:
            avg_discount = df.loc[has_discount, ticket_discount_num].mean()
            print(f"Sconto medio (solo righe con codice): {avg_discount:.2f}")

    # === Check-in =============================================================
    present_checkin_cols = [c for c in checkin_columns if c in df.columns]
    report_missing(df, present_checkin_cols, "Check-in")

    for col_name in present_checkin_cols:
        unique_values = df[col_name].dropna().unique()[:10]
        print(f"\nEsempi per '{col_name}': {unique_values}")
        counts = df[col_name].astype(str).str.strip().str.lower().value_counts(dropna=False)
        print(counts)

    # === Duplicati & qualita dati ============================================
    dup_targets = [
        attendee_email_col,
        buyer_email_col,
        order_number_col,
        ticket_type_col,
        ticket_total_num,
    ]
    report_missing(df, dup_targets, "Duplicate / Qualita")

    if attendee_email_col in df.columns:
        vc = df[attendee_email_col].value_counts()
        dup_att = vc[vc > 1]
        print(f"\nEmail attendee con piu di un record: {len(dup_att)}")
        print(dup_att.head(20))

    if order_number_col in df.columns:
        vc_ord = df[order_number_col].value_counts()
        dup_orders = vc_ord[vc_ord > 1]
        print(f"\nOrdini con piu di una riga: {len(dup_orders)}")
        print(dup_orders.head(20))

    if ticket_total_num in df.columns and order_status_col in df.columns:
        suspect = df[
            (df[ticket_total_num].fillna(0) == 0)
            & (df[order_status_col].astype(str).str.lower() == "paid")
        ]
        print(f"\nRighe con {ticket_total_num} = 0 ma ordine 'Paid': {len(suspect)}")
        if not suspect.empty:
            cols = [
                order_number_col,
                order_status_col,
                ticket_total_col,
                ticket_total_num,
                discount_col,
            ]
            cols = [c for c in cols if c in suspect.columns]
            print(suspect[cols].head(10).to_string(index=False))

    key_missing_cols = [
        c
        for c in [attendee_email_col, buyer_email_col, order_number_col, ticket_type_col, ticket_total_num]
        if c
    ]
    if key_missing_cols:
        missing_summary = (
            df[key_missing_cols].isna().mean().sort_values(ascending=False) * 100
        )
        print("\nPercentuale di valori NaN sui campi chiave:")
        print(missing_summary.round(1))

    # === Esportazioni ========================================================
    export_summary_tables(
        df,
        geo_country_cols,
        geo_city_cols,
        ticket_type_col,
        ticket_total_num,
        payment_gateway_col,
        csv_dir,
        plots_dir,
        plot_format,
    )

    if bool(pdf_cfg.get("enabled", True)):
        export_detailed_pdf_report(
            output_dir=output_dir,
            csv_path=csv_path,
            df_raw=df_raw,
            df=df,
            csv_dir=csv_dir,
            plots_dir=plots_dir,
            timeline_markers=timeline_markers,
            ticket_type_col=ticket_type_col,
            ticket_total_num=ticket_total_num,
            order_total_num=order_total_num,
            order_status_col=order_status_col,
            country_col=country_col,
            city_col=city_col,
            dob_col=dob_col,
            order_status_counts=order_status_counts,
            by_type=by_type_df,
            phase_table=phase_table_df,
            amb_table=amb_table_df,
        )

    if bool(narrative_pdf_cfg.get("enabled", True)):
        export_narrative_pdf_report(
            output_dir=output_dir,
            csv_path=csv_path,
            df_raw=df_raw,
            df=df,
            timeline_markers=timeline_markers,
            ticket_type_col=ticket_type_col,
            ticket_total_num=ticket_total_num,
            order_total_num=order_total_num,
            order_status_counts=order_status_counts,
            by_type=by_type_df,
            phase_table=phase_table_df,
            amb_table=amb_table_df,
        )

    print("\nAnalisi completata.")


if __name__ == "__main__":
    main()

