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
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

# Usa backend "Agg" per salvare i grafici anche su server/headless.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (dipende dal backend impostato)
from datetime import datetime

READ_KWARGS: Dict[str, object] = {
    "sep": ",",
    "quotechar": '"',
    "encoding": "utf-8",
    "engine": "python",
    "dtype": str,
    "skip_blank_lines": True,
}

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


def load_config(path: Path) -> Dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Config non trovato: {path}")
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


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

    plots_cfg = config.get("plots", {})
    plots_enabled = plots_cfg.get("enabled", True)
    plot_format = plots_cfg.get("format", "png")
    plots_dir = output_dir / "plots"
    timeline_markers = config.get("timeline_markers", []) or []

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
        return name

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
            ts = pd.to_datetime(date_str, errors="coerce")
            if pd.isna(ts):
                continue
            parsed.append({"label": label, "date": ts.normalize(), "color": color})
        return parsed

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
    parsed_timeline_markers = parse_timeline_markers(timeline_markers)
    attendee_email_col = ensure_existing(col_value("attendee_email", "Attendee E-mail"))
    buyer_email_col = ensure_existing(col_value("buyer_email", "Buyer E-Mail"))
    order_number_col = ensure_existing(col_value("order_number", "Order Number"))
    order_status_col = ensure_existing(col_value("order_status", "Order Status"))
    dob_preferred = col_value("date_of_birth", "Date of birth / Data di nascita (Campi ticket holder)")
    dob_col = ensure_existing(
        dob_preferred,
        dob_preferred.replace("\u00e0", "\ufffd") if dob_preferred else None,
    )

    if payment_date_col and payment_date_col in df.columns:
        df[PARSED_DATE_COL] = df[payment_date_col].map(parse_payment_date)
    else:
        df[PARSED_DATE_COL] = pd.NaT

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

    clean_path = output_dir / "tickets_clean.csv"
    df.to_csv(clean_path, index=False, encoding="utf-8")
    print(f"\nFile pulito salvato in: {clean_path}")

    missing_report_path = output_dir / "column_missing_report.txt"
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
    key_fields = [c for c in key_fields if c]
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
            .sort_values(["revenue", "tickets"], ascending=False)
        )
        print("\nVendite per tipo di ticket:")
        print(by_type.head(20))

        if plots_enabled:
            type_counts = by_type["tickets"].sort_values(ascending=True)
            height = max(4, 0.6 * len(type_counts))
            fig, ax = plt.subplots(figsize=(10, height))
            type_counts.plot(kind="barh", ax=ax, color="#1976d2")
            ax.set_title("Conteggio biglietti per tipo")
            ax.set_xlabel("Biglietti")
            ax.set_ylabel("")
            fig.tight_layout()
            save_plot(fig, plots_dir, "ticket_type_counts", plot_format)

    # Timeline vendite
    if PARSED_DATE_COL in df.columns:
        ts = df.dropna(subset=[PARSED_DATE_COL]).copy()
        if not ts.empty:
            ts["date"] = ts[PARSED_DATE_COL].dt.date
            daily = ts.groupby("date").size()
            print("\nTimeline vendite (prime righe):")
            print(daily.head())
            if plots_enabled:
                fig, ax = plt.subplots(figsize=(10, 4))
                daily.plot(ax=ax, marker="o", color="#388e3c", label="Vendite")
                ax.set_title("Biglietti venduti per giorno")
                ax.set_xlabel("Data")
                ax.set_ylabel("Biglietti")
                ax.grid(True, alpha=0.3)
                legend_handles = []
                if parsed_timeline_markers:
                    for marker in parsed_timeline_markers:
                        mdate = marker["date"].date()
                        line = ax.axvline(
                            mdate, color=marker["color"], linestyle="--", alpha=0.8, linewidth=1.2, label=marker["label"]
                        )
                        legend_handles.append(line)
                if legend_handles:
                    ax.legend()
                fig.tight_layout()
                save_plot(fig, plots_dir, "vendite_giornaliere", plot_format)

                fig, ax = plt.subplots(figsize=(10, 4))
                daily.cumsum().plot(ax=ax, marker="o", color="#d32f2f", label="Cumulato")
                ax.set_title("Biglietti cumulati")
                ax.set_xlabel("Data")
                ax.set_ylabel("Cumulato")
                ax.grid(True, alpha=0.3)
                legend_handles = []
                if parsed_timeline_markers:
                    for marker in parsed_timeline_markers:
                        mdate = marker["date"].date()
                        line = ax.axvline(
                            mdate, color=marker["color"], linestyle="--", alpha=0.8, linewidth=1.2, label=marker["label"]
                        )
                        legend_handles.append(line)
                if legend_handles:
                    ax.legend()
                fig.tight_layout()
                save_plot(fig, plots_dir, "vendite_cumulative", plot_format)
        else:
            print("\nNessuna data valida per la timeline vendite.")

    # === Provenienza geografica ==============================================
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
                fig, ax = plt.subplots(figsize=(10, 4))
                country_plot.plot(kind="bar", ax=ax, color="#6a1b9a")
                ax.set_title(f"Top 15 paesi - {col}")
                ax.set_ylabel("Biglietti")
                fig.tight_layout()
                save_plot(fig, plots_dir, f"top_paesi_{slugify(col)}", plot_format)

    for col in geo_city_cols:
        by_city = df.groupby(col, dropna=False).size().sort_values(ascending=False)
        print(f"\nTop citt\u00e0 ({col}):")
        print(by_city.head(20))
        if plots_enabled:
            city_plot = drop_nan_categories(by_city).head(15)
            if city_plot.empty:
                print(f" - Nessun dato valido per il grafico citt\u00e0 ({col}).")
            else:
                fig, ax = plt.subplots(figsize=(10, 4))
                city_plot.plot(kind="bar", ax=ax, color="#00838f")
                ax.set_title(f"Top 15 citt\u00e0 - {col}")
                ax.set_ylabel("Biglietti")
                fig.tight_layout()
                save_plot(fig, plots_dir, f"top_citta_{slugify(col)}", plot_format)


    # === Demografia (date di nascita) =========================================
    if dob_col in df.columns:
        dob_parsed_col = "BirthDate_parsed"
        df[dob_parsed_col] = df[dob_col].map(parse_birth_date)
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
                fig, ax = plt.subplots(figsize=(8, 4))
                ages.plot(kind="hist", bins=min(10, len(dob_valid)), ax=ax, color="#5d4037")
                ax.set_title("Distribuzione Eta partecipanti")
                ax.set_xlabel("Eta (anni)")
                ax.set_ylabel("Partecipanti")
                fig.tight_layout()
                save_plot(fig, plots_dir, "eta_partecipanti", plot_format)

                fig, ax = plt.subplots(figsize=(8, 4))
                birth_year_counts.tail(40).plot(kind="bar", ax=ax, color="#3949ab")
                ax.set_title("Anno di nascita (ultimi 40 anni)")
                ax.set_xlabel("Anno")
                ax.set_ylabel("Partecipanti")
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
    exports: Dict[str, pd.DataFrame] = {}
    if ticket_type_col in df.columns:
        exports["by_type.csv"] = (
            df.groupby(ticket_type_col, dropna=False)
            .agg(
                tickets=(ticket_type_col, "size"),
                revenue=(ticket_total_num, "sum") if ticket_total_num in df.columns else (ticket_type_col, "size"),
                avg_price=(ticket_total_num, "mean")
                if ticket_total_num in df.columns
                else (ticket_type_col, "size"),
            )
            .sort_values(["revenue", "tickets"], ascending=False)
        )
    for idx, col in enumerate(geo_country_cols):
        fname = "by_country.csv" if idx == 0 else f"by_country_{slugify(col)}.csv"
        exports[fname] = (
            df.groupby(col, dropna=False)
            .size()
            .to_frame("tickets")
            .sort_values("tickets", ascending=False)
        )

    for name, table in exports.items():
        out_path = output_dir / name
        table.to_csv(out_path, encoding="utf-8")
        print(f"Esportato: {out_path}")

    print("\nAnalisi completata.")


if __name__ == "__main__":
    main()
