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

    columns_cfg: Dict[str, str] = config.get("columns", {})
    checkin_columns = config.get("checkin_columns", DEFAULT_CHECKIN_COLUMNS)

    def col_value(key: str, fallback: Optional[str]) -> Optional[str]:
        value = columns_cfg.get(key)
        return value or fallback

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

    payment_date_col = col_value("payment_date", "Payment Date")
    ticket_type_col = col_value("ticket_type", "Ticket Type")
    ticket_total_col = col_value("ticket_total", "Ticket Total")
    order_total_col = col_value("order_total", "Order Total")
    discount_col = col_value("discount_code", "Discount Code")
    ticket_discount_col = col_value("ticket_discount", "Ticket Discount")
    country_col = col_value(
        "country", "Country of residence / Paese di residenza (Campi ticket holder)"
    )
    city_col = col_value(
        "city", "City of residence / Citt\u00e0 di residenza (Campi ticket holder)"
    )
    attendee_email_col = col_value("attendee_email", "Attendee E-mail")
    buyer_email_col = col_value("buyer_email", "Buyer E-Mail")
    order_number_col = col_value("order_number", "Order Number")
    order_status_col = col_value("order_status", "Order Status")

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

    clean_path = output_dir / "tickets_clean.csv"
    df.to_csv(clean_path, index=False, encoding="utf-8")
    print(f"\nFile pulito salvato in: {clean_path}")

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
        country_col,
        city_col,
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
            fig, ax = plt.subplots(figsize=(10, 5))
            by_type["tickets"].plot(kind="bar", ax=ax, color="#1976d2")
            ax.set_title("Conteggio biglietti per tipo")
            ax.set_ylabel("Biglietti")
            ax.set_xlabel(ticket_type_col)
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
                daily.plot(ax=ax, marker="o", color="#388e3c")
                ax.set_title("Biglietti venduti per giorno")
                ax.set_xlabel("Data")
                ax.set_ylabel("Biglietti")
                ax.grid(True, alpha=0.3)
                fig.tight_layout()
                save_plot(fig, plots_dir, "vendite_giornaliere", plot_format)

                fig, ax = plt.subplots(figsize=(10, 4))
                daily.cumsum().plot(ax=ax, marker="o", color="#d32f2f")
                ax.set_title("Biglietti cumulati")
                ax.set_xlabel("Data")
                ax.set_ylabel("Cumulato")
                ax.grid(True, alpha=0.3)
                fig.tight_layout()
                save_plot(fig, plots_dir, "vendite_cumulative", plot_format)
        else:
            print("\nNessuna data valida per la timeline vendite.")

    # === Provenienza geografica ==============================================
    report_missing(df, [country_col, city_col], "Geografia")

    if country_col in df.columns:
        by_country = df.groupby(country_col, dropna=False).size().sort_values(ascending=False)
        print("\nTop paesi:")
        print(by_country.head(20))
        if plots_enabled:
            fig, ax = plt.subplots(figsize=(10, 4))
            by_country.head(15).plot(kind="bar", ax=ax, color="#6a1b9a")
            ax.set_title("Top 15 paesi")
            ax.set_ylabel("Biglietti")
            fig.tight_layout()
            save_plot(fig, plots_dir, "top_paesi", plot_format)

    if city_col in df.columns:
        by_city = df.groupby(city_col, dropna=False).size().sort_values(ascending=False)
        print("\nTop citt\u00e0:")
        print(by_city.head(20))
        if plots_enabled:
            fig, ax = plt.subplots(figsize=(10, 4))
            by_city.head(15).plot(kind="bar", ax=ax, color="#00838f")
            ax.set_title("Top 15 citt\u00e0")
            ax.set_ylabel("Biglietti")
            fig.tight_layout()
            save_plot(fig, plots_dir, "top_citta", plot_format)

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
    if country_col in df.columns:
        exports["by_country.csv"] = (
            df.groupby(country_col, dropna=False)
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
