#!/usr/bin/env python3
"""
Genera grafici di vendite giornaliere e cumulative a partire da CSV Tickera.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.dates import DateFormatter, MonthLocator

READ_KWARGS = {
    "sep": ",",
    "quotechar": '"',
    "encoding": "utf-8",
    "engine": "python",
    "dtype": str,
    "skip_blank_lines": True,
}


def normalize_columns(columns: Iterable[str]) -> List[str]:
    seen: dict[str, int] = {}
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
    for fmt in (
        "%d/%m/%Y - %H:%M",
        "%d/%m/%Y %H:%M",
        "%d/%m/%Y",
        "%d-%m-%Y %H:%M",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
    ):
        try:
            return pd.Timestamp(datetime.strptime(s, fmt))
        except ValueError:
            continue
    parsed = pd.to_datetime(s, errors="coerce", dayfirst=True)
    return parsed if not pd.isna(parsed) else pd.NaT


def resolve_column(df: pd.DataFrame, preferred: str) -> Optional[str]:
    if preferred in df.columns:
        return preferred
    preferred_lower = preferred.lower()
    for col in df.columns:
        if col.lower() == preferred_lower:
            return col
    return None


def slugify(text: str) -> str:
    safe = "".join(ch.lower() if ch.isalnum() else "_" for ch in text)
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe.strip("_")[:120] or "file"


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, **READ_KWARGS)
    df.columns = normalize_columns(df.columns)
    return df


def build_daily_counts(df: pd.DataFrame, date_col: str) -> pd.Series:
    parsed_col = "_payment_date_parsed"
    df[parsed_col] = df[date_col].map(parse_payment_date)
    valid = df.dropna(subset=[parsed_col]).copy()
    if valid.empty:
        return pd.Series(dtype=int)
    daily = valid.groupby(valid[parsed_col].dt.date).size()
    daily.index = pd.to_datetime(daily.index)
    daily = daily.sort_index()
    full_index = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    return daily.reindex(full_index, fill_value=0)


def plot_daily_sales(daily: pd.Series, label: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(daily.index, daily.values, marker="o", color="#1e88e5")
    ax.set_title(f"Vendite giornaliere - {label}")
    ax.set_xlabel("Data")
    ax.set_ylabel("Biglietti")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_cumulative_sales(daily: pd.Series, label: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    cumulative = daily.cumsum()
    ax.plot(cumulative.index, cumulative.values, marker="o", color="#d81b60")
    ax.set_title(f"Vendite cumulative - {label}")
    ax.set_xlabel("Data")
    ax.set_ylabel("Cumulato")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def build_season_index(start_month: int, season_months: int) -> pd.DatetimeIndex:
    base_year = 2000
    start = pd.Timestamp(base_year, start_month, 1)
    end = start + pd.DateOffset(months=season_months) - pd.Timedelta(days=1)
    return pd.date_range(start, end, freq="D")


def map_date_to_season(
    ts: pd.Timestamp,
    start_month: int,
    first_real_date: pd.Timestamp,
) -> pd.Timestamp:
    base_year = 2000
    if ts.month > start_month:
        year = base_year
    elif ts.month < start_month:
        year = base_year + 1
    else:
        year = base_year + 1 if ts > first_real_date else base_year
    return pd.Timestamp(year, ts.month, ts.day)


def normalize_daily_to_season(
    daily: pd.Series,
    start_month: int,
    season_months: int,
    first_real_date: pd.Timestamp,
) -> pd.Series:
    if daily.empty:
        return daily
    normalized_dates = []
    for ts in daily.index:
        normalized_dates.append(map_date_to_season(ts, start_month, first_real_date))
    normalized = pd.Series(daily.values, index=pd.to_datetime(normalized_dates))
    normalized = normalized.groupby(level=0).sum().sort_index()
    season_index = build_season_index(start_month, season_months)
    return normalized.reindex(season_index, fill_value=0)


def trim_to_span(
    series: pd.Series,
    first_date: pd.Timestamp,
    last_date: pd.Timestamp,
) -> pd.Series:
    trimmed = series.copy()
    if first_date <= last_date:
        mask = (trimmed.index >= first_date) & (trimmed.index <= last_date)
    else:
        # Il range attraversa il confine Luglio->Luglio.
        mask = (trimmed.index >= first_date) | (trimmed.index <= last_date)
    trimmed[~mask] = pd.NA
    return trimmed


def plot_combined_daily(series_map: Dict[str, pd.Series], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    first_series: Optional[pd.Series] = None
    for label, daily in series_map.items():
        if first_series is None:
            first_series = daily
        ax.plot(daily.index, daily.values, linewidth=1.3, label=label)
        nonzero = daily[daily > 0]
        ax.scatter(
            nonzero.index,
            nonzero.values,
            s=16,
            marker="o",
            alpha=0.9,
        )
    ax.set_title("Vendite giornaliere - confronto stagionale (Luglio->Luglio)")
    ax.set_xlabel("Mese")
    ax.set_ylabel("Biglietti")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    if first_series is not None:
        ax.set_xlim(first_series.index.min(), first_series.index.max())
    ax.xaxis.set_major_locator(MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(DateFormatter("%b"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_combined_cumulative(series_map: Dict[str, pd.Series], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    first_series: Optional[pd.Series] = None
    for label, daily in series_map.items():
        if first_series is None:
            first_series = daily
        daily_filled = daily.fillna(0)
        cumulative = daily_filled.cumsum()
        cumulative[daily.isna()] = pd.NA
        ax.plot(
            cumulative.index,
            cumulative.values,
            marker="o",
            markersize=3,
            linewidth=1.3,
            label=label,
        )
    ax.set_title("Vendite cumulative - confronto stagionale (Luglio->Luglio)")
    ax.set_xlabel("Mese")
    ax.set_ylabel("Cumulato")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    if first_series is not None:
        ax.set_xlim(first_series.index.min(), first_series.index.max())
    ax.xaxis.set_major_locator(MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(DateFormatter("%b"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_combined_cumulative_normalized(
    series_map: Dict[str, pd.Series],
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    first_series: Optional[pd.Series] = None
    for label, daily in series_map.items():
        if first_series is None:
            first_series = daily
        daily_filled = daily.fillna(0)
        cumulative = daily_filled.cumsum()
        final_value = cumulative.max()
        if final_value and not pd.isna(final_value):
            cumulative = cumulative / final_value * 100.0
        cumulative[daily.isna()] = pd.NA
        ax.plot(
            cumulative.index,
            cumulative.values,
            marker="o",
            markersize=3,
            linewidth=1.3,
            label=label,
        )
    ax.set_title("Vendite cumulative - confronto stagionale (normalizzato)")
    ax.set_xlabel("Mese")
    ax.set_ylabel("Cumulato (%)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    if first_series is not None:
        ax.set_xlim(first_series.index.min(), first_series.index.max())
    ax.xaxis.set_major_locator(MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(DateFormatter("%b"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_combined_cumulative_aligned_end(
    series_map: Dict[str, pd.Series],
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    for label, daily in series_map.items():
        daily_filled = daily.fillna(0)
        cumulative = daily_filled.cumsum()
        cumulative[daily.isna()] = pd.NA
        valid = cumulative.dropna()
        if valid.empty:
            continue
        # Allinea l'ultima entry a x=0 (giorni prima in negativo).
        x_vals = list(range(-len(valid) + 1, 1))
        ax.plot(
            x_vals,
            valid.values,
            marker="o",
            markersize=3,
            linewidth=1.3,
            label=label,
        )
    ax.set_title("Vendite cumulative - allineate all'ultima entry")
    ax.set_xlabel("Giorni prima dell'ultima entry")
    ax.set_ylabel("Cumulato")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def process_file(
    path: Path,
    output_dir: Path,
    payment_date_col: str,
    fmt: str,
    season_start_month: int,
    season_months: int,
) -> Optional[Tuple[pd.Series, pd.Timestamp]]:
    if not path.exists():
        print(f"[SKIP] File non trovato: {path}")
        return None
    df = load_csv(path)
    resolved_col = resolve_column(df, payment_date_col)
    if resolved_col is None:
        print(f"[SKIP] Colonna '{payment_date_col}' non trovata in {path.name}")
        return None

    daily = build_daily_counts(df, resolved_col)
    if daily.empty:
        print(f"[SKIP] Nessuna data valida in {path.name}")
        return None

    label = path.stem
    slug = slugify(label)
    daily_path = output_dir / f"{slug}_vendite_giornaliere.{fmt}"
    cumulative_path = output_dir / f"{slug}_vendite_cumulative.{fmt}"

    plot_daily_sales(daily, label, daily_path)
    plot_cumulative_sales(daily, label, cumulative_path)

    print(f"[OK] Salvati: {daily_path.name}, {cumulative_path.name}")
    first_real_date = pd.Timestamp(daily.index.min())
    last_real_date = pd.Timestamp(daily.index.max())
    normalized = normalize_daily_to_season(
        daily,
        season_start_month,
        season_months,
        first_real_date,
    )
    first_date = map_date_to_season(first_real_date, season_start_month, first_real_date)
    last_date = map_date_to_season(last_real_date, season_start_month, first_real_date)
    return normalized, first_date, last_date


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crea grafici di vendite giornaliere e cumulative da CSV Tickera."
    )
    parser.add_argument(
        "--inputs",
        nargs="*",
        default=[
            "Documenti/Tickets/Lista_ticket_7chakras_2019_onlyGood_VERIFIED_FLAT.csv",
            "Documenti/Tickets/Lista_ticket_7chakras_2025_onlyGood_VERIFIED_FLAT.csv",
            "Documenti/Tickets/Attendee_List_Paid_19Gen_16.18pm_FLAT.csv",
        ],
        help="Lista dei CSV di input (default: i 3 file forniti).",
    )
    parser.add_argument(
        "--output-dir",
        default="output/vendite_distribuzione",
        help="Cartella di output per i grafici.",
    )
    parser.add_argument(
        "--payment-date-col",
        default="Payment Date",
        help="Nome colonna con la data pagamento.",
    )
    parser.add_argument(
        "--format",
        default="png",
        help="Formato immagine (es. png, jpg).",
    )
    parser.add_argument(
        "--season-start-month",
        type=int,
        default=7,
        help="Mese di inizio stagione per la comparazione (default: 7 = Luglio).",
    )
    parser.add_argument(
        "--season-months",
        type=int,
        default=13,
        help="Durata finestra stagionale in mesi (default: 13).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    season_series: Dict[str, pd.Series] = {}
    for input_path in args.inputs:
        path = Path(input_path)
        result = process_file(
            path,
            output_dir,
            args.payment_date_col,
            args.format,
            args.season_start_month,
            args.season_months,
        )
        if result is not None:
            normalized, first_date, last_date = result
            if not normalized.empty:
                season_series[path.stem] = trim_to_span(
                    normalized,
                    first_date,
                    last_date,
                )

    if season_series:
        combined_daily = output_dir / f"vendite_giornaliere_comparativa.{args.format}"
        combined_cumulative = output_dir / f"vendite_cumulative_comparativa.{args.format}"
        combined_cumulative_norm = (
            output_dir / f"vendite_cumulative_comparativa_norm.{args.format}"
        )
        combined_cumulative_aligned = (
            output_dir / f"vendite_cumulative_comparativa_allineata.{args.format}"
        )
        plot_combined_daily(season_series, combined_daily)
        plot_combined_cumulative(season_series, combined_cumulative)
        plot_combined_cumulative_normalized(season_series, combined_cumulative_norm)
        plot_combined_cumulative_aligned_end(season_series, combined_cumulative_aligned)
        print(
            "[OK] Salvati: "
            f"{combined_daily.name}, {combined_cumulative.name}, "
            f"{combined_cumulative_norm.name}, {combined_cumulative_aligned.name}"
        )


if __name__ == "__main__":
    main()
