#!/usr/bin/env python3
"""
Genera grafici di vendite giornaliere e cumulative a partire da CSV Tickera.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import re

READ_KWARGS = {
    "sep": ",",
    "quotechar": '"',
    "encoding": "utf-8",
    "engine": "python",
    "dtype": str,
    "skip_blank_lines": True,
}


EVENT_END_DATES: Dict[str, pd.Timestamp] = {
    "Lista_ticket_7chakras_2019_onlyGood_VERIFIED_FLAT": pd.Timestamp("2019-08-28"),
    "Lista_ticket_7chakras_2025_onlyGood_VERIFIED_FLAT": pd.Timestamp("2025-06-23"),
    "Attendee_List_Paid_19Gen_16.18pm_FLAT": pd.Timestamp("2026-07-13"),
}

REFERENCE_EVENT_DATE = max(EVENT_END_DATES.values())


def format_month_label(days: int) -> str:
    if days == 0:
        return "Event"
    months = days / 30
    if abs(months - round(months)) < 1e-9:
        months_int = int(round(months))
        return f"{months_int} month" if months_int == 1 else f"{months_int} months"
    if abs(months - 0.5) < 1e-9:
        return "1/2mont"
    return f"{months:.1f}mont"


def build_event_ticks(max_days: int) -> Tuple[np.ndarray, List[str]]:
    step = 15
    ticks = np.arange(0, max_days + step, step)
    labels = [format_month_label(int(day)) for day in ticks]
    return ticks, labels


LEGEND_FONTSIZE = 12


def resolve_event_target(stem: str) -> Tuple[Optional[str], Optional[pd.Timestamp]]:
    exact = EVENT_END_DATES.get(stem)
    if exact is not None:
        return stem, exact
    lower = stem.lower()
    for key in EVENT_END_DATES:
        if key.lower() == lower:
            return key, EVENT_END_DATES[key]
    for key in EVENT_END_DATES:
        key_lower = key.lower()
        if key_lower in lower or lower in key_lower:
            return key, EVENT_END_DATES[key]
    print(
        "[WARN] Nessuna EVENT_END_DATE trovata per "
        f"'{stem}'. Disponibili: {list(EVENT_END_DATES.keys())}. "
        "Aggiungi la chiave corretta se manca."
    )
    lower = stem.lower()
    if "2019" in lower:
        return "__year_2019__", EVENT_END_DATES["Lista_ticket_7chakras_2019_onlyGood_VERIFIED_FLAT"]
    if "2025" in lower:
        return "__year_2025__", EVENT_END_DATES["Lista_ticket_7chakras_2025_onlyGood_VERIFIED_FLAT"]
    if "2026" in lower:
        return "__year_2026__", EVENT_END_DATES["Attendee_List_Paid_19Gen_16.18pm_FLAT"]
    return None, None


def infer_year_from_stem(stem: str) -> str:
    match = re.search(r"(20\d{2})", stem)
    return match.group(1) if match else "unknown"


def legend_label_from_stem(stem: str) -> str:
    year = infer_year_from_stem(stem)
    return f"7 Chakras Festival {year}"


def export_ticket_rows(df: pd.DataFrame, payment_col: str, year: str, output_dir: Path) -> None:
    date_strings = []
    for value in df[payment_col]:
        parsed = parse_payment_date(value)
        date_strings.append(parsed.strftime("%Y-%m-%d") if not pd.isna(parsed) else "")
    entries = pd.DataFrame(
        {
            "Data": date_strings,
            "Biglietti": [1] * len(date_strings),
        }
    )
    path = output_dir / f"Festival_{year}_vendite_giornaliere.csv"
    entries.to_csv(path, index=False, encoding="utf-8")
    print(f"[OK] CSV giornaliero salvato in {path.name}")


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


def build_monthly_summary(daily: pd.Series) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame(columns=["month_start", "month_label", "tickets_month", "avg_tickets_per_day"])
    monthly_total = daily.resample("MS").sum()
    monthly_days = daily.resample("MS").size()
    monthly_avg = monthly_total / monthly_days
    summary = pd.DataFrame(
        {
            "month_start": monthly_total.index,
            "month_label": monthly_total.index.strftime("%Y-%m"),
            "tickets_month": monthly_total.values.astype(int),
            "avg_tickets_per_day": monthly_avg.values,
        }
    )
    return summary


def plot_daily_sales(daily: pd.Series, label: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    legend_name = legend_label_from_stem(label)
    ax.plot(daily.index, daily.values, marker="o", color="#1e88e5", label=legend_name)
    ax.set_title(f"Daily Sales - {legend_name}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Tickets")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=LEGEND_FONTSIZE)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_cumulative_sales(daily: pd.Series, label: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    cumulative = daily.cumsum()
    legend_name = legend_label_from_stem(label)
    ax.plot(cumulative.index, cumulative.values, marker="o", color="#d81b60", label=legend_name)
    ax.set_title(f"Cumulative Sales - {legend_name}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Tickets")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=LEGEND_FONTSIZE)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_monthly_sales_histogram(monthly: pd.DataFrame, label: str, out_path: Path) -> None:
    if monthly.empty:
        return
    fig, ax = plt.subplots(figsize=(11, 5))
    legend_name = legend_label_from_stem(label)
    x = np.arange(len(monthly))
    bars = ax.bar(
        x,
        monthly["tickets_month"].to_numpy(),
        color="#1565c0",
        alpha=0.9,
        width=0.75,
        label="Tickets/month",
    )
    ax.plot(
        x,
        monthly["avg_tickets_per_day"].to_numpy(),
        color="#e65100",
        marker="o",
        linewidth=1.8,
        label="Avg tickets/day in month",
    )
    for idx, bar in enumerate(bars):
        value = int(bar.get_height())
        avg = float(monthly.iloc[idx]["avg_tickets_per_day"])
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            f"{value}\navg {avg:.2f}/d",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_title(f"Monthly sales histogram - {legend_name}")
    ax.set_xlabel("Month")
    ax.set_ylabel("Tickets")
    ax.set_xticks(x)
    ax.set_xticklabels(monthly["month_label"].tolist(), rotation=45, ha="right")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=LEGEND_FONTSIZE - 1)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_daily_histogram_for_single_month(
    month_daily: pd.Series,
    label: str,
    month_label: str,
    out_path: Path,
) -> None:
    if month_daily.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    days = month_daily.index.day.to_numpy()
    values = month_daily.values.astype(int)
    avg = float(np.mean(values))
    ax.bar(days, values, color="#2e7d32", alpha=0.9, width=0.9)
    ax.axhline(avg, color="#e65100", linestyle="--", linewidth=1.5, label=f"Avg/day: {avg:.2f}")
    legend_name = legend_label_from_stem(label)
    ax.set_title(f"Daily sales histogram - {legend_name} - {month_label}")
    ax.set_xlabel("Day of month")
    ax.set_ylabel("Tickets")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=LEGEND_FONTSIZE - 1)
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


def reorder_wrap_series(
    series: pd.Series,
    first_date: pd.Timestamp,
    last_date: pd.Timestamp,
) -> pd.Series:
    if first_date <= last_date:
        return series.loc[first_date:last_date]
    part1 = series.loc[first_date:]
    part2 = series.loc[:last_date]
    return pd.concat([part1, part2])



def plot_daily_aligned_event(
    series_map: Dict[str, Tuple[pd.Series, pd.Timestamp]],
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    min_x = float("inf")
    max_x = 0
    for label, payload in series_map.items():
        series, target_end = payload
        if series.empty or target_end is None:
            continue
        valid = series[series.index <= target_end]
        if valid.empty:
            continue
        delta_days = (target_end.normalize() - valid.index.normalize()).days
        x_vals = delta_days
        min_x = min(min_x, int(x_vals.min()))
        max_x = max(max_x, int(x_vals.max()))
        ax.plot(
            x_vals,
            valid.values,
            marker="o",
            markersize=3,
            linewidth=1.3,
            label=f"{legend_label_from_stem(label)} ({target_end.strftime('%d/%m/%Y')})",
        )
    ax.set_title("Daily sales - aligned to the event")
    ax.set_xlabel("Months before the 7 Chakras")
    ax.set_ylabel("Daily Tickets")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=LEGEND_FONTSIZE)
    if min_x == float("inf"):
        min_x = 0
    ticks, labels = build_event_ticks(max_x)
    ax.set_xlim(max_x, min_x)
    ax.invert_xaxis()
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.invert_xaxis()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_daily_event_histogram(
    series_map: Dict[str, Tuple[pd.Series, pd.Timestamp]],
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    cmap = plt.get_cmap("tab10")
    min_x = float("inf")
    max_x = 0
    for idx, (label, payload) in enumerate(series_map.items()):
        series, target_end = payload
        if series.empty or target_end is None:
            continue
        valid = series[series.index <= target_end]
        if valid.empty:
            continue
        delta_days = (target_end.normalize() - valid.index.normalize()).days
        x_vals = delta_days
        min_x = min(min_x, int(x_vals.min()))
        max_x = max(max_x, int(x_vals.max()))
        base_color = cmap(idx % cmap.N)
        alpha = 0.25 + (idx * 0.1)
        ax.bar(
            x_vals,
            valid.values,
            width=1,
            color=base_color,
            alpha=min(alpha, 0.5),
            label=f"{legend_label_from_stem(label)} ({target_end.strftime('%d/%m/%Y')})",
            align="edge",
        )
    ax.set_title("Daily sales histogram - aligned to the event")
    ax.set_xlabel("Months before the 7 Chakras")
    ax.set_ylabel("Daily Tickets")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=LEGEND_FONTSIZE)
    if min_x == float("inf"):
        min_x = 0
    ticks, labels = build_event_ticks(max_x)
    ax.set_xlim(max_x, min_x)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.invert_xaxis()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_daily_event_histogram_labeled(
    series_map: Dict[str, Tuple[pd.Series, pd.Timestamp]],
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    cmap = plt.get_cmap("tab10")
    min_x = float("inf")
    max_x = 0
    for idx, (label, payload) in enumerate(series_map.items()):
        series, target_end = payload
        if series.empty or target_end is None:
            continue
        valid = series[series.index <= target_end]
        if valid.empty:
            continue
        delta_days = (target_end.normalize() - valid.index.normalize()).days
        x_vals = delta_days
        min_x = min(min_x, int(x_vals.min()))
        max_x = max(max_x, int(x_vals.max()))
        base_color = cmap(idx % cmap.N)
        alpha = 0.25 + (idx * 0.1)
        bars = ax.bar(
            x_vals,
            valid.values,
            width=1,
            color=base_color,
            alpha=min(alpha, 0.5),
            label=f"{legend_label_from_stem(label)} ({target_end.strftime('%d/%m/%Y')})",
            align="edge",
        )
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 0.5,
                    str(int(height)),
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    alpha=0.8,
                )
    ax.set_title("Daily sales histogram (labeled) - aligned to the event")
    ax.set_xlabel("Months before the 7 Chakras")
    ax.set_ylabel("Daily Tickets")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=LEGEND_FONTSIZE)
    if min_x == float("inf"):
        min_x = 0
    ticks, labels = build_event_ticks(max_x)
    ax.set_xlim(max_x, min_x)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_cumulative_aligned_event(
    series_map: Dict[str, Tuple[pd.Series, pd.Timestamp]],
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    min_x = float("inf")
    max_x = 0
    for label, payload in series_map.items():
        series, target_end = payload
        if series.empty or target_end is None:
            continue
        valid = series[series.index <= target_end]
        if valid.empty:
            continue
        delta_days = (target_end.normalize() - valid.index.normalize()).days
        x_vals = delta_days
        min_x = min(min_x, int(x_vals.min()))
        max_x = max(max_x, int(x_vals.max()))
        ax.plot(
            x_vals,
            valid.values,
            marker="o",
            markersize=3,
            linewidth=1.3,
            label=f"{legend_label_from_stem(label)} ({target_end.strftime('%d/%m/%Y')})",
        )
    ax.set_title("Cumulative sales - aligned to the event")
    ax.set_xlabel("Months before the 7 Chakras")
    ax.set_ylabel("Cumulative Tickets")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=LEGEND_FONTSIZE)
    if min_x == float("inf"):
        min_x = 0
    ticks, labels = build_event_ticks(max_x)
    ax.set_xlim(min_x, max_x)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=45, ha="right")
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
) -> Optional[Tuple[pd.Series, pd.Timestamp, pd.Timestamp, Optional[Tuple[pd.Series, pd.Timestamp]]]]:
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

    year = infer_year_from_stem(path.stem)
    export_ticket_rows(df, resolved_col, year, output_dir)

    label = path.stem
    slug = slugify(label)
    daily_path = output_dir / f"{slug}_vendite_giornaliere.{fmt}"
    cumulative_path = output_dir / f"{slug}_vendite_cumulative.{fmt}"
    monthly_hist_path = output_dir / f"{slug}_vendite_mensili_hist.{fmt}"
    monthly_csv_path = output_dir / f"{slug}_vendite_mensili.csv"
    monthly_daily_dir = output_dir / f"{slug}_vendite_giornaliere_per_mese"
    monthly_daily_dir.mkdir(parents=True, exist_ok=True)

    plot_daily_sales(daily, label, daily_path)
    plot_cumulative_sales(daily, label, cumulative_path)
    monthly_summary = build_monthly_summary(daily)
    plot_monthly_sales_histogram(monthly_summary, label, monthly_hist_path)
    monthly_saved = []
    month_index = daily.index.to_period("M")
    for month in sorted(month_index.unique()):
        mask = month_index == month
        month_daily = daily.loc[mask]
        if month_daily.empty:
            continue
        month_label = str(month)
        month_out = monthly_daily_dir / f"{slug}_vendite_giornaliere_{month_label}.{fmt}"
        plot_daily_histogram_for_single_month(month_daily, label, month_label, month_out)
        monthly_saved.append(month_out.name)
    if not monthly_summary.empty:
        monthly_summary.to_csv(monthly_csv_path, index=False, encoding="utf-8")

    print(
        f"[OK] Salvati: {daily_path.name}, {cumulative_path.name}, "
        f"{monthly_hist_path.name}, {monthly_csv_path.name}, "
        f"{len(monthly_saved)} istogrammi giornalieri mensili in {monthly_daily_dir.name}"
    )
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
    match_key, target_end = resolve_event_target(path.stem)
    print(f"[DEBUG] stem={path.stem} event_target={target_end}")
    event_payload = (daily, target_end) if target_end is not None else None
    return normalized, first_date, last_date, event_payload


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
    event_series: Dict[str, Tuple[pd.Series, pd.Timestamp]] = {}
    for input_path in args.inputs:
        path = Path(input_path)
        print(f"[DEBUG] input={path.name} stem={path.stem}")
        result = process_file(
            path,
            output_dir,
            args.payment_date_col,
            args.format,
            args.season_start_month,
            args.season_months,
        )
        if result is not None:
            _, _, _, event_payload = result
            if event_payload is not None and not event_payload[0].empty:
                event_series[path.stem] = event_payload

    if event_series:
        print(f"[DEBUG] event_series keys = {list(event_series.keys())}")
        event_cum_series: Dict[str, Tuple[pd.Series, pd.Timestamp]] = {}
        combined_daily_event = output_dir / f"vendite_giornaliere_comparativa_event.{args.format}"
        plot_daily_aligned_event(event_series, combined_daily_event)
        saved_names = [combined_daily_event.name]
        for label, payload in event_series.items():
            cum_series = payload[0].sort_index().cumsum()
            event_cum_series[label] = (cum_series, payload[1])
        combined_cum_event = output_dir / f"vendite_cumulative_comparativa_event.{args.format}"
        plot_cumulative_aligned_event(event_cum_series, combined_cum_event)
        saved_names.append(combined_cum_event.name)
        combined_daily_event_hist = (
            output_dir / f"vendite_giornaliere_comparativa_event_hist.{args.format}"
        )
        plot_daily_event_histogram(event_series, combined_daily_event_hist)
        saved_names.append(combined_daily_event_hist.name)
        combined_daily_event_hist_label = (
            output_dir / f"vendite_giornaliere_comparativa_event_hist_label.{args.format}"
        )
        plot_daily_event_histogram_labeled(event_series, combined_daily_event_hist_label)
        saved_names.append(combined_daily_event_hist_label.name)
        print(f"[OK] Salvati: {', '.join(saved_names)}")


if __name__ == "__main__":
    main()
