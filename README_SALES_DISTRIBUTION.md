# README - Distribuzione Vendite

Questo documento descrive lo script `Code/ticket_sales_distribution_plots.py`,
che genera grafici di vendite giornaliere e cumulative dai CSV Tickera.

## Scopo

- Legge uno o piu CSV con colonna `Payment Date`
- Calcola le vendite per giorno
- Esporta 2 grafici per ogni CSV: vendite giornaliere e vendite cumulative
- Genera 2 grafici comparativi con tutte le distribuzioni allineate su Agosto->Agosto

## Input di default

I file di input sono impostati di default (modificabili con `--inputs`):

- `Documenti/Tickets/Lista_ticket_7chakras_2019_onlyGood_VERIFIED_FLAT.csv`
- `Documenti/Tickets/Lista_ticket_7chakras_2025_onlyGood_VERIFIED_FLAT.csv`
- `Documenti/Tickets/Attendee_List_Paid_19Gen_16.18pm_FLAT.csv`

## Output

Cartella di output (default):

- `output/vendite_distribuzione`

Per ogni CSV vengono generati:

- `<nome_csv>_vendite_giornaliere.png`
- `<nome_csv>_vendite_cumulative.png`
- `<nome_csv>_vendite_mensili_hist.png` (istogramma: totale ticket per mese + media giornaliera del mese)
- `<nome_csv>_vendite_mensili.csv` (riepilogo mensile con `tickets_month` e `avg_tickets_per_day`)
- cartella `<nome_csv>_vendite_giornaliere_per_mese/` con un grafico per mese (`giorno del mese` -> `ticket venduti`)
- `Festival_{anno}_vendite_giornaliere.csv` (giorno e biglietti per ricostruire il grafico)

Grafici comparativi (tutti i CSV sovrapposti, allineati per mese):

- `vendite_giornaliere_comparativa.png`
- `vendite_cumulative_comparativa.png`
- `vendite_cumulative_comparativa_norm.png` (normalizzato al 100% finale)
- `vendite_cumulative_comparativa_allineata.png` (ultima entry allineata)
- `vendite_giornaliere_comparativa_allineata.png` (giornaliere allineate all'ultima entry)
- `vendite_giornaliere_comparativa_event.png` (vendite giornaliere allineate al giorno del festival per ciascun anno)
- `vendite_cumulative_comparativa_event.png` (vendite cumulative allineate all’evento)
- `vendite_giornaliere_comparativa_event_hist.png` (istogramma giornaliero allineato all’evento)
- `vendite_giornaliere_comparativa_event_hist_label.png` (istogramma con valori per barra)

## Esecuzione

Da terminale:

```bash
python Code/ticket_sales_distribution_plots.py
```

Con input personalizzati:

```bash
python Code/ticket_sales_distribution_plots.py --inputs Documenti/Tickets/file1.csv Documenti/Tickets/file2.csv
```

Per cambiare il mese di inizio stagione (default: Luglio):

```bash
python Code/ticket_sales_distribution_plots.py --season-start-month 7
```

Per allargare la finestra stagionale (default: 13 mesi):

```bash
python Code/ticket_sales_distribution_plots.py --season-months 15
```

Per analisi periodicita picchi (smooth 3 giorni):

```bash
python Code/ticket_sales_distribution_plots.py --peak-analysis
```

Per eseguire piu quantili in una sola run:

```bash
python Code/ticket_sales_distribution_plots.py --peak-analysis --peak-quantiles 0.8 0.9
```

## Grafico giornaliero allineato all’evento

Basta eseguire lo script: il file `vendite_giornaliere_comparativa_event.png` mostra le vendite giornaliere per ciascun CSV dopo aver spostato la loro ultima entry sul giorno dell’evento (2019→28/08, 2025→23/06, 2026→14/07). Questo permette di confrontare gli step di vendita con il “countdown” fino al festival, anche se le date reali cadono in mesi diversi.

## Launch VS Code

Configurazione disponibile:

- `Ticket Sales Distributions`
