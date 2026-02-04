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

Grafici comparativi (tutti i CSV sovrapposti, allineati per mese):

- `vendite_giornaliere_comparativa.png`
- `vendite_cumulative_comparativa.png`
- `vendite_cumulative_comparativa_norm.png` (normalizzato al 100% finale)
- `vendite_cumulative_comparativa_allineata.png` (ultima entry allineata)
- `vendite_giornaliere_comparativa_allineata.png` (giornaliere allineate all'ultima entry)
- `vendite_giornaliere_comparativa_event.png` (vendite giornaliere allineate al giorno del festival per ciascun anno)

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

## Grafico giornaliero allineato all’evento

Basta eseguire lo script: il file `vendite_giornaliere_comparativa_event.png` mostra le vendite giornaliere per ciascun CSV dopo aver spostato la loro ultima entry sul giorno dell’evento (2019→28/08, 2025→23/06, 2026→14/07). Questo permette di confrontare gli step di vendita con il “countdown” fino al festival, anche se le date reali cadono in mesi diversi.

## Launch VS Code

Configurazione disponibile:

- `Ticket Sales Distributions`
