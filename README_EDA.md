# README - EDA Ticket 2026

Questo file descrive i report generati da `Code/ticket_eda.py`.
Tutte le statistiche sono calcolate **solo** su righe con `Order Status = Paid`.

## Report principali (CSV)

- `tickets_clean.csv`
  Dataset filtrato su Paid, con colonne normalizzate e campi numerici convertiti.

- `by_type.csv`
  Biglietti per `Ticket Type` con conteggi e totale in euro.
  La colonna `avg_price` non e' piu' presente; viene aggiunta la riga `TOTAL`.
  Immagine: `plots/table_by_type.png`.

- `by_country*.csv`
  Conteggi per paese (usa "Country of residence / Paese di residenza").
  Immagine: `plots/table_by_country*.png`.

- `by_city*.csv`
  Conteggi per citta di residenza (usa "City of residence / Citta di residenza").
  Al momento **non** viene calcolata la regione.
  Immagine: `plots/table_by_city*.png`.

- `by_payment_gateway.csv`
  Conteggi per `Payment Gateway` con totale in euro.
  Immagine: `plots/table_by_payment_gateway.png`.

- `phase_revenue_from_ticket_type.csv`
  Somma in euro per fase (phase 0, phase 1, etc) estratta dal testo di `Ticket Type`.
  Include anche il totale generale e quante righe non hanno importo nel testo.
  Immagine: `plots/table_phase_revenue_from_ticket_type.png`.

- `ambassador_sales.csv`
  Vendite per ambassador, riconosciuti quando `Ticket Type` o `Ticket ID`
  contengono la stringa "Ambassador NOME".
  Immagine: `plots/table_ambassador_sales.png`.

- `column_missing_report.txt`
  Report dei valori mancanti (dataset completo + solo Paid).

## Note

- La colonna `Ticket Type` contiene prezzi tipo "PHASE 1: 160€".
  Lo script estrae quell'importo per i report "per fase".
- Se il prezzo non e' presente nel testo di `Ticket Type`,
  la riga viene conteggiata come "amount_missing".
