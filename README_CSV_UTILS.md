# README - Utility CSV

Questo documento descrive gli script in `Code/` per filtrare e "flattenare" CSV.

## 1) tickets_paid_filter.py

Percorso: `Code/tickets_paid_filter.py`

Scopo:
- legge un CSV ticket
- filtra solo righe con `Order Status = Paid`
- salva due output nello **stesso** path del file originale:
  - `Attendee_List_Paid_VERIFIED__<data>.csv`
  - `Attendee_List_Paid_VERIFIED_FLAT__<data>.csv` (newline nei campi sostituiti da spazi)

Esempio output:
- `Documenti/Tickets/Attendee_List_Paid_VERIFIED__19-01-26_12-15.csv`
- `Documenti/Tickets/Attendee_List_Paid_VERIFIED_FLAT__19-01-26_12-15.csv`

Launch VS Code:
- `Ticket Paid Verified`

## 2) csv_flatten.py

Percorso: `Code/csv_flatten.py`

Scopo:
- prende un CSV qualsiasi e crea una versione FLAT
- i newline dentro i campi vengono sostituiti con spazi
- output nello stesso path: `<nome>_FLAT.csv`

Esempio output:
- `Documenti/Tickets/Attendee_List_Paid_19-01-26_12-15_FLAT.csv`

Launch VS Code:
- `CSV Flatten (generic)`
