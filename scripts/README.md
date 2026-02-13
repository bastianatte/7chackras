# Filter tickets

Script per filtrare un export ticket in modo da conservare solo le righe in cui la colonna **Ticket Type** (o **Ticket ID**) contiene una keyword specifica.

## Perché usarlo
- Normalizza i nomi delle colonne (ignorando spazi, trattini e underscore) e individua automaticamente la colonna ticket corretta.
- Normalizza anche la keyword in input, quindi `early bird`, `EarlyBird`, `early-bird` e simili vengono trattati allo stesso modo.
- L’output contiene solo le colonne essenziali: `First Name`, `Last Name`, `Name`, `Attendee E-mail`, `Payment Date`.
- Se non passi `--output`, il file viene salvato accanto all’input con suffisso `_filtered.csv`.

## Esempio

```bash
python scripts/filter_tickets.py \
  --input "Documenti/Tickets/UntilPhase1_2026_definitivo_FLAT_EarlyBirdOnly.csv" \
  --ticket_keyword "early bird"
```

## Argomenti
- `--input` **(obbligatorio)**: percorso del CSV da filtrare.
- `--ticket_keyword`: keyword da cercare nella colonna ticket (default `early bird`).
- `--output`: percorso esplicito del CSV di destinazione; se omesso viene costruito un nome basato su quello dell’input con suffisso `_filtered`.

## Comportamento
1. Verifica che il CSV abbia intestazioni; altrimenti termina con errore.
2. Cerca la colonna ticket (prima `Ticket Type`, poi `Ticket ID`; altrimenti applica un matching fuzzy e segnala l’errore se fallisce).
3. Normalizza la keyword e la cerca in modo case-insensitive e tolerante a spazi/trattini/underscore.
4. Verifica che le cinque colonne richieste siano presenti (anche con varianti di nome); se manca una, esce spiegando quali intestazioni erano disponibili.
5. Produce un file CSV con solo le righe filtrate e le colonne ordinate come richiesto.

## Debug e VS Code
Usa la configurazione `Filter tickets by type (CSV → subset)` in `.vscode/launch.json` per lanciare rapidamente lo script con il CSV di default e la keyword `early bird`.
