# 7chakras Data Processing Hub

Benvenuto nel repository di elaborazione dati per il progetto **7chakras**. Questo hub centrale coordina l'analisi dei ticket, la gestione dei soci e la riconciliazione dei dati per l'evento 2026.

## 🚀 Flusso di Lavoro Principale

Il progetto si divide in tre aree principali:

1.  **Analisi Esplorativa (EDA):** Statistiche sulle vendite e report finanziari.
2.  **Matching Soci/Ticket:** Riconciliazione tra i partecipanti (Tickera) e il libro soci.
3.  **Utility CSV:** Strumenti per pulire, unire e normalizzare i dati.

---

## 📂 Struttura del Codice (`Code/`)

### Core Scripts

- **[`ticket_eda.py`](file:///c:/Users/spina/Documents/Other_Codes/7chackras/Code/ticket_eda.py):** Analizza i ticket "Paid", calcola entrate per tipo/paese e genera grafici.
- **[`members_ticket_match.py`](file:///c:/Users/spina/Documents/Other_Codes/7chackras/Code/members_ticket_match.py):** Esegue il matching tra liste soci e attendee, categorizzando i risultati (match sicuri, nomi simili, scarti).
- **[`ticket_phase_summary.py`](file:///c:/Users/spina/Documents/Other_Codes/7chackras/Code/ticket_phase_summary.py):** Genera un riepilogo delle vendite suddiviso per fasi (Early Bird, Phase 0, Phase 1, ecc.).

### Utility & Pre-processing

- **[`tickets_merge.py`](file:///c:/Users/spina/Documents/Other_Codes/7chackras/Code/tickets_merge.py):** Unisce diversi export di Tickera in un unico file master.
- **[`tickets_paid_filter.py`](file:///c:/Users/spina/Documents/Other_Codes/7chackras/Code/tickets_paid_filter.py):** Filtra solo le transazioni confermate e normalizza i campi.
- **[`csv_flatten.py`](file:///c:/Users/spina/Documents/Other_Codes/7chackras/Code/csv_flatten.py):** Rimuove newline dai campi CSV per evitare errori di parsing in altri software.

---

## 📖 Documentazione Dettagliata

Per approfondire ogni modulo, consulta i README specifici:

- 📄 **[Dettagli EDA](file:///c:/Users/spina/Documents/Other_Codes/7chackras/README_EDA.md):** Report generati, colonne calcolate e immagini prodotte.
- 📄 **[Dettagli Matching](file:///c:/Users/spina/Documents/Other_Codes/7chackras/README_MATCHING.md):** Logica di confronto nomi/email e spiegazione dei file di output.
- 📄 **[Manuale Utility CSV](file:///c:/Users/spina/Documents/Other_Codes/7chackras/README_CSV_UTILS.md):** Guida all'uso degli script di supporto.

---

## 📓 Notebook di Analisi (`Code/nb/`)

Nella cartella `nb/` sono presenti Jupyter Notebook per analisi interattive:

- `7chakras_eda.ipynb`: Prototipazione dell'analisi esplorativa.
- `estrai_email_ticket.ipynb`: Tool rapido per l'estrazione bulk di contatti.

---

## 🛠️ Come Iniziare

Assicurati di avere `pandas` installato:

```bash
pip install pandas
```

Per eseguire l'analisi EDA standard:

```bash
python Code/ticket_eda.py --input Documenti/Tickets/tuo_file.csv
```

---

_Configurato con amore da Antigravity 🌀_
