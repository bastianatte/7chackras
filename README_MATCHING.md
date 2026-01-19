# README - Report Match Soci/Ticket 2026

Questo documento descrive tutti i file di output generati dallo script
`Code/members_ticket_match.py` e il significato di ogni categoria.

## Match principali

- `match2026_{suffix}.csv`
  Persone con nome+cognome (normalizzati) uguali e **Attendee E-mail** uguale.
  Questo e' il match "sicuro".

- `match2026_plus{suffix}.csv`
  Sottoinsieme dei match sicuri che hanno anche **altri campi** coincidenti
  (data nascita, luogo nascita, residenza, codice fiscale).

- `match2026_email_diversa{suffix}.csv`
  Ticket Paid con nome+cognome uguali ma **Attendee E-mail diversa** rispetto al libro soci.
  L'email e' presente in entrambe le liste ma non coincide; serve per revisione manuale.

- `match2026_scarti{suffix}.csv`
  Tutti i candidati con nome+cognome uguali che **non passano** il controllo email.
  Include sia:
  - email diversa (quindi anche quelli presenti in `match2026_email_diversa{suffix}.csv`)
  - email mancante da una delle due parti
  Non sono match.

- `match2026_nome_simile{suffix}.csv`
  Persone con **email uguale** ma nome/cognome non identici; sono casi simili:
  - nome e cognome invertiti
  - nome parziale (es. primo nome vs primo+secondo)
  - nome+cognome inseriti nello stesso campo
  Questo report serve per intercettare errori di compilazione.

## Altri report di controllo

- `duplicati2026{suffix}.csv`
  Biglietti **Paid** con **nome+cognome duplicati** nel file ticket.
  Serve per capire se la stessa persona compare piu' volte nella lista ticket.

- `noSoci2025{suffix}.csv`
  Ticket **Paid** che **non hanno nessun socio corrispondente per nome+cognome**.
  Questi sono presenti nei ticket ma non nel libro soci.

- `soci_senza_match2026{suffix}.csv`
  Soci del libro 2025 **senza nessun ticket corrispondente per nome+cognome**.
  Questi sono presenti nel libro soci ma non nei ticket.

## Note importanti

- Il match usa sempre l'email **Attendee E-mail** (non la Buyer E-Mail).
- La colonna `buyer_email_match` e' un controllo secondario, non decide il match.
- Lo script salva un controllo di coerenza: se ci sono errori, crea
  `verifica_inversa_errori{suffix}.csv`.
