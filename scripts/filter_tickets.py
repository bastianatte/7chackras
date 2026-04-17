#!/usr/bin/env python3
"""Filter ticket CSV exports by keyword and emit a trimmed subset."""

import argparse
import csv
import os
import re
import sys
from difflib import SequenceMatcher

REQUIRED_COLUMNS = [
    "First Name",
    "Last Name",
    "Name",
    "Attendee E-mail",
    "Payment Date",
]
TICKET_COLUMN_PREFERENCES = ["Ticket Type", "Ticket ID"]
NORMALIZE_REGEX = re.compile(r"[\s\-_]+")
FUZZY_THRESHOLD = 0.75


def die(message: str, code: int = 1) -> None:
    """Print a clean error and exit with the given code."""
    print(f"Error: {message}", file=sys.stderr)
    sys.exit(code)


def normalize_identifier(value: str) -> str:
    """Normalize column names and values to compare while ignoring spaces and separators."""
    if value is None:
        return ""
    return NORMALIZE_REGEX.sub("", value).lower()


def find_ticket_column(fieldnames: list[str]) -> str:
    """Locate the column containing the ticket descriptor."""
    normalized = {normalize_identifier(name): name for name in fieldnames}

    for preferred in TICKET_COLUMN_PREFERENCES:
        match = normalized.get(normalize_identifier(preferred))
        if match:
            return match

    best_match = None
    best_score = 0.0
    best_target = None
    for name in fieldnames:
        norm_name = normalize_identifier(name)
        for target in TICKET_COLUMN_PREFERENCES:
            score = SequenceMatcher(
                None, norm_name, normalize_identifier(target)
            ).ratio()
            if score > best_score:
                best_score = score
                best_match = name
                best_target = target

    if best_score >= FUZZY_THRESHOLD and best_match:
        print(
            f"Using column '{best_match}' as best match for '{best_target}'.",
            file=sys.stderr,
        )
        return best_match

    die(
        "Impossibile individuare una colonna 'Ticket Type' o 'Ticket ID'. "
        f"Colonne disponibili: {', '.join(fieldnames)}"
    )


def resolve_required_columns(fieldnames: list[str]) -> dict[str, str]:
    """Map the canonical column names to the actual headers present in the CSV."""
    normalized = {normalize_identifier(name): name for name in fieldnames}
    resolved: dict[str, str] = {}

    for canonical in REQUIRED_COLUMNS:
        matched = normalized.get(normalize_identifier(canonical))
        if not matched:
            die(
                f"Colonna obbligatoria '{canonical}' assente. "
                f"Colonne disponibili: {', '.join(fieldnames)}"
            )
        resolved[canonical] = matched

    return resolved


def build_output_path(input_path: str, custom_output: str | None) -> str:
    """Return the destination path either from --output or deriving from the input file."""
    if custom_output:
        return custom_output

    directory = os.path.dirname(input_path)
    directory = directory or "."
    base_name, extension = os.path.splitext(os.path.basename(input_path))
    return os.path.join(directory, f"{base_name}_filtered{extension}")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Filtra un export ticket lasciando solo i biglietti che contengono una keyword"
    )
    parser.add_argument("--input", required=True, help="Percorso del CSV in input")
    parser.add_argument(
        "--ticket_keyword",
        default="early bird",
        help="Keyword da cercare nella colonna ticket",
    )
    parser.add_argument("--output", help="Percorso del CSV in output (facoltativo)")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    if not os.path.isfile(args.input):
        die(f"File di input non trovato: {args.input}")

    keyword = args.ticket_keyword.strip()
    if not keyword:
        die("La keyword non può essere vuota.")

    # Match the keyword literally (including spaces), ignoring letter case only.
    keyword_pattern = re.compile(re.escape(keyword), re.IGNORECASE)

    output_path = build_output_path(args.input, args.output)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(args.input, newline="", encoding="utf-8") as input_file:
        reader = csv.DictReader(input_file)
        if not reader.fieldnames:
            die("CSV di input privo di intestazione.")

        ticket_column = find_ticket_column(reader.fieldnames)
        resolved_columns = resolve_required_columns(reader.fieldnames)

        filtered_count = 0
        with open(output_path, "w", newline="", encoding="utf-8") as output_file:
            writer = csv.writer(output_file)
            writer.writerow(REQUIRED_COLUMNS)

            for row in reader:
                value = row.get(ticket_column, "")
                if value is None:
                    value = ""
                if keyword_pattern.search(value):
                    filtered_count += 1
                    writer.writerow(
                        [row.get(resolved_columns[col], "") for col in REQUIRED_COLUMNS]
                    )

    print(
        f"Filtrate {filtered_count} righe con keyword '{args.ticket_keyword}'. "
        f"Output scritto su {output_path}"
    )


if __name__ == "__main__":
    main()

