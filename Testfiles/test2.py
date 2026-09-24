import sys
from importlib import metadata
from rich.panel import Panel
from rich import print as rprint

# Henter versjonen til den 'rich'-pakken som faktisk er lastet inn av pip
try:
    rich_versjon = metadata.version("rich")
except metadata.PackageNotFoundError:
    rich_versjon = "Ukjent (ikke installert)"

# Meldingstekst med riktig versjonsnummer
melding = f"Dette rich-biblioteket (v{rich_versjon}) er hentet fra nettet! Ikke lockfile."

# Printer ut en pen boks
rprint(Panel(melding, title="PIP Installasjonstest", expand=False))

