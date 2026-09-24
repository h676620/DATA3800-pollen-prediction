import importlib.metadata
import tomllib  # Innebygd i Python 3.11+
from rich import print

# 1. Hent faktisk installert versjon
lokal_v = importlib.metadata.version("rich")

# 2. Les automatisk fra poetry.lock for å finne sannheten
forventet_laast_versjon = None
try:
    with open("poetry.lock", "rb") as f:
        lock_data = tomllib.load(f)
        # Leter etter rich i lockfilen
        for package in lock_data.get("package", []):
            if package.get("name") == "rich":
                forventet_laast_versjon = package.get("version")
                break
except FileNotFoundError:
    print("[red]✗ Fant ikke poetry.lock i denne mappen![/red]")

# 3. Definer en tilfeldig nyere versjon (for illustrasjon)
nett_v = "14.1.0" 

if not forventet_laast_versjon:
    print("[red]✗ Fant ikke 'rich' i poetry.lock.[/red]")
else:
    print(f"\n[yellow]Faktisk låst versjon funnet i poetry.lock:[/] [bold]{forventet_laast_versjon}[/]")
    print(f"[blue]Faktisk installert i miljøet:[/]            [bold]{lokal_v}[/]")
    print(f"[cyan]Nyeste versjon på PyPI (eksempel):[/]       [bold]{nett_v}[/]\n")

    # Korrekt logisk sjekk
    if lokal_v == forventet_laast_versjon:
        print(f"[green]✓ Suksess! Miljøet bruker den låste versjonen ({lokal_v}) direkte fra lockfilen.[/green]")
    else:
        print(f"[red]✗ Feil: Miljøet bruker versjon {lokal_v}, men poetry.lock sier den burde være {forventet_laast_versjon}.[/red]")
