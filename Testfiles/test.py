
from rich.console import Console
from rich.panel import Panel
import importlib.metadata  # Built-in tool to read the exact locked version

console = Console()

console.print("\n--- [bold cyan]Lockfile Library Verification[/bold cyan] ---")

console.print(Panel(
    "[bold green]Lockfile test successful![/bold green] Rich is rendering terminal styles.",
    title="Success", 
    expand=False
))
installed_version = importlib.metadata.version("rich")

print(f"Active Rich Version in Environment: {installed_version}") #<----- The newest version is 14.1.0 and not 13.0.0

