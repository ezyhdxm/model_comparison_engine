"""Module entry point: python -m model_comparison_engine."""
# CLI LOGIC: Importing the package is inert; module execution runs the explicit command.
from .cli import main

if __name__ == '__main__':
    main()
