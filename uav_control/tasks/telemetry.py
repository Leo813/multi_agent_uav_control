"""Optional legacy CSV output, independent of the simulation and learning loop."""

import csv
from itertools import zip_longest
from pathlib import Path
import re
import warnings


def write_legacy_csv(directory, filename, columns):
    """Keep legacy columns; pad unequal observation/action sample counts with blanks."""
    if directory is None:
        return
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", filename)
    try:
        destination = Path(directory).expanduser()
        destination.mkdir(parents=True, exist_ok=True)
        with (destination / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(columns)
            writer.writerows(zip_longest(*columns.values(), fillvalue=""))
    except OSError as error:
        warnings.warn(f"Could not write optional telemetry {name}: {error}", RuntimeWarning)
