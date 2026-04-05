"""Allow running the CLI as ``python -m cli``."""

import sys

from biasbuster.cli.main import main

sys.exit(main())
