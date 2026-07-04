"""Database initialization via Alembic migrations.

Usage:
    python init_db.py              # dry-run (shows what would happen)
    python init_db.py --force      # run alembic upgrade head
"""

import argparse
import subprocess
import sys

parser = argparse.ArgumentParser(description="Initialize CVEE database schema via Alembic")
parser.add_argument(
    "--force",
    action="store_true",
    help="Run alembic upgrade head (applies all pending migrations)",
)
args = parser.parse_args()

if not args.force:
    print("This script will run all pending Alembic migrations.")
    print("Run with --force to proceed:")
    print(f"  python {__file__} --force")
    sys.exit(1)

try:
    subprocess.run(
        ["uv", "run", "alembic", "upgrade", "head"],
        check=True,
    )
    print("Database migration completed successfully.")
except subprocess.CalledProcessError as e:
    print(f"Migration failed: {e}")
    sys.exit(1)
