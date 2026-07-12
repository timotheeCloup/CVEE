"""add jobs_silver.title_tsv generated column

Revision ID: 3a9f1c7b2d84
Revises: 8cef33abbe6d
Create Date: 2026-07-12 18:20:00.000000

"""

from collections.abc import Sequence

from alembic import op

revision: str = "3a9f1c7b2d84"
down_revision: str | Sequence[str] | None = "8cef33abbe6d"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Precompute the French tsvector of the job title as a STORED generated
    # column so the hybrid search no longer recomputes to_tsvector('french',
    # intitule) for the whole corpus on every request (title_rank signal).
    # Expression matches the previous runtime one exactly (result-preserving).
    op.execute(
        """
        ALTER TABLE jobs_silver
        ADD COLUMN IF NOT EXISTS title_tsv tsvector
        GENERATED ALWAYS AS (to_tsvector('french', COALESCE(intitule, ''))) STORED;
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_silver_title_tsv ON jobs_silver USING gin (title_tsv);"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_silver_title_tsv;")
    op.execute("ALTER TABLE jobs_silver DROP COLUMN IF EXISTS title_tsv;")
