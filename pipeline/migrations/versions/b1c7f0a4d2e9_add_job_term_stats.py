"""add job_term_stats table and refresh function

Revision ID: b1c7f0a4d2e9
Revises: 3a9f1c7b2d84
Create Date: 2026-09-20 10:00:00.000000

"""

from collections.abc import Sequence

from alembic import op

revision: str = "b1c7f0a4d2e9"
down_revision: str | Sequence[str] | None = "3a9f1c7b2d84"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # unaccent is required to build CV lexemes with the same normalization as
    # the job fts_tokens (see fn_fill_fts_from_silver).
    op.execute("CREATE EXTENSION IF NOT EXISTS unaccent;")

    # Document frequency per lexeme over the job corpus. Used to weight CV
    # terms by rarity (IDF): rare skills like "python" outrank generic words
    # like "equipe" that appear in almost every offer.
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS job_term_stats (
            term TEXT PRIMARY KEY,
            df INTEGER NOT NULL
        );
        """
    )

    # Rebuild the stats from jobs_gold. Cheap on the current corpus size and
    # called once per ingest run, so the table never needs manual maintenance.
    # DELETE (not TRUNCATE) keeps the API reads non-blocking: TRUNCATE would take
    # an ACCESS EXCLUSIVE lock and stall the search while the stats rebuild.
    op.execute(
        """
        CREATE OR REPLACE FUNCTION refresh_job_term_stats() RETURNS void AS $$
        BEGIN
            DELETE FROM job_term_stats;
            INSERT INTO job_term_stats (term, df)
            SELECT word, ndoc
            FROM ts_stat('SELECT fts_tokens FROM jobs_gold WHERE fts_tokens IS NOT NULL');
        END;
        $$ LANGUAGE plpgsql;
        """
    )

    # Populate immediately so the API has data before the next ingest run.
    op.execute("SELECT refresh_job_term_stats();")


def downgrade() -> None:
    op.execute("DROP FUNCTION IF EXISTS refresh_job_term_stats();")
    op.execute("DROP TABLE IF EXISTS job_term_stats;")
