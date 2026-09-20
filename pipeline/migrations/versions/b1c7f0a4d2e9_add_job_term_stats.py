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

    # jobs_gold.fts_tokens and its fill trigger predate this revision in prod
    # but were never captured in any migration. Declared here (idempotently) so
    # a database built from migrations alone matches prod before
    # refresh_job_term_stats() reads the column below.
    op.execute(
        """
        ALTER TABLE jobs_gold
        ADD COLUMN IF NOT EXISTS fts_tokens tsvector;
        """
    )
    op.execute(
        """
        CREATE OR REPLACE FUNCTION fn_fill_fts_from_silver() RETURNS trigger AS $$
        DECLARE
            source_record RECORD;
            clean_competences TEXT;
            clean_qualites TEXT;
        BEGIN
            SELECT intitule, description, competences, qualitesprofessionnelles
            INTO source_record
            FROM jobs_silver
            WHERE job_id = NEW.job_id;

            SELECT string_agg(elem->>'libelle', ' ')
            INTO clean_competences
            FROM jsonb_array_elements(source_record.competences) AS elem;

            SELECT string_agg((elem->>'libelle') || ' ' || (elem->>'description'), ' ')
            INTO clean_qualites
            FROM jsonb_array_elements(source_record.qualitesprofessionnelles) AS elem;

            UPDATE jobs_gold
            SET fts_tokens =
                setweight(to_tsvector('french', unaccent(COALESCE(source_record.intitule, ''))), 'A') ||
                setweight(to_tsvector('french', unaccent(COALESCE(source_record.description, ''))), 'B') ||
                setweight(to_tsvector('french', unaccent(COALESCE(clean_competences, ''))), 'C') ||
                setweight(to_tsvector('french', unaccent(COALESCE(clean_qualites, ''))), 'C')
            WHERE job_id = NEW.job_id;

            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
        """
    )
    op.execute("DROP TRIGGER IF EXISTS tr_update_gold_fts ON jobs_gold;")
    op.execute(
        """
        CREATE TRIGGER tr_update_gold_fts AFTER INSERT ON jobs_gold
        FOR EACH ROW EXECUTE FUNCTION fn_fill_fts_from_silver();
        """
    )

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
    op.execute("DROP TRIGGER IF EXISTS tr_update_gold_fts ON jobs_gold;")
    op.execute("DROP FUNCTION IF EXISTS fn_fill_fts_from_silver();")
    op.execute("ALTER TABLE jobs_gold DROP COLUMN IF EXISTS fts_tokens;")
