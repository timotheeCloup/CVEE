"""refresh jobs_gold.fts_tokens on insert or update

Revision ID: d4a1b2c3e5f6
Revises: b1c7f0a4d2e9
Create Date: 2026-10-03 10:00:00.000000

"""

from collections.abc import Sequence

from alembic import op

revision: str = "d4a1b2c3e5f6"
down_revision: str | Sequence[str] | None = "b1c7f0a4d2e9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


_FILL_FTS_BEFORE = """
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

        NEW.fts_tokens :=
            setweight(to_tsvector('french', unaccent(COALESCE(source_record.intitule, ''))), 'A') ||
            setweight(to_tsvector('french', unaccent(COALESCE(source_record.description, ''))), 'B') ||
            setweight(to_tsvector('french', unaccent(COALESCE(clean_competences, ''))), 'C') ||
            setweight(to_tsvector('french', unaccent(COALESCE(clean_qualites, ''))), 'C');

        RETURN NEW;
    END;
    $$ LANGUAGE plpgsql;
"""

_FILL_FTS_AFTER = """
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


def upgrade() -> None:
    # The ingest now upserts jobs_silver, so a job's text fields can change
    # after its gold row already exists. Compute fts_tokens in a
    # BEFORE INSERT OR UPDATE trigger that sets NEW directly, instead of the
    # previous AFTER INSERT trigger that ran a separate UPDATE: this keeps the
    # tokens in sync on updates too, without any recursive trigger.
    op.execute(_FILL_FTS_BEFORE)
    op.execute("DROP TRIGGER IF EXISTS tr_update_gold_fts ON jobs_gold;")
    op.execute(
        """
        CREATE TRIGGER tr_update_gold_fts BEFORE INSERT OR UPDATE ON jobs_gold
        FOR EACH ROW EXECUTE FUNCTION fn_fill_fts_from_silver();
        """
    )


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS tr_update_gold_fts ON jobs_gold;")
    op.execute(_FILL_FTS_AFTER)
    op.execute(
        """
        CREATE TRIGGER tr_update_gold_fts AFTER INSERT ON jobs_gold
        FOR EACH ROW EXECUTE FUNCTION fn_fill_fts_from_silver();
        """
    )
