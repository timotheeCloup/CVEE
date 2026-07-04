"""initial_schema

Revision ID: 8cef33abbe6d
Revises:
Create Date: 2026-07-04 15:39:06.534809

"""

from collections.abc import Sequence

from alembic import op

revision: str = "8cef33abbe6d"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    op.execute(
        """
        CREATE TABLE IF NOT EXISTS jobs_silver (
            job_id TEXT PRIMARY KEY,
            intitule TEXT,
            description TEXT,
            vector_text_input TEXT,
            dateCreation TEXT,
            dateActualisation TEXT,
            lieuTravail JSONB,
            entreprise JSONB,
            contact JSONB,
            agence JSONB,
            origineOffre JSONB,
            contexteTravail JSONB,
            salaire JSONB,
            competences JSONB,
            formations JSONB,
            langues JSONB,
            qualitesProfessionnelles JSONB,
            permis JSONB,
            romeCode TEXT,
            romeLibelle TEXT,
            appellationlibelle TEXT,
            typeContrat TEXT,
            typeContratLibelle TEXT,
            natureContrat TEXT,
            experienceExige TEXT,
            experienceLibelle TEXT,
            dureeTravailLibelle TEXT,
            dureeTravailLibelleConverti TEXT,
            alternance BOOLEAN,
            nombrePostes INTEGER,
            accessibleTH BOOLEAN,
            qualificationCode TEXT,
            qualificationLibelle TEXT,
            codeNAF TEXT,
            secteurActivite TEXT,
            secteurActiviteLibelle TEXT,
            trancheEffectifEtab TEXT,
            offresManqueCandidats BOOLEAN,
            entrepriseAdaptee BOOLEAN,
            employeurHandiEngage BOOLEAN,
            deplacementCode TEXT,
            deplacementLibelle TEXT,
            experienceCommentaire TEXT,
            complementExercice TEXT,
            ingestion_date DATE
        );
        """
    )

    op.execute(
        """
        CREATE TABLE IF NOT EXISTS jobs_gold (
            job_id TEXT PRIMARY KEY,
            embedding vector(384),
            CONSTRAINT fk_job
                FOREIGN KEY (job_id)
                REFERENCES jobs_silver(job_id)
                ON DELETE CASCADE
        );
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS jobs_gold CASCADE;")
    op.execute("DROP TABLE IF EXISTS jobs_silver CASCADE;")
