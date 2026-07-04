from pydantic import BaseModel, Field, field_validator


class JobResult(BaseModel):
    job_id: str
    similarity_score: float = Field(ge=0.0, le=1.0)
    embedding_score: float
    fts_score: float
    combined_score: float
    intitule: str = ""
    entreprise: str = ""
    lieu: str = ""
    type_contrat: str = ""
    date_creation: str = ""
    matching_terms: list[str] = []

    model_config = {"from_attributes": True}

    @field_validator("intitule", "entreprise", "lieu", "type_contrat", "date_creation", mode="before")
    @classmethod
    def _none_to_empty(cls, v: object) -> str:
        return v if isinstance(v, str) else ""


class EmbedResponse(BaseModel):
    top_jobs: list[JobResult] = []


class HealthResponse(BaseModel):
    status: str
