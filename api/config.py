from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    db_host: str = ""
    db_port: int = 5432
    db_user: str = ""
    db_password: str = ""
    db_name: str = ""
    port: int = 8080


settings = Settings()
