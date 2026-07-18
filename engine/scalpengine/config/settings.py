"""Central runtime settings, loaded from .env / environment.

Why: a single validated settings object keeps paper-mode enforcement,
credentials, and fallback selection (SQLite / in-process pub-sub) in one
auditable place. The paper-URL assertion itself lives in execution.broker —
close to the only code that can talk to Alpaca.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # mode
    trading_mode: str = "paper"  # "live" only via Phase 4 gates (§8.5)

    # alpaca paper credentials
    alpaca_api_key: str = ""
    alpaca_secret_key: str = ""
    alpaca_paper_base_url: str = "https://paper-api.alpaca.markets"

    # alpaca live credentials (Phase 4; absent -> live structurally impossible)
    alpaca_live_api_key: str = ""
    alpaca_live_secret_key: str = ""
    max_live_equity_usd: float | None = None

    # persistence / pub-sub
    database_url: str = ""
    db_backend: str = "sqlite"
    sqlite_path: str = "scalpengine.db"
    redis_url: str = ""

    # api auth
    app_password: str = "change-me"
    jwt_secret: str = "change-me"
    jwt_expiry_hours: int = 24
    cors_origins: str = "http://localhost:3000"  # comma-separated

    # engine
    risk_tier: str = "medium"
    scalp_profile: str = "off"  # off | auto | small | mid | large
    # dir holding model.joblib/features.json/inference.json (export_model.py);
    # empty -> bracket paths active but the second-cadence model loop is off
    scalp_artifact_dir: str = ""
    # absolute equity floor: kill + halt before the account sinks under the
    # day-trading minimum (small-account mode). 0 = disabled.
    min_equity_halt_usd: float = 0.0
    bar_interval_s: int = 60
    signal_bar_interval_s: int = 300
    history_warmup_days: int = 30
    staleness_pause_s: int = 60
    staleness_kill_s: int = 180
    broker_error_kill_count: int = 5
    broker_error_kill_window_s: int = 60
    heartbeat_interval_s: int = 10
    models_dir: str = "models"

    def resolved_database_url(self) -> str:
        """Postgres when DATABASE_URL is set; otherwise SQLite fallback."""
        if self.database_url:
            return self.database_url
        return f"sqlite:///{self.sqlite_path}"


def get_settings() -> Settings:
    return Settings()
