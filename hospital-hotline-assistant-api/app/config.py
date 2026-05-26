from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    app_name: str = "Hospital Hotline Assistant API"
    environment: str = "development"
    database_url: str = "postgresql://postgres:postgres@localhost:5432/hospital_hotline"
    cors_origins: list[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]
    # Regex companion to ``cors_origins`` so dev setups (WSL2, LAN IPs,
    # custom Vite ports) are accepted without having to enumerate every
    # variant. The default covers http(s)://localhost:<port>,
    # http(s)://127.0.0.1:<port>, and any private LAN IPv4 on any port.
    # Set to ``None`` (or override via env) to disable the regex match.
    cors_origin_regex: str | None = (
        r"^https?://("
        r"localhost"
        r"|127\.0\.0\.1"
        r"|10\.\d{1,3}\.\d{1,3}\.\d{1,3}"
        r"|192\.168\.\d{1,3}\.\d{1,3}"
        r"|172\.(1[6-9]|2\d|3[0-1])\.\d{1,3}\.\d{1,3}"
        r")(:\d+)?$"
    )
    slack_webhook_url: str | None = None
    alert_severity_threshold: str = "emergency"
    alert_cooldown_seconds: int = 300
    google_cloud_project: str | None = None
    google_cloud_location: str = "us-central1"
    google_model_name: str = "gemini-2.5-flash"
    google_application_credentials: str | None = None
    google_ai_enabled: bool = False
    # Upper bound on how many times the ADK LoopAgent re-runs the
    # triage reasoner per HTTP turn. The inner LlmAgent already does
    # its own multi-tool micro-loop in a single run, so this is a
    # safety cap, not a normal-path tuning knob. Keep small to bound
    # latency.
    adk_max_tool_iterations: int = 3

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

settings = Settings()