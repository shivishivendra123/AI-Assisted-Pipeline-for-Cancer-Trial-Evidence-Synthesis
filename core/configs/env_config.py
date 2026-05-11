"""
Central environment configuration loader for REASON project.
Loads all configuration from environment variables with sensible defaults.
"""
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load .env file from project root
env_path = Path(__file__).resolve().parent.parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
    print(f"[INFO] Loaded environment variables from: {env_path}")
else:
    print(f"[WARN] No .env file found at {env_path}. Using defaults or environment variables.")


class Config:
    """Centralized configuration from environment variables."""

    # ==============================================
    # Google Cloud Vertex AI Configuration
    # ==============================================
    GCP_PROJECT_ID: str = os.getenv("GCP_PROJECT_ID", "evidence-synthesis-gemma")
    GCP_LOCATION: str = os.getenv("GCP_LOCATION", "us-east4")
    GCP_ENDPOINT_ID: str = os.getenv("GCP_ENDPOINT_ID", "mg-endpoint-fc4b2334-585c-4e64-9c0d-df7caee0cf01")
    GCP_DEDICATED_DNS: str = os.getenv(
        "GCP_DEDICATED_DNS",
        "mg-endpoint-fc4b2334-585c-4e64-9c0d-df7caee0cf01.us-east4-132817493282.prediction.vertexai.goog"
    )

    # Vertex AI Gemini (separate from the dedicated medgemma endpoint above).
    # Gemini 2.5 Flash is broadly available in us-central1; the medgemma
    # endpoint's region (us-east4) doesn't host every Gemini variant.
    GEMINI_LOCATION: str = os.getenv("GEMINI_LOCATION", "us-central1")
    GEMINI_MODEL:    str = os.getenv("GEMINI_MODEL",    "gemini-2.5-flash")

    # ==============================================
    # PubMed/NCBI API Configuration
    # ==============================================
    NCBI_EMAIL: str = os.getenv("NCBI_EMAIL", "sgupta13@mail.yu.edu")
    NCBI_API_KEY: str = os.getenv("NCBI_API_KEY", "dc7de5bfc1cb115021baf6b463aa52728408")
    PUBMED_MAX_RESULTS: int = int(os.getenv("PUBMED_MAX_RESULTS", "500"))

    # ==============================================
    # OpenAI Configuration (Optional)
    # ==============================================
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4")

    # ==============================================
    # LLM Model Parameters
    # ==============================================
    # LLM_TEMPERATURE=0.0 → deterministic sampling across every pipeline
    # stage (mesh expansion, PICO extraction, screening, eligibility,
    # study-characteristics, outcome extraction, evidence synthesis).
    # Override to 0.2-0.4 in .env if you want exploratory variability.
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.0"))
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "800"))
    LLM_TIMEOUT: int = int(os.getenv("LLM_TIMEOUT", "300"))

    # ==============================================
    # Pipeline Configuration
    # ==============================================
    DEFAULT_MAX_STUDIES: int = int(os.getenv("DEFAULT_MAX_STUDIES", "5"))
    DEFAULT_SCORE_THRESHOLD: float = float(os.getenv("DEFAULT_SCORE_THRESHOLD", "3.0"))
    DEFAULT_MAX_WORKERS: int = int(os.getenv("DEFAULT_MAX_WORKERS", "5"))

    # ==============================================
    # Artifact Storage Paths
    # ==============================================
    ARTIFACTS_BASE_DIR: str = os.getenv("ARTIFACTS_BASE_DIR", "./artifacts")
    ARTIFACTS_PICO_DIR: str = os.getenv("ARTIFACTS_PICO_DIR", "artifacts_day1")
    ARTIFACTS_AUGMENTED_DIR: str = os.getenv("ARTIFACTS_AUGMENTED_DIR", "artifacts_day2")
    ARTIFACTS_MESH_DIR: str = os.getenv("ARTIFACTS_MESH_DIR", "artifacts_day3")
    ARTIFACTS_PUBMED_DIR: str = os.getenv("ARTIFACTS_PUBMED_DIR", "artifacts_day4")
    ARTIFACTS_SCREENING_DIR: str = os.getenv("ARTIFACTS_SCREENING_DIR", "artifacts_day5")
    ARTIFACTS_EXTRACTION_DIR: str = os.getenv("ARTIFACTS_EXTRACTION_DIR", "artifacts_day6")
    ARTIFACTS_SYNTHESIS_DIR: str = os.getenv("ARTIFACTS_SYNTHESIS_DIR", "artifacts_day7")

    # ==============================================
    # Retry Configuration
    # ==============================================
    API_RETRY_ATTEMPTS: int = int(os.getenv("API_RETRY_ATTEMPTS", "5"))
    API_RETRY_WAIT: int = int(os.getenv("API_RETRY_WAIT", "1"))

    @classmethod
    def validate(cls):
        """Validate critical configuration values."""
        errors = []

        # Check for required sensitive values
        if cls.NCBI_EMAIL == "sgupta13@mail.yu.edu":
            errors.append("⚠️  NCBI_EMAIL is using default value. Please set your own email in .env")

        if cls.NCBI_API_KEY == "f5fae3152f30c1cbe67e396db8f3a9247508":
            errors.append("⚠️  NCBI_API_KEY is using default value. Please set your own API key in .env")

        if errors:
            print("\n".join(errors))
            print("\nCreate a .env file based on .env.example and set your credentials.\n")

        return len(errors) == 0

    @classmethod
    def display(cls):
        """Display current configuration (hiding sensitive values)."""
        print("\n" + "="*60)
        print("REASON Configuration")
        print("="*60)
        print(f"GCP Project ID:        {cls.GCP_PROJECT_ID}")
        print(f"GCP Location:          {cls.GCP_LOCATION}")
        print(f"GCP Endpoint ID:       {cls.GCP_ENDPOINT_ID[:20]}...")
        print(f"NCBI Email:            {cls.NCBI_EMAIL}")
        print(f"NCBI API Key:          {'*' * 20 if cls.NCBI_API_KEY else 'Not Set'}")
        print(f"PubMed Max Results:    {cls.PUBMED_MAX_RESULTS}")
        print(f"LLM Temperature:       {cls.LLM_TEMPERATURE}")
        print(f"LLM Max Tokens:        {cls.LLM_MAX_TOKENS}")
        print(f"Default Max Studies:   {cls.DEFAULT_MAX_STUDIES}")
        print(f"Default Score Thresh:  {cls.DEFAULT_SCORE_THRESHOLD}")
        print(f"Default Max Workers:   {cls.DEFAULT_MAX_WORKERS}")
        print(f"Artifacts Base Dir:    {cls.ARTIFACTS_PICO_DIR}")
        print("="*60 + "\n")


# Create a singleton instance
config = Config()

# Validate on import (optional - can be disabled)
# config.validate()
