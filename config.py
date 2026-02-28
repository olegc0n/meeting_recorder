"""
Configuration and settings persistence.
"""
from PySide6.QtCore import QSettings

APP_NAME = "MeetingTranscriber"
SETTINGS_ORG = "MeetingTranscriber"


def get_settings() -> QSettings:
    """Return QSettings instance for the app."""
    return QSettings(SETTINGS_ORG, APP_NAME)


def default_transcription_config() -> dict:
    """Default transcription settings."""
    return {
        "model_size": "tiny",
        "device": "cpu",
        "compute_type": "int8",
        "beam_size": 5,
        "buffer_duration": 3.0,
        "vad_filter": True,
        "use_initial_prompt": True,
        "llm_analysis_enabled": True,
        "llm_analysis_interval": 30,
    }


def load_transcription_config() -> dict:
    """Load transcription config from QSettings."""
    s = get_settings()
    cfg = default_transcription_config()
    cfg["model_size"] = s.value("model_size", cfg["model_size"], type=str)
    cfg["device"] = s.value("device", cfg["device"], type=str)
    cfg["compute_type"] = s.value("compute_type", cfg["compute_type"], type=str)
    cfg["beam_size"] = s.value("beam_size", cfg["beam_size"], type=int)
    cfg["buffer_duration"] = s.value("buffer_duration", cfg["buffer_duration"], type=float)
    cfg["vad_filter"] = s.value("vad_filter", cfg["vad_filter"], type=bool)
    cfg["use_initial_prompt"] = s.value("use_initial_prompt", cfg["use_initial_prompt"], type=bool)
    cfg["llm_analysis_enabled"] = s.value("llm_analysis_enabled", cfg["llm_analysis_enabled"], type=bool)
    cfg["llm_analysis_interval"] = s.value("llm_analysis_interval", cfg["llm_analysis_interval"], type=int)
    return cfg


def save_transcription_config(cfg: dict) -> None:
    """Save transcription config to QSettings."""
    s = get_settings()
    for k, v in cfg.items():
        s.setValue(k, v)
    s.sync()


def load_audio_source() -> str:
    """Return the last-selected audio source device id (or empty string)."""
    return get_settings().value("audio_source_id", "", type=str)


def save_audio_source(device_id: str) -> None:
    """Persist the selected audio source device id."""
    s = get_settings()
    s.setValue("audio_source_id", device_id)
    s.sync()
