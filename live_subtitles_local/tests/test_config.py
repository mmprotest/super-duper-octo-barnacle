from live_subtitles_local.asr.schemas import SessionConfig
from live_subtitles_local.config.loader import load_yaml_file


def test_config_loading() -> None:
    raw = load_yaml_file("live_subtitles_local/config/default.yaml")
    config = SessionConfig(**{key: value for key, value in raw.items() if key in SessionConfig.__dataclass_fields__})
    assert config.whisper_device == "cuda"
    assert config.target_language == "en"
    assert config.receiver_queue_size == 1024
    assert config.asr_poll_interval_ms == 350
