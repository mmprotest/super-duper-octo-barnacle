import json

from live_subtitles_local.rtc.rtc_config import resolve_rtc_settings


ENV_KEYS = [
    "LOCAL_DEV",
    "REMOTE_DEPLOYMENT",
    "RTC_ICE_SERVERS_JSON",
    "STUN_URLS",
    "TURN_URLS",
    "TURN_USERNAME",
    "TURN_CREDENTIAL",
    "TWILIO_ACCOUNT_SID",
    "TWILIO_AUTH_TOKEN",
]


def _clear_env(monkeypatch) -> None:
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_local_dev_uses_direct_config_without_turn(monkeypatch) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv("LOCAL_DEV", "true")

    settings = resolve_rtc_settings()

    assert settings.mode == "local_direct"
    assert settings.turn_configured is False
    assert settings.frontend_rtc_configuration == {"iceServers": []}
    assert settings.server_rtc_configuration == {"iceServers": []}


def test_remote_mode_uses_explicit_turn_env(monkeypatch) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv("REMOTE_DEPLOYMENT", "true")
    monkeypatch.setenv("STUN_URLS", "stun:stun.cloudflare.com:3478")
    monkeypatch.setenv("TURN_URLS", "turn:turn.example.com:3478?transport=udp,turns:turn.example.com:5349")
    monkeypatch.setenv("TURN_USERNAME", "user")
    monkeypatch.setenv("TURN_CREDENTIAL", "pass")

    settings = resolve_rtc_settings()

    assert settings.mode == "remote_env"
    assert settings.turn_configured is True
    assert settings.ice_server_count == 2
    assert settings.sources == ["STUN_URLS", "TURN_URLS"]


def test_remote_mode_accepts_json_ice_servers(monkeypatch) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv("REMOTE_DEPLOYMENT", "true")
    monkeypatch.setenv(
        "RTC_ICE_SERVERS_JSON",
        json.dumps(
            [
                {"urls": ["stun:stun.example.com:3478"]},
                {"urls": ["turn:turn.example.com:3478"], "username": "u", "credential": "p"},
            ]
        ),
    )

    settings = resolve_rtc_settings()

    assert settings.turn_configured is True
    assert settings.ice_server_count == 2
    assert settings.sources == ["RTC_ICE_SERVERS_JSON"]
