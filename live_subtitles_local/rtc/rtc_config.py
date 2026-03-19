from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


TRUE_VALUES = {"1", "true", "yes", "on"}


@dataclass(slots=True)
class RTCSettings:
    mode: str
    frontend_rtc_configuration: dict[str, Any]
    server_rtc_configuration: dict[str, Any]
    turn_configured: bool
    using_explicit_ice_config: bool
    ice_server_count: int
    description: str
    connection_guidance: str
    warnings: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)


def _env_flag(name: str) -> bool:
    value = os.getenv(name)
    return bool(value and value.strip().lower() in TRUE_VALUES)


def _split_csv_env(name: str) -> list[str]:
    value = os.getenv(name, "")
    return [item.strip() for item in value.split(",") if item.strip()]


def _build_ice_server(urls: list[str], username: str | None = None, credential: str | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"urls": urls if len(urls) > 1 else urls[0]}
    if username:
        payload["username"] = username
    if credential:
        payload["credential"] = credential
    return payload


def _load_json_ice_servers(raw: str) -> list[dict[str, Any]]:
    data = json.loads(raw)
    if not isinstance(data, list):
        raise ValueError("RTC_ICE_SERVERS_JSON must decode to a list of ICE server objects.")
    servers: list[dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict) or "urls" not in item:
            raise ValueError("Each RTC ICE server entry must be an object with a urls field.")
        servers.append(dict(item))
    return servers


def _load_twilio_ice_servers() -> tuple[list[dict[str, Any]], str | None]:
    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")
    if not account_sid and not auth_token:
        return [], None
    if not account_sid or not auth_token:
        return [], "Twilio TURN env vars are incomplete; set both TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN or remove them."
    try:
        from streamlit_webrtc.credentials import get_twilio_ice_servers
    except Exception as exc:  # pragma: no cover - import failure depends on environment
        return [], f"Twilio TURN was requested but streamlit-webrtc could not load Twilio support: {exc}"

    try:
        servers = get_twilio_ice_servers(account_sid, auth_token)
    except Exception as exc:
        return [], f"Failed to fetch TURN credentials from Twilio: {exc}"
    return [dict(server) for server in servers], None


def resolve_rtc_settings() -> RTCSettings:
    local_dev = _env_flag("LOCAL_DEV")
    remote_deployment = _env_flag("REMOTE_DEPLOYMENT")
    warnings: list[str] = []
    sources: list[str] = []

    if local_dev and remote_deployment:
        warnings.append("Both LOCAL_DEV and REMOTE_DEPLOYMENT are true; LOCAL_DEV takes precedence and TURN is disabled.")
        remote_deployment = False

    if local_dev or not remote_deployment:
        if not local_dev and not remote_deployment:
            warnings.append("Neither LOCAL_DEV nor REMOTE_DEPLOYMENT is set; defaulting to local/direct ICE mode for local-first development.")
        description = "Local/direct WebRTC mode. No external STUN/TURN servers are configured."
        guidance = (
            "For localhost browser-to-local-server use, TURN is usually unnecessary. "
            "If connection fails here, check browser microphone permission, localhost/HTTPS, and general WebRTC support first."
        )
        settings = RTCSettings(
            mode="local_direct",
            frontend_rtc_configuration={"iceServers": []},
            server_rtc_configuration={"iceServers": []},
            turn_configured=False,
            using_explicit_ice_config=True,
            ice_server_count=0,
            description=description,
            connection_guidance=guidance,
            warnings=warnings,
            sources=sources,
        )
        logger.info("RTC mode resolved to local/direct; TURN disabled.")
        for warning in warnings:
            logger.warning(warning)
        return settings

    ice_servers: list[dict[str, Any]] = []
    raw_ice_servers = os.getenv("RTC_ICE_SERVERS_JSON")
    if raw_ice_servers:
        try:
            ice_servers = _load_json_ice_servers(raw_ice_servers)
            sources.append("RTC_ICE_SERVERS_JSON")
        except ValueError as exc:
            warnings.append(str(exc))

    if not ice_servers:
        stun_urls = _split_csv_env("STUN_URLS")
        turn_urls = _split_csv_env("TURN_URLS")
        turn_username = os.getenv("TURN_USERNAME")
        turn_credential = os.getenv("TURN_CREDENTIAL")
        if stun_urls:
            ice_servers.append(_build_ice_server(stun_urls))
            sources.append("STUN_URLS")
        if turn_urls:
            ice_servers.append(_build_ice_server(turn_urls, username=turn_username, credential=turn_credential))
            sources.append("TURN_URLS")
            if not turn_username or not turn_credential:
                warnings.append("TURN_URLS is set without both TURN_USERNAME and TURN_CREDENTIAL; some TURN servers will reject authentication.")

    if not any("turn:" in str(server.get("urls", "")) for server in ice_servers):
        twilio_servers, twilio_warning = _load_twilio_ice_servers()
        if twilio_warning:
            warnings.append(twilio_warning)
        elif twilio_servers:
            ice_servers = twilio_servers
            sources.append("TWILIO_ACCOUNT_SID/TWILIO_AUTH_TOKEN")

    turn_configured = any("turn:" in json.dumps(server) for server in ice_servers)
    if not ice_servers:
        warnings.append(
            "REMOTE_DEPLOYMENT is enabled but no RTC ICE servers are configured. The app will still start, but remote browser connections may fail because TURN is not configured."
        )

    description = (
        "Remote deployment RTC mode. "
        + ("TURN is configured via environment variables." if turn_configured else "TURN is not configured.")
    )
    guidance = (
        "Remote deployments often need TURN for browsers behind NAT/firewalls. "
        "If a remote connection fails and TURN is not configured, the RTC config is the likely cause."
        if not turn_configured
        else "TURN is configured. If a remote connection still fails, verify the TURN credentials, ICE server reachability, and browser secure-context requirements."
    )
    settings = RTCSettings(
        mode="remote_env",
        frontend_rtc_configuration={"iceServers": ice_servers},
        server_rtc_configuration={"iceServers": ice_servers},
        turn_configured=turn_configured,
        using_explicit_ice_config=True,
        ice_server_count=len(ice_servers),
        description=description,
        connection_guidance=guidance,
        warnings=warnings,
        sources=sources,
    )
    logger.info(
        "RTC mode resolved to remote_env; ice_servers=%s turn_configured=%s sources=%s",
        settings.ice_server_count,
        settings.turn_configured,
        settings.sources,
    )
    for warning in warnings:
        logger.warning(warning)
    return settings
