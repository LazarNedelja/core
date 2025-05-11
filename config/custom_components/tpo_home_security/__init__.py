"""The tpo_home_security integration."""

from __future__ import annotations

import os

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant
from homeassistant.helpers.typing import ConfigType

from .const import DOMAIN
from .listener import register_listeners
from .notifications import EmailNotifier, PushNotifier

_PLATFORMS: list[Platform] = [Platform.LIGHT]


async def async_setup_entry(hass, entry) -> bool:  # noqa: D103
    # Read options (fall back to defaults if not set)
    # service_name = entry.options.get("email_service", "email_alerts")
    # push_svc = entry.options.get("push_service", "mobile_app_mydevice")
    recipients = entry.options.get("email_recipients", [])

    # Create your notifier instance
    email_notifier = EmailNotifier(hass)
    push_notifier = PushNotifier(hass)

    # Keep it somewhere for later (e.g. attach to hass.data)
    hass.data.setdefault(DOMAIN, {})
    hass.data[DOMAIN]["email_notifier"] = email_notifier
    hass.data[DOMAIN]["email_recipients"] = recipients
    hass.data[DOMAIN]["push_notifier"] = push_notifier

    # Register your event listener (from listener.py)
    register_listeners(hass)

    hass.http.register_static_path(
        f"/local/{DOMAIN}",
        os.path.dirname(__file__),
        False,  # don’t require auth
    )

    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    return await hass.config_entries.async_unload_platforms(entry, _PLATFORMS)


def setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up tpo_home_security for YAML-based configuration."""
    # only proceed if user actually has the domain in configuration.yaml
    if DOMAIN in config:
        # read your YAML keys (you could allow overriding service/name here too)
        notifier = EmailNotifier(hass, service_name="email_alerts")
        recipients = config[DOMAIN].get("email_recipients", [])

        # populate hass.data just as in async_setup_entry
        hass.data.setdefault(DOMAIN, {})["email_notifier"] = notifier
        hass.data[DOMAIN]["email_recipients"] = recipients

    # register_listeners(hass)
    return True
