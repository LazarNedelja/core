from abc import ABC, abstractmethod

from homeassistant.core import HomeAssistant


class NotificationChannel(ABC):
    """Interface for all notification channels."""

    @abstractmethod
    def send(self, title: str, message: str, targets: list[str] | None = None) -> None:
        """Send a notification.

        :param title: The notification title (or email subject).
        :param message: The notification body.
        :param targets: Optional list of targets (emails, device IDs, etc.).
        """


class EmailNotifier(NotificationChannel):
    """A dedicated Python class to send email notifications via Home Assistant's SMTP notify integration."""

    def __init__(self, hass: HomeAssistant, service_name: str = "email_alerts"):
        """Initialize the EmailNotifier.

        :param hass: Home Assistant instance
        :param service_name: the notify service (after notify.) to call
        """
        self.hass = hass
        self.service_name = service_name

    def send(
        self, subject: str, message: str, targets: list[str] | None = None
    ) -> None:
        """Send an email notification via HA notify service.

        :param subject: Email subject
        :param message: Email body
        :param targets: Optional list of recipient emails
        """
        data = {"title": subject, "message": message}
        if targets:
            data["target"] = targets
        self.hass.services.call("notify", self.service_name, data)


class PushNotifier(NotificationChannel):
    """A dedicated Python class to send push notifications via Home Assistant mobile_app for SM-S928B."""

    def __init__(self, hass: HomeAssistant, service_name: str = "mobile_app_sm_s928b"):
        """Initialize the PushNotifier for device SM-S928B.

        :param hass: Home Assistant instance
        :param service_name: the mobile_app notify service name for SM-S928B
        """
        self.hass = hass
        self.service_name = service_name

    def send(self, title: str, message: str, targets: list[str] | None = None) -> None:
        """Send a push notification via HA mobile_app notify service for SM-S928B.

        :param title: Notification title
        :param message: Notification body
        :param targets: Optional list of target device IDs (not needed for a single device)
        """
        data = {"title": title, "message": message}
        if targets:
            data["target"] = targets
        self.hass.services.call("notify", self.service_name, data)


class PushNotifier(NotificationChannel):
    """A dedicated Python class to send push notifications via Home Assistant mobile_app for SM-S928B."""

    def __init__(self, hass: HomeAssistant, service_name: str = "mobile_app_sm_s928b"):
        """Initialize the PushNotifier for device SM-S928B.

        :param hass: Home Assistant instance
        :param service_name: the mobile_app notify service name for SM-S928B
        """
        self.hass = hass
        self.service_name = service_name

    def send(self, title: str, message: str, targets: list[str] | None = None) -> None:
        """Send a push notification via HA mobile_app notify service for SM-S928B.

        :param title: Notification title
        :param message: Notification body
        :param targets: Optional list of target device IDs (not needed for a single device)
        """
        data = {"title": title, "message": message}
        if targets:
            data["target"] = targets
        self.hass.services.call("notify", self.service_name, data)


class NotificationService:
    """Manages multiple NotificationChannel instances and broadcasts to all of them."""

    def __init__(self) -> None:
        self.channels: list[NotificationChannel] = []

    def add_channel(self, channel: NotificationChannel) -> None:
        """Register a new channel (if not already present)."""
        if channel not in self.channels:
            self.channels.append(channel)

    def delete_channel(self, channel: NotificationChannel) -> None:
        """Remove an existing channel."""
        if channel in self.channels:
            self.channels.remove(channel)

    def set_channel(self, channel: NotificationChannel) -> None:
        """Drop all channels and use only this one."""
        self.channels = [channel]

    def notify(self, subject: str, body: str) -> None:
        """Send the same notification to every registered channel."""
        for ch in self.channels:
            ch.send(subject, body)


# Usage example for SM-S928B:
#
# from .notifications import EmailNotifier, PushNotifier
# from .const import DOMAIN
#
# async def async_setup_entry(hass, entry):
#     hass.data.setdefault(DOMAIN, {})
#
#     # Email notifier
#     email_notifier = EmailNotifier(hass)
#     hass.data[DOMAIN]["email_notifier"] = email_notifier
#
#     # Push notifier for SM-S928B
#     push_notifier = PushNotifier(hass)
#     hass.data[DOMAIN]["push_notifier"] = push_notifier
#
#     return True
#
# In the event handler:
#
# data = hass.data[DOMAIN]
# email_notifier = data.get("email_notifier")
# push_notifier = data.get("push_notifier")
# recipients = data.get("email_recipients")
#
# if email_notifier and push_notifier:
#     email_notifier.send("🏠 Home Security Alert", "Person detected!", recipients)
#     push_notifier.send("🏠 Home Security Alert", "Person detected!")
