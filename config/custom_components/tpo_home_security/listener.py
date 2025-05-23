"""Event listeners for the tpo_home_security integration, refactored using the Observer pattern and a Facade structural pattern."""

from abc import ABC, abstractmethod
import itertools
import logging
from pathlib import Path
from typing import Any

import cv2
import librosa
import numpy as np
import torch

from homeassistant.core import Event, HomeAssistant, callback
from homeassistant.helpers.event import async_call_later

from .const import DOMAIN

np.complex = complex

# Force full weights load so ultralytics can load YOLOv5
_orig_torch_load = torch.load


def _torch_load_force_full(*args: Any, **kwargs: Any) -> Any:
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)


torch.load = _torch_load_force_full

from ultralytics import YOLO  # noqa: E402

_LOGGER = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
CAMERA_ENTITY = "camera.192_168_1_218"  # ← change this to your camera entity_id
SNAPSHOT_DIR = Path(__file__).parent  # config/www
SNAPSHOT_FILE = SNAPSHOT_DIR / "last_snapshot.jpg"

VIDEO_FILE = Path(__file__).parent / "SecurityCam.mp4"
MODEL_FILE = Path(__file__).parent / "yolov8n.pt"

BLINK_ENTITY = "input_boolean.blinking_lights"

PERSON_CLASS = 0
ANIMAL_CLASS_IDS = [
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
]
CONFIDENCE = 0.5
# ─────────────────────────────────────────────────────────────────────────────

# Load the model **once** at import time
try:
    MODEL = YOLO(str(MODEL_FILE))
    _LOGGER.info("Loaded YOLO model from %s", MODEL_FILE)
except Exception as e:
    MODEL = None
    _LOGGER.error("Failed loading YOLO model: %s", e)


class YoloModelSingleton:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            try:
                cls._instance = super().__new__(cls)
                cls._instance.model = YOLO(str(MODEL_FILE))
                _LOGGER.info("Loaded YOLO model from %s", MODEL_FILE)
            except Exception as e:
                _LOGGER.error("Failed loading YOLO model: %s", e)
                cls._instance = None
        return cls._instance

    def detect_person(self, image_path: str) -> bool:
        if not self.model:
            return False
        try:
            results = self.model(image_path)[0]
            return any(
                int(box.cls) == PERSON_CLASS and box.conf.cpu().item() >= CONFIDENCE
                for box in results.boxes
            )
        except Exception as e:
            _LOGGER.error("YOLO error: %s", e)
            return False

    def detect_animals(self, image_path: str) -> list[str]:
        """Return list of animal class names detected in the image."""
        if not self.model:
            return []
        try:
            results = self.model.predict(source=image_path, classes=ANIMAL_CLASS_IDS)[0]
            animals = [self.model.names[int(box.cls)] for box in results.boxes]
            return list(dict.fromkeys(animals))
        except Exception as e:
            _LOGGER.error("YOLO error during animal detection: %s", e)
            return []


# ── Observer Pattern ───────────────────────────────────────────────────────────
class Observer(ABC):
    """Interface for all sensor observers."""

    @abstractmethod
    def update(self, state: str) -> None:
        """Handle an update from the Sensor.

        :param state: "DETECTED" or "CLEAR"
        """


class Sensor:
    def __init__(self) -> None:
        """Initialize the Sensor with an empty list of subscribers."""

        self.subscribers = []

    def register(self, subscriber):
        self.subscribers.append(subscriber)

    def notify(self, state):
        for sub in self.subscribers:
            sub.update(state)


class Alarm(Observer):
    """Observer that handles alarm state changes."""

    def __init__(self, hass: HomeAssistant) -> None:
        """Initialize the Alarm observer with the Home Assistant instance."""

        self.hass = hass

    def update(self, state: str) -> None:
        """Handle an update from the Sensor.

        :param state: "DETECTED" or "CLEAR"
        """
        # svc = "turn_on" if state == "DETECTED" else "turn_off"
        # self.hass.services.call(
        #     "input_boolean",
        #     svc,
        #     {"entity_id": "input_boolean.alarm_toggle"},
        # )
        # _LOGGER.info("Alarm turned %s", "ON" if state == "DETECTED" else "OFF")

        if state == "DETECTED":
            # Turn it on immediately
            self.hass.services.call(
                "input_boolean",
                "turn_on",
                {"entity_id": "input_boolean.alarm_toggle"},
            )
            _LOGGER.info("Alarm turned ON")

            self.hass.services.call(
                "input_boolean",
                "turn_on",
                {"entity_id": BLINK_ENTITY},
            )
            _LOGGER.info("Blinking lights turned ON")

            # Schedule the delayed turn-off back on the main loop thread
            # so async_call_later is invoked from the event loop.
            self.hass.loop.call_soon_threadsafe(
                lambda: async_call_later(self.hass, 10, self._auto_turn_off)
            )
        else:
            # Immediate turn-off if CLEAR
            self.hass.services.call(
                "input_boolean",
                "turn_off",
                {"entity_id": "input_boolean.alarm_toggle"},
            )
            _LOGGER.info("Alarm turned OFF")

            # Also turn blinking lights OFF
            self.hass.services.call(
                "input_boolean",
                "turn_off",
                {"entity_id": BLINK_ENTITY},
            )
            _LOGGER.info("Blinking lights turned OFF")

    def _auto_turn_off(self, now) -> None:
        """Pass callback to async_call_later to switch the alarm off."""

        self.hass.services.call(
            "input_boolean",
            "turn_off",
            {"entity_id": "input_boolean.alarm_toggle"},
        )
        _LOGGER.info("Alarm automatically turned OFF after timeout")

        # Turn blinking lights OFF
        self.hass.services.call(
            "input_boolean",
            "turn_off",
            {"entity_id": BLINK_ENTITY},
        )
        _LOGGER.info("Blinking lights automatically turned OFF after timeout")


class Notifier(Observer):
    """Observer that handles notifications."""

    def update(self, state: str) -> None:
        """Handle an update from the Sensor.

        :param state: "DETECTED" or "CLEAR"
        """
        if state == "DETECTED":
            _LOGGER.info("Notifier: Person detected alert triggered")


# ── Structural Pattern: Facade Pattern ─────────────────────────────────────────
class SecurityFacade:
    """Facade to simplify detection, state decision, and notification flow."""

    def __init__(self, hass: HomeAssistant):
        # initialize subject and observers
        self.sensor = Sensor()
        self.sensor.register(Alarm(hass))
        self.sensor.register(Notifier())
        # reuse singleton for detection
        self.yolo = YoloModelSingleton()

    def process_frame(self, image_path: str) -> bool:
        # detect person and notify subscribers via Sensor
        detected = self.yolo.detect_person(image_path)
        state = "DETECTED" if detected else "CLEAR"
        self.sensor.notify(state)
        return detected

    def process_animal(self, image_path: str) -> list[str]:
        # detect animals and notify subscribers via Sensor
        animals = self.yolo.detect_animals(image_path)
        state = "ANIMAL DETECTED" if animals else "NO ANIMAL"
        self.sensor.notify(state)
        return animals


# Global facade instance (to be created in register_listeners)
security_facade: SecurityFacade | None = None

# VIDEO_ENTITY = "home_sec.security_cam_video"


@callback
def handle_sensor_toggle_update(hass: HomeAssistant, event: Event) -> None:
    email_notifier = hass.data[DOMAIN]["email_notifier"]
    push_notifier = hass.data[DOMAIN]["push_notifier"]
    recipients = hass.data[DOMAIN]["email_recipients"]

    entity_id = event.data.get("entity_id")

    sensor_state = hass.states.get("input_boolean.motion_error")

    if entity_id == "input_boolean.motion_error":
        # Send the push notification if there is an error on the motion sensor
        if sensor_state and sensor_state.state == "on":
            push_notifier.send(
                title="🏠 Home Security Alert",
                message="Motion sensor error detected.",
            )
            # Send the email
            email_notifier.send(
                subject="🏠 Home Security Alert",
                message="Motion sensor error detected.",
                targets=recipients,
            )

    if entity_id not in ("input_boolean.sensor_toggle", "input_select.home_mode"):
        return

    sensor_state = hass.states.get("input_boolean.sensor_toggle")
    if sensor_state:
        hass.states.set(
            "home_sec.sensor_toggle_state",
            sensor_state.state,
            {
                "friendly_name": "Sensor Toggle State",
                "original_entity": "input_boolean.sensor_toggle",
            },
        )

    mode_state = hass.states.get("input_select.home_mode")
    if not sensor_state or not mode_state:
        _LOGGER.debug("Missing entities, skipping detection")
        return

    if sensor_state.state != "on" or mode_state.state != "AWAY":
        _LOGGER.debug(
            "Conditions not met (sensor=%s, mode=%s)",
            sensor_state.state,
            mode_state.state,
        )
        return

    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

    # Test camera snapshot
    hass.services.call(
        "camera",
        "snapshot",
        {
            "entity_id": CAMERA_ENTITY,
            "filename": str(SNAPSHOT_FILE),
        },
    )
    _LOGGER.info("Saved camera snapshot to %s", SNAPSHOT_FILE)

    # Test video snapshot

    # Izvlacenje prvog frejma iz videa
    # cap = cv2.VideoCapture(str(VIDEO_FILE))
    # success, frame = cap.read()
    # cap.release()
    # if not success:
    #     _LOGGER.error("Failed to read frame from %s", VIDEO_FILE)
    #     return

    # tmp = VIDEO_FILE.parent / "_snapshot.jpg"

    # cv2.imwrite(str(tmp), frame)

    # Delegate detection & notification to the Facade
    if security_facade:
        detected = security_facade.process_frame(str(SNAPSHOT_FILE))
        if detected:
            # Send the email
            email_notifier.send(
                subject="🏠 Home Security Alert",
                message="A person was detected by your camera.",
                targets=recipients,
            )
            # send the push notification
            push_notifier.send(
                title="🏠 Home Security Alert",
                message="A person was detected by your camera.",
            )
        animals = security_facade.process_animal(str(SNAPSHOT_FILE))
        if animals:
            # send the push notification
            push_notifier.send(
                title="🏠 Home Security Alert",
                message=f"An animal was detected by your camera: {', '.join(animals)}",
            )


import asyncio

PHONE_ENTITY_ID = (
    "device_tracker.sm_s928b"  # Change this to your actual device_tracker entity ID
)


async def phone_presence_check(hass: HomeAssistant):
    previous_state = None
    while True:
        state = hass.states.get(PHONE_ENTITY_ID)
        if state is None:
            print(f"{PHONE_ENTITY_ID} entity not found")
        else:
            current_state = state.state
            if current_state == "home":
                print(f"{PHONE_ENTITY_ID} is connected to local network (HOME)")
                # Do nothing when user is home
            elif current_state == "not_home":
                print(f"{PHONE_ENTITY_ID} is NOT connected (state: {current_state})")
                # If previous state was "home" and now "not_home", set home_mode to AWAY
                if previous_state == "home":
                    _LOGGER.info(f"User left home, setting home_mode to AWAY")
                    # Call the service asynchronously on the event loop
                    hass.async_create_task(
                        hass.services.async_call(
                            "input_select",
                            "select_option",
                            {
                                "entity_id": "input_select.home_mode",
                                "option": "AWAY",
                            },
                        )
                    )
            else:
                print(f"{PHONE_ENTITY_ID} is in state: {current_state}")

            previous_state = current_state
        await asyncio.sleep(5)


async def vacation_blink_lights(hass: HomeAssistant):
    BLINK_ENTITY = "input_boolean.blinking_lights"
    MODE_ENTITY = "input_select.home_mode"

    while True:
        mode_state = hass.states.get(MODE_ENTITY)
        if mode_state and mode_state.state == "VACATION":
            _LOGGER.info("Vacation mode active: starting blinking cycle")

            cycle_duration = 30
            blink_interval = 2
            elapsed = 0

            while elapsed < cycle_duration:
                # Toggle ON
                await hass.services.async_call(
                    "input_boolean",
                    "turn_on",
                    {"entity_id": BLINK_ENTITY},
                    blocking=True,
                )
                await asyncio.sleep(blink_interval)
                elapsed += blink_interval

                # Toggle OFF
                await hass.services.async_call(
                    "input_boolean",
                    "turn_off",
                    {"entity_id": BLINK_ENTITY},
                    blocking=True,
                )
                await asyncio.sleep(blink_interval)
                elapsed += blink_interval

            _LOGGER.info("Vacation blinking cycle complete, starting new cycle")
            # Continue immediately for the next cycle

        else:
            # Not vacation mode: ensure blinking lights are OFF
            current = hass.states.get(BLINK_ENTITY)
            if current and current.state == "on":
                await hass.services.async_call(
                    "input_boolean",
                    "turn_off",
                    {"entity_id": BLINK_ENTITY},
                    blocking=True,
                )
                _LOGGER.info("Vacation mode off: turned blinking lights OFF")
            await asyncio.sleep(5)  # check less frequently when not vacation


def detect(file_path):
    """
    Analyze an audio file for glass break sounds
    Returns True if glass break detected, False otherwise
    """
    try:
        # Load audio file
        y, sr = librosa.load(file_path, sr=None)

        # Convert to mono if stereo
        if len(y.shape) > 1:
            y = np.mean(y, axis=1)

        # Calculate short-time Fourier transform
        n_fft = min(2048, len(y))
        hop_length = n_fft // 4

        # Get spectrogram
        stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
        spectrogram = np.abs(stft)

        # Calculate RMS energy over time
        rms_energy = librosa.feature.rms(S=spectrogram)[0]
        max_energy = np.max(rms_energy)

        # Energy spike detection (transients)
        energy_diff = np.diff(rms_energy)
        energy_diff = np.append(energy_diff, 0)
        transient_score = np.max(energy_diff)

        # Extract high frequency energy (typically above 4kHz for glass breaks)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        high_freq_mask = freqs >= 4000
        high_freq_energy = np.sum(spectrogram[high_freq_mask, :], axis=0)
        high_freq_score = np.max(high_freq_energy)

        # Decision logic
        energy_condition = max_energy > 0.17889
        transient_condition = transient_score > 3181.55908
        high_freq_condition = high_freq_score > 0.09099

        # Combined decision
        is_glass_break = energy_condition and (
            transient_condition or high_freq_condition
        )

        return is_glass_break

    except Exception as e:
        print(f"Error processing audio file: {e}")
        return False


def handle_sound_detection(hass: HomeAssistant, event: Event) -> None:
    """Handle sound detection event."""

    entity_id = event.data.get("entity_id")

    if entity_id != "input_boolean.sound_toggle":
        return
    sensor_state = hass.states.get("input_boolean.sound_toggle")

    if sensor_state.state != "on":
        return

    # Check if the home mode is set to AWAY/VACATION/SLEEPING
    mode_state = hass.states.get("input_select.home_mode")
    if mode_state.state == "HOME":
        return

    audio_file = Path(__file__).parent / "./glass/glass_break.wav"
    if detect(audio_file):
        # Send the push notification
        push_notifier = hass.data[DOMAIN]["push_notifier"]
        push_notifier.send(
            title="🏠 Home Security Alert",
            message="A glass break was detected.",
        )
        # Send the email
        email_notifier = hass.data[DOMAIN]["email_notifier"]
        recipients = hass.data[DOMAIN]["email_recipients"]
        email_notifier.send(
            subject="🏠 Home Security Alert",
            message="A glass break was detected.",
            targets=recipients,
        )
        _LOGGER.info("Glass break detected!")
    else:
        _LOGGER.info("No glass break detected.")


def register_listeners(hass: HomeAssistant) -> None:
    global security_facade
    # create and configure the SecurityFacade
    security_facade = SecurityFacade(hass)

    # bind manual alarm_toggle flips into our Observer
    hass.bus.async_listen("state_changed", lambda e: _alarm_toggle_listener(hass, e))

    # register Home Assistant event listener
    hass.bus.async_listen(
        "state_changed", lambda event: handle_sensor_toggle_update(hass, event)
    )

    hass.bus.async_listen(
        "state_changed", lambda event: handle_sound_detection(hass, event)
    )

    hass.loop.create_task(phone_presence_check(hass))

    hass.loop.create_task(vacation_blink_lights(hass))

    # initialize UI toggle state
    initial = hass.states.get("input_boolean.sensor_toggle")
    init_state = initial.state if initial else "unknown"
    hass.states.async_set(
        "home_sec.sensor_toggle_state",
        init_state,
        {
            "friendly_name": "Sensor Toggle State",
            "original_entity": "input_boolean.sensor_toggle",
        },
    )
    # hass.states.async_set(
    #     VIDEO_ENTITY,
    #     "playing",
    #     {
    #         "friendly_name": "Security Cam Loop (off)",
    #         "video_path": str(VIDEO_FILE),
    #         "content_type": "video/mp4",
    #         "loop": True,
    #     },
    # )


# New listener to catch manual toggles of the alarm switch
@callback
def _alarm_toggle_listener(hass: HomeAssistant, event: Event) -> None:
    if event.data.get("entity_id") != "input_boolean.alarm_toggle":
        return
    new_state = event.data.get("new_state")
    notify = "DETECTED" if (new_state and new_state.state == "on") else "CLEAR"
    if security_facade:
        security_facade.sensor.notify(notify)
