from typing import Optional

from homeassistant.core import HomeAssistant

from .model import YoloModelSingleton
from .sensors import Alarm, Notifier, Sensor

security_facade: Optional["SecurityFacade"] = None


class SecurityFacade:
    def __init__(self, hass: HomeAssistant):
        self.sensor = Sensor()
        self.sensor.register(Alarm(hass))
        self.sensor.register(Notifier())
        self.yolo = YoloModelSingleton()

    def process_frame(self, image_path: str) -> bool:
        detected = self.yolo.detect_person(image_path)
        state = "DETECTED" if detected else "CLEAR"
        self.sensor.notify(state)
        return detected
