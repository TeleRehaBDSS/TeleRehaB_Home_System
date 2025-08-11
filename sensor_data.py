
# sensor_data.py

import json

class SensorData:
    def __init__(self, device_mac_address, timestamp, w, x, y, z):
        self.device_mac_address = device_mac_address
        self.timestamp = timestamp
        self.w = w
        self.x = x
        self.y = y
        self.z = z

    def __str__(self):
        return f"SensorData(device_mac_address={self.device_mac_address}, timestamp={self.timestamp}, w={self.w}, x={self.x}, y={self.y}, z={self.z})"

    @staticmethod
    def from_json(json_str):
        data = json.loads(json_str)
        return SensorData(
            data["deviceMacAddress"],
            data["timestamp"],
            data["w"],
            data["x"],
            data["y"],
            data["z"]
        )

    def to_json(self):
        return json.dumps(self, default=lambda o: o.__dict__)
