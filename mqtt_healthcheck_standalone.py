#!/usr/bin/env python3
import paho.mqtt.client as mqtt
import time
import threading
import argparse
import sys
import subprocess
import os
import configparser
import logging
from datetime import datetime
from pathlib import Path

from shared_variables import (
    MQTT_BROKER_HOST, MQTT_BROKER_PORT, MQTT_KEEP_ALIVE_INTERVAL
)

# --- Logging: one file per start/boot ---
SCRIPT_DIR = Path(__file__).resolve().parent
LOG_DIR = SCRIPT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / f"heartbeat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()]
)
log = logging.getLogger("heartbeat")


class HeartbeatChecker:
    """
    Listens for pings on MQTT and replies with acks.
    If no ping arrives within `timeout`, powers off the machine.
    """

    def __init__(self, clinic_id, broker_host, broker_port=1883, timeout=120):
        self.ping_topic = f"heartbeat/{clinic_id}/ping"
        self.ack_topic = f"heartbeat/{clinic_id}/ack"
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.timeout = timeout

        self.client_is_alive = False
        self.timer = None

        self.client = mqtt.Client(client_id=f"heartbeat_server_{clinic_id}")
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message

    # ---- MQTT callbacks ----
    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            log.info(f"Connected to MQTT {self.broker_host}:{self.broker_port}")
            log.info(f"Subscribing to: {self.ping_topic}")
            self.client.subscribe(self.ping_topic, qos=1)
        else:
            log.error(f"Failed to connect, rc={rc}")
            sys.exit(1)

    def on_message(self, client, userdata, msg):
        payload = msg.payload.decode(errors="ignore")
        log.info(f"Ping on {msg.topic}: '{payload}'")
        # Reply ACK
        client.publish(self.ack_topic, payload="ack", qos=1)
        # Bump liveness timer
        self._reset_timeout()

    # ---- Timer / watchdog ----
    def _reset_timeout(self):
        if not self.client_is_alive:
            log.info("Client marked ONLINE")
            self.client_is_alive = True
        if self.timer:
            self.timer.cancel()
        self.timer = threading.Timer(self.timeout, self._mark_client_as_offline)
        self.timer.daemon = True
        self.timer.start()

    def _mark_client_as_offline(self):
        self.client_is_alive = False
        log.warning("-" * 60)
        log.warning(f"CRITICAL: Heartbeat timeout (> {self.timeout}s). Android app OFFLINE.")
        log.warning("-" * 60)
        if self.timer:
            self.timer.cancel()
            self.timer = None
        self._shutdown_system()

    # ---- Actions ----
    def _shutdown_system(self):
        if os.getenv("DRY_RUN", "0") == "1":
            log.info("[DRY_RUN] Would power off now.")
            return
        try:
            log.info("Powering off system...")
            # Requires passwordless sudo: youruser ALL=NOPASSWD: /bin/systemctl poweroff
            subprocess.run(["sudo", "systemctl", "poweroff"], check=False)
        except Exception as e:
            log.error(f"Shutdown failed: {e}")

    # ---- Lifecycle ----
    def run(self):
        log.info("Starting Heartbeat Checker...")
        try:
            self.client.connect(self.broker_host, self.broker_port, MQTT_KEEP_ALIVE_INTERVAL)
            # Start the timer upon connection attempt (will be refreshed on first ping)
            self._reset_timeout()
            self.client.loop_forever()
        except ConnectionRefusedError:
            log.error(f"Connection refused to {self.broker_host}:{self.broker_port}. Is the broker running?")
        except KeyboardInterrupt:
            self.stop()
        except Exception as e:
            log.error(f"Unexpected error: {e}")

    def stop(self):
        log.info("Stopping Heartbeat Checker...")
        if self.timer:
            self.timer.cancel()
            self.timer = None
        self.client.loop_stop()
        self.client.disconnect()
        log.info("Disconnected cleanly.")


def main():
    parser = argparse.ArgumentParser(description="MQTT Heartbeat Server for monitoring an application.")
    parser.add_argument("--timeout", type=int, default=90,
                        help="Seconds to wait before marking client as offline (default: 90).")
    args = parser.parse_args()

    # Read clinic_id from clinic.ini
    config = configparser.ConfigParser()
    ini_path = SCRIPT_DIR / "clinic.ini"
    if not ini_path.exists():
        log.error(f"clinic.ini not found at {ini_path}")
        sys.exit(1)

    config.read(ini_path)
    try:
        clinic_id = config["CLINIC"]["clinic_id"].strip().strip('"').strip("'")
        if not clinic_id:
            raise KeyError("clinic_id is empty")
    except KeyError:
        log.error("clinic.ini missing [CLINIC] section or non-empty clinic_id key.")
        sys.exit(1)

    api_config = configparser.ConfigParser()
    api_ini_path = SCRIPT_DIR / "config.ini"
    if not api_ini_path.exists():
        log.error(f"Required configuration file config.ini not found at {api_ini_path}")
        sys.exit(1)

    api_config.read(api_ini_path)
    try:
        key_edge = api_config["API"]["key_edge"].strip()
    except KeyError:
        log.error("config.ini is missing the [API] section or the key_edge key.")
        sys.exit(1)

    # Check for the special '00000' key to disable the service
    if key_edge == "00000":
        log.info("key_edge in config.ini is '00000'.")
        log.info("As per configuration, the heartbeat service will NOT start.")
        log.info("The system will not be monitored for shutdown.")
        sys.exit(0)  # Exit gracefully

    log.info("Valid key_edge found. Heartbeat service will proceed to start.")

    checker = HeartbeatChecker(
        clinic_id=clinic_id,
        broker_host=MQTT_BROKER_HOST,
        broker_port=MQTT_BROKER_PORT,
        timeout=args.timeout
    )

    try:
        checker.run()
    except KeyboardInterrupt:
        checker.stop()
        sys.exit(0)


if __name__ == "__main__":
    main()
# This script is intended to be run as a standalone service.
# It will listen for pings on the MQTT broker and respond with acks.
# If no ping is received within the specified timeout, it will power off the system.
# Make sure to run it with appropriate permissions to allow system shutdown.
# Example usage: python mqtt_healthcheck_standalone.py --timeout 120
# Ensure you have the required MQTT broker running and configured.
# The script uses a configuration file `clinic.ini` to read the clinic ID. 
