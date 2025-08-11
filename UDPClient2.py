import socket
import time
import FindMyIP as ip 
from shared_variables import MQTT_BROKER_HOST, MQTT_BROKER_PORT

from multiprocessing import Process

MULTICAST_GROUP = '224.1.1.1'
PORT = 10001
MESSAGE_PREFIX = 'NEW IP - '


def broadcast_ip(times=5, interval=2):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)

    local_ip = ip.internal()
    print(f"Broadcasting broker IP: {local_ip}")

    message = f"{MESSAGE_PREFIX}{MQTT_BROKER_HOST} - {local_ip}".encode()

    for i in range(times):
        sock.sendto(message, (MULTICAST_GROUP, PORT))
        print(f"[{i + 1}/{times}] Broadcasted: {message.decode()}")
        time.sleep(interval)

    sock.close()
    print("Broadcast complete.")