import socket
import struct
import json
import FindMyIP as ip

def start_multicast_server(queueData):
    multicast_group = '224.1.1.1'
    server_address = ('', 10000)


    # Create the socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # Bind to the server address
    sock.bind(server_address)

    # Tell the operating system to add the socket to the multicast group on all interfaces
    group = socket.inet_aton(multicast_group)
    mreq = struct.pack('4sL', group, socket.INADDR_ANY)
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)

    print("Listening for multicast messages for data...")

    try:
        while True:
            data, address = sock.recvfrom(1024)
            json_data = json.loads(data.decode())
            queueData.put(json.dumps(json_data))  # Convert back to a JSON string
    finally:
        print('Closing socket')
        sock.close()

import socket
import json

def start_unicast_server(queueData):
    server_ip = ''  # Use '' to bind to all interfaces, or '192.168.x.x' for specific one
    server_port = 10000

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((server_ip, server_port))

    print(f"Listening for UDP messages on port {server_port}...")

    try:
        while True:
            data, address = sock.recvfrom(1024)
            try:
                json_data = json.loads(data.decode())
                queueData.put(json.dumps(json_data))  # Store as JSON string
                #print(f"Received from {address}: {json_data}")
            except json.JSONDecodeError:
                print(f"Recprintved non-JSON from {address}: {data}")
    finally:
        print('Closing socket')
        sock.close()
