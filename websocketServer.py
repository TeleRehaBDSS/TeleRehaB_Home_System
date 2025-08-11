import asyncio
import websockets
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

connected_clients = set()

async def register(websocket):
    connected_clients.add(websocket)
    print(f"Client connected: {websocket.remote_address} at {datetime.now()}")
    try:
        await websocket.wait_closed()
    finally:
        connected_clients.remove(websocket)
        print(f"Client disconnected: {websocket.remote_address} at {datetime.now()}")

async def handle_client(websocket, path):
    await register(websocket)
    while True:
        try:
            message = await websocket.recv()
            print(f"Received message -> {message}")
        except websockets.exceptions.ConnectionClosed:
            print(f"Connection closed: {websocket.remote_address}")
            break
        except Exception as e:
            print(f"Error receiving message: {e}")

async def heartbeat():
    while True:
        await asyncio.sleep(60)  # Heartbeat interval
        for websocket in list(connected_clients):
            try:
                await websocket.send('{"type": "heartbeat"}')
            except websockets.exceptions.ConnectionClosed:
                print(f"Connection closed during heartbeat: {websocket.remote_address}")
                connected_clients.remove(websocket)
            except Exception as e:
                print(f"Error sending heartbeat: {e}")
                connected_clients.remove(websocket)

async def start_websocket_server():
    print("Starting WebSocket server...")
    server = await websockets.serve(handle_client, "0.0.0.0", 8765)
    print("WebSocket server started")
    await asyncio.gather(server.wait_closed(), heartbeat())

def run_websocket_server():
    asyncio.run(start_websocket_server())
    