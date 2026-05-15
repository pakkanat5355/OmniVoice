import asyncio
import struct
import sys
import argparse
import websockets

async def ws_to_audiosocket(ws, writer):
    """Receive from WebSocket, send to AudioSocket"""
    try:
        while True:
            payload = await ws.recv()
            if isinstance(payload, bytes):
                # Type 0x10 = SLIN, Length = len(payload) (Big Endian)
                header = struct.pack(">BH", 0x10, len(payload))
                writer.write(header + payload)
                await writer.drain()
    except websockets.exceptions.ConnectionClosed:
        print("[Proxy] WebSocket closed")
    except Exception as e:
        print(f"[Proxy] Error in ws_to_audiosocket: {e}")
    finally:
        writer.close()

async def audiosocket_to_ws(reader, ws):
    """Receive from AudioSocket, send to WebSocket"""
    try:
        while True:
            header = await reader.readexactly(3)
            msg_type, length = struct.unpack(">BH", header)
            
            payload = await reader.readexactly(length)
            
            if msg_type == 0x01: # KIND_ID (UUID)
                print(f"[Proxy] Call connected! UUID: {payload.hex()}")
            elif msg_type == 0x10: # KIND_SLIN
                await ws.send(payload)
            elif msg_type == 0x00 or msg_type == 0x02: # KIND_HANGUP / KIND_ERROR
                print("[Proxy] AudioSocket hangup received.")
                break
    except asyncio.IncompleteReadError:
        print("[Proxy] AudioSocket connection closed")
    except Exception as e:
        print(f"[Proxy] Error in audiosocket_to_ws: {e}")
    finally:
        await ws.close()

async def handle_client(reader, writer, ws_url):
    print(f"[Proxy] New AudioSocket connection. Proxying to {ws_url}...")
    try:
        async with websockets.connect(ws_url) as ws:
            task1 = asyncio.create_task(ws_to_audiosocket(ws, writer))
            task2 = asyncio.create_task(audiosocket_to_ws(reader, ws))
            
            done, pending = await asyncio.wait(
                [task1, task2],
                return_when=asyncio.FIRST_COMPLETED
            )
            for p in pending:
                p.cancel()
    except Exception as e:
        print(f"[Proxy] Connection error: {e}")
    finally:
        writer.close()
        print("[Proxy] Session ended.")

async def main(port, ws_url):
    server = await asyncio.start_server(
        lambda r, w: handle_client(r, w, ws_url),
        '127.0.0.1', port
    )
    addrs = ', '.join(str(sockets.getsockname()) for sockets in server.sockets)
    print(f"[Proxy] Listening for Asterisk AudioSocket on {addrs}")
    print(f"[Proxy] Proxying to WebSocket {ws_url}")
    print(f"[Proxy] In Asterisk dialplan: exten => 100,1,AudioSocket(uuid, 127.0.0.1:{port})")
    
    async with server:
        await server.serve_forever()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Asterisk AudioSocket to WebSocket Proxy")
    parser.add_argument("--port", type=int, default=8003, help="Local TCP port for AudioSocket")
    parser.add_argument("--ws-url", type=str, required=True, help="WebSocket URL of the Voicebot (e.g. wss://xxx.ngrok.app/asterisk_ws)")
    
    args = parser.parse_args()
    
    try:
        asyncio.run(main(args.port, args.ws_url))
    except KeyboardInterrupt:
        print("Exiting...")
