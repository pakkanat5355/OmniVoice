#!/usr/bin/env python3
"""
Start ngrok tunnel for Genesys AudioHook testing.

Usage:
    uv run python voicebot/start_ngrok.py --token YOUR_NGROK_TOKEN

Then copy the printed wss:// URL into Genesys Admin → Telephony → Audio Hooks.
"""
import argparse
import sys

from pyngrok import conf, ngrok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True, help="ngrok authtoken from https://dashboard.ngrok.com")
    parser.add_argument("--port", type=int, default=8002)
    args = parser.parse_args()

    conf.get_default().auth_token = args.token

    tunnel = ngrok.connect(args.port, "http")
    http_url: str = tunnel.public_url
    wss_url = http_url.replace("http://", "wss://").replace("https://", "wss://")

    print("\n" + "=" * 60)
    print("  ngrok tunnel is LIVE")
    print("=" * 60)
    print(f"  HTTP URL : {http_url}")
    print(f"  AudioHook: {wss_url}/audiohook")
    print("=" * 60)
    print("\nConfigure in Genesys Cloud:")
    print("  Admin → Telephony → Audio Hooks → Add")
    print(f"  URI: {wss_url}/audiohook")
    print("\nPress Ctrl+C to stop.\n")

    try:
        ngrok.run()   # blocks until interrupted
    except KeyboardInterrupt:
        print("\nStopping ngrok...")
        ngrok.disconnect(tunnel.public_url)
        ngrok.kill()


if __name__ == "__main__":
    main()
