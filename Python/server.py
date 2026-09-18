"""Executable entrypoint for the AutoPhotoEditor API."""

from __future__ import annotations

import argparse
import threading
import webbrowser

import uvicorn

from autophotoeditor.api.app import app


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the AutoPhotoEditor API server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--log-level", default="info")
    parser.add_argument("--no-browser", action="store_true", help="Do not open the API in a browser")
    args = parser.parse_args()

    if not args.no_browser:
        browser_host = "127.0.0.1" if args.host in {"0.0.0.0", "::"} else args.host
        threading.Timer(
            1.0,
            webbrowser.open,
            args=(f"http://{browser_host}:{args.port}/docs",),
        ).start()

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        reload=False,
    )


if __name__ == "__main__":
    main()
