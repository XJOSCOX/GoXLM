"""Launch the local GoXLM inference service on IPv4 loopback."""
import argparse
from pathlib import Path

import uvicorn

from .api import create_app
from .engine import Settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=Path(__file__).with_name("models.example.json"))
    args = parser.parse_args()
    try:
        app = create_app(Settings.model_validate_json(args.models.read_bytes()))
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    uvicorn.run(app, host="127.0.0.1", port=18000, access_log=False, log_level="warning",
                proxy_headers=False)


if __name__ == "__main__":
    main()
