"""Entry point for the student attention detection server."""

from __future__ import annotations

import argparse
import logging

from configs.config import DEFAULT_HOST, DEFAULT_PORT


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Student Attention Detection API Server",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=DEFAULT_HOST,
        help=f"Host to bind (default: {DEFAULT_HOST})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Port to bind (default: {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the ONNX emotion model",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["online", "face-to-face"],
        default="online",
        help="Operating mode (default: online)",
    )
    return parser.parse_args()


def main() -> None:
    """Initialise the app and start the development server."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    args = parse_args()

    # Import here to avoid circular imports at module level
    from src.api.app import create_app

    app = create_app(model_path=args.model_path)

    banner = (
        "\n"
        "============================================\n"
        "  Student Attention Detection Server\n"
        "============================================\n"
        f"  Host      : {args.host}\n"
        f"  Port      : {args.port}\n"
        f"  Model     : {args.model_path}\n"
        f"  Mode      : {args.mode}\n"
        f"  Dashboard : http://{args.host}:{args.port}/dashboard\n"
        "============================================\n"
    )
    print(banner)

    app.run(host=args.host, port=args.port, debug=True)


if __name__ == "__main__":
    main()
