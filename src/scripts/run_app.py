"""Run the simple Gradio app to check the user's adherence to European values."""

import logging

from european_values.app import create_app

logging.basicConfig(
    format="%(asctime)s ⋅ %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO
)


def main() -> None:
    """Run the Gradio app."""
    app = create_app()
    app.launch()


if __name__ == "__main__":
    main()
