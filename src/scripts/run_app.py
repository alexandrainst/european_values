"""Run the simple Gradio app to check the user's adherence to European values."""

import logging

import click

from european_values.app import create_app

logging.basicConfig(
    format="%(asctime)s ⋅ %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO
)


@click.command()
@click.option(
    "--share", is_flag=True, help="Share the app publicly via Gradio's share link."
)
def main(share: bool) -> None:
    """Run the Gradio app."""
    app = create_app()
    app.launch(share=share)


if __name__ == "__main__":
    main()
