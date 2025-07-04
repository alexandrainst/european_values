"""Extraction of survey questions.

Usage:
    uv run extract_questions_from_html.py --input INPUT_FILE --output OUTPUT_FILE
"""

import json
import logging
from pathlib import Path

import click

from european_values.question_extraction import process_survey_html

logger = logging.getLogger("extract_questions_from_html")


@click.command()
@click.option(
    "--input",
    required=True,
    type=click.Path(exists=True),
    help="Path to HTML survey description",
)
@click.option(
    "--output", required=True, type=click.Path(), help="Path to output JSON dictionary"
)
def main(input: str, output: str) -> None:
    """Main function to extract survey questions from an HTML file.

    Args:
        input:
            Path to the input HTML file containing the survey description.
        output:
            Path to the output JSON file where the extracted questions will be saved.
    """
    # Extract the questions from the HTML file
    logger.info(f"Processing HTML file: {input!r}")
    with Path(input).open(encoding="utf-8") as f:
        html_content = f.read()
    questions = process_survey_html(html_content=html_content)
    logger.info(f"Successfully extracted {len(questions):,} questions")

    # Store the questions in a JSON file
    with Path(output).open(mode="w", encoding="utf-8") as f:
        f.write(json.dumps(questions, indent=4, ensure_ascii=False))
    logger.info(f"Saved to: {output}")


if __name__ == "__main__":
    main()
