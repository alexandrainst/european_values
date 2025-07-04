"""Extraction of survey questions from HTML text."""

import logging
import re
import typing as t

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


SURVEYS = ["EVS5", "WVS7"]


def process_survey_html(html_content: str) -> dict[str, t.Any]:
    """Process HTML content and extract survey questions.

    Args:
        html_content:
            The HTML content as a string.

    Returns:
        A dictionary containing the extracted questions and their components.
    """
    soup = BeautifulSoup(html_content, "html.parser")
    questions = {}

    # Find all div elements with exploredata IDs
    question_divs = soup.find_all("div", id=re.compile(r"exploredata-.*_Var.*"))

    logger.info(f"Found {len(question_divs)} question div(s) to process")

    for div in question_divs:
        assert isinstance(div, BeautifulSoup), (
            "Expected `div` to be a BeautifulSoup object"
        )
        div_id = div.get(key="id", default="")
        assert isinstance(div_id, str), "Expected `div id` to be a string"
        question_id = extract_question_id(div_id=div_id)

        if not question_id:
            logger.warning(f"Could not extract question ID from div id: {div_id!r}")
            continue

        logger.info(f"Processing question: {question_id!r}")

        missing_components: list[str] = []

        # Extract all components
        title = extract_title(div=div, question_id=question_id)
        if not title:
            missing_components.append("title")

        item_question_keys, item_number, item_text = extract_item(div=div)
        if not item_text:
            missing_components.append("item")
        if not item_number:
            missing_components.append("item_number")

        question_keys, question_texts = extract_question_text(div=div)
        question_keys = question_keys if question_keys else item_question_keys
        if not question_keys:
            missing_components.append("question_keys")
        if not question_texts:
            missing_components.append("question_texts")

        pre_question_text = extract_pre_question_text(div=div)
        if not pre_question_text:
            missing_components.append("pre_question_text")

        answers = extract_answers(div=div)
        if not answers:
            missing_components.append("answers")

        notes = extract_notes(div=div)
        if not notes:
            missing_components.append("notes")

        # Post-processing for specific fields
        # cntry_AN (mapping stored in notes)
        if question_id == "cntry_AN" and not answers and notes:
            country_mapping = re.findall(
                pattern=r"([A-Z]{2})\s{2}([A-Za-z\s]+?)(?=\s+[A-Z]{2}|$)", string=notes
            )
            if country_mapping:
                answers = {code: country for code, country in country_mapping}

        if missing_components:
            logger.warning(
                f"Missing components for {question_id}: {', '.join(missing_components)}"
            )
        else:
            logger.info(f"All components extracted for {question_id}")

        # post-processing for survey-specific questions
        if any(question_id.endswith(f"_{survey}") for survey in SURVEYS):
            survey = question_id.split("_")[-1]
            if title:
                title = re.sub(rf"\s*\({survey}\)", "", title)
            if question_texts and question_keys:
                question_keys = {
                    s: t if s == survey else None for s, t in question_keys.items()
                }
                question_texts = {
                    s: t if s == survey else None for s, t in question_texts.items()
                }

        # Build question dictionary
        question_data: dict[str, t.Any] = {}
        if title:
            question_data["title"] = title
        if question_keys:
            question_data["key"] = question_keys
        if pre_question_text:
            question_data["preamble"] = pre_question_text
        if question_texts:
            question_data["text"] = question_texts
        if item_text:
            question_data["item"] = item_text
        if answers:
            question_data["answers"] = answers
        if notes:
            question_data["notes"] = notes

        questions[question_id] = question_data

        logger.info(f"Extracted {len(answers):,} answer options")

    return questions


def extract_question_id(div_id: str) -> str | None:
    """Extract question ID from div id attribute.

    Args:
        div_id:
            The id attribute of the div element.

    Returns:
        The extracted question ID if found, otherwise None.
    """
    match = re.search(r"_Var([A-Za-z\d_]+)$", div_id)
    if match:
        return match.group(1)
    else:
        logger.warning(f"Could not extract question ID from {div_id!r}.")
        return None


def extract_title(div: BeautifulSoup, question_id: str) -> str | None:
    """Extract question title from the div.

    Args:
        div:
            The BeautifulSoup object representing the div element.
        question_id:
            The ID of the question to match against the title.

    Returns:
        The extracted title if found, otherwise None.
    """
    title_div = div.find("div", class_="title")
    if title_div:
        assert isinstance(title_div, BeautifulSoup), (
            "Expected `title_div` to be a BeautifulSoup object"
        )
        h3_tags = title_div.find_all(
            "h3"
        )  # For some reason, there's always an empty h3
        for h3_tag in h3_tags:
            assert isinstance(h3_tag, BeautifulSoup), (
                "Expected `h3_tag` to be a BeautifulSoup object"
            )
            if anchor_tag := h3_tag.find("a"):
                # Extract text after the dash
                full_text = anchor_tag.get_text(strip=True)
                match = re.search(rf"^{re.escape(question_id)}\s*-\s*(.+)", full_text)
                if match:
                    return match.group(1)
                return None
    return None


def extract_item(
    div: BeautifulSoup,
) -> tuple[dict[str, str | None] | None, str | None, str | None]:
    """Extract a single item from a `div` HTML element.

    Args:
        div:
            The BeautifulSoup object representing the div element.

    Returns:
        A tuple containing:
        - A dictionary with survey keys and their corresponding question keys.
        - The item number if found, otherwise None.
        - The item text if found, otherwise None.
    """
    item_div = div.find("div", class_="item")
    if item_div:
        item_text = item_div.get_text(strip=True)

        # extract item number
        item_number = None
        item_number_match = re.match(r"Item:?\s*(\d+)\s*-\s*", item_text)
        if item_number_match:
            item_number = item_number_match.group(1)
            item_text = (
                item_text[: item_number_match.start()]
                + item_text[item_number_match.end() :]
            )
            item_text = item_text.strip()

        # check for question keys
        question_keys: dict[str, str | None] = {s: None for s in SURVEYS}
        question_key_match = re.search(
            rf"Master Question(?: in ({'|'.join(SURVEYS)}) \(([^)]*)\);?)+:", item_text
        )
        question_key_matches = re.findall(
            rf"in ({'|'.join(SURVEYS)}) \(([^)]*)\);?", item_text
        )
        if question_key_match:
            for survey, question_key in question_key_matches:
                question_keys[survey] = question_key
            # remove from question text
            item_text = (
                item_text[: question_key_match.start()]
                + item_text[question_key_match.end() :]
            )
            item_text = item_text.strip()

        return question_keys, item_number, item_text
    return None, None, None


def extract_question_text(
    div: BeautifulSoup,
) -> tuple[dict[str, str | None] | None, dict[str, str | None] | None]:
    """Extract the question text and keys from a `div` HTML tag.

    Args:
        div:
            The BeautifulSoup object representing the div element.

    Returns:
        A tuple containing:
        - A dictionary with survey keys and their corresponding question keys.
        - A dictionary with survey keys and their corresponding question texts.
    """
    text = None

    details_div = div.find("div", class_="details")
    if details_div:
        assert isinstance(details_div, BeautifulSoup), (
            "Expected `details_div` to be a BeautifulSoup object"
        )
        question_text_div = details_div.find("div", class_="question_text")
        if question_text_div:
            assert isinstance(question_text_div, BeautifulSoup), (
                "Expected `question_text_div` to be a BeautifulSoup object"
            )
            span = question_text_div.find("span", class_="more")
            if span:
                text = span.get_text(strip=True)

    # Fallback: try abstract_preview
    if text is None:
        abstract_div = div.find("div", class_="abstract_preview")
        if abstract_div:
            assert isinstance(abstract_div, BeautifulSoup), (
                "Expected `abstract_div` to be a BeautifulSoup object"
            )
            span = abstract_div.find("span", class_="more_question")
            if span:
                text = span.get_text(strip=True)

    if text:
        # strip text
        text = re.sub(r"<br\s*/?>", "", text)
        text = re.sub(r"\n+", " ", text)

        # extract question keys
        question_keys: dict[str, str | None] | None = None
        question_key_match = re.search(
            rf"Master Question(?: in ({'|'.join(SURVEYS)}) \(([^)]*)\);?)+:", text
        )
        question_key_matches = re.findall(
            rf"in ({'|'.join(SURVEYS)}) \(([^)]*)\);?", text
        )
        if question_key_match:
            question_keys = {s: None for s in SURVEYS}
            for survey, question_key in question_key_matches:
                question_keys[survey] = question_key
            # remove from question text
            text = text[: question_key_match.start()] + text[question_key_match.end() :]
            text = text.strip()

        # split question into survey-specific versions
        # by default, both surveys have the same question text
        question_texts: dict[str, str | None] = {s: text for s in SURVEYS}
        survey_texts = re.match(rf"({'|'.join(SURVEYS)}):(.*)" * len(SURVEYS), text)
        if survey_texts:
            for survey_idx, survey in enumerate(SURVEYS):
                survey = survey_texts.group(1 + (survey_idx * 2))
                survey_text = survey_texts.group(1 + (survey_idx * 2) + 1)
                question_texts[survey] = survey_text.strip()
        return question_keys, question_texts

    return None, None


def extract_pre_question_text(div: BeautifulSoup) -> str | None:
    """Extract the text preceeding a question in a `div` HTML tag.

    Args:
        div:
            The BeautifulSoup object representing the div element.

    Returns:
        The pre-question text if found, otherwise None.
    """
    details_div = div.find("div", class_="details")
    if details_div:
        assert isinstance(details_div, BeautifulSoup), (
            "Expected `details_div` to be a BeautifulSoup object"
        )
        question_label_div = details_div.find("div", class_="question_label")
        if question_label_div:
            text = question_label_div.get_text(strip=True)
            # Remove the bold tag text
            text = re.sub(r"^Pre question text:\s*", "", text)
            text = re.sub(r"<br\s*/?>", "", text)
            return text
    return None


def extract_answers(div: BeautifulSoup) -> dict[str, str | None]:
    """Extract all the answers within a `div` HTML tag.

    Args:
        div:
            The BeautifulSoup object representing the div element.

    Returns:
        A dictionary where keys are answer values and values are their corresponding
        labels.
    """
    answers: dict[str, str | None] = {}

    # Look for answer_categories div
    details_div = div.find("div", class_="details")
    if details_div:
        assert isinstance(details_div, BeautifulSoup), (
            "Expected `details_div` to be a BeautifulSoup object"
        )
        answer_div = details_div.find("div", class_="variable_code_list")
        if answer_div:
            assert isinstance(answer_div, BeautifulSoup), (
                "Expected `answer_div` to be a BeautifulSoup object"
            )
            table = answer_div.find("table", class_="variables_code_list_table")
            if table:
                assert isinstance(table, BeautifulSoup), (
                    "Expected `table` to be a BeautifulSoup object"
                )
                tbody = table.find("tbody")
                if tbody:
                    assert isinstance(tbody, BeautifulSoup), (
                        "Expected `tbody` to be a BeautifulSoup object"
                    )
                    rows = tbody.find_all("tr")
                    for row in rows:
                        assert isinstance(row, BeautifulSoup), (
                            "Expected `row` to be a BeautifulSoup object"
                        )
                        cells = row.find_all("td")
                        if len(cells) >= 2:
                            value = cells[0].get_text(strip=True)
                            label = cells[1].get_text(strip=True)
                            # Clean up any HTML artifacts
                            label = re.sub(r"<[^>]+>", "", label)
                            answers[value] = label if label else None
    return answers


def extract_notes(div: BeautifulSoup) -> str | None:
    """Extract additional notes related to a survey question from a `div` HTML tag.

    Args:
        div:
            The BeautifulSoup object representing the div element.

    Returns:
        The notes text if found, otherwise None.
    """
    details_div = div.find("div", class_="details")
    if details_div:
        assert isinstance(details_div, BeautifulSoup), (
            "Expected `details_div` to be a BeautifulSoup object"
        )
        notes_div = details_div.find("div", class_="notes")
        if notes_div:
            notes = re.sub(
                r"\s*Notes?\s*:\s*", "", notes_div.get_text(separator=" ", strip=True)
            )
            return notes
    return None
