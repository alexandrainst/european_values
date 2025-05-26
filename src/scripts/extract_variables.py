import argparse
import re
import json
from bs4 import BeautifulSoup
from typing import Dict, Any, Optional


SURVEYS = ['EVS5', 'WVS7']


def parse_arguments():
    parser = argparse.ArgumentParser(description="Extract Survey Questions")
    parser.add_argument('--input', required=True, help='path to HTML survey description')
    parser.add_argument('--output', required=True, help='path to output JSON dictionary')
    return parser.parse_args()


def extract_question_id(div_id: str) -> Optional[str]:
    """Extract question ID from div id attribute."""
    match = re.search(r'_Var([A-Za-z\d_]+)$', div_id)
    if match:
        return match.group(1)
    else:
        print(f"[Warning] Could not extract question ID from '{div_id}'.")
    return None


def extract_title(div: BeautifulSoup, question_id: str) -> Optional[str]:
    """Extract question title from the div."""
    title_div = div.find('div', class_='title')
    if title_div:
        h3_tags = title_div.find_all('h3')  # for some reason, there's always an empty h3
        for h3_tag in h3_tags:
            if h3_tag.find('a'):
                # Extract text after the dash
                full_text = h3_tag.find('a').get_text(strip=True)
                match = re.search(rf'^{re.escape(question_id)}\s*-\s*(.+)', full_text)
                if match:
                    return match.group(1)
                return match.group(1)
    return None


def extract_item(div: BeautifulSoup):
    item_div = div.find('div', class_='item')
    if item_div:
        item_text = item_div.get_text(strip=True)

        # extract item number
        item_number = None
        item_number_match = re.match(r'Item:?\s*(\d+)\s*-\s*', item_text)
        if item_number_match:
            item_number = item_number_match.group(1)
            item_text = item_text[:item_number_match.start()] + item_text[item_number_match.end():]
            item_text = item_text.strip()

        # check for question keys
        question_keys = {s:None for s in SURVEYS}
        question_key_match = re.search(rf'Master Question(?: in ({"|".join(SURVEYS)}) \(([^)]*)\);?)+:', item_text)
        question_key_matches = re.findall(rf'in ({"|".join(SURVEYS)}) \(([^)]*)\);?', item_text)
        if question_key_match:
            for survey, question_key in question_key_matches:
                question_keys[survey] = question_key
            # remove from question text
            item_text = item_text[:question_key_match.start()] + item_text[question_key_match.end():]
            item_text = item_text.strip()

        return question_keys, item_number, item_text
    return None, None, None


def extract_question_text(div):
    text = None

    details_div = div.find('div', class_='details')
    if details_div:
        question_text_div = details_div.find('div', class_='question_text')
        if question_text_div:
            span = question_text_div.find('span', class_='more')
            if span:
                text = span.get_text(strip=True)

    # Fallback: try abstract_preview
    if text is None:
        abstract_div = div.find('div', class_='abstract_preview')
        if abstract_div:
            span = abstract_div.find('span', class_='more_question')
            if span:
                text = span.get_text(strip=True)

    if text:
        # strip text
        text = re.sub(r'<br\s*/?>', '', text)
        text = re.sub(r'\n+', ' ', text)

        # extract question keys
        question_keys = None
        question_key_match = re.search(rf'Master Question(?: in ({"|".join(SURVEYS)}) \(([^)]*)\);?)+:', text)
        question_key_matches = re.findall(rf'in ({"|".join(SURVEYS)}) \(([^)]*)\);?', text)
        if question_key_match:
            question_keys = {s:None for s in SURVEYS}
            for survey, question_key in question_key_matches:
                question_keys[survey] = question_key
            # remove from question text
            text = text[:question_key_match.start()] + text[question_key_match.end():]
            text = text.strip()

        # split question into survey-specific versions
        # by default, both surveys have the same question text
        question_texts = {s:text for s in SURVEYS}
        survey_texts = re.match(rf'({"|".join(SURVEYS)}):(.*)'*len(SURVEYS), text)
        if survey_texts:
            for survey_idx, survey in enumerate(SURVEYS):
                survey = survey_texts.group(1 + (survey_idx*2))
                survey_text = survey_texts.group(1 + (survey_idx*2) + 1)
                question_texts[survey] = survey_text.strip()
        return question_keys, question_texts

    return None, None


def extract_pre_question_text(div: BeautifulSoup) -> Optional[str]:
    details_div = div.find('div', class_='details')
    if details_div:
        question_label_div = details_div.find('div', class_='question_label')
        if question_label_div:
            text = question_label_div.get_text(strip=True)
            # Remove the bold tag text
            text = re.sub(r'^Pre question text:\s*', '', text)
            text = re.sub(r'<br\s*/?>', '', text)
            return text
    return None


def extract_answers(div: BeautifulSoup) -> Dict[str, str]:
    answers = {}

    # Look for answer_categories div
    details_div = div.find('div', class_='details')
    if details_div:
        answer_div = details_div.find('div', class_='variable_code_list')
        if answer_div:
            table = answer_div.find('table', class_='variables_code_list_table')
            if table:
                tbody = table.find('tbody')
                if tbody:
                    rows = tbody.find_all('tr')
                    for row in rows:
                        cells = row.find_all('td')
                        if len(cells) >= 2:
                            value = cells[0].get_text(strip=True)
                            label = cells[1].get_text(strip=True)
                            # Clean up any HTML artifacts
                            label = re.sub(r'<[^>]+>', '', label)
                            # # convert if possible
                            # try:
                            #     value = int(value)
                            # except ValueError:
                            #     pass
                            # try:
                            #     label = int(label)
                            # except ValueError:
                            #     pass
                            answers[value] = label if label else None

    return answers


def extract_notes(div: BeautifulSoup) -> Optional[str]:
    details_div = div.find('div', class_='details')
    if details_div:
        notes_div = details_div.find('div', class_='notes')
        if notes_div:
            notes = re.sub(r'\s*Notes?\s*:\s*', '', notes_div.get_text(separator=' ', strip=True))
            return notes
    return None


def process_survey_html(html_content: str) -> Dict[str, Any]:
    """Process HTML content and extract survey questions."""
    soup = BeautifulSoup(html_content, 'html.parser')
    questions = {}

    # Find all div elements with exploredata IDs
    question_divs = soup.find_all('div', id=re.compile(r'exploredata-.*_Var.*'))

    print(f"Found {len(question_divs)} question div(s) to process")

    for div in question_divs:
        div_id = div.get('id', '')
        question_id = extract_question_id(div_id)

        if not question_id:
            print(f"[Warning]: Could not extract question ID from div id: {div_id}")
            continue

        print(f"Processing question: {question_id}")

        missing_components = []

        # Extract all components
        title = extract_title(div, question_id)
        if not title:
            missing_components.append("title")

        item_question_keys, item_number, item_text = extract_item(div)
        if not item_text:
            missing_components.append("item")
        if not item_number:
            missing_components.append("item_number")

        question_keys, question_texts = extract_question_text(div)
        question_keys = question_keys if question_keys else item_question_keys
        if not question_keys:
            missing_components.append("question_keys")
        if not question_texts:
            missing_components.append("question_texts")

        pre_question_text = extract_pre_question_text(div)
        if not pre_question_text:
            missing_components.append("pre_question_text")

        answers = extract_answers(div)
        if not answers:
            missing_components.append("answers")

        notes = extract_notes(div)
        if not notes:
            missing_components.append('notes')

        # post-processing for specific fields
        # cntry_AN (mapping stored in notes)
        if (question_id == 'cntry_AN') and (not answers):
            country_mapping = re.findall(r'([A-Z]{2})\s{2}([A-Za-z\s]+?)(?=\s+[A-Z]{2}|$)', notes)
            if country_mapping:
                answers = {code:country for code, country in country_mapping}

        if missing_components:
            print(f"  [Warning]: Missing components for {question_id}: {', '.join(missing_components)}")
        else:
            print(f"  All components extracted for {question_id}")

        # post-processing for survey-specific questions
        if any(question_id.endswith(f'_{survey}') for survey in SURVEYS):
            survey = question_id.split('_')[-1]
            if title:
                title = re.sub(rf'\s*\({survey}\)', '', title)
            if question_texts:
                question_keys = {s:t if s == survey else None for s, t in question_keys.items()}
                question_texts = {s:t if s == survey else None for s, t in question_texts.items()}


        # Build question dictionary
        question_data = {}
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

        print(f"  Extracted {len(answers)} answer options")

    return questions


def main():
    args = parse_arguments()

    with open(args.input, 'r', encoding='utf-8') as f:
        html_content = f.read()

    print(f"Processing HTML file: {args.input}")
    print("=" * 50)

    questions = process_survey_html(html_content)

    print("=" * 50)
    print(f"Successfully processed {len(questions)} question(s)")

    # export JSON
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(json.dumps(questions, indent=4, ensure_ascii=False))
    print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
