"""A simple Gradio app to check the user's adherence to European values."""

import logging
from pathlib import Path

import cloudpickle
import gradio as gr
import pandas as pd
from datasets import Dataset, load_dataset
from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)


QUESTIONS_TO_INCLUDE = [
    "F025:1",
    "F025:5",
    "A124_09",
    "F025:3",
    "F118",
    "D081",
    "C001_01:1",
    "F122",
    "E025",
    "D059",
    "D054",
    "D078",
    "D026_05",
    "E069_01",
    "C041",
    "E003:4",
    "E116",
    "G007_36_B",
    "G007_35_B",
    "E228",
    "E001:2",
    "E265_08",
    "E114",
    "E265_01",
    "C039",
    "E233",
    "E233B",
    "G062",
    "E028",
    "E265_07",
    "E265_06",
    "E265_02",
    "A080_01",
    "E069_02",
    "A080_02",
    "G052",
    "E037",
    "A072",
    "G005",
    "G063",
    "A068",
    "A078",
    "A079",
    "E036",
    "A003",
    "G257",
    "D001_B",
    "F025:8",
    "F025:7",
    "E264:4",
    "A009",
    "E001:4",
    "F025:4",
]


def create_app() -> gr.Blocks:
    """Build the complete Gradio interface for the survey application.

    Returns:
        The constructed Gradio Blocks object, ready to be launched.
    """
    logger.info("Loading the survey questions...")
    dataset = load_dataset(
        "EuropeanValuesProject/za7505", name="en-clean", split="train"
    )
    assert isinstance(dataset, Dataset)
    df = dataset.to_pandas()
    assert isinstance(df, pd.DataFrame)
    df.set_index("question_id", inplace=True)

    # Replace "Mentioned" and "Not mentioned" with "Yes" and "No" respectively, in the
    # answer choices
    df.choices = df.choices.apply(
        lambda choices: {
            key: ("Yes" if value == "Mentioned" else "No")
            if value in ["Mentioned", "Not mentioned"]
            else value
            for key, value in choices.items()
        }
    )

    logger.info("Selecting the questions that we care about...")
    question_ids: list[str] = list()
    for question_id_and_choice in QUESTIONS_TO_INCLUDE:
        question_id = question_id_and_choice.split(":")[0]
        if question_id not in question_ids:
            question_ids.append(question_id)
    df = df.loc[question_ids]
    assert isinstance(df, pd.DataFrame)

    logger.info("Loading the pipeline used for scoring...")
    pipeline_dir = snapshot_download(repo_id="EuroEval/european-values-pipeline")
    with Path(pipeline_dir, "pipeline.pkl").open("rb") as f:
        pipeline = cloudpickle.load(f)

    with gr.Blocks(title="European Values Quiz") as demo:
        gr.Markdown(
            """
            # Are your values European?

            Test yourself to see if your values align with answers from tens of
            thousands of people from the EU.

            Please answer each question to the best of your ability.
            """
        )

        # This keeps track of which question the user is currently answering
        state_question_index = gr.State(value=0)

        # This keeps track of all the answers the user has given
        state_answers = gr.State(value=[])

        # Main group containing a single question and the answer choices
        with gr.Group() as quiz_group:
            question_text = gr.Markdown(line_breaks=True, height=125)
            answer_radio = gr.Radio(label="Select your answer:")

        # This group will be displayed when the quiz is done
        with gr.Group(visible=False) as results_group:
            gr.Markdown("## Survey Complete!\n\nThank you for your participation.\n\n")
            score_display = gr.Markdown()

        def get_score(answers_list: list[int]) -> float:
            """Calculate the score based on the user's answers.

            Args:
                answers_list:
                    The list of all the previous answers given by the user.

            Returns:
                The score as a float, which is the percentage of correct answers.
            """
            # Convert the answers list to match the questions we care about
            new_to_old_question_idx = {
                idx: question_ids.index(QUESTIONS_TO_INCLUDE[idx].split(":")[0])
                for idx in range(len(QUESTIONS_TO_INCLUDE))
            }
            new_answers_list = list()
            for new_idx, question_id_and_choice in enumerate(QUESTIONS_TO_INCLUDE):
                old_idx = new_to_old_question_idx[new_idx]
                if ":" not in question_id_and_choice:
                    new_answers_list.append(answers_list[old_idx])
                else:
                    choice_idx = question_id_and_choice.split(":")[1]
                    new_answers_list.append(
                        1 if answers_list[old_idx] == int(choice_idx) else 0
                    )

            # Compute the score using the pipeline
            return pipeline.transform([new_answers_list])[0].item()

        def process_answer(
            current_index: int, answers_list: list[int], current_answer: str
        ) -> tuple:
            """Process the user's answer and return updates for the Gradio UI.

            Args:
                current_index:
                    The index that we are currently processing.
                answers_list:
                    The list of all the previous answers given by the user.
                current_answer:
                    The answer that the user has given to the current question.

            Returns:
                A tuple, consisting of the following:

                - `state_question_index`, with the updated question index.
                - `state_answers`, with the updated list of answers.
                - `question_text`, with the next question.
                - `answer_radio`, with the choices for the next question.
                - `quiz_group`, with the configuration of the quiz group, which is the
                  block containing the current question. Only used to show/hide this
                  block.
                - `results_group`, with the configuration of the results group, being
                  the block containing the final results. Only used to show/hide this
                  block.
                - `score_display`, being the Markdown component in the results group
                  containing the actual score.
            """
            current_choices = df.iloc[current_index].choices
            answer_to_choice_idx = {
                choice: idx for idx, choice in current_choices.items()
            }
            answer_idx = answer_to_choice_idx[current_answer]
            logger.info(
                f"User answered question {current_index + 1} with choice "
                f"{current_answer} (index {answer_idx})"
            )

            answers_list.append(answer_idx)
            next_index = current_index + 1

            if next_index >= len(question_ids):
                final_score = get_score(answers_list=answers_list)
                return (
                    next_index,
                    answers_list,
                    gr.update(),
                    gr.update(),
                    gr.update(visible=False),
                    gr.update(visible=True),
                    f"### Your final score is: {final_score:.2%}",
                )
            else:
                next_question_data = df.iloc[next_index]
                return (
                    next_index,
                    answers_list,
                    (
                        f"### Question {1 + next_index}/{len(question_ids)}\n\n"
                        f"{next_question_data.question}"
                    ),
                    gr.update(
                        choices=[
                            choice
                            for choice in next_question_data.choices.values()
                            if choice is not None
                        ],
                        value=None,
                        label="Select your answer:",
                    ),
                    gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(),
                )

        def start_survey() -> tuple:
            """Prepare the initial state of the survey UI when the demo loads."""
            first_question_data = df.iloc[0]
            return (
                f"### Question 1/{len(question_ids)}\n\n{first_question_data.question}",
                gr.update(
                    choices=[
                        choice
                        for choice in first_question_data.choices.values()
                        if choice is not None
                    ],
                    label="Select your answer:",
                ),
            )

        # Add first question when the demo loads
        demo.load(fn=start_survey, inputs=[], outputs=[question_text, answer_radio])

        # Process answer when an answer is selected
        answer_radio.input(
            fn=process_answer,
            inputs=[state_question_index, state_answers, answer_radio],
            outputs=[
                state_question_index,
                state_answers,
                question_text,
                answer_radio,
                quiz_group,
                results_group,
                score_display,
            ],
        )

    return demo
