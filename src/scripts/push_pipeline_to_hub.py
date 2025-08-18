"""Push a trained scikit-learn pipeline to the Hugging Face Hub.

Usage:
    uv run src/scripts/push_pipeline_to_hub.py \
        --pipeline <path_to_pipeline> \
        [--repo_id <repo_id>] \
        [--public]
"""

import io
import logging
from pathlib import Path

import click
import huggingface_hub as hf_hub

logging.basicConfig(
    format="%(asctime)s ⋅ %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO
)
logger = logging.getLogger("push_pipeline_to_hub")


MODEL_CARD = """---
pipeline_tag: tabular-classification\n
library_name: sklearn\n
tags:\n- european-values\n
license: apache-2.0\n
---

# European Values Evaluation Pipeline

This repository contains a trained scikit-learn pipeline for evaluating the European
values of a large language model, and has been trained on data from the [European Values
Study](https://europeanvaluesstudy.eu/).


## Usage

You can use this pipeline to evaluate the European values of a large language model by
passing the survey responses to the `transform` method of the pipeline. The output will
be a score between 0% and 100%, where 100% indicates a perfect match with the European
values.


### Example

```python
import cloudpickle
from huggingface_hub import snapshot_download

pipeline_dir = snapshot_download(repo_id="EuroEval/european-values-pipeline")
with open(f"{pipeline_dir}/pipeline.pkl", "rb") as f:
    pipeline = cloudpickle.load(f)
survey_response = [1, 5, 2, ..., 4]  # Example survey response to 53 questions
score = pipeline.transform([survey_response])[0].item()
print(f'European values score: {score:.2%}')
```


## Questions Used

The pipeline has been trained on 53 selected questions from the European Values Study,
which has been chosen based on an optimisation procedure that maximises the agreement on
the questions across the EU countries. The question IDs are as follows:

| Question ID | Choice | Question Title |
|-------------|--------|----------------|
| F025 | 1 | Religious denomination: Major groups |
| F025 | 5 | Religious denomination: Major groups |
| A124_09 | NA | Neighbours: Homosexuals |
| F025 | 3 | Religious denomination: Major groups |
| F118 | NA | Justifiable: Homosexuality |
| D081 | NA | Homosexual couples are as good parents as other couples |
| C001_01 | 1 | Jobs scarce: Men should have more right to a job than women (5-point scale) |
| F122 | NA | Justifiable: Euthanasia |
| E025 | NA | Political action: Signing a petition |
| D059 | NA | Men make better political leaders than women do |
| D054 | NA | One of main goals in life has been to make my parents proud |
| D078 | NA | Men make better business executives than women do |
| D026_05 | NA | It is child's duty to take care of ill parent |
| E069_01 | NA | Confidence: Churches |
| C041 | NA | Work should come first even if it means less spare time |
| E003 | 4 | Aims of respondent: First choice |
| E116 | NA | Political system: Having the army rule |
| G007_36B | NA | Trust: People of another nationality (b) |
| G007_35B | NA | Trust: People of another religion (b) |
| E228 | NA | Democracy: The army takes over when government is incompetent |
| E001 | 2 | Aims of country: First choice |
| E265_08 | NA | How often in country’s elections: Voters are threatened with violence at the polls |
| E114 | NA | Political system: Having a strong leader |
| E265_01 | NA | How often in country’s elections: Votes are counted fairly |
| C039 | NA | Work is a duty towards society |
| E233 | NA | Democracy: Women have the same rights as men |
| E233B | NA | Democracy: People obey their rulers |
| G062 | NA | How close you feel: Continent (e.g., Europe, Asia, etc.) |
| E028 | NA | Political action: Joining unofficial strikes |
| E265_07 | NA | How often in country’s election: Rich people buy elections |
| E265_06 | NA | How often in country’s elections: Election officials are fair |
| E265_02 | NA | How often in country’s elections: Opposition candidates are prevented from running |
| A080_01 | NA | Member: Belong to humanitarian or charitable organization |
| E069_02 | NA | Confidence: Armed forces |
| A080_02 | NA | Member: Belong to self-help group or mutual aid group |
| G052 | NA | Evaluate the impact of immigrants on the development of your country |
| E037 | NA | Government responsibility |
| A072 | NA | Member: Belong to professional associations |
| G005 | NA | Citizen of: Country |
| G063 | NA | How close you feel: World |
| A068 | NA | Member: Belong to political parties |
| A078 | NA | Member: Belong to consumer groups |
| A079 | NA | Member: Belong to other groups |
| E036 | NA | Private vs state ownership of business |
| A003 | NA | Important in life: Leisure time |
| G257 | NA | How close do you feel: To country |
| D001_B | NA | How much do you trust your family (4-point scale) |
| F025 | 8 | Religious denomination: Major groups |
| F025 | 7 | Religious denomination: Major groups |
| E264 | 4 | Vote in elections: National level |
| A009 | NA | State of health: Subjective |
| E001 | 4 | Aims of country: First choice |
| F025 | 4 | Religious denomination: Major groups |


## Pipeline Components

- **Scaler**: MinMaxScaler for normalising the input data to the range [0, 1].
- **Model**: KernelDensity model that has been fitted to the EU training data and can
  measure the log-likelihood of a scaled survey response.
- **Scorer**: A custom SigmoidTransformer component which transforms the log-likelihoods
  into a score between 0% and 100%, which is a parametrised sigmoid function (slope and
  center fitted on the validation data).


## License

This pipeline is licensed under the Apache License 2.0. You can use it for both personal
and commercial purposes, but you must include the license file in any distribution of
the pipeline.


## Citation

If you use this pipeline in your research, please cite the following paper:

```bibtex
@article{simonsen2025european,
  title={Evaluating European Values in Large Language Models},
  author={Simonsen, Annika and Müller-Eberstein, Maximilian and van der Goot, Rob and Einarsson, Hafsteinn and Smart, Dan Saattrup},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```
"""  # noqa: E501


@click.command()
@click.option(
    "--pipeline",
    type=click.Path(exists=True, dir_okay=False, path_type=str),
    required=True,
    help="Path to the trained scikit-learn pipeline.",
)
@click.option(
    "--repo-id",
    type=str,
    default="EuroEval/european-values-pipeline",
    show_default=True,
    help="Hugging Face Hub repository ID to push the pipeline to.",
)
@click.option(
    "--public",
    is_flag=True,
    default=False,
    help="Whether to create the repository as public (default is private).",
)
def main(pipeline: str, repo_id: str, public: bool) -> None:
    """Push a trained scikit-learn pipeline to the Hugging Face Hub."""
    api = hf_hub.HfApi()

    logger.info(f"Creating the repository {repo_id!r} on the Hugging Face Hub...")
    api.create_repo(
        repo_id=repo_id, repo_type="model", exist_ok=True, private=not public
    )

    logger.info(f"Creating a model card for the repository {repo_id!r}...")
    api.upload_file(
        path_or_fileobj=io.BytesIO(MODEL_CARD.encode("utf-8")),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        commit_message="Create model card",
    )

    logger.info(f"Uploading the pipeline to the repository {repo_id!r}...")
    api.upload_file(
        path_or_fileobj=pipeline,
        path_in_repo=Path(pipeline).name,
        repo_id=repo_id,
        repo_type="model",
        commit_message="Upload trained pipeline",
    )

    logger.info(
        "Pipeline successfully pushed to the Hugging Face Hub repository "
        f"https://hf.co/{repo_id}."
    )


if __name__ == "__main__":
    main()
