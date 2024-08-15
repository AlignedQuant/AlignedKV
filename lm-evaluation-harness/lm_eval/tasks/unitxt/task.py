"""
In the dynamic landscape of generative NLP, traditional text processing pipelines limit research flexibility and reproducibility, as they are tailored to specific dataset, task, and model combinations. The escalating complexity, involving system prompts, model-specific formats, instructions, and more, calls for a shift to a structured, modular, and customizable solution.

Addressing this need, we present Unitxt, an innovative library for customizable textual data preparation and evaluation tailored to generative language models. Unitxt natively integrates with common libraries like HuggingFace and LM-eval-harness and deconstructs processing flows into modular components, enabling easy customization and sharing between practitioners. These components encompass model-specific formats, task prompts, and many other comprehensive dataset processing definitions. The Unitxt-Catalog centralizes these components, fostering collaboration and exploration in modern textual data workflows. Beyond being a tool, Unitxt is a community-driven platform, empowering users to build, share, and advance their pipelines collaboratively.
"""

from functools import partial
from typing import Optional

import evaluatetask

from lm_eval.api.instance import Instance
from lm_eval.api.task import ConfigurableTask


_CITATION = """
@misc{bandel2024unitxt,
      title={Unitxt: Flexible, Shareable and Reusable Data Preparation and Evaluation for Generative AI},
      author={Elron Bandel and Yotam Perlitz and Elad Venezian and Roni Friedman-Melamed and Ofir Arviv and Matan Orbach and Shachar Don-Yehyia and Dafna Sheinwald and Ariel Gera and Leshem Choshen and Michal Shmueli-Scheuer and Yoav Katz},
      year={2024},
      eprint={2401.14019},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
"""


def score(items, metric):
    predictions, references = zip(*items)
    evaluator = evaluate.load("unitxt/metric")
    for reference in references:
        reference["metrics"] = [metric]
    results = evaluator.compute(predictions=predictions, references=references)
    return results[0]["score"]["global"]["score"]


class Unitxt(ConfigurableTask):
    VERSION = 0

    def __init__(
        self,
        config: Optional[dict] = None,
    ) -> None:
        assert "recipe" in config, "Unitxt task must have a 'recipe' string."
        super().__init__(
            config={
                "metadata": {"version": self.VERSION},
                "dataset_kwargs": {"trust_remote_code": True},
                "dataset_name": config["recipe"],
                "dataset_path": "unitxt/data",
            }
        )
        self.metrics = self.dataset["test"][0]["metrics"]

    def has_training_docs(self):
        return "train" in self.dataset

    def has_validation_docs(self):
        return "validation" in self.dataset

    def has_test_docs(self):
        return "test" in self.dataset

    def training_docs(self):
        return self.dataset["train"]

    def validation_docs(self):
        return self.dataset["validation"]

    def test_docs(self):
        return self.dataset["test"]

    def doc_to_text(self, doc):
        return doc["source"]

    def should_decontaminate(self):
        return False

    def doc_to_target(self, doc):
        doc["target"]

    def construct_requests(self, doc, ctx, **kwargs):
        """Uses RequestFactory to construct Requests and returns an iterable of
        Requests which will be sent to the LM.

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param ctx: str
            The context string, generated by fewshot_context. This includes the natural
            language description, as well as the few shot examples, and the question
            part of the document for `doc`.
        """

        return [
            Instance(
                request_type="generate_until",
                doc=doc,
                arguments=(ctx, {"until": ["\n"]}),
                idx=0,
                **kwargs,
            )
        ]

    def process_results(self, doc, results):
        """Take a single document and the LM results and evaluates, returning a
        dict where keys are the names of submetrics and values are the values of
        the metric for that one document

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param results:
            The results of the requests created in construct_requests.
        """

        continuation = results[0]

        predictions = continuation

        references = doc
        return {
            metric.replace("metrics.", ""): (predictions, references)
            for metric in self.metrics
        }

    def aggregation(self):
        """
        :returns: {str: [float] -> float}
            A dictionary where keys are the names of submetrics and values are
            functions that aggregate a list of metrics
        """
        return {
            metric.replace("metrics.", ""): partial(score, metric=metric)
            for metric in self.metrics
        }

    def higher_is_better(self):
        """
        :returns: {str: bool}
            A dictionary where keys are the names of submetrics and values are
            whether a higher value of the submetric is better
        """
        return {metric.replace("metrics.", ""): True for metric in self.metrics}
