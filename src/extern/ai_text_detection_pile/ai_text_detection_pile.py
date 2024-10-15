# custom_ai_detection_dataset.py

from typing import Dict, Generator, Tuple

import datasets
from datasets import load_dataset, concatenate_datasets, Dataset


class CustomAIDetectionDataset(datasets.GeneratorBasedBuilder):
    """A wrapper around the AI Text Detection Pile dataset ensuring balanced 'human' and 'ai' examples."""

    VERSION = datasets.Version("1.0.0")
    _DESCRIPTION = "A custom wrapper for the ai-text-detection-pile dataset with balanced classes."
    _TEXT = "text"
    _SOURCE = "source"

    _URLS = {
        "url": "artem9k/ai-text-detection-pile"
    }

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="default",
            version=VERSION,
            description="Load 20k samples with equal 'human' and 'ai' examples."
        )
    ]
    DEFAULT_CONFIG_NAME = "default"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Dictionary to store balanced datasets for each split
        self.balanced_data = {
            "train": [],
            "test": [],
            "validation": []
        }

    def _info(self) -> datasets.DatasetInfo:
        """Defines the dataset's features."""
        return datasets.DatasetInfo(
            description=self._DESCRIPTION,
            features=datasets.Features({
                "id": datasets.Value("string"),
                self._TEXT: datasets.Value("string"),
                self._SOURCE: datasets.ClassLabel(names=["human", "ai"])
            }),
            supervised_keys=None,
            homepage="https://huggingface.co/artem9k/ai-text-detection-pile",
            citation="",
        )

    def _split_generators(self, dl_manager) -> list:
        """
        Defines the splits of the dataset.

        Splits:
            - train: 45% of the balanced data
            - test: 10% of the balanced data
            - validation: 45% of the balanced data
        """
        # Load the entire 'train' split from the dataset
        dataset = load_dataset(self._URLS["url"], split="train")

        # Shuffle the dataset to randomize the distribution of 'human' and 'ai' examples
        shuffled_dataset = dataset.shuffle(seed=42)

        # Filter the dataset into 'human' and 'ai' subsets
        human_ds = shuffled_dataset.filter(lambda example: example[self._SOURCE] == "human", num_proc=4)
        ai_ds = shuffled_dataset.filter(lambda example: example[self._SOURCE] == "ai", num_proc=4)

        # Determine the minimum count to ensure balance
        min_count = min(len(human_ds), len(ai_ds))
        if min_count == 0:
            raise ValueError("One of the classes ('human' or 'ai') has no examples in the split.")

        # Select the first 'min_count' examples from each subset
        balanced_human_ds = human_ds.select(range(min_count))
        balanced_ai_ds = ai_ds.select(range(min_count))

        # Concatenate the balanced subsets
        balanced_dataset = concatenate_datasets([balanced_human_ds, balanced_ai_ds])

        # Shuffle the balanced dataset to mix 'human' and 'ai' examples
        balanced_dataset = balanced_dataset.shuffle(seed=42)

        # Define split sizes
        total = len(balanced_dataset)
        train_size = int(0.45 * total)
        test_size = int(0.10 * total)
        validation_size = total - train_size - test_size  # Remaining examples

        # Split the dataset
        train_ds = balanced_dataset.select(range(train_size))
        test_ds = balanced_dataset.select(range(train_size, train_size + test_size))
        validation_ds = balanced_dataset.select(range(train_size + test_size, total))

        # Store the splits
        self.balanced_data["train"] = train_ds
        self.balanced_data["test"] = test_ds
        self.balanced_data["validation"] = validation_ds

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"split": "train"}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"split": "test"}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"split": "validation"}
            ),
        ]

    def _generate_examples(self, split: str) -> Generator[Tuple[str, Dict], None, None]:
        """
        Yields examples for the specified split.

        Args:
            split (str): One of 'train', 'test', 'validation'.

        Yields:
            Tuple[str, Dict]: A tuple of example ID and the example data.
        """
        split_dataset = self.balanced_data[split]
        for idx, example in enumerate(split_dataset):
            yield str(idx), {
                "id": str(idx),
                self._TEXT: example[self._TEXT],
                self._SOURCE: example[self._SOURCE]
            }