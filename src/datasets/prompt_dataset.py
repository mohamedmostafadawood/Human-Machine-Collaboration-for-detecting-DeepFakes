from datasets import load_dataset

from src.arguments.dataset_args import DatasetArgs
from src.arguments.env_args import EnvArgs


class PromptDataset:

    def __init__(self, dataset_args: DatasetArgs, env_args: EnvArgs = None):
        self.dataset_args = dataset_args
        self.env_args = env_args or EnvArgs()

        # state variables
        self.ds = None


    def load(self):
        self.ds = load_dataset("sentiment140", split="train[:5%]")  # Using 5% for demonstration