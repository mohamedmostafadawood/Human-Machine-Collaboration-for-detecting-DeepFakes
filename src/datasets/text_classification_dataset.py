import datasets
from datasets import load_dataset, Dataset, concatenate_datasets
from torch.utils.data import DataLoader

from src.arguments.dataset_args import DatasetArgs
from src.arguments.env_args import EnvArgs

class TextClassificationDataset:

    def __init__(self, dataset_args: DatasetArgs, env_args: EnvArgs = None):
        self.dataset_args = dataset_args
        self.env_args = env_args or EnvArgs()

        # State variables
        self.ds = None

    def get_data_loader(self, partition='train'):
        return DataLoader(self.ds[partition], shuffle=True, batch_size=self.env_args.batch_size, drop_last=True)

    def load(self):
        """
        Loads the dataset, shuffles it, and creates a balanced subset with equal numbers of 'human' and 'ai' examples.
        """
        self.ds = load_dataset(
            "src/extern/ai_text_detection_pile",
            trust_remote_code=True
        )
        return self