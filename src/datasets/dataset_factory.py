from src.arguments.dataset_args import DatasetArgs
from src.arguments.env_args import EnvArgs
from src.datasets.prompt_dataset import PromptDataset
from src.datasets.text_classification_dataset import TextClassificationDataset


class DatasetFactory:


    @staticmethod
    def from_dataset_args(dataset_args: DatasetArgs, env_args: EnvArgs = None) -> PromptDataset | TextClassificationDataset:
        """
        Instantiate an autoregressive language Model
        """
        if dataset_args.ds_task in ["text-classification"]:
            if dataset_args.dataset_name in ["src/extern/ai_text_detection_pile"]:
                return TextClassificationDataset(dataset_args, env_args)
        else:
            raise NotImplemented
        raise ValueError(dataset_args.dataset_name)