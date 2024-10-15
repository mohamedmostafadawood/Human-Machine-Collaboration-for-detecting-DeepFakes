from src.arguments.dataset_args import DatasetArgs
from src.arguments.env_args import EnvArgs
from src.datasets.prompt_dataset import PromptDataset


class DatasetFactory:


    @staticmethod
    def from_dataset_args(dataset_args: DatasetArgs, env_args: EnvArgs = None) -> PromptDataset:
        """
        Instantiate an autoregressive language Model
        """
        if dataset_args.dataset_name in ["meta-llama/Llama-3.2-1B-Instruct"]:
            return AutoregressiveLM(model_args, env_args)
        raise ValueError(model_args.model_name)