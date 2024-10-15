import transformers

from src.arguments.config_args import ConfigArgs
from src.arguments.dataset_args import DatasetArgs
from src.arguments.env_args import EnvArgs
from src.arguments.model_args import ModelArgs
from src.models.autoregressive_lm import AutoregressiveLM
from src.models.model_factory import ModelFactory


def parse_args():
    parser = transformers.HfArgumentParser((ModelArgs,
                                            DatasetArgs,
                                            EnvArgs,
                                            ConfigArgs))
    return parser.parse_args_into_dataclasses()


def train_detector_backprop(
            model_args: ModelArgs,
            dataset_args: DatasetArgs,
            env_args: EnvArgs,
            config_args: ConfigArgs):
    """
    Train an LLM to detect text deepfakes generated with 2 different models with a 0/1 accuracy loss.

    """
    if config_args.exists():  # a configuration file was provided. yaml files always overwrite other settings!
        model_args = config_args.get_model_args()  # params to instantiate the generator.
        dataset_args = config_args.get_dataset_args()
        env_args = config_args.get_env_args()

    dataset = PromptDataset
    language_model: AutoregressiveLM = ModelFactory.from_model_args(model_args, env_args)



if __name__ == "__main__":
    train_detector_backprop(*parse_args())