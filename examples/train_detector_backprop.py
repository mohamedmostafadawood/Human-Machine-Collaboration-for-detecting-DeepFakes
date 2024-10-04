import transformers

from src.arguments.config_args import ConfigArgs
from src.arguments.env_args import EnvArgs
from src.arguments.model_args import ModelArgs


def parse_args():
    parser = transformers.HfArgumentParser((ModelArgs,
                                            EnvArgs,
                                            ConfigArgs))
    return parser.parse_args_into_dataclasses()


def train_detector_backprop(model_args: ModelArgs,
           env_args: EnvArgs,
           config_args: ConfigArgs):
    """
    Train an LLM to detect text deepfakes generated with 2 different models with a 0/1 accuracy loss.

    """
    if config_args.exists():  # a configuration file was provided. yaml files always overwrite other settings!
        model_args = config_args.get_model_args()  # params to instantiate the generator.
        env_args = config_args.get_env_args()

    # reference training code here


if __name__ == "__main__":
    train_detector_backprop(*parse_args())