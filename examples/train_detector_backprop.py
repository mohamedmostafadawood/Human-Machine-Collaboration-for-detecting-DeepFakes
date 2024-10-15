import transformers

from src.arguments.config_args import ConfigArgs
from src.arguments.dataset_args import DatasetArgs
from src.arguments.env_args import EnvArgs
from src.arguments.model_args import ModelArgs
from src.arguments.optimizer_args import OptimizerArgs
from src.datasets.dataset_factory import DatasetFactory
from src.datasets.text_classification_dataset import TextClassificationDataset
from src.models.autoregressive_lm import AutoregressiveLM
from src.models.model_factory import ModelFactory
from src.models.sequence_classifier_lm import SequenceClassifierLM


def parse_args():
    parser = transformers.HfArgumentParser((ModelArgs,
                                            DatasetArgs,
                                            OptimizerArgs,
                                            EnvArgs,
                                            ConfigArgs))
    return parser.parse_args_into_dataclasses()


def train_detector_backprop(
            model_args: ModelArgs,
            dataset_args: DatasetArgs,
            opt_args: OptimizerArgs,
            env_args: EnvArgs,
            config_args: ConfigArgs):
    """
    Train an LLM to detect text deepfakes generated with 2 different models with a 0/1 accuracy loss.

    """
    if config_args.exists():  # a configuration file was provided. yaml files always overwrite other settings!
        model_args = config_args.get_model_args()  # params to instantiate the generator.
        dataset_args = config_args.get_dataset_args()
        opt_args = config_args.get_opt_args()
        env_args = config_args.get_env_args()

    dataset: TextClassificationDataset = DatasetFactory.from_dataset_args(dataset_args, env_args).load()
    sequence_classifier: SequenceClassifierLM = ModelFactory.from_model_args(model_args, env_args).load()

    print(sequence_classifier)
    sequence_classifier.fine_tune(dataset, opt_args=opt_args)



if __name__ == "__main__":
    train_detector_backprop(*parse_args())