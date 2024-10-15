from src.arguments.env_args import EnvArgs
from src.arguments.model_args import ModelArgs
from src.models.autoregressive_lm import AutoregressiveLM
from src.models.sequence_classifier_lm import SequenceClassifierLM


class ModelFactory:


    @staticmethod
    def from_model_args(model_args: ModelArgs, env_args: EnvArgs = None) -> AutoregressiveLM | SequenceClassifierLM:
        """
        Instantiate an autoregressive language Model
        """
        if model_args.task == "sequence-classification":
            if model_args.model_name in ["meta-llama/Llama-3.2-1B-Instruct"]:
                return SequenceClassifierLM(model_args, env_args)
        else:
            if model_args.model_name in ["meta-llama/Llama-3.2-1B-Instruct"]:
                return AutoregressiveLM(model_args, env_args)
        raise ValueError(model_args.model_name)