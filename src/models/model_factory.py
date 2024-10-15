from src.arguments.env_args import EnvArgs
from src.arguments.model_args import ModelArgs
from src.models.autoregressive_lm import AutoregressiveLM


class ModelFactory:


    @staticmethod
    def from_model_args(model_args: ModelArgs, env_args: EnvArgs = None) -> AutoregressiveLM:
        """
        Instantiate an autoregressive language Model
        """
        if model_args.model_name in ["meta-llama/Llama-3.2-1B-Instruct"]:
            return AutoregressiveLM(model_args, env_args)
        raise ValueError(model_args.model_name)