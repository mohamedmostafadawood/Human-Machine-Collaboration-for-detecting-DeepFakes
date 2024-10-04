from src.arguments.env_args import EnvArgs
from src.arguments.model_args import ModelArgs


class ModelFactory:


    @staticmethod
    def from_model_args(model_args: ModelArgs, env_args: EnvArgs = None) -> None:
        """
        Instantiate an image generator. If watermarking keys are provided, we load a specific subclass of
        the generator that implements the watermarking method.
        """
        pass