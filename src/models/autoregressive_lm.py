from typing import List

from src.arguments.env_args import EnvArgs
from src.arguments.model_args import ModelArgs


class AutoregressiveLM:
    def __init__(self, model_args: ModelArgs, env_args: EnvArgs = None):

        self.env_args = env_args or EnvArgs()
        self.model_args = model_args


    def generate(self, prompt: str | List[str]):
        """ Generate text using an LM
        """
        pass
