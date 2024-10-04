from dataclasses import dataclass, field


@dataclass
class ModelArgs:
    CONFIG_KEY = "model_args"
    """ This class contains all parameters for the diffusion model generator. """

    # The following fields denote the key names in a checkpoint file.
    MODEL_KEY = "language_model"  # state-dict
    MODEL_ARGS_KEY = "language_model_args"  # model args

    model_name: str = field(default="meta-llama/Llama-3.2-1B-Instruct", metadata={
        "help": "reference to the language model on the hub",
        "choices": ["meta-llama/Llama-3.2-1B-Instruct"]
    })
