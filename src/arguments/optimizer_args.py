from dataclasses import dataclass, field

from importlib_metadata import metadata


@dataclass
class OptimizerArgs:
    CONFIG_KEY = "optimizer_args"
    """ This class contains all parameters for the optimizer. """

    # The following fields denote the key names in a checkpoint file.
    OPT_KEY = "optimizer_args"  # state-dict
    OPT_ARGS_KEY = "optimizer_args_key"  # model args

    opt_name: str = field(default="adamW", metadata={
        "help": "which optimizer to use",
        "choices": ["adamW"]
    })

    lr: float = field(default=1e-4, metadata={
        "help": "the learning rate"
    })

    weight_decay: float = field(default=0, metadata={
        "help": "the weight decay"
    })

