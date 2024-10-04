from dataclasses import dataclass, field


@dataclass
class DatasetArgs:
    CONFIG_KEY = "dataset_args"
    """ This class contains all parameters for instantiating a Huggingface dataset. """


    dataset_name: str = field(default="artem9k/ai-text-detection-pil", metadata={
        "help": "reference to the language model on the hub",
        "choices": ["artem9k/ai-text-detection-pile"]
    })
