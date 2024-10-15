from dataclasses import dataclass, field

from importlib_metadata import metadata


@dataclass
class DatasetArgs:
    CONFIG_KEY = "dataset_args"
    """ This class contains all parameters for instantiating a Huggingface dataset. """


    dataset_name: str = field(default="src/extern/ai_text_detection_pile", metadata={
        "help": "reference to the language model on the hub",
        "choices": ["src/extern/ai_text_detection_pile"]
    })

    ds_task: str = field(default="text-classification", metadata={
        "help": "the task for which we load the text classification dataset",
        "choices": ["text-classification", "text-generation"]
    })
