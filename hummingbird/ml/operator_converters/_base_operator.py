from abc import ABC


class BaseOperator(ABC):
    """
    Abstract class definig the basic structure for operator implementations in Hummingbird.
    """

    def __init__(self):
        super().__init__()

        # An operator can be either a model or a transformer. In the latter case, self.transformer must be set to True.
        # In the former case, if a model is doing regression, then self.regression must be set to True. Otherwise the operator is doing some classificaiton task.
        self.regression = False
        self.transformer = False
