from abc import ABC


class BaseOperator(ABC):
    """
    Abstract class defining the basic structure for operator implementations in Hummingbird.
    """

    def __init__(self, regression=False, classification=False, transformer=False, anomaly_detection=False, **kwargs):
        super().__init__()

        # An operator can be either a model or a transformer. In the latter case, self.transformer must be set to True.
        # In the former case, if a model is doing regression, then self.regression must be set to True.
        # Otherwise classification must be True.
        self.regression = regression
        self.classification = classification
        self.transformer = transformer
        self.anomaly_detection = anomaly_detection
