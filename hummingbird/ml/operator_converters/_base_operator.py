from abc import ABC


class BaseOperator(ABC):
    """
    Abstract class defining the basic structure for operator implementations in Hummingbird.
    """

    def __init__(self, regression=False, transformer=False, anomaly_detection=False, **kwargs):
        super().__init__()

        # An operator can be either a model or a transformer. In the latter case, self.transformer must be set to True.
        # In the former case, if a model is doing regression, then self.regression must be set to True. If it's doing
        # anomaly detection, then self.anomaly_detection must be set to True. Otherwise the operator is doing
        # some classificaiton task.
        self.regression = regression
        self.transformer = transformer
        self.anomaly_detection = anomaly_detection
