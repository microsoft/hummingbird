from abc import ABC


class PhysicalOperator(ABC):
    """
    Abstract class defining the basic structure for operator implementations in Hummingbird.
    """

    def __init__(self, operator, regression=False, classification=False, transformer=False, anomaly_detection=False, **kwargs):
        """
        Args:
            regression: Whether operator is a regression model.
            classification: Whether the operator is a classification model.
            transformer: Whether the operator is a feature transformer.
            anomaly_detection: Whether the operator is an anomaly detection model.
            kwargs: Other keyword arguments.
        """

        super().__init__()
        # Get the base information from the operator.
        self.name = operator.full_name
        self.inputs = [input_.full_name for input_ in operator.inputs]
        self.outputs = [output_.full_name for output_ in operator.outputs]
        # An operator can be either a model or a transformer. In the latter case, self.transformer must be set to True.
        # In the former case, if a model is doing regression, then self.regression must be set to True.
        # Otherwise classification must be True.
        self.regression = regression
        self.classification = classification
        self.transformer = transformer
        self.anomaly_detection = anomaly_detection
