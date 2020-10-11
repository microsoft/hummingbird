from abc import ABC


class BaseOperator(ABC):
    """
    Abstract class defining the basic structure for operator implementations in Hummingbird.
    """

    # Placeholder variable to capture the original input tensor(s) received by the forward method.
    input_x = None

    def __init__(self, regression=False, classification=False, transformer=False, anomaly_detection=False, input_indices=None, append_output=False, **kwargs):
        """
        Args:
            regression: Whether operator is a regression model.
            classification: Whether the operator is a classification model.
            transformer: Whether the operator is a feature transformer.
            anomaly_detection: Whether the operator is an anomaly detection model.
            input_indices: The column indices of the inputs for this operator that needs to be selected from the set of all inputs. Used for supporting Spark-ML.
            append_output: Whether the output of this operator has to be appended to the input in the final output. Used for supporting Spark-ML.
            kwargs: Other keyword arguments.
        """

        super().__init__()

        # An operator can be either a model or a transformer. In the latter case, self.transformer must be set to True.
        # In the former case, if a model is doing regression, then self.regression must be set to True.
        # Otherwise classification must be True.
        self.regression = regression
        self.classification = classification
        self.transformer = transformer
        self.anomaly_detection = anomaly_detection
        self.input_indices = input_indices
        self.append_output = append_output

    # Helper function to select the input for the current operator.
    def select_input_if_needed(self, x):
        if self.append_output:
            self.input_x = x

        if self.input_indices is not None and (isinstance(x, list) or isinstance(x, tuple)):
            if len(self.input_indices) > 1:
                x = tuple([x[i] for i in self.input_indices])
            else:
                x = x[self.input_indices[0]]
        elif isinstance(x, tuple) and len(x) == 1:
            x = x[0]

        return x

    # Helper function to append the output of the current operator to the input received. Used for Spark-ML conversion.
    def get_appended_output_if_needed(self, x):
        if self.append_output:
            if type(self.input_x) == tuple:
                self.input_x = self.input_x + (x,)
                return self.input_x
            else:
                return (self.input_x, x)
        else:
            return x
