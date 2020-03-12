import re

import numpy as np
from hummingbird.common.data_types import TensorType

from . import _registration
from ._container import Skl2PyTorchModel
from .exceptions import MissingConverter, MissingShapeCalculator

type_fct = type


class Variable:
    """
    Defines a variable which holds any data defined
    from data types.
    """

    def __init__(self, raw_name, pytorch_name, type=None):
        """
        :param raw_name: A string indicating the variable's name in the
                         original model. Usually, it's the seed string
                         used to created its PyTorch name (i.e., the
                         field *pytorch_name* below).
        :param pytorch_name: A string indicating the variable's name in
                          the converted model
        :param type: A type object defined in .common.data_types.py;
                     e.g., FloatTensorType
        """

        self.raw_name = raw_name
        self.pytorch_name = pytorch_name
        self.type = type

        # The following fields are bool variables used in parsing and
        # compiling stages
        self.is_fed = None
        self.is_root = None
        self.is_leaf = None
        self.is_abandoned = False

        if isinstance(self.type, TensorType):
            shape = self.type.shape
            if not isinstance(shape, (list, tuple)):
                try:
                    shape = list(shape)
                except TypeError:
                    raise TypeError("shape must be a tuple or a list not "
                                    "{}.".format(type_fct(shape)))
            for dim in shape:
                if dim is None:
                    continue
                if not isinstance(dim, (int, np.int32, np.int64)):
                    raise TypeError("shape must contains integers not "
                                    "'{}'.".format(dim))

    @property
    def full_name(self):
        """
        Return a globally unique variable ID
        """
        return self.pytorch_name

    def __repr__(self):
        return ("Variable(raw_name='{0}', pytorch_name='{1}', type={2})".format(
            self.raw_name, self.pytorch_name, self.type))


class Operator():
    """
    Defines an operator available in PyTorch.
    """

    def __init__(self, pytorch_name, type, raw_operator):
        """
        :param pytorch_name: A unique ID, which is a string
        :param type: A object which uniquely characterizes the type of
                     this operator. For example, it can be a string.
        :param raw_operator: The original operator which defines this operator;
                             for example, a scikit-learn Imputer.F
        """

        if isinstance(raw_operator, str):
            raise RuntimeError("Parameter raw_operator must be an object not "
                               "a string '{0}'.".format(raw_operator))
        # operator name in the converted model
        self.pytorch_name = pytorch_name
        self.type = type
        self.raw_operator = raw_operator

        self.inputs = []
        self.outputs = []

        self.is_evaluated = None
        self.is_abandoned = False

    @property
    def full_name(self):
        """
        Return a globally unique operator ID
        """
        return self.pytorch_name

    @property
    def input_full_names(self):
        """
        Return all input variables' names
        """
        return [variable.full_name for variable in self.inputs]

    @property
    def output_full_names(self):
        """
        Return all output variables' names
        """
        return [variable.full_name for variable in self.outputs]

    @property
    def original_operator(self):
        """
        Return the original operator/layer
        """
        return self.raw_operator


class Topology:
    """
    Holds instances on :class:`Scope <hummingbird.common._topology.Scope>` and
    :class:`SklearnModelContainer <hummingbird.common._container.SklearnModelContainer>`.
    These are filled by the converters while a pipeline is being converted.
    When all converters were called, method
    :meth:`Topology.compile <hummingbird.common._topology.Topology.compile>`
    must be called to convert the topological graph into PyTorch model.
    """  # noqa

    def __init__(self, model, default_batch_size=1, initial_types=None, variable_name_set=None, operator_name_set=None):
        """
        Initializes a *Topology* object, which is an intermediate
        representation of a computational graph.
        :param model: RawModelContainer object or one of its derived
                      classes. It contains the original model.
        :param default_batch_size: batch_size prepend to scalar and
                                   array types. It's usually 1 or None.
        :param initial_types: A list providing some types for some
                              root variables. Each element is a tuple
                               of a variable name and a type defined
                               in *data_types.py*.
        """
        # self.scopes = []
        self.raw_model = model
        # self.scope_names = set()
        self.variable_name_set = set()
        self.operator_name_set = set()
        self.initial_types = initial_types if initial_types else list()
        self.default_batch_size = default_batch_size

        self.pytorch_variable_names = (
            variable_name_set if variable_name_set is not None else set())
        self.pytorch_operator_names = (
            operator_name_set if operator_name_set is not None else set())

        # An one-to-many map from raw variable name to PyTorch variable
        # names. It looks like
        # (key, value) = (raw_name, [pytorch_name, pytorch_name1, pytorch_name2, ..., pytorch_nameN]) # noqa
        # The last name may hide all other names in this scope.
        self.variable_name_mapping = {}

        # A map of local variables defined in this scope.
        # (key, value) = (pytorch_name, variable)
        self.variables = {}

        # A map of local operators defined in this scope.
        # (key, value) = (pytorch_name, operator)
        self.operators = {}

        # This attribute is used in optimizing the graph structure. If
        # root_names is not empty, only the variables specified will be
        # treated as the roots (i.e., set is_fed to True in the
        # beginning of a graph evaluation) of the graph. Specifying all
        # root variables in this list and leaving it empty are
        # equivalent. This attribute directly affects
        # _initialize_graph_status_for_traversing function and
        # indirectly affects _infer_all_shapes and _prune functions.
        self.root_names = list()

    @staticmethod
    def _generate_unique_name(seed, existing_names):
        """
        Produce an unique string based on the seed
        :param seed: a string
        :param existing_names: a set containing strings which cannot be
                               produced
        :return: a string similar to the seed
        """
        if seed == '':
            raise ValueError('Name seed must be a non-empty string.')

        # Make the seed meet C-style naming convention
        # Only alphabets and numbers are allowed
        seed = re.sub('[^0-9a-zA-Z]', '_', seed)

        # The first symbol cannot be a number
        if re.match('^[0-9]', seed):
            seed = '_' + seed

        # If seed has never been seen, we return it as it is. Otherwise,
        # we will append an number to make it unique.
        if seed not in existing_names:
            existing_names.add(seed)
            return seed
        else:
            i = 1
            while seed + str(i) in existing_names:
                i += 1
            new_name = seed + str(i)
            existing_names.add(new_name)
            return new_name

    def get_unique_scope_name(self, seed):
        return Topology._generate_unique_name(seed, self.scope_names)

    def declare_variable(self, raw_name, type=None, prepend=False):
        """
        This function may create a new variable in this scope. If
        *raw_name* has been used to create other variables, the new
        variable will hide all other variables created using *raw_name*.
        """
        # Get unique ID for the new variable
        pytorch = self.get_unique_variable_name(raw_name)

        # Create the variable
        variable = Variable(raw_name, pytorch, type)
        self.variables[pytorch] = variable

        if raw_name in self.variable_name_mapping:
            # Hide existing variables with the same raw_name
            if not prepend:
                self.variable_name_mapping[raw_name].append(pytorch)
            else:
                self.variable_name_mapping[raw_name].insert(0, pytorch)
        else:
            self.variable_name_mapping[raw_name] = [pytorch]
        return variable

    def delete_variable(self, pytorch_name):
        """
        Removes the variable whose *pytorch_name* is the input *pytorch_name*.
        """
        if (pytorch_name not in self.pytorch_variable_names or
                pytorch_name not in self.variables):
            raise RuntimeError('The variable to remove was not found.')
        self.pytorch_variable_names.discard(pytorch_name)
        raw_name = self.variables[pytorch_name].raw_name
        self.variable_name_mapping[raw_name].remove(pytorch_name)
        del self.variables[pytorch_name]

    def get_unique_variable_name(self, seed):
        """
        Creates a unique variable ID based on the given seed.
        """
        if not isinstance(seed, str):
            raise TypeError("Parameter seed must be a string not {}."
                            "".format(type(seed)))
        name = Topology._generate_unique_name(
            seed, self.pytorch_variable_names)
        return name

    def delete_operator(self, pytorch_name):
        """
        Removes the operator whose pytorch_name is the input *pytorch_name*.
        """
        if (pytorch_name not in self.pytorch_operator_names or
                pytorch_name not in self.operators):
            raise RuntimeError('The operator to remove was not found.')

        self.pytorch_operator_names.discard(pytorch_name)
        del self.operators[pytorch_name]

    def declare_operator(self, type, raw_model=None):
        """
        This function is used to declare new local operator.
        """
        pytorch_name = self.get_unique_operator_name(str(type))
        operator = Operator(pytorch_name, type, raw_model)
        self.operators[pytorch_name] = operator
        return operator

    def get_op_producing_variable(self, variable):
        """
        This function is used to find the operator which produces the given variable
        """
        for operator in self.operators.values():
            if any([variable.pytorch_name == x.pytorch_name for x in operator.outputs]):
                return operator
        return None

    def get_ops_consuming_variable(self, variable):
        """
        This function is used to find the operators which consumes the given variable
        """
        return [op for op in self.operators.values() if
                any(variable.pytorch_name == var.pytorch_name for var in op.inputs)]

    def get_unique_operator_name(self, seed):
        """
        Creates a unique operator ID based on the given seed.
        """
        return Topology._generate_unique_name(seed, self.pytorch_operator_names)

    def unordered_operator_iterator(self):
        for operator in self.operators.values():
            yield operator

    def unordered_variable_iterator(self):
        for variable in self.variables.values():
            yield variable

    def topological_operator_iterator(self):
        """
        This is an iterator of all operators in Topology object.
        Operators may be produced in a topological order. If you want to
        simply go though all operators without considering their
        topological structure, please use another function,
        unordered_operator_iterator.
        """
        self._initialize_graph_status_for_traversing()

        while not all(operator.is_evaluated for operator in self.operators.values()):
            is_evaluation_happened = False
            for operator in self.unordered_operator_iterator():
                if (all(variable.is_fed for variable in operator.inputs)
                        and not operator.is_evaluated):
                    # Check if over-writing problem occurs (i.e., multiple
                    # operators produce results on one variable).
                    for variable in operator.outputs:
                        # Throw an error if this variable has been treated as
                        # an output somewhere
                        if variable.is_fed:
                            raise RuntimeError(
                                'One variable can only be assigned once.')
                        # Mark this variable as filled
                        variable.is_fed = True
                    # Make this operator as handled
                    operator.is_evaluated = True
                    is_evaluation_happened = True
                    # Send out an operator
                    yield operator
            # After scanning through the whole computational graph, at
            # least one operator should be evaluated. If not, we need
            # to terminate this procedure to avoid dead lock.
            if not is_evaluation_happened:
                break

    def _initialize_graph_status_for_traversing(self):
        """
        Initialize the status of all variables and operators for
        traversing the underline graph
        """
        # In the beginning, we set is_root and is_leaf true. For is_fed,
        # we have two different behaviors depending on whether
        # root_names is empty.
        for variable in self.unordered_variable_iterator():
            # If root_names is set, we only set those variable to be
            # fed. Otherwise, all roots would be fed.
            variable.is_fed = (False if self.root_names and variable.pytorch_name
                               not in self.root_names else True)
            variable.is_root = True
            variable.is_leaf = True

        # Then, we flip some flags by applying some simple rules so
        # that only
        #   1. all roots get is_root=True and is_fed=True
        #   2. all leaves get is_leaf=True

        for operator in self.unordered_operator_iterator():
            # All operators are not processed in the beginning
            operator.is_evaluated = False
            for variable in operator.outputs:
                # Output cannot be fed before graph traversing
                variable.is_fed = False
                # If the variable is an output of one operator,
                # it must not be a root
                variable.is_root = False

            for variable in operator.inputs:
                # If the variable is an input of one operator,
                # it must not be a leaf
                variable.is_leaf = False

    def infer_all_shapes(self):
        """
        Infer all variables' shapes in the computational graph.
        """
        self._initialize_graph_status_for_traversing()

        # Deliver user-specified types to root variables
        for raw_name, initial_type in self.initial_types:
            # Check all variables declared using raw_name in
            # the whole graph

            # Assign initial_type to all variables declared using
            # raw_name
            for pytorch_name in self.variable_name_mapping[raw_name]:
                variable = self.variables[pytorch_name]
                if variable.is_root:
                    # Assign type to the root; existing type
                    # produced by parser may be overwritten
                    variable.type = initial_type

        # Traverse the graph from roots to leaves
        for operator in self.topological_operator_iterator():

            try:
                _registration.get_shape_calculator(operator.type)(operator)
            except ValueError:
                raise MissingShapeCalculator(
                    "Unable to find shape calculator for alias '{}' type "
                    "'{}'. You may raise an issue at "
                    "https://cisl.visualstudio.com/Hummingbird/issues."
                    "".format(operator.type,
                              type(getattr(operator, 'raw_model', None))))


def convert_topology(topology, device=None, extra_config={}):
    """
    This function is used to convert our Topology object defined in
    _parser.py into a PyTorch model.
    :param topology: The Topology object we are going to convert
    :param device: torch.device which device to translate the model
    :param extra_config: Extra configurations to be used by individual operator converters
    :return: a PyTorch model
    """

    operator_map = {}
    for operator in topology.topological_operator_iterator():
        try:
            converter = _registration.get_converter(operator.type)
            operator_map[operator.full_name] = converter(
                operator, device, extra_config)
        except ValueError:
            raise MissingConverter(
                "Unable to find converter for alias '{}' type "
                "'{}'. You may raise an issue at "
                "https://cisl.visualstudio.com/Hummingbird/issues."
                "".format(operator.type,
                          type(getattr(operator, 'raw_model', None))))

    pytorch_model = Skl2PyTorchModel(topology.raw_model.input_names, topology.raw_model.output_names, operator_map,
                                     topology, device, extra_config)
    return pytorch_model
