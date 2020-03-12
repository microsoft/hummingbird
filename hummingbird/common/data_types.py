class DataType(object):
    def __init__(self, shape=None, doc_string=''):
        self.shape = shape
        self.doc_string = doc_string


class TensorType(DataType):

    def __init__(self, shape=None, doc_string='', denotation=None, channel_denotations=None):
        super(TensorType, self).__init__(
            [] if not shape else shape, doc_string)
        self.denotation = denotation
        self.channel_denotations = channel_denotations


class TensorListType(DataType):

    def __init__(self, shape=None, doc_string='', denotation=None, channel_denotations=None):
        super(TensorListType, self).__init__(
            [] if not shape else shape, doc_string)
        self.denotation = denotation
        self.channel_denotations = channel_denotations


class UInt8TensorType(TensorType):

    def __init__(self, shape=None, doc_string=''):
        super(UInt8TensorType, self).__init__(shape, doc_string)

    def __repr__(self):
        return "Uint8TensorType(shape={0})".format(self.shape)


class Int32TensorType(TensorType):

    def __init__(self, shape=None, doc_string=''):
        super(Int32TensorType, self).__init__(shape, doc_string)

    def __repr__(self):
        return "Int32TensorType(shape={0})".format(self.shape)


class Int64TensorType(TensorType):

    def __init__(self, shape=None, doc_string=''):
        super(Int64TensorType, self).__init__(shape, doc_string)

    def __repr__(self):
        return "Int64TensorType(shape={0})".format(self.shape)


class Float32TensorType(TensorType):
    def __init__(self, shape=None, color_space=None, doc_string='', denotation=None, channel_denotations=None):
        super(Float32TensorType, self).__init__(
            shape, doc_string, denotation, channel_denotations)
        self.color_space = color_space

    def __repr__(self):
        return "Float32TensorType(shape={0})".format(self.shape)


class UInt8TensorTypeList(TensorListType):
    def __init__(self, shape=None, color_space=None, doc_string='', denotation=None, channel_denotations=None):
        super(UInt8TensorTypeList, self).__init__(
            shape, doc_string, denotation, channel_denotations)
        self.color_space = color_space

    def __repr__(self):
        return "UInt8TensorTypeList(shape={0})".format(self.shape)
