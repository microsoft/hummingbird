from hummingbird.common._registration import get_all_optimization_passes


class OptimizationPass:

    def optimize(self, topplogy):
        raise RuntimeError('Optimize method not implemented...')


def optimize(topology):
    """
    This function optimizes a given topology based on the registered optimization passes
    :param topology:
    :return:
    """
    continue_passes = True
    while continue_passes:
        continue_passes = False
        for _, optimization_pass in get_all_optimization_passes().items():
            topology.infer_all_shapes()
            continue_passes = optimization_pass().optimize(topology) or continue_passes
            topology.infer_all_shapes()

    return topology
