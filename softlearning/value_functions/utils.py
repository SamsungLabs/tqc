from copy import deepcopy

from softlearning.preprocessors.utils import get_preprocessor_from_params
from . import vanilla


def create_multiple_value_functions(value_fn, *args,  num_q, **kwargs):
    value_fns = tuple(value_fn(*args, **kwargs) for i in range(num_q))
    return value_fns


VALUE_FUNCTIONS = {
    'feedforward_V_function': (
        vanilla.create_feedforward_V_function),
    'double_feedforward_Q_function': lambda *args, **kwargs: (
        create_multiple_value_functions(
            vanilla.create_feedforward_Q_function, *args, **kwargs)),
}


def get_Q_function_from_variant(variant, env, *args, **kwargs):
    Q_params = variant['Q_params']
    Q_type = Q_params['type']
    Q_kwargs = deepcopy(Q_params['kwargs'])
    alg_type = variant['algorithm_params']['type']
    num_q = variant['algorithm_params']['kwargs']['num_q']
    if alg_type == 'TQC':
        num_outputs = variant['algorithm_params']['kwargs']['num_quantiles']
    else:
        num_outputs = 1

    preprocessor_params = Q_kwargs.pop('preprocessor_params', None)
    preprocessor = get_preprocessor_from_params(env, preprocessor_params)

    return VALUE_FUNCTIONS[Q_type](
        observation_shape=env.active_observation_shape,
        action_shape=env.action_space.shape,
        *args,
        num_q=num_q,
        observation_preprocessor=preprocessor,
        output_size=num_outputs,
        **Q_kwargs,
        **kwargs)


def get_V_function_from_variant(variant, env, *args, **kwargs):
    V_params = variant['V_params']
    V_type = V_params['type']
    V_kwargs = deepcopy(V_params['kwargs'])

    preprocessor_params = V_kwargs.pop('preprocessor_params', None)
    preprocessor = get_preprocessor_from_params(env, preprocessor_params)

    return VALUE_FUNCTIONS[V_type](
        observation_shape=env.active_observation_shape,
        *args,
        observation_preprocessor=preprocessor,
        **V_kwargs,
        **kwargs)
