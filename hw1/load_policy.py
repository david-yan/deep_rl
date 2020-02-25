import pickle, tensorflow as tf, tf_util, numpy as np

def load_pickle_policy(filename):
    tf.compat.v1.disable_eager_execution()
    with open(filename, 'rb') as f:
        data = pickle.loads(f.read())

    # assert len(data.keys()) == 2
    nonlin_type = data['nonlin_type']
    policy_type = [k for k in data.keys() if k != 'nonlin_type'][0]
    activation = tf_util.lrelu if nonlin_type == 'lrelu' else tf.keras.activations.tanh

    assert policy_type == 'GaussianPolicy', 'Policy type {} not supported'.format(policy_type)
    policy_params = data[policy_type]

    assert set(policy_params.keys()) == {'logstdevs_1_Da', 'hidden', 'obsnorm', 'out'}

    # Keep track of input and output dims (i.e. observation and action dims) for the user

    def build_policy():
        # Build the policy. First, observation normalization.
        assert list(policy_params['obsnorm'].keys()) == ['Standardizer']
        assert list(policy_params['hidden'].keys()) == ['FeedforwardNet']
        layer_params = policy_params['hidden']['FeedforwardNet']
        # Initialize model with input layer
        model = tf.keras.Sequential()
        layer_keys = sorted(layer_params.keys())
        model.add(tf.keras.Input(layer_params[layer_keys[0]]['AffineLayer']['W'].shape[0], name='input'))
        for layer_name in layer_keys:
            l = layer_params[layer_name]
            model.add(tf.keras.layers.Dense(l['AffineLayer']['W'].shape[1], activation=activation, name=layer_name))
            layer = model.get_layer(name=layer_name)
            layer.set_weights([l['AffineLayer']['W'], l['AffineLayer']['b'][0,:]])

        # Output layer
        out = policy_params['out']
        model.add(tf.keras.layers.Dense(out['AffineLayer']['W'].shape[1], name='out'))
        output_layer = model.get_layer(name='out')
        output_layer.set_weights([out['AffineLayer']['W'], out['AffineLayer']['b'][0,:]])
        return model

    model = build_policy()
    return model

def init_empty_policy(filename):
    with open(filename, 'rb') as f:
        data = pickle.loads(f.read())

    # assert len(data.keys()) == 2
    nonlin_type = data['nonlin_type']
    policy_type = [k for k in data.keys() if k != 'nonlin_type'][0]
    activation = tf_util.lrelu if nonlin_type == 'lrelu' else tf.keras.activations.tanh

    assert policy_type == 'GaussianPolicy', 'Policy type {} not supported'.format(policy_type)
    policy_params = data[policy_type]

    assert set(policy_params.keys()) == {'logstdevs_1_Da', 'hidden', 'obsnorm', 'out'}

    # Keep track of input and output dims (i.e. observation and action dims) for the user

    def build_policy():
        # Build the policy. First, observation normalization.
        assert list(policy_params['obsnorm'].keys()) == ['Standardizer']
        assert list(policy_params['hidden'].keys()) == ['FeedforwardNet']
        layer_params = policy_params['hidden']['FeedforwardNet']
        # Initialize model with input layer
        model = tf.keras.Sequential()
        for layer_name in sorted(layer_params.keys()):
            l = layer_params[layer_name]
            model.add(tf.keras.layers.Dense(l['AffineLayer']['W'].shape[1], activation=activation))

        # Output layer
        model.add(tf.keras.layers.Dense(policy_params['out']['AffineLayer']['W'].shape[1]))
        return model

    model = build_policy()
    return model

