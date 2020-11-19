import torch


def get_linear_layers(net):
    linear_layers = []
    for child in net.children():
        if type(child) == torch.nn.Linear:
            linear_layers.append(child)
    if len(linear_layers) <= 0:
        linear_layers = [net]
    return linear_layers


def count_neurons(net, return_layers=False):
    first = None
    extra = 0
    last = 0
    layers = 0
    for layer in get_linear_layers(net):
        layers += 1
        in_neurons = layer.in_features
        if first is None:
            first = in_neurons
        else:
            extra += in_neurons 
        last = layer.out_features

    if return_layers:
        return first, extra, last, layers
    return first, extra, last


def import_ffnn(ffnn, activation):
    """Ingest a FFNN into the PAGNN architecture"""

    # create equivalent adjacency structure
    first, extra, last, layers = count_neurons(ffnn, return_layers=True)
    pagnn = PAGNNLayer(first, last, extra, steps=layers, activation=activation, retain_state=False)

    # import synaptic weightings
    pagnn.zero_params()
    last_i = 0
    seen_output_neurons = 0
    for i, layer in enumerate(get_linear_layers(ffnn)):
        in_neurons = layer.in_features
        out_neurons = layer.out_features

        pW = pagnn.weight
        pb = pagnn.bias
        lW = layer.weight
        lb = layer.bias
        
        new_last_i = last_i + in_neurons

        pW.data[last_i:last_i+in_neurons, new_last_i:new_last_i+out_neurons] = lW.T
        if lb is not None:
            pb.data[new_last_i:new_last_i+out_neurons] = lb

        last_i = new_last_i

        seen_output_neurons += out_neurons

    return pagnn


def _pagnn_op(state, weight, bias=None):
    if state.dim() == 2 and bias is not None:
        # fused op is marginally faster
        ret = torch.addmm(bias, state, weight)
    else:
        output = state.matmul(weight)
        if bias is not None:
            output += bias
        ret = output
    # return state.matmul(weight)
    return ret


class PAGNNLayer(torch.nn.Module):
    def __init__(self, input_neurons, output_neurons, extra_neurons, steps=1, sparsity=0, retain_state=True, activation=None, sequence_inputs=False):
        super(PAGNNLayer, self).__init__()

        assert input_neurons > 0 
        assert output_neurons > 0 
        assert extra_neurons >= 0 
        assert sparsity >= 0 
        assert steps >= 1

        if sparsity > 0:
            raise NotImplemented()

        self._total_neurons = input_neurons + output_neurons + extra_neurons
        self._input_neurons = input_neurons
        self._output_neurons = output_neurons
        self._extra_neurons = extra_neurons
        self._sparsity = sparsity
        self._retain_state = retain_state
        self._sequence_inputs = sequence_inputs
        self._steps = steps

        if activation is None:
            activation = lambda x: x
        # elif steps == 1:
            # raise Exception('If activation is provided, but steps = 1, the activation will not be used UNLESS input is a sequence')
        
        self.weight = torch.nn.Parameter(torch.zeros((self._total_neurons, self._total_neurons)))
        self.bias = torch.nn.Parameter(torch.zeros(self._total_neurons))
        self.state = None
        self.activation = activation
        torch.nn.init.kaiming_uniform_(self.weight, mode='fan_in')

        # if retain_state:
            # self.reset_state(self._total_neurons)

    @torch.no_grad()
    def zero_params(self):
        self.weight.data = torch.zeros_like(self.weight.data)
        self.bias.data = torch.zeros_like(self.bias.data)
        if self.state is not None:
            self.state.data = torch.zeros_like(self.state.data)

    def to(self, device, *args, **kwargs):
        super().to(device, *args, **kwargs)
        if self.state is not None:
            self.state = self.state.to(device)
        return self

    def reset_state(self, state_shape):
        self.state = torch.zeros(state_shape, device=self.weight.device)

    def step(self, n=1):
        for step in range(n):
            self.state = _pagnn_op(self.state, self.weight, self.bias)
            if step < n-1:
                self.state = self.activation(self.state)

    def forward(self, x):
        if len(x.shape) == 1 and x.shape[0] != self._input_neurons:
            # treat input data as a sequence
            if self._input_neurons != 1:
                raise NotImplemented('TODO')

            for idx, sample in enumerate(x.unsqueeze(-1)):
                self.load_input_neurons(sample, force_retain_state=idx != 0)
                self.step(n=self._steps)

        elif self._sequence_inputs:
            if len(x) <= 1:
                raise Exception('using sequence inputs but only passed a seq of size 1')

            for idx, sample in enumerate(x):
                self.load_input_neurons(sample)
                self.step(n=self._steps)
    
        else:
            self.load_input_neurons(x)
            self.step(n=self._steps)
        
        return self.extract_output_neurons_data()

    def load_input_neurons(self, x, force_retain_state=None):
        retain_state = self._retain_state if force_retain_state is None else force_retain_state

        if len(x.shape) == 1:
            assert x.shape[0] == self._input_neurons
            if not retain_state or self.state is None:
                self.reset_state(self._total_neurons)
            self.state[:self._input_neurons] = x

        elif len(x.shape) == 2:
            assert x.shape[1] == self._input_neurons
            if not retain_state or self.state is None:
                self.reset_state((x.shape[0], self._total_neurons))
            self.state[:, :self._input_neurons] = x

        else:
            raise Exception()

    def extract_output_neurons_data(self):
        if len(self.state.shape) == 1:
            return self.state[-self._output_neurons:]
        return self.state[:, -self._output_neurons:]


    torch.no_grad()
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        use_super_load = False
        tensors = []

        for name, param in self.named_parameters():
            key = prefix + name
            if key in state_dict:
                input_param = state_dict[key]
                
                # if shapes are equal, can defer to default super loader
                if input_param.shape == param.shape:
                    use_super_load = True
                    break

                tensors.append(input_param)

        if use_super_load:
            super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                          missing_keys, unexpected_keys, error_msgs)
        else:
            W, b = tensors
            net = torch.nn.Linear(W.shape[1], W.shape[0])
            net.weight.data = W
            net.bias.data = b
            new_pagnn = import_ffnn(net, None)
            self.weight.data = new_pagnn.weight.data
            self.bias.data = new_pagnn.bias.data


    def extra_repr(self):
        return 'input_neurons=%i, output_neurons=%i, extra_neurons=%i' % (self._input_neurons, self._output_neurons, self._extra_neurons)
