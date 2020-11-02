import torch


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
    def __init__(self, input_neurons, output_neurons, extra_neurons, steps=1, sparsity=0, retain_state=True, activation=None, device=torch.device('cpu')):
        self.device = device

        super(PAGNNLayer, self).__init__()

        assert input_neurons > 0 and output_neurons > 0 and extra_neurons >= 0 and sparsity >= 0 and steps >= 1

        if sparsity > 0:
            raise NotImplemented()

        self._total_neurons = input_neurons + output_neurons + extra_neurons
        self._input_neurons = input_neurons
        self._output_neurons = output_neurons
        self._sparsity = sparsity
        self._retain_state = retain_state
        self._steps = steps

        if activation is None:
            activation = lambda x: x
        elif steps == 1:
            raise Exception('If activation is provided, but steps = 1, the activation will not be used.')
        
        self.weight = torch.nn.Parameter(torch.zeros((self._total_neurons, self._total_neurons)))
        self.bias = torch.nn.Parameter(torch.zeros(self._total_neurons))
        self.state = None
        self.activation = activation
        torch.nn.init.kaiming_uniform_(self.weight, mode='fan_in')

        if retain_state:
            self.reset_state(self._total_neurons)

    def to(self, device, *args, **kwargs):
        self.device = device
        super().to(device, *args, **kwargs)
        if self.state is not None:
            self.state = self.state.to(device)
        return self

    def reset_state(self, state_shape):
        self.state = torch.zeros(state_shape, device=self.device)

    def step(self, n=1):
        for step in range(n):
            self.state = _pagnn_op(self.state, self.weight, self.bias)
            if step < n-1:
                self.state = self.activation(self.state)

    def forward(self, x):
        self.load_input_neurons(x)
        self.step(n=self._steps)
        return self.extract_output_neurons_data()

    def load_input_neurons(self, x):
        if len(x.shape) == 1:
            assert x.shape[0] == self._input_neurons
            if not self._retain_state:
                self.reset_state(self._total_neurons)
            self.state[:self._input_neurons] = x

        elif len(x.shape) == 2:
            assert x.shape[1] == self._input_neurons
            if not self._retain_state:
                self.reset_state((x.shape[0], self._total_neurons))
            self.state[:, :self._input_neurons] = x

        else:
            raise Exception()

    def extract_output_neurons_data(self):
        if len(self.state.shape) == 1:
            return self.state[-self._output_neurons:]
        return self.state[:, -self._output_neurons:]

