# PAGNNs

**WARNING: PAGNNs are currently under VERY ACTIVE research. These results & comparisons are very early in it's lifecycle and there is plenty of ground to cover.**

**This is also the very first fully open source AI research project for my new research company that is powered by [SharpestMinds](https://www.sharpestminds.com/?r=dyllan-mccreary) mentees. If you are interested in being part of my research team, you may reach out to me via `mccreary@dyllan.ai` or you can apply to be my mentee [here](https://app.sharpestminds.com/u/yGyFBQvfv44iG2JC5?r=dyllan-mccreary). If you'd like to see an idea of the types of researchers I'm looking for, the [GitHub Projects](https://github.com/McCrearyD/PAGNNs/projects) or [Issues](https://github.com/McCrearyD/PAGNNs/issues) pages of this repository has a nice list of things to do!**

## What are PAGNNs?
- PAGNN stands for `Persistent Artificial Graph-based Neural Networks`. 
- It is an artificial neural network architecture inspired by the brain that has no layers, but rather it has 1 square weight matrix `W` that denotes the synaptic weightings between neurons. More generally, this is an [Adjacency Matrix](https://en.wikipedia.org/wiki/Adjacency_matrix) that represents the costs between nodes in a [topological graph](https://en.wikipedia.org/wiki/Topological_graph). 
- It also has a state vector `sₜ` that represents each neuron's (or node) internal state values. If `W` is a matrix of size `NxN`, then the corresponding `sₜ` is a vector of size `N`. 
- The intuition is that neurons transfer their state according to the `W` matrix's weights to the next state. 
- The formal math equation is: `sₜ₊₁ = σ(sₜ @ W + b)` where `@` is matrix multiplication.
- Interestingly, the activation function `σ(...)` can be an identity function and still produce nonlinearity. It is in active research to what extent this is useful, although currently using an activation function like ReLU is shown to generalize better. Though I speculate that we may be able to learn this nonlinearity thus eliminating heuristic activation function decisions entirely.

## How do PAGNNs load data?
- Traditionally, neural networks operate upon an input feature vector, typically denoted as `x`. Let's use real numbers for an example. Let's say `x` is a feature vector of size 4. Traditionally, we would turn this into a column vector and matrix multiply with the transposed `W` for the input layer. `W` in this case is `Nx4` making the transposed multiply inner dimensions match `(1, 4) & (4, N)`.
- For PAGNNs, we operate upon the `sₜ` vector which generally should be larger than the input features (we want to allocate more neurons beyond the number of input neurons). If we have the input feature data `x` that is also of size 4, we simply populate the first 4 elements in `sₜ` with this data. 
- If our PAGNN has 10 total neurons (`W` is `10x10` & `sₜ` is of size `10`), the first 4 elements of `sₜ` now has the contents of `x`. Now, if we do the matrix multiply between the `sₜ` (size `10`) vector & `W` (size `10x10`), our output is `sₜ₊₁` which is still of size `10`!

## How do PAGNNs output data?
- We do the same process as data inputs, but rather than putting data into the `sₜ` vector, we pull out data from the end. If we have a PAGNN with 10 neurons, 4 of which are our input neurons, and we want our output data to be of size 2 then we simply allocate 2 of these neurons to be output neurons. After we do a sufficient number of time steps (this is a hyperparameter), we then pull the data from the last 2 elements of `sₜ`. These are our predicted values.

## How do PAGNNs train?
- After extracting our predicted values in the previous section, we can calculate the loss. PyTorch makes this very easy as it builds a dynamic compute graph that allows us to backpropagate the error. Then, we simply call the optimizer's step function!

**Note: In the current state of the repository, PAGNNs are fully dense. I hypothesize that significantly large, but also sparse PAGNNs can perform significantly better.**

## Environment Setup:

1. Clone this repository.
2. `cd path/to/cloned/PAGNNs/`
3. `python -m venv env`
4. `source env/bin/activate`
5. `pip install -e .`

## Comparisons:

### Time Series Prediction:
[Source Code](examples/time_series.py)

Observe the "param" count in the visualization's legend. **The PAGNN here only has 156 parameters and outperforms the LSTM with 41,301 parameters by a very significant amount**. This is using the same LR & same # epochs. This PAGNN is fully dense.

A quick note here, increasing PAGNNs neuron count count doesn't necessarily increase performance, in some cases it can reduce it. This is most likely due to the quadratic increase in noise as you increase the number of neurons. **I hypothesize that sparse training will reduce the amount of noise and boost performance by an extreme amount in comparison to fully dense PAGNNs if executed correctly**.

![](examples/figures/time_series.png)

### Iris Classification:
[Source Code](examples/iris_classification.py)
![](examples/figures/iris_classification.png)

### MNIST:
[Source Code](examples/mnist.py)
![](examples/figures/mnist.png)

### Mushroom Classification:
[Source Code](examples/mushroom_classification.py)
![](examples/figures/mushroom_classification.png)
