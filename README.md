# Residue-RNN
Residue Recurrent/Recursive Neural Network with Torch

This is a Residue Recurrent/Recursive Neural Network library that extends Torch's nn. 
You can use it to build RRNN.
This library includes documentation for the following objects:

Modules that `forward` entire sequences :

 * [FastResidueRecurrent](#rrnn.FastResidueRecurrent) : A Fast Effective Whole Sequence BPTT Residue Recurrent Neural Network

<a name='rrnn.FastResidueRecurrent'></a>
## FastResidueRecurrent ##
References about Recurrent:
 * A. [Sutsekever Thesis Sec. 2.5 and 2.8](http://www.cs.utoronto.ca/~ilya/pubs/ilya_sutskever_phd_thesis.pdf)
 * B. [Mikolov Thesis Sec. 3.2 and 3.3](http://www.fit.vutbr.cz/~imikolov/rnnlm/thesis.pdf)
 * C. [RNN and Backpropagation Guide](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.3.9311&rep=rep1&type=pdf)

A [composite Module](https://github.com/torch/nn/blob/master/doc/containers.md#containers) for implementing Recurrent Neural Networks (RNN), excluding the output layer.

The `nn.FastResidueRecurrent(inid, input, nstate, rinput, rstate, merge, transfer)` constructor takes 7 arguments:
* `inid` : the table contains keys: "state-1", "state0" and "input0" means the state at time -1 and 0 and the input at time 0, the state must be of the same size with the `merge` Modules output, and the input must be of the same size as the `input` Modules input.
 * `input` : a Module that processes input Tensor. Output must be of same size as `nstate`, `rinput` and `rstate` Module.
 * `nstate` : a Module that processes the previous step's output of `merge` Mudule up to the `merge` Module.
 * `rinput` : a Module that processes the previous step's input to the `input` Mudule up to the `merge` Module.
 * `rstate` : a Module that processes the previous second step's output of the `merge` Mudule up to the `merge` Module.
 * `merge` : a [table Module](https://github.com/torch/nn/blob/master/doc/table.md#table-layers) that merges the outputs of the `input`, `nstate`,`rinput` and `rstate` Module before being forwarded through the `transfer` Module.
 * `transfer` : a Module that processes the output of the `merge` Module and output a time-step's output of the FastResidueRecurrent Module.

Note that current implementation backward gradOutput through the whole sequence in an effective and fast way, so there is no `rho` parameters, Due to the vanishing gradients effect, references A and B recommend rho = 5 (or lower) for Recurrent, while this Module can effectively weak the vanishing gradients effect.

### [OutputTable] forward(inputTable) ###
Process a input sequece table `inputTable` with Fast Residue Recurrent Neural Network and output the Module's output sequence table `OutputTable`.

### [gradInputTable] backward(inputTable, gradOutputTable, scale) ###
Process backward with the Fast Residue Recurrent Neural Network and output the Module's gradInput sequence(a table). `inputTable` is the Modules input sequence table. `gradOutputTable` is the gradOutput sequence table given by the layer on it. `scale` is the scale will be used by accGradParameters, default is 1.

### zeroGradParameters() ###
This will zero the accumulation of the gradients with respect to the parameters, accumulated through accGradParameters(input, gradOutput,scale) calls.

### updateParameters(learningRate) ###
This will update the parameters, according to the accumulation of the gradients with respect to the parameters, accumulated through backward() calls.
