# Residue-RNN
Residue Recurrent/Recursive Neural Network with Torch

This is a Residue Recurrent/Recursive Neural Network library that extends Torch's nn. 
You can use it to build RRNN.
This library includes documentation for the following objects:

Modules that `forward` entire sequences :

 * [FastResidueRecurrent](#rrnn.FastResidueRecurrent) : A Fast Effective Whole Sequence BPTT Residue Recurrent Neural Network, can not correctly deal sequece length smaller than 4;
 * [ClipGradientFastResidueRecurrent](#rrnn.ClipGradientFastResidueRecurrent) : A Fast Effective Whole Sequence BPTT Residue Recurrent Neural Network with Clip Gradient, which can help avoid gradient explosion, can not correctly deal sequece length smaller than 4;
 * [SecureFastResidueRecurrent](#rrnn.SecureFastResidueRecurrent) : A secure version of [FastResidueRecurrent](#rrnn.FastResidueRecurrent), can accept any sequence length;
 * [SecureClipGradientFastResidueRecurrent](#rrnn.SecureClipGradientFastResidueRecurrent) : A secure version of  [ClipGradientFastResidueRecurrent](#rrnn.ClipGradientFastResidueRecurrent), can accept any sequence length;

<a name='rrnn.FastResidueRecurrent'></a>
## FastResidueRecurrent ##
References about Recurrent:
 * A. [Sutsekever Thesis Sec. 2.5 and 2.8](http://www.cs.utoronto.ca/~ilya/pubs/ilya_sutskever_phd_thesis.pdf)
 * B. [Mikolov Thesis Sec. 3.2 and 3.3](http://www.fit.vutbr.cz/~imikolov/rnnlm/thesis.pdf)
 * C. [RNN and Backpropagation Guide](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.3.9311&rep=rep1&type=pdf)

A [composite Module](https://github.com/torch/nn/blob/master/doc/containers.md#containers) for implementing Recurrent Neural Networks (RNN), excluding the output layer.

The `rrnn.FastResidueRecurrent(inid, input, nstate, rinput, rstate, merge, transfer)` constructor takes 7 arguments:
* `inid` : the table contains keys: "state-1", "state0" and "input0" means the state at time -1 and 0 and the input at time 0, the state must be of the same size with the `merge` Modules output, and the input must be of the same size as the `input` Modules input.
 * `input` : a Module that processes input Tensor. Output must be of same size as `nstate`, `rinput` and `rstate` Module.
 * `nstate` : a Module that processes the previous step's output of `merge` Mudule up to the `merge` Module.
 * `rinput` : a Module that processes the previous step's input to the `input` Mudule up to the `merge` Module.
 * `rstate` : a Module that processes the previous second step's output of the `merge` Mudule up to the `merge` Module.
 * `merge` : a [table Module](https://github.com/torch/nn/blob/master/doc/table.md#table-layers) that merges the outputs of the `input`, `nstate`,`rinput` and `rstate` Module before being forwarded through the `transfer` Module.
 * `transfer` : a Module that processes the output of the `merge` Module and output a time-step's output of the FastResidueRecurrent Module.

Note that current implementation backward gradOutput through the whole sequence in an effective and fast way, so there is no `rho` parameters, Due to the vanishing gradients effect, references A and B recommend rho = 5 (or lower) for Recurrent, while this Module can effectively weak the vanishing gradients effect.

### [OutputTable] updateOutput(inputTable) ###
Process a input sequece table `inputTable` with Fast Residue Recurrent Neural Network and output the Module's output sequence table `OutputTable`.

### [OutputTable] forward(inputTable) ###
Do the same thing as `updateOutput`.

### [gradInputTable] backward(inputTable, gradOutputTable, scale) ###
Process backward with the Fast Residue Recurrent Neural Network and output the Module's gradInput sequence(a table). `inputTable` is the Modules input sequence table. `gradOutputTable` is the gradOutput sequence table given by the layer on it. `scale` is the scale will be used by accGradParameters, default is 1.

### zeroGradParameters() ###
This will zero the accumulation of the gradients with respect to the parameters, accumulated through accGradParameters(input, gradOutput,scale) calls.

### updateParameters(learningRate) ###
This will update the parameters, according to the accumulation of the gradients with respect to the parameters, accumulated through backward() calls.

### training() ###
In training mode, the network remembers all previous states. This is necessary for BPTT.

### evaluate() ###
During evaluation, since their is no need to perform BPTT at a later time, only the previous two steps is remembered. This is very efficient memory-wise, such that evaluation can be performed using potentially infinite-length sequence.

<a name='rrnn.ClipGradientFastResidueRecurrent'></a>
## ClipGradientFastResidueRecurrent ##
The `rrnn.ClipGradientFastResidueRecurrent(inid, input, nstate, rinput, rstate, merge, transfer, maxgradient)` constructor takes 8 arguments, most arguments are the same with [FastResidueRecurrent](#rrnn.FastResidueRecurrent), except `maxgradient`:
 * `maxgradient` : the maximum value that the gradient can be while bptt, a value larger then that will clip to `maxgradient`, this can effectively keep the Module from gradient explosion, make the Module easier to use.

The method is same with [FastResidueRecurrent](#rrnn.FastResidueRecurrent).

<a name='rrnn.SecureFastResidueRecurrent'></a>
## SecureFastResidueRecurrent ##
A secure version of [FastResidueRecurrent](#rrnn.FastResidueRecurrent), can accept any sequence length;

<a name='rrnn.SecureClipGradientFastResidueRecurrent'></a>
## SecureClipGradientFastResidueRecurrent ##
A secure version of [ClipGradientFastResidueRecurrent](#rrnn.SecureClipGradientFastResidueRecurrent), can accept any sequence length;
