# Residue-RNN
Residue Recurrent/Recursive Neural Network with Torch

This is a Residue Recurrent Neural Network library that extends Torch's nn. 
You can use it to build RRNN.
This library includes documentation for the following objects:

Modules that `forward` entire sequences :

 * [ResidueRecurrent](#rrnn.ResidueRecurrent) : A Residue Recurrent Nueral Network;

<a name='rrnn.ResidueRecurrent'></a>
## ResidueRecurrent ##
The `nn.ResidueRecurrent(inid, input, nstate, rinput, rstate, merge, transfer, rho)` constructor takes 8 arguments:
* `inid` : the table contains keys: "state-1", "state0" and "input0" means the state at time -1 and 0 and the input at time 0, the state must be of the same size with the `merge` Modules output, and the input must be of the same size as the `input` Modules input.
 * `input` : a Module that processes input Tensor. Output must be of same size as `nstate`, and same size as the output of the `merge` `rstate` Module.
 * `nstate` : a Module that processes the previous step's output of `merge` Mudule up to the `merge` Module.
 * `rinput` : a Module that processes the previous step's input to the `input` Mudule up to the `merge` Module.
 * `rstate` : a Module that processes the previous second step's output of the `merge` Mudule up to the `merge` Module.
 * `merge` : a [table Module](https://github.com/torch/nn/blob/master/doc/table.md#table-layers) that merges the outputs of the `input`, `nstate`,`rinput` and `rstate` Module before being forwarded through the `transfer` Module.
 * `transfer` : a Module that processes the output of the `merge` Module and output a time-step's output of the ResidueRecurrent Module.
 * `rho` : the maximum amount of backpropagation steps to take back in time. Limits the number of previous steps kept in memory. Due to the vanishing gradients effect, references A and B recommend `rho = 5` (or lower). Defaults to ?.

### [OutputTable] forward(inputTable) ###
Process a input sequece table `inputTable` with Residue Recurrent Neural Network and output the Module's output sequence table `OutputTable`.

### [gradInputTable] backward(inputTable, gradOutputTable, scale) ###
Process backward with the Residue Recurrent Neural Network and output the Module's gradInput sequence(a table). `inputTable` is the Modules input sequence table. `gradOutputTable` is the gradOutput sequence table given by the layer on it. `scale` is the scale will be used by accGradParameters, default is 1.

### zeroGradParameters() ###
this will zero the accumulation of the gradients with respect to the parameters, accumulated through accGradParameters(input, gradOutput,scale) calls.

### updateParameters(learningRate) ###
this will update the parameters, according to the accumulation of the gradients with respect to the parameters, accumulated through backward() calls.
