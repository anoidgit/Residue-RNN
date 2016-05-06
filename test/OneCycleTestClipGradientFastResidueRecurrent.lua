require 'paths'
require 'rnn'
require 'ClipGradientFastResidueRecurrent'
dl = require 'dataload'


startlr=0.05
minlr=0.00001
saturate=400--'epoch at which linear decayed LR will reach minlr'
momentum=0.9
maxnormout=-1
batchsize=1
maxepoch=1000
earlystop=50
cutoff=5
seqlen=10
hiddensize=200
progress=true
savepath=paths.concat(dl.SAVE_PATH, 'rrnnlm')
id="20160504rrnn"

trainset, validset, testset = dl.loadPTB({batchsize,1,1})

lm = nn.Sequential()

-- input layer (i.e. word embedding space)
lookup = nn.LookupTable(#trainset.ivocab, hiddensize)
lookup.maxnormout = -1 -- prevent weird maxnormout behaviour
lm:add(lookup) -- input is seqlen x batchsize
--lm:add(nn.Reshape(seqlen,hiddensize)) -- ano add, Clip batch
lm:add(nn.SplitTable(1)) -- tensor to table of tensors

-- rrnn layers
inputsize = hiddensize
inid={}
inid["state-1"]=torch.Tensor(batchsize,hiddensize):fill(0)
inid["state0"]=torch.Tensor(batchsize,hiddensize):fill(0)
inid["input0"]=torch.Tensor(batchsize,inputsize):fill(0)
--rrnn = nn.FastResidueRecurrent(inid,nn.Linear(inputsize,hiddensize),nn.Linear(hiddensize,hiddensize),nn.Linear(inputsize,hiddensize),nn.Linear(hiddensize,hiddensize),nn.Sequential():add(nn.CAddTable()):add(nn.Sigmoid()),nn.Sequential():add(nn.Linear(inputsize, #trainset.ivocab)):add(nn.LogSoftMax()):add(nn.Reshape(1,#trainset.ivocab)))

rrnn = nn.ClipGradientFastResidueRecurrent(inid,nn.Linear(inputsize,hiddensize),nn.Linear(hiddensize,hiddensize),nn.Linear(inputsize,hiddensize),nn.Linear(hiddensize,hiddensize),nn.Sequential():add(nn.CAddTable()):add(nn.Sigmoid()),nn.Sequential():add(nn.Linear(inputsize, #trainset.ivocab)):add(nn.LogSoftMax()),cutoff)

print(rrnn)

--lm:add(rrnn)

-- target is also seqlen x batchsize.
targetmodule = nn.SplitTable(1)

--[[ loss function ]]--
crit = nn.ClassNLLCriterion()
criterion = nn.SequencerCriterion(crit)

epoch = 1
lr = startlr
trainsize=trainset:size()
validsize=validset:size()
for i, inputs, targets in trainset:subiter(seqlen, trainsize) do
	testi=inputs
	testt=targets
	break
end
outputslm = lm:forward(testi)
outputs=rrnn:forward(outputslm)
err = criterion:forward(outputs,testt)
gradOutputs = criterion:backward(outputs, testt)
rrnn:zeroGradParameters()
lm:zeroGradParameters()
gradOutTable=rrnn:backward(outputslm, gradOutputs)
print(gradOutTable)
lm:backward(testi, gradOutTable)
