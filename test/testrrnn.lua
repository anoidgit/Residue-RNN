require 'paths'
require 'rnn'
require 'ClipGradientFastResidueRecurrent'
local dl = require 'dataload'

function savemodel(fname,modelsave)
	file=torch.DiskFile(fname,'w')
	file:writeObject(modelsave)
	file:close()
end

startlr=0.05
minlr=0.00001
saturate=400--'epoch at which linear decayed LR will reach minlr'
batchsize=128
maxepoch=30
earlystop=5
cutoff=5
seqlen=64
hiddensize=200
progress=true
savepath=paths.concat(dl.SAVE_PATH, 'rrnnlm')
id="20160504rrnn"

local trainset, validset, testset = dl.loadPTB({batchsize,1,1})
print("Vocabulary size : "..#trainset.ivocab) 
print("Train set split into "..batchsize.." sequences of length "..trainset:size())

local lm = nn.Sequential()

-- input layer (i.e. word embedding space)
local lookup = nn.LookupTable(#trainset.ivocab, hiddensize)
lookup.maxnormout = -1 -- prevent weird maxnormout behaviour
lm:add(lookup) -- input is seqlen x batchsize
lm:add(nn.SplitTable(1)) -- tensor to table of tensors

--rrnn layers
inputsize = hiddensize
inid={}
inid["state-1"]=torch.Tensor(1,hiddensize):zero()
inid["state0"]=torch.Tensor(1,hiddensize):zero()
inid["input0"]=torch.Tensor(1,inputsize):zero()
rrnn = nn.ClipGradientFastResidueRecurrent(inid,nn.Linear(inputsize,hiddensize),nn.Linear(hiddensize,hiddensize),nn.Linear(inputsize,hiddensize),nn.Linear(hiddensize,hiddensize),nn.Sequential():add(nn.CAddTable()):add(nn.Sigmoid()),nn.Sequential():add(nn.Linear(inputsize, #trainset.ivocab)):add(nn.LogSoftMax()),cutoff)

print"Language Model:"
print(lm)

print"RRNN:"
print(rrnn)

-- target is also seqlen x batchsize.
local targetmodule = nn.SplitTable(1)

--[[ loss function ]]--
local crit = nn.ClassNLLCriterion()
local criterion = nn.SequencerCriterion(crit)

--[[ experiment log ]]--

-- is saved to file every time a new validation minima is found
local xplog = {}
xplog.dataset = 'PennTreeBank'
xplog.vocab = trainset.vocab
-- will only serialize params
xplog.lmmodel = nn.Serial(lm)
xplog.lmmodel:mediumSerial()
xplog.rrnnmodel = nn.Serial(rrnn)
xplog.rrnnmodel:mediumSerial()
--xplog.model = lm
xplog.criterion = criterion
xplog.targetmodule = targetmodule
-- keep a log of NLL for each epoch
xplog.trainppl = {}
xplog.valppl = {}
-- will be used for early-stopping
xplog.minvalppl = 99999999
xplog.epoch = 0
local ntrial = 0
paths.mkdir(savepath)

local epoch = 1
lr = startlr
trainsize=trainset:size()
validsize=validset:size()
while epoch <= maxepoch do
	print("")
	print("Epoch #"..epoch.." :")

	-- 1. training
	
	local a = torch.Timer()
	lm:training()
	local sumErr = 0
	for i, inputs, targets in trainset:subiter(seqlen, trainsize) do
		targets = targetmodule:forward(targets)
		
		-- forward
		local outputslm = lm:forward(inputs)
		local outputs = rrnn:forward(outputslm)
		local err = criterion:forward(outputs, targets)
		sumErr = sumErr + err
		
		-- backward 
		local gradOutputs = criterion:backward(outputs, targets)
		rrnn:zeroGradParameters()
		local gradOutputsrrnn = rrnn:backward(outputslm, gradOutputs)
		lm:zeroGradParameters()
		lm:backward(inputs, gradOutputsrrnn)
		
		-- update
		rrnn:updateParameters(lr)
		lm:updateParameters(lr) -- affects params

		if progress then
			xlua.progress(math.min(i + seqlen, trainsize), trainsize)
		end

		if i % 1000 == 0 then
			collectgarbage()
		end

	end
	
	-- learning rate decay
	lr = lr + (minlr - startlr)/saturate
	lr = math.max(minlr, lr)

	print("learning rate", lr)

	local speed = a:time().real/trainsize
	print(string.format("Speed : %f sec/batch ", speed))

	local ppl = torch.exp(sumErr/trainsize)
	print("Training PPL : "..ppl)

	xplog.trainppl[epoch] = ppl

	-- 2. cross-validation

	lm:evaluate()
	local sumErr = 0
	for i, inputs, targets in validset:subiter(seqlen, validsize) do
		targets = targetmodule:forward(targets)
		outputslm = lm:forward(inputs)
		print(outputslm)
		outputs = rrnn:forward(outputslm)
		err = criterion:forward(outputs, targets)
		sumErr = sumErr + err
	end

	local ppl = torch.exp(sumErr/validsize)
	print("Validation PPL : "..ppl)

	xplog.valppl[epoch] = ppl
	ntrial = ntrial + 1

	-- early-stopping
	if ppl < xplog.minvalppl then
		-- save best version of model
		xplog.minvalppl = ppl
		xplog.epoch = epoch 
		local filename = paths.concat(savepath, id..'.t7')
		print("Found new minima. Saving to "..filename)
		torch.save(filename, xplog)
		savemodel(filename..'.lm',lm)
		savemodel(filename..'.rrnn',rrnn)
		ntrial = 0
	elseif ntrial >= earlystop then
		print("No new minima found after "..ntrial.." epochs.")
		print("Stopping experiment.")
		break
	end

	collectgarbage()
	epoch = epoch + 1
end
print("Evaluate model using : ")
print("th scripts/evaluate-rnnlm.lua --xplogpath "..paths.concat(savepath, id..'.t7'))
