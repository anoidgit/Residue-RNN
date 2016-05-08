require 'paths'
require 'rnn'
local dl = require 'dataload'

torch.setdefaulttensortype('torch.FloatTensor')

function savemodel(fname,modelsave)
	file=torch.DiskFile(fname,'w')
	file:writeObject(modelsave)
	file:close()
end

startlr=0.005
minlr=0.000001
saturate=120--'epoch at which linear decayed LR will reach minlr'
batchsize=256
maxepoch=200
earlystop=15
cutoff=5
seqlen=64
hiddensize=200
progress=true
savepath=paths.concat(dl.SAVE_PATH, 'rnnlm')
id="20160504rnn"

local trainset, validset, testset = dl.loadPTB({batchsize,1,1})
print("Vocabulary size : "..#trainset.ivocab) 
print("Train set split into "..batchsize.." sequences of length "..trainset:size())

local lm = nn.Sequential()

-- input layer (i.e. word embedding space)
local lookup = nn.LookupTable(#trainset.ivocab, hiddensize)
lookup.maxnormout = -1 -- prevent weird maxnormout behaviour
lm:add(lookup) -- input is seqlen x batchsize
lm:add(nn.SplitTable(1)) -- tensor to table of tensors

-- rnn layers
local stepmodule = nn.Sequential() -- applied at each time-step
local inputsize = hiddensize
local rm =  nn.Sequential() -- input is {x[t], h[t-1]}
	:add(nn.ParallelTable()
		:add(i==1 and nn.Identity() or nn.Linear(inputsize, hiddensize)) -- input layer
		:add(nn.Linear(hiddensize, hiddensize))) -- recurrent layer
	:add(nn.CAddTable()) -- merge
	:add(nn.Sigmoid()) -- transfer
rnn = nn.Recurrence(rm, hiddensize, 1)
stepmodule:add(rnn)
inputsize = hiddensize

-- output layer
stepmodule:add(nn.Linear(inputsize, #trainset.ivocab))
stepmodule:add(nn.LogSoftMax())

-- encapsulate stepmodule into a Sequencer
lm:add(nn.Sequencer(stepmodule))

print"Language Model:"
print(lm)

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
xplog.model = nn.Serial(lm)
xplog.model:mediumSerial()
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
		local outputs = lm:forward(inputs)
		local err = criterion:forward(outputs, targets)
		sumErr = sumErr + err
		
		-- backward 
		local gradOutputs = criterion:backward(outputs, targets)
		lm:zeroGradParameters()
		lm:backward(inputs, gradOutputs)
		
		-- update
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
		local outputs = lm:forward(inputs)
		local err = criterion:forward(outputs, targets)
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
		ntrial = 0
	elseif ntrial >= earlystop then
		print("No new minima found after "..ntrial.." epochs.")
		print("Stopping experiment.")
		break
	else
		lr=lr/2
	end

	collectgarbage()
	epoch = epoch + 1
end
print("Evaluate model using : ")
print("th scripts/evaluate-rnnlm.lua --xplogpath "..paths.concat(savepath, id..'.t7'))
