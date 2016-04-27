------------------------------------------------------------------------
--ano build at 2016/04/27
------------------------------------------------------------------------
local ClipGradientFastResidueRecurrent, parent = torch.class('nn.ClipGradientFastResidueRecurrent', 'nn.Container')

function ClipGradientFastResidueRecurrent:__init(inid, input, nstate, rinput, rstate, merge, transfer, maxgradient)
	parent.__init(self)

	self.statem1=inid["state-1"]
	self.state0=inid["state0"]
	self.input0=inid["input0"]
	self.clipgradient=maxgradient
	local parrelModel=nn.ParallelTable()
		:add(input)
		:add(nstate)
		:add(rinput)
		:add(rstate)
	self.stateModel=nn.Sequential()
		:add(parrelModel)
		:add(merge)
	self.outputModel=transfer
end

function ClipGradientFastResidueRecurrent:forward(inputTable)
	-- output(t) = transfer(state(t-1) + input(t) + state(t-2) + input(t-1))
	local tmp
	self.output={}
	inputTable[0]=self.input0
	self.state={}
	self.state[1]=self.statem1
	self.state[2]=self.state0
	for step=1,#inputTable do
		tmp=self.stateModel:updateOutput({inputTable[step],self.state[step+1],inputTable[step-1],self.state[step]})
		self.state[step+2]=tmp
		self.output[step]=self.outputModel:updateOutput(tmp)
	end
	return self.output
end

function ClipGradientFastResidueRecurrent:backward(inputTable, gradOutputTable, scale)
	scale = scale or 1
	local gradState,input,state_1,input_1,state_2
	for step=#gradOutputTable,3,-1 do
		if gradOutputTable[step]>self.clipgradient then
			gradOutputTable[step]=self.clipgradient
		end
		gradState=self.outputModel:backward(self.state[step+2],gradOutputTable[step],scale)
		input,state_1,input_1,state_2=unpack(self.stateModel:backward({inputTable[step],self.state[step+1],inputTable[step-1],self.state[step]},gradState))
		gradOutputTable[step-1]+=state_1
		gradOutputTable[step-2]+=state_2
		self.gradInput[step]+=input
		self.gradInput[step-1]=input_1
	end
	if gradOutputTable[2]>self.clipgradient then
		gradOutputTable[2]=self.clipgradient
	end
	gradState=self.outputModel:backward(self.state[4],gradOutputTable[2],scale)
	input,state_1,input_1,state_2=unpack(self.stateModel:backward({inputTable[2],self.state[3],inputTable[1],self.state[2]},gradState))
	gradOutputTable[1]+=state_1
	self.updstate0=state_2--state 0 update here,step=2;gradOutputTable[step-2]+=state_2
	self.gradInput[2]+=input
	self.gradInput[1]=input_1
	if gradOutputTable[1]>self.clipgradient then
		gradOutputTable[1]=self.clipgradient
	end
	gradState=self.outputModel:backward(self.state[3],gradOutputTable[1],scale)
	input,state_1,input_1,state_2=unpack(self.stateModel:backward({inputTable[1],self.state[2],self.input0,self.state[1]},gradState))
	self.updstate0+=state_1--state 0 update here,step=1;gradOutputTable[0]+=state_1
	self.updstatem1=state_2--state -1 update here;gradOutputTable[-1]+=state_2
	self.gradInput[1]+=input
	self.updinput0=input_1--input 0 update here;self.gradInput[0]=input_1
	return self.gradInput
end

function ClipGradientFastResidueRecurrent:zeroGradParameters()
	self.stateModel:zeroGradParameters()
	self.outputModel:zeroGradParameters()
	self.updstate0=0
	self.updstatem1=0
	self.updinput0=0
end

function ClipGradientFastResidueRecurrent:updateParameters(learningRate)
	self.stateModel:updateParameters(learningRate)
	self.outputModel:updateParameters(learningRate)
	self.state0:add(-learningRate,self.updstate0)
	self.statem1:add(-learningRate,self.updstatem1)
	self.input0:add(-learningRate,self.updinput0)
end

function ClipGradientFastResidueRecurrent:__tostring__()
	return torch.type(self)
end
