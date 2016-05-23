require "nn"
local FastRNN, parent = torch.class('nn.FastRNN', 'nn.Module')

function FastRNN:__init(inistate, input, nstate, merge, transfer, maxgradient)
	parent.__init(self)
	self.output={}
	self.gradInput={}
	self.state={}
	self.srcstate0=inistate
	self.batchsize=1
	self.statesize=self.srcstate0:size()[2]
	self.state0=self.srcstate0:clone()
	self.srcupdstate0=self.state0:clone():zero()
	self.updstate0=self.srcupdstate0:clone()
	self.clipgradient=maxgradient
	self.train=true
	local parrelModel=nn.ParallelTable()
		:add(input)
		:add(nstate)
	self.stateModel=nn.Sequential()
		:add(parrelModel)
		:add(merge)
	self.outputModel=transfer
end

function FastRNN:updateOutput(inputTable)
	-- output(t) = transfer(state(t-1) + input(t) + state(t-2) + input(t-1))
	self.input=inputTable or self.input
	local bsize=self.input[1]:size()[1]
	if bsize~=self.batchsize then
		self.batchsize=bsize
		self.state0=self.srcstate0:expand(self.batchsize,self.statesize)
		self.updstate0=self.srcupdstate0:repeatTensor(self.batchsize,1)
	end
	self.state={}
	self.output={}
	if self.train then
		self.state[1]=self.state0
		for step=1,#self.input do
			self.state[step+1]=self.stateModel:updateOutput({self.input[step],self.state[step]}):clone()
			self.output[step]=self.outputModel:updateOutput(self.state[step+1]):clone()
		end
	else
		local evastate=self.state0
		for step=1,#self.input do
			evastate=self.stateModel:updateOutput({self.input[step],evastate}):clone()
			self.output[step]=self.outputModel:updateOutput(evastate):clone()
		end
	end
	return self.output
end

function FastRNN:backward(inputTable, gradOutputTable, scale)
	self.gradOutput = gradOutputTable or self.gradOutput
	self.input = inputTable or self.input
	scale = scale or 1
	local input,state_1
	local gradState={}
	for step=#self.gradOutput,1,-1 do
		gradState[step]=self.outputModel:backward(self.state[step+1],self.gradOutput[step],scale):clone()
	end
	local cstep=#gradState
	gradState[cstep]:cmin(self.clipgradient)
	input,state_1=unpack(self.stateModel:backward({self.input[cstep],self.state[cstep]},gradState[cstep]))
	gradState[cstep-1]:add(state_1)
	self.gradInput[cstep]=input:clone()
	for step=(#gradState-1),2,-1 do
		gradState[step]:cmin(self.clipgradient)
		input,state_1=unpack(self.stateModel:backward({self.input[step],self.state[step]},gradState[step]))
		gradState[step-1]:add(state_1)
		self.gradInput[step]=input:clone()
	end
	gradState[1]:cmin(self.clipgradient)
	input,state_1=unpack(self.stateModel:backward({self.input[1],self.state[1]},gradState[1]))
	self.updstate0:add(scale,state_1)--state 0 update here,step=1;self.gradOutput[0]+=state_1
	self.gradInput[1]=input:clone()
	return self.gradInput
end

function FastRNN:zeroGradParameters()
	self.stateModel:zeroGradParameters()
	self.outputModel:zeroGradParameters()
	self.updstate0:zero()
end

function FastRNN:updateParameters(learningRate)
	self.stateModel:updateParameters(learningRate)
	self.outputModel:updateParameters(learningRate)
	local slrate=-learningRate/self.batchsize
	for step=1,self.batchsize do
		self.state0:add(slrate,self.updstate0[step])
	end
end

function FastRNN:training()
	self.train=true
end

function FastRNN:evaluate()
	self.train=false
end

function FastRNN:__tostring__()
	return torch.type(self) .. "{\n	State Module:" ..  tostring(self.stateModel):gsub('\n', '\n' .. "	") .."\n	Output Module:" .. tostring(self.outputModel):gsub('\n', '\n' .. "	") .. "\n}"
end
