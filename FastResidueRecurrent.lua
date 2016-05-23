require "nn"
local FastResidueRecurrent, parent = torch.class('nn.FastResidueRecurrent', 'nn.Module')

function FastResidueRecurrent:__init(inid, input, nstate, rinput, rstate, merge, transfer)
	parent.__init(self)
	self.output={}
	self.gradInput={}
	self.state={}
	self.srcstatem1=inid["state-1"]
	self.srcstate0=inid["state0"]
	self.srcinput0=inid["input0"]
	self.batchsize=1
	self.statesize=self.srcstatem1:size()[2]
	self.inputsize=self.srcinput0:size()[2]
	self.statem1=self.srcstatem1
	self.state0=self.srcstate0
	self.input0=self.srcinput0
	self.srcupdstate0=self.statem1:clone():zero()
	self.srcupdstatem1=self.srcupdstate0:clone()
	self.srcupdinput0=self.input0:clone():zero()
	self.updstate0=self.srcupdstate0:clone()
	self.updstatem1=self.srcupdstatem1:clone()
	self.updinput0=self.srcupdinput0:clone()
	self.train=true
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

function FastResidueRecurrent:updateOutput(inputTable)
	-- output(t) = transfer(state(t-1) + input(t) + state(t-2) + input(t-1))
	self.input=inputTable or self.input
	local bsize=self.input[1]:size()[1]
	if bsize~=self.batchsize then
		self.batchsize=bsize
		self.input0=self.srcinput0:expandAs(self.input[1])
		self.updinput0=self.srcupdinput0:expandAs(self.input0)
		self.statem1=self.srcstatem1:expand(self.batchsize,self.statesize)
		self.state0=self.srcstate0:expandAs(self.statem1)
		self.updstate0=self.srcupdstate0:repeatTensor(self.batchsize,1)
		self.updstatem1=self.srcupdstatem1:repeatTensor(self.batchsize,1)
	end
	self.input[0]=self.input0
	self.state={}
	self.output={}
	if self.train then
		self.state[1]=self.statem1
		self.state[2]=self.state0
		for step=1,#self.input do
			self.state[step+2]=self.stateModel:updateOutput({self.input[step],self.state[step+1],self.input[step-1],self.state[step]}):clone()
			self.output[step]=self.outputModel:updateOutput(self.state[step+2]):clone()
		end
	else
		local evastatem1=self.statem1
		local evastate0=self.state0
		local evastate
		for step=1,#self.input do
			evastate=self.stateModel:updateOutput({self.input[step],evastate0,self.input[step-1],evastatem1}):clone()
			self.output[step]=self.outputModel:updateOutput(evastate):clone()
			evastatem1=evastate0
			evastate0=evastate
		end
	end
	return self.output
end

function FastResidueRecurrent:backward(inputTable, gradOutputTable, scale)
	self.gradOutput = gradOutputTable or self.gradOutput
	self.input = inputTable or self.input
	scale = scale or 1
	local input,state_1,input_1,state_2
	local gradState={}
	for step=#self.gradOutput,1,-1 do
		gradState[step]=self.outputModel:backward(self.state[step+2],self.gradOutput[step],scale):clone()
	end
	local cstep=#gradState
	input,state_1,input_1,state_2=unpack(self.stateModel:backward({self.input[cstep],self.state[cstep+1],self.input[cstep-1],self.state[cstep]},gradState[cstep]))
	gradState[cstep-1]:add(state_1)
	gradState[cstep-2]:add(state_2)
	self.gradInput[cstep]=input:clone()
	self.gradInput[cstep-1]=input_1:clone()
	for step=(#gradState-1),3,-1 do
		input,state_1,input_1,state_2=unpack(self.stateModel:backward({self.input[step],self.state[step+1],self.input[step-1],self.state[step]},gradState[step]))
		gradState[step-1]:add(state_1)
		gradState[step-2]:add(state_2)
		self.gradInput[step]:add(input)
		self.gradInput[step-1]=input_1:clone()
	end
	input,state_1,input_1,state_2=unpack(self.stateModel:backward({self.input[2],self.state[3],self.input[1],self.state[2]},gradState[2]))
	gradState[1]:add(state_1)
	if (scale~=1) then
		state_2:mul(scale)
	end
	self.updstate0=state_2:clone()--state 0 update here,step=2;self.gradOutput[step-2]+=state_2
	self.gradInput[2]:add(input)
	self.gradInput[1]=input_1:clone()
	input,state_1,input_1,state_2=unpack(self.stateModel:backward({self.input[1],self.state[2],self.input0,self.state[1]},gradState[1]))
	self.updstate0:add(scale,state_1)--state 0 update here,step=1;self.gradOutput[0]+=state_1
	if (scale~=1) then
		state_2:mul(scale)
	end
	self.updstatem1=state_2:clone()--state -1 update here;self.gradOutput[-1]+=state_2
	self.gradInput[1]:add(input)
	if (scale~=1) then
		input_1:mul(scale)
	end
	self.updinput0=input_1:clone()--input 0 update here;gradInput[0]=input_1
	return self.gradInput
end

function FastResidueRecurrent:zeroGradParameters()
	self.stateModel:zeroGradParameters()
	self.outputModel:zeroGradParameters()
	self.updstate0:zero()
	self.updstatem1:zero()
	self.updinput0:zero()
end

function FastResidueRecurrent:updateParameters(learningRate)
	self.stateModel:updateParameters(learningRate)
	self.outputModel:updateParameters(learningRate)
	local slrate=-learningRate/self.batchsize
	for step=1,self.batchsize do
		self.state0:add(slrate,self.updstate0[step])
		self.statem1:add(slrate,self.updstatem1[step])
		self.input0:add(slrate,self.updinput0[step])
	end
end

function FastResidueRecurrent:training()
	self.train=true
end

function FastResidueRecurrent:evaluate()
	self.train=false
end

function FastResidueRecurrent:__tostring__()
	return torch.type(self) .. "{\n	State Module:" ..  tostring(self.stateModel):gsub('\n', '\n' .. "	") .."\n	Output Module:" .. tostring(self.outputModel):gsub('\n', '\n' .. "	") .. "\n}"
end
