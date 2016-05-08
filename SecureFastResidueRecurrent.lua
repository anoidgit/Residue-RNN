require "nn"
local SecureFastResidueRecurrent, parent = torch.class('nn.SecureFastResidueRecurrent', 'nn.Module')

function SecureFastResidueRecurrent:__init(inid, input, nstate, rinput, rstate, merge, transfer)
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
	self.updstate0=self.srcupdstate0
	self.updstatem1=self.srcupdstatem1
	self.updinput0=self.srcupdinput0
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

function SecureFastResidueRecurrent:updateOutput(inputTable)
	-- output(t) = transfer(state(t-1) + input(t) + state(t-2) + input(t-1))
	self.input=inputTable or self.input
	local bsize=self.input[1]:size()[1]
	if bsize~=self.batchsize then
		self.batchsize=bsize
		self.input0=self.srcinput0:expandAs(self.input[1])
		self.updinput0=self.srcupdinput0:expandAs(self.input0)
		self.statem1=self.srcstatem1:expand(self.batchsize,self.statesize)
		self.state0=self.srcstate0:expandAs(self.statem1)
		self.updstate0=self.srcupdstate0:expandAs(self.statem1)
		self.updstatem1=self.srcupdstatem1:expandAs(self.statem1)
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

function SecureFastResidueRecurrent:forward(inputTable)
	-- output(t) = transfer(state(t-1) + input(t) + state(t-2) + input(t-1))
	self.input=inputTable or self.input
	local bsize=self.input[1]:size()[1]
	if bsize~=self.batchsize then
		self.batchsize=bsize
		self.input0=self.srcinput0:expandAs(self.input[1])
		self.updinput0=self.srcupdinput0:expandAs(self.input0)
		self.statem1=self.srcstatem1:expand(self.batchsize,self.statesize)
		self.state0=self.srcstate0:expandAs(self.statem1)
		self.updstate0=self.srcupdstate0:expandAs(self.statem1)
		self.updstatem1=self.srcupdstatem1:expandAs(self.statem1)
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

function SecureFastResidueRecurrent:backward(inputTable, gradOutputTable, scale)
	self.gradOutput = gradOutputTable or self.gradOutput
	self.input = inputTable or self.input
	scale = scale or 1
	local input,state_1,input_1,state_2
	local gradState={}
	for step=#self.gradOutput,1,-1 do
		gradState[step]=self.outputModel:backward(self.state[step+2],self.gradOutput[step],scale):clone()
	end
	local cstep=#gradState
	if cstep>2 then
		input,state_1,input_1,state_2=unpack(self.stateModel:backward({self.input[cstep],self.state[cstep+1],self.input[cstep-1],self.state[cstep]},gradState[cstep]))
		gradState[cstep-1]:add(state_1)
		gradState[cstep-2]:add(state_2)
		self.gradInput[cstep]=input:clone()
		self.gradInput[cstep-1]=input_1:clone()
	end
	if cstep>3 then
		for step=(#gradState-1),3,-1 do
			input,state_1,input_1,state_2=unpack(self.stateModel:backward({self.input[step],self.state[step+1],self.input[step-1],self.state[step]},gradState[step]))
			gradState[step-1]:add(state_1)
			gradState[step-2]:add(state_2)
			self.gradInput[step]:add(input)
			self.gradInput[step-1]=input_1:clone()
		end
	end
	if cstep>2 then
		input,state_1,input_1,state_2=unpack(self.stateModel:backward({self.input[2],self.state[3],self.input[1],self.state[2]},gradState[2]))
		gradState[1]:add(state_1)
		if (scale~=1) then
			state_2:mul(scale)
		end
		self.updstate0=state_2:clone()--state 0 update here,step=2;self.gradOutput[step-2]+=state_2
		self.gradInput[2]:add(input)
		self.gradInput[1]=input_1:clone()
	end
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

function SecureFastResidueRecurrent:zeroGradParameters()
	self.stateModel:zeroGradParameters()
	self.outputModel:zeroGradParameters()
	self.updstate0:zero()
	self.updstatem1:zero()
	self.updinput0:zero()
end

function SecureFastResidueRecurrent:updateParameters(learningRate)
	self.stateModel:updateParameters(learningRate)
	self.outputModel:updateParameters(learningRate)
	self.state0:add(-learningRate,self.updstate0)
	self.statem1:add(-learningRate,self.updstatem1)
	self.input0:add(-learningRate,self.updinput0)
end

function SecureFastResidueRecurrent:training()
	self.train=true
end

function SecureFastResidueRecurrent:evaluate()
	self.train=false
end

function SecureFastResidueRecurrent:__tostring__()
	return torch.type(self) .. "{\n	State Module:" ..  tostring(self.stateModel):gsub('\n', '\n' .. "	") .."\n	Output Module:" .. tostring(self.outputModel):gsub('\n', '\n' .. "	") .. "\n}"
end