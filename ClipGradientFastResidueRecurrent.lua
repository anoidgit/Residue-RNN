require "nn"
local ClipGradientFastResidueRecurrent, parent = torch.class('nn.ClipGradientFastResidueRecurrent', 'nn.Module')

function ClipGradientFastResidueRecurrent:__init(inid, input, nstate, rinput, rstate, merge, transfer, maxgradient)
	parent.__init(self)
	self.output={}
	self.state={}
	self.statem1=inid["state-1"]
	self.state0=inid["state0"]
	self.input0=inid["input0"]
	self.updstate0=self.statem1:clone():fill(0)
	self.updstatem1=self.updstate0:clone()
	self.updinput0=self.input0:clone():fill(0)
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
	local outputTable={}
	inputTable[0]=self.input0
	self.state={}
	self.state[1]=self.statem1
	self.state[2]=self.state0
	for step=1,#inputTable do
		tmp=self.stateModel:updateOutput({inputTable[step],self.state[step+1],inputTable[step-1],self.state[step]})
		self.state[step+2]=tmp
		outputTable[step]=self.outputModel:updateOutput(tmp)
	end
	return outputTable
end

function ClipGradientFastResidueRecurrent:backward(inputTable, gradOutputTable, scale)
	scale = scale or 1
	local input,state_1,input_1,state_2
	local gradInput={}
	local gradState={}
	for step=#gradOutputTable,1,-1 do
		gradState[step]=self.outputModel:backward(self.state[step+2],gradOutputTable[step],scale)
	end
	local cstep=#gradState
	gradState[cstep]:cmin(self.clipgradient)
	input,state_1,input_1,state_2=unpack(self.stateModel:backward({inputTable[cstep],self.state[cstep+1],inputTable[cstep-1],self.state[cstep]},gradState[cstep]))
	gradState[cstep-1]:add(state_1)
	gradState[cstep-2]:add(state_2)
	gradInput[cstep]=input
	gradInput[cstep-1]=input_1
	for step=(#gradState-1),3,-1 do
		gradState[step]:cmin(self.clipgradient)
		input,state_1,input_1,state_2=unpack(self.stateModel:backward({inputTable[step],self.state[step+1],inputTable[step-1],self.state[step]},gradState[step]))
		gradState[step-1]:add(state_1)
		gradState[step-2]:add(state_2)
		gradInput[step]:add(input)
		gradInput[step-1]=input_1
	end
	gradState[2]:cmin(self.clipgradient)
	input,state_1,input_1,state_2=unpack(self.stateModel:backward({inputTable[2],self.state[3],inputTable[1],self.state[2]},gradState[2]))
	gradState[1]:add(state_1)
	if (scale~=1) then
		state_2:mul(scale)
	end
	self.updstate0=state_2--state 0 update here,step=2;gradOutputTable[step-2]+=state_2
	gradInput[2]:add(input)
	gradInput[1]=input_1
	gradState[1]:cmin(self.clipgradient)
	input,state_1,input_1,state_2=unpack(self.stateModel:backward({inputTable[1],self.state[2],self.input0,self.state[1]},gradState[1]))
	self.updstate0:add(scale,state_1)--state 0 update here,step=1;gradOutputTable[0]+=state_1
	if (scale~=1) then
		state_2:mul(scale)
	end
	self.updstatem1=state_2--state -1 update here;gradOutputTable[-1]+=state_2
	gradInput[1]:add(input)
	if (scale~=1) then
		input_1:mul(scale)
	end
	self.updinput0=input_1--input 0 update here;gradInput[0]=input_1
	return gradInput
end

function ClipGradientFastResidueRecurrent:zeroGradParameters()
	self.stateModel:zeroGradParameters()
	self.outputModel:zeroGradParameters()
	self.updstate0:fill(0)
	self.updstatem1:fill(0)
	self.updinput0:fill(0)
end

function ClipGradientFastResidueRecurrent:updateParameters(learningRate)
	self.stateModel:updateParameters(learningRate)
	self.outputModel:updateParameters(learningRate)
	self.state0:add(-learningRate,self.updstate0)
	self.statem1:add(-learningRate,self.updstatem1)
	self.input0:add(-learningRate,self.updinput0)
end

function ClipGradientFastResidueRecurrent:__tostring__()
	return torch.type(self) .. "{\nOutput Module:" .. tostring(self.outputModel) .. "\nState Module:" ..  tostring(self.stateModel) .."\n}"
end
