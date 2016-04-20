------------------------------------------------------------------------
--ano build at 2016/04/20
------------------------------------------------------------------------
local ResidueRecurrent, parent = torch.class('nn.ResidueRecurrent', 'nn.Container')

function ResidueRecurrent:__init(inid, input, nstate, rinput, rstate, merge, transfer, rho)
	parent.__init(self)

	self.statem1=inid["state-1"]
	self.state0=inid["state0"]
	self.input0=inid["input0"]
	self.rho=rho
	local parrelnet=nn.ParallelTable()
		:add(input)
		:add(nstate)
		:add(rinput)
		:add(rstate)
	self.statenet=nn.Sequential()
		:add(parrelnet)
		:add(merge)
	self.outputnet=transfer
end

function ResidueRecurrent:forward(inputTable)
	-- output(t) = transfer(state(t-1) + input(t) + state(t-2) + input(t-1))
	self.maxBPTT=(#inputTable<self.rho) and #inputTable or self.rho
	self.output={}
	inputTable[0]=self.input0
	self.state={}
	self.state[1]=self.statem1
	self.state[2]=self.state0
	for step=1,#inputTable do
		self.state[step+2]=self.statenet:updateOutput({inputTable[step],self.state[step+1],inputTable[step-1],self.state[step]})
		self.output[step]=self.outputnet:updateOutput(self.state[step+2])
	end
	return self.output
end

function ResidueRecurrent:mergeCache(dstCache,srcCache)
	if not srcCache then
		for key,value in ipairs(srcCache) do
			if not dstCache[key] then
				dstCache[key]+=srcCache[key]
			else
				dstCache[key]=srcCache[key]
			end
		end
	end
	return dstCache
end

function ResidueRecurrent:BPTTStep(stepstart,steps,gradin,inputTable)
	local rs={}
	local irs={}
	if steps>0 then
		local rs1={}
		local rs2={}
		local irs1={}
		local irs2={}
		irs1[stepstart],rs1[stepstart-1],irs2[stepstart-1],rs2[stepstart-2] = unpack(self.statenet:updateGradInput({inputTable[stepstart],self.state[stepstart+1],inputTable[stepstart-1],self.state[stepstart]},gradin))
		local tr,tir=BPTTStep(stepstart-1,steps-1,rs1[stepstart-1])
		rs1=mergeCache(rs1,tr)
		irs1=mergeCache(irs1,tir)
		tr,tir=BPTTStep(stepstart-1,steps-1,rs1[stepstart-1])
		rs2=mergeCache(rs2,tr)
		irs2=mergeCache(irs2,tir)
		rs=mergeCache(rs1,rs2)
		irs=mergeCache(irs1,irs2)
	end
	return rs,irs
end

function ResidueRecurrent:updateGradInput(inputTable, gradOutputTable)
	self.gradInput = {}
	self.sgradOutput = {}
	for step=#gradOutputTable,1,-1 do
		sgradOutput[step] = self.outputnet:updateGradInput(self.state[step+2], gradOutputTable[step])
	end
	-- BPTT sgradOutput
	local gradBPTTCache={}
	local gradCache={}
	local gradiBPTTCache={}
	local gradiCache={}
	if #gradOutputTable>=self.maxBPTT+2 then
		for step=#gradOutputTable,self.maxBPTT+2,-1 do
			gradCache,gradiCache=BPTTStep(step,self.maxBPTT,sgradOutput[step],inputTable)
			gradBPTTCache=mergeCache(gradBPTTCache,gradCache)
			gradiBPTTCache=mergeCache(gradiBPTTCache,gradiCache)
		end
	end
	local tstep=(#inputTable<self.maxBPTT+1) and #inputTable or self.maxBPTT+1
	if tstep>=3 then
		for step=tstep,3,-1 do
			--next line error with steps bptt
			gradCache,gradiCache=BPTTStep(step,step,sgradOutput[step],inputTable)
			gradBPTTCache=mergeCache(gradBPTTCache,gradCache)
			gradiBPTTCache=mergeCache(gradiBPTTCache,gradiCache)
		end
	end
	self.gradInput=gradiBPTTCache
	self.sgradOutput=mergeCache(self.sgradOutput,gradBPTTCache)
	return self.gradInput
end

function ResidueRecurrent:accGradParameters(inputTable, gradOutputTable, scale)
	for step=#gradOutputTable,1,-1 do
		self.outputnet:accGradParameters(self.state[step+2], gradOutputTable[step], scale)
		self.statenet:accGradParameters({inputTable[step],self.state[step+1],inputTable[step-1],self.state[step]}, self.sgradOutput[step], scale)
	end
end

function ResidueRecurrent:zeroGradParameters()
	statenet:zeroGradParameters()
	outputnet:zeroGradParameters()
end

function ResidueRecurrent:__tostring__()
   local tab = '  '
   local line = '\n'
   local next = ' -> '
   local str = torch.type(self)
   str = str .. ' {' .. line .. tab .. '[{input(t), output(t-1)}'
   for i=1,3 do
      str = str .. next .. '(' .. i .. ')'
   end
   str = str .. next .. 'output(t)]'
   
   local tab = '  '
   local line = '\n  '
   local next = '  |`-> '
   local ext = '  |    '
   local last = '   ... -> '
   str = str .. line ..  '(1): ' .. ' {' .. line .. tab .. 'input(t)'
   str = str .. line .. tab .. next .. '(t==0): ' .. tostring(self.startModule):gsub('\n', '\n' .. tab .. ext)
   str = str .. line .. tab .. next .. '(t~=0): ' .. tostring(self.inputModule):gsub('\n', '\n' .. tab .. ext)
   str = str .. line .. tab .. 'output(t-1)'
   str = str .. line .. tab .. next .. tostring(self.feedbackModule):gsub('\n', line .. tab .. ext)
   str = str .. line .. "}"
   local tab = '  '
   local line = '\n'
   local next = ' -> '
   str = str .. line .. tab .. '(' .. 2 .. '): ' .. tostring(self.mergeModule):gsub(line, line .. tab)
   str = str .. line .. tab .. '(' .. 3 .. '): ' .. tostring(self.transferModule):gsub(line, line .. tab)
   str = str .. line .. '}'
   return str
end
