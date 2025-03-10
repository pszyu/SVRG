--[[
Copyright (c) 2016 Michael Wilber

This software is provided 'as-is', without any express or implied
warranty. In no event will the authors be held liable for any damages
arising from the use of this software.

Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it
freely, subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not
   claim that you wrote the original software. If you use this software
   in a product, an acknowledgement in the product documentation would be
   appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be
   misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
--]]

--[[
adjusted by Shuzhi Yu for own experiments
--]]


require 'optim'
TrainingHelpers = {}

function evaluateModel(model, datasetTest, batchSize)
   print("Evaluating...")
   model:evaluate()
   local correct1 = 0
   local correct5 = 0
   local total = 0
   local batches = torch.range(1, datasetTest:size()):long():split(batchSize)
   for i=1,#batches do
       collectgarbage(); collectgarbage();
       local results = datasetTest:sampleIndices(nil, batches[i])
       local batch, labels = results.inputs, results.outputs
       labels = labels:long()
       local y = model:forward(batch)
       local _, indices = torch.sort(y, 2, true)
       -- indices has shape (batchSize, nClasses)
       local top1 = indices:select(2, 1)
       local top5 = indices:narrow(2, 1,5)
       correct1 = correct1 + torch.eq(top1, labels):sum()
       correct5 = correct5 + torch.eq(top5, labels:view(-1, 1):expandAs(top5)):sum()
       total = total + indices:size(1)
   end
   return {correct1=correct1/total, correct5=correct5/total}
end



-- write tensors to file
-- change line based on the first dimension
function writeTensors(T, fileWriter, dim, precision)
  if T:dim()==1 then --begin to write
    for i = 1,T:size(1) do
      fileWriter:write(round(T[i], precision).." ")
    end
  else
    for i = 1,T:size(1) do
      writeTensors(T[i], fileWriter, dim, precision)
      if T:dim()==dim and i~=T:size(1) then
        fileWriter:write("\n") --change line
      end
    end
  end
end


-- write tensors to file v2
-- only record numbers not string label
-- in binary
function writeTensorsV2(T, fileWriter, dim)
  if T:dim()==1 then --begin to write
    for i = 1,T:size(1) do
      fileWriter:writeFloat(T[i])
    end
  else
    for i = 1,T:size(1) do
      writeTensorsV2(T[i], fileWriter, dim)
    end
  end
end


-- function of writing down the name of the term, number of elements, data
-- using writeTensorsV2
function write(fw, name, data)
  fw:writeChar(torch.CharStorage():string(name))
  fw:writeInt(data:nElement())
  writeTensorsV2(data, fw, data:dim())
end

-- round a number to a decimal point
function round(num, numDecimalPlaces)
  local mult = 10^(numDecimalPlaces or 0)
  return math.floor(num * mult + 0.5) / mult
end


function TrainingHelpers.trainForever(forwardBackwardBatch, weights, sgdState, epochSize, afterEpoch)
   local d = Date{os.date()}
   local modelTag = string.format("%04d%02d%02d-%d",
      d:year(), d:month(), d:day(), torch.random())
  
    sgdState.epochCounter = sgdState.epochCounter or 1
    
    
   -- record data begins
   -- create file writers
   dir = "/Users/PaulYu/Documents/CEE670/workspace_sgd/"
   fileWriter = nil

   -- algorithm implemented as in paper Optimization Methods for Large-Scale Machine Learning
   alpha = 0.1
   
   -- initial state w1 is the initialization by kaiming init method
   -- get the weights and gradients vector (pointers)
   weights, gradients = model:getParameters()
   -- define number of training samples
   nSams = epochSize

   -- compute the batch gradients
   forwardBackwardBatch(0)

   while true do -- Each epoch
      collectgarbage(); collectgarbage()
      -- begin sgd step
      local ij = math.random(nSams)
      forwardBackwardBatch(ij)
      -- update weights
      weights:csub(gradients:mul(alpha))


      -- begin to record data
      
      fw_binary = torch.DiskFile(dir.."resnet_Itr"..sgdState.epochCounter..".dat", "w"):binary()
      for _, layer in ipairs(model.modules) do
        -- write layer name and number of data types
        if string.find(tostring(layer), "Convolution") then
          fw_binary:writeChar(torch.CharStorage():string("conv"))
          fw_binary:writeInt(2)
          -- gradient Weight
          write(fw_binary, "gwei", layer.gradWeight)
          -- gradient bias
          write(fw_binary, "gbia", layer.gradBias)
        end
      end
      
      fw_binary:close() 
      
      -- record data ends

      if afterEpoch then afterEpoch() end
      sgdState.epochCounter = sgdState.epochCounter + 1
      print("\n\n----- Epoch "..sgdState.epochCounter.." -----")
      
   end
end

