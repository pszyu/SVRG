path = require 'pl.path'
require 'image'

Dataset = {}
local CIFAR, parent = torch.class("Dataset.CIFAR")

function CIFAR:__init(path, mode, batchSize)
   local trsize = 500
   local tesize = 50

   self.batchSize = batchSize
   self.mode = mode

   if mode == "train" then
      self.data = torch.Tensor(trsize, 3*32*32)
      self.labels = torch.Tensor(trsize)
      self.size = function() return trsize end
      
      local subset = torch.load(path..'/data_batch_' .. (1) .. '.t7', 'ascii')
      self.data[{ {1, trsize} }] = subset.data:t()[{{1, trsize}}]
      self.labels[{ {1, trsize} }] = subset.labels[1][{{1, trsize}}]
      
      self.labels = self.labels + 1
   elseif mode == "test" then
      local subset = torch.load(path..'/test_batch.t7', 'ascii')
      self.data[{ {1, trsize} }] = subset.data:t()[{{1, tesize}}]
      self.labels[{ {1, trsize} }] = subset.labels[1][{{1, tsize}}]
      self.size = function() return tesize end
      self.labels = self.labels + 1
   end

   self.data = self.data[{ {1, self:size()} }] -- Allow using a subset :)
   self.data = self.data:reshape(self:size(), 3, 32,32)

end

function CIFAR:preprocess(mean, std)
   mean = mean or self.data:mean(1)
   std = std or self.data:std() -- Complete std!
   self.data:add(-mean:expandAs(self.data)):mul(1/std)
   return mean,std
end

function CIFAR:sampleIndices(batch, indices)
   if not indices then
      indices = batch
      batch = nil
   end
   local n = indices:size(1)
   batch = batch or {inputs = torch.zeros(n, 3, 32,32),
                     outputs = torch.zeros(n)
                  }
   batch.outputs:copy(self.labels:index(1, indices))
   if self.mode == "train" then
      batch.inputs:zero()
      for i,index in ipairs(torch.totable(indices)) do
         -- Copy self.data[index] into batch.inputs[i], with
         -- preprocessing
         local input = batch.inputs[i]
         input:zero()
         local xoffs, yoffs = torch.random(-4,4), torch.random(-4,4)
         local input_y = {math.max(1,   1 + yoffs),
                          math.min(32, 32 + yoffs)}
         local data_y = {math.max(1,   1 - yoffs),
                         math.min(32, 32 - yoffs)}
         local input_x = {math.max(1,   1 + xoffs),
                          math.min(32, 32 + xoffs)}
         local data_x = {math.max(1,   1 - xoffs),
                         math.min(32, 32 - xoffs)}
         local xmin, xmax = math.max(1, xoffs),  math.min(32, 32+xoffs)

         input[{ {}, input_y, input_x }] = self.data[index][{ {}, data_y, data_x }]
         -- Horizontal flip!!
         if torch.random(1,2)==1 then
            input:copy(image.hflip(input))
         end
      end
   elseif self.mode=="test" then
      batch.inputs:copy(self.data:index(1, indices))
   end
   return batch
end

function CIFAR:sample(batch, batch_size)
   if not batch_size then
      batch_size = batch
      batch = nil
   end
   return self:sampleIndices(
      batch,
      (torch.rand(batch_size) * self:size()):long():add(1)
   )
end

function CIFAR:size()
   return self.data:size(1)
end

function CIFAR:getData()
  inputs = torch.zeros(500, 3, 32,32)
  outputs = torch.zeros(500)
  
  outputs:copy(self.labels)
  inputs:copy(self.data)
  
  return inputs, outputs
end

-- cifar = Dataset.CIFAR("/Users/michael/cifar10/cifar-10-batches-t7", "train")
-- collectgarbage()
-- collectgarbage()
-- display=require'display'
-- display.image(cifar:sample(32).inputs)
