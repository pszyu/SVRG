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

require 'residual-layers'
require 'nn'
require 'data.cifar-dataset'
require 'nngraph'
require 'train-helpers_sgd'
local nninit = require 'nninit'

-- notes here
print [[done by original code; 100 epochs in total; \n 
      learning rate 0.1;\n 
      record gradients weight, gradients bias by epoch;\n
      store data in binary]]

opt = {}
opt["batchSize"] = 10
opt["iterSize"] = 1
opt["Nsize"] = 3
opt["dataRoot"] = "/Users/PaulYu/Documents/bitbucket/resnet_torch_exp_local/dataset/cifar-10-batches-t7"
opt["loadFrom"] = ""
opt["experimentName"] = ""
print(opt)

-- create data loader
dataTrain = Dataset.CIFAR(opt.dataRoot, "train", opt.batchSize)
-- dataTest = Dataset.CIFAR(opt.dataRoot, "test", opt.batchSize)
local mean,std = dataTrain:preprocess()
-- dataTest:preprocess(mean,std)
print("Dataset size: ", dataTrain:size())


-- Residual network.
-- Input: 3x32x32
-- local N = opt.Nsize
local N = 1 -- 32 layers
if opt.loadFrom == "" then
    input = nn.Identity()()
    ------> 3, 32,32
    model = nn.SpatialConvolution(3, 16, 3,3, 1,1, 1,1)
                :init('weight', nninit.kaiming, {gain = 'relu'})
                :init('bias', nninit.constant, 0)(input)
    model = nn.SpatialBatchNormalization(16)(model)
    model = nn.ReLU(true)(model)
    ------> 16, 32,32   First Group
    for i=1,N do   model = addResidualLayer2(model, 16)   end
    ------> 32, 16,16   Second Group
    model = addResidualLayer2(model, 16, 32, 2)
    for i=1,N-1 do   model = addResidualLayer2(model, 32)   end
    ------> 64, 8,8     Third Group
    model = addResidualLayer2(model, 32, 64, 2)
    for i=1,N-1 do   model = addResidualLayer2(model, 64)   end
    ------> 10, 8,8     Pooling, Linear, Softmax
    model = nn.SpatialAveragePooling(8,8)(model)
    model = nn.Reshape(64)(model)
    model = nn.Linear(64, 10)(model)
    model = nn.LogSoftMax()(model)

    model = nn.gModule({input}, {model})

else
    print("Loading model from "..opt.loadFrom)
    cutorch.setDevice(1)
    model = torch.load(opt.loadFrom)
    print "Done"
end

loss = nn.ClassNLLCriterion()


sgdState = {
}


if opt.loadFrom ~= "" then
    print("Trying to load sgdState from "..string.gsub(opt.loadFrom, "model", "sgdState"))
    collectgarbage(); collectgarbage(); collectgarbage()
    sgdState = torch.load(""..string.gsub(opt.loadFrom, "model", "sgdState"))
    collectgarbage(); collectgarbage(); collectgarbage()
    print("Got", sgdState.nSampledImages,"images")
end


-- Actual Training! -----------------------------
weights, gradients = model:getParameters()
function forwardBackwardBatch(ind_sam)
    -- parameter batch is the index of the one sample in batch to use
    -- batch = 0 includes all the samples

    -- actually updated.
    -- Zero the gradient parameters so we can accumulate them again.
    model:training()
    gradients:zero()

    local loss_val = 0
    local inputs, labels, sel_input, sel_label
    inputs, labels = dataTrain:getData()
    if ind_sam > 0 then
      ind_sam = torch.LongTensor{ind_sam}
      sel_input = inputs:index(1, ind_sam)
      sel_label = labels:index(1, ind_sam)
    else
      sel_input = inputs
      sel_label = labels
    end

    collectgarbage(); collectgarbage();
    local y = model:forward(sel_input)
    loss_val = loss_val + loss:forward(y, sel_label)
    local df_dw = loss:backward(y, sel_label)
    model:backward(sel_input, df_dw)
    -- The above call will accumulate all GPUs' parameters onto GPU #1

    loss_val = loss_val / N
    gradients:mul( 1.0 / N )

end


-- files recording training and testing error
---[[
dir = "/Users/PaulYu/Documents/CEE670/workspace_sgd/"
foTrTop1 = io.open(dir.."trTop1_sgd.txt", "a")
foTrTop5 = io.open(dir.."trTop5_sgd.txt", "a")
-- foTeTop1 = io.open(dir.."teTop1_SVRG.txt", "a")
-- foTeTop5 = io.open(dir.."teTop5_SVRG.txt", "a")
--]]
function evalModel()
    ---[[
    -- training error
    local trResults = evaluateModel(model, dataTrain, opt.batchSize)
    foTrTop1:write(trResults.correct1.."  ")
    foTrTop1:flush()
    foTrTop5:write(trResults.correct5.."  ")
    foTrTop5:flush()

    --[[
    -- testing error
    local teResults = evaluateModel(model, dataTest, opt.batchSize)
    foTeTop1:write(teResults.correct1.."  ")
    foTeTop1:flush()
    foTeTop5:write(teResults.correct5.."  ")
    foTeTop5:flush()
    --]]
    
    if (sgdState.epochCounter or 0) > 100 then
        print("Training complete, go home")
        -- close files
        foTrTop1:close()
        foTrTop5:close()
        foTeTop1:close()
        foTeTop5:close()
        torch.save(dir.."resnet.t7", model)
        os.exit()
    end
end

evalModel()



TrainingHelpers.trainForever(
forwardBackwardBatch,
weights,
sgdState,
dataTrain:size(),
evalModel
)
