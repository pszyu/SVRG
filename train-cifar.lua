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

require 'residual-layers'
require 'nn'
require 'data.cifar-dataset'
require 'cutorch'
require 'cunn'
require 'cudnn'
require 'nngraph'
require 'train-helpers'
local nninit = require 'nninit'

-- notes here
print [[done by original code; 70 epochs in total; \n 
      learning rate divide by 10 at 50 epoch, and 60 epoch;\n 
      record layer weights, bias and gradients weight, gradients bias, and output, by iteration;\n
      store data in binary]]

opt = lapp[[
      --batchSize       (default 128)      Sub-batch size
      --iterSize        (default 1)       How many sub-batches in each batch
      --Nsize           (default 3)       Model has 6*n+2 layers.
      --dataRoot        (default /mnt/cifar) Data root folder
      --loadFrom        (default "")      Model to load
      --experimentName  (default "snapshots/cifar-residual-experiment1")
]]
print(opt)

-- create data loader
dataTrain = Dataset.CIFAR(opt.dataRoot, "train", opt.batchSize)
dataTest = Dataset.CIFAR(opt.dataRoot, "test", opt.batchSize)
local mean,std = dataTrain:preprocess()
dataTest:preprocess(mean,std)
print("Dataset size: ", dataTrain:size())


-- Residual network.
-- Input: 3x32x32
-- local N = opt.Nsize
local N = 5 -- 32 layers
if opt.loadFrom == "" then
    input = nn.Identity()()
    ------> 3, 32,32
    model = cudnn.SpatialConvolution(3, 16, 3,3, 1,1, 1,1)
                :init('weight', nninit.kaiming, {gain = 'relu'})
                :init('bias', nninit.constant, 0)(input)
    model = cudnn.SpatialBatchNormalization(16)(model)
    model = cudnn.ReLU(true)(model)
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
    model:cuda()
    --print(#model:forward(torch.randn(100, 3, 32,32):cuda()))
else
    print("Loading model from "..opt.loadFrom)
    cutorch.setDevice(1)
    model = torch.load(opt.loadFrom)
    print "Done"
end

loss = nn.ClassNLLCriterion()
loss:cuda()

sgdState = {
   --- For SGD with momentum ---
   ----[[
   -- My semi-working settings
   learningRate   = "will be set later",
   weightDecay    = 1e-4,
   -- Settings from their paper
   --learningRate = 0.1,
   --weightDecay    = 1e-4,

   momentum     = 0.9,
   dampening    = 0,
   nesterov     = true,
   --]]
   --- For rmsprop, which is very fiddly and I don't trust it at all ---
   --[[
   learningRate = "Will be set later",
   alpha = 0.9,
   whichOptimMethod = 'rmsprop',
   --]]
   --- For adadelta, which sucks ---
   --[[
   rho              = 0.3,
   whichOptimMethod = 'adadelta',
   --]]
   --- For adagrad, which also sucks ---
   --[[
   learningRate = "Will be set later",
   whichOptimMethod = 'adagrad',
   --]]
   --- For adam, which also sucks ---
   --[[
   learningRate = 0.005,
   whichOptimMethod = 'adam',
   --]]
   --- For the alternate implementation of NAG ---
   --[[
   learningRate = 0.01,
   weightDecay = 1e-6,
   momentum = 0.9,
   whichOptimMethod = 'nag',
   --]]
   --

   --whichOptimMethod = opt.whichOptimMethod,
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
weightsTab, gradientsTab = model:parameters()
function forwardBackwardBatch(batch)
    -- After every batch, the different GPUs all have different gradients
    -- (because they saw different data), and only the first GPU's weights were
    -- actually updated.
    -- We have to do two changes:
    --   - Copy the new parameters from GPU #1 to the rest of them;
    --   - Zero the gradient parameters so we can accumulate them again.
    model:training()
    gradients:zero()

    --[[
    -- Reset BN momentum, nvidia-style
    model:apply(function(m)
        if torch.type(m):find('BatchNormalization') then
            m.momentum = 1.0  / ((m.count or 0) + 1)
            m.count = (m.count or 0) + 1
            print("--Resetting BN momentum to", m.momentum)
            print("-- Running mean is", m.running_mean:mean(), "+-", m.running_mean:std())
        end
    end)
    --]]

    -- From https://github.com/bgshih/cifar.torch/blob/master/train.lua#L119-L128
    if sgdState.epochCounter < 50 then
        sgdState.learningRate = 0.1
    elseif sgdState.epochCounter < 60 then
        sgdState.learningRate = 0.01
    else
        sgdState.learningRate = 0.001
    end

    local loss_val = 0
    local N = opt.iterSize
    local inputs, labels
    for i=1,N do
        inputs, labels = dataTrain:getBatch()
        inputs = inputs:cuda()
        labels = labels:cuda()
        collectgarbage(); collectgarbage();
        local y = model:forward(inputs)
        loss_val = loss_val + loss:forward(y, labels)
        local df_dw = loss:backward(y, labels)
        model:backward(inputs, df_dw)
        -- The above call will accumulate all GPUs' parameters onto GPU #1
    end
    loss_val = loss_val / N
    gradients:mul( 1.0 / N )

    return loss_val, gradients, inputs:size(1) * N
end


-- files recording training and testing error
---[[
dir = "/usr/project/xtmp/shuzhiyu/resnet_torch_exp/workspace/"
foTrTop1 = io.open(dir.."trTop1.txt", "a")
--foTrTop5 = io.open(dir.."trTop5.txt", "a")
foTeTop1 = io.open(dir.."teTop1.txt", "a")
--foTeTop5 = io.open(dir.."teTop5.txt", "a")
--]]
function evalModel()
    ---[[
    -- training error
    local trResults = evaluateModel(model, dataTrain, opt.batchSize)
    foTrTop1:write(trResults.."  ")
    foTrTop1:flush()
    --[[
    foTrTop1:write(trResults.correct1.."  ")
    foTrTop1:flush()
    foTrTop5:write(trResults.correct5.."  ")
    foTrTop5:flush()
    --]]
    -- testing error
    local teResults = evaluateModel(model, dataTest, opt.batchSize)
    foTeTop1:write(teResults.."  ")
    foTeTop1:flush()
    --[[
    foTeTop1:write(teResults.correct1.."  ")
    foTeTop1:flush()
    foTeTop5:write(teResults.correct5.."  ")
    foTeTop5:flush()
    --]]
    --[[
    if sgdState.epochCounter then
      print("the sgdState.epochCounter is: "..sgdState.epochCounter)
    end
    --]]
    if (sgdState.epochCounter or 0) > 70 then
        print("Training complete, go home")
        -- close files
        ---[[
        foTrTop1:close()
        --foTrTop5:close()
        foTeTop1:close()
        --foTeTop5:close()
        --]]
        torch.save(dir.."resnet.t7", model)
        os.exit()
    end
end

evalModel()

--[[
require 'graph'
graph.dot(model.fg, 'MLP', '/tmp/MLP')
os.execute('convert /tmp/MLP.svg /tmp/MLP.png')
display.image(image.load('/tmp/MLP.png'), {title="Network Structure", win=23})
--]]

--[[
require 'ncdu-model-explore'
local y = model:forward(torch.randn(opt.batchSize, 3, 32,32):cuda())
local df_dw = loss:backward(y, torch.zeros(opt.batchSize):cuda())
model:backward(torch.randn(opt.batchSize,3,32,32):cuda(), df_dw)
exploreNcdu(model)
--]]



-- --[[
TrainingHelpers.trainForever(
forwardBackwardBatch,
weights,
sgdState,
dataTrain:size(),
evalModel
)
--]]
