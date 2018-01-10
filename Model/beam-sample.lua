-- Sampled summaries will be saved as HDF5 files in the root directory of the model.
-- For example sampling summaries for the items of the testing set (based on their corresponding triples)
-- with a beam size of 3 will create the following file: "./Summaries-Testing-Beam-3.h5".

-- IMPORTANT: Make sure that the language of the pre-trained model matches the version of the dataset that
-- is loaded in the dataset.lua.

local params = {checkpoint = './Checkpoints/ar_model.t7', -- The filepath to the saved pre-trained model.
		beam_size = 4 -- Sets the beam size that will be used during decoding.
}

print('Network Parameters')
print(params)
local ok1, cunn = pcall(require, 'cunn')
local ok2, cutorch = pcall(require, 'cutorch')
if not (ok1 and ok2) then
    print('Warning: Either cunn or cutorch was not found. Falling gracefully to CPU...')
    params.gpuidx = 0
    pcall(require, 'nn')
end


require('nngraph')
require('optim')
require('utilities')
dataset = require('dataset')
require('LookupTableMaskZero')
require('MaskedClassNLLCriterionInheritingNLLCriterion')


local function reset_state(state)
    state.batchidx = 1
    print('State: ' .. string.format("%s", state.name) .. ' has been reset.')

end


local function main()
    local checkpoint = torch.load(params.checkpoint)
    encoder, decoder = checkpoint.model.encoder, checkpoint.model.decoder
    extendTable(params, checkpoint.details.params)

    local triples_dictionary = dataset.triples_dictionary()
    local summaries_dictionary = dataset.summaries_dictionary()
    assert(length(triples_dictionary['item2id']) == length(triples_dictionary['id2item']))
    assert(length(summaries_dictionary['word2id']) == length(summaries_dictionary['id2word']))
    local start_token = summaries_dictionary['word2id']['<start>']
    local end_token = summaries_dictionary['word2id']['<end>']
    local pad_token = summaries_dictionary['word2id']['<PAD>']

    validation = {triples = transfer_to_gpu(dataset.validate_triples(params.numAlignedTriples, params.batch_size), params.gpuidx),
		  summaries = transfer_to_gpu(dataset.validate_summaries(params.batch_size), params.gpuidx),
		  name = 'Validation'}
    testing = {triples = transfer_to_gpu(dataset.test_triples(params.numAlignedTriples, params.batch_size), params.gpuidx),
	       summaries = transfer_to_gpu(dataset.test_summaries(params.batch_size), params.gpuidx),
	       name = 'Testing'}

    reset_state(validation)
    reset_state(testing)
    collectgarbage()
    collectgarbage()


    

    next_word = transfer_to_gpu(torch.zeros(params.batch_size), params.gpuidx)
    LogSoftMax = transfer_to_gpu(torch.zeros(params.batch_size, params.target_vocab_size * params.beam_size), params.gpuidx)
    
    encoder.network:evaluate()
    for j = 1, #decoder.rnns do decoder.rnns[j]:evaluate() end

    decoder.s = {}
    decoder.tempState = {}
    for j = 0, params.beam_size do
	decoder.s[j] = {}
	decoder.tempState[j] = {}
	if params.layers == 1 then
	    decoder.tempState[j] = transfer_to_gpu(torch.zeros(params.batch_size, params.rnn_size), params.gpuidx)
	    decoder.s[j] = transfer_to_gpu(torch.zeros(params.batch_size, params.rnn_size), params.gpuidx)
	else
	    for d = 1, params.layers do
		decoder.tempState[j][d] = transfer_to_gpu(torch.zeros(params.batch_size, params.rnn_size), params.gpuidx)
		decoder.s[j][d] = transfer_to_gpu(torch.zeros(params.batch_size, params.rnn_size), params.gpuidx)
	    end
	end
    end


    -- Set the state from which we'd like to start sampling.
    local state = testing
    local summaries_filename = string.format('./Summaries-%s-Beam-%s.h5', state.name, params.beam_size)
    local summaries_file = hdf5.open(summaries_filename, 'w')
    summaries = torch.zeros(params.beam_size, params.batch_size * state.triples:size(1), params.timesteps + 1)

    
    while state.batchidx <= state.triples:size(1) do
	local batchTriples = state.triples[{state.batchidx, {}, {}, {}}]:reshape(params.batch_size * params.numAlignedTriples, 3)
	encoder.s = encoder.network:forward(batchTriples)
	if params.gpuidx > 0 then cutorch.synchronize() end


	validMask = torch.ones(params.batch_size, params.beam_size)
	
	print('Generating summaries for '.. string.format("%d", state.batchidx).. '. Batch...')
	
	-- We initialise the decoder.
	next_word:fill(start_token)
	summaries:sub(1, params.beam_size, 1, summaries:size(2), 1, 1):fill(start_token)
	if params.layers == 1 then decoder.s[0]:copy(encoder.s)
	else
	    for d = 1, #decoder.s[0] do 
		if d == 2 then decoder.s[0][2]:copy(encoder.s)
		else decoder.s[0][d]:zero() end
	    end
	end
	
	local tempPrediction, tempState = unpack(decoder.rnns[1]:forward({next_word,
									  decoder.s[0]}))
	
	for j = 1, params.beam_size do
	    if params.layers == 1 then decoder.s[j]:copy(tempState)
	    else copyTable(tempState, decoder.s[j]) end
	end
	if params.gpuidx > 0 then cutorch.synchronize() end
	
	batchProbabilities, batchIndeces = tempPrediction:topk(params.beam_size, true, true)
	-- In which case the results are returned from smallest to k-th smallest (dir == false)
	-- or highest to k-th highest (dir == true).

	for i = 2, params.timesteps do

	    
	    LogSoftMax:zero()
	    for beamidx = 1, params.beam_size do
		next_word:copy(batchIndeces:sub(1, params.batch_size, beamidx, beamidx))
		
		
		local tempPrediction, tempState = unpack(decoder.rnns[i]:forward({next_word, decoder.s[beamidx]}))
		if params.layers == 1 then decoder.tempState[beamidx]:copy(tempState)
		else copyTable(tempState, decoder.tempState[beamidx]) end
		if params.gpuidx > 0 then cutorch.synchronize() end

		prediction = tempPrediction + batchProbabilities:sub(1, params.batch_size, beamidx, beamidx)
		    :reshape(params.batch_size, 1):expand(params.batch_size, params.target_vocab_size)
		
		LogSoftMax:sub(1, params.batch_size, (beamidx - 1) * params.target_vocab_size + 1, beamidx * params.target_vocab_size):copy(prediction)
	    end
	    
	    for j = 1, params.batch_size do

		for b = 1, params.beam_size do
		    if validMask[j][s] == 0 then
			
			LogSoftMax:sub(j, j, (b - 1) * params.target_vocab_size + 1, b * params.target_vocab_size):fill(-math.huge)
		    end
		end

		candidates = torch.Tensor(batchIndeces[j]:size()):copy(batchIndeces[j])

		tempProbabilities, tempCandidates = LogSoftMax[j]:topk(params.beam_size, true, true)
		tempMask = torch.Tensor(validMask[j]:size()):copy(validMask[j])
				
		batchProbabilities:sub(j, j, 1, params.beam_size):copy(tempProbabilities)
		tempSummaries = torch.Tensor(summaries:sub(1, params.beam_size, (state.batchidx - 1) * params.batch_size + j, (state.batchidx - 1) * params.batch_size + j, 1, i - 1):size())
		    :copy(summaries:sub(1, params.beam_size, (state.batchidx - 1) * params.batch_size + j, (state.batchidx - 1) * params.batch_size + j, 1, i - 1))
		
		for beamidx = 1, params.beam_size do
		    
		    if validMask[j][beamidx] == 1 then
			candidatesIndex = math.floor((tempCandidates[beamidx - validMask[j]:sub(1, beamidx):eq(0):sum()] - 1) / params.target_vocab_size) + 1
			

			-- The indexing here is preverted!
			summaries[beamidx][(state.batchidx - 1) * params.batch_size + j]:sub(1, i - 1):copy(tempSummaries[candidatesIndex])
			summaries[beamidx][(state.batchidx - 1) * params.batch_size + j][i] = candidates[candidatesIndex]
			
			
			
			batchIndeces[j][beamidx] = tempCandidates[beamidx - validMask[j]:sub(1, beamidx):eq(0):sum()] % params.target_vocab_size
			batchProbabilities[j][beamidx] = tempProbabilities[beamidx - validMask[j]:sub(1, beamidx):eq(0):sum()]
			
			
			
			if params.layers == 1 then 
			    decoder.s[beamidx][j]:copy(decoder.tempState[candidatesIndex][j])
			else
			    for d = 1, params.layers do
				decoder.s[beamidx][d][j]:copy(decoder.tempState[candidatesIndex][d][j])
			    end
			end
			if batchIndeces[j][beamidx] == 0 then
			    summaries[beamidx][(state.batchidx - 1) * params.batch_size + j][i + 1] = end_token
			    batchProbabilities[j][beamidx] = LogSoftMax[j]:min()
			    tempMask[beamidx] = 0

			end
		    else
			batchIndeces[j][beamidx] = 0
			batchProbabilities[j][beamidx] = LogSoftMax[j][(beamidx - 1) * params.target_vocab_size + 1]
		    end

		end
		
		validMask[j]:copy(tempMask)
	    end
	end

	state.batchidx = state.batchidx + 1
	collectgarbage()
	collectgarbage()
    end
    summaries_file:write(tostring('triples'), state.triples:int())
    summaries_file:write(tostring('summaries'), summaries)
    summaries_file:write(tostring('actual_summaries'), state.summaries:int())
    summaries_file:close()
end

main()
