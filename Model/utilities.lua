function copyTable(fromTable, toTable)
    assert(#fromTable == #toTable)
    for i = 1, #toTable do
        toTable[i]:copy(fromTable[i])
    end
end

-- Returns the length of an arbitrary table
-- Adapted from: https://stackoverflow.com/questions/2705793/how-to-get-number-of-entries-in-a-lua-table
function length(array)
    count = 0
    for _ in pairs(array) do
	count = count + 1
    end
    return count
end

function extendTable(extendedTable, inputTable)
    for k, v in pairs(inputTable) do 
        if (type(extendedTable[k]) == 'table' and type(v) == 'table') then
            extend(extendedTable[k], v)
        else
            extendedTable[k] = v 
        end
    end
end

-- Transfer input tensor to GPU
function transfer_to_gpu(x, gpuidx)
    if gpuidx > 0 then
        return x:cuda()
    else
        return x
    end
end

