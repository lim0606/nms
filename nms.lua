require 'torch'
local C = nms.C

local function gpu_nms(boxes, nms_overlap_thresh)
  assert(torch.type(boxes) == 'torch.CudaTensor')
  assert(boxes:size(1) <= 67000, ('Current implementation only support the case when boxes:size(1) <= 67000: boxes:size(1) = %d'):format(boxes:size(1)))
  -- init
  local boxes = boxes

  -- sort  
  local scores = boxes[{{}, 5}]
  local v, inds = torch.sort(scores, 1, true)
  local boxes = boxes:index(1, inds)

  -- gpu nms 
  local mask = torch.CudaTensor()
  local keep_out = torch.IntTensor()
  keep_out:resize(boxes:size(1))
  local num_out = torch.IntTensor()
  num_out:resize(1)

  C.nms_run(cutorch.getState(),
            keep_out:cdata(),
            num_out:cdata(),
            boxes:cdata(),
            mask:cdata(),
            nms_overlap_thresh)

  -- concat up-to num_out
  keep_out = keep_out:sub(1, num_out[1])

  -- int to long 
  local keep_inds = inds:index(1, keep_out:long()):long()

  return keep_inds 
end
rawset(nms, 'gpu_nms', gpu_nms)

--[[
  This function is copied from https://github.com/fmassa/object-detection.torch
]]
local function cpu_nms(boxes, overlap)

  local pick = torch.LongTensor()

  if boxes:numel() == 0 then
    return pick
  end

  local x1 = boxes[{{},1}]
  local y1 = boxes[{{},2}]
  local x2 = boxes[{{},3}]
  local y2 = boxes[{{},4}]
  local s = boxes[{{},-1}]

  local area = boxes.new():resizeAs(s):zero()
  --if torch.type(boxes) == 'torch.CudaTensor' then
  --  area:copy(x2-x1+1)
  --  area:cmul(y2-y1+1)
  --else
    area:map2(x2,x1,function(xx,xx2,xx1) return xx2-xx1+1 end)
    area:map2(y2,y1,function(xx,xx2,xx1) return xx*(xx2-xx1+1) end)
  --end

  local vals, I = s:sort(1)

  pick:resize(s:size()):zero()
  local counter = 1
  local xx1 = boxes.new()
  local yy1 = boxes.new()
  local xx2 = boxes.new()
  local yy2 = boxes.new()

  local w-- = boxes.new()
  local h-- = boxes.new()

  while I:numel()>0 do
    local last = I:size(1)
    local i = I[last]
    pick[counter] = i
    counter = counter + 1
    if last == 1 then
      break
    end
    I = I[{{1,last-1}}]

    xx1:index(x1,1,I)
    xx1:cmax(x1[i])
    yy1:index(y1,1,I)
    yy1:cmax(y1[i])
    xx2:index(x2,1,I)
    xx2:cmin(x2[i])
    yy2:index(y2,1,I)
    yy2:cmin(y2[i])

    --if torch.type(boxes) == 'torch.CudaTensor' then
    --  w = torch.cmax(xx2-xx1+1, 0)
    --  h = torch.cmax(yy2-yy1+1, 0)
    --else
      w = boxes.new()
      h = boxes.new()
      w:resizeAs(xx2):zero()
      w:map2(xx2,xx1,function(xx,xxx2,xxx1) return math.max(xxx2-xxx1+1,0) end)
      h:resizeAs(yy2):zero()
      h:map2(yy2,yy1,function(xx,yyy2,yyy1) return math.max(yyy2-yyy1+1,0) end)
    --end
    local inter = w
    inter:cmul(h)

    local o = h
    xx1:index(area,1,I)
    torch.cdiv(o,inter,xx1+area[i]-inter)
    I = I[o:le(overlap)]
  end

  pick = pick[{{1,counter-1}}]
  return pick
end
rawset(nms, 'cpu_nms', cpu_nms)

local function func(boxes, nms_overlap_thresh)
  if torch.type(boxes) == 'torch.CudaTensor' then 
    return gpu_nms(boxes, nms_overlap_thresh)
  else
    return cpu_nms(boxes, nms_overlap_thresh) 
  end
end
rawset(nms, 'func', func)
