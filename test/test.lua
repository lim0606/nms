require 'nms'

torch.manualSeed(0)
local timer = torch.Timer()

local height = 128 
local width  = 256
local n_rois = 67000
local overlap = 0.3

local boxes = torch.zeros(n_rois, 5)
for i = 1, n_rois do
  boxes[i][1] = math.floor(0.5 * width * torch.rand(1)[1])
  boxes[i][3] = boxes[i][1] + math.floor(0.5 * width * torch.rand(1)[1])
  boxes[i][2] = math.floor(0.5 * height * torch.rand(1)[1])
  boxes[i][4] = boxes[i][2] + math.floor(0.5 * height * torch.rand(1)[1])
  boxes[i][5] = torch.rand(1)[1] 
end 
--print(boxes)

timer:reset()
local keep_inds_cpu = nms.cpu_nms(boxes:cuda(), overlap)
--print(keep_inds_cpu)
--print(boxes:index(1, keep_inds_cpu))
print(('Elapsed time (cpu) %.4f sec'):format(timer:time().real))

timer:reset()
local keep_inds_gpu = nms.gpu_nms(boxes:cuda(), overlap)
--print(keep_inds_gpu)
--print(boxes:index(1, keep_inds_gpu))
print(('Elapsed time (gpu) %.4f sec'):format(timer:time().real))

local diff = torch.abs(keep_inds_cpu - keep_inds_gpu):sum()
print(('diff: %d'):format(diff))
