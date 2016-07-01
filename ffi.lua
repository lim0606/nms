local ffi = require 'ffi'

local libpath = package.searchpath('libnms', package.cpath)
if not libpath then return end

require 'cunn'

ffi.cdef[[
void nms_run(THCState* state,
             THIntTensor* keep_out,
             THIntTensor* num_out,
             THCudaTensor* boxes,
             THCudaTensor* mask,
             float nms_overlap_thresh);
]]

return ffi.load(libpath)
