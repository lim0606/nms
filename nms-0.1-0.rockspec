package = "nms"
version = "0.1-0"

source = {
   url = "git://github.com/lim0606/nms",
   tag = "master"
}

description = {
   summary = "torch nms repository",
   detailed = [[
Torch/Lua wrapper for the gpu implementation of non-maximum suppression, which is in https://github.com/rbgirshick/py-faster-rcnn/tree/master/lib/nms
   ]],
   homepage = "https://github.com/lim0606/nms",
   license = "BSD"
}

dependencies = {
   "torch >= 7.0",
   "nn",
}

build = {
   type = "command",
   build_command = [[
cmake -E make_directory build;
cd build;
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(LUA_BINDIR)/.." -DCMAKE_INSTALL_PREFIX="$(PREFIX)"
   ]],
   install_command = "cd build && $(MAKE) install"
}
