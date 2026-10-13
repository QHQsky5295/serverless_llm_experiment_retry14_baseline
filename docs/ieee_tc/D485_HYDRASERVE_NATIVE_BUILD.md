# D485：HydraServe 原生扩展与 fastload 编译

2026-10-13。沿用D484已验证独立依赖环境、D482已有CUDA12.4工具链和官方
commit `8ae605de354ccfa2e9095514cdb4a9e9c56aa56b`。不改原生源码或算法，
不改其他项目k3s，不替换为标准vLLM，不重新安装环境或下载模型。

直接使用官方 CMakeLists 的 `default` 目标（源码中明确提供的直接构建入口）
及 `scripts/image/fastload`。仅指定本机SM86、独立build/output目录、已有
Python/工具链/运行库位置；一次一个编译任务，NVCC线程1，控制构建峰值。
不是推理性能配置搜索；LoRA所需Punica目标保持开启。

## 已完成与已保留的首轮失败

两套CMake配置实际通过：NVCC12.4.131、GCC11.4、Torch2.3.0；CUDA头文件
版本12.4。fastload三步编译和链接实际成功，产生原生libst_pybinding.so。
库尚未实际加载，不能据此宣称CUDA、异步加载或模型正确。

首轮推理扩展编译在第一个cache_kernels.cu因 `cusparse.h` 找不到而停止。
实际文件存在于经过D484 RECORD核验的CUDA数学库目录。CMAKE_INCLUDE_PATH
只供find_path搜索，不自动成为编译器include参数；原标准CUDA镜像中的
统一目录布局与本机分散安装布局不同。这不是kernel算法错误或GPU性能失败。

首轮89.204960秒，峰值2,073,468,928B，high/max/OOM/swap全0；实际递归
成员为空，按InvocationID/inode核验后关闭。全部配置、错误和关闭记录保留。

## 必要修正和当前状态

复用同一build脚本，在单独 `header_retry/` 中给C++/CUDA编译器显式传递
cublas/cudnn/cufft/curand/cusolver/cusparse/nvtx的数学头文件路径。
不把wheel的CUDA12.1 runtime头文件覆盖到NVCC12.4工具链头文件前；
不改任何模型计算、批处理、调度、pipeline、合并或启动重叠源码。
已成功编译的fastload复用，路径/大小/SHA单独记录，不再编一次。

修正后CMake再次通过，cache_kernels.cu实际编译通过，后续原生对象编译
正在继续。以header_retry/build_result.json及真实进程状态判断终态，
不把“开始编译”写成全部扩展已经成功。未进行模型、远端LoRA或性能实验。

辅助资源域3/4GiB、swap0、CPU2,3,26,27，构建单任务/单NVCC线程，
3600秒总上限、16GiB主机内存/100GiB磁盘运行护栏。新增峰值估计8GiB，
启动150GiB/增长门槛不变。不与第二重任务或性能实验同时执行。

后续：检验原生动态库实际加载、专用NCCL/flash-attn依赖；将已验证的
D479层映射、D480生命周期接口、D481IPC/地址适配接入真实worker，验证
实际LoRA数值与generation/timing合同，再100→500-ID→1000→4000。
一台四卡计算主机仍只是一物理节点，机制未自然触发就如实报告。
之后完整可用基线比较，再优化PrimeLoRA；不能以构建成功关闭完整目标。

原始目录：实验仓库 `results/ieee_tc/hydraserve_qualification/d485_20261013/`。
继续同一有进展的构建，不重新启动。正式对比前仍按冻结V1/单轮V2验收。
