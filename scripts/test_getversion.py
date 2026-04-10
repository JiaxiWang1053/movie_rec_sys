import torch
import sys
import subprocess
import pkg_resources

def get_cuda_version():
    try:
        result = subprocess.check_output(["nvcc", "--version"]).decode()
        for line in result.split("\n"):
            if "release" in line:
                return line.strip()
    except:
        return "未检测到 nvcc（可能未安装 CUDA Toolkit）"
    return "未知"

def get_gpu_info():
    try:
        result = subprocess.check_output(["nvidia-smi"]).decode()
        return result.split("\n")[0]
    except:
        return "未检测到 GPU"

def get_python_version():
    return sys.version

def get_pytorch_info():
    return {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda
    }

def get_top_packages(n=20):
    packages = sorted(
        [(d.project_name, d.version) for d in pkg_resources.working_set],
        key=lambda x: x[0].lower()
    )
    return packages[:n]

print("="*50)
print("1. Python 版本")
print("="*50)
print(get_python_version())

print("\n" + "="*50)
print("2. GPU 信息")
print("="*50)
print(get_gpu_info())

print("\n" + "="*50)
print("3. CUDA 信息（系统）")
print("="*50)
print(get_cuda_version())

print("\n" + "="*50)
print("4. PyTorch 信息")
print("="*50)
pt = get_pytorch_info()
for k, v in pt.items():
    print(f"{k}: {v}")

print("\n" + "="*50)
print("5. 常用包（前20个）")
print("="*50)
for name, version in get_top_packages():
    print(f"{name}=={version}")