"""
环境检查脚本

检查运行手部重建demo所需的环境和依赖是否正确安装
"""

import sys
import os
from pathlib import Path


def check_python_version():
    """检查Python版本"""
    print("=" * 60)
    print("检查 Python 版本")
    print("=" * 60)
    
    version = sys.version_info
    print(f"当前Python版本: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print("❌ Python版本过低，需要Python 3.7+")
        return False
    else:
        print("✅ Python版本符合要求")
        return True


def check_packages():
    """检查必要的Python包"""
    print("\n" + "=" * 60)
    print("检查 Python 依赖包")
    print("=" * 60)
    
    required_packages = {
        'numpy': 'numpy',
        'torch': 'torch',
        'cv2': 'opencv-python',
        'matplotlib': 'matplotlib',
        'tqdm': 'tqdm',
        'PIL': 'Pillow',
        'pytorch3d': 'pytorch3d',  # 新增依赖
    }
    
    all_ok = True
    
    for module_name, package_name in required_packages.items():
        try:
            __import__(module_name)
            print(f"✅ {package_name:20s} - 已安装")
        except ImportError:
            print(f"❌ {package_name:20s} - 未安装")
            all_ok = False
    
    if not all_ok:
        print("\n请运行以下命令安装缺失的依赖:")
        print("  pip install -r requirements.txt")
    
    return all_ok


def check_cuda():
    """检查CUDA是否可用"""
    print("\n" + "=" * 60)
    print("检查 CUDA 支持")
    print("=" * 60)
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA 可用")
            print(f"   CUDA版本: {torch.version.cuda}")
            print(f"   GPU数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
            return True
        else:
            print("⚠️  CUDA 不可用，将使用CPU模式（速度较慢）")
            return False
    except ImportError:
        print("❌ PyTorch未安装，无法检查CUDA")
        return False


def check_model_files():
    """检查模型文件是否存在"""
    print("\n" + "=" * 60)
    print("检查 模型权重文件")
    print("=" * 60)
    
    # 返回到VITRA根目录
    project_root = Path(__file__).parent.parent
    
    model_files = {
        'HaWoR模型': project_root / 'weights' / 'hawor' / 'checkpoints' / 'hawor.ckpt',
        '手部检测器': project_root / 'weights' / 'hawor' / 'external' / 'detector.pt',
        'MANO模型': project_root / 'weights' / 'mano',
    }
    
    all_ok = True
    
    for name, path in model_files.items():
        if path.exists():
            print(f"✅ {name:15s} - {path}")
        else:
            print(f"❌ {name:15s} - 未找到: {path}")
            all_ok = False
    
    if not all_ok:
        print("\n请下载并放置所需的模型权重文件")
    
    return all_ok


def check_vitra_modules():
    """检查VITRA项目的必要模块"""
    print("\n" + "=" * 60)
    print("检查 VITRA 项目模块")
    print("=" * 60)
    
    # 将VITRA根目录添加到Python路径
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    required_modules = [
        ('data.tools.utils_hawor', 'HaWoR流水线'),
        ('libs.models.mano_wrapper', 'MANO包装器'),
    ]
    
    all_ok = True
    
    for module_path, description in required_modules:
        try:
            __import__(module_path)
            print(f"✅ {description:20s} - {module_path}")
        except ImportError as e:
            print(f"❌ {description:20s} - 无法导入: {e}")
            all_ok = False
    
    if not all_ok:
        print("\n请确保VITRA项目的依赖模块已正确安装")
    
    return all_ok


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("手部重建Demo - 环境检查")
    print("=" * 60)
    
    results = []
    
    # 执行各项检查
    results.append(("Python版本", check_python_version()))
    results.append(("Python包", check_packages()))
    results.append(("CUDA支持", check_cuda()))
    results.append(("模型文件", check_model_files()))
    results.append(("VITRA模块", check_vitra_modules()))
    
    # 总结
    print("\n" + "=" * 60)
    print("检查结果汇总")
    print("=" * 60)
    
    for name, status in results:
        status_str = "✅ 通过" if status else "❌ 失败"
        print(f"{name:15s}: {status_str}")
    
    all_passed = all(status for _, status in results)
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ 所有检查通过！环境配置正确")
        print("可以运行demo了:")
        print("  python demo.py --help")
    else:
        print("❌ 部分检查未通过，请根据上述提示解决问题")
    print("=" * 60 + "\n")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
