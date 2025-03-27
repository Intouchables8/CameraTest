import sys
from pathlib import Path
ROOTPATH = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOTPATH))
import shutil
import ast
from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import importlib.util
import subprocess
import numpy as np

build_dir = ROOTPATH / 'build'
def _get_all_modules(file_name, project_root):
    """ 解析 Python 文件，提取 import 语句涉及的 Python 文件路径 """
    file_name = Path(file_name).resolve()
    project_root = Path(project_root).resolve()

    with file_name.open("r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=str(file_name))

    imported_modules = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):  # 处理 `import xxx`
            for alias in node.names:
                imported_modules.add(alias.name)

        elif isinstance(node, ast.ImportFrom):  # 处理 `from Common import utils`
            if node.module:
                for alias in node.names:
                    full_module = f"{node.module}.{alias.name}"  # 形成完整路径
                    module_path = project_root / node.module.replace(".", "/") / f"{alias.name}.py"
                    # 确保 `Common/utils.py` 存在才添加
                    if module_path.exists():
                        imported_modules.add(full_module)
                    else:
                        module_path = project_root / node.module.replace(".", "/") / f"{alias.name}.pyd"
                        if module_path.exists():
                            imported_modules.add(full_module)
    return imported_modules

def _find_module_file(module_name, project_root):
    """ 通过 importlib 查找模块路径，并确保它在当前目录下 """
    spec = importlib.util.find_spec(module_name)
    if spec and spec.origin:
        module_path = Path(spec.origin)
        # 仅保留当前项目下的 .py 文件
        if module_path.suffix in {".py", ".pyd"} and project_root in module_path.parents or project_root == module_path.parents:
            return module_path
    return None

def _get_all_imported_files(file_name, project_root, visited=None):
    """
    递归获取所有直接和间接 import 的 Python 文件路径，仅限 project_root 下的文件
    """
    file_name = Path(file_name).resolve()
    if visited is None:
        visited = set()
    
    if file_name in visited:
        return visited  # 避免循环依赖

    visited.add(file_name)

    if file_name.suffix == ".py":
        imported_modules = _get_all_modules(file_name, project_root)
    else:
        return visited
    for module in imported_modules:
        module_path = _find_module_file(module, project_root)
        if module_path and module_path not in visited:
            _get_all_imported_files(module_path, project_root, visited)  # 递归查找
    return visited

def _copy_imported_files(source_file, destination_dir):
    """
    获取所有被 import 的 Python 文件，并将它们复制到 destination_dir,保持层级结构
    """
    source_file = Path(source_file).resolve()
    project_root = Path(source_file.parent).resolve()
    destination_dir = Path(destination_dir).resolve()
    destination_dir.mkdir(parents=True, exist_ok=True)  # 确保目标目录存在

    # 获取所有需要复制的 .py 文件
    imported_files = _get_all_imported_files(source_file, project_root)

    for file in imported_files:
        try:
            relative_path = file.relative_to(project_root)  # 计算相对路径
            target_path = destination_dir / relative_path  # 目标路径
            target_path.parent.mkdir(parents=True, exist_ok=True)  # 确保目录存在
            shutil.copy(file, target_path)  # 复制文件
            print(f"✅ Copied: {file} → {target_path}")
        except Exception as e:
            print(f"⚠️ 复制 {file} 时出错: {e}")

    print(f"🎯 {len(imported_files)} files copied to {destination_dir}")

def _build_pyd(target_dir):
    """
    遍历 source_dir 目录，逐个编译 .py 文件为 .pyd，并保留原目录结构
    """
    target_dir = Path(target_dir).resolve()

    # 查找所有 .py 文件
    py_files = list(target_dir.rglob("*.py"))
    if not py_files:
        print("❌ No Python files found to compile.")
        return

    extensions = []
    try:
        extensions = [
            Extension(
                name=str(py_file.relative_to(target_dir)).replace("/", ".").replace("\\", ".").rsplit(".", 1)[0],
                sources=[str(py_file)],
                include_dirs=[np.get_include()]  # 兼容 NumPy
            )
            for py_file in py_files
        ]
        
        setup(
            name="custom_pyd_package",
            ext_modules=cythonize(extensions, language_level="3"),
            packages=find_packages(where=target_dir),  # 自动查找 `G:/Test` 目录中的包
            package_dir={"": target_dir},  # 让 `setup.py` 识别 `G:/Test` 目录下的结构
            script_args=["build_ext", "--inplace"]
        )
        
        
        # for pyd_file in (build_dir).rglob("*.pyd"):
        #     relative_path = pyd_file.relative_to(build_dir)  # 计算相对路径
        #     target_path = target_dir / relative_path  # 复制到目标文件夹的相应位置
        #     target_path.parent.mkdir(parents=True, exist_ok=True)  # 确保目录存在
        #     shutil.copy2(pyd_file, target_path)  # 复制文件

    except subprocess.CalledProcessError as e:
        print(f"❌ Compilation failed: {e}")

def _delete_target_py(target_dir):
    # 清除py c
    for py_file in target_dir.rglob("*.py"):
        if py_file.name != "setup.py":  # 保留 setup.py 直到编译完成
            py_file.unlink()
        for c_file in target_dir.rglob("*.c"):  # 删除 .c 文件
            c_file.unlink()
        

def convert_2_pyd(source_file, target_dir, encripted):
    try:
        source_file = Path(source_file).resolve()
        target_dir = Path(target_dir).resolve()
        target_dir.mkdir(parents=True, exist_ok=True)  # 确保目标目录存在
        _delete_target_py(target_dir)
        # 复制
        # if build_dir.exists():
        #     shutil.rmtree(build_dir)  # 递归删除
        _copy_imported_files(source_file, target_dir)
        
        # 打包
        if encripted:
            _build_pyd(target_dir)
            
        _delete_target_py(target_dir)
        
        # # 清除build
        # if build_dir.exists():
        #     shutil.rmtree(build_dir)  # 递归删除
        print("🗑️  Cleaned up .py and .c files, leaving only .pyd files.")
            
    except subprocess.CalledProcessError as e:
        _delete_target_py(target_dir)
        if build_dir.exists():
            shutil.rmtree(build_dir)  # 递归删除
        print(f"❌ Compilation failed: {e}")



if __name__ == '__main__':
    source_file = r"G:\CameraTest\et_light.py"  # 替换为你的 .py 文件路径
    target_dir = r"G:\Projects\ET"  # 目标文件夹
    encripted = True
    convert_2_pyd(source_file, target_dir, encripted)