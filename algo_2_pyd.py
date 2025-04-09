from Common.convert_2_pyd import convert_2_pyd
from enum import Enum
import shutil
from pathlib import Path
ROOTPATH = Path(__file__).resolve().parent
build_dir = ROOTPATH / 'build' 

class Project(Enum):
    ET = 'et'
    RGB = 'rgb'
    CV = 'cv'
    Empty = 'None'

def pack_item(params, name, encrypt, item, target_dir):
    if params[item]:    
        convert_2_pyd(f"./{name}_{item}.py", target_dir, encrypt)
        print(f'✅{name}_{item} -->> : {target_dir}  encrypt: {encrypt}')
    
def pack_algo_2_pyd(params, target_dir):
    name = params['name']
    encrypt = params['encrypt']
    config = params['config']
    if build_dir.exists():
        shutil.rmtree(build_dir)  # 递归删除
    if name == 'None':
        print('⚠️ None project need to be encrpted')
        exit()
    else:        
        for item in ['light', 'sfr', 'dark']:
            pack_item(params, name, encrypt, item, target_dir)
        
        if config:
            target_dir = Path(target_dir)
            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f"./Config/config_{name}.yaml", target_dir)
        
    if build_dir.exists():
        shutil.rmtree(build_dir)  # 递归删除

if __name__ == '__main__':
    params = {'name':Project.CV.value,'light':True, 'sfr':True, 'dark':True, 'config':True, 'encrypt':True}
    target_dir = r'G:\Projects\CV'
    pack_algo_2_pyd(params, target_dir)
    
    
    
    
    



