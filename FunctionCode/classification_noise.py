import os
import shutil

# # 给 ********************* Noise图像分组 ********************
# # 对噪音图片进行五五分组
# def chunk_it(seq, size):
#     """将序列分为指定大小的块"""
#     it = iter(seq)
#     return iter(lambda: tuple(islice(it, size)), ())

# # 假设你的图片存储在这个路径下
# folder_path = r'E:\Wrok\Temp\Oregon\20250314\dark raw image'

# # 读取文件夹中所有文件的名称
# file_names = sorted([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])

# # 按照每5个一组分组
# groups = list(chunk_it(file_names, 5))

# # 输出每个分组的内容
# for i, group in enumerate(groups, start=1):
#     if len(str(i)) == 1:
#         i = f'0{i}'
#     for name in group:
#         fileName = os.path.join(folder_path, name)
#         tagetPath = os.path.join(folder_path, 'group', str(i))
#         if not os.path.exists(tagetPath):
#             os.makedirs(tagetPath)
#         tagetName = os.path.join(tagetPath, name)
#         shutil.copyfile(fileName,tagetName)


# # ***************************把light图像多复制几份 **********************
import os
import shutil

# 源文件夹路径
source_folder = r'E:\Wrok\ERS\Diamond RGB\Module Images (for algo correlation)\Light (Fail)'

# 获取文件夹中的所有文件
files = os.listdir(source_folder)

# 复制每个文件四次
idx = 0
for file in files:
    # 构建文件的完整路径
    source_file = os.path.join(source_folder, file)
    
    # 检查是否为文件（避免复制子文件夹）
    if os.path.isfile(source_file) and file.endswith('.raw'):
        savePath = os.path.join(source_folder, 'class', str(idx))
        if not os.path.exists(savePath):
            os.makedirs(savePath)
        idx += 1
        for i in range(0, 5):  # 复制四次
            # 构建新文件的名称
            new_file_name = f"{os.path.splitext(file)[0]}_{i}{os.path.splitext(file)[1]}"
            new_file_path = os.path.join(savePath, new_file_name)
            # 复制文件
            shutil.copy(source_file, new_file_path)

print("复制完成！")


# ## 把五五一组的图像只保留一个*****************************
# import glob
# def delete_files_with_extensions(folder_path):
#     # 查找所有文件夹中的所有文件
#     for ext in ['1.raw', '2.raw', '3.raw', '4.raw']:
#         # 使用glob获取指定后缀的所有文件
#         files = glob.glob(os.path.join(folder_path, f'*{ext}'))

#         # 删除找到的文件
#         for file in files:
#             try:
#                 os.remove(file)
#                 print(f"已删除文件: {file}")
#             except Exception as e:
#                 print(f"删除文件失败: {file}, 错误: {e}")

# # 设置目标文件夹路径
# folder_path = r"D:\tem\ERS\Oregon\对标数据\最新对标\Light RAW 241030\Light RAW 241030"
# delete_files_with_extensions(folder_path)


## ******************************** 递归分组 ********************
# import os
# import shutil

# # 设置大文件夹路径
# big_folder = "E:\Wrok\ERS\Diamond CV\Module Images (fo algo correlation)"  # 修改为你的大文件夹路径

# # 遍历大文件夹中的所有小文件夹
# for sub_folder in os.listdir(big_folder):
#     sub_folder_path = os.path.join(big_folder, sub_folder)
    
#     # 确保是文件夹
#     if os.path.isdir(sub_folder_path):
#         # 在当前小文件夹内创建目标文件夹
#         dark_folder = os.path.join(sub_folder_path, "dark_files")
#         light_folder = os.path.join(sub_folder_path, "light_files")
#         sfr_folder = os.path.join(sub_folder_path, "sfr_files")

#         for folder in [dark_folder, light_folder, sfr_folder]:
#             os.makedirs(folder, exist_ok=True)

#         # 遍历小文件夹中的所有文件
#         for filename in os.listdir(sub_folder_path):
#             file_path = os.path.join(sub_folder_path, filename)

#             # 只处理文件，不处理目录
#             if os.path.isfile(file_path):
#                 if "dark" in filename.lower():
#                     shutil.move(file_path, os.path.join(dark_folder, filename))
#                 elif "light" in filename.lower():
#                     shutil.move(file_path, os.path.join(light_folder, filename))
#                 elif "sfr" in filename.lower():
#                     shutil.move(file_path, os.path.join(sfr_folder, filename))

# print("所有文件夹内的文件已分类完成！")
