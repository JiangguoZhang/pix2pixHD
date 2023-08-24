import os
import shutil

def folder_reorganize(root, out_dir):
    for i, k in enumerate(os.listdir(root)):
        os.makedirs(os.path.join(out_dir, f"train/group_{i}/{k}"))
        os.makedirs(os.path.join(out_dir, f"train/group_{i}/1"))
        os.makedirs(os.path.join(out_dir, f"test/group_{i}/{k}"))
        os.makedirs(os.path.join(out_dir, f"test/group_{i}/1"))
        for order in range(5, 10):
            for j in range(1, 100):
                name1 = f'Filtered_{2**order}by{2**order}_seed={j * 2 ** (order - 5)}.tif'
                name2 = f'GroundTruth_{2**order}by{2**order}_seed={j * 2 ** (order - 5)}.tif'
                shutil.copy(os.path.join(root, k, name1), os.path.join(out_dir, f"train/group_{i}/{k}/{2**order}by{2**order}_seed={j * 2 ** (order - 5)}.tif"))

                shutil.copy(os.path.join(root, k, name2), os.path.join(out_dir,
                                                                       f"train/group_{i}/1/{2 ** order}by{2 ** order}_seed={j * 2 ** (order - 5)}.tif"))
            j = 100
            name1 = f'Filtered_{2 ** order}by{2 ** order}_seed={j * 2 ** (order - 5)}.tif'
            name2 = f'GroundTruth_{2 ** order}by{2 ** order}_seed={j * 2 ** (order - 5)}.tif'
            shutil.copy(os.path.join(root, k, name1), os.path.join(out_dir,
                                                                   f"test/group_{i}/{k}/{2 ** order}by{2 ** order}_seed={j * 2 ** (order - 5)}.tif"))

            shutil.copy(os.path.join(root, k, name2), os.path.join(out_dir,
                                                                   f"test/group_{i}/1/{2 ** order}by{2 ** order}_seed={j * 2 ** (order - 5)}.tif"))

        # for name in os.listdir(os.path.join(root, k)):
        #     if name.startswith("Filtered"):
        #         shutil.copy(os.path.join(root, k, name), os.path.join(out_dir, f"group_{i}/{k}", name[9:]))
        #     else:
        #         shutil.copy(os.path.join(root, k, name), os.path.join(out_dir, f"group_{i}/1", name[12:]))

folder_reorganize('/home/xavier/Documents/Tao-ImageSet/TotalSynthesizedImage/20230703/group1',
                  '/home/xavier/Documents/Tao-ImageSet/TotalSynthesizedImage/20230703-dataset')
