######## Multi-GPU training example #######
python train.py --name label2city_new --batchSize 2 --label_nc 0 --no_instance --dataroot ./datasets/cityscapes3/

python train.py --name liver_1024p --batchSize 2 --label_nc 0 --no_instance --save_epoch_freq 50 --niter 700 --dataroot ./datasets/liver/ --continue_train

python test.py --name liver_1024p_pre --batchSize 2 --label_nc 0 --no_instance --dataroot /home/xavier/Documents/Tao-ImageSet/OneDrive_1_2023-5-22/20230519-worm/worm/in_256