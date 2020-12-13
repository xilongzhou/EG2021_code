

## stage 1 4D train
SynData = ""

python train.py --dataroot $SynData --MyTest ALL_4D --name tt --niter_decay 0 --niter 30 --batchSize 4 --output_nc 3 --resize_or_crop crop --fineSize 256 --netG NewVA_Net --augment_input

## stage 2 5D train

python train.py --dataroot $SynData --MyTest ALL_5D_Render --name tt --niter_decay 0 --niter 30 --batchSize 4 --output_nc 3 --resize_or_crop crop --fineSize 256 --netG NewVA_Net_Light --augment_input


## stage 3 real train

loadckpt = ""
RealData = ""

python train.py --dataroot $SynData --MyTest ALL_5D_Render --name 5DRe4D7L0.5_rou1_2 --checkpoints_dir ./checkpoints/VA_Net 
--niter_decay 0 --niter 30 --batchSize 4 --resize_or_crop crop --fineSize 256 --netG NewVA_Net_Light --augment_input --continue_train --load_pretrain $loadckpt --rand_light 0.5 
--real_train --real_dataroot $RealData 