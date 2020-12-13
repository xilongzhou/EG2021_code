import os
from collections import OrderedDict
from torch.autograd import Variable
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from models.renderer import *

import util.util as util
from util.visualizer import Visualizer
from util import html
import torch
import ntpath

if __name__ == '__main__':

    opt = TestOptions().parse(seed=0,save=False)
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip

    myseed = 0#random.randint(0, 2**31 - 1)
    torch.manual_seed(myseed)
    torch.cuda.manual_seed(myseed)
    torch.cuda.manual_seed_all(myseed) 
    np.random.seed(myseed)

    data_loader = CreateDataLoader(opt)
    
    if opt.real_train:
        dataset, real_dataset = data_loader.load_data()
        mydata=real_dataset
    else:
        dataset = data_loader.load_data()
        mydata=dataset

    visualizer = Visualizer(opt)
    # create website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
    webpage = html.HTML(web_dir,opt.savename, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

    ## gamma correction for display
    Gamma_Correciton=False
    if opt.MyTest == 'Diff' or opt.MyTest == 'Spec':
        Gamma_Correciton=True
    print('Gamma Correction: ',Gamma_Correciton)

    # test
    model = create_model(opt)
    # model.eval()
    print(model)


    if opt.log_loss:
        log_path = os.path.join(opt.results_dir, opt.name, 'Totallog.txt')

        TestLog_Each=open(os.path.join(opt.results_dir, opt.name, '{:s}_epoch.txt'.format(opt.which_epoch)),'w')
        length=len(mydata)
        VGG_total=0
        L1_total=0

    if opt.txt:
      txtpath_index=os.path.join(web_dir,'index_{}'.format(opt.savename))
      if not os.path.exists(txtpath_index):
          os.makedirs(txtpath_index) 
      txtfile_index=open(os.path.join(txtpath_index,'files.txt'),'w')

    allight=torch.empty(0,3).cuda()
    # model.eval()
    for i, data in enumerate(mydata):

        # test on real pairs
        if opt.real_train:
            real_data_pair = torch.cat((data['label'],data['image']),0)
            real_A, real_B, fake_B, fake_feature_A = model.inference_train(data['label'], data['inst'],real_data_pair)
             
            visuals = OrderedDict([('real_A', util.tensor2im(real_A[0], gamma=False)),
                                   ('real_B', util.tensor2im(real_B[0], gamma=False)),                         
                                   ('fake_B', util.tensor2im(fake_B[0], gamma=True, normalize=False)),
                                   ('Normal Fake', util.tensor2im(fake_feature_A[0,0:3,:,:], gamma=False, normalize=False)),
                                   ('Diff Fake', util.tensor2im(fake_feature_A[0,3:6,:,:], gamma=True, normalize=False)),
                                   ('Rough Fake', util.tensor2im(fake_feature_A[0,8:9,:,:].repeat(3,1,1), gamma=False, normalize=False)),
                                   ('Spec Fake', util.tensor2im(fake_feature_A[0,9:12,:,:], gamma=True, normalize=False)),
                                   # ('Render Fake', util.tensor2im(rendered['Fake'][0,:,:,:], gamma=True))
                                   ])  

            img_path = data['path']
            print('process image... %s' % img_path)
            visualizer.save_images(webpage, visuals, img_path)

        else:

            if opt.MyTest=='ALL_5D_Render':
                generated,loss, rendered,inputimage,fakelight = model.inference_train(data['label'], data['inst'], data['image'])

            else: 
                generated,loss,inputimage,fakelight = model.inference_train(data['label'], data['inst'], data['image'])
               

            if opt.mode =='Syn':

                #[0,1] -> [-1,1]
                # normal_image = generated.data[0,0:3,:,:].detach()*2-1
                #[-1,1] 
                Normal_vec = normalize_vec(generated.data[0,0:3,:,:].detach().permute(1,2,0))

                if opt.MyTest=='ALL_1D' or opt.MyTest=='ALL_4D':
                    visuals = OrderedDict([('input_label', util.tensor2label(inputimage[0], opt.label_nc)),
                                           ('Normal Fake', util.tensor2im(Normal_vec.permute(2,0,1), gamma=False)),
                                           ('Normal Real', util.tensor2im(data['image'][0,0:3,:,:], gamma=False)),
                                           ('Diff Fake', util.tensor2im(generated.data[0,3:6,:,:], gamma=True)),
                                           ('Diff Real', util.tensor2im(data['image'][0,3:6,:,:], gamma=True)),
                                           ('Rough Fake', util.tensor2im(generated.data[0,6:9,:,:], gamma=False)),
                                           ('Rough Real', util.tensor2im(data['image'][0,6:9,:,:], gamma=False)),
                                           ('Spec Fake', util.tensor2im(generated.data[0,9:12,:,:], gamma=True)),
                                           ('Spec Real', util.tensor2im(data['image'][0,9:12,:,:], gamma=True))])
                elif opt.MyTest=='ALL_5D_Render':
                    visuals = OrderedDict([('input_label', util.tensor2label(inputimage[0], opt.label_nc)),
                                           ('Normal Fake', util.tensor2im(Normal_vec.permute(2,0,1), gamma=False)),
                                           ('Normal Real', util.tensor2im(data['image'][0,0:3,:,:], gamma=False)),
                                           ('Diff Fake', util.tensor2im(generated.data[0,3:6,:,:], gamma=True)),
                                           ('Diff Real', util.tensor2im(data['image'][0,3:6,:,:], gamma=True)),
                                           ('Rough Fake', util.tensor2im(generated.data[0,6:9,:,:], gamma=False)),
                                           ('Rough Real', util.tensor2im(data['image'][0,6:9,:,:], gamma=False)),
                                           ('Spec Fake', util.tensor2im(generated.data[0,9:12,:,:], gamma=True)),
                                           ('Spec Real', util.tensor2im(data['image'][0,9:12,:,:], gamma=True)),
                                           ('Render Fake', util.tensor2im(rendered[0,:,:,:], gamma=True,normalize=False))
                                           # ('Render Real', util.tensor2im(rendered['Real'][0,:,:,:], gamma=True))

                                           ])
                else:
                    visuals = OrderedDict([('input_label', util.tensor2label(data['label'][0], opt.label_nc)),
                                           ('real_image', util.tensor2im(data['image'][0,3:6,:,:], gamma=Gamma_Correciton)),
                                           ('synthesized_image', util.tensor2im(generated.data[0], gamma=Gamma_Correciton))])
            elif opt.mode =='Real':
              
              # normal_image = generated.data[0,0:3,:,:].detach()*2-1
              # Normal_vec = (normalize_vec(normal_image)+1)*0.5

              if opt.MyTest=='ALL_5D_Render':

                  #[0,1] -> [-1,1]
                  normal_image = generated.data[0,0:3,:,:].detach()*2-1
                  #[-1,1] -> [0,1]
                  Normal_vec = (normalize_vec(normal_image.permute(1,2,0))+1)*0.5

                  visuals = OrderedDict([('input_label', util.tensor2im(inputimage[0], gamma=False)),
                                         ('Normal Fake', util.tensor2im(Normal_vec.permute(2,0,1), gamma=False, normalize=False)),
                                         ('Diff Fake', util.tensor2im(generated.data[0,3:6,:,:], gamma=True, normalize=False)),
                                         ('Rough Fake', util.tensor2im(generated.data[0,8:9,:,:].repeat(3,1,1), gamma=False, normalize=False)),
                                         ('Spec Fake', util.tensor2im(generated.data[0,9:12,:,:], gamma=True, normalize=False)),
                                         ('Render Fake', util.tensor2im(rendered[0,:,:,:], gamma=True,normalize=False))
                                         ])            
              else:

                  #[-1,1]
                  Normal_vec = normalize_vec(generated.data[0,0:3,:,:].detach().permute(1,2,0))

                  visuals = OrderedDict([('input_label', util.tensor2im(inputimage[0], gamma=False)),
                                         ('Normal Fake', util.tensor2im(Normal_vec.permute(2,0,1), gamma=False)),
                                         ('Diff Fake', util.tensor2im(generated.data[0,3:6,:,:], gamma=True)),
                                         ('Rough Fake', util.tensor2im(generated.data[0,8:9,:,:].repeat(3,1,1), gamma=False)),
                                         ('Spec Fake', util.tensor2im(generated.data[0,9:12,:,:], gamma=True))
                                         # ('Render Fake', util.tensor2im(rendered[0,:,:,:], gamma=True))
                                         ]) 

            img_path = data['path']
            print('process image... %s' % img_path)
            visualizer.save_images(webpage, visuals, img_path)

            allight=torch.cat([allight,fakelight],dim=0)

            if opt.txt:
                short_path = ntpath.basename(img_path[0])
                name = os.path.splitext(short_path)[0]
                txtpath_li=os.path.join(web_dir,'light_{}'.format(opt.savename))
                if not os.path.exists(txtpath_li):
                    os.makedirs(txtpath_li)                

                # print(fakelight)
                # print(fakelight.shape)
                txtfile_index.write('{}\n'.format(name[:4]))

                txtfile_li=open(os.path.join(txtpath_li,'{}.txt'.format(name[:4])),'w')
                txtfile_li.write('predefined N: {}\n'.format(1))
                txtfile_li.write('{:.6f},{:.6f},{:.6f}\t'.format(fakelight[0,0].cpu().numpy(),fakelight[0,1].cpu().numpy(),fakelight[0,2].cpu().numpy()))
                txtfile_li.write('{:.6f},{:.6f},{:.6f}\n'.format(0.0,0.0,2.14))
                txtfile_li.close()

        if opt.log_loss:
            VGG_total+=loss['G_VGG']
            L1_total+=loss['G_L1']
            TestLog_Each.write('Number {:d}: VGG total {:.3f}, L1 Total {:.3f} \n'.format(i,loss['G_VGG'],loss['G_L1']))

    if opt.txt:
        txtfile_index.close()

    webpage.save()

    # print(allight)
    # npy_path='F:/LoganZhou/Research/OtherPaper/Pix2Pix/Paper/EG2021/comparison/Realimages'
    # np.save(os.path.join(npy_path , 'light2.npy'),allight.cpu())


    if opt.log_loss:
        TestLog_Each.close()                    
        VGG_total=VGG_total/length
        L1_total=L1_total/length
        TestLog=open(log_path,'a')
        TestLog.write('at epoch {:s}: VGG total {:.3f}, L1 Total {:.3f} \n'.format(opt.which_epoch,VGG_total,L1_total))
        TestLog.close()
