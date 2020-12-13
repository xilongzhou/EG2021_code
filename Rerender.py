import torch
import os
from os import listdir
from os.path import join
import numpy as np
import argparse
import torch.utils.data as data
from scipy.misc import imread,imresize, imsave

from torch.utils.data import DataLoader
from PIL import Image
import imageio

import cv2
from cv2 import VideoWriter, VideoWriter_fourcc 

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from util.util import *
import torch.distributions as tdist

from models.renderer import *
from pygifsicle import optimize

def savenpy(path,tensor):
	np.save(path,tensor.detach().cpu())

def Cosine_Distribution_Number(Number, r_max, mydevice):

	# u_1= torch.rand((Number,1),device=mydevice)*r_max #+ r_max	# rmax: 0.95 (default)
	# u_2= torch.rand((Number,1),device=mydevice)

	u_1= torch.rand((Number,1),device=mydevice)*0.0 + r_max	# gif
	u_2= 1.0/Number*torch.arange(1,Number+1,dtype=torch.float32).unsqueeze(-1).to(mydevice) # gif
	# print('u_2:', u_2)
	# print('u_22:', u_22)

	r = torch.sqrt(u_1)
	theta = 2*PI*u_2

	x = r*torch.cos(theta)
	y = r*torch.sin(theta)
	z = torch.sqrt(1-r*r)

	temp_out = torch.cat([x,y,z],1)

	return temp_out



def N_Light(Near_Number,r_max, mydevice):

	rand_light = Cosine_Distribution_Number(Near_Number, r_max, mydevice)

	# Origin ([-1,1],[-1,1],0)
	# Origin=torch.tensor([0.0,0.0],device=mydevice)
	# Origin_xy = torch.rand((Near_Number,2),device=mydevice)*2-1
	Origin_xy = torch.rand((Near_Number,2),device=mydevice)*0
	Origin = torch.cat([Origin_xy,torch.zeros((Near_Number,1),device=mydevice)],1)

	m=tdist.Normal(torch.tensor([1.0]),torch.tensor([0.75]))
	Distance=m.sample((Near_Number,2)).to(mydevice)
	Light_po=Origin+rand_light*2.14#torch.exp(Distance[:,0])

	return Light_po



def InverseLog(tensor):
	temp=tensor*(np.log(1.01)-np.log(0.01))+np.log(0.01)
	return torch.exp(temp)
	# return  (tf.log(tf.add(tensor,0.01)) - tf.log(0.01)) / (tf.log(1.01)-tf.log(0.01))

def saveimage(image_numpy, image_path, gamma=True):
	
	if gamma:
		image_numpy = image_numpy**(1/2.2)*255.0
	else:
		image_numpy = image_numpy*255.0

	image_numpy = np.clip(image_numpy, 0, 255)
	image_numpy = image_numpy.astype(np.uint8)
	print(image_numpy.shape)
	image_pil = Image.fromarray(image_numpy)
	# image_pil.save(image_path, format='JPEG', subsampling=0, quality=100)
	image_pil.save(image_path, format='PNG', subsampling=0, quality=100)


def load_light_txt(name):
    with open(name,'r') as f:
        lines = f.readlines()
        wlvs = []
        for line in lines[0:]:
            line = line[:-1]
            camera_pos = [float(i) for i in line.split(',')]
            # camera_pos = [i for i in line.split(',')]
            # print(camera_pos)
            wlvs.append(camera_pos)
        wlvs=np.array(wlvs)
        print(wlvs)
        return wlvs



class DataLoaderHelper(data.Dataset):
	def __init__(self, image_dir):
		super(DataLoaderHelper, self).__init__()
		self.path = image_dir
		# self.image_filenames = glob('{:s}/*.png'.format(image_dir),recursive=True)
		self.image_filenames = [x for x in listdir(image_dir)]
		print(self.image_filenames)
	def __getitem__(self, index):
	   
		CROP_SIZE=256

		fullimage = mpimg.imread(join(self.path,self.image_filenames[index]))
		[height,width,channel] = fullimage.shape
		# print(fullimage.dtype)

		if fullimage.dtype=='uint8':
			fullimage = fullimage.astype(np.float32)
			fullimage=torch.from_numpy(fullimage/255.0)
		elif fullimage.dtype=='float32':
			fullimage=torch.from_numpy(fullimage)
		else:
			raise('data type error')

		### crop the image to desired size
		temp_crop=int((height-CROP_SIZE)*0.5)

		xyCropping=np.array([temp_crop,temp_crop])

		#split the image into five subimages
		# inputimage = fullimage[xyCropping[0] : xyCropping[0] + CROP_SIZE, xyCropping[1] : xyCropping[1] + CROP_SIZE, :]
		# normal = fullimage[xyCropping[0] : xyCropping[0] + CROP_SIZE, xyCropping[1]+height : xyCropping[1] + CROP_SIZE+height, :]
		# diff = fullimage[xyCropping[0] : xyCropping[0] + CROP_SIZE, xyCropping[1]+height*2 : xyCropping[1] + CROP_SIZE+height*2, :]
		# rough = fullimage[xyCropping[0] : xyCropping[0] + CROP_SIZE, xyCropping[1]+height*3 : xyCropping[1] + CROP_SIZE+height*3, :]
		# spec = fullimage[xyCropping[0] : xyCropping[0] + CROP_SIZE, xyCropping[1]+height*4 : xyCropping[1] + CROP_SIZE+height*4, :]

		diff = fullimage[xyCropping[0] : xyCropping[0] + CROP_SIZE, xyCropping[1] : xyCropping[1] + CROP_SIZE, :]
		normal = fullimage[xyCropping[0] : xyCropping[0] + CROP_SIZE, xyCropping[1]+height : xyCropping[1] + CROP_SIZE+height, :]
		rough = fullimage[xyCropping[0] : xyCropping[0] + CROP_SIZE, xyCropping[1]+height*2 : xyCropping[1] + CROP_SIZE+height*2, :]
		spec = fullimage[xyCropping[0] : xyCropping[0] + CROP_SIZE, xyCropping[1]+height*3 : xyCropping[1] + CROP_SIZE+height*3, :]
		
		# normal = fullimage[xyCropping[0] : xyCropping[0] + CROP_SIZE, xyCropping[1] : xyCropping[1] + CROP_SIZE, :]
		# diff = fullimage[xyCropping[0] : xyCropping[0] + CROP_SIZE, xyCropping[1]+height : xyCropping[1] + CROP_SIZE+height, :]
		# rough = fullimage[xyCropping[0] : xyCropping[0] + CROP_SIZE, xyCropping[1]+height*2 : xyCropping[1] + CROP_SIZE+height*2, :]
		# spec = fullimage[xyCropping[0] : xyCropping[0] + CROP_SIZE, xyCropping[1]+height*3 : xyCropping[1] + CROP_SIZE+height*3, :]

		return normal,diff,rough,spec

		#return fullimage

	def __len__(self):
		return len(self.image_filenames)

##### the input of normalize_vec function must be an image with 3 channels


class DataLoaderHelper_LoadSingleImage(data.Dataset):
	def __init__(self, image_dir):
		super(DataLoaderHelper_LoadSingleImage, self).__init__()
		self.path = image_dir
		self.image_filenames = [x for x in listdir(image_dir)]
		print(self.path)

	def __getitem__(self, index):

		fullimage = mpimg.imread(join(self.path,self.image_filenames[index]))#,format='jpg')
		if fullimage.dtype=='uint8':
			inputimage = fullimage.astype(np.float32)
			inputimage=torch.from_numpy(inputimage/255.0)
		elif fullimage.dtype=='float32':
			inputimage=torch.from_numpy(fullimage)
		else:
			raise('data type error')

		if inputimage.shape[-1]!=3:
			inputimage=inputimage.unsqueeze(-1).repeat(1,1,3)

		# return inputimage,self.image_filenames[index][:4]
		return inputimage,self.image_filenames[index]

	def __len__(self):
		return len(self.image_filenames)

if __name__ == '__main__':

	EPSILON=0.0000001
	PI = np.pi

	width = 256
	height = 256
	FPS = 12

	fourcc = cv2.VideoWriter_fourcc(*'XVID')

	mydevice=torch.device('cuda')

	parser = argparse.ArgumentParser(description='DeepRendering-implemention')
	parser.add_argument('--input_dir', required=True, type=str, default=0, help='0: my results considered; 1: only their model test on real data')
	parser.add_argument('--save_dir', required=True, type=str, default=0, help='0: my results considered; 1: only their model test on real data')
	parser.add_argument('--batch_size', type=int, default=5, help='0: my results considered; 1: only their model test on real data')
	parser.add_argument('--WhichTest', required=True, help='0: my results considered; 1: only their model test on real data')
	# parser.add_argument('--mode', default='Syn', help='Syn or Real')
	parser.add_argument('--name', default='Syn', help='Syn or Real')
	parser.add_argument('--video', action='store_true', help='make video for rendering images')
	parser.add_argument('--gif', action='store_true', help='make gif for rendering images')
	parser.add_argument('--load_light_syn', action='store_true', help='Syn or Real')
	parser.add_argument('--load_light_real', action='store_true', help='Syn or Real')
	parser.add_argument('--CoCamLi', action='store_true', help='colocated camera or not')        
	parser.add_argument('--image', action='store_true', help='just dealing with images')        
	parser.add_argument('--lightpos', type=float, default=0.5, help='light position')

	opt = parser.parse_args()


	input_dir = opt.input_dir
	if not os.path.exists(input_dir):
		raise('error no file found')

	# if opt.WhichTest=='VA'or opt.WhichTest=='gt' or opt.WhichTest=='MY':
	# 	batchsize=5
	# elif opt.WhichTest=='IN':
	# 	batchsize=4
	# else:
	# 	batchsize=1
	batchsize=opt.batch_size
	# if opt.mode =='Real' and opt.WhichTest=='MY':
	# 	batchsize=5


	if opt.WhichTest=='VA' or opt.WhichTest=='MY'or opt.WhichTest=='gt' or opt.WhichTest=='Gao'or opt.WhichTest=='Opt':
		test_set = DataLoaderHelper_LoadSingleImage(input_dir)
	else:
		test_set = DataLoaderHelper(input_dir)

	test_data = DataLoader(dataset=test_set, batch_size=batchsize, shuffle=False)

	Position_map_cpu=PositionMap(256,256,3).to(mydevice)

	if opt.video:
		N_renderings=int(FPS*3)
		save_dir=join(opt.save_dir,'video_{:s}'.format(opt.name))
	elif opt.gif or opt.image:
		N_renderings=int(FPS*3)
		save_dir=join(opt.save_dir,'gif_{:s}'.format(opt.name))
	else:
		save_dir=join(opt.save_dir,'Rerendered_{:s}'.format(opt.name))
		N_renderings=1

	if not os.path.exists(save_dir):
		os.makedirs(save_dir)

	# MyFile=open('render_center.txt','w')
	# MyFile=open('render_center_newmodel.txt','w')

	print(len(test_data))

	# if opt.load_light:
	# 	lightfolder = listdir ('F:/LoganZhou/Research2/Dataset/Light/MGReal7_train')

	for i,mydata in enumerate(test_data):

		data,dataname=mydata

		# i_name = dataname[0]
		# if opt.WhichTest=='MY':
		# 	i_name = dataname[0][0:len(dataname[0])-16]
		# elif opt.WhichTest=='VA':
		# 	# i_name = dataname[0][0:len(dataname[0])-30] # des19
		# 	i_name = dataname[0][0:len(dataname[0])-13] # des18
		# elif opt.WhichTest=="Gao":
		# 	i_name = dataname[0][0:len(dataname[0])-9]
		# elif opt.WhichTest=="Opt":
		# 	i_name = dataname[0][0:len(dataname[0])-8]

		if i<10:
			i_name='000'+str(i)
		elif i<100:
			i_name='00'+str(i)
		else:
			i_name='0'+str(i)

		# if opt.video:
		# 	videopath=join(save_dir,'./video_{}.avi'.format(i_name))
		# 	video = VideoWriter(videopath, fourcc, float(FPS), (width, height))

		if opt.gif:
			gif = []
			gifpath=join(save_dir,'gif_{}.gif'.format(i_name))

		if batchsize==6 and opt.WhichTest=='MY':
			fake_diff=data[0:1,:,:,:].to(mydevice)**(2.2)
			fake_normal=data[2:3,:,:,:].to(mydevice)
			fake_rough=data[4:5,:,:,:].to(mydevice)
			fake_spec=data[5:6,:,:,:].to(mydevice)**(2.2)
		elif opt.WhichTest=='VA':
			fake_in=data[0:1,:,:,:].to(mydevice)
			fake_diff=data[2:3,:,:,:].to(mydevice)**(2.2)
			fake_normal=data[1:2,:,:,:].to(mydevice)
			fake_rough=data[3:4,:,:,:].to(mydevice)
			fake_spec=data[4:5,:,:,:].to(mydevice)**(2.2)
		elif opt.WhichTest=='gt':
			fake_diff=data[0:1,:,:,:].to(mydevice)**(2.2)
			fake_normal=data[2:3,:,:,:].to(mydevice)
			fake_rough=data[3:4,:,:,:].to(mydevice)
			fake_spec=data[4:5,:,:,:].to(mydevice)**(2.2)
		elif batchsize==5 and (opt.WhichTest=='MY' or opt.WhichTest=='Gao'or opt.WhichTest=='Opt'):
			fake_diff=data[0:1,:,:,:].to(mydevice)**(2.2)
			fake_normal=data[2:3,:,:,:].to(mydevice)
			fake_rough=data[3:4,:,:,:].to(mydevice)
			fake_spec=data[4:5,:,:,:].to(mydevice)**(2.2)
		else:
			fake_normal,fake_diff,fake_rough,fake_spec=mydata
			fake_diff=fake_diff.to(mydevice)**(2.2)
			fake_spec=fake_spec.to(mydevice)**(2.2)
			fake_normal=fake_normal.to(mydevice)
			fake_rough=fake_rough.to(mydevice)

		# cat_image=torch.cat((fake_diff**(1/2.2),fake_normal,fake_rough,fake_spec**(1/2.2)),dim=2)


		
		if opt.image:
			# input_name = dataname[0].split('.')
			# norm_name = dataname[1].split('.')
			# diff_name = dataname[2].split('.')
			# rou_name = dataname[3].split('.')
			# spec_name = dataname[4].split('.')

			# print(input_name[0])
			# print(norm_name[0])
			# print(diff_name[0])
			# print(rou_name[0])
			# print(spec_name[0])

			# saveimage(fake_diff[0,:,:,:].cpu().numpy(), join(save_dir,'{}.jpg'.format(diff_name[0])), gamma=True)
			# saveimage(fake_normal[0,:,:,:].cpu().numpy(), join(save_dir,'{}.jpg'.format(norm_name[0])), gamma=False)
			# saveimage(fake_rough[0,:,:,:].cpu().numpy(), join(save_dir,'{}.jpg'.format(rou_name[0])), gamma=False)
			# saveimage(fake_spec[0,:,:,:].cpu().numpy(), join(save_dir,'{}.jpg'.format(spec_name[0])), gamma=True)
			# saveimage(fake_in[0,:,:,:].cpu().numpy(), join(save_dir,'{}.jpg'.format(input_name[0])), gamma=False)
			print(fake_diff.dtype)
			saveimage(fake_diff[0,:,:,:].cpu().numpy(), join(save_dir,'{}.jpg'.format(i)), gamma=True)
			saveimage(fake_normal[0,:,:,:].cpu().numpy(), join(save_dir,'{}.jpg'.format(i)), gamma=False)
			saveimage(fake_rough[0,:,:,:].cpu().numpy(), join(save_dir,'{}.jpg'.format(i)), gamma=False)
			saveimage(fake_spec[0,:,:,:].cpu().numpy(), join(save_dir,'{}.jpg'.format(i)), gamma=True)
			saveimage(fake_in[0,:,:,:].cpu().numpy(), join(save_dir,'{}.jpg'.format(i)), gamma=False)

		else:
			# npy_path='F:/LoganZhou/Research/OtherPaper/Pix2Pix/Paper/EG2021/comparison/Realimages'
			npy_path='F:/LoganZhou/Research/OtherPaper/Comparison/Syn'
			# npy_path='F:/LoganZhou/Research/OtherPaper/Comparison/Real'
			if opt.load_light_real:

				LightPo1=torch.from_numpy(load_light_txt(join('F:/LoganZhou/Research2/Dataset/Light/MGReal7_train','{}'.format(lightfolder[int(i)])))).to(mydevice).float()
				LightPo2=torch.from_numpy(load_light_txt(join('F:/LoganZhou/Research2/Dataset/Light/MGReal2_test','{}'.format(lightfolder[int(i)])))).to(mydevice).float()

				txtfile=open(join('F:/LoganZhou/Research/OtherPaper/MultiGao/singleimage/data/Real_MG/MG_wlvs1','{}'.format(lightfolder[int(i)])),'w')
				for i in range(1):
					txtfile.write('{:.6f},{:.6f},{:.6f}\n'.format(LightPo1[i][0].cpu().numpy(),LightPo1[i][1].cpu().numpy(),LightPo1[i][2].cpu().numpy()))
					# txtfile.write('{:.6f},{:.6f},{:.6f}\n'.format(LightPos[i][0].cpu().numpy(),LightPos[i][1].cpu().numpy(),LightPos[i][2].cpu().numpy()))
				txtfile.close()


				LightPo = torch.cat((LightPo1[1:,...],LightPo2),dim=0)
				print(LightPo.dtype)
				N_renderings=LightPo.shape[0]
			elif opt.load_light_syn:
				LightPo=torch.from_numpy(np.load('F:/LoganZhou/Research/OtherPaper/Comparison/Syn/light.npy')).to(mydevice)
				# LightPos=torch.from_numpy(np.load('F:/LoganZhou/Research2/Dataset/Light/5_Co/light.npy')).to(mydevice)
				N_renderings=LightPo.shape[0]
			else:
				# LightPo=N_Light(N_renderings,0.9,mydevice)
				LightPo=N_Light(N_renderings,opt.lightpos,mydevice) # gif
				# np.save(join(npy_path , 'light.npy'),LightPo.cpu())

			# fake_render = SingleRender_NumberPointLight(fake_diff, fake_spec, fake_normal, fake_rough, LightPo, Position_map_cpu, mydevice, N_renderings, opt.CoCamLi)	
			fake_render = SingleRender_NumberPointLight_FixedCamera(fake_diff, fake_spec, fake_normal, fake_rough, LightPo, Position_map_cpu, mydevice, N_renderings, False)	



			for j in range(N_renderings):

				if opt.video:
					fake_video = fake_render[j,:,:,:].cpu().numpy()**(1/2.2)*255.0

					fake_video = np.clip(fake_video, 0, 255)
					fake_video = fake_video.astype(np.uint8)
					RGB_img = cv2.cvtColor(fake_video, cv2.COLOR_RGB2BGR)

					video.write(RGB_img)
					# saveimage(fake_render[j,:,:,:].cpu().numpy(), join(save_dir,'{}fakerender_at{}.jpg'.format(i,j)))
				elif opt.gif:

					gif_numpy = fake_render[j,:,:,:].cpu().numpy()**(1/2.2)*255.0
					gif_numpy = np.clip(gif_numpy, 0, 255)
					gif_numpy = gif_numpy.astype(np.uint8)
					gif.append(gif_numpy)
					# gif_Image=Image.fromarray(gif_numpy)
					# gif.append(gif_Image)

					# saveimage(fake_render[j,:,:,:].cpu().numpy(), join(save_dir,'{}fakerender_at{}.jpg'.format(i_name,j)))
				else:
					# saveimage(fake_render[j,:,:,:].cpu().numpy(), join(save_dir,'{}_input.jpg'.format(i_name)))
					saveimage(fake_render[j,:,:,:].cpu().numpy(), join(save_dir,'{}_{}.png'.format(i_name,j+1)))

			if opt.video:			
				video.release()
				print('save the video',i)

			if opt.gif:
				# gif[0].save(gifpath, save_all=True, append_images=gif[1:], optimize=False, duration=4, loop=0)
				imageio.mimwrite(gifpath, gif, format='GIF', fps=FPS)
				print('save the gif',i)
				# print(gifpath)
				# optimize(gifpath)


		# MyFile.write('the {}th input image: \n'.format(i))
		# MyFile.write('paper loss:{:.4f}, nolight loss:{:.4f}, light loss:{:.4f} \n'.format(MyDifference,MyDifference_nolight,MyDifference_light))
		# MyFile.write('Random_Light, paper loss:{:.4f}, nolight loss:{:.4f}, light loss:{:.4f} \n'.format(Difference.item(),Difference_nolight.item(),Difference_light.item()))

	# TotalLoss=TotalLoss/iterations
	# TotalLoss_nolight=TotalLoss_nolight/iterations
	# TotalLoss_light=TotalLoss_light/iterations
	# # TheirDifference_total=TheirDifference/iterations
	# print('|paper loss:',TotalLoss,'|nolight loss:',TotalLoss_nolight,'|light loss:',TotalLoss_light)

	# MyFile.write('Total paper loss:{:.4f}, nolight loss:{:.4f}, light loss:{:.4f} \n'.format(TotalLoss,TotalLoss_nolight,TotalLoss_light))
	# MyFile.close()

	# print('final loss (averaged on all validation datasets) of my test: ',MyDifference_total,'| their test: ', TheirDifference_total)
		


