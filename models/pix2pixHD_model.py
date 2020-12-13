import numpy as np
import torch
import os
from torch.autograd import Variable
from util.image_pool import ImagePool
from util.util import *
from .base_model import BaseModel
from . import networks
from .renderer import *
import matplotlib.pyplot as plt


class Pix2PixHDModel(BaseModel):
	def name(self):
		return 'Pix2PixHDModel'
	
	def init_loss_filter(self, use_gan_feat_loss, use_vgg_loss, use_l1_loss, real_train):
		flags = (True, use_gan_feat_loss, use_vgg_loss, use_l1_loss, True, True, True, real_train, real_train)
		def loss_filter(g_gan, g_gan_feat, g_vgg, g_l1, d_real, d_fake, light_l1, real_d_real, real_d_fake):
			return [l for (l,f) in zip((g_gan, g_gan_feat, g_l1, d_real, d_fake, light_l1, real_d_real, real_d_fake),flags) if f]
		return loss_filter

	def init_loss_filter_L1(self, use_l1_loss):
		flags = (use_l1_loss, True)
		def loss_filter(g_l1, light_l1):
			return [l for (l,f) in zip((g_l1, light_l1),flags) if f]
		return loss_filter

	def init_loss_filter_1D(self, use_gan_feat_loss, use_vgg_loss, use_l1_loss, real_train):
		flags = (True, use_gan_feat_loss, use_vgg_loss, use_l1_loss,True,True, True, real_train, real_train)
		def loss_filter_1D(g_gan,g_gan_feat, g_vgg, g_l1,d_real_render,d_fake_render,light_l1, real_d_real, real_d_fake):
			return [l for (l,f) in zip((g_gan, g_gan_feat,g_vgg, g_l1,d_real_render,d_fake_render, light_l1, real_d_real, real_d_fake),flags) if f]
		return loss_filter_1D

	def init_loss_filter_4D(self, use_gan_feat_loss, use_vgg_loss, use_l1_loss, real_train):
		flags = (True, use_gan_feat_loss, use_vgg_loss, use_l1_loss,True,True,True,True,True,True,True,True, True, real_train, real_train)
		def loss_filter_4D(g_gan,g_gan_feat, g_vgg, g_l1,d_real_norm,d_fake_norm,d_real_diff,d_fake_diff,d_real_rough,d_fake_rough,d_real_spec,d_fake_spec, light_l1, real_d_real, real_d_fake):
			return [l for (l,f) in zip((g_gan, g_gan_feat, g_vgg, g_l1,d_real_norm,d_fake_norm,d_real_diff,d_fake_diff,d_real_rough,d_fake_rough,d_real_spec,d_fake_spec, 
										light_l1, real_d_real, real_d_fake),flags) if f]
		return loss_filter_4D

	def init_loss_filter_5D(self, use_gan_feat_loss, use_vgg_loss, use_l1_loss, real_train):
		flags = (True, use_gan_feat_loss, use_vgg_loss, use_l1_loss,True,True,True,True,True,True,True,True,True,True, True, real_train, real_train)
		def loss_filter_5D(g_gan,g_gan_feat, g_vgg, g_l1,d_real_norm,d_fake_norm,d_real_diff,d_fake_diff,d_real_rough,d_fake_rough,d_real_spec,d_fake_spec,d_real_render,d_fake_render,
							 light_l1, real_d_real, real_d_fake):
			return [l for (l,f) in zip((g_gan, g_gan_feat, g_vgg, g_l1,d_real_norm,d_fake_norm,d_real_diff,d_fake_diff,d_real_rough,d_fake_rough,d_real_spec,d_fake_spec,
									d_real_render,d_fake_render, light_l1, real_d_real, real_d_fake),flags) if f]
		return loss_filter_5D

	def initialize(self, opt):
		BaseModel.initialize(self, opt)
		if opt.resize_or_crop != 'none' or not opt.isTrain: # when training at full res this causes OOM
			torch.backends.cudnn.benchmark = True
		self.isTrain = opt.isTrain
		self.use_features = opt.instance_feat or opt.label_feat
		self.gen_features = self.use_features and not self.opt.load_features
		input_nc = opt.label_nc if opt.label_nc != 0 else opt.input_nc

		##### define networks        
		# Generator network
		netG_input_nc = input_nc        
		if not opt.no_instance:
			netG_input_nc += 1
		if self.use_features:
			netG_input_nc += opt.feat_num                  
		
		self.netG = networks.define_G(netG_input_nc, opt.output_nc, opt.rough_nc, opt.ngf, opt.netG, opt.use_dropout_G,
									  opt.n_downsample_global, opt.n_blocks_global, opt.n_blocks_branch, opt.n_local_enhancers, 
									  opt.n_blocks_local, opt.norm, gpu_ids=self.gpu_ids)        

		# Discriminator network
		if self.isTrain:
			
			use_sigmoid = opt.no_lsgan

			if opt.MyTest=='ALL_4D':
				netD_input_nc = input_nc + opt.input_nc_D

				self.netD_Norm = networks.define_D(netD_input_nc, opt.use_dropout_D, opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid, opt.num_D, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)               
				self.netD_Diff = networks.define_D(netD_input_nc, opt.use_dropout_D, opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid, opt.num_D, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)               
				self.netD_Rough = networks.define_D(netD_input_nc, opt.use_dropout_D, opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid, opt.num_D, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)               
				self.netD_Spec = networks.define_D(netD_input_nc, opt.use_dropout_D, opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid, opt.num_D, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)               
			elif opt.MyTest=='ALL_1D_Render':
				netD_input_nc = input_nc + 3
				self.netD_Render = networks.define_D(netD_input_nc, opt.use_dropout_D, opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid, opt.num_D, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids) 

			elif opt.MyTest=='ALL_5D_Render':

				netD_input_nc = input_nc + opt.input_nc_D
				netD_input_nc_render = input_nc + 3

				self.netD_Norm = networks.define_D(netD_input_nc, opt.use_dropout_D, opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid, opt.num_D, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)               
				self.netD_Diff = networks.define_D(netD_input_nc, opt.use_dropout_D, opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid, opt.num_D, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)               
				self.netD_Rough = networks.define_D(netD_input_nc, opt.use_dropout_D, opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid, opt.num_D, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)               
				self.netD_Spec = networks.define_D(netD_input_nc, opt.use_dropout_D, opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid, opt.num_D, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)               
				self.netD_Render = networks.define_D(netD_input_nc_render, opt.use_dropout_D, opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid, opt.num_D, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)               
			elif opt.MyTest=='L1_Render' or opt.MyTest=='L1' :
				print('no discriminator')	
			else:
				raise ('error !')

			if opt.real_train:
				netD_input_nc_render = input_nc + 3
				self.netD_Render_Real = networks.define_D(netD_input_nc_render, opt.use_dropout_D, opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid, opt.num_D, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)               


		### Encoder network
		if self.gen_features:          
			self.netE = networks.define_G(opt.output_nc, opt.feat_num, opt.nef, 'encoder', 
										  opt.n_downsample_E, norm=opt.norm, gpu_ids=self.gpu_ids)  
		if self.opt.verbose:
				print('---------- Networks initialized -------------')

		# load networks
		if not self.isTrain or opt.continue_train or opt.load_pretrain:
			pretrained_path = '' if not self.isTrain else opt.load_pretrain
			self.load_network(self.netG, 'G', opt.which_epoch, pretrained_path)            
			if self.isTrain:
				if opt.MyTest=='ALL_4D':
					self.load_network(self.netD_Norm, 'D_Norm', opt.which_epoch, pretrained_path)  
					self.load_network(self.netD_Diff, 'D_Diff', opt.which_epoch, pretrained_path)  
					self.load_network(self.netD_Rough, 'D_Rough', opt.which_epoch, pretrained_path)  
					self.load_network(self.netD_Spec, 'D_Spec', opt.which_epoch, pretrained_path) 
				elif opt.MyTest=='ALL_5D_Render':
					self.load_network(self.netD_Norm, 'D_Norm', opt.which_epoch, pretrained_path)  
					self.load_network(self.netD_Diff, 'D_Diff', opt.which_epoch, pretrained_path)  
					self.load_network(self.netD_Rough, 'D_Rough', opt.which_epoch, pretrained_path)  
					self.load_network(self.netD_Spec, 'D_Spec', opt.which_epoch, pretrained_path)
					self.load_network(self.netD_Render, 'D_Render', opt.which_epoch, pretrained_path)  
				elif opt.MyTest=='ALL_1D_Render': 
					self.load_network(self.netD_Render, 'D_Render', opt.which_epoch, pretrained_path)  
				else:
					print('No D to load')

				if opt.real_train:
					self.load_network(self.netD_Render_Real, 'D_Render_Real', opt.which_epoch, pretrained_path)  

			if self.gen_features:
				self.load_network(self.netE, 'E', opt.which_epoch, pretrained_path)              

		self.criterionFeat = torch.nn.L1Loss()
		if not opt.no_vgg_loss:  
			print('VGG')          
			self.criterionVGG = networks.VGGLoss(self.gpu_ids)
		self.Position_map=PositionMap(256,256,3).cuda(self.gpu_ids[0])

		self.real_train = opt.real_train

		# set loss functions and optimizers
		if self.isTrain:
			if opt.pool_size > 0 and (len(self.gpu_ids)) > 1:
				raise NotImplementedError("Fake Pool Not Implemented for MultiGPU")
			self.fake_pool = ImagePool(opt.pool_size)
			self.old_lr = opt.lr

			# define loss functions
			if opt.MyTest=='ALL_4D':
				self.loss_filter = self.init_loss_filter_4D(not opt.no_ganFeat_loss, not opt.no_vgg_loss, not opt.no_l1_loss, opt.real_train)
				self.loss_names = self.loss_filter('G_GAN','G_GAN_Feat','G_VGG','G_L1','D_real_norm', 'D_fake_norm',
									'D_real_diff', 'D_fake_diff','D_real_rough', 'D_fake_rough','D_real_spec', 'D_fake_spec','light_l1', 'realD_real','realD_fake')
			elif opt.MyTest=='ALL_1D_Render':
				self.loss_filter = self.init_loss_filter_1D(not opt.no_ganFeat_loss, not opt.no_vgg_loss, not opt.no_l1_loss, opt.real_train)
				self.loss_names = self.loss_filter('G_GAN','G_GAN_Feat','G_VGG','G_L1','D_real_render','D_fake_render','light_l1','realD_real','realD_fake')
			elif opt.MyTest=='ALL_5D_Render':				
				self.loss_filter = self.init_loss_filter_5D(not opt.no_ganFeat_loss, not opt.no_vgg_loss, not opt.no_l1_loss,  opt.real_train)
				self.loss_names = self.loss_filter('G_GAN','G_GAN_Feat','G_VGG','G_L1','D_real_norm', 'D_fake_norm',
									'D_real_diff', 'D_fake_diff','D_real_rough', 'D_fake_rough','D_real_spec', 'D_fake_spec','D_real_render','D_fake_render','light_l1', 'realD_real','realD_fake')
			elif opt.MyTest=='L1' or opt.MyTest=='L1_Render':
				self.loss_filter = self.init_loss_filter_L1( not opt.no_l1_loss)
				self.loss_names = self.loss_filter('G_L1','light_l1')				
			else:
				raise('error')


			self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)   
  

			# initialize optimizers
			# optimizer G
			params = list(self.netG.parameters())
			self.optimizer_G = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))                            

			# optimizer D
			if opt.MyTest=='ALL_4D':
				params_Normal = list(self.netD_Norm.parameters())    
				self.optimizer_D_Norm = torch.optim.Adam(params_Normal, lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)   
				params_Diff = list(self.netD_Diff.parameters())    
				self.optimizer_D_Diff = torch.optim.Adam(params_Diff, lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)   
				params_Rough = list(self.netD_Rough.parameters())    
				self.optimizer_D_Rough = torch.optim.Adam(params_Rough, lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)   
				params_Spec = list(self.netD_Spec.parameters())    
				self.optimizer_D_Spec = torch.optim.Adam(params_Spec, lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay) 
			elif opt.MyTest=='ALL_1D_Render':
				params_Render = list(self.netD_Render.parameters())    
				self.optimizer_D_Render = torch.optim.Adam(params_Render, lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay) 	
			elif opt.MyTest=='ALL_5D_Render':
				params_Normal = list(self.netD_Norm.parameters())    
				self.optimizer_D_Norm = torch.optim.Adam(params_Normal, lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)   
				params_Diff = list(self.netD_Diff.parameters())    
				self.optimizer_D_Diff = torch.optim.Adam(params_Diff, lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)   
				params_Rough = list(self.netD_Rough.parameters())    
				self.optimizer_D_Rough = torch.optim.Adam(params_Rough, lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)   
				params_Spec = list(self.netD_Spec.parameters())    
				self.optimizer_D_Spec = torch.optim.Adam(params_Spec, lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay) 	
				params_Render = list(self.netD_Render.parameters())    
				self.optimizer_D_Render = torch.optim.Adam(params_Render, lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay) 			
			
			elif opt.MyTest=='L1' or opt.MyTest=='L1_Render':
				print('no optimization for D')


			if opt.real_train:
				params_Render_Real = list(self.netD_Render_Real.parameters())    
				self.optimizer_D_Render_Real = torch.optim.Adam(params_Render_Real, lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay) 			

		if self.opt.rand_light==0.3:
			# light randon 0.3
			self.Light_Mean=torch.tensor([-0.0019,0.00028,3.3212],device='cuda')
			self.Light_STD=torch.tensor([1.3158,1.3153,2.9029],device='cuda')
		elif self.opt.rand_light==0.5:
			# light random 0.9
			self.Light_Mean=torch.tensor([-0.0012,0.0018,3.0988],device='cuda')
			self.Light_STD=torch.tensor([1.6891,1.6833,2.7202],device='cuda')

		elif self.opt.rand_light==0.9:
			# light random 0.9
			self.Light_Mean=torch.tensor([0.0063,-0.0058,2.5871],device='cuda')
			self.Light_STD=torch.tensor([2.2784,2.2784,2.4076],device='cuda')


	def encode_input(self, label_map, inst_map=None, real_image=None, feat_map=None, infer=False):             
		
		if self.opt.label_nc == 0:
			input_label = label_map.data.cuda()
		else:
			# create one-hot vector for label map 
			size = label_map.size()
			oneHot_size = (size[0], self.opt.label_nc, size[2], size[3])
			input_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
			input_label = input_label.scatter_(1, label_map.data.long().cuda(), 1.0)
			if self.opt.data_type == 16:
				input_label = input_label.half()

		# get edges from instance map
		if not self.opt.no_instance:
			inst_map = inst_map.data.cuda()
			edge_map = self.get_edges(inst_map)
			input_label = torch.cat((input_label, edge_map), dim=1)         
		input_label = Variable(input_label, volatile=infer)

		# real images for training
		if real_image is not None:
			real_image = Variable(real_image.data.cuda())

		# instance map for feature encoding
		if self.use_features:
			# get precomputed feature maps
			if self.opt.load_features:
				feat_map = Variable(feat_map.data.cuda())
			if self.opt.label_feat:
				inst_map = label_map.cuda()

		return input_label, inst_map, real_image, feat_map

	def discriminate(self, input_label, test_image, use_pool=False, real=False):

		if self.MyTest == 'ALL_4D' and not real:
			##normal
			input_concat_normal = torch.cat((input_label, test_image[:,0:3,:,:].detach()), dim=1)           
			## diff
			input_concat_diff = torch.cat((input_label, test_image[:,3:6,:,:].detach()), dim=1)
			## rough
			input_concat_rough = torch.cat((input_label, test_image[:,6:9,:,:].detach()), dim=1)
			## spec
			input_concat_spec = torch.cat((input_label, test_image[:,9:12,:,:].detach()), dim=1)

			if use_pool:          
				fake_query_normal = self.fake_pool.query(input_concat_normal)
				fake_query_diff = self.fake_pool.query(input_concat_diff)
				fake_query_rough = self.fake_pool.query(input_concat_rough)
				fake_query_spec = self.fake_pool.query(input_concat_spec)
				
				return [self.netD_Norm.forward(fake_query_normal),
						self.netD_Diff.forward(fake_query_diff),
						self.netD_Rough.forward(fake_query_rough),
						self.netD_Spec.forward(fake_query_spec)]
			else:
				
				return [self.netD_Norm.forward(input_concat_normal),
						self.netD_Diff.forward(input_concat_diff),
						self.netD_Rough.forward(input_concat_rough),
						self.netD_Spec.forward(input_concat_spec)]

		elif self.MyTest =='ALL_5D_Render' and not real:
			##normal
			input_concat_normal = torch.cat((input_label, test_image[:,0:3,:,:].detach()), dim=1)           
			## diff
			input_concat_diff = torch.cat((input_label, test_image[:,3:6,:,:].detach()), dim=1)
			## rough
			input_concat_rough = torch.cat((input_label, test_image[:,6:9,:,:].detach()), dim=1)
			## spec
			input_concat_spec = torch.cat((input_label, test_image[:,9:12,:,:].detach()), dim=1)

			input_concat_render = torch.cat((input_label, test_image[:,12:15,:,:].detach()), dim=1)

			if use_pool:          
				fake_query_normal = self.fake_pool.query(input_concat_normal)
				fake_query_diff = self.fake_pool.query(input_concat_diff)
				fake_query_rough = self.fake_pool.query(input_concat_rough)
				fake_query_spec = self.fake_pool.query(input_concat_spec)
				fake_query_render = self.fake_pool.query(input_concat_render)
				
				return [self.netD_Norm.forward(fake_query_normal),
						self.netD_Diff.forward(fake_query_diff),
						self.netD_Rough.forward(fake_query_rough),
						self.netD_Spec.forward(fake_query_spec),
						self.netD_Render.forward(fake_query_render)]
			else:
				
				return [self.netD_Norm.forward(input_concat_normal),
						self.netD_Diff.forward(input_concat_diff),
						self.netD_Rough.forward(input_concat_rough),
						self.netD_Spec.forward(input_concat_spec),
						self.netD_Render.forward(input_concat_render)]

		elif self.MyTest =='ALL_1D_Render' and not real:		
			input_concat = torch.cat((input_label, test_image.detach()), dim=1)
			if use_pool:            
				fake_query = self.fake_pool.query(input_concat)
				return self.netD_Render.forward(fake_query)
			else:
				return self.netD_Render.forward(input_concat)
		elif real:
			# print('real image discriminate...')
			input_concat = torch.cat((input_label, test_image.detach()), dim=1)
			if use_pool:            
				fake_query = self.fake_pool.query(input_concat)
				return self.netD_Render_Real.forward(fake_query)
			else:
				return self.netD_Render_Real.forward(input_concat)			
				


	def forward(self, label, inst, image, feat, input_real, infer=False):
		# Encode Inputs
		input_syn, inst_map, real_syn, feat_map = self.encode_input(label, inst, image, feat)  
		
		# [-1,1] ->[0,1]
		real_image_temp=tensorNormalize(real_syn.permute(0,2,3,1))

		# augment input images
		if self.augment_input:
			#[B,C,W,H]
			real_synlight = Create_NumberPointLightPosition(self.opt.batchSize, self.opt.rand_light, self.gpu_ids[0])
			# camera_spos = Create_NumberPointLightPosition(self.opt.batchSize, 0.2, self.gpu_ids[0])

			input_syn=SingleRender_NumberPointLight_FixedCamera(real_image_temp[:,:,:,3:6], real_image_temp[:,:,:,9:12], real_image_temp[:,:,:,0:3],
									real_image_temp[:,:,:,6:9], real_synlight, self.Position_map, self.gpu_ids[0], self.opt.batchSize, self.opt.LowCam).permute(0,3,1,2)	
			
			# input_syn=SingleRender_NumberPointLightCamera(real_image_temp[:,:,:,3:6], real_image_temp[:,:,:,9:12], real_image_temp[:,:,:,0:3],
			# 						real_image_temp[:,:,:,6:9], real_synlight, camera_pos, self.Position_map, self.gpu_ids[0], self.opt.batchSize).permute(0,3,1,2)	

			input_syn=torch.clamp(input_syn,0,1)

			# linear to the log, [0,1] -> [-1,1]
			input_syn=2*input_syn**(1/2.2)-1
			# input_syn=2*logTensor(input_syn)-1


		# Fake Generation
		if self.use_features:
			if not self.opt.load_features:
				feat_map = self.netE.forward(real_syn, inst_map)                     
			input_concat = torch.cat((input_syn, feat_map), dim=1)                        
		else:
			input_concat = input_syn

		# add real images into input if needed
		if self.real_train:
			# gamma, [0,1] -> [-1,1]
			input_real_gamma=2*input_real**(1/2.2)-1
			input_concat = torch.cat((input_concat, input_real_gamma), dim=0)                        


		fake_image,fake_light = self.netG.forward(input_concat)

		# seperate syn and real images if needed
		if not self.real_train:
			fake_syn = fake_image
			fake_synlight = fake_light
		else:
			fake_syn = fake_image[0:int(self.opt.batchSize),...]
			fake_real = fake_image[int(self.opt.batchSize):int(self.opt.batchSize)+int(self.opt.real_batchSize)*2,...]
			if fake_light is not None:
				fake_synlight = fake_light[0:int(self.opt.batchSize),...]
				fake_reallight = fake_light[int(self.opt.batchSize):int(self.opt.batchSize)+int(self.opt.real_batchSize)*2,...]

		############################ Synthetic images training #####################
		# 4 discriminators
		if self.MyTest=='ALL_4D':

			# Fake Detection and Loss
			pred_fake_pool = self.discriminate(input_syn, fake_syn, use_pool=True)
			loss_D_fake_norm = self.criterionGAN(pred_fake_pool[0], False)        
			loss_D_fake_diff = self.criterionGAN(pred_fake_pool[1], False)        
			loss_D_fake_rough = self.criterionGAN(pred_fake_pool[2], False)        
			loss_D_fake_spec = self.criterionGAN(pred_fake_pool[3], False)        

			# Real Detection and Loss        
			pred_real = self.discriminate(input_syn, real_syn)
			loss_D_real_norm = self.criterionGAN(pred_real[0], True)
			loss_D_real_diff = self.criterionGAN(pred_real[1], True)
			loss_D_real_rough = self.criterionGAN(pred_real[2], True)
			loss_D_real_spec = self.criterionGAN(pred_real[3], True)
			# pred_real_average = 0.25*(pred_real[0] + pred_real[1] + pred_real[2] + pred_real[3])

			# GAN loss (Fake Passability Loss) 
			pred_fake_norm = self.netD_Norm.forward(torch.cat((input_syn, fake_syn[:,0:3,:,:]), dim=1))
			pred_fake_diff = self.netD_Diff.forward(torch.cat((input_syn, fake_syn[:,3:6,:,:]), dim=1))        
			pred_fake_rough = self.netD_Rough.forward(torch.cat((input_syn, fake_syn[:,6:9,:,:]), dim=1))        
			pred_fake_spec = self.netD_Spec.forward(torch.cat((input_syn, fake_syn[:,9:12,:,:]), dim=1))  
			# pred_fake_average = 0.25*(pred_fake_norm + pred_fake_diff + pred_fake_rough + pred_fake_spec)

			loss_G_GAN_norm = self.criterionGAN(pred_fake_norm, True)               
			loss_G_GAN_diff = self.criterionGAN(pred_fake_diff, True)               
			loss_G_GAN_rough = self.criterionGAN(pred_fake_rough, True)               
			loss_G_GAN_spec = self.criterionGAN(pred_fake_spec, True)               
			loss_G_GAN = 0.25*(loss_G_GAN_norm + loss_G_GAN_diff + loss_G_GAN_rough + loss_G_GAN_spec)
		# 1 discriminators (with render)
		elif self.MyTest=='ALL_1D_Render':

			## convert from [-1,1] to [0,1]
			fake_image_temp=tensorNormalize(fake_syn.permute(0,2,3,1))

			LightPosition=Create_NumberPointLightPosition(self.opt.batchSize, 0.95, self.gpu_ids[0])
			# CameraPosition=Create_NumberPointLightPosition(1, 0.2, self.gpu_ids[0])
			
			#[B,C,W,H] HDR image
			# Render_Fake=SingleRender_camera(fake_image_temp[:,:,:,3:6], fake_image_temp[:,:,:,9:12], fake_image_temp[:,:,:,0:3], 
			# 													fake_image_temp[:,:,:,6:9], LightPosition,CameraPosition, self.Position_map, self.gpu_ids[0]).permute(0,3,1,2)
			# Render_Real=SingleRender_camera(real_image_temp[:,:,:,3:6], real_image_temp[:,:,:,9:12], real_image_temp[:,:,:,0:3],
			# 													real_image_temp[:,:,:,6:9], LightPosition,CameraPosition, self.Position_map, self.gpu_ids[0]).permute(0,3,1,2)
			
			Render_Fake=SingleRender_NumberPointLight_FixedCamera(fake_image_temp[:,:,:,3:6], fake_image_temp[:,:,:,9:12], fake_image_temp[:,:,:,0:3],
									fake_image_temp[:,:,:,6:9], LightPosition, self.Position_map, self.gpu_ids[0], self.opt.batchSize, self.opt.LowCam).permute(0,3,1,2)	

			Render_Real=SingleRender_NumberPointLight_FixedCamera(real_image_temp[:,:,:,3:6], real_image_temp[:,:,:,9:12], real_image_temp[:,:,:,0:3],
									real_image_temp[:,:,:,6:9], LightPosition, self.Position_map, self.gpu_ids[0], self.opt.batchSize, self.opt.LowCam).permute(0,3,1,2)	

			#[B,C,W,H] LDR image
			Render_Fake=torch.clamp(Render_Fake,0,1)
			Render_Real=torch.clamp(Render_Real,0,1)

			#gamma correction LDR to log domain
			Render_Fake=2*(Render_Fake+EPSILON)**(1/2.2)-1
			Render_Real=2*(Render_Real+EPSILON)**(1/2.2)-1
			renderimage = {'Fake':Render_Fake,'Real':Render_Real}

			# Fake Detection and Loss
			pred_fake_pool = self.discriminate(input_syn, Render_Fake, use_pool=True)
			loss_D_fake_render = self.criterionGAN(pred_fake_pool, False)        

			# Real Detection and Loss        
			pred_real = self.discriminate(input_syn, Render_Real)
			loss_D_real_render = self.criterionGAN(pred_real, True)

			# GAN loss (Fake Passability Loss) 
			pred_fake = self.netD_Render.forward(torch.cat((input_syn, Render_Fake), dim=1))        
			loss_G_GAN = self.criterionGAN(pred_fake, True)*self.opt.lambda_render
		# 5 discriminator
		elif self.MyTest=='ALL_5D_Render':

			## convert from [-1,1] to [0,1]
			fake_image_temp=tensorNormalize(fake_syn.permute(0,2,3,1))

			LightPosition=Create_NumberPointLightPosition(self.opt.batchSize, 0.9, self.gpu_ids[0])
			# CameraPosition=Create_NumberPointLightPosition(1, 0.2, self.gpu_ids[0])
			
			#[B,C,W,H] HDR image
			# Render_Fake=SingleRender_camera(fake_image_temp[:,:,:,3:6], fake_image_temp[:,:,:,9:12], fake_image_temp[:,:,:,0:3], 
			# 													fake_image_temp[:,:,:,6:9], LightPosition,CameraPosition, self.Position_map, self.gpu_ids[0]).permute(0,3,1,2)
			# Render_Real=SingleRender_camera(real_image_temp[:,:,:,3:6], real_image_temp[:,:,:,9:12], real_image_temp[:,:,:,0:3],
			# 													real_image_temp[:,:,:,6:9], LightPosition,CameraPosition, self.Position_map, self.gpu_ids[0]).permute(0,3,1,2)
			
			Render_Fake=SingleRender_NumberPointLight_FixedCamera(fake_image_temp[:,:,:,3:6], fake_image_temp[:,:,:,9:12], fake_image_temp[:,:,:,0:3],
									fake_image_temp[:,:,:,6:9], LightPosition, self.Position_map, self.gpu_ids[0], self.opt.batchSize, self.opt.LowCam).permute(0,3,1,2)	

			Render_Real=SingleRender_NumberPointLight_FixedCamera(real_image_temp[:,:,:,3:6], real_image_temp[:,:,:,9:12], real_image_temp[:,:,:,0:3],
									real_image_temp[:,:,:,6:9], LightPosition, self.Position_map, self.gpu_ids[0], self.opt.batchSize, self.opt.LowCam).permute(0,3,1,2)	

			#[B,C,W,H] LDR image
			Render_Fake=torch.clamp(Render_Fake,0,1)
			Render_Real=torch.clamp(Render_Real,0,1)

			#gamma correction LDR to log domain
			Render_Fake=2*(Render_Fake+EPSILON)**(1/2.2)-1
			Render_Real=2*(Render_Real+EPSILON)**(1/2.2)-1

			renderimage = {'Fake':Render_Fake,'Real':Render_Real}

			concat_Fake=torch.cat((fake_syn,Render_Fake),dim=1)
			concat_Real=torch.cat((real_syn,Render_Real),dim=1)

			# print(concat_Real.shape)
			# Fake Detection and Loss
			pred_fake_pool = self.discriminate(input_syn, concat_Fake, use_pool=True)
			loss_D_fake_norm = self.criterionGAN(pred_fake_pool[0], False)        
			loss_D_fake_diff = self.criterionGAN(pred_fake_pool[1], False)        
			loss_D_fake_rough = self.criterionGAN(pred_fake_pool[2], False)        
			loss_D_fake_spec = self.criterionGAN(pred_fake_pool[3], False)        
			loss_D_fake_render = self.criterionGAN(pred_fake_pool[4], False)        

			# Real Detection and Loss        
			pred_real = self.discriminate(input_syn, concat_Real)
			loss_D_real_norm = self.criterionGAN(pred_real[0], True)
			loss_D_real_diff = self.criterionGAN(pred_real[1], True)
			loss_D_real_rough = self.criterionGAN(pred_real[2], True)
			loss_D_real_spec = self.criterionGAN(pred_real[3], True)
			loss_D_real_render = self.criterionGAN(pred_real[4], True)

			# GAN loss (Fake Passability Loss) 
			pred_fake_norm = self.netD_Norm.forward(torch.cat((input_syn, fake_syn[:,0:3,:,:]), dim=1))
			pred_fake_diff = self.netD_Diff.forward(torch.cat((input_syn, fake_syn[:,3:6,:,:]), dim=1))        
			pred_fake_rough = self.netD_Rough.forward(torch.cat((input_syn, fake_syn[:,6:9,:,:]), dim=1))        
			pred_fake_spec = self.netD_Spec.forward(torch.cat((input_syn, fake_syn[:,9:12,:,:]), dim=1))  
			pred_fake_render = self.netD_Render.forward(torch.cat((input_syn, Render_Fake), dim=1))  

			loss_G_GAN_norm = self.criterionGAN(pred_fake_norm, True)               
			loss_G_GAN_diff = self.criterionGAN(pred_fake_diff, True)               
			loss_G_GAN_rough = self.criterionGAN(pred_fake_rough, True)               
			loss_G_GAN_spec = self.criterionGAN(pred_fake_spec, True)
			loss_G_GAN_render = self.criterionGAN(pred_fake_render, True)
			loss_G_GAN = 0.2*(loss_G_GAN_norm + loss_G_GAN_diff + loss_G_GAN_rough + loss_G_GAN_spec + loss_G_GAN_render*self.opt.lambda_render)

			renderimage = {'Fake':Render_Fake,'Real':Render_Real}
		# 0 discriminators 
		elif self.MyTest=='L1_Render':# or self.MyTest=='L1':

			## convert from [-1,1] to [0,1]
			fake_image_temp=tensorNormalize(fake_syn.permute(0,2,3,1))

			LightPosition=Create_NumberPointLightPosition(self.opt.batchSize, 0.9, self.gpu_ids[0])
			# CameraPosition=Create_NumberPointLightPosition(1, 0.2, self.gpu_ids[0])
			
			#[B,C,W,H] HDR image
			Render_Fake=SingleRender_NumberPointLight_FixedCamera(fake_image_temp[:,:,:,3:6], fake_image_temp[:,:,:,9:12], fake_image_temp[:,:,:,0:3],
									fake_image_temp[:,:,:,6:9], LightPosition, self.Position_map, self.gpu_ids[0], self.opt.batchSize, self.opt.LowCam).permute(0,3,1,2)	

			Render_Real=SingleRender_NumberPointLight_FixedCamera(real_image_temp[:,:,:,3:6], real_image_temp[:,:,:,9:12], real_image_temp[:,:,:,0:3],
									real_image_temp[:,:,:,6:9], LightPosition, self.Position_map, self.gpu_ids[0], self.opt.batchSize, self.opt.LowCam).permute(0,3,1,2)	

			#[B,C,W,H] LDR image
			Render_Fake=torch.clamp(Render_Fake,0,1)
			Render_Real=torch.clamp(Render_Real,0,1)

			#gamma correction LDR to log domain
			Render_Fake=2*(Render_Fake+EPSILON)**(1/2.2)-1
			Render_Real=2*(Render_Real+EPSILON)**(1/2.2)-1

			renderimage = {'Fake':Render_Fake,'Real':Render_Real}

		# else:
		# 	# Fake Detection and Loss
		# 	pred_fake_pool = self.discriminate(input_syn, fake_syn, use_pool=True)
		# 	loss_D_fake = self.criterionGAN(pred_fake_pool, False)        

		# 	# Real Detection and Loss        
		# 	pred_real = self.discriminate(input_syn, real_syn)
		# 	loss_D_real = self.criterionGAN(pred_real, True)

		# 	# GAN loss (Fake Passability Loss) 
		# 	pred_fake = self.netD.forward(torch.cat((input_syn, fake_syn), dim=1))        
		# 	loss_G_GAN = self.criterionGAN(pred_fake, True)  
			# print('pred_fake',pred_fake)

		# GAN feature matching loss
		loss_G_GAN_Feat = 0
		if not self.opt.no_ganFeat_loss and self.MyTest != 'L1' and self.MyTest != 'L1_Render':
			feat_weights = 4.0 / (self.opt.n_layers_D + 1)
			D_weights = 1.0 / self.opt.num_D
			if self.MyTest =='ALL_4D':
				pre_fake_list=[pred_fake_norm,pred_fake_diff,pred_fake_rough,pred_fake_spec]
				for k in range(4):
					for i in range(self.opt.num_D):
						for j in range(len(pre_fake_list[k][i])-1):
							loss_G_GAN_Feat += 0.25 * D_weights * feat_weights * self.criterionFeat(pre_fake_list[k][i][j], pred_real[k][i][j].detach()) * self.opt.lambda_feat

			elif self.MyTest =='ALL_5D_Render':
				pre_fake_list=[pred_fake_norm,pred_fake_diff,pred_fake_rough,pred_fake_spec,pred_fake_render]
				for k in range(5):
					for i in range(self.opt.num_D):
						for j in range(len(pre_fake_list[k][i])-1):
							# more weight on the rendering discriminator
							if k==4:
								loss_G_GAN_Feat += 0.2 * self.opt.lambda_render * D_weights * feat_weights * self.criterionFeat(pre_fake_list[k][i][j], pred_real[k][i][j].detach()) * self.opt.lambda_feat	
							else:
								loss_G_GAN_Feat += 0.2 * D_weights * feat_weights * self.criterionFeat(pre_fake_list[k][i][j], pred_real[k][i][j].detach()) * self.opt.lambda_feat
			
			elif self.MyTest =='ALL_1D_Render':
				for i in range(self.opt.num_D):
					for j in range(len(pred_fake[i])-1):
						loss_G_GAN_Feat += D_weights * feat_weights * \
							self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach()) * self.opt.lambda_feat

		# VGG feature loss
		loss_G_VGG = 0
		if not self.opt.no_vgg_loss:
			## only compute vgg loss for the diffuse map
			if self.MyTest=='Diff':
				fake_vg=fake_syn
				real_vg=real_syn
			elif self.MyTest=='ALL_1D_Render' or self.MyTest=='ALL_5D_Render':
				fake_vg=VGGpreprocess(tensorNormalize(Render_Fake))	
				real_vg=VGGpreprocess(tensorNormalize(Render_Real))	
			else:
				fake_vg=VGGpreprocess(tensorNormalize(fake_syn[:,3:6,:,:]))	
				real_vg=VGGpreprocess(tensorNormalize(real_syn[:,3:6,:,:]))	

			loss_G_VGG = self.criterionVGG(fake_vg, real_vg) * self.opt.lambda_feat

		# L1 loss
		loss_G_L1 = 0
		if not self.opt.no_l1_loss:
			if self.MyTest=='ALL_1D_Render':
				loss_G_L1 = ( self.criterionFeat(fake_syn, real_syn)*4 + self.criterionFeat(Render_Fake, Render_Real)*self.opt.lambda_render )*0.2 * self.opt.lambda_l1

			elif self.MyTest=='ALL_5D_Render' or self.MyTest=='L1_Render':
				loss_G_L1 = ( self.criterionFeat(fake_syn, real_syn) + self.criterionFeat(Render_Fake, Render_Real)*self.opt.lambda_render ) * self.opt.lambda_l1

			elif self.MyTest=='ALL_4D' or self.MyTest=='L1':
				loss_G_L1 = self.criterionFeat(fake_syn, real_syn) * self.opt.lambda_l1
			else:
				raise('error')

		# L1 loss for light position
		loss_Light_L1=0
		Light=None
		if fake_synlight is not None:
			loss_Light_L1 = self.criterionFeat(fake_synlight, NormMeanStd(real_synlight,self.Light_Mean,self.Light_STD)) * self.opt.lambda_l1

			# syn_rendered_estilight=SingleRender_NumberPointLight_FixedCamera(real_image_temp[:,:,:,3:6], real_image_temp[:,:,:,9:12], real_image_temp[:,:,:,0:3],
			# 						real_image_temp[:,:,:,6:9], Inverese_NormMeanStd(fake_synlight, self.Light_Mean, self.Light_STD), self.Position_map, self.gpu_ids[0], self.opt.batchSize, self.opt.LowCam).permute(0,3,1,2)
			# loss_Light_L1 += self.criterionFeat(syn_rendered_estilight,input_syn)* self.opt.lambda_l1
			# loss_Light_L1 = loss_Light_L1*0.5

			Light = {'Fake':Inverese_NormMeanStd(fake_synlight, self.Light_Mean, self.Light_STD), 'Real':real_synlight}

		############################ Real images training #####################
		Real_loss_D_fake = 0
		Real_loss_D_real = 0
		Real_loss_G_GAN = 0
		Real_loss_G_GAN_Feat = 0
		Real_loss_G_L1=0
		Real_loss_G_VGG = 0

		if self.real_train:
			# render using feature from image A and light from image B

			# [-1,1] -> [0,1]
			fake_real_A=tensorNormalize(fake_real[0:1,...].permute(0,2,3,1))

			# inverse normalize light
			fakeLightA=Inverese_NormMeanStd(fake_reallight[0:1,...].cuda(), self.Light_Mean, self.Light_STD)
			fakeLightB=Inverese_NormMeanStd(fake_reallight[1:2,...].cuda(), self.Light_Mean, self.Light_STD)
			if fakeLightB[:,2] <= 0.01 or fakeLightA[:,2] <= 0.01:
				print('lighting error!!!')

			# linear HDR
			fake_render_B = SingleRender(fake_real_A[:,:,:,3:6], fake_real_A[:,:,:,9:12], fake_real_A[:,:,:,0:3],
									fake_real_A[:,:,:,6:9], fakeLightB.detach() , self.Position_map, self.gpu_ids[0], self.opt.LowCam).permute(0,3,1,2)

			fake_render_A = SingleRender(fake_real_A[:,:,:,3:6], fake_real_A[:,:,:,9:12], fake_real_A[:,:,:,0:3],
									fake_real_A[:,:,:,6:9], fakeLightA.detach() , self.Position_map, self.gpu_ids[0], self.opt.LowCam).permute(0,3,1,2)

			
			# linear [0,1]
			real_A = input_real[0:int(self.opt.real_batchSize),...]
			real_B = input_real[int(self.opt.real_batchSize):2*int(self.opt.real_batchSize),...]

			## this is used for invariant sclaing
			# input fake_render_B (HDR,linear) and real_B linear LDR [0,1]
			# output scaled fake_render_B (linear LDR clamp[0,1])
			if self.opt.In_scale:
				fake_render_B_before=fake_render_B.clone().detach()
				fake_render_B = InvariantScaling(fake_render_B,real_B,50)

				fake_render_A_before=fake_render_A.clone().detach()
				fake_render_A = InvariantScaling(fake_render_A,real_A,50)				
			else:
				fake_render_B=torch.clamp(fake_render_B,0.01,1)
				fake_render_B_before=fake_render_B.clone().detach()

				fake_render_A=torch.clamp(fake_render_A,0.01,1)
				fake_render_A_before=fake_render_A.clone().detach()

			# print('fake_B: ',fake_render_B)
			# print('real_B: ',real_B)

			# linear to log, [0,1] -> [-1,1] 
			fake_render_A=2*(fake_render_A+EPSILON)**(1/2.2)-1			
			fake_render_A_before=2*(fake_render_A_before+EPSILON)**(1/2.2)-1			
			fake_render_B=2*(fake_render_B+EPSILON)**(1/2.2)-1			
			fake_render_B_before=2*(fake_render_B_before+EPSILON)**(1/2.2)-1

			real_A=2*(real_A+EPSILON)**(1/2.2)-1			
			real_B=2*(real_B+EPSILON)**(1/2.2)-1


			# Fake Detection and Loss
			Real_pred_fake_pool = self.discriminate(real_A, fake_render_B, use_pool=True, real=True)
			Real_loss_D_fake = self.criterionGAN(Real_pred_fake_pool, False)        

			# Real Detection and Loss        
			Real_pred_real = self.discriminate(real_A, real_B, real=True)
			Real_loss_D_real = self.criterionGAN(Real_pred_real, True)

			# GAN loss (Fake Passability Loss) 
			Real_pred_fake = self.netD_Render_Real.forward(torch.cat((real_A, fake_render_B), dim=1))        
			Real_loss_G_GAN = self.criterionGAN(Real_pred_fake, True)*self.opt.lambda_render


			if self.opt.gan_realA:
				# Fake Detection and Loss
				Real_pred_fake_pool = self.discriminate(real_A, fake_render_A, use_pool=True, real=True)
				Real_loss_D_fake += self.criterionGAN(Real_pred_fake_pool, False)        
				
				# Real Detection and Loss        
				Real_pred_real = self.discriminate(real_A, real_A, real=True)
				Real_loss_D_real += self.criterionGAN(Real_pred_real, True)

				# GAN loss (Fake Passability Loss) 
				Real_pred_fake = self.netD_Render_Real.forward(torch.cat((real_A, fake_render_A), dim=1))        
				Real_loss_G_GAN += self.criterionGAN(Real_pred_fake, True)*self.opt.lambda_render	
							
				Real_loss_D_fake *= 0.5
				Real_loss_D_real *= 0.5
				Real_loss_G_GAN *= 0.5


			# GAN feature matching loss
			if not self.opt.no_ganFeat_loss:
				feat_weights = 4.0 / (self.opt.n_layers_D + 1)
				D_weights = 1.0 / self.opt.num_D
				for i in range(self.opt.num_D):
					for j in range(len(Real_pred_fake[i])-1):
						Real_loss_G_GAN_Feat += self.opt.lambda_feat *self.opt.lambda_render * D_weights * feat_weights * \
							self.criterionFeat(Real_pred_fake[i][j], Real_pred_real[i][j].detach()) 

			if not self.opt.no_reall1_loss:
				# make it consistent with Real rendering discriminator
				Real_loss_G_L1 = self.criterionFeat(real_B, fake_render_B)*self.opt.lambda_render *self.opt.lambda_reall1
				if self.opt.L1_realA:
					Real_loss_G_L1 += self.criterionFeat(real_A, fake_render_A)*self.opt.lambda_render *self.opt.lambda_reall1
					Real_loss_G_L1 = Real_loss_G_L1*0.5


			# VGG feature loss
			if not self.opt.no_real_vgg_loss:
				Real_loss_G_VGG = self.criterionVGG(real_B, fake_render_B) * self.opt.lambda_feat
				if self.opt.vg_realA:
					Real_loss_G_VGG += self.criterionVGG(real_A, fake_render_A) * self.opt.lambda_feat
					Real_loss_G_VGG = Real_loss_G_VGG*0.5

			# add one more term Fake_render_
			realimages = {'image_A':real_A, 'image_B': real_B, 'fake_B':fake_render_B,'fake_B_before':fake_render_B_before,'fakeLightB':fakeLightB,
						'Diff_A':fake_real[0:1, 3:6, :, :], 'Spec_A':fake_real[0:1, 9:12, :, :], 'Norm_A':fake_real[0:1, 0:3, :, :], 'Rough_A':fake_real[0:1, 6:9, :, :],
						'Diff_B':fake_real[1:2, 3:6, :, :], 'Spec_B':fake_real[1:2, 9:12, :, :], 'Norm_B':fake_real[1:2, 0:3, :, :], 'Rough_B':fake_real[1:2, 6:9, :, :]
						 }



		# output
		if self.MyTest =='ALL_4D':	
			# sum up GAN loss for G (real and syn)
			loss_G_GAN += Real_loss_G_GAN*0.25 *self.opt.lambda_real
			loss_G_GAN_Feat += Real_loss_G_GAN_Feat*0.25*self.opt.lambda_real
			loss_G_L1 += Real_loss_G_L1*self.opt.lambda_real
			loss_G_VGG += Real_loss_G_VGG*self.opt.lambda_real
			Real_loss_L1_log=Real_loss_G_L1*self.opt.lambda_real

			return [ self.loss_filter( loss_G_GAN, loss_G_GAN_Feat, loss_G_VGG, loss_G_L1, 
										loss_D_real_norm, loss_D_fake_norm, 
										loss_D_real_diff, loss_D_fake_diff,
										loss_D_real_rough, loss_D_fake_rough, 
										loss_D_real_spec, loss_D_fake_spec,
										loss_Light_L1, Real_loss_D_real, Real_loss_D_fake), 
					None if not infer else fake_syn, None, input_syn, Light, None if not self.opt.real_train else realimages, None if not self.opt.real_train else Real_loss_L1_log]

		elif self.MyTest=='ALL_5D_Render':

			# sum up GAN loss for G (real and syn)
			loss_G_GAN += Real_loss_G_GAN*0.2*self.opt.lambda_real
			loss_G_L1 += Real_loss_G_L1*self.opt.lambda_real
			loss_G_GAN_Feat += Real_loss_G_GAN_Feat*0.2*self.opt.lambda_real
			loss_G_VGG += Real_loss_G_VGG*self.opt.lambda_real

			Real_loss_L1_log = Real_loss_G_L1*self.opt.lambda_real

			return [ self.loss_filter( loss_G_GAN, loss_G_GAN_Feat, loss_G_VGG, loss_G_L1, 
										loss_D_real_norm, loss_D_fake_norm, 
										loss_D_real_diff, loss_D_fake_diff,
										loss_D_real_rough, loss_D_fake_rough, 
										loss_D_real_spec, loss_D_fake_spec,
										loss_D_real_render, loss_D_fake_render,
										loss_Light_L1, Real_loss_D_real, Real_loss_D_fake), 
					None if not infer else fake_syn, renderimage, input_syn, Light, None if not self.opt.real_train else realimages, None if not self.opt.real_train else Real_loss_L1_log]			
		
		elif self.MyTest=='L1_Render':
			return [ self.loss_filter( loss_G_L1, loss_Light_L1), 
					None if not infer else fake_syn, renderimage, input_syn, Light, None, None ]

		elif self.MyTest=='ALL_1D_Render':

			loss_G_GAN += Real_loss_G_GAN*0.2*self.opt.lambda_real
			loss_G_L1 += Real_loss_G_L1*self.opt.lambda_real
			loss_G_GAN_Feat += Real_loss_G_GAN_Feat*0.2*self.opt.lambda_real

			Real_loss_L1_log = Real_loss_G_L1*self.opt.lambda_real

			return [ self.loss_filter( loss_G_GAN, loss_G_GAN_Feat, loss_G_VGG, loss_G_L1, 
										loss_D_real_render, loss_D_fake_render,
										loss_Light_L1, Real_loss_D_real, Real_loss_D_fake), 
					None if not infer else fake_syn, renderimage, input_syn, Light, None if not self.opt.real_train else realimages, None if not self.opt.real_train else Real_loss_L1_log]			
		
		elif self.MyTest=='L1':
			return [ self.loss_filter( loss_G_L1, loss_Light_L1), 
					None if not infer else fake_syn, None, input_syn, Light, None, None ]


	def inference(self, label, inst, image=None):
		# Encode Inputs        
		image = Variable(image) if image is not None else None
		input_label, inst_map, real_image, _ = self.encode_input(Variable(label), Variable(inst), image, infer=True)

		# Fake Generation
		if self.use_features:
			if self.opt.use_encoded_image:
				# encode the real image to get feature map
				feat_map = self.netE.forward(real_image, inst_map)
			else:
				# sample clusters from precomputed features             
				feat_map = self.sample_features(inst_map)
			input_concat = torch.cat((input_label, feat_map), dim=1)                        
		else:
			input_concat = input_label        
		   
		with torch.no_grad():
			fake_image = self.netG.forward(input_concat)

		return fake_image


	def inference_train(self, label, inst, image=None):
		# Encode Inputs        
		image = Variable(image) if image is not None else None
		input_label, inst_map, real_image, _ = self.encode_input(Variable(label), Variable(inst), image, infer=True)


		# augment input images
		if self.mode =='Syn':
			# real_image=real_image.cuda()
			real_image_temp=tensorNormalize(real_image.permute(0,2,3,1))
			light=Create_NumberPointLightPosition(1, 0.3, self.gpu_ids[0])
			#[B,C,W,H]
			input_label=SingleRender(real_image_temp[:,:,:,3:6], real_image_temp[:,:,:,9:12], real_image_temp[:,:,:,0:3],
									real_image_temp[:,:,:,6:9], light, self.Position_map, self.gpu_ids[0], False).permute(0,3,1,2)
			
			input_label=torch.clamp(input_label,0,1)
			# gamma correction to the input then from [0,1] to [-1,1]
			input_label=2*input_label**(1/2.2)-1

		elif self.mode=='Real':
			# gamma, then log, then normalize from [0,1] to [-1,1]
			if self.opt.real_train:
				input_label=2*real_image**(1/2.2)-1
			else:
				if self.opt.Gamma_test:
					input_label=2*input_label**(1/2.2)-1
				else:
					input_label=2*input_label-1

		input_concat = input_label        

		with torch.no_grad():
			fake_image,fake_light = self.netG.forward(input_concat)



		if self.mode=='Syn':

			# # L1 loss
			# loss_G_L1 = 0

			# if self.MyTest=='ALL_5D_Render':
			# 	if not self.opt.no_l1_loss:
			# 		loss_G_L1 = (self.criterionFeat(fake_image, real_image)+self.criterionFeat(FakeBatch, RealBatch)) * self.opt.lambda_l1			
			# else:
			# 	if not self.opt.no_l1_loss:
			# 		loss_G_L1 = self.criterionFeat(fake_image, real_image) * self.opt.lambda_l1

			All_Loss={'G_VGG':0, 'G_L1':0}
	
			if self.MyTest=='ALL_5D_Render':
				# fake_image=tensorNormalize(fake_image[0:1,...].permute(0,2,3,1))
				# inverse normalize light
				fake_light=Inverese_NormMeanStd(fake_light[0:1,...].cuda(), self.Light_Mean, self.Light_STD)

				# linear HDR [B,W,H,C]
				renderimage = SingleRender(real_image_temp[:,:,:,3:6], real_image_temp[:,:,:,9:12], real_image_temp[:,:,:,0:3],
										real_image_temp[:,:,:,6:9], fake_light.detach() , self.Position_map, self.gpu_ids[0], False).permute(0,3,1,2)

				return fake_image,All_Loss,renderimage,input_label,fake_light
			else:
				fake_light=Inverese_NormMeanStd(fake_light[0:1,...].cuda(), self.Light_Mean, self.Light_STD)
				return fake_image,All_Loss,input_label,fake_light


		elif self.mode=='Real':
			if self.opt.real_train:
				# [-1,1] -> [0,1]
				fake_feature_A=tensorNormalize(fake_image[0:1,...].permute(0,2,3,1))
				# inverse normalize light
				fake_light_B=Inverese_NormMeanStd(fake_light[1:2,...].cuda(), self.Light_Mean, self.Light_STD)
				if fake_light_B[:,2] <= 0.01:
					print('lighting error!!!')

				# linear HDR
				fake_B = SingleRender(fake_feature_A[:,:,:,3:6], fake_feature_A[:,:,:,9:12], fake_feature_A[:,:,:,0:3],
										fake_feature_A[:,:,:,6:9], fake_light_B.detach() , self.Position_map, self.gpu_ids[0], self.opt.LowCam).permute(0,3,1,2)
				
				# linear [0,1]
				real_A = input_concat[0:1,...]
				real_B = input_concat[1:2,...]

				return real_A, real_B, fake_B, fake_feature_A.permute(0,3,1,2)

			else:
				renderimage = {'Fake':0}
				All_Loss={'G_VGG':0, 'G_L1':0}
				if self.MyTest=='ALL_5D_Render':
					fake_image=tensorNormalize(fake_image[0:1,...].permute(0,2,3,1))
					# inverse normalize light
					fake_light=Inverese_NormMeanStd(fake_light[0:1,...].cuda(), self.Light_Mean, self.Light_STD)

					# linear HDR [B,W,H,C]
					renderimage = SingleRender(fake_image[:,:,:,3:6], fake_image[:,:,:,9:12], fake_image[:,:,:,0:3],
											fake_image[:,:,:,6:9], fake_light.detach() , self.Position_map, self.gpu_ids[0], self.opt.LowCam).permute(0,3,1,2)


					return fake_image.permute(0,3,1,2),All_Loss,renderimage,input_label,fake_light
				else:
					fake_light=Inverese_NormMeanStd(fake_light[0:1,...].cuda(), self.Light_Mean, self.Light_STD)
					return fake_image,All_Loss,input_label,fake_light


	def sample_features(self, inst): 
		# read precomputed feature clusters 
		cluster_path = os.path.join(self.opt.checkpoints_dir, self.opt.name, self.opt.cluster_path)        
		features_clustered = np.load(cluster_path, encoding='latin1').item()

		# randomly sample from the feature clusters
		inst_np = inst.cpu().numpy().astype(int)                                      
		feat_map = self.Tensor(inst.size()[0], self.opt.feat_num, inst.size()[2], inst.size()[3])
		for i in np.unique(inst_np):    
			label = i if i < 1000 else i//1000
			if label in features_clustered:
				feat = features_clustered[label]
				cluster_idx = np.random.randint(0, feat.shape[0]) 
											
				idx = (inst == int(i)).nonzero()
				for k in range(self.opt.feat_num):                                    
					feat_map[idx[:,0], idx[:,1] + k, idx[:,2], idx[:,3]] = feat[cluster_idx, k]
		if self.opt.data_type==16:
			feat_map = feat_map.half()
		return feat_map

	def encode_features(self, image, inst):
		image = Variable(image.cuda(), volatile=True)
		feat_num = self.opt.feat_num
		h, w = inst.size()[2], inst.size()[3]
		block_num = 32
		feat_map = self.netE.forward(image, inst.cuda())
		inst_np = inst.cpu().numpy().astype(int)
		feature = {}
		for i in range(self.opt.label_nc):
			feature[i] = np.zeros((0, feat_num+1))
		for i in np.unique(inst_np):
			label = i if i < 1000 else i//1000
			idx = (inst == int(i)).nonzero()
			num = idx.size()[0]
			idx = idx[num//2,:]
			val = np.zeros((1, feat_num+1))                        
			for k in range(feat_num):
				val[0, k] = feat_map[idx[0], idx[1] + k, idx[2], idx[3]].data[0]            
			val[0, feat_num] = float(num) / (h * w // block_num)
			feature[label] = np.append(feature[label], val, axis=0)
		return feature

	def get_edges(self, t):
		edge = torch.cuda.ByteTensor(t.size()).zero_()
		edge[:,:,:,1:] = edge[:,:,:,1:] | (t[:,:,:,1:] != t[:,:,:,:-1])
		edge[:,:,:,:-1] = edge[:,:,:,:-1] | (t[:,:,:,1:] != t[:,:,:,:-1])
		edge[:,:,1:,:] = edge[:,:,1:,:] | (t[:,:,1:,:] != t[:,:,:-1,:])
		edge[:,:,:-1,:] = edge[:,:,:-1,:] | (t[:,:,1:,:] != t[:,:,:-1,:])
		if self.opt.data_type==16:
			return edge.half()
		else:
			return edge.float()

	def save(self, which_epoch):
		if self.MyTest=='ALL_4D':
			self.save_network(self.netG, 'G', which_epoch, self.gpu_ids)
			self.save_network(self.netD_Norm, 'D_Norm', which_epoch, self.gpu_ids)
			self.save_network(self.netD_Diff, 'D_Diff', which_epoch, self.gpu_ids)
			self.save_network(self.netD_Rough, 'D_Rough', which_epoch, self.gpu_ids)
			self.save_network(self.netD_Spec, 'D_Spec', which_epoch, self.gpu_ids)
		elif self.MyTest=='ALL_5D_Render':
			self.save_network(self.netG, 'G', which_epoch, self.gpu_ids)
			self.save_network(self.netD_Norm, 'D_Norm', which_epoch, self.gpu_ids)
			self.save_network(self.netD_Diff, 'D_Diff', which_epoch, self.gpu_ids)
			self.save_network(self.netD_Rough, 'D_Rough', which_epoch, self.gpu_ids)
			self.save_network(self.netD_Spec, 'D_Spec', which_epoch, self.gpu_ids)
			self.save_network(self.netD_Render, 'D_Render', which_epoch, self.gpu_ids)
		elif self.MyTest=='ALL_1D_Render':
			self.save_network(self.netG, 'G', which_epoch, self.gpu_ids)
			self.save_network(self.netD_Render, 'D_Render', which_epoch, self.gpu_ids)
		elif self.MyTest=='L1' or self.MyTest=='L1_Render':
			self.save_network(self.netG, 'G', which_epoch, self.gpu_ids)


		if self.opt.real_train:
			self.save_network(self.netD_Render_Real, 'D_Render_Real', which_epoch, self.gpu_ids)


		if self.gen_features:
			self.save_network(self.netE, 'E', which_epoch, self.gpu_ids)

	def update_fixed_params(self):
		# after fixing the global generator for a number of iterations, also start finetuning it
		params = list(self.netG.parameters())
		if self.gen_features:
			params += list(self.netE.parameters())           
		self.optimizer_G = torch.optim.Adam(params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
		if self.opt.verbose:
			print('------------ Now also finetuning global generator -----------')

	def update_learning_rate(self):
		lrd = self.opt.lr / self.opt.niter_decay
		lr = self.old_lr - lrd    

		if self.MyTest=='ALL_4D':
			for param_group in self.optimizer_D_Norm.param_groups:
				param_group['lr'] = lr	
			for param_group in self.optimizer_D_Diff.param_groups:
				param_group['lr'] = lr			
			for param_group in self.optimizer_D_Rough.param_groups:
				param_group['lr'] = lr	
			for param_group in self.optimizer_D_Spec.param_groups:
				param_group['lr'] = lr	
		elif self.MyTest=='ALL_5D_Render':
			for param_group in self.optimizer_D_Norm.param_groups:
				param_group['lr'] = lr	
			for param_group in self.optimizer_D_Diff.param_groups:
				param_group['lr'] = lr			
			for param_group in self.optimizer_D_Rough.param_groups:
				param_group['lr'] = lr	
			for param_group in self.optimizer_D_Spec.param_groups:
				param_group['lr'] = lr	
			for param_group in self.optimizer_D_Render.param_groups:
				param_group['lr'] = lr		
		elif self.MyTest=='ALL_1D_Render':
			for param_group in self.optimizer_D_Render.param_groups:
				param_group['lr'] = lr	


		if self.opt.real_train:
			for param_group in self.optimizer_D_Render_Real.param_groups:
				param_group['lr'] = lr	

		for param_group in self.optimizer_G.param_groups:
			param_group['lr'] = lr
		if self.opt.verbose:
			print('update learning rate: %f -> %f' % (self.old_lr, lr))
		self.old_lr = lr

class InferenceModel(Pix2PixHDModel):
	# def initialize(self, opt):
	# 	self.gpu_ids=opt.gpu_ids
	# 	self.criterionFeat = torch.nn.L1Loss()
	# 	if not opt.no_vgg_loss:             
	# 		self.criterionVGG = networks.VGGLoss(self.gpu_ids)
	def forward(self, inp):
		label, inst = inp
		return self.inference(label, inst)

		
