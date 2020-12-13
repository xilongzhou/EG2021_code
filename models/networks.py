import torch
import torch.nn as nn
import functools
from torch.autograd import Variable
import numpy as np

###############################################################################
# Functions
###############################################################################
def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		m.weight.data.normal_(0.0, 0.02)
	elif classname.find('BatchNorm2d') != -1:
		m.weight.data.normal_(1.0, 0.02)
		m.bias.data.fill_(0)
	elif classname.find('Linear') != -1:
		n = m.in_features
		y = np.sqrt(1/float(n))
		# print('input features: ',n)
		m.weight.data.normal_(0.0, 0.01*y)
		if m.bias is not None:
			m.weight.data.normal_(0.0, y)   
			m.bias.data.normal_(0.0, 0.002) 


def VA_weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		m.weight.data.normal_(0.0, 0.02)
	elif classname.find('InstanceNorm2d') != -1:
		m.weight.data.normal_(1.0, 0.02)
		m.bias.data.fill_(0)    
	elif classname.find('Linear') != -1:
		n = m.in_features
		y = np.sqrt(1/float(n))
		# print('input features: ',n)
		m.weight.data.normal_(0.0, 0.01*y)
		if m.bias is not None:
			m.weight.data.normal_(0.0, y)   
			m.bias.data.normal_(0.0, 0.002) 


def get_norm_layer(norm_type='instance'):
	if norm_type == 'batch':
		norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
	elif norm_type == 'instance':
		norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
	else:
		raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
	return norm_layer

def define_G(input_nc, output_nc,rough_channel, ngf, netG, dropout, n_downsample_global=3, n_blocks_global=9, n_blocks_branch=4, n_local_enhancers=1, 
			 n_blocks_local=3, norm='instance', gpu_ids=[]):    
	norm_layer = get_norm_layer(norm_type=norm)     
	if netG == 'global':    
		netG = GlobalGenerator(input_nc, output_nc, dropout, ngf, n_downsample_global, n_blocks_global, norm_layer)
	elif netG == 'global_Light':    
		netG = GlobalGenerator_Light(input_nc, output_nc, dropout, ngf, n_downsample_global, n_blocks_global, norm_layer)
	elif netG == 'newarch':    
		netG = NewGenerator(input_nc, output_nc, dropout, ngf, n_downsample_global, n_blocks_global, n_blocks_branch, norm_layer)   
	elif netG == 'newarch_Light':
		netG = NewGenerator_Light(input_nc, output_nc, dropout, ngf, n_downsample_global, n_blocks_global, n_blocks_branch, norm_layer)   
	elif netG == 'local':        
		netG = LocalEnhancer(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global,n_local_enhancers, n_blocks_local, norm_layer)
	elif netG == 'encoder':
		netG = Encoder(input_nc, output_nc, ngf, n_downsample_global, norm_layer)
	elif netG == 'VA_Net':
		netG = VA_Net(input_nc,output_nc)
	elif netG =='LocalVA_Net':
		netG = LocalVA_Net(input_nc,output_nc)
	elif netG =='NewVA_Net':
		netG = NewVA_Net(input_nc,output_nc,rough_channel)
	elif netG =='NewVA_Net_Light':
		netG = NewVA_Net_Light(input_nc,output_nc,rough_channel)
	else:
		raise('generator not implemented!')
	print(netG)

	if len(gpu_ids) > 0:
		assert(torch.cuda.is_available())   
		netG.cuda(gpu_ids[0])

	# if netG == 'VA_Net' or netG =='LocalVA_Net' or netG=='NewVA_Net' or netG =='NewVA_Net_Light':
	print('VA Net initialization')
	netG.apply(VA_weights_init)
	# else:
	# 	print('Other Net initialization')
	# 	netG.apply(weights_init)

	return netG

def define_D(input_nc, dropout, ndf, n_layers_D, norm='instance', use_sigmoid=False, num_D=1, getIntermFeat=False, gpu_ids=[]):        
	norm_layer = get_norm_layer(norm_type=norm)   
	netD = MultiscaleDiscriminator(input_nc, ndf, n_layers_D, norm_layer, use_sigmoid, num_D, getIntermFeat,dropout)   
	print(netD)
	if len(gpu_ids) > 0:
		assert(torch.cuda.is_available())
		netD.cuda(gpu_ids[0])
	netD.apply(weights_init)
	return netD

def print_network(net):
	if isinstance(net, list):
		net = net[0]
	num_params = 0
	for param in net.parameters():
		num_params += param.numel()
	print(net)
	print('Total number of parameters: %d' % num_params)

##############################################################################
# Losses
##############################################################################
class GANLoss(nn.Module):
	def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
				 tensor=torch.FloatTensor):
		super(GANLoss, self).__init__()
		self.real_label = target_real_label
		self.fake_label = target_fake_label
		self.real_label_var = None
		self.fake_label_var = None
		self.Tensor = tensor
		if use_lsgan:
			self.loss = nn.MSELoss()
		else:
			self.loss = nn.BCELoss()

	def get_target_tensor(self, input, target_is_real):
		target_tensor = None
		if target_is_real:
			create_label = ((self.real_label_var is None) or
							(self.real_label_var.numel() != input.numel()))
			if create_label:
				real_tensor = self.Tensor(input.size()).fill_(self.real_label)
				self.real_label_var = Variable(real_tensor, requires_grad=False)
			target_tensor = self.real_label_var
		else:
			create_label = ((self.fake_label_var is None) or
							(self.fake_label_var.numel() != input.numel()))
			if create_label:
				fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
				self.fake_label_var = Variable(fake_tensor, requires_grad=False)
			target_tensor = self.fake_label_var
		return target_tensor

	def __call__(self, input, target_is_real):
		if isinstance(input[0], list):
			loss = 0
			for input_i in input:
				pred = input_i[-1]
				target_tensor = self.get_target_tensor(pred, target_is_real)
				loss += self.loss(pred, target_tensor)
			return loss
		else:            
			target_tensor = self.get_target_tensor(input[-1], target_is_real)
			return self.loss(input[-1], target_tensor)

class GANLoss_Smooth(nn.Module):
	def __init__(self, use_lsgan=True, target_real_label=0.9, target_fake_label=0.0,
				 tensor=torch.FloatTensor):
		super(GANLoss_Smooth, self).__init__()
		self.real_label = target_real_label
		self.fake_label = target_fake_label
		self.real_label_var = None
		self.fake_label_var = None
		self.Tensor = tensor
		if use_lsgan:
			self.loss = nn.MSELoss()
		else:
			self.loss = nn.BCELoss()

	def get_target_tensor(self, input, target_is_real):
		target_tensor = None
		if target_is_real:
			create_label = ((self.real_label_var is None) or
							(self.real_label_var.numel() != input.numel()))
			if create_label:
				real_tensor = self.Tensor(input.size()).fill_(self.real_label)
				self.real_label_var = Variable(real_tensor, requires_grad=False)
			target_tensor = self.real_label_var
		else:
			create_label = ((self.fake_label_var is None) or
							(self.fake_label_var.numel() != input.numel()))
			if create_label:
				fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
				self.fake_label_var = Variable(fake_tensor, requires_grad=False)
			target_tensor = self.fake_label_var
		return target_tensor

	def __call__(self, input, target_is_real):
		if isinstance(input[0], list):
			loss = 0
			for input_i in input:
				pred = input_i[-1]
				target_tensor = self.get_target_tensor(pred, target_is_real)
				loss += self.loss(pred, target_tensor)
			return loss
		else:            
			target_tensor = self.get_target_tensor(input[-1], target_is_real)
			return self.loss(input[-1], target_tensor)

class VGGLoss(nn.Module):
	def __init__(self, gpu_ids):
		super(VGGLoss, self).__init__()        
		self.vgg = Vgg19().cuda()
		self.criterion = nn.L1Loss()
		self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]        

	def forward(self, x, y):              
		x_vgg, y_vgg = self.vgg(x), self.vgg(y)
		loss = 0
		for i in range(len(x_vgg)):
			loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())        
		return loss

##############################################################################
# Generator
##############################################################################
class LocalEnhancer(nn.Module):
	def __init__(self, input_nc, output_nc, ngf=32, n_downsample_global=3, n_blocks_global=9, 
				 n_local_enhancers=1, n_blocks_local=3, norm_layer=nn.BatchNorm2d, padding_type='reflect'):        
		super(LocalEnhancer, self).__init__()
		self.n_local_enhancers = n_local_enhancers
		
		###### global generator model #####           
		ngf_global = ngf * (2**n_local_enhancers)
		model_global = GlobalGenerator(input_nc, output_nc, ngf_global, n_downsample_global, n_blocks_global, norm_layer).model        
		model_global = [model_global[i] for i in range(len(model_global)-3)] # get rid of final convolution layers        
		self.model = nn.Sequential(*model_global)                

		###### local enhancer layers #####
		for n in range(1, n_local_enhancers+1):
			### downsample            
			ngf_global = ngf * (2**(n_local_enhancers-n))
			model_downsample = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf_global, kernel_size=7, padding=0), 
								norm_layer(ngf_global), nn.ReLU(True),
								nn.Conv2d(ngf_global, ngf_global * 2, kernel_size=3, stride=2, padding=1), 
								norm_layer(ngf_global * 2), nn.ReLU(True)]
			### residual blocks
			model_upsample = []
			for i in range(n_blocks_local):
				model_upsample += [ResnetBlock(ngf_global * 2, padding_type=padding_type, norm_layer=norm_layer)]

			### upsample
			model_upsample += [nn.ConvTranspose2d(ngf_global * 2, ngf_global, kernel_size=3, stride=2, padding=1, output_padding=1), 
							   norm_layer(ngf_global), nn.ReLU(True)]      

			### final convolution
			if n == n_local_enhancers:                
				model_upsample += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]                       
			
			setattr(self, 'model'+str(n)+'_1', nn.Sequential(*model_downsample))
			setattr(self, 'model'+str(n)+'_2', nn.Sequential(*model_upsample))                  
		
		self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

	def forward(self, input): 
		### create input pyramid
		input_downsampled = [input]
		for i in range(self.n_local_enhancers):
			input_downsampled.append(self.downsample(input_downsampled[-1]))

		### output at coarest level
		output_prev = self.model(input_downsampled[-1])        
		### build up one layer at a time
		for n_local_enhancers in range(1, self.n_local_enhancers+1):
			model_downsample = getattr(self, 'model'+str(n_local_enhancers)+'_1')
			model_upsample = getattr(self, 'model'+str(n_local_enhancers)+'_2')            
			input_i = input_downsampled[self.n_local_enhancers-n_local_enhancers]            
			output_prev = model_upsample(model_downsample(input_i) + output_prev)
		return output_prev

class GlobalGenerator(nn.Module):
	def __init__(self, input_nc, output_nc,dropout, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d, 
				 padding_type='reflect'):
		assert(n_blocks >= 0)
		super(GlobalGenerator, self).__init__()        
		activation = nn.ReLU(True)        

		model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
		### downsample
		for i in range(n_downsampling):
			mult = 2**i
			model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
					  norm_layer(ngf * mult * 2), activation]

		### resnet blocks
		mult = 2**n_downsampling
		for i in range(n_blocks):
			model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer, use_dropout=dropout)]
		
		### upsample         
		for i in range(n_downsampling):
			mult = 2**(n_downsampling - i)
			model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
					   norm_layer(int(ngf * mult / 2)), activation]

			# model += [nn.Upsample(scale_factor = 2, mode='bilinear'),
			#           nn.ReflectionPad2d(1),
			#           nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1, padding=0),
			#           norm_layer(int(ngf * mult / 2)), activation]

		model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]        
		self.model = nn.Sequential(*model)
			
	def forward(self, input):
		return self.model(input), None            


class NewGenerator(nn.Module):
	def __init__(self, input_nc, output_nc,dropout, ngf=64, n_downsampling=3, n_blocks=8, n_block_branch=4, norm_layer=nn.BatchNorm2d, 
				 padding_type='reflect'):
		assert(n_blocks >= 0)
		super(NewGenerator, self).__init__()        
		activation = nn.ReLU(True)        

		model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
		### downsample
		for i in range(n_downsampling):
			mult = 2**i
			model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
					  norm_layer(ngf * mult * 2), activation]

		### resnet blocks
		mult = 2**n_downsampling
		for i in range(n_blocks-n_block_branch):
			model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer,use_dropout=dropout)]
		
		############################## My Updates ###################################
		Decoder_norm=[]
		Decoder_diff=[]
		Decoder_rough=[]
		Decoder_spec=[]

		for i in range(n_block_branch):
			Decoder_norm += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer,use_dropout=dropout)]
			Decoder_diff += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer,use_dropout=dropout)]
			Decoder_rough += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer,use_dropout=dropout)]
			Decoder_spec += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer,use_dropout=dropout)]

		### upsample         
		for i in range(n_downsampling):
			mult = 2**(n_downsampling - i)
			Decoder_norm += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
						   norm_layer(int(ngf * mult / 2)), activation]
			Decoder_diff += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
						   norm_layer(int(ngf * mult / 2)), activation]
			Decoder_rough += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
						   norm_layer(int(ngf * mult / 2)), activation]
			Decoder_spec += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
						   norm_layer(int(ngf * mult / 2)), activation]

		Decoder_norm += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]        
		Decoder_diff += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()] 
		Decoder_rough += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()] 
		Decoder_spec += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()] 

		self.Encoder = nn.Sequential(*model)
		self.Decoder_norm = nn.Sequential(*Decoder_norm)
		self.Decoder_diff = nn.Sequential(*Decoder_diff)
		self.Decoder_rough = nn.Sequential(*Decoder_rough)
		self.Decoder_spec = nn.Sequential(*Decoder_spec)

			
	def forward(self, input):

		####################### My Updates ########################33
		Encoder_results=self.Encoder(input)
		Normal=self.Decoder_norm(Encoder_results)
		Diff=self.Decoder_diff(Encoder_results)
		Rough=self.Decoder_rough(Encoder_results)
		Spec=self.Decoder_spec(Encoder_results)
		# print('norm', Normal.shape)
		Output = torch.cat((Normal,Diff,Rough,Spec),1)
		return Output, None



# define VA paper nework
class FC(nn.Module):
	def __init__(self,input_channel,output_channel,BIAS):
		super(FC,self).__init__()
		self.fc_layer=nn.Linear(input_channel,output_channel,bias=BIAS)

	def forward(self,input):
		# if input is 4D [b,c,1,1],output [b,c,1,1]
		if len(input.shape) == 4:
			[b,c,w,h]=input.shape
			out=self.fc_layer(input.view(b,c))
			out=out.unsqueeze(2).unsqueeze(2)
		# if input is 2D [b,c] otuput [b,c]
		elif len(input.shape) == 2:
			out=self.fc_layer(input)
		# otherwise, error
		else:
			print('incorrectly input to FC layer')
			sys.exit(1) 

		return out


class GlobalGenerator_Light(nn.Module):
	def __init__(self, input_nc, output_nc,dropout, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d, 
				 padding_type='reflect'):
		assert(n_blocks >= 0)
		super(GlobalGenerator_Light, self).__init__()        
		activation = nn.ReLU(True)        

		model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
		### downsample
		for i in range(n_downsampling):
			mult = 2**i
			model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
					  norm_layer(ngf * mult * 2), activation]

		### resnet blocks
		mult = 2**n_downsampling
		mult_light = 2**n_downsampling

		for i in range(n_blocks):
			model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer, use_dropout=dropout)]
		
		self.Encoder = nn.Sequential(*model)

		Decoder = []
		Decoder_Light =[]
		### upsample         
		for i in range(n_downsampling):
			mult = 2**(n_downsampling - i)
			Decoder += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
					   norm_layer(int(ngf * mult / 2)), activation]

		Decoder_Light += [nn.Conv2d(int(ngf * mult_light), int(ngf * mult_light/4), kernel_size=4, stride=2, padding=1),
				   norm_layer(int(ngf * mult_light/4)), activation]				   			   
		Decoder_Light += [nn.Conv2d(int(ngf * mult_light/4), int(ngf * mult_light/16), kernel_size=4, stride=2, padding=1),
				   norm_layer(int(ngf * mult_light/16)), activation]


		Decoder += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]        
		self.Decoder = nn.Sequential(*Decoder)
		self.Decoder_Light=nn.Sequential(*Decoder_Light)

		Decoder_Light_FC = []
		Decoder_Light_FC += [FC(int(ngf * mult_light), int(ngf * mult_light/4),True), nn.LeakyReLU(0.2)]
		Decoder_Light_FC += [FC(int(ngf * mult_light/4), int(ngf * mult_light/8),True), nn.LeakyReLU(0.2)]
		Decoder_Light_FC += [FC(int(ngf * mult_light/8), int(ngf * mult_light/16),True), nn.LeakyReLU(0.2)]
		Decoder_Light_FC += [FC(int(ngf * mult_light/16), int(ngf * mult_light/32),True), nn.LeakyReLU(0.2)]
		Decoder_Light_FC += [FC(int(ngf * mult_light/32), int(ngf * mult_light/64),True), nn.LeakyReLU(0.2)]
		Decoder_Light_FC += [FC(int(ngf * mult_light/64),3,True)]
		self.Decoder_Light_FC=nn.Sequential(*Decoder_Light_FC)
	  

	def forward(self, input):

		####################### My Updates ########################33
		Encoder_results=self.Encoder(input)
		Out=self.Decoder(Encoder_results)
		# print('Encoder_results0: ',Out[0,...])
		# print('Encoder_results1: ',Out[1,...])

		Light_Encoder=self.Decoder_Light(Encoder_results)
		# print('Light_Encoder0',Light_Encoder[0,...])
		# print('Light_Encoder1',Light_Encoder[1,...])

		flat_Light=Light_Encoder.view(-1,self.num_flat_features(Light_Encoder))
		Light = self.Decoder_Light_FC(flat_Light)
		# print('Light0',Light[0,...])
		# print('Light1',Light[1,...])

		return Out,Light

	def num_flat_features(self, x):
		size = x.size()[1:]  # all dimensions except the batch dimension
		num_features = 1
		for s in size:
			num_features *= s
		return num_features


class NewGenerator_Light(nn.Module):
	def __init__(self, input_nc, output_nc,dropout, ngf=64, n_downsampling=3, n_blocks=8, n_block_branch=4, norm_layer=nn.BatchNorm2d, 
				 padding_type='reflect'):
		assert(n_blocks >= 0)
		super(NewGenerator_Light, self).__init__()        
		activation = nn.ReLU(True)        

		model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
		### downsample
		for i in range(n_downsampling):
			mult = 2**i
			model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
					  norm_layer(ngf * mult * 2), activation]

		### resnet blocks
		mult = 2**n_downsampling
		mult_light = 2**n_downsampling
		for i in range(n_blocks-n_block_branch):
			model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer,use_dropout=dropout)]
		
		############################## My Updates ###################################
		Decoder_norm=[]
		Decoder_diff=[]
		Decoder_rough=[]
		Decoder_spec=[]

		for i in range(n_block_branch):
			Decoder_norm += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer,use_dropout=dropout)]
			Decoder_diff += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer,use_dropout=dropout)]
			Decoder_rough += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer,use_dropout=dropout)]
			Decoder_spec += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer,use_dropout=dropout)]
			# Decoder_Light += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer,use_dropout=dropout)]

		### upsample         
		for i in range(n_downsampling):
			mult = 2**(n_downsampling - i)
			Decoder_norm += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
						   norm_layer(int(ngf * mult / 2)), activation]
			Decoder_diff += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
						   norm_layer(int(ngf * mult / 2)), activation]
			Decoder_rough += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
						   norm_layer(int(ngf * mult / 2)), activation]
			Decoder_spec += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
						   norm_layer(int(ngf * mult / 2)), activation]
			
		Decoder_norm += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]        
		Decoder_diff += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()] 
		Decoder_rough += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()] 
		Decoder_spec += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()] 

		self.Encoder = nn.Sequential(*model)
		self.Decoder_norm = nn.Sequential(*Decoder_norm)
		self.Decoder_diff = nn.Sequential(*Decoder_diff)
		self.Decoder_rough = nn.Sequential(*Decoder_rough)
		self.Decoder_spec = nn.Sequential(*Decoder_spec)


		mult_light = 2**n_downsampling
		Decoder_Light =[]
		Decoder_Light += [nn.Conv2d(int(ngf * mult_light), int(ngf * mult_light/4), kernel_size=4, stride=2, padding=1),
				   norm_layer(int(ngf * mult_light/4)), activation]				   			   
		Decoder_Light += [nn.Conv2d(int(ngf * mult_light/4), int(ngf * mult_light/16), kernel_size=4, stride=2, padding=1),
				   norm_layer(int(ngf * mult_light/16)), activation]
		self.Decoder_Light=nn.Sequential(*Decoder_Light)


		Decoder_Light_FC = []
		Decoder_Light_FC += [FC(int(ngf * mult_light), int(ngf * mult_light/4),True), nn.LeakyReLU(0.2)]
		Decoder_Light_FC += [FC(int(ngf * mult_light/4), int(ngf * mult_light/8),True), nn.LeakyReLU(0.2)]
		Decoder_Light_FC += [FC(int(ngf * mult_light/8), int(ngf * mult_light/16),True), nn.LeakyReLU(0.2)]
		Decoder_Light_FC += [FC(int(ngf * mult_light/16), int(ngf * mult_light/32),True), nn.LeakyReLU(0.2)]
		Decoder_Light_FC += [FC(int(ngf * mult_light/32), int(ngf * mult_light/64),True), nn.LeakyReLU(0.2)]
		Decoder_Light_FC += [FC(int(ngf * mult_light/64),3,True)]
		self.Decoder_Light_FC=nn.Sequential(*Decoder_Light_FC)

			
	def forward(self, input):

		####################### My Updates ########################33
		Encoder_results=self.Encoder(input)
		Normal=self.Decoder_norm(Encoder_results)
		Diff=self.Decoder_diff(Encoder_results)
		Rough=self.Decoder_rough(Encoder_results)
		Spec=self.Decoder_spec(Encoder_results)
		Output = torch.cat((Normal,Diff,Rough,Spec),1)

		Light_Encoder=self.Decoder_Light(Encoder_results)
		flat_Light=Light_Encoder.view(-1,self.num_flat_features(Light_Encoder))
		Light = self.Decoder_Light_FC(flat_Light)

		return Output,Light

	def num_flat_features(self, x):
		size = x.size()[1:]  # all dimensions except the batch dimension
		num_features = 1
		for s in size:
			num_features *= s
		return num_features



# Define a resnet block
class ResnetBlock(nn.Module):
	def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
		super(ResnetBlock, self).__init__()
		self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

	def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
		conv_block = []
		p = 0
		if padding_type == 'reflect':
			conv_block += [nn.ReflectionPad2d(1)]
		elif padding_type == 'replicate':
			conv_block += [nn.ReplicationPad2d(1)]
		elif padding_type == 'zero':
			p = 1
		else:
			raise NotImplementedError('padding [%s] is not implemented' % padding_type)

		conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
					   norm_layer(dim),
					   activation]
		if use_dropout:
			print('dropout for G')
			conv_block += [nn.Dropout(0.5)]

		p = 0
		if padding_type == 'reflect':
			conv_block += [nn.ReflectionPad2d(1)]
		elif padding_type == 'replicate':
			conv_block += [nn.ReplicationPad2d(1)]
		elif padding_type == 'zero':
			p = 1
		else:
			raise NotImplementedError('padding [%s] is not implemented' % padding_type)
		conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
					   norm_layer(dim)]

		return nn.Sequential(*conv_block)

	def forward(self, x):
		out = x + self.conv_block(x)
		return out


class Deconv(nn.Module):

	def __init__(self,input_channel,output_channel):

		super(Deconv,self).__init__()
		## upsampling method (non-deterministic in pytorch)
		# self.upsampling=nn.Upsample(scale_factor=2, mode='nearest')

		self.temp_conv1 = nn.Conv2d(input_channel,output_channel,4,stride=1,bias=False)
		self.temp_conv2 = nn.Conv2d(output_channel,output_channel,4,stride=1,bias=False)

		# realize same padding in tensorflow
		self.padding=nn.ConstantPad2d((1, 2, 1, 2), 0)

	def forward(self,input):

		# print('Deco input shape,',input.shape[1])
		# Upsamp=self.upsampling(input)
		## hack upsampling method to make is deterministic
		Upsamp = input[:, :, :, None, :, None].expand(-1, -1, -1, 2, -1, 2).reshape(input.size(0), input.size(1), input.size(2)*2, input.size(3)*2)

		out=self.temp_conv1(self.padding(Upsamp))
		out=self.temp_conv2(self.padding(out))

		# print('output shape,',out.shape)
		return out

def mymean(input):
	[b,c,w,h]=input.shape
	mean=input.view(b,c,-1).mean(2)
	mean=mean.unsqueeze(-1).unsqueeze(-1)
	return mean#.reshape(b,c,1,1)


class VA_Net(nn.Module):
	def __init__(self,input_channel,output_channel):
		super(VA_Net,self).__init__()

		## define local networks
		#encoder and downsampling
		self.conv1 = nn.Conv2d(input_channel,64,4,2,1,bias=False)
		self.conv2 = nn.Conv2d(64,128,4,2,1,bias=False)
		self.conv3 = nn.Conv2d(128,256,4,2,1,bias=False)
		self.conv4 = nn.Conv2d(256,512,4,2,1,bias=False)
		self.conv5 = nn.Conv2d(512,512,4,2,1,bias=False)
		self.conv6 = nn.Conv2d(512,512,4,2,1,bias=False)
		self.conv7 = nn.Conv2d(512,512,4,2,1,bias=False)
		self.conv8 = nn.Conv2d(512,512,4,2,1,bias=False)

		#decoder
		self.deconv1 = Deconv(512, 512)
		self.deconv2 = Deconv(1024, 512)
		self.deconv3 = Deconv(1024, 512)
		self.deconv4 = Deconv(1024, 512)
		self.deconv5 = Deconv(1024, 256)
		self.deconv6 = Deconv(512, 128)
		self.deconv7 = Deconv(256, 64)
		self.deconv8 = Deconv(128, output_channel)

		# #decoder
		# self.deconv1 = nn.ConvTranspose2d(512, 512, 4, 2, 1)
		# self.deconv2 = nn.ConvTranspose2d(1024, 512, 4, 2, 1)
		# self.deconv3 = nn.ConvTranspose2d(1024, 512, 4, 2, 1)
		# self.deconv4 = nn.ConvTranspose2d(1024, 512, 4, 2, 1)
		# self.deconv5 = nn.ConvTranspose2d(1024, 256, 4, 2, 1)
		# self.deconv6 = nn.ConvTranspose2d(512, 128, 4, 2, 1)
		# self.deconv7 = nn.ConvTranspose2d(256, 64, 4, 2, 1)
		# self.deconv8 = nn.ConvTranspose2d(128, output_channel, 4, 2, 1)

		self.sig = nn.Sigmoid()
		self.tan = nn.Tanh()

		self.leaky_relu = nn.LeakyReLU(0.2)

		# self.instance_normal1 = nn.InstanceNorm2d(64,affine=True)
		self.instance_normal2 = nn.InstanceNorm2d(128,affine=True)
		self.instance_normal3 = nn.InstanceNorm2d(256,affine=True)
		self.instance_normal4 = nn.InstanceNorm2d(512,affine=True)
		self.instance_normal5 = nn.InstanceNorm2d(512,affine=True)
		self.instance_normal6 = nn.InstanceNorm2d(512,affine=True)
		self.instance_normal7 = nn.InstanceNorm2d(512,affine=True)

		self.instance_normal_de_1 = nn.InstanceNorm2d(512,affine=True)
		self.instance_normal_de_2 = nn.InstanceNorm2d(512,affine=True)
		self.instance_normal_de_3 = nn.InstanceNorm2d(512,affine=True)
		self.instance_normal_de_4 = nn.InstanceNorm2d(512,affine=True)
		self.instance_normal_de_5 = nn.InstanceNorm2d(256,affine=True)
		self.instance_normal_de_6 = nn.InstanceNorm2d(128,affine=True)
		self.instance_normal_de_7 = nn.InstanceNorm2d(64,affine=True)


		self.dropout = nn.Dropout(0.5)

		## define global networks
		self.global_fc1=FC(input_channel,128,True)

		self.global_fc2=FC(256,256,True)
		self.global_to_local_fc2=FC(128,128,False)

		self.global_fc3=FC(512,512,True)
		self.global_to_local_fc3=FC(256,256,False)

		self.global_fc4=FC(1024,512,True)
		self.global_to_local_fc4=FC(512,512,False)

		self.global_fc5=FC(1024,512,True)
		self.global_to_local_fc5=FC(512,512,False)

		self.global_fc6=FC(1024,512,True)
		self.global_to_local_fc6=FC(512,512,False)

		self.global_fc7=FC(1024,512,True)
		self.global_to_local_fc7=FC(512,512,False)

		self.global_fc8=FC(1024,512,True)
		self.global_to_local_fc8=FC(512,512,False)

		############### decoder #########################
		self.global_to_local_de_fc1=FC(512,512,False)
		self.global_de_fc1=FC(1024,512,True)

		self.global_to_local_de_fc2=FC(512,512,False)
		self.global_de_fc2=FC(1024,512,True)

		self.global_to_local_de_fc3=FC(512,512,False)
		self.global_de_fc3=FC(1024,512,True)

		self.global_to_local_de_fc4=FC(512,512,False)
		self.global_de_fc4=FC(1024,512,True) #[1024,512]

		self.global_to_local_de_fc5=FC(512,256,False) #[512,256,False]
		self.global_de_fc5=FC(768,256,True) #[768,256,true]

		self.global_to_local_de_fc6=FC(256,128,False)#[256,128,False]
		self.global_de_fc6=FC(384,128,True)#[384,128,true]

		self.global_to_local_de_fc7=FC(128,64,False)#[128,64,False]
		self.global_de_fc7=FC(192,64,True)#[192,64,true]

		self.global_to_local_de_fc8=FC(64,output_channel,False)#[64,9,False]

		# self.Mean=mymean()

		self.selu=nn.SELU()


	def forward(self, input):

		#input:[b,3,h,w] -> [b,3,1,1] (mean) -> [b,128,1,1]
		# GlobalNetwork_FC1=self.selu(self.global_fc1(torch.mean(input,dim=[2,3]).unsqueeze(-1).unsqueeze(-1))) #global
		GlobalNetwork_FC1=self.selu(self.global_fc1(mymean(input))) #global
		# [batch,64,h/2,w/2]
		encoder1 = self.conv1(input) #local network


		# [batch,128,h/4,w/4]        
		encoder2 = self.conv2(self.leaky_relu(encoder1)) #local network
		# [b,256,1,1]
		# GlobalInput2=torch.cat((GlobalNetwork_FC1,torch.mean(encoder2,dim=[2,3]).unsqueeze(-1).unsqueeze(-1)),1)   #global network
		GlobalInput2=torch.cat((GlobalNetwork_FC1,mymean(encoder2)),1)   #global network
		# print('2st mean:',mymean(encoder2))

		# [b,256,1,1]
		GlobalNetwork_FC2=self.selu(self.global_fc2(GlobalInput2)) #global network
		# [b,128,h/4,w/4]
		encoder2=self.instance_normal2(encoder2)+self.global_to_local_fc2(GlobalNetwork_FC1) #local 


		# [batch,256,h/8,w/8]        
		encoder3 = self.conv3(self.leaky_relu(encoder2)) #local network
		# [b,512,1,1]
		# GlobalInput3=torch.cat((GlobalNetwork_FC2,torch.mean(encoder3,dim=[2,3]).unsqueeze(-1).unsqueeze(-1)),1)   #global network
		GlobalInput3=torch.cat((GlobalNetwork_FC2,mymean(encoder3)),1)   #global network

		# [b,512,1,1]
		GlobalNetwork_FC3=self.selu(self.global_fc3(GlobalInput3)) #global network
		# [b,256,h/8,w/8]
		encoder3=self.instance_normal3(encoder3)+self.global_to_local_fc3(GlobalNetwork_FC2) #local 


		# [batch,512,h/16,w/16]        
		encoder4 = self.conv4(self.leaky_relu(encoder3)) #local network
		# [b,1024,1,1]
		# GlobalInput4=torch.cat((GlobalNetwork_FC3,torch.mean(encoder4,dim=[2,3]).unsqueeze(-1).unsqueeze(-1)),1)   #global network
		GlobalInput4=torch.cat((GlobalNetwork_FC3,mymean(encoder4)),1)   #global network
		
		# [b,512,1,1]
		GlobalNetwork_FC4=self.selu(self.global_fc4(GlobalInput4)) #global network
		# [b,512,h/16,w/16]
		encoder4=self.instance_normal4(encoder4)+self.global_to_local_fc4(GlobalNetwork_FC3) #local 


		# [batch,512,h/32,w/32]        
		encoder5 = self.conv5(self.leaky_relu(encoder4)) #local network
		# [b,1024,1,1]
		# GlobalInput5=torch.cat((GlobalNetwork_FC4,torch.mean(encoder5,dim=[2,3]).unsqueeze(-1).unsqueeze(-1)),1)   #global network
		GlobalInput5=torch.cat((GlobalNetwork_FC4,mymean(encoder5)),1)   #global network
		
		# [b,512,1,1]
		GlobalNetwork_FC5=self.selu(self.global_fc5(GlobalInput5)) #global network
		# [b,512,h/32,w/32]
		encoder5=self.instance_normal5(encoder5)+self.global_to_local_fc5(GlobalNetwork_FC4) #local 


		# [batch,512,h/64,w/64]        
		encoder6 = self.conv6(self.leaky_relu(encoder5)) #local network
		# [b,1024,1,1]
		# GlobalInput6=torch.cat((GlobalNetwork_FC5,torch.mean(encoder6,dim=[2,3]).unsqueeze(-1).unsqueeze(-1)),1)   #global network
		GlobalInput6=torch.cat((GlobalNetwork_FC5,mymean(encoder6)),1)   #global network

		# [b,512,1,1]
		GlobalNetwork_FC6=self.selu(self.global_fc6(GlobalInput6)) #global network
		# [b,512,h/64,w/64]
		encoder6=self.instance_normal6(encoder6)+self.global_to_local_fc6(GlobalNetwork_FC5) #local 


		# [batch,512,h/128,w/128]        
		encoder7 = self.conv7(self.leaky_relu(encoder6)) #local network
		# [b,1024,1,1]
		# GlobalInput7=torch.cat((GlobalNetwork_FC6,torch.mean(encoder7,dim=[2,3]).unsqueeze(-1).unsqueeze(-1)),1)   #global network
		GlobalInput7=torch.cat((GlobalNetwork_FC6,mymean(encoder7)),1)   #global network

		# [b,512,1,1]
		GlobalNetwork_FC7=self.selu(self.global_fc7(GlobalInput7)) #global network
		# [b,512,h/128,w/128]
		encoder7=self.instance_normal7(encoder7)+self.global_to_local_fc7(GlobalNetwork_FC6) #local 



		# # [batch,512,h/256,w/256]
		encoder8 = self.conv8(self.leaky_relu(encoder7)) # local
		# [b,512,h/256,w/256]
		encoder8=encoder8+self.global_to_local_fc8(GlobalNetwork_FC7) #local 

		# [b,1024,1,1]
		GlobalInput8=torch.cat((GlobalNetwork_FC7,encoder8),1)   #global network
		# [b,512,1,1]
		GlobalNetwork_FC8=self.selu(self.global_fc8(GlobalInput8)) #global network

		################################## decoder #############################################
		# [batch,512,h/128,w/128]
		decoder1 = self.deconv1(self.leaky_relu(encoder8))
		# [b,1024,1,1]
		# GlobalInput_de1=torch.cat((GlobalNetwork_FC8,torch.mean(decoder1,dim=[2,3]).unsqueeze(-1).unsqueeze(-1)),1)   #global network
		GlobalInput_de1=torch.cat((GlobalNetwork_FC8,mymean(decoder1)),1)   #global network

		# [b,512,1,1]
		GlobalNetwork_de_FC1=self.selu(self.global_de_fc1(GlobalInput_de1)) #global network
		# [b,512,h/128,w/128]
		decoder1=self.instance_normal_de_1(decoder1)+self.global_to_local_de_fc1(GlobalNetwork_FC8) #local 
		# [batch,1024,h/128,w/128]
		decoder1 = torch.cat((self.dropout(decoder1), encoder7), 1)


		# [batch,512,h/64,w/64]
		decoder2 = self.deconv2(self.leaky_relu(decoder1))
		# [b,1024,1,1]
		# GlobalInput_de2=torch.cat((GlobalNetwork_de_FC1,torch.mean(decoder2,dim=[2,3]).unsqueeze(-1).unsqueeze(-1)),1)   #global network
		GlobalInput_de2=torch.cat((GlobalNetwork_de_FC1,mymean(decoder2)),1)   #global network

		# [b,512,1,1]
		GlobalNetwork_de_FC2=self.selu(self.global_de_fc2(GlobalInput_de2)) #global network
		# [b,512,h/64,w/64]
		decoder2=self.instance_normal_de_2(decoder2)+self.global_to_local_de_fc2(GlobalNetwork_de_FC1) #local 
		# [batch,1024,h/64,w/64]
		decoder2 = torch.cat((self.dropout(decoder2), encoder6), 1)


		# [batch,512,h/32,w/32]
		decoder3 = self.deconv3(self.leaky_relu(decoder2))
		# [b,1024,1,1]
		# GlobalInput_de3=torch.cat((GlobalNetwork_de_FC2,torch.mean(decoder3,dim=[2,3]).unsqueeze(-1).unsqueeze(-1)),1)   #global network
		GlobalInput_de3=torch.cat((GlobalNetwork_de_FC2,mymean(decoder3)),1)   #global network

		# [b,512,1,1]
		GlobalNetwork_de_FC3=self.selu(self.global_de_fc3(GlobalInput_de3)) #global network
		# [b,512,h/32,w/32]
		decoder3=self.instance_normal_de_3(decoder3)+self.global_to_local_de_fc3(GlobalNetwork_de_FC2) #local 
		# [batch,1024,h/32,w/32]
		decoder3 = torch.cat((self.dropout(decoder3), encoder5), 1)


		# [batch,512,h/16,w/16]
		decoder4 = self.deconv4(self.leaky_relu(decoder3))
		# [b,1024,1,1]
		# GlobalInput_de4=torch.cat((GlobalNetwork_de_FC3,torch.mean(decoder4,dim=[2,3]).unsqueeze(-1).unsqueeze(-1)),1)   #global network
		GlobalInput_de4=torch.cat((GlobalNetwork_de_FC3,mymean(decoder4)),1)   #global network

		# [b,512,1,1]
		GlobalNetwork_de_FC4=self.selu(self.global_de_fc4(GlobalInput_de4)) #global network
		# [b,512,h/16,w/16]
		decoder4=self.instance_normal_de_4(decoder4)+self.global_to_local_de_fc4(GlobalNetwork_de_FC3) #local 
		# [batch,1024,h/16,w/16]
		decoder4 = torch.cat((decoder4, encoder4), 1)


		# [batch,256,h/8,w/8]
		decoder5 = self.deconv5(self.leaky_relu(decoder4))
		# print('mean: ',self.Mean(decoder5).shape,'|global: ',GlobalNetwork_de_FC4.shape)
		# [b,768,1,1]
		# GlobalInput_de5=torch.cat((GlobalNetwork_de_FC4,torch.mean(decoder5,dim=[2,3]).unsqueeze(-1).unsqueeze(-1)),1)   #global network
		GlobalInput_de5=torch.cat((GlobalNetwork_de_FC4,mymean(decoder5)),1)   #global network

		# [b,256,1,1]
		GlobalNetwork_de_FC5=self.selu(self.global_de_fc5(GlobalInput_de5)) #global network
		# [b,256,h/8,w/8]
		decoder5=self.instance_normal_de_5(decoder5)+self.global_to_local_de_fc5(GlobalNetwork_de_FC4) #local 
		# [batch,512,h/8,w/8]
		decoder5 = torch.cat((decoder5, encoder3), 1)


		# [batch,128,h/4,w/4]
		decoder6 = self.deconv6(self.leaky_relu(decoder5))
		# [b,384,1,1]
		# GlobalInput_de6=torch.cat((GlobalNetwork_de_FC5,torch.mean(decoder6,dim=[2,3]).unsqueeze(-1).unsqueeze(-1)),1)   #global network
		GlobalInput_de6=torch.cat((GlobalNetwork_de_FC5,mymean(decoder6)),1)   #global network

		# [b,128,1,1]
		GlobalNetwork_de_FC6=self.selu(self.global_de_fc6(GlobalInput_de6)) #global network
		# [b,128,h/4,w/4]
		decoder6=self.instance_normal_de_6(decoder6)+self.global_to_local_de_fc6(GlobalNetwork_de_FC5) #local 
		# [batch,256,h/4,w/4]
		decoder6 = torch.cat((decoder6, encoder2), 1)


		# [batch,64,h/2,w/2]
		decoder7 = self.deconv7(self.leaky_relu(decoder6))
		# [b,192,1,1]
		# GlobalInput_de7=torch.cat((GlobalNetwork_de_FC6,torch.mean(decoder7,dim=[2,3]).unsqueeze(-1).unsqueeze(-1)),1)   #global network
		GlobalInput_de7=torch.cat((GlobalNetwork_de_FC6,mymean(decoder7)),1)   #global network

		# [b,64,1,1]
		GlobalNetwork_de_FC7=self.selu(self.global_de_fc7(GlobalInput_de7)) #global network
		# [b,64,h/2,w/2]
		decoder7=self.instance_normal_de_7(decoder7)+self.global_to_local_de_fc7(GlobalNetwork_de_FC6) #local 
		# [batch,128,h/2,w/2]
		decoder7 = torch.cat((decoder7, encoder1), 1)
		# print(decoder7.shape)


		# [batch,9,h,w]
		decoder8 = self.deconv8(self.leaky_relu(decoder7))
		# [batch,9,h,w]
		decoder8=decoder8+self.global_to_local_de_fc8(GlobalNetwork_de_FC7) #local 

		output = self.tan(decoder8)
		# print(output.shape)
		# print('decoder: ',output.permute(0,2,3,1))

		return output,None



class LocalVA_Net(nn.Module):
	def __init__(self,input_channel,output_channel):
		super(LocalVA_Net,self).__init__()

		## define local networks
		#encoder and downsampling
		self.conv1 = nn.Conv2d(input_channel,64,4,2,1,bias=False)
		self.conv2 = nn.Conv2d(64,128,4,2,1,bias=False)
		self.conv3 = nn.Conv2d(128,256,4,2,1,bias=False)
		self.conv4 = nn.Conv2d(256,512,4,2,1,bias=False)
		self.conv5 = nn.Conv2d(512,512,4,2,1,bias=False)
		self.conv6 = nn.Conv2d(512,512,4,2,1,bias=False)
		self.conv7 = nn.Conv2d(512,512,4,2,1,bias=False)
		self.conv8 = nn.Conv2d(512,512,4,2,1,bias=False)

		#decoder
		self.deconv1 = Deconv(512, 512)
		self.deconv2 = Deconv(1024, 512)
		self.deconv3 = Deconv(1024, 512)
		self.deconv4 = Deconv(1024, 512)
		self.deconv5 = Deconv(1024, 256)
		self.deconv6 = Deconv(512, 128)
		self.deconv7 = Deconv(256, 64)
		self.deconv8 = Deconv(128, output_channel)

		self.sig = nn.Sigmoid()
		self.tan = nn.Tanh()


		self.leaky_relu = nn.LeakyReLU(0.2)

		# self.instance_normal1 = nn.InstanceNorm2d(64,affine=True)
		self.instance_normal2 = nn.InstanceNorm2d(128,affine=True)
		self.instance_normal3 = nn.InstanceNorm2d(256,affine=True)
		self.instance_normal4 = nn.InstanceNorm2d(512,affine=True)
		self.instance_normal5 = nn.InstanceNorm2d(512,affine=True)
		self.instance_normal6 = nn.InstanceNorm2d(512,affine=True)
		self.instance_normal7 = nn.InstanceNorm2d(512,affine=True)

		self.instance_normal_de_1 = nn.InstanceNorm2d(512,affine=True)
		self.instance_normal_de_2 = nn.InstanceNorm2d(512,affine=True)
		self.instance_normal_de_3 = nn.InstanceNorm2d(512,affine=True)
		self.instance_normal_de_4 = nn.InstanceNorm2d(512,affine=True)
		self.instance_normal_de_5 = nn.InstanceNorm2d(256,affine=True)
		self.instance_normal_de_6 = nn.InstanceNorm2d(128,affine=True)
		self.instance_normal_de_7 = nn.InstanceNorm2d(64,affine=True)

		self.dropout = nn.Dropout(0.5)


	def forward(self, input):


		# [batch,64,h/2,w/2]
		encoder1 = self.conv1(input) #local network

		# [batch,128,h/4,w/4]        
		encoder2 = self.conv2(self.leaky_relu(encoder1)) #local network

		# [b,128,h/4,w/4]
		encoder2=self.instance_normal2(encoder2)
		# print(self.conv2)

		# [batch,256,h/8,w/8]        
		encoder3 = self.conv3(self.leaky_relu(encoder2)) #local network

		# [b,256,h/8,w/8]
		encoder3=self.instance_normal3(encoder3) #local 


		# [batch,512,h/16,w/16]        
		encoder4 = self.conv4(self.leaky_relu(encoder3)) #local network

		# [b,512,h/16,w/16]
		encoder4=self.instance_normal4(encoder4) #local 


		# [batch,512,h/32,w/32]        
		encoder5 = self.conv5(self.leaky_relu(encoder4)) #local network

		# [b,512,h/32,w/32]
		encoder5=self.instance_normal5(encoder5) #local 


		# [batch,512,h/64,w/64]        
		encoder6 = self.conv6(self.leaky_relu(encoder5)) #local network

		# [b,512,h/64,w/64]
		encoder6=self.instance_normal6(encoder6) #local 


		# [batch,512,h/128,w/128]        
		encoder7 = self.conv7(self.leaky_relu(encoder6)) #local network

		# [b,512,h/128,w/128]
		encoder7=self.instance_normal7(encoder7) #local 


		# # [batch,512,h/256,w/256]
		encoder8 = self.conv8(self.leaky_relu(encoder7)) # local


		################################## decoder #############################################
		# [batch,512,h/128,w/128]
		decoder1 = self.deconv1(self.leaky_relu(encoder8))
		# [b,512,h/128,w/128]
		decoder1=self.instance_normal_de_1(decoder1) #local 
		# [batch,1024,h/128,w/128]
		decoder1 = torch.cat((self.dropout(decoder1), encoder7), 1)


		# [batch,512,h/64,w/64]
		decoder2 = self.deconv2(self.leaky_relu(decoder1))
		# [b,512,h/64,w/64]
		decoder2=self.instance_normal_de_2(decoder2)#local 
		# [batch,1024,h/64,w/64]
		decoder2 = torch.cat((self.dropout(decoder2), encoder6), 1)


		# [batch,512,h/32,w/32]
		decoder3 = self.deconv3(self.leaky_relu(decoder2))
		# [b,512,h/32,w/32]
		decoder3=self.instance_normal_de_3(decoder3) #local 
		# [batch,1024,h/32,w/32]
		decoder3 = torch.cat((self.dropout(decoder3), encoder5), 1)


		# [batch,512,h/16,w/16]
		decoder4 = self.deconv4(self.leaky_relu(decoder3))
		# [b,512,h/16,w/16]
		decoder4=self.instance_normal_de_4(decoder4)#local 
		# [batch,1024,h/16,w/16]
		decoder4 = torch.cat((decoder4, encoder4), 1)


		# [batch,256,h/8,w/8]
		decoder5 = self.deconv5(self.leaky_relu(decoder4))
		# [b,256,h/8,w/8]
		decoder5=self.instance_normal_de_5(decoder5) #local 
		# [batch,512,h/8,w/8]
		decoder5 = torch.cat((decoder5, encoder3), 1)


		# [batch,128,h/4,w/4]
		decoder6 = self.deconv6(self.leaky_relu(decoder5))
		# [b,128,h/4,w/4]
		decoder6=self.instance_normal_de_6(decoder6) #local 
		# [batch,256,h/4,w/4]
		decoder6 = torch.cat((decoder6, encoder2), 1)


		# [batch,64,h/2,w/2]
		decoder7 = self.deconv7(self.leaky_relu(decoder6))
		# [b,64,h/2,w/2]
		decoder7=self.instance_normal_de_7(decoder7) #local 
		# [batch,128,h/2,w/2]
		decoder7 = torch.cat((decoder7, encoder1), 1)


		# [batch,9,h,w]
		decoder8 = self.deconv8(self.leaky_relu(decoder7))

		output = self.tan(decoder8)
		# print(output.shape)

		return output, None



class NewVA_Net(nn.Module):
	def __init__(self,input_channel,output_channel, rough_channel):
		super(NewVA_Net,self).__init__()
		
		self.rough_nc=rough_channel

		## define local networks
		#encoder and downsampling
		self.conv1 = nn.Conv2d(input_channel,64,4,2,1,bias=False)
		self.conv2 = nn.Conv2d(64,128,4,2,1,bias=False)
		self.conv3 = nn.Conv2d(128,256,4,2,1,bias=False)
		self.conv4 = nn.Conv2d(256,512,4,2,1,bias=False)
		self.conv5 = nn.Conv2d(512,512,4,2,1,bias=False)
		self.conv6 = nn.Conv2d(512,512,4,2,1,bias=False)
		self.conv7 = nn.Conv2d(512,512,4,2,1,bias=False)
		self.conv8 = nn.Conv2d(512,512,4,2,1,bias=False)

		#decoder(diff)
		self.deconv1_diff = Deconv(512, 512)
		self.deconv2_diff = Deconv(1024, 512)
		self.deconv3_diff = Deconv(1024, 512)
		self.deconv4_diff = Deconv(1024, 512)
		self.deconv5_diff = Deconv(1024, 256)
		self.deconv6_diff = Deconv(512, 128)
		self.deconv7_diff = Deconv(256, 64)
		self.deconv8_diff = Deconv(128, output_channel)

		#decoder(normal)
		self.deconv1_normal = Deconv(512, 512)
		self.deconv2_normal = Deconv(1024, 512)
		self.deconv3_normal = Deconv(1024, 512)
		self.deconv4_normal = Deconv(1024, 512)
		self.deconv5_normal = Deconv(1024, 256)
		self.deconv6_normal = Deconv(512, 128)
		self.deconv7_normal = Deconv(256, 64)
		self.deconv8_normal = Deconv(128, output_channel)

		#decoder(rough)
		self.deconv1_rough = Deconv(512, 512)
		self.deconv2_rough = Deconv(1024, 512)
		self.deconv3_rough = Deconv(1024, 512)
		self.deconv4_rough = Deconv(1024, 512)
		self.deconv5_rough = Deconv(1024, 256)
		self.deconv6_rough = Deconv(512, 128)
		self.deconv7_rough = Deconv(256, 64)
		self.deconv8_rough = Deconv(128, rough_channel)

		#decoder(spec)
		self.deconv1_spec = Deconv(512, 512)
		self.deconv2_spec = Deconv(1024, 512)
		self.deconv3_spec = Deconv(1024, 512)
		self.deconv4_spec = Deconv(1024, 512)
		self.deconv5_spec = Deconv(1024, 256)
		self.deconv6_spec = Deconv(512, 128)
		self.deconv7_spec = Deconv(256, 64)
		self.deconv8_spec = Deconv(128, output_channel)

		self.sig = nn.Sigmoid()
		self.tan = nn.Tanh()


		self.leaky_relu = nn.LeakyReLU(0.2)

		# self.instance_normal1 = nn.InstanceNorm2d(64,affine=True)
		self.instance_normal2 = nn.InstanceNorm2d(128,affine=True)
		self.instance_normal3 = nn.InstanceNorm2d(256,affine=True)
		self.instance_normal4 = nn.InstanceNorm2d(512,affine=True)
		self.instance_normal5 = nn.InstanceNorm2d(512,affine=True)
		self.instance_normal6 = nn.InstanceNorm2d(512,affine=True)
		self.instance_normal7 = nn.InstanceNorm2d(512,affine=True)

		self.instance_normal_de_1_diff = nn.InstanceNorm2d(512,affine=True)
		self.instance_normal_de_2_diff = nn.InstanceNorm2d(512,affine=True)
		self.instance_normal_de_3_diff = nn.InstanceNorm2d(512,affine=True)
		self.instance_normal_de_4_diff = nn.InstanceNorm2d(512,affine=True)
		self.instance_normal_de_5_diff = nn.InstanceNorm2d(256,affine=True)
		self.instance_normal_de_6_diff = nn.InstanceNorm2d(128,affine=True)
		self.instance_normal_de_7_diff = nn.InstanceNorm2d(64,affine=True)

		self.instance_normal_de_1_normal = nn.InstanceNorm2d(512,affine=True)
		self.instance_normal_de_2_normal = nn.InstanceNorm2d(512,affine=True)
		self.instance_normal_de_3_normal = nn.InstanceNorm2d(512,affine=True)
		self.instance_normal_de_4_normal = nn.InstanceNorm2d(512,affine=True)
		self.instance_normal_de_5_normal = nn.InstanceNorm2d(256,affine=True)
		self.instance_normal_de_6_normal = nn.InstanceNorm2d(128,affine=True)
		self.instance_normal_de_7_normal = nn.InstanceNorm2d(64,affine=True)

		self.instance_normal_de_1_rough = nn.InstanceNorm2d(512,affine=True)
		self.instance_normal_de_2_rough = nn.InstanceNorm2d(512,affine=True)
		self.instance_normal_de_3_rough = nn.InstanceNorm2d(512,affine=True)
		self.instance_normal_de_4_rough = nn.InstanceNorm2d(512,affine=True)
		self.instance_normal_de_5_rough = nn.InstanceNorm2d(256,affine=True)
		self.instance_normal_de_6_rough = nn.InstanceNorm2d(128,affine=True)
		self.instance_normal_de_7_rough = nn.InstanceNorm2d(64,affine=True)

		self.instance_normal_de_1_spec = nn.InstanceNorm2d(512,affine=True)
		self.instance_normal_de_2_spec = nn.InstanceNorm2d(512,affine=True)
		self.instance_normal_de_3_spec = nn.InstanceNorm2d(512,affine=True)
		self.instance_normal_de_4_spec = nn.InstanceNorm2d(512,affine=True)
		self.instance_normal_de_5_spec = nn.InstanceNorm2d(256,affine=True)
		self.instance_normal_de_6_spec = nn.InstanceNorm2d(128,affine=True)
		self.instance_normal_de_7_spec = nn.InstanceNorm2d(64,affine=True)

		self.dropout = nn.Dropout(0.5)


	def forward(self, input):

		# [batch,64,h/2,w/2]
		encoder1 = self.conv1(input) #local network
		# [batch,128,h/4,w/4]        
		encoder2 = self.instance_normal2(self.conv2(self.leaky_relu(encoder1))) #local network
		# [batch,256,h/8,w/8]        
		encoder3 = self.instance_normal3(self.conv3(self.leaky_relu(encoder2))) #local network
		# [batch,512,h/16,w/16]        
		encoder4 = self.instance_normal4(self.conv4(self.leaky_relu(encoder3))) #local network
		# [batch,512,h/32,w/32]        
		encoder5 = self.instance_normal5(self.conv5(self.leaky_relu(encoder4))) #local network
		# [batch,512,h/64,w/64]        
		encoder6 = self.instance_normal6(self.conv6(self.leaky_relu(encoder5))) #local network
		# [batch,512,h/128,w/128]        
		encoder7 = self.instance_normal7(self.conv7(self.leaky_relu(encoder6))) #local network
		# [batch,512,h/256,w/256]
		encoder8 = self.conv8(self.leaky_relu(encoder7)) # local


		################################## decoder (diff) #############################################
		# [batch,512,h/128,w/128]
		decoder1_diff = self.instance_normal_de_1_diff(self.deconv1_diff(self.leaky_relu(encoder8)))
		# [batch,1024,h/128,w/128]
		decoder1_diff = torch.cat((self.dropout(decoder1_diff), encoder7), 1)

		# [batch,512,h/64,w/64]
		decoder2_diff = self.instance_normal_de_2_diff(self.deconv2_diff(self.leaky_relu(decoder1_diff)))
		# [batch,1024,h/64,w/64]
		decoder2_diff = torch.cat((self.dropout(decoder2_diff), encoder6), 1)

		# [batch,512,h/32,w/32]
		decoder3_diff = self.instance_normal_de_3_diff(self.deconv3_diff(self.leaky_relu(decoder2_diff)))
		# [batch,1024,h/32,w/32]
		decoder3_diff = torch.cat((self.dropout(decoder3_diff), encoder5), 1)

		# [batch,512,h/16,w/16]
		decoder4_diff = self.instance_normal_de_4_diff(self.deconv4_diff(self.leaky_relu(decoder3_diff)))
		# [batch,1024,h/16,w/16]
		decoder4_diff = torch.cat((decoder4_diff, encoder4), 1)

		# [batch,256,h/8,w/8]
		decoder5_diff = self.instance_normal_de_5_diff(self.deconv5_diff(self.leaky_relu(decoder4_diff)))
		# [batch,512,h/8,w/8]
		decoder5_diff = torch.cat((decoder5_diff, encoder3), 1)

		# [batch,128,h/4,w/4]
		decoder6_diff = self.instance_normal_de_6_diff(self.deconv6_diff(self.leaky_relu(decoder5_diff)))
		# [batch,256,h/4,w/4]
		decoder6_diff = torch.cat((decoder6_diff, encoder2), 1)

		# [batch,64,h/2,w/2]
		decoder7_diff = self.instance_normal_de_7_diff(self.deconv7_diff(self.leaky_relu(decoder6_diff)))
		# [batch,128,h/2,w/2]
		decoder7_diff = torch.cat((decoder7_diff, encoder1), 1)

		# [batch,out_c,h,w]
		decoder8_diff = self.deconv8_diff(self.leaky_relu(decoder7_diff))

		diff = self.tan(decoder8_diff)
		# print(output.shape)

		################################## decoder (normal) #############################################
		# [batch,512,h/128,w/128]
		decoder1_normal = self.instance_normal_de_1_normal(self.deconv1_normal(self.leaky_relu(encoder8)))
		# [batch,1024,h/128,w/128]
		decoder1_normal = torch.cat((self.dropout(decoder1_normal), encoder7), 1)

		# [batch,512,h/64,w/64]
		decoder2_normal = self.instance_normal_de_2_normal(self.deconv2_normal(self.leaky_relu(decoder1_normal)))
		# [batch,1024,h/64,w/64]
		decoder2_normal = torch.cat((self.dropout(decoder2_normal), encoder6), 1)

		# [batch,512,h/32,w/32]
		decoder3_normal = self.instance_normal_de_3_normal(self.deconv3_normal(self.leaky_relu(decoder2_normal)))
		# [batch,1024,h/32,w/32]
		decoder3_normal = torch.cat((self.dropout(decoder3_normal), encoder5), 1)

		# [batch,512,h/16,w/16]
		decoder4_normal = self.instance_normal_de_4_normal(self.deconv4_normal(self.leaky_relu(decoder3_normal)))
		# [batch,1024,h/16,w/16]
		decoder4_normal = torch.cat((decoder4_normal, encoder4), 1)

		# [batch,256,h/8,w/8]
		decoder5_normal = self.instance_normal_de_5_normal(self.deconv5_normal(self.leaky_relu(decoder4_normal)))
		# [batch,512,h/8,w/8]
		decoder5_normal = torch.cat((decoder5_normal, encoder3), 1)

		# [batch,128,h/4,w/4]
		decoder6_normal = self.instance_normal_de_6_normal(self.deconv6_normal(self.leaky_relu(decoder5_normal)))
		# [batch,256,h/4,w/4]
		decoder6_normal = torch.cat((decoder6_normal, encoder2), 1)

		# [batch,64,h/2,w/2]
		decoder7_normal = self.instance_normal_de_7_normal(self.deconv7_normal(self.leaky_relu(decoder6_normal)))
		# [batch,128,h/2,w/2]
		decoder7_normal = torch.cat((decoder7_normal, encoder1), 1)

		# [batch,out_c,h,w]
		decoder8_normal = self.deconv8_normal(self.leaky_relu(decoder7_normal))

		normal = self.tan(decoder8_normal)
		# print(output.shape)
	 
		################################## decoder (normal) #############################################
		# [batch,512,h/128,w/128]
		decoder1_rough = self.instance_normal_de_1_rough(self.deconv1_rough(self.leaky_relu(encoder8)))
		# [batch,1024,h/128,w/128]
		decoder1_rough = torch.cat((self.dropout(decoder1_rough), encoder7), 1)

		# [batch,512,h/64,w/64]
		decoder2_rough = self.instance_normal_de_2_rough(self.deconv2_rough(self.leaky_relu(decoder1_rough)))
		# [batch,1024,h/64,w/64]
		decoder2_rough = torch.cat((self.dropout(decoder2_rough), encoder6), 1)

		# [batch,512,h/32,w/32]
		decoder3_rough = self.instance_normal_de_3_rough(self.deconv3_rough(self.leaky_relu(decoder2_rough)))
		# [batch,1024,h/32,w/32]
		decoder3_rough = torch.cat((self.dropout(decoder3_rough), encoder5), 1)

		# [batch,512,h/16,w/16]
		decoder4_rough = self.instance_normal_de_4_rough(self.deconv4_rough(self.leaky_relu(decoder3_rough)))
		# [batch,1024,h/16,w/16]
		decoder4_rough = torch.cat((decoder4_rough, encoder4), 1)

		# [batch,256,h/8,w/8]
		decoder5_rough = self.instance_normal_de_5_rough(self.deconv5_rough(self.leaky_relu(decoder4_rough)))
		# [batch,512,h/8,w/8]
		decoder5_rough = torch.cat((decoder5_rough, encoder3), 1)

		# [batch,128,h/4,w/4]
		decoder6_rough = self.instance_normal_de_6_rough(self.deconv6_rough(self.leaky_relu(decoder5_rough)))
		# [batch,256,h/4,w/4]
		decoder6_rough = torch.cat((decoder6_rough, encoder2), 1)

		# [batch,64,h/2,w/2]
		decoder7_rough = self.instance_normal_de_7_rough(self.deconv7_rough(self.leaky_relu(decoder6_rough)))
		# [batch,128,h/2,w/2]
		decoder7_rough = torch.cat((decoder7_rough, encoder1), 1)

		# [batch,_out_c,h,w]
		decoder8_rough = self.deconv8_rough(self.leaky_relu(decoder7_rough))

		rough = self.tan(decoder8_rough)
		# print(output.shape)
		if self.rough_nc==1:
			rough=rough.repeat(1,3,1,1)

		################################## decoder (normal) #############################################
		# [batch,512,h/128,w/128]
		decoder1_spec = self.instance_normal_de_1_spec(self.deconv1_spec(self.leaky_relu(encoder8)))
		# [batch,1024,h/128,w/128]
		decoder1_spec = torch.cat((self.dropout(decoder1_spec), encoder7), 1)

		# [batch,512,h/64,w/64]
		decoder2_spec = self.instance_normal_de_2_spec(self.deconv2_spec(self.leaky_relu(decoder1_spec)))
		# [batch,1024,h/64,w/64]
		decoder2_spec = torch.cat((self.dropout(decoder2_spec), encoder6), 1)

		# [batch,512,h/32,w/32]
		decoder3_spec = self.instance_normal_de_3_spec(self.deconv3_spec(self.leaky_relu(decoder2_spec)))
		# [batch,1024,h/32,w/32]
		decoder3_spec = torch.cat((self.dropout(decoder3_spec), encoder5), 1)

		# [batch,512,h/16,w/16]
		decoder4_spec = self.instance_normal_de_4_spec(self.deconv4_spec(self.leaky_relu(decoder3_spec)))
		# [batch,1024,h/16,w/16]
		decoder4_spec = torch.cat((decoder4_spec, encoder4), 1)

		# [batch,256,h/8,w/8]
		decoder5_spec = self.instance_normal_de_5_spec(self.deconv5_spec(self.leaky_relu(decoder4_spec)))
		# [batch,512,h/8,w/8]
		decoder5_spec = torch.cat((decoder5_spec, encoder3), 1)

		# [batch,128,h/4,w/4]
		decoder6_spec = self.instance_normal_de_6_spec(self.deconv6_spec(self.leaky_relu(decoder5_spec)))
		# [batch,256,h/4,w/4]
		decoder6_spec = torch.cat((decoder6_spec, encoder2), 1)

		# [batch,64,h/2,w/2]
		decoder7_spec = self.instance_normal_de_7_spec(self.deconv7_spec(self.leaky_relu(decoder6_spec)))
		# [batch,128,h/2,w/2]
		decoder7_spec = torch.cat((decoder7_spec, encoder1), 1)

		# [batch,out_c,h,w]
		decoder8_spec = self.deconv8_spec(self.leaky_relu(decoder7_spec))

		spec = self.tan(decoder8_spec)

		output=torch.cat((normal,diff,rough,spec),1)

		# print('shape: ',output.shape)

		return output, None




class NewVA_Net_Light(nn.Module):
	def __init__(self,input_channel,output_channel,rough_channel):
		super(NewVA_Net_Light,self).__init__()

		self.rough_nc=rough_channel
		## define local networks
		#encoder and downsampling
		self.conv1 = nn.Conv2d(input_channel,64,4,2,1,bias=False)
		self.conv2 = nn.Conv2d(64,128,4,2,1,bias=False)
		self.conv3 = nn.Conv2d(128,256,4,2,1,bias=False)
		self.conv4 = nn.Conv2d(256,512,4,2,1,bias=False)
		self.conv5 = nn.Conv2d(512,512,4,2,1,bias=False)
		self.conv6 = nn.Conv2d(512,512,4,2,1,bias=False)
		self.conv7 = nn.Conv2d(512,512,4,2,1,bias=False)
		self.conv8 = nn.Conv2d(512,512,4,2,1,bias=False)

		#decoder(diff)
		self.deconv1_diff = Deconv(512, 512)
		self.deconv2_diff = Deconv(1024, 512)
		self.deconv3_diff = Deconv(1024, 512)
		self.deconv4_diff = Deconv(1024, 512)
		self.deconv5_diff = Deconv(1024, 256)
		self.deconv6_diff = Deconv(512, 128)
		self.deconv7_diff = Deconv(256, 64)
		self.deconv8_diff = Deconv(128, output_channel)

		#decoder(normal)
		self.deconv1_normal = Deconv(512, 512)
		self.deconv2_normal = Deconv(1024, 512)
		self.deconv3_normal = Deconv(1024, 512)
		self.deconv4_normal = Deconv(1024, 512)
		self.deconv5_normal = Deconv(1024, 256)
		self.deconv6_normal = Deconv(512, 128)
		self.deconv7_normal = Deconv(256, 64)
		self.deconv8_normal = Deconv(128, output_channel)

		#decoder(rough)
		self.deconv1_rough = Deconv(512, 512)
		self.deconv2_rough = Deconv(1024, 512)
		self.deconv3_rough = Deconv(1024, 512)
		self.deconv4_rough = Deconv(1024, 512)
		self.deconv5_rough = Deconv(1024, 256)
		self.deconv6_rough = Deconv(512, 128)
		self.deconv7_rough = Deconv(256, 64)
		self.deconv8_rough = Deconv(128, rough_channel)

		#decoder(spec)
		self.deconv1_spec = Deconv(512, 512)
		self.deconv2_spec = Deconv(1024, 512)
		self.deconv3_spec = Deconv(1024, 512)
		self.deconv4_spec = Deconv(1024, 512)
		self.deconv5_spec = Deconv(1024, 256)
		self.deconv6_spec = Deconv(512, 128)
		self.deconv7_spec = Deconv(256, 64)
		self.deconv8_spec = Deconv(128, output_channel)

		self.sig = nn.Sigmoid()
		self.tan = nn.Tanh()


		self.leaky_relu = nn.LeakyReLU(0.2)

		# self.instance_normal1 = nn.InstanceNorm2d(64,affine=True)
		self.instance_normal2 = nn.InstanceNorm2d(128,affine=True)
		self.instance_normal3 = nn.InstanceNorm2d(256,affine=True)
		self.instance_normal4 = nn.InstanceNorm2d(512,affine=True)
		self.instance_normal5 = nn.InstanceNorm2d(512,affine=True)
		self.instance_normal6 = nn.InstanceNorm2d(512,affine=True)
		self.instance_normal7 = nn.InstanceNorm2d(512,affine=True)

		self.instance_normal_de_1_diff = nn.InstanceNorm2d(512,affine=True)
		self.instance_normal_de_2_diff = nn.InstanceNorm2d(512,affine=True)
		self.instance_normal_de_3_diff = nn.InstanceNorm2d(512,affine=True)
		self.instance_normal_de_4_diff = nn.InstanceNorm2d(512,affine=True)
		self.instance_normal_de_5_diff = nn.InstanceNorm2d(256,affine=True)
		self.instance_normal_de_6_diff = nn.InstanceNorm2d(128,affine=True)
		self.instance_normal_de_7_diff = nn.InstanceNorm2d(64,affine=True)

		self.instance_normal_de_1_normal = nn.InstanceNorm2d(512,affine=True)
		self.instance_normal_de_2_normal = nn.InstanceNorm2d(512,affine=True)
		self.instance_normal_de_3_normal = nn.InstanceNorm2d(512,affine=True)
		self.instance_normal_de_4_normal = nn.InstanceNorm2d(512,affine=True)
		self.instance_normal_de_5_normal = nn.InstanceNorm2d(256,affine=True)
		self.instance_normal_de_6_normal = nn.InstanceNorm2d(128,affine=True)
		self.instance_normal_de_7_normal = nn.InstanceNorm2d(64,affine=True)

		self.instance_normal_de_1_rough = nn.InstanceNorm2d(512,affine=True)
		self.instance_normal_de_2_rough = nn.InstanceNorm2d(512,affine=True)
		self.instance_normal_de_3_rough = nn.InstanceNorm2d(512,affine=True)
		self.instance_normal_de_4_rough = nn.InstanceNorm2d(512,affine=True)
		self.instance_normal_de_5_rough = nn.InstanceNorm2d(256,affine=True)
		self.instance_normal_de_6_rough = nn.InstanceNorm2d(128,affine=True)
		self.instance_normal_de_7_rough = nn.InstanceNorm2d(64,affine=True)

		self.instance_normal_de_1_spec = nn.InstanceNorm2d(512,affine=True)
		self.instance_normal_de_2_spec = nn.InstanceNorm2d(512,affine=True)
		self.instance_normal_de_3_spec = nn.InstanceNorm2d(512,affine=True)
		self.instance_normal_de_4_spec = nn.InstanceNorm2d(512,affine=True)
		self.instance_normal_de_5_spec = nn.InstanceNorm2d(256,affine=True)
		self.instance_normal_de_6_spec = nn.InstanceNorm2d(128,affine=True)
		self.instance_normal_de_7_spec = nn.InstanceNorm2d(64,affine=True)

		self.dropout = nn.Dropout(0.5)

		########################### This is for light ##########################
		# self.Mean=mymean()
		self.FC1_Light=FC(512*1*1,256,True)
		self.FC2_Light=FC(256,128,True)
		self.FC3_Light=FC(128,64,True)
		self.FC4_Light=FC(64,32,True)
		self.FC5_Light=FC(32,16,True)
		self.FC6_Light=FC(16,3,True)


	def forward(self, input):

		# [batch,64,h/2,w/2]
		encoder1 = self.conv1(input) #local network
		# [batch,128,h/4,w/4]        
		encoder2 = self.instance_normal2(self.conv2(self.leaky_relu(encoder1))) #local network
		# [batch,256,h/8,w/8]        
		encoder3 = self.instance_normal3(self.conv3(self.leaky_relu(encoder2))) #local network
		# [batch,512,h/16,w/16]        
		encoder4 = self.instance_normal4(self.conv4(self.leaky_relu(encoder3))) #local network
		# [batch,512,h/32,w/32]        
		encoder5 = self.instance_normal5(self.conv5(self.leaky_relu(encoder4))) #local network
		# [batch,512,h/64,w/64]        
		encoder6 = self.instance_normal6(self.conv6(self.leaky_relu(encoder5))) #local network
		# [batch,512,h/128,w/128]        
		encoder7 = self.instance_normal7(self.conv7(self.leaky_relu(encoder6))) #local network
		# [batch,512,h/256,w/256]
		encoder8 = self.conv8(self.leaky_relu(encoder7)) # local


		################################## decoder (diff) #############################################
		# [batch,512,h/128,w/128]
		decoder1_diff = self.instance_normal_de_1_diff(self.deconv1_diff(self.leaky_relu(encoder8)))
		# [batch,1024,h/128,w/128]
		decoder1_diff = torch.cat((self.dropout(decoder1_diff), encoder7), 1)

		# [batch,512,h/64,w/64]
		decoder2_diff = self.instance_normal_de_2_diff(self.deconv2_diff(self.leaky_relu(decoder1_diff)))
		# [batch,1024,h/64,w/64]
		decoder2_diff = torch.cat((self.dropout(decoder2_diff), encoder6), 1)

		# [batch,512,h/32,w/32]
		decoder3_diff = self.instance_normal_de_3_diff(self.deconv3_diff(self.leaky_relu(decoder2_diff)))
		# [batch,1024,h/32,w/32]
		decoder3_diff = torch.cat((self.dropout(decoder3_diff), encoder5), 1)

		# [batch,512,h/16,w/16]
		decoder4_diff = self.instance_normal_de_4_diff(self.deconv4_diff(self.leaky_relu(decoder3_diff)))
		# [batch,1024,h/16,w/16]
		decoder4_diff = torch.cat((decoder4_diff, encoder4), 1)

		# [batch,256,h/8,w/8]
		decoder5_diff = self.instance_normal_de_5_diff(self.deconv5_diff(self.leaky_relu(decoder4_diff)))
		# [batch,512,h/8,w/8]
		decoder5_diff = torch.cat((decoder5_diff, encoder3), 1)

		# [batch,128,h/4,w/4]
		decoder6_diff = self.instance_normal_de_6_diff(self.deconv6_diff(self.leaky_relu(decoder5_diff)))
		# [batch,256,h/4,w/4]
		decoder6_diff = torch.cat((decoder6_diff, encoder2), 1)

		# [batch,64,h/2,w/2]
		decoder7_diff = self.instance_normal_de_7_diff(self.deconv7_diff(self.leaky_relu(decoder6_diff)))
		# [batch,128,h/2,w/2]
		decoder7_diff = torch.cat((decoder7_diff, encoder1), 1)

		# [batch,out_c,h,w]
		decoder8_diff = self.deconv8_diff(self.leaky_relu(decoder7_diff))

		diff = self.tan(decoder8_diff)
		# print(output.shape)

		################################## decoder (normal) #############################################
		# [batch,512,h/128,w/128]
		decoder1_normal = self.instance_normal_de_1_normal(self.deconv1_normal(self.leaky_relu(encoder8)))
		# [batch,1024,h/128,w/128]
		decoder1_normal = torch.cat((self.dropout(decoder1_normal), encoder7), 1)

		# [batch,512,h/64,w/64]
		decoder2_normal = self.instance_normal_de_2_normal(self.deconv2_normal(self.leaky_relu(decoder1_normal)))
		# [batch,1024,h/64,w/64]
		decoder2_normal = torch.cat((self.dropout(decoder2_normal), encoder6), 1)

		# [batch,512,h/32,w/32]
		decoder3_normal = self.instance_normal_de_3_normal(self.deconv3_normal(self.leaky_relu(decoder2_normal)))
		# [batch,1024,h/32,w/32]
		decoder3_normal = torch.cat((self.dropout(decoder3_normal), encoder5), 1)

		# [batch,512,h/16,w/16]
		decoder4_normal = self.instance_normal_de_4_normal(self.deconv4_normal(self.leaky_relu(decoder3_normal)))
		# [batch,1024,h/16,w/16]
		decoder4_normal = torch.cat((decoder4_normal, encoder4), 1)

		# [batch,256,h/8,w/8]
		decoder5_normal = self.instance_normal_de_5_normal(self.deconv5_normal(self.leaky_relu(decoder4_normal)))
		# [batch,512,h/8,w/8]
		decoder5_normal = torch.cat((decoder5_normal, encoder3), 1)

		# [batch,128,h/4,w/4]
		decoder6_normal = self.instance_normal_de_6_normal(self.deconv6_normal(self.leaky_relu(decoder5_normal)))
		# [batch,256,h/4,w/4]
		decoder6_normal = torch.cat((decoder6_normal, encoder2), 1)

		# [batch,64,h/2,w/2]
		decoder7_normal = self.instance_normal_de_7_normal(self.deconv7_normal(self.leaky_relu(decoder6_normal)))
		# [batch,128,h/2,w/2]
		decoder7_normal = torch.cat((decoder7_normal, encoder1), 1)

		# [batch,out_c,h,w]
		decoder8_normal = self.deconv8_normal(self.leaky_relu(decoder7_normal))

		normal = self.tan(decoder8_normal)
		# print(output.shape)

		################################## decoder (normal) #############################################
		# [batch,512,h/128,w/128]
		decoder1_rough = self.instance_normal_de_1_rough(self.deconv1_rough(self.leaky_relu(encoder8)))
		# [batch,1024,h/128,w/128]
		decoder1_rough = torch.cat((self.dropout(decoder1_rough), encoder7), 1)

		# [batch,512,h/64,w/64]
		decoder2_rough = self.instance_normal_de_2_rough(self.deconv2_rough(self.leaky_relu(decoder1_rough)))
		# [batch,1024,h/64,w/64]
		decoder2_rough = torch.cat((self.dropout(decoder2_rough), encoder6), 1)

		# [batch,512,h/32,w/32]
		decoder3_rough = self.instance_normal_de_3_rough(self.deconv3_rough(self.leaky_relu(decoder2_rough)))
		# [batch,1024,h/32,w/32]
		decoder3_rough = torch.cat((self.dropout(decoder3_rough), encoder5), 1)

		# [batch,512,h/16,w/16]
		decoder4_rough = self.instance_normal_de_4_rough(self.deconv4_rough(self.leaky_relu(decoder3_rough)))
		# [batch,1024,h/16,w/16]
		decoder4_rough = torch.cat((decoder4_rough, encoder4), 1)

		# [batch,256,h/8,w/8]
		decoder5_rough = self.instance_normal_de_5_rough(self.deconv5_rough(self.leaky_relu(decoder4_rough)))
		# [batch,512,h/8,w/8]
		decoder5_rough = torch.cat((decoder5_rough, encoder3), 1)

		# [batch,128,h/4,w/4]
		decoder6_rough = self.instance_normal_de_6_rough(self.deconv6_rough(self.leaky_relu(decoder5_rough)))
		# [batch,256,h/4,w/4]
		decoder6_rough = torch.cat((decoder6_rough, encoder2), 1)

		# [batch,64,h/2,w/2]
		decoder7_rough = self.instance_normal_de_7_rough(self.deconv7_rough(self.leaky_relu(decoder6_rough)))
		# [batch,128,h/2,w/2]
		decoder7_rough = torch.cat((decoder7_rough, encoder1), 1)

		# [batch,_out_c,h,w]
		decoder8_rough = self.deconv8_rough(self.leaky_relu(decoder7_rough))

		rough = self.tan(decoder8_rough)
		# print(output.shape)

		################################## decoder (normal) #############################################
		# [batch,512,h/128,w/128]
		decoder1_spec = self.instance_normal_de_1_spec(self.deconv1_spec(self.leaky_relu(encoder8)))
		# [batch,1024,h/128,w/128]
		decoder1_spec = torch.cat((self.dropout(decoder1_spec), encoder7), 1)

		# [batch,512,h/64,w/64]
		decoder2_spec = self.instance_normal_de_2_spec(self.deconv2_spec(self.leaky_relu(decoder1_spec)))
		# [batch,1024,h/64,w/64]
		decoder2_spec = torch.cat((self.dropout(decoder2_spec), encoder6), 1)

		# [batch,512,h/32,w/32]
		decoder3_spec = self.instance_normal_de_3_spec(self.deconv3_spec(self.leaky_relu(decoder2_spec)))
		# [batch,1024,h/32,w/32]
		decoder3_spec = torch.cat((self.dropout(decoder3_spec), encoder5), 1)

		# [batch,512,h/16,w/16]
		decoder4_spec = self.instance_normal_de_4_spec(self.deconv4_spec(self.leaky_relu(decoder3_spec)))
		# [batch,1024,h/16,w/16]
		decoder4_spec = torch.cat((decoder4_spec, encoder4), 1)

		# [batch,256,h/8,w/8]
		decoder5_spec = self.instance_normal_de_5_spec(self.deconv5_spec(self.leaky_relu(decoder4_spec)))
		# [batch,512,h/8,w/8]
		decoder5_spec = torch.cat((decoder5_spec, encoder3), 1)

		# [batch,128,h/4,w/4]
		decoder6_spec = self.instance_normal_de_6_spec(self.deconv6_spec(self.leaky_relu(decoder5_spec)))
		# [batch,256,h/4,w/4]
		decoder6_spec = torch.cat((decoder6_spec, encoder2), 1)

		# [batch,64,h/2,w/2]
		decoder7_spec = self.instance_normal_de_7_spec(self.deconv7_spec(self.leaky_relu(decoder6_spec)))
		# [batch,128,h/2,w/2]
		decoder7_spec = torch.cat((decoder7_spec, encoder1), 1)

		# [batch,out_c,h,w]
		decoder8_spec = self.deconv8_spec(self.leaky_relu(decoder7_spec))

		spec = self.tan(decoder8_spec)

		if self.rough_nc==1:
			rough=rough.repeat(1,3,1,1)

		output=torch.cat((normal,diff,rough,spec),1)

		# print('shape: ',output.shape)

		########################################## Estimate Light #################################################################
		#[B,1,512]
		flat_encoder8=encoder8.view(-1,self.num_flat_features(encoder8))
		#[B,1,256]
		LightPo= self.leaky_relu(self.FC1_Light(flat_encoder8))
		#[B,1,128]
		LightPo= self.leaky_relu(self.FC2_Light(LightPo))
		#[B,1,64]
		LightPo= self.leaky_relu(self.FC3_Light(LightPo))
		#[B,1,32]
		LightPo= self.leaky_relu(self.FC4_Light(LightPo))
		#[B,1,16]
		LightPo= self.leaky_relu(self.FC5_Light(LightPo))
		#[B,1,3]
		LightPo= self.FC6_Light(LightPo)

		return output, LightPo

	def num_flat_features(self, x):
		size = x.size()[1:]  # all dimensions except the batch dimension
		num_features = 1
		for s in size:
			num_features *= s
		return num_features



class Encoder(nn.Module):
	def __init__(self, input_nc, output_nc, ngf=32, n_downsampling=4, norm_layer=nn.BatchNorm2d):
		super(Encoder, self).__init__()        
		self.output_nc = output_nc        

		model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), 
				 norm_layer(ngf), nn.ReLU(True)]             
		### downsample
		for i in range(n_downsampling):
			mult = 2**i
			model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
					  norm_layer(ngf * mult * 2), nn.ReLU(True)]

		### upsample         
		for i in range(n_downsampling):
			mult = 2**(n_downsampling - i)
			model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
					   norm_layer(int(ngf * mult / 2)), nn.ReLU(True)]        

		model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
		self.model = nn.Sequential(*model) 

	def forward(self, input, inst):
		outputs = self.model(input)

		# instance-wise average pooling
		outputs_mean = outputs.clone()
		inst_list = np.unique(inst.cpu().numpy().astype(int))        
		for i in inst_list:
			for b in range(input.size()[0]):
				indices = (inst[b:b+1] == int(i)).nonzero() # n x 4            
				for j in range(self.output_nc):
					output_ins = outputs[indices[:,0] + b, indices[:,1] + j, indices[:,2], indices[:,3]]                    
					mean_feat = torch.mean(output_ins).expand_as(output_ins)                                        
					outputs_mean[indices[:,0] + b, indices[:,1] + j, indices[:,2], indices[:,3]] = mean_feat                       
		return outputs_mean

class MultiscaleDiscriminator(nn.Module):
	def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, 
				 use_sigmoid=False, num_D=3, getIntermFeat=False, use_dropout=False):
		super(MultiscaleDiscriminator, self).__init__()
		self.num_D = num_D
		self.n_layers = n_layers
		self.getIntermFeat = getIntermFeat
	 
		for i in range(num_D):
			netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat, use_dropout)
			if getIntermFeat:                                
				for j in range(n_layers+2):
					setattr(self, 'scale'+str(i)+'_layer'+str(j), getattr(netD, 'model'+str(j)))                                   
			else:
				setattr(self, 'layer'+str(i), netD.model)

		self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

	def singleD_forward(self, model, input):
		if self.getIntermFeat:
			result = [input]
			for i in range(len(model)):
				result.append(model[i](result[-1]))
			return result[1:]
		else:
			return [model(input)]

	def forward(self, input):        
		num_D = self.num_D
		result = []
		input_downsampled = input
		for i in range(num_D):
			if self.getIntermFeat:
				# print('number i: ',i, 'input: ', input_downsampled.shape)

				model = [getattr(self, 'scale'+str(num_D-1-i)+'_layer'+str(j)) for j in range(self.n_layers+2)]
			else:
				model = getattr(self, 'layer'+str(num_D-1-i))
			result.append(self.singleD_forward(model, input_downsampled))
			if i != (num_D-1):
				input_downsampled = self.downsample(input_downsampled)
		return result
		
# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
	def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=False, use_dropout=False):
		super(NLayerDiscriminator, self).__init__()
		self.getIntermFeat = getIntermFeat
		self.n_layers = n_layers

		kw = 4
		padw = int(np.ceil((kw-1.0)/2))
		sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

		nf = ndf
		for n in range(1, n_layers):
			nf_prev = nf
			nf = min(nf * 2, 512)
			if use_dropout:
				print('dropout for D')
				sequence += [[
				nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
				norm_layer(nf), nn.LeakyReLU(0.2, True),nn.Dropout(0.5)
				]]
			else:
				sequence += [[
					nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
					norm_layer(nf), nn.LeakyReLU(0.2, True)
				]]   

		nf_prev = nf
		nf = min(nf * 2, 512)
		if use_dropout:
			print('dropout for D')            
			sequence += [[
			nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
			norm_layer(nf),
			nn.LeakyReLU(0.2, True),nn.Dropout(0.5)
			]]
		else:
			sequence += [[
				nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
				norm_layer(nf),
				nn.LeakyReLU(0.2, True)
			]]

		sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

		if use_sigmoid:
			sequence += [[nn.Sigmoid()]]

		if getIntermFeat:
			for n in range(len(sequence)):
				setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
		else:
			sequence_stream = []
			for n in range(len(sequence)):
				sequence_stream += sequence[n]
			self.model = nn.Sequential(*sequence_stream)

	def forward(self, input):
		if self.getIntermFeat:
			res = [input]
			for n in range(self.n_layers+2):
				model = getattr(self, 'model'+str(n))
				res.append(model(res[-1]))
			return res[1:]
		else:
			return self.model(input)        

from torchvision import models
class Vgg19(torch.nn.Module):
	def __init__(self, requires_grad=False):
		super(Vgg19, self).__init__()
		vgg_pretrained_features = models.vgg19(pretrained=True).features
		self.slice1 = torch.nn.Sequential()
		self.slice2 = torch.nn.Sequential()
		self.slice3 = torch.nn.Sequential()
		self.slice4 = torch.nn.Sequential()
		self.slice5 = torch.nn.Sequential()
		for x in range(2):
			self.slice1.add_module(str(x), vgg_pretrained_features[x])
		for x in range(2, 7):
			self.slice2.add_module(str(x), vgg_pretrained_features[x])
		for x in range(7, 12):
			self.slice3.add_module(str(x), vgg_pretrained_features[x])
		for x in range(12, 21):
			self.slice4.add_module(str(x), vgg_pretrained_features[x])
		for x in range(21, 30):
			self.slice5.add_module(str(x), vgg_pretrained_features[x])
		if not requires_grad:
			for param in self.parameters():
				param.requires_grad = False

	def forward(self, X):
		h_relu1 = self.slice1(X)
		h_relu2 = self.slice2(h_relu1)        
		h_relu3 = self.slice3(h_relu2)        
		h_relu4 = self.slice4(h_relu3)        
		h_relu5 = self.slice5(h_relu4)                
		out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
		return out
