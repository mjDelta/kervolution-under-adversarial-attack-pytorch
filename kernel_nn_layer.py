# -*- coding: utf-8 -*-
# @Author: Mengji Zhang
# @Date:   2019-05-18 21:12:00
# @Last Modified by:   WIN10
# @Last Modified time: 2019-05-19 21:14:17
# @E-mail: zmj_xy@sjtu.edu.cn

from torch import nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
class PolynomialKernelConv(nn.Conv2d):
	def __init__(self,in_channels,out_channels,kernel_size,stride=1,padding=0,dilation=1,groups=1,
		cp=0.5,dp=2,cp_require_grad=False,device="cuda"):
		super(PolynomialKernelConv,self).__init__(in_channels,out_channels,kernel_size,stride,padding,dilation,groups)
		if cp_require_grad:
			self.cp=Variable(torch.FloatTensor([cp]),requires_grad=cp_require_grad).to(device)
		else:
			self.cp=cp
		self.dp=dp
	def compute_outshape(self,h,w):
		kernel_h=self.kernel_size[0]+(self.kernel_size[0]-1)*(self.dilation[0]-1)
		kernel_w=self.kernel_size[1]+(self.kernel_size[1]-1)*(self.dilation[1]-1)

		out_h=(h+2*self.padding[0]-(kernel_h-1)-1)//self.stride[0]+1
		out_w=(w+2*self.padding[1]-(kernel_w-1)-1)//self.stride[1]+1
		return out_h,out_w
	def forward(self,x,require_bias=True):
		batch,_,height,width=x.shape
		out_h,out_w=self.compute_outshape(height,width)

		x_unfold=F.unfold(x,self.kernel_size,self.dilation,self.padding,self.stride)
		w=self.weight
		w=w.view(self.out_channels,-1)
		mul=x_unfold.transpose(1,2).matmul(w.t()).transpose(1,2)
		
		if require_bias:
			mul=mul.transpose(1,2)
			mul+=self.bias
			mul=mul.transpose(1,2)
		mul=(self.cp+mul)**self.dp
		kernel_out=F.fold(mul,(out_h,out_w),(1,1))
		return kernel_out

class GaussianKernelConv(nn.Conv2d):
	def __init__(self,in_channels,out_channels,kernel_size,stride=1,padding=0,dilation=1,groups=1,
		gamma=0.5,gamma_require_grad=False,device="cuda"):
		super(GaussianKernelConv,self).__init__(in_channels,out_channels,kernel_size,stride,padding,dilation,groups)
		if gamma_require_grad:
			self.gamma=Variable(torch.FloatTensor([gamma]),requires_grad=gamma_require_grad).to(device)
		else:
			self.gamma=gamma

	def compute_outshape(self,h,w):
		kernel_h=self.kernel_size[0]+(self.kernel_size[0]-1)*(self.dilation[0]-1)
		kernel_w=self.kernel_size[1]+(self.kernel_size[1]-1)*(self.dilation[1]-1)

		out_h=(h+2*self.padding[0]-(kernel_h-1)-1)//self.stride[0]+1
		out_w=(w+2*self.padding[1]-(kernel_w-1)-1)//self.stride[1]+1
		return out_h,out_w
	def forward(self,x):
		batch,_,height,width=x.shape
		out_h,out_w=self.compute_outshape(height,width)

		x_unfold=F.unfold(x,self.kernel_size,self.dilation,self.padding,self.stride)
		w=self.weight
		w=w.view(self.out_channels,-1)

		minus=x_unfold[:,None,:,:]-w[None,:,:,None]
		minus=minus**2
		minus=minus.sum(dim=2)
		
		minus=torch.exp(-self.gamma*minus)

		kernel_out=F.fold(minus,(out_h,out_w),(1,1))
		return kernel_out

class LpNormKernelConv(nn.Conv2d):
	def __init__(self,in_channels,out_channels,kernel_size,p,stride=1,padding=0,dilation=1,groups=1,device="cuda"):
		super(LpNormKernelConv,self).__init__(in_channels,out_channels,kernel_size,stride,padding,dilation,groups)
		self.p=p

	def compute_outshape(self,h,w):
		kernel_h=self.kernel_size[0]+(self.kernel_size[0]-1)*(self.dilation[0]-1)
		kernel_w=self.kernel_size[1]+(self.kernel_size[1]-1)*(self.dilation[1]-1)

		out_h=(h+2*self.padding[0]-(kernel_h-1)-1)//self.stride[0]+1
		out_w=(w+2*self.padding[1]-(kernel_w-1)-1)//self.stride[1]+1
		return out_h,out_w
	def forward(self,x):
		batch,_,height,width=x.shape
		out_h,out_w=self.compute_outshape(height,width)

		x_unfold=F.unfold(x,self.kernel_size,self.dilation,self.padding,self.stride)
		w=self.weight
		w=w.view(self.out_channels,-1)

		minus=x_unfold[:,None,:,:]-w[None,:,:,None]
		minus=minus**self.p

		minus=minus.sum(dim=2)
		
		kernel_out=F.fold(minus,(out_h,out_w),(1,1))
		return kernel_out
if __name__=="__main__":
	from torchvision import datasets,transforms
	device=torch.device("cuda" if torch.cuda.is_available() else "cpu")


	test_loader=torch.utils.data.DataLoader(
    datasets.MNIST("data",train=False,download=True,transform=transforms.Compose([
                transforms.ToTensor(),
            ])),batch_size=128,shuffle=False)
	conv1=nn.Conv2d(1,10,5)
	kernel_conv=GaussianKernelConv(10,20,5,gamma=1)
	conv2=nn.Conv2d(10,20,5)
	conv2.weight=kernel_conv.weight
	conv2.bias=kernel_conv.bias
	conv1.to(device)
	kernel_conv.to(device)
	for d,t in test_loader:
		d=d.to(device)
		tmp=conv1(d)
		print("input: ",tmp.shape)
		output=kernel_conv(tmp)
		print("output:",output.shape)
		output2=conv2(tmp)
		print((output2-output).abs().max())
		break




