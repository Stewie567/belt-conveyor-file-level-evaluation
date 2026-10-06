import torch
from torch import nn

def block(a,b,k,activation=True):
    return nn.Sequential(nn.Conv1d(a,b,k,padding='same',bias=False),nn.BatchNorm1d(b),nn.ReLU() if activation else nn.Identity())
class Residual(nn.Module):
    def __init__(self,a,b):
        super().__init__();self.path=nn.Sequential(block(a,b,8),block(b,b,5),block(b,b,3,False));self.skip=block(a,b,1,False) if a!=b else nn.Identity()
    def forward(self,x):return torch.relu(self.path(x)+self.skip(x))
class Inception(nn.Module):
    def __init__(self,a):
        super().__init__();self.bottle=nn.Conv1d(a,32,1,bias=False) if a>1 else nn.Identity();b=32 if a>1 else a
        self.branches=nn.ModuleList([nn.Conv1d(b,32,k,padding='same',bias=False) for k in [10,20,40]])
        self.pool=nn.Sequential(nn.MaxPool1d(3,stride=1,padding=1),nn.Conv1d(a,32,1,bias=False));self.bn=nn.BatchNorm1d(128)
    def forward(self,x):
        b=self.bottle(x);return torch.relu(self.bn(torch.cat([f(b) for f in self.branches]+[self.pool(x)],1)))
class InceptionGroup(nn.Module):
    def __init__(self,a):
        super().__init__();self.path=nn.Sequential(Inception(a),Inception(128),Inception(128));self.skip=block(a,128,1,False)
    def forward(self,x):return torch.relu(self.path(x)+self.skip(x))
class Reverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,scale):ctx.scale=scale;return x.view_as(x)
    @staticmethod
    def backward(ctx,g):return -ctx.scale*g,None
class Network(nn.Module):
    def __init__(self,name,channels,domains=2):
        super().__init__()
        if name in ['FCN','SI-FCN']:self.features=nn.Sequential(block(channels,128,8),block(128,256,5),block(256,128,3))
        elif name=='ResNet':self.features=nn.Sequential(Residual(channels,64),Residual(64,128),Residual(128,128))
        elif name=='InceptionTime':self.features=nn.Sequential(InceptionGroup(channels),InceptionGroup(128))
        else:raise ValueError(name)
        self.classifier=nn.Linear(128,6)
        self.discriminator=nn.Sequential(nn.Linear(128,64),nn.ReLU(),nn.Linear(64,domains)) if name=='SI-FCN' else None
    def forward(self,x,lam=0):
        z=self.features(x).mean(-1);y=self.classifier(z)
        d=self.discriminator(Reverse.apply(z,lam)) if self.discriminator is not None else None
        return y,d,z
