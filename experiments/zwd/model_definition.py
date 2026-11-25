"""DA-TransUNet architecture for paired and stagewise experiments."""
import numpy as np
import torch
from torch import nn
import math
from torch.nn import Module,BatchNorm1d,BatchNorm2d,LayerNorm,Conv2d,MaxPool2d,AvgPool2d,Dropout,LeakyReLU,ReLU,PReLU,Sigmoid,Tanh,ELU,Softmax,Linear,Flatten,ZeroPad2d,Upsample,TransformerEncoderLayer,TransformerEncoder

def build_model(input_channels, device='cpu'):
    timestep=6; downnum=4; covnum=2; layer_time=1; baselayer=16
    cov_kernelsize=3; cov_strides=1; pool_method='maxpooling'; pool_kernel_size=2; pool_strides=2
    ifnormalization='no';normalization_method=None;trans_num=1;num_heads=1;dim_feedforward=128
    trans_dropout_rate=0.;activate='tanh';nlp_activate='relu';da_attention_order='space_first'
    if_bottom_attention_only='yes';if_resnet='yes';if_last_act='no';task_mode='regression'
    devices=torch.device(device)
    class TimeDistributed(nn.Module):
        def __init__(self, module, batch_first=False):
            super(TimeDistributed, self).__init__()
            self.module = module
            self.batch_first = batch_first
        def forward(self, x): 
            if self.batch_first:
                batch_size, seq_len = x.shape[0], x.shape[1]
            else:
                seq_len, batch_size = x.shape[0], x.shape[1]
            x_reshaped = x.view(batch_size * seq_len, *x.shape[2:])
            y = self.module(x_reshaped)
            if self.batch_first:
                y = y.view(batch_size, seq_len, *y.shape[1:])
            else:
                y = y.view(seq_len, batch_size, *y.shape[1:])

            return y
    
    class Model(nn.Module):
        def __init__(self,trainx,trainy,timestep,downnum,covnum,layer_time,baselayer,cov_kernelsize,cov_strides,pool_method,pool_kernel_size,pool_strides,ifnormalization,normalization_method,trans_num,num_heads,dim_feedforward,trans_dropout_rate,activate,nlp_activate,da_attention_order,if_bottom_attention_only,if_resnet):
            super(Model,self).__init__()
            exec('from torch.nn import Module,BatchNorm1d,BatchNorm2d,LayerNorm,Conv2d,MaxPool2d,AvgPool2d,Dropout,LeakyReLU,ReLU,PReLU,Sigmoid,Tanh,ELU,Softmax,Linear,Flatten,ZeroPad2d,Upsample,TransformerEncoderLayer,TransformerEncoder', globals(), self.__dict__)
            exec('import numpy as np', globals(), self.__dict__)
            exec('import math', globals(), self.__dict__)
            exec('import torch', globals(), self.__dict__)
            exec('from torch import nn', globals(), self.__dict__)
            self.timestep=timestep
            self.downnum=downnum
            self.covnum=covnum
            self.layer_time=layer_time
            self.baselayer=baselayer
            self.cov_kernelsize=cov_kernelsize
            self.cov_strides=cov_strides
            self.pool_method=pool_method
            self.pool_kernel_size=pool_kernel_size
            self.pool_strides=pool_strides
            self.ifnormalization=ifnormalization
            self.normalization_method=normalization_method
            self.trans_num=trans_num
            self.num_heads=num_heads
            self.dim_feedforward=dim_feedforward
            self.trans_dropout_rate=trans_dropout_rate
            self.activate=activate
            self.nlp_activate=nlp_activate
            self.da_attention_order = da_attention_order
            if self.da_attention_order not in ['time_first', 'space_first']:
                raise ValueError("da_attention_order must be 'time_first' or 'space_first'")
            self.if_bottom_attention_only = if_bottom_attention_only
            self.if_resnet = if_resnet
            
            if self.if_bottom_attention_only not in ['yes', 'no']:
                raise ValueError("if_bottom_attention_only must be 'yes' or 'no'")
            
            if self.if_resnet not in ['yes', 'no']:
                raise ValueError("if_resnet must be 'yes' or 'no'")
            self.trainx=trainx
            self.trainy=trainy
            self.hight0=self.trainx.shape[3]
            self.weight0=self.trainx.shape[4]
            self.upsize=pool_strides
            self.__dict__['self']=self
            self.__dict__['TimeDistributed']=TimeDistributed
            self.k=0
            self.total_covnum=0
            def align_residual(self, shortcut, target):
                """
                shortcut / target:
                    encoder: (B, T, C, H, W)
                    decoder: (B, C, H, W)
                将 shortcut 的 H/W 对齐到 target。
                """
                sh, sw = shortcut.shape[-2], shortcut.shape[-1]
                th, tw = target.shape[-2], target.shape[-1]
    
                # crop height
                if sh > th:
                    dh = sh - th
                    shortcut = shortcut[..., dh // 2: dh // 2 + th, :]
                # crop width
                if sw > tw:
                    dw = sw - tw
                    shortcut = shortcut[..., :, dw // 2: dw // 2 + tw]
    
                sh, sw = shortcut.shape[-2], shortcut.shape[-1]
    
                # pad height / width
                pad_h = th - sh
                pad_w = tw - sw
    
                if pad_h > 0 or pad_w > 0:
                    pad_top = pad_h // 2
                    pad_bottom = pad_h - pad_top
                    pad_left = pad_w // 2
                    pad_right = pad_w - pad_left
                    shortcut = torch.nn.functional.pad(
                        shortcut,
                        (pad_left, pad_right, pad_top, pad_bottom)
                    )
    
                return shortcut
            self.align_residual = lambda shortcut, target: align_residual(self, shortcut, target)
            
            W=trainx.shape[3]
            H=trainx.shape[4]
            for i in range(int(self.downnum-1)):
                W=math.floor(((W-self.pool_kernel_size)/self.pool_strides)+1)
                H=math.floor(((H-self.pool_kernel_size)/self.pool_strides)+1)
            if W<1 or H<1:
                print('下采样次数过多,导致最下层图形长度/宽度小于1!')
                return model,predicty,testy,r,p
            if self.pool_strides!=self.pool_kernel_size:
                print('池化步长和池化核不等,将导致下采样前后通道数相除不为整数！')
                return model,predicty,testy,r,p
            for i in range(int(self.downnum)):
                for j in range(int(self.covnum)):
                    if self.total_covnum==0:
                        exec('self.pad'+str(i+1)+'_'+str(j+1)+'=TimeDistributed(ZeroPad2d(padding=(math.floor(math.floor((self.trainx.shape[3]-(self.trainx.shape[3]/self.cov_strides)))/2.0),math.ceil(math.floor((self.trainx.shape[3]-(self.trainx.shape[3]/self.cov_strides)))/2.0),math.floor(math.floor((self.trainx.shape[2]-(self.trainx.shape[2]/self.cov_strides)))/2.0),math.ceil(math.floor((self.trainx.shape[2]-(self.trainx.shape[2]/self.cov_strides)))/2.0))), batch_first=True)', globals(), self.__dict__)
                        exec('self.conv'+str(i+1)+'_'+str(j+1)+'=TimeDistributed(Conv2d(self.trainx.shape[2],self.baselayer,(self.cov_kernelsize,self.cov_kernelsize),stride=self.cov_strides,padding="same"), batch_first=True)', globals(), self.__dict__)
                        if self.if_resnet == 'yes':
                            res_in_channels = int(self.trainx.shape[2])
                            res_out_channels = int(self.baselayer)
                            if res_in_channels != res_out_channels or self.cov_strides != 1:
                                setattr(
                                    self,
                                    'res' + str(i+1) + '_' + str(j+1),
                                    TimeDistributed(
                                        Conv2d(res_in_channels, res_out_channels, (1, 1), stride=self.cov_strides, padding=0),
                                        batch_first=True
                                    )
                                )
                            else:
                                setattr(self, 'res' + str(i+1) + '_' + str(j+1), None)
                        if self.activate=='elu':
                            exec('self.act'+str(i+1)+'_'+str(j+1)+'=ELU()', globals(), self.__dict__)
                        elif self.activate=='leakyrelu':
                            exec('self.act'+str(i+1)+'_'+str(j+1)+'=LeakyReLU()', globals(), self.__dict__)
                        elif self.activate=='prelu':
                            exec('self.act'+str(i+1)+'_'+str(j+1)+'=PReLU()', globals(), self.__dict__)
                        elif self.activate=='relu':
                            exec('self.act'+str(i+1)+'_'+str(j+1)+'=ReLU()', globals(), self.__dict__)
                        elif self.activate=='sigmoid':
                            exec('self.act'+str(i+1)+'_'+str(j+1)+'=Sigmoid()', globals(), self.__dict__)
                        elif self.activate=='tanh':
                            exec('self.act'+str(i+1)+'_'+str(j+1)+'=Tanh()', globals(), self.__dict__)
                        elif self.activate=='softmax':
                            exec('self.act'+str(i+1)+'_'+str(j+1)+'=Softmax()', globals(), self.__dict__)
                        exec('in_channels=self.baselayer', globals(), self.__dict__)
                    else:
                        exec('self.pad'+str(i+1)+'_'+str(j+1)+'=TimeDistributed(ZeroPad2d(padding=(math.floor(math.floor((self.weight'+str(i)+'-(self.weight'+str(i)+'/self.cov_strides)))/2.0),math.ceil(math.floor((self.weight'+str(i)+'-(self.weight'+str(i)+'/self.cov_strides)))/2.0),math.floor(math.floor((self.hight'+str(i)+'-(self.hight'+str(i)+'/self.cov_strides)))/2.0),math.ceil(math.floor((self.hight'+str(i)+'-(self.hight'+str(i)+'/self.cov_strides)))/2.0))), batch_first=True)', globals(), self.__dict__)
                        exec('self.conv'+str(i+1)+'_'+str(j+1)+'=TimeDistributed(Conv2d(self.in_channels,self.baselayer,(self.cov_kernelsize,self.cov_kernelsize),stride=self.cov_strides,padding="same"), batch_first=True)', globals(), self.__dict__)
                        if self.if_resnet == 'yes':
                            res_in_channels = int(self.in_channels)
                            res_out_channels = int(self.baselayer)
                            if res_in_channels != res_out_channels or self.cov_strides != 1:
                                setattr(
                                    self,
                                    'res' + str(i+1) + '_' + str(j+1),
                                    TimeDistributed(
                                        Conv2d(res_in_channels, res_out_channels, (1, 1), stride=self.cov_strides, padding=0),
                                        batch_first=True
                                    )
                                )
                            else:
                                setattr(self, 'res' + str(i+1) + '_' + str(j+1), None)
                        if self.activate=='elu':
                            exec('self.act'+str(i+1)+'_'+str(j+1)+'=ELU()', globals(), self.__dict__)
                        elif self.activate=='leakyrelu':
                            exec('self.act'+str(i+1)+'_'+str(j+1)+'=LeakyReLU()', globals(), self.__dict__)
                        elif self.activate=='prelu':
                            exec('self.act'+str(i+1)+'_'+str(j+1)+'=PReLU()', globals(), self.__dict__)
                        elif self.activate=='relu':
                            exec('self.act'+str(i+1)+'_'+str(j+1)+'=ReLU()', globals(), self.__dict__)
                        elif self.activate=='sigmoid':
                            exec('self.act'+str(i+1)+'_'+str(j+1)+'=Sigmoid()', globals(), self.__dict__)
                        elif self.activate=='tanh':
                            exec('self.act'+str(i+1)+'_'+str(j+1)+'=Tanh()', globals(), self.__dict__)
                        elif self.activate=='softmax':
                            exec('self.act'+str(i+1)+'_'+str(j+1)+'=Softmax()', globals(), self.__dict__)
                        exec('in_channels=self.baselayer', globals(), self.__dict__)
                    self.total_covnum=self.total_covnum+1
                    if self.ifnormalization=='yes':
                        if self.normalization_method=='batchnormalization':
                            exec('self.norm'+str(i+1)+'_'+str(j+1)+'=TimeDistributed(BatchNorm2d(in_channels), batch_first=True)', globals(), self.__dict__)
                        elif self.normalization_method=='layernormalization':
                            exec('self.norm'+str(i+1)+'_'+str(j+1)+'=LayerNorm(in_channels)', globals(), self.__dict__)
                    da_layer_id = i + 1
                    da_channels = int(self.in_channels)
                    da_h = int(getattr(self, 'hight' + str(i)))
                    da_w = int(getattr(self, 'weight' + str(i)))
                    da_time_dim = da_channels * da_h * da_w
                if self.if_bottom_attention_only == 'no' or i == int(self.downnum) - 1:

                    da_layer_id = i + 1

                    # current encoder level final feature shape:
                    # (B, T, C, H, W)
                    #
                    # C is current in_channels after the last conv of this level.
                    # H/W are hight{i}/weight{i} before pooling.
                    da_channels = int(self.in_channels)
                    da_h = int(getattr(self, 'hight' + str(i)))
                    da_w = int(getattr(self, 'weight' + str(i)))

                    # time transformer:
                    # token dimension = T
                    # feature dimension = C * H * W
                    da_time_dim = da_channels * da_h * da_w

                    if da_time_dim % self.num_heads != 0:
                        raise ValueError(
                            "时间 Transformer 的 d_model=C*H*W 必须能被 num_heads 整除。"
                            f" 当前层={da_layer_id}, C={da_channels}, H={da_h}, W={da_w}, "
                            f"C*H*W={da_time_dim}, num_heads={self.num_heads}"
                        )

                    # space transformer:
                    # token dimension = H * W
                    # feature dimension = C
                    if da_channels % self.num_heads != 0:
                        raise ValueError(
                            "空间 Transformer 的 d_model=C 必须能被 num_heads 整除。"
                            f" 当前层={da_layer_id}, C={da_channels}, num_heads={self.num_heads}"
                        )

                    setattr(
                        self,
                        'time_transformer' + str(da_layer_id),
                        TransformerEncoder(
                            TransformerEncoderLayer(
                                da_time_dim,
                                self.num_heads,
                                dim_feedforward=self.dim_feedforward,
                                dropout=self.trans_dropout_rate,
                                activation=self.nlp_activate,
                                batch_first=True
                            ),
                            self.trans_num
                        )
                    )

                    setattr(
                        self,
                        'space_transformer' + str(da_layer_id),
                        TransformerEncoder(
                            TransformerEncoderLayer(
                                da_channels,
                                self.num_heads,
                                dim_feedforward=self.dim_feedforward,
                                dropout=self.trans_dropout_rate,
                                activation=self.nlp_activate,
                                batch_first=True
                            ),
                            self.trans_num
                        )
                    )

                    setattr(
                        self,
                        'time_embed' + str(da_layer_id),
                        nn.Parameter(torch.zeros(1, self.timestep, da_time_dim))
                    )
                    nn.init.trunc_normal_(
                        getattr(self, 'time_embed' + str(da_layer_id)),
                        std=0.02
                    )

                    setattr(
                        self,
                        'space_embed' + str(da_layer_id),
                        nn.Parameter(torch.zeros(1, da_h * da_w, da_channels))
                    )
                    nn.init.trunc_normal_(
                        getattr(self, 'space_embed' + str(da_layer_id)),
                        std=0.02
                    )

                    # learnable timestep aggregation:
                    # input:  (B, C*H*W, T)
                    # output: (B, C*H*W, 1)
                    setattr(
                        self,
                        'time_pool' + str(da_layer_id),
                        Linear(self.timestep, 1)
                    )
                self.k=self.k+1
                if i!=(self.downnum-1):
                    self.baselayer=int(self.baselayer*self.layer_time)
                if self.pool_method=='maxpooling' and i!=(self.downnum-1):
                    exec('self.hight'+str(i+1)+'=1+math.floor((self.hight'+str(i)+'-self.pool_kernel_size)/self.pool_strides)', globals(), self.__dict__)
                    exec('self.weight'+str(i+1)+'=1+math.floor((self.weight'+str(i)+'-self.pool_kernel_size)/self.pool_strides)', globals(), self.__dict__)
                    exec('self.pool'+str(i+1)+'=TimeDistributed(MaxPool2d((self.pool_kernel_size,self.pool_kernel_size),stride=self.pool_strides), batch_first=True)', globals(), self.__dict__)
                elif self.pool_method=='avepooling' and i!=(self.downnum-1):
                    exec('self.hight'+str(i+1)+'=1+math.floor((self.hight'+str(i)+'-self.pool_kernel_size)/self.pool_strides)', globals(), self.__dict__)
                    exec('self.weight'+str(i+1)+'=1+math.floor((self.weight'+str(i)+'-self.pool_kernel_size)/self.pool_strides)', globals(), self.__dict__)
                    exec('self.pool'+str(i+1)+'=TimeDistributed(AvgPool2d((self.pool_kernel_size,self.pool_kernel_size),stride=self.pool_strides), batch_first=True)', globals(), self.__dict__)
            self.l=self.k-1
            self.k=self.k+1
            for m in range(self.downnum,2*self.downnum-1):
                self.baselayer=int(self.baselayer/self.layer_time)
                if m==self.downnum:
                    exec('self.pad'+str(m+1)+'=ZeroPad2d(padding=(math.floor(math.floor((self.weight'+str(m-1)+'-(self.weight'+str(m-1)+'/self.cov_strides)))/2.0),math.ceil(math.floor((self.weight'+str(m-1)+'-(self.weight'+str(m-1)+'/self.cov_strides)))/2.0),math.floor(math.floor((self.hight'+str(m-1)+'-(self.hight'+str(m-1)+'/self.cov_strides)))/2.0),math.ceil(math.floor((self.hight'+str(m-1)+'-(self.hight'+str(m-1)+'/self.cov_strides)))/2.0)))', globals(), self.__dict__)
                else:
                    exec('self.pad'+str(m+1)+'=ZeroPad2d(padding=(math.floor(math.floor((self.weight'+str(m)+'-(self.weight'+str(m)+'/self.cov_strides)))/2.0),math.ceil(math.floor((self.weight'+str(m)+'-(self.weight'+str(m)+'/self.cov_strides)))/2.0),math.floor(math.floor((self.hight'+str(m)+'-(self.hight'+str(m)+'/self.cov_strides)))/2.0),math.ceil(math.floor((self.hight'+str(m)+'-(self.hight'+str(m)+'/self.cov_strides)))/2.0)))', globals(), self.__dict__)
                exec('self.conv'+str(m+1)+'=Conv2d(self.in_channels,self.baselayer,(self.cov_kernelsize,self.cov_kernelsize),stride=self.cov_strides,padding="same")', globals(), self.__dict__)
                if self.if_resnet == 'yes':
                    res_in_channels = int(self.in_channels)
                    res_out_channels = int(self.baselayer)
                    if res_in_channels != res_out_channels or self.cov_strides != 1:
                        setattr(
                            self,
                            'res' + str(m+1),
                            Conv2d(res_in_channels, res_out_channels, (1, 1), stride=self.cov_strides, padding=0)
                        )
                    else:
                        setattr(self, 'res' + str(m+1), None)
                if self.activate=='elu':
                    exec('self.act'+str(m+1)+'=ELU()', globals(), self.__dict__)
                elif self.activate=='leakyrelu':
                    exec('self.act'+str(m+1)+'=LeakyReLU()', globals(), self.__dict__)
                elif self.activate=='prelu':
                    exec('self.act'+str(m+1)+'=PReLU()', globals(), self.__dict__)
                elif self.activate=='relu':
                    exec('self.act'+str(m+1)+'=ReLU()', globals(), self.__dict__)
                elif self.activate=='sigmoid':
                    exec('self.act'+str(m+1)+'=Sigmoid()', globals(), self.__dict__)
                elif self.activate=='tanh':
                    exec('self.act'+str(m+1)+'=Tanh()', globals(), self.__dict__)
                elif self.activate=='softmax':
                    exec('self.act'+str(m+1)+'=Softmax()', globals(), self.__dict__)
                exec('in_channels=self.baselayer', globals(), self.__dict__)
                if self.ifnormalization=='yes':
                    if self.normalization_method=='batchnormalization':
                        exec('self.norm'+str(m+1)+'=BatchNorm2d(in_channels)', globals(), self.__dict__)
                    elif self.normalization_method=='layernormalization':
                        exec('self.norm'+str(m+1)+'=LayerNorm(in_channels)', globals(), self.__dict__)
                exec('self.hight'+str(m+1)+'=self.hight'+str(self.l)+'*self.upsize', globals(), self.__dict__)
                exec('self.weight'+str(m+1)+'=self.weight'+str(self.l)+'*self.upsize', globals(), self.__dict__)
                exec('self.conc'+str(m+1)+'=torch.concatenate', globals(), self.__dict__)   
                exec('self.uppad'+str(m+1)+'=ZeroPad2d(padding=(math.floor((self.weight'+str(self.l-1)+'-self.weight'+str(self.k)+')/2.0),math.ceil((self.weight'+str(self.l-1)+'-self.weight'+str(self.k)+')/2.0),math.floor((self.hight'+str(self.l-1)+'-self.hight'+str(self.k)+')/2.0),math.ceil((self.hight'+str(self.l-1)+'-self.hight'+str(self.k)+')/2.0)))', globals(), self.__dict__)
                exec('self.up'+str(m+1)+'=Upsample(scale_factor=self.upsize, mode="bilinear")', globals(), self.__dict__)
                exec('in_channels=in_channels*2', globals(), self.__dict__)
                for n in range(int(self.covnum)):
                    exec('self.pad'+str(m+1)+'_'+str(n+1)+'=ZeroPad2d(padding=(math.floor(math.floor((self.weight'+str(m+1)+'-(self.weight'+str(m+1)+'/self.cov_strides)))/2.0),math.ceil(math.floor((self.weight'+str(m+1)+'-(self.weight'+str(m+1)+'/self.cov_strides)))/2.0),math.floor(math.floor((self.hight'+str(m+1)+'-(self.hight'+str(m+1)+'/self.cov_strides)))/2.0),math.ceil(math.floor((self.hight'+str(m+1)+'-(self.hight'+str(m+1)+'/self.cov_strides)))/2.0)))', globals(), self.__dict__)
                    exec('self.conv'+str(m+1)+'_'+str(n+1)+'=Conv2d(self.in_channels,self.baselayer,(self.cov_kernelsize,self.cov_kernelsize),stride=self.cov_strides,padding="same")', globals(), self.__dict__)
                    if self.if_resnet == 'yes':
                        res_in_channels = int(self.in_channels)
                        res_out_channels = int(self.baselayer)
                        if res_in_channels != res_out_channels or self.cov_strides != 1:
                            setattr(
                                self,
                                'res' + str(m+1) + '_' + str(n+1),
                                Conv2d(res_in_channels, res_out_channels, (1, 1), stride=self.cov_strides, padding=0)
                            )
                        else:
                            setattr(self, 'res' + str(m+1) + '_' + str(n+1), None)
                    if self.activate=='elu':
                        exec('self.act'+str(m+1)+'_'+str(n+1)+'=ELU()', globals(), self.__dict__)
                    elif self.activate=='leakyrelu':
                        exec('self.act'+str(m+1)+'_'+str(n+1)+'=LeakyReLU()', globals(), self.__dict__)
                    elif self.activate=='prelu':
                        exec('self.act'+str(m+1)+'_'+str(n+1)+'=PReLU()', globals(), self.__dict__)
                    elif self.activate=='relu':
                        exec('self.act'+str(m+1)+'_'+str(n+1)+'=ReLU()', globals(), self.__dict__)
                    elif self.activate=='sigmoid':
                        exec('self.act'+str(m+1)+'_'+str(n+1)+'=Sigmoid()', globals(), self.__dict__)
                    elif self.activate=='tanh':
                        exec('self.act'+str(m+1)+'_'+str(n+1)+'=Tanh()', globals(), self.__dict__)
                    elif self.activate=='softmax':
                        exec('self.act'+str(m+1)+'_'+str(n+1)+'=Softmax()', globals(), self.__dict__)
                    exec('in_channels=self.baselayer', globals(), self.__dict__)
                    if self.ifnormalization=='yes':
                        if self.normalization_method=='batchnormalization':
                            exec('self.norm'+str(m+1)+'_'+str(n+1)+'=BatchNorm2d(in_channels)', globals(), self.__dict__)
                        elif self.normalization_method=='layernormalization':
                            exec('self.norm'+str(m+1)+'_'+str(n+1)+'=LayerNorm(in_channels)', globals(), self.__dict__)
                self.k=self.k+1
                self.l=self.l-1
            if task_mode=='multi_classify':
                exec('self.convlast=Conv2d(self.in_channels,np.max(trainy)+1,(1,1),stride=1,padding="same")', globals(), self.__dict__)
            else:
                exec('self.convlast=Conv2d(self.in_channels,self.trainy.shape[1],(1,1),stride=1,padding="same")', globals(), self.__dict__)
            if if_last_act!='no':
                if if_last_act=='elu':
                    exec('self.actlast=ELU()', globals(), self.__dict__)
                elif if_last_act=='leakyrelu':
                    exec('self.actlast=LeakyReLU()', globals(), self.__dict__)
                elif if_last_act=='prelu':
                    exec('self.actlast=PReLU()', globals(), self.__dict__)
                elif if_last_act=='relu':
                    exec('self.actlast=ReLU()', globals(), self.__dict__)
                elif if_last_act=='sigmoid':
                    exec('self.actlast=Sigmoid()', globals(), self.__dict__)
                elif if_last_act=='tanh':
                    exec('self.actlast=Tanh()', globals(), self.__dict__)
                elif if_last_act=='softmax':
                    exec('self.actlast=Softmax()', globals(), self.__dict__)
        def da_time_attention(self, x, layer_id):
            """
            x: (B, T, C, H, W)
            时间 Transformer:
            token 维度 = T
            feature 维度 = C*H*W
            输出仍然是 (B, T, C, H, W)
            """
            B, T, C, H, W = x.shape
            x_time = x.reshape(B, T, C * H * W)
            x_time = x_time + getattr(self, 'time_embed' + str(layer_id))
            x_time = getattr(self, 'time_transformer' + str(layer_id))(x_time)
            x_time = x_time.reshape(B, T, C, H, W)
            return x_time

        def da_space_attention(self, x, layer_id):
            """
            x: (B, T, C, H, W)
            空间 Transformer:
            对每个 timestep 独立做空间 self-attention
            token 维度 = H*W
            feature 维度 = C
            输出仍然是 (B, T, C, H, W)
            """
            B, T, C, H, W = x.shape
            x_space = x.permute(0, 1, 3, 4, 2).contiguous()
            x_space = x_space.reshape(B * T, H * W, C)
            x_space = x_space + getattr(self, 'space_embed' + str(layer_id))
            x_space = getattr(self, 'space_transformer' + str(layer_id))(x_space)
            x_space = x_space.reshape(B, T, H, W, C)
            x_space = x_space.permute(0, 1, 4, 2, 3).contiguous()
            return x_space

        def da_time_pooling(self, x, layer_id):
            """
            x: (B, T, C, H, W)
            用可学习 Linear(T -> 1) 聚合 timestep 维度
            输出: (B, C, H, W)
            """
            B, T, C, H, W = x.shape
            x_pool = x.reshape(B, T, C * H * W)
            x_pool = x_pool.permute(0, 2, 1).contiguous()
            x_pool = getattr(self, 'time_pool' + str(layer_id))(x_pool)
            x_pool = x_pool.squeeze(-1)
            x_pool = x_pool.reshape(B, C, H, W)
            return x_pool

        def dual_attention_block(self, x, layer_id):
            """
            x: (B, T, C, H, W)
            输出: (B, C, H, W)
            """
            if self.da_attention_order == 'time_first':
                x = self.da_time_attention(x, layer_id)
                x = self.da_space_attention(x, layer_id)
            elif self.da_attention_order == 'space_first':
                x = self.da_space_attention(x, layer_id)
                x = self.da_time_attention(x, layer_id)
            else:
                raise ValueError("da_attention_order must be 'time_first' or 'space_first'")

            x = self.da_time_pooling(x, layer_id)
            return x
        def forward(self, x):
            self.__dict__['x']=x
            self.k=0

            # Encoder
            for i in range(int(self.downnum)):
                for j in range(int(self.covnum)):
                    if i==0 and j==0:
                        exec('model_res_input'+str(i+1)+'_'+str(j+1)+'=x', globals(), self.__dict__)
                    else:
                        if j==0:
                            exec('model_res_input'+str(i+1)+'_'+str(j+1)+'=model_pool'+str(i), globals(), self.__dict__)
                        else:
                            if self.ifnormalization=='yes':
                                exec('model_res_input'+str(i+1)+'_'+str(j+1)+'=model_norm'+str(i+1)+'_'+str(j), globals(), self.__dict__)
                            else:
                                exec('model_res_input'+str(i+1)+'_'+str(j+1)+'=model_act'+str(i+1)+'_'+str(j), globals(), self.__dict__)

                    exec('model_conv'+str(i+1)+'_'+str(j+1)+'=self.conv'+str(i+1)+'_'+str(j+1)+'(model_res_input'+str(i+1)+'_'+str(j+1)+')', globals(), self.__dict__)
                    exec('model_pad'+str(i+1)+'_'+str(j+1)+'=self.pad'+str(i+1)+'_'+str(j+1)+'(model_conv'+str(i+1)+'_'+str(j+1)+')', globals(), self.__dict__)
                    exec('model_act'+str(i+1)+'_'+str(j+1)+'=self.act'+str(i+1)+'_'+str(j+1)+'(model_pad'+str(i+1)+'_'+str(j+1)+')', globals(), self.__dict__)
                    if self.ifnormalization=='yes':
                        exec('model_norm'+str(i+1)+'_'+str(j+1)+'=self.norm'+str(i+1)+'_'+str(j+1)+'(model_act'+str(i+1)+'_'+str(j+1)+')', globals(), self.__dict__)

                    # Encoder residual shortcut
                    if self.if_resnet == 'yes':
                        res_layer = getattr(self, 'res' + str(i+1) + '_' + str(j+1))
                        self.__dict__['res_layer'] = res_layer
                        if res_layer is not None:
                            exec('model_res_shortcut'+str(i+1)+'_'+str(j+1)+'=res_layer(model_res_input'+str(i+1)+'_'+str(j+1)+')', globals(), self.__dict__)
                        else:
                            exec('model_res_shortcut'+str(i+1)+'_'+str(j+1)+'=model_res_input'+str(i+1)+'_'+str(j+1), globals(), self.__dict__)

                        if self.ifnormalization=='yes':
                            exec('model_res_shortcut'+str(i+1)+'_'+str(j+1)+'=self.align_residual(model_res_shortcut'+str(i+1)+'_'+str(j+1)+', model_norm'+str(i+1)+'_'+str(j+1)+')', globals(), self.__dict__)
                            exec('model_norm'+str(i+1)+'_'+str(j+1)+'=model_norm'+str(i+1)+'_'+str(j+1)+' + model_res_shortcut'+str(i+1)+'_'+str(j+1), globals(), self.__dict__)
                        else:
                            exec('model_res_shortcut'+str(i+1)+'_'+str(j+1)+'=self.align_residual(model_res_shortcut'+str(i+1)+'_'+str(j+1)+', model_act'+str(i+1)+'_'+str(j+1)+')', globals(), self.__dict__)
                            exec('model_act'+str(i+1)+'_'+str(j+1)+'=model_act'+str(i+1)+'_'+str(j+1)+' + model_res_shortcut'+str(i+1)+'_'+str(j+1), globals(), self.__dict__)

                # DA skip feature
                #     only bottleneck uses dual_attention_block;
                #     shallow skips use the last timestep.
                #     every skip uses dual_attention_block.
                if self.if_bottom_attention_only == 'yes' and i != int(self.downnum) - 1:
                    if self.ifnormalization=='yes':
                        exec('model_skip'+str(i+1)+'=model_norm'+str(i+1)+'_'+str(j+1)+'[:,-1,:,:,:]', globals(), self.__dict__)
                    else:
                        exec('model_skip'+str(i+1)+'=model_act'+str(i+1)+'_'+str(j+1)+'[:,-1,:,:,:]', globals(), self.__dict__)
                else:
                    if self.ifnormalization=='yes':
                        exec('model_skip'+str(i+1)+'=self.dual_attention_block(model_norm'+str(i+1)+'_'+str(j+1)+','+str(i+1)+')', globals(), self.__dict__)
                    else:
                        exec('model_skip'+str(i+1)+'=self.dual_attention_block(model_act'+str(i+1)+'_'+str(j+1)+','+str(i+1)+')', globals(), self.__dict__)

                self.k=self.k+1

                if i!=(self.downnum-1):
                    if self.ifnormalization=='yes':
                        exec('model_pool'+str(i+1)+'=self.pool'+str(i+1)+'(model_norm'+str(i+1)+'_'+str(j+1)+')', globals(), self.__dict__)
                    else:
                        exec('model_pool'+str(i+1)+'=self.pool'+str(i+1)+'(model_act'+str(i+1)+'_'+str(j+1)+')', globals(), self.__dict__)

            self.l=self.k-1
            self.k=self.k+1

            # bottom feature after dual attention and learnable timestep pooling
            exec('model_transformer'+str(i+1)+'=model_skip'+str(i+1), globals(), self.__dict__)

            # Decoder
            for m in range(self.downnum,2*self.downnum-1):
                # Decoder main conv input
                if m==self.downnum:
                    exec('model_res_input'+str(m+1)+'=model_transformer'+str(i+1), globals(), self.__dict__)
                else:
                    if self.ifnormalization=='yes':
                        exec('model_res_input'+str(m+1)+'=model_norm'+str(m)+'_'+str(n+1), globals(), self.__dict__)
                    else:
                        exec('model_res_input'+str(m+1)+'=model_act'+str(m)+'_'+str(n+1), globals(), self.__dict__)

                exec('model_conv'+str(m+1)+'=self.conv'+str(m+1)+'(model_res_input'+str(m+1)+')', globals(), self.__dict__)
                exec('model_pad'+str(m+1)+'=self.pad'+str(m+1)+'(model_conv'+str(m+1)+')', globals(), self.__dict__)
                exec('model_act'+str(m+1)+'=self.act'+str(m+1)+'(model_pad'+str(m+1)+')', globals(), self.__dict__)
                if self.ifnormalization=='yes':
                    exec('model_norm'+str(m+1)+'=self.norm'+str(m+1)+'(model_act'+str(m+1)+')', globals(), self.__dict__)

                # Decoder main residual shortcut
                if self.if_resnet == 'yes':
                    res_layer = getattr(self, 'res' + str(m+1))
                    self.__dict__['res_layer'] = res_layer
                    if res_layer is not None:
                        exec('model_res_shortcut'+str(m+1)+'=res_layer(model_res_input'+str(m+1)+')', globals(), self.__dict__)
                    else:
                        exec('model_res_shortcut'+str(m+1)+'=model_res_input'+str(m+1), globals(), self.__dict__)

                    if self.ifnormalization=='yes':
                        exec('model_res_shortcut'+str(m+1)+'=self.align_residual(model_res_shortcut'+str(m+1)+', model_norm'+str(m+1)+')', globals(), self.__dict__)
                        exec('model_norm'+str(m+1)+'=model_norm'+str(m+1)+' + model_res_shortcut'+str(m+1), globals(), self.__dict__)
                    else:
                        exec('model_res_shortcut'+str(m+1)+'=self.align_residual(model_res_shortcut'+str(m+1)+', model_act'+str(m+1)+')', globals(), self.__dict__)
                        exec('model_act'+str(m+1)+'=model_act'+str(m+1)+' + model_res_shortcut'+str(m+1), globals(), self.__dict__)
                        # original decoder uses model_conv{self.k} as covup when no normalization.
                        # Overwrite it so the residual path really affects the following upsample.
                        exec('model_conv'+str(m+1)+'=model_act'+str(m+1), globals(), self.__dict__)

                if self.ifnormalization=='yes':
                    exec('self.covup'+str(m+1)+'=model_norm'+str(self.k), globals(), self.__dict__)
                else:
                    exec('self.covup'+str(m+1)+'=model_conv'+str(self.k), globals(), self.__dict__)

                # DA skip has already been converted to 4D: (B, C, H, W)
                exec('self.covdown'+str(m+1)+'=model_skip'+str(self.l), globals(), self.__dict__)

                exec('model_up'+str(m+1)+'=self.up'+str(m+1)+'(self.covup'+str(m+1)+')', globals(), self.__dict__)
                exec('model_uppad'+str(m+1)+'=self.uppad'+str(m+1)+'(model_up'+str(m+1)+')', globals(), self.__dict__)
                exec('model_conc'+str(m+1)+'=self.conc'+str(m+1)+'((model_uppad'+str(m+1)+',self.covdown'+str(m+1)+'),axis=1)', globals(), self.__dict__)

                for n in range(int(self.covnum)):
                    if n==0:
                        exec('model_res_input'+str(m+1)+'_'+str(n+1)+'=model_conc'+str(m+1), globals(), self.__dict__)
                    else:
                        if self.ifnormalization=='yes':
                            exec('model_res_input'+str(m+1)+'_'+str(n+1)+'=model_norm'+str(m+1)+'_'+str(n), globals(), self.__dict__)
                        else:
                            exec('model_res_input'+str(m+1)+'_'+str(n+1)+'=model_act'+str(m+1)+'_'+str(n), globals(), self.__dict__)

                    exec('model_conv'+str(m+1)+'_'+str(n+1)+'=self.conv'+str(m+1)+'_'+str(n+1)+'(model_res_input'+str(m+1)+'_'+str(n+1)+')', globals(), self.__dict__)
                    exec('model_pad'+str(m+1)+'_'+str(n+1)+'=self.pad'+str(m+1)+'_'+str(n+1)+'(model_conv'+str(m+1)+'_'+str(n+1)+')', globals(), self.__dict__)
                    exec('model_act'+str(m+1)+'_'+str(n+1)+'=self.act'+str(m+1)+'_'+str(n+1)+'(model_pad'+str(m+1)+'_'+str(n+1)+')', globals(), self.__dict__)
                    if self.ifnormalization=='yes':
                        exec('model_norm'+str(m+1)+'_'+str(n+1)+'=self.norm'+str(m+1)+'_'+str(n+1)+'(model_act'+str(m+1)+'_'+str(n+1)+')', globals(), self.__dict__)

                    # Decoder post-concat residual shortcut
                    if self.if_resnet == 'yes':
                        res_layer = getattr(self, 'res' + str(m+1) + '_' + str(n+1))
                        self.__dict__['res_layer'] = res_layer
                        if res_layer is not None:
                            exec('model_res_shortcut'+str(m+1)+'_'+str(n+1)+'=res_layer(model_res_input'+str(m+1)+'_'+str(n+1)+')', globals(), self.__dict__)
                        else:
                            exec('model_res_shortcut'+str(m+1)+'_'+str(n+1)+'=model_res_input'+str(m+1)+'_'+str(n+1), globals(), self.__dict__)

                        if self.ifnormalization=='yes':
                            exec('model_res_shortcut'+str(m+1)+'_'+str(n+1)+'=self.align_residual(model_res_shortcut'+str(m+1)+'_'+str(n+1)+', model_norm'+str(m+1)+'_'+str(n+1)+')', globals(), self.__dict__)
                            exec('model_norm'+str(m+1)+'_'+str(n+1)+'=model_norm'+str(m+1)+'_'+str(n+1)+' + model_res_shortcut'+str(m+1)+'_'+str(n+1), globals(), self.__dict__)
                        else:
                            exec('model_res_shortcut'+str(m+1)+'_'+str(n+1)+'=self.align_residual(model_res_shortcut'+str(m+1)+'_'+str(n+1)+', model_act'+str(m+1)+'_'+str(n+1)+')', globals(), self.__dict__)
                            exec('model_act'+str(m+1)+'_'+str(n+1)+'=model_act'+str(m+1)+'_'+str(n+1)+' + model_res_shortcut'+str(m+1)+'_'+str(n+1), globals(), self.__dict__)

                self.k=self.k+1
                self.l=self.l-1

            if if_last_act!='no':
                if self.ifnormalization=='yes':
                    exec('model_convlast=self.convlast(model_norm'+str(m+1)+'_'+str(n+1)+')', globals(), self.__dict__)
                else:
                    exec('model_convlast=self.convlast(model_act'+str(m+1)+'_'+str(n+1)+')', globals(), self.__dict__)
                outputs=eval('self.actlast(model_convlast)', globals(), self.__dict__)
            else:
                if self.ifnormalization=='yes':
                    outputs=eval('self.convlast(model_norm'+str(m+1)+'_'+str(n+1)+')', globals(), self.__dict__)
                else:
                    outputs=eval('self.convlast(model_act'+str(m+1)+'_'+str(n+1)+')', globals(), self.__dict__)
            return outputs
    trainx=np.zeros((2,timestep,input_channels,49,73),dtype=np.float32)
    trainy=np.zeros((2,9,49,73),dtype=np.float32)
    model=Model(trainx,trainy,timestep,downnum,covnum,layer_time,baselayer,cov_kernelsize,cov_strides,pool_method,pool_kernel_size,pool_strides,ifnormalization,normalization_method,trans_num,num_heads,dim_feedforward,trans_dropout_rate,activate,nlp_activate,da_attention_order,if_bottom_attention_only,if_resnet)
    return model.to(devices)
