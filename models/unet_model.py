def Auto_Unet_Based_Model_pytorch(vy,vx,downnum,covnum,timestep=None,unet_mode='single',test_size=0.2,valid_size=0.1,k_fold=None,task_mode='regression',if_best_mode='no',modelpath=None,ifrandom_split='yes',if_bottom_attention_only='no',if_resnet='no',if_bottom_res_only='no',layer_time=2,baselayer=64,cov_kernelsize=3,cov_strides=1,pool_method='maxpooling',pool_kernel_size=2,pool_strides=2,ifnormalization='no',normalization_method=None,da_order='space_first',ta_order='cts',trans_num=2,num_heads=2,dim_feedforward=256,trans_dropout_rate=0.0,mamba_class='mamba2',mamba_state=64,mamba_conv=4,mamba_expand=2,activate='tanh',nlp_activate='relu',if_last_act='no',if_print_model='yes',loss_function='default',optimizer='SGD',metrics='default',learning_rate=0.01,lr_decay=None,regularization=None,regularization_lambda=1e-5,epochs=2000,batch_size=20,if_early_stopping=None,ifheatmap='no',ifweight='yes',ifmute='no',ifsave='no',savepath=None,device='cpu'):
    from torch.nn import Module,BatchNorm1d,BatchNorm2d,LayerNorm,GroupNorm,Conv2d,MaxPool2d,AvgPool2d,Dropout,LeakyReLU,ReLU,PReLU,Sigmoid,Tanh,ELU,Softmax,Linear,Flatten,ZeroPad2d,Upsample,TransformerEncoderLayer,TransformerEncoder
    from torch.optim import Adam,NAdam,SGD
    from torch.optim.lr_scheduler import ExponentialLR
    import torch
    from torch import nn
    import torch.nn.functional as F
    import math
    import numpy as np
    import os
    import copy
    import datetime
    import warnings
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import KFold
    from scipy.stats import pearsonr
    from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score
    import sklearn
    try:
        import torchmetrics
    except Exception:
        torchmetrics = None
    try:
        import shap
    except Exception:
        shap = None
    try:
        from tqdm import tqdm
    except Exception:
        def tqdm(x):
            return x
    warnings.filterwarnings('ignore')

    if lr_decay is not None:
        lr_decay = float(lr_decay)
        if lr_decay <= 0:
            raise ValueError("lr_decay 必须为 None 或正浮点数。例如 None、0.99、0.95。")

    if regularization is not None:
        regularization = str(regularization).lower()
        if regularization not in ['l1', 'l2']:
            raise ValueError("regularization 只能为 None、'l1' 或 'l2'。")
        if regularization_lambda is None:
            regularization_lambda = 1e-5
        regularization_lambda = float(regularization_lambda)
        if regularization_lambda < 0:
            raise ValueError("regularization_lambda 必须 >= 0。")

    def add_regularization_loss(base_loss, model_instance):
        if regularization is None or regularization_lambda == 0:
            return base_loss

        reg_loss = torch.zeros((), dtype=base_loss.dtype, device=base_loss.device)

        for name, param in model_instance.named_parameters():
            if not param.requires_grad:
                continue

            if param.ndim <= 1 or name.endswith('.bias'):
                continue

            if regularization == 'l1':
                reg_loss = reg_loss + torch.sum(torch.abs(param))
            elif regularization == 'l2':
                reg_loss = reg_loss + torch.sum(param ** 2)

        return base_loss + regularization_lambda * reg_loss

    unet_mode=str(unet_mode).lower()
    valid_unet_modes=['single','trans','mamba','stt','stm','dat','dam','tat','tam']
    if unet_mode not in valid_unet_modes:
        raise ValueError("unet_mode must be one of ['single','trans','mamba','stt','stm','dat','dam','tat','tam']")

    if unet_mode in ['mamba','stm','dam','tam']:
        import mamba_ssm.modules.mamba2 as mamba2_module
        mamba2_module.causal_conv1d_fn = None
        mamba2_module.causal_conv1d_update = None
        import mamba_ssm.modules.mamba_simple as mamba_module
        mamba_module.causal_conv1d_fn = None
        mamba_module.causal_conv1d_update = None
        from mamba_ssm import Mamba,Mamba2
        globals()['mamba2_module']=mamba2_module
        globals()['mamba_module']=mamba_module
        globals()['Mamba']=Mamba
        globals()['Mamba2']=Mamba2

    temporal_modes=['stt','stm','dat','dam','tat','tam']
    mamba_modes=['mamba','stm','dam','tam']
    transformer_modes=['trans','stt','dat','tat']
    if unet_mode in temporal_modes and timestep is None:
        raise ValueError("When unet_mode is one of ['stt','stm','dat','dam','tat','tam'], timestep must not be None.")
    devices = torch.device(os.environ.get('YANGTZE_DEVICE', 'cuda') if device == 'gpu' else device)

    if task_mode=='binary_classify' or task_mode=='multi_classify':
        def get_class_weights_from_labels(vy, device=None, ignore_index=None, num_classes=None):
            y = torch.as_tensor(vy, device=device)
            if y.ndim >= 1 and y.shape[-1] == 1:
                y = y.squeeze(-1)
            y = y.reshape(-1)
            if ignore_index is not None:
                y = y[y != ignore_index]
            if y.is_floating_point():
                y = y[~torch.isnan(y)]
            y = y.long()
            if num_classes is None:
                num_classes = int(y.max().item()) + 1
            cnt = torch.bincount(y, minlength=num_classes).float()
            w = cnt.sum() / (num_classes * cnt.clamp_min(1.0))
            return w
        class_weights = get_class_weights_from_labels(vy, device=devices)

    if task_mode=='regression':
        if type(loss_function) is not str:
            loss=loss_function
        else:
            if loss_function=='default' or loss_function=='MSELoss':
                loss=torch.nn.MSELoss()
            elif loss_function=='L1Loss':
                loss=torch.nn.L1Loss()
            elif loss_function=='PoissonNLLLoss':
                loss=torch.nn.PoissonNLLLoss()
            elif loss_function=='GaussianNLLLoss':
                loss=torch.nn.GaussianNLLLoss()
            elif loss_function=='KLDivLoss':
                loss=torch.nn.KLDivLoss()
            elif loss_function=='HuberLoss':
                loss=torch.nn.HuberLoss()
            elif loss_function=='SmoothL1Loss':
                loss=torch.nn.SmoothL1Loss()
            elif loss_function=='Pearsonr':
                class loss_pearsonr(nn.Module):
                    def __init__(self):
                        super().__init__()
                    def forward(self, y, x):
                        y_true_mean=torch.nanmean(y,dim=0,keepdim=True)
                        y_pred_mean=torch.nanmean(x,dim=0,keepdim=True)
                        cov=torch.nansum((y-y_true_mean)*(x-y_pred_mean),dim=0,keepdim=True)
                        y_true_v=torch.nansum(torch.square((y-y_true_mean)),dim=0,keepdim=True)
                        y_pred_v=torch.nansum(torch.square((x-y_pred_mean)),dim=0,keepdim=True)
                        y_true_v=torch.sqrt(y_true_v)
                        y_pred_v=torch.sqrt(y_pred_v)
                        pearson=cov/(y_true_v*y_pred_v)
                        return torch.nanmean((1-pearson)**1.5)
                loss=loss_pearsonr()
            else:
                raise ValueError('Unsupported loss_function for regression.')
        if type(metrics) is not str:
            metric=metrics
        else:
            if metrics=='default' or metrics=='MSELoss':
                metric=torch.nn.MSELoss()
            elif metrics=='L1Loss':
                metric=torch.nn.L1Loss()
            elif metrics=='PoissonNLLLoss':
                metric=torch.nn.PoissonNLLLoss()
            elif metrics=='GaussianNLLLoss':
                metric=torch.nn.GaussianNLLLoss()
            elif metrics=='KLDivLoss':
                metric=torch.nn.KLDivLoss()
            elif metrics=='HuberLoss':
                metric=torch.nn.HuberLoss()
            elif metrics=='SmoothL1Loss':
                metric=torch.nn.SmoothL1Loss()
            elif metrics=='Pearsonr':
                class metric_pearsonr(nn.Module):
                    def __init__(self):
                        super().__init__()
                    def forward(self, y, x):
                        y_true_mean=torch.nanmean(y,dim=0,keepdim=True)
                        y_pred_mean=torch.nanmean(x,dim=0,keepdim=True)
                        cov=torch.nansum((y-y_true_mean)*(x-y_pred_mean),dim=0,keepdim=True)
                        y_true_v=torch.nansum(torch.square((y-y_true_mean)),dim=0,keepdim=True)
                        y_pred_v=torch.nansum(torch.square((x-y_pred_mean)),dim=0,keepdim=True)
                        y_true_v=torch.sqrt(y_true_v)
                        y_pred_v=torch.sqrt(y_pred_v)
                        pearson=cov/(y_true_v*y_pred_v)
                        return torch.nanmean((1-pearson)**1.5)
                metric=metric_pearsonr()
            else:
                raise ValueError('Unsupported metrics for regression.')
    elif task_mode=='binary_classify':
        if type(loss_function) is not str:
            loss=loss_function
        else:
            if loss_function=='default' or loss_function=='BCELoss':
                loss=torch.nn.BCELoss()
            elif loss_function=='BCEWithLogitsLoss':
                pos_w = (class_weights[1] / class_weights[0]).to(devices).view(1)
                loss=torch.nn.BCEWithLogitsLoss(pos_weight=pos_w)
            elif loss_function=='SoftMarginLoss':
                loss=torch.nn.SoftMarginLoss()
            elif loss_function=='MultiLabelSoftMarginLoss':
                loss=torch.nn.MultiLabelSoftMarginLoss()
            else:
                raise ValueError('Unsupported loss_function for binary_classify.')
        if type(metrics) is not str:
            metric=metrics
        else:
            if torchmetrics is not None and (metrics=='default' or metrics=='f1'):
                metric=torchmetrics.F1Score(task="binary").to(devices)
            elif torchmetrics is not None and metrics=='accuracy':
                metric=torchmetrics.Accuracy(task="binary").to(devices)
            elif torchmetrics is not None and metrics=='precision':
                metric=torchmetrics.Precision(task="binary").to(devices)
            elif torchmetrics is not None and metrics=='recall':
                metric=torchmetrics.Recall(task="binary").to(devices)
            elif metrics=='BCELoss':
                metric=torch.nn.BCELoss()
            elif metrics=='BCEWithLogitsLoss':
                metric=torch.nn.BCEWithLogitsLoss()
            else:
                metric=torch.nn.BCELoss()
    elif task_mode=='multi_classify':
        if type(loss_function) is not str:
            loss=loss_function
        else:
            if loss_function=='default' or loss_function=='CrossEntropyLoss':
                loss=torch.nn.CrossEntropyLoss(weight=class_weights)
            elif loss_function=='NLLLoss':
                loss=torch.nn.NLLLoss(weight=class_weights)
            elif loss_function=='TripletMarginLoss':
                loss=torch.nn.TripletMarginLoss()
            elif loss_function=='HingeEmbeddingLoss':
                loss=torch.nn.HingeEmbeddingLoss()
            elif loss_function=='MultiLabelMarginLoss':
                loss=torch.nn.MultiLabelMarginLoss()
            elif loss_function=='TripletMarginWithDistanceLoss':
                loss=torch.nn.TripletMarginWithDistanceLoss()
            else:
                raise ValueError('Unsupported loss_function for multi_classify.')
        if type(metrics) is not str:
            metric=metrics
        else:
            if torchmetrics is not None and (metrics=='default' or metrics=='accuracy'):
                metric=torchmetrics.Accuracy(task="multiclass",num_classes = int(np.max(vy))+1, average="macro").to(devices)
            elif torchmetrics is not None and metrics=='recall' :
                metric=torchmetrics.Recall(task="multiclass", num_classes=int(np.max(vy))+1, average="macro").to(devices)
            elif torchmetrics is not None and metrics=='precision' :
                metric=torchmetrics.Precision(task="multiclass", num_classes=int(np.max(vy))+1, average="macro").to(devices)
            elif torchmetrics is not None and metrics=='f1' :
                metric=torchmetrics.F1Score(task="multiclass", num_classes=int(np.max(vy))+1, average="macro").to(devices)
            elif metrics=='CrossEntropyLoss':
                metric=torch.nn.CrossEntropyLoss()
            elif metrics=='NLLLoss':
                metric=torch.nn.NLLLoss()
            else:
                metric=torch.nn.CrossEntropyLoss()
    else:
        raise ValueError("task_mode must be 'regression', 'binary_classify', or 'multi_classify'.")

    weights=0
    heatmap=0
    model=0
    predicty=None
    testy=None
    r=None
    p=None

    class ShapWrapper(Module):
        def __init__(self, original_model, x_sample_shape, output_channel_idx=None):
            super().__init__()
            self.original_model = original_model
            self.x_sample_shape = x_sample_shape
            self.x_flat_size = np.prod(x_sample_shape)
            self.output_channel_idx = output_channel_idx
        def forward(self, x_flat_tensor):
            if x_flat_tensor.dim() == 1:
                x_flat_tensor = x_flat_tensor.unsqueeze(0)
            x_reshaped = x_flat_tensor.reshape(-1, *self.x_sample_shape)
            model_output = self.original_model(x_reshaped)
            if model_output.is_floating_point() is False:
                model_output = model_output.float()
            if self.output_channel_idx is not None and model_output.ndim == 4:
                return torch.mean(model_output[:, self.output_channel_idx, :, :].reshape(model_output.size(0), -1), dim=1).unsqueeze(1)
            elif model_output.ndim == 4:
                return torch.mean(model_output.reshape(model_output.size(0), -1), dim=1).unsqueeze(1)
            else:
                return torch.mean(model_output.reshape(model_output.size(0), -1), dim=1).unsqueeze(1)

    class EarlyStopping:
        def __init__(self, patience, delta=0):
            self.patience = patience
            self.counter = 0
            self.best_score = None
            self.early_stop = False
            self.val_loss_min = np.Inf
            self.delta = delta
            self.best_model_state_dict = None
        def __call__(self, val_loss, model):
            if np.isnan(val_loss):
                print("Validation loss is NaN. Stopping early and will restore best model.")
                self.early_stop = True
                return
            score = -val_loss
            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(val_loss, model)
            elif score < self.best_score + self.delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint(val_loss, model)
                self.counter = 0
        def save_checkpoint(self, val_loss, model):
            self.best_model_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
            self.val_loss_min = val_loss
        def load_best_checkpoint(self, model):
            if self.best_model_state_dict is not None:
                model.load_state_dict(self.best_model_state_dict)
            else:
                print("Warning: load_best_checkpoint called but no best_model_state_dict was saved.")

    if vy.ndim==3:
        vy=vy.reshape(vy.shape[0],vy.shape[1],vy.shape[2],1)
    vy=vy.transpose(0,3,1,2)

    if unet_mode in temporal_modes:
        vys=vy[timestep-1:,:,:,:]
        vxs=np.zeros((vx.shape[0]-timestep+1,timestep,vx.shape[1],vx.shape[2],vx.shape[3]))
        for i in range(vx.shape[0]-timestep+1):
            vxs[i,:,:,:,:]=vx[i:i+timestep,:,:,:]
        vxs=vxs.transpose(0,1,4,2,3)
    else:
        vxs=vx.transpose(0,3,1,2)
        vys=vy

    if ifrandom_split=='yes':
        trainy,testy,trainx,testx = train_test_split(vys,vxs,test_size=test_size,random_state=25)
    elif ifrandom_split=='no':
        index=int((1-test_size)*vys.shape[0])
        trainy=vys[:index]
        testy=vys[index:]
        trainx=vxs[:index]
        testx=vxs[index:]
    elif ifrandom_split=='all_train' or ifrandom_split=='all_test' or ifrandom_split=='just_model':
        trainy=vys
        testy=vys
        trainx=vxs
        testx=vxs
    else:
        raise ValueError("ifrandom_split must be 'yes', 'no', 'all_train', 'all_test', or 'just_model'.")

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
            x_reshaped = x.contiguous().view(batch_size * seq_len, *x.shape[2:])
            y = self.module(x_reshaped)
            if self.batch_first:
                y = y.contiguous().view(batch_size, seq_len, *y.shape[1:])
            else:
                y = y.contiguous().view(seq_len, batch_size, *y.shape[1:])
            return y

    class Model(nn.Module):
        def __init__(self,trainx,trainy,timestep,unet_mode,downnum,covnum,layer_time,baselayer,cov_kernelsize,cov_strides,pool_method,pool_kernel_size,pool_strides,ifnormalization,normalization_method,trans_num,num_heads,dim_feedforward,trans_dropout_rate,mamba_class,mamba_state,mamba_conv,mamba_expand,activate,nlp_activate,da_order,ta_order,if_bottom_attention_only,if_resnet,if_bottom_res_only):
            super(Model,self).__init__()
            exec('from torch.nn import Module,BatchNorm1d,BatchNorm2d,LayerNorm,GroupNorm,Conv2d,MaxPool2d,AvgPool2d,Dropout,LeakyReLU,ReLU,PReLU,Sigmoid,Tanh,ELU,Softmax,Linear,Flatten,ZeroPad2d,Upsample,TransformerEncoderLayer,TransformerEncoder', globals(), self.__dict__)
            exec('import numpy as np', globals(), self.__dict__)
            exec('import math', globals(), self.__dict__)
            exec('import torch', globals(), self.__dict__)
            exec('from torch import nn', globals(), self.__dict__)
            exec('import torch.nn.functional as F', globals(), self.__dict__)
            self.timestep=timestep
            self.unet_mode=unet_mode
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
            self.mamba_class=mamba_class
            self.mamba_state=mamba_state
            self.mamba_conv=mamba_conv
            self.mamba_expand=mamba_expand
            self.activate=activate
            self.nlp_activate=nlp_activate
            self.da_order=da_order
            self.ta_order=ta_order
            self.if_bottom_attention_only=if_bottom_attention_only
            self.if_resnet=if_resnet
            self.if_bottom_res_only=if_bottom_res_only
            if self.if_bottom_attention_only not in ['yes','no']:
                raise ValueError("if_bottom_attention_only must be 'yes' or 'no'")
            if self.if_resnet not in ['yes','no']:
                raise ValueError("if_resnet must be 'yes' or 'no'")
            if self.if_bottom_res_only not in ['yes','no']:
                raise ValueError("if_bottom_res_only must be 'yes' or 'no'")
            self.trainx=trainx
            self.trainy=trainy
            self.is_temporal = self.unet_mode in ['stt','stm','dat','dam','tat','tam']
            self.is_mamba = self.unet_mode in ['mamba','stm','dam','tam']
            self.is_transformer = self.unet_mode in ['trans','stt','dat','tat']
            if self.is_mamba:
                exec('import mamba_ssm.modules.mamba_simple as mamba_module', globals(), self.__dict__)
                exec('mamba_module.causal_conv1d_fn = None', globals(), self.__dict__)
                exec('mamba_module.causal_conv1d_update = None', globals(), self.__dict__)
                exec('import mamba_ssm.modules.mamba2 as mamba2_module', globals(), self.__dict__)
                exec('mamba2_module.causal_conv1d_fn = None', globals(), self.__dict__)
                exec('mamba2_module.causal_conv1d_update = None', globals(), self.__dict__)
                exec('from mamba_ssm import Mamba,Mamba2', globals(), self.__dict__)
            if self.is_temporal:
                self.hight0=int(self.trainx.shape[3])
                self.weight0=int(self.trainx.shape[4])
                self.input_channels=int(self.trainx.shape[2])
            else:
                self.hight0=int(self.trainx.shape[2])
                self.weight0=int(self.trainx.shape[3])
                self.input_channels=int(self.trainx.shape[1])
            self.upsize=pool_strides
            self.__dict__['self']=self
            self.__dict__['TimeDistributed']=TimeDistributed
            if self.pool_strides!=self.pool_kernel_size:
                print('池化步长和池化核不等,将导致下采样前后通道数相除不为整数！')
            W=self.hight0
            H=self.weight0
            for i in range(int(self.downnum-1)):
                W=math.floor(((W-self.pool_kernel_size)/self.pool_strides)+1)
                H=math.floor(((H-self.pool_kernel_size)/self.pool_strides)+1)
            if W<1 or H<1:
                raise ValueError('下采样次数过多,导致最下层图形长度/宽度小于1!')
            self.encoder_channels=[]
            self.encoder_h=[]
            self.encoder_w=[]
            self.k=0
            self.total_covnum=0

            def _get_act_string(act_name):
                if act_name=='elu':
                    return 'ELU()'
                elif act_name=='leakyrelu':
                    return 'LeakyReLU()'
                elif act_name=='prelu':
                    return 'PReLU()'
                elif act_name=='relu':
                    return 'ReLU()'
                elif act_name=='sigmoid':
                    return 'Sigmoid()'
                elif act_name=='tanh':
                    return 'Tanh()'
                elif act_name=='softmax':
                    return 'Softmax(dim=1)'
                else:
                    return 'Tanh()'
            self._get_act_string=_get_act_string

            def build_mamba_block(d_model):
                if self.mamba_class == 'mamba2':
                    mamba_inner = self.mamba_expand * d_model
                    mamba_ssm = None
                    mamba_headdim = 64 if mamba_inner % 64 == 0 else 32 if mamba_inner % 32 == 0 else 16 if mamba_inner % 16 == 0 else 8 if mamba_inner % 8 == 0 else 4 if mamba_inner % 4 == 0 else 2 if mamba_inner % 2 == 0 else 1
                    return Mamba2(d_model=d_model,d_state=self.mamba_state,d_conv=self.mamba_conv,expand=self.mamba_expand,headdim=mamba_headdim,d_ssm=mamba_ssm,use_mem_eff_path=False)
                else:
                    return Mamba(d_model=d_model,d_state=self.mamba_state,d_conv=self.mamba_conv,expand=self.mamba_expand)
            self.build_mamba_block=build_mamba_block

            current_channels=self.input_channels
            current_h=self.hight0
            current_w=self.weight0
            for i in range(int(self.downnum)):
                out_channels=int(baselayer*(self.layer_time**i))
                for j in range(int(self.covnum)):
                    layer_id=str(i+1)+'_'+str(j+1)
                    in_channels=current_channels if j==0 else out_channels
                    self.__dict__['in_channels']=int(in_channels)
                    self.__dict__['out_channels']=int(out_channels)
                    if self.is_temporal:
                        exec('self.pad'+layer_id+'=TimeDistributed(ZeroPad2d(0), batch_first=True)', globals(), self.__dict__)
                        exec('self.conv'+layer_id+'=TimeDistributed(Conv2d(in_channels,out_channels,(self.cov_kernelsize,self.cov_kernelsize),stride=self.cov_strides,padding="same"), batch_first=True)', globals(), self.__dict__)
                    else:
                        exec('self.pad'+layer_id+'=ZeroPad2d(0)', globals(), self.__dict__)
                        exec('self.conv'+layer_id+'=Conv2d(in_channels,out_channels,(self.cov_kernelsize,self.cov_kernelsize),stride=self.cov_strides,padding="same")', globals(), self.__dict__)
                    exec('self.act'+layer_id+'='+self._get_act_string(self.activate), globals(), self.__dict__)
                    if self.ifnormalization=='yes':
                        if self.normalization_method=='batchnormalization':
                            if self.is_temporal:
                                exec('self.norm'+layer_id+'=TimeDistributed(BatchNorm2d(out_channels), batch_first=True)', globals(), self.__dict__)
                            else:
                                exec('self.norm'+layer_id+'=BatchNorm2d(out_channels)', globals(), self.__dict__)
                        elif self.normalization_method=='layernormalization':
                            if self.is_temporal:
                                exec('self.norm'+layer_id+'=TimeDistributed(GroupNorm(1,out_channels), batch_first=True)', globals(), self.__dict__)
                            else:
                                exec('self.norm'+layer_id+'=GroupNorm(1,out_channels)', globals(), self.__dict__)
                    if self.if_resnet=='yes' and (self.if_bottom_res_only=='no' or i==int(self.downnum)-1):
                        if int(in_channels)!=int(out_channels) or self.cov_strides!=1:
                            if self.is_temporal:
                                setattr(self,'res'+layer_id,TimeDistributed(Conv2d(int(in_channels),int(out_channels),(1,1),stride=self.cov_strides,padding=0), batch_first=True))
                            else:
                                setattr(self,'res'+layer_id,Conv2d(int(in_channels),int(out_channels),(1,1),stride=self.cov_strides,padding=0))
                        else:
                            setattr(self,'res'+layer_id,None)
                    current_channels=out_channels
                    self.total_covnum=self.total_covnum+1
                self.encoder_channels.append(int(out_channels))
                self.encoder_h.append(int(current_h))
                self.encoder_w.append(int(current_w))
                if self._need_attention_layer(i):
                    self._build_attention_layer(i+1,int(out_channels),int(current_h),int(current_w))
                if i != int(self.downnum)-1:
                    if self.pool_method=='maxpooling':
                        if self.is_temporal:
                            exec('self.pool'+str(i+1)+'=TimeDistributed(MaxPool2d((self.pool_kernel_size,self.pool_kernel_size),stride=self.pool_strides), batch_first=True)', globals(), self.__dict__)
                        else:
                            exec('self.pool'+str(i+1)+'=MaxPool2d((self.pool_kernel_size,self.pool_kernel_size),stride=self.pool_strides)', globals(), self.__dict__)
                    elif self.pool_method=='avepooling':
                        if self.is_temporal:
                            exec('self.pool'+str(i+1)+'=TimeDistributed(AvgPool2d((self.pool_kernel_size,self.pool_kernel_size),stride=self.pool_strides), batch_first=True)', globals(), self.__dict__)
                        else:
                            exec('self.pool'+str(i+1)+'=AvgPool2d((self.pool_kernel_size,self.pool_kernel_size),stride=self.pool_strides)', globals(), self.__dict__)
                    current_h=1+math.floor((current_h-self.pool_kernel_size)/self.pool_strides)
                    current_w=1+math.floor((current_w-self.pool_kernel_size)/self.pool_strides)

            decoder_in_channels=self.encoder_channels[-1]
            decoder_index=0
            for skip_level in range(int(self.downnum)-2,-1,-1):
                decoder_index=decoder_index+1
                m=int(self.downnum)+decoder_index
                skip_channels=self.encoder_channels[skip_level]
                self.__dict__['in_channels']=int(decoder_in_channels)
                self.__dict__['out_channels']=int(skip_channels)
                exec('self.conv'+str(m)+'=Conv2d(in_channels,out_channels,(self.cov_kernelsize,self.cov_kernelsize),stride=self.cov_strides,padding="same")', globals(), self.__dict__)
                exec('self.pad'+str(m)+'=ZeroPad2d(0)', globals(), self.__dict__)
                exec('self.act'+str(m)+'='+self._get_act_string(self.activate), globals(), self.__dict__)
                if self.ifnormalization=='yes':
                    if self.normalization_method=='batchnormalization':
                        exec('self.norm'+str(m)+'=BatchNorm2d(out_channels)', globals(), self.__dict__)
                    elif self.normalization_method=='layernormalization':
                        exec('self.norm'+str(m)+'=GroupNorm(1,out_channels)', globals(), self.__dict__)
                if self.if_resnet=='yes' and (self.if_bottom_res_only=='no' or decoder_index==1):
                    if int(decoder_in_channels)!=int(skip_channels) or self.cov_strides!=1:
                        setattr(self,'res'+str(m),Conv2d(int(decoder_in_channels),int(skip_channels),(1,1),stride=self.cov_strides,padding=0))
                    else:
                        setattr(self,'res'+str(m),None)
                exec('self.up'+str(m)+'=Upsample(scale_factor=self.upsize, mode="bilinear")', globals(), self.__dict__)
                exec('self.conc'+str(m)+'=torch.concatenate', globals(), self.__dict__)
                decoder_conv_in=skip_channels*2
                for n in range(int(self.covnum)):
                    layer_id=str(m)+'_'+str(n+1)
                    in_channels=decoder_conv_in if n==0 else skip_channels
                    self.__dict__['in_channels']=int(in_channels)
                    self.__dict__['out_channels']=int(skip_channels)
                    exec('self.pad'+layer_id+'=ZeroPad2d(0)', globals(), self.__dict__)
                    exec('self.conv'+layer_id+'=Conv2d(in_channels,out_channels,(self.cov_kernelsize,self.cov_kernelsize),stride=self.cov_strides,padding="same")', globals(), self.__dict__)
                    exec('self.act'+layer_id+'='+self._get_act_string(self.activate), globals(), self.__dict__)
                    if self.ifnormalization=='yes':
                        if self.normalization_method=='batchnormalization':
                            exec('self.norm'+layer_id+'=BatchNorm2d(out_channels)', globals(), self.__dict__)
                        elif self.normalization_method=='layernormalization':
                            exec('self.norm'+layer_id+'=GroupNorm(1,out_channels)', globals(), self.__dict__)
                    if self.if_resnet=='yes' and (self.if_bottom_res_only=='no' or decoder_index==1):
                        if int(in_channels)!=int(skip_channels) or self.cov_strides!=1:
                            setattr(self,'res'+layer_id,Conv2d(int(in_channels),int(skip_channels),(1,1),stride=self.cov_strides,padding=0))
                        else:
                            setattr(self,'res'+layer_id,None)
                decoder_in_channels=skip_channels

            if task_mode=='multi_classify':
                self.__dict__['decoder_in_channels']=int(decoder_in_channels)
                exec('self.convlast=Conv2d(decoder_in_channels,int(np.max(self.trainy))+1,(1,1),stride=1,padding="same")', globals(), self.__dict__)
            else:
                self.__dict__['decoder_in_channels']=int(decoder_in_channels)
                exec('self.convlast=Conv2d(decoder_in_channels,self.trainy.shape[1],(1,1),stride=1,padding="same")', globals(), self.__dict__)
            if if_last_act!='no':
                exec('self.actlast='+self._get_act_string(if_last_act), globals(), self.__dict__)

        def _need_attention_layer(self, i):
            is_bottom = i == int(self.downnum)-1
            if self.unet_mode in ['single']:
                return False
            if self.unet_mode in ['trans','mamba','stt','stm']:
                return is_bottom
            if self.unet_mode in ['dat','dam','tat','tam']:
                return (self.if_bottom_attention_only=='no') or is_bottom
            return False

        def _build_transformer(self, d_model, nhead=None):
            if nhead is None:
                nhead=self.num_heads
            encoder_layer=TransformerEncoderLayer(d_model=d_model,nhead=nhead,dim_feedforward=self.dim_feedforward,dropout=self.trans_dropout_rate,activation=self.nlp_activate,batch_first=True)
            return TransformerEncoder(encoder_layer,num_layers=self.trans_num)

        def _build_attention_layer(self, layer_id, channels, hight, weight):
            if self.unet_mode in ['trans']:
                if channels % self.num_heads != 0:
                    raise ValueError('For spatial Transformer, channels must be divisible by num_heads.')
                setattr(self,'space_transformer'+str(layer_id),self._build_transformer(channels,self.num_heads))
                setattr(self,'space_embed'+str(layer_id),nn.Parameter(torch.zeros(1,hight*weight,channels)))
                nn.init.trunc_normal_(getattr(self,'space_embed'+str(layer_id)),std=0.02)
            elif self.unet_mode in ['mamba']:
                setattr(self,'space_mamba'+str(layer_id),self.build_mamba_block(channels))
                setattr(self,'space_embed'+str(layer_id),nn.Parameter(torch.zeros(1,hight*weight,channels)))
                nn.init.trunc_normal_(getattr(self,'space_embed'+str(layer_id)),std=0.02)
            elif self.unet_mode in ['stt']:
                d_model=channels*hight*weight
                if d_model % self.num_heads != 0:
                    raise ValueError('For temporal Transformer, C*H*W must be divisible by num_heads.')
                setattr(self,'time_transformer'+str(layer_id),self._build_transformer(d_model,self.num_heads))
                setattr(self,'time_embed'+str(layer_id),nn.Parameter(torch.zeros(1,self.timestep,d_model)))
                nn.init.trunc_normal_(getattr(self,'time_embed'+str(layer_id)),std=0.02)
            elif self.unet_mode in ['stm']:
                d_model=channels*hight*weight
                setattr(self,'time_mamba'+str(layer_id),self.build_mamba_block(d_model))
                setattr(self,'time_embed'+str(layer_id),nn.Parameter(torch.zeros(1,self.timestep,d_model)))
                nn.init.trunc_normal_(getattr(self,'time_embed'+str(layer_id)),std=0.02)
            elif self.unet_mode in ['dat','tat']:
                d_time=channels*hight*weight
                if d_time % self.num_heads != 0:
                    raise ValueError('For time Transformer, C*H*W must be divisible by num_heads.')
                if channels % self.num_heads != 0:
                    raise ValueError('For space Transformer, C must be divisible by num_heads.')
                setattr(self,'time_transformer'+str(layer_id),self._build_transformer(d_time,self.num_heads))
                setattr(self,'time_embed'+str(layer_id),nn.Parameter(torch.zeros(1,self.timestep,d_time)))
                nn.init.trunc_normal_(getattr(self,'time_embed'+str(layer_id)),std=0.02)
                setattr(self,'space_transformer'+str(layer_id),self._build_transformer(channels,self.num_heads))
                setattr(self,'space_embed'+str(layer_id),nn.Parameter(torch.zeros(1,hight*weight,channels)))
                nn.init.trunc_normal_(getattr(self,'space_embed'+str(layer_id)),std=0.02)
                if self.unet_mode=='tat':
                    setattr(self,'channel_transformer'+str(layer_id),self._build_transformer(2,1))
                    setattr(self,'channel_embed'+str(layer_id),nn.Parameter(torch.zeros(1,channels,2)))
                    nn.init.trunc_normal_(getattr(self,'channel_embed'+str(layer_id)),std=0.02)
                    setattr(self,'channel_score'+str(layer_id),Linear(2,1))
                setattr(self,'time_pool'+str(layer_id),Linear(self.timestep,1))
            elif self.unet_mode in ['dam','tam']:
                d_time=channels*hight*weight
                setattr(self,'time_mamba'+str(layer_id),self.build_mamba_block(d_time))
                setattr(self,'time_embed'+str(layer_id),nn.Parameter(torch.zeros(1,self.timestep,d_time)))
                nn.init.trunc_normal_(getattr(self,'time_embed'+str(layer_id)),std=0.02)
                setattr(self,'space_mamba'+str(layer_id),self.build_mamba_block(channels))
                setattr(self,'space_embed'+str(layer_id),nn.Parameter(torch.zeros(1,hight*weight,channels)))
                nn.init.trunc_normal_(getattr(self,'space_embed'+str(layer_id)),std=0.02)
                if self.unet_mode=='tam':
                    setattr(self,'channel_mamba'+str(layer_id),self.build_mamba_block(2))
                    setattr(self,'channel_embed'+str(layer_id),nn.Parameter(torch.zeros(1,channels,2)))
                    nn.init.trunc_normal_(getattr(self,'channel_embed'+str(layer_id)),std=0.02)
                    setattr(self,'channel_score'+str(layer_id),Linear(2,1))
                setattr(self,'time_pool'+str(layer_id),Linear(self.timestep,1))

        def align_residual(self, shortcut, target):
            sh, sw = shortcut.shape[-2], shortcut.shape[-1]
            th, tw = target.shape[-2], target.shape[-1]
            if sh > th:
                dh = sh - th
                shortcut = shortcut[..., dh // 2: dh // 2 + th, :]
            if sw > tw:
                dw = sw - tw
                shortcut = shortcut[..., :, dw // 2: dw // 2 + tw]
            sh, sw = shortcut.shape[-2], shortcut.shape[-1]
            pad_h = th - sh
            pad_w = tw - sw
            if pad_h > 0 or pad_w > 0:
                pad_top = pad_h // 2
                pad_bottom = pad_h - pad_top
                pad_left = pad_w // 2
                pad_right = pad_w - pad_left
                shortcut = F.pad(shortcut,(pad_left,pad_right,pad_top,pad_bottom))
            return shortcut

        def _apply_resnet(self, x_input, x_output, layer_name):
            if self.if_resnet=='yes' and hasattr(self,'res'+layer_name):
                res_layer=getattr(self,'res'+layer_name)
                if res_layer is not None:
                    res_x=res_layer(x_input)
                else:
                    res_x=x_input
                res_x=self.align_residual(res_x,x_output)
                x_output=x_output+res_x
            return x_output

        def spatial_transformer_block(self, x, layer_id):
            B,C,H,W=x.shape
            x_space=x.permute(0,2,3,1).contiguous().reshape(B,H*W,C)
            x_space=x_space+getattr(self,'space_embed'+str(layer_id))
            x_space=getattr(self,'space_transformer'+str(layer_id))(x_space)
            x_space=x_space.reshape(B,H,W,C).permute(0,3,1,2).contiguous()
            return x_space

        def spatial_mamba_block(self, x, layer_id):
            B,C,H,W=x.shape
            x_space=x.permute(0,2,3,1).contiguous().reshape(B,H*W,C)
            x_space=x_space+getattr(self,'space_embed'+str(layer_id))
            x_space=getattr(self,'space_mamba'+str(layer_id))(x_space)
            x_space=x_space.reshape(B,H,W,C).permute(0,3,1,2).contiguous()
            return x_space

        def time_transformer_last(self, x, layer_id):
            B,T,C,H,W=x.shape
            x_time=x.contiguous().reshape(B,T,C*H*W)
            x_time=x_time+getattr(self,'time_embed'+str(layer_id))
            x_time=getattr(self,'time_transformer'+str(layer_id))(x_time)
            x_time=x_time.reshape(B,T,C,H,W)
            return x_time[:,-1,:,:,:]

        def time_mamba_last(self, x, layer_id):
            B,T,C,H,W=x.shape
            x_time=x.contiguous().reshape(B,T,C*H*W)
            x_time=x_time+getattr(self,'time_embed'+str(layer_id))
            x_time=getattr(self,'time_mamba'+str(layer_id))(x_time)
            x_time=x_time.reshape(B,T,C,H,W)
            return x_time[:,-1,:,:,:]

        def da_time_attention(self, x, layer_id):
            B,T,C,H,W=x.shape
            x_time=x.contiguous().reshape(B,T,C*H*W)
            x_time=x_time+getattr(self,'time_embed'+str(layer_id))
            x_time=getattr(self,'time_transformer'+str(layer_id))(x_time)
            x_time=x_time.reshape(B,T,C,H,W)
            return x_time

        def da_space_attention(self, x, layer_id):
            B,T,C,H,W=x.shape
            x_space=x.permute(0,1,3,4,2).contiguous().reshape(B*T,H*W,C)
            x_space=x_space+getattr(self,'space_embed'+str(layer_id))
            x_space=getattr(self,'space_transformer'+str(layer_id))(x_space)
            x_space=x_space.reshape(B,T,H,W,C).permute(0,1,4,2,3).contiguous()
            return x_space

        def da_time_mamba(self, x, layer_id):
            B,T,C,H,W=x.shape
            x_time=x.contiguous().reshape(B,T,C*H*W)
            x_time=x_time+getattr(self,'time_embed'+str(layer_id))
            x_time=getattr(self,'time_mamba'+str(layer_id))(x_time)
            x_time=x_time.reshape(B,T,C,H,W)
            return x_time

        def da_space_mamba(self, x, layer_id):
            B,T,C,H,W=x.shape
            x_space=x.permute(0,1,3,4,2).contiguous().reshape(B*T,H*W,C)
            x_space=x_space+getattr(self,'space_embed'+str(layer_id))
            x_space=getattr(self,'space_mamba'+str(layer_id))(x_space)
            x_space=x_space.reshape(B,T,H,W,C).permute(0,1,4,2,3).contiguous()
            return x_space

        def ta_channel_attention(self, x, layer_id):
            B,T,C,H,W=x.shape
            x_avg=torch.mean(x,dim=(3,4))
            x_max=torch.amax(x,dim=(3,4))
            x_channel=torch.stack((x_avg,x_max),dim=-1).contiguous().reshape(B*T,C,2)
            x_channel=x_channel+getattr(self,'channel_embed'+str(layer_id))
            x_channel=getattr(self,'channel_transformer'+str(layer_id))(x_channel)
            channel_weight=getattr(self,'channel_score'+str(layer_id))(x_channel)
            channel_weight=torch.sigmoid(channel_weight).reshape(B,T,C,1,1)
            return x*channel_weight

        def ta_channel_mamba(self, x, layer_id):
            B,T,C,H,W=x.shape
            x_avg=torch.mean(x,dim=(3,4))
            x_max=torch.amax(x,dim=(3,4))
            x_channel=torch.stack((x_avg,x_max),dim=-1).contiguous().reshape(B*T,C,2)
            x_channel=x_channel+getattr(self,'channel_embed'+str(layer_id))
            x_channel=getattr(self,'channel_mamba'+str(layer_id))(x_channel)
            channel_weight=getattr(self,'channel_score'+str(layer_id))(x_channel)
            channel_weight=torch.sigmoid(channel_weight).reshape(B,T,C,1,1)
            return x*channel_weight

        def da_time_pooling(self, x, layer_id):
            B,T,C,H,W=x.shape
            x_pool=x.contiguous().reshape(B,T,C*H*W)
            x_pool=x_pool.permute(0,2,1).contiguous()
            x_pool=getattr(self,'time_pool'+str(layer_id))(x_pool)
            x_pool=x_pool.squeeze(-1).reshape(B,C,H,W)
            return x_pool

        def dual_attention_block(self, x, layer_id):
            order=self.da_order
            if order=='time_first':
                x=self.da_time_attention(x,layer_id)
                x=self.da_space_attention(x,layer_id)
            elif order=='space_first':
                x=self.da_space_attention(x,layer_id)
                x=self.da_time_attention(x,layer_id)
            else:
                raise ValueError("da_order must be 'time_first' or 'space_first'")
            return self.da_time_pooling(x,layer_id)

        def dual_mamba_block(self, x, layer_id):
            order=self.da_order
            if order=='time_first':
                x=self.da_time_mamba(x,layer_id)
                x=self.da_space_mamba(x,layer_id)
            elif order=='space_first':
                x=self.da_space_mamba(x,layer_id)
                x=self.da_time_mamba(x,layer_id)
            else:
                raise ValueError("da_order must be 'time_first' or 'space_first'")
            return self.da_time_pooling(x,layer_id)

        def triple_attention_block(self, x, layer_id):
            order=self.ta_order
            if order=='time_first':
                order='tsc'
            elif order=='space_first':
                order='stc'
            if order not in ['cts','cst','sct','stc','tcs','tsc']:
                raise ValueError("ta_order must be one of ['cts','cst','sct','stc','tcs','tsc']")
            for axis_name in order:
                if axis_name=='c':
                    x=self.ta_channel_attention(x,layer_id)
                elif axis_name=='t':
                    x=self.da_time_attention(x,layer_id)
                elif axis_name=='s':
                    x=self.da_space_attention(x,layer_id)
            return self.da_time_pooling(x,layer_id)

        def triple_mamba_block(self, x, layer_id):
            order=self.ta_order
            if order=='time_first':
                order='tsc'
            elif order=='space_first':
                order='stc'
            if order not in ['cts','cst','sct','stc','tcs','tsc']:
                raise ValueError("ta_order must be one of ['cts','cst','sct','stc','tcs','tsc']")
            for axis_name in order:
                if axis_name=='c':
                    x=self.ta_channel_mamba(x,layer_id)
                elif axis_name=='t':
                    x=self.da_time_mamba(x,layer_id)
                elif axis_name=='s':
                    x=self.da_space_mamba(x,layer_id)
            return self.da_time_pooling(x,layer_id)

        def _make_skip(self, feature, layer_id):
            if self.unet_mode=='single':
                return feature
            elif self.unet_mode=='trans':
                if layer_id==self.downnum:
                    return self.spatial_transformer_block(feature,layer_id)
                else:
                    return feature
            elif self.unet_mode=='mamba':
                if layer_id==self.downnum:
                    return self.spatial_mamba_block(feature,layer_id)
                else:
                    return feature
            elif self.unet_mode=='stt':
                if layer_id==self.downnum:
                    return self.time_transformer_last(feature,layer_id)
                else:
                    return feature[:,-1,:,:,:]
            elif self.unet_mode=='stm':
                if layer_id==self.downnum:
                    return self.time_mamba_last(feature,layer_id)
                else:
                    return feature[:,-1,:,:,:]
            elif self.unet_mode=='dat':
                if self.if_bottom_attention_only=='yes' and layer_id!=self.downnum:
                    return feature[:,-1,:,:,:]
                else:
                    return self.dual_attention_block(feature,layer_id)
            elif self.unet_mode=='dam':
                if self.if_bottom_attention_only=='yes' and layer_id!=self.downnum:
                    return feature[:,-1,:,:,:]
                else:
                    return self.dual_mamba_block(feature,layer_id)
            elif self.unet_mode=='tat':
                if self.if_bottom_attention_only=='yes' and layer_id!=self.downnum:
                    return feature[:,-1,:,:,:]
                else:
                    return self.triple_attention_block(feature,layer_id)
            elif self.unet_mode=='tam':
                if self.if_bottom_attention_only=='yes' and layer_id!=self.downnum:
                    return feature[:,-1,:,:,:]
                else:
                    return self.triple_mamba_block(feature,layer_id)

        def forward(self, x):
            self.__dict__['x']=x
            skip_features=[]
            current_name='x'
            for i in range(int(self.downnum)):
                for j in range(int(self.covnum)):
                    layer_id=str(i+1)+'_'+str(j+1)
                    input_tensor=eval(current_name, globals(), self.__dict__)
                    self.__dict__['model_res_input'+layer_id]=input_tensor
                    exec('model_conv'+layer_id+'=self.conv'+layer_id+'(model_res_input'+layer_id+')', globals(), self.__dict__)
                    exec('model_pad'+layer_id+'=self.pad'+layer_id+'(model_conv'+layer_id+')', globals(), self.__dict__)
                    exec('model_act'+layer_id+'=self.act'+layer_id+'(model_pad'+layer_id+')', globals(), self.__dict__)
                    if self.ifnormalization=='yes':
                        exec('model_norm'+layer_id+'=self.norm'+layer_id+'(model_act'+layer_id+')', globals(), self.__dict__)
                        output_tensor=eval('model_norm'+layer_id, globals(), self.__dict__)
                    else:
                        output_tensor=eval('model_act'+layer_id, globals(), self.__dict__)
                    output_tensor=self._apply_resnet(input_tensor,output_tensor,layer_id)
                    self.__dict__['model_feature'+layer_id]=output_tensor
                    current_name='model_feature'+layer_id
                feature=eval(current_name, globals(), self.__dict__)
                skip_feature=self._make_skip(feature,i+1)
                self.__dict__['model_skip'+str(i+1)]=skip_feature
                skip_features.append(skip_feature)
                if i != int(self.downnum)-1:
                    self.__dict__['model_pool_input'+str(i+1)]=feature
                    exec('model_pool'+str(i+1)+'=self.pool'+str(i+1)+'(model_pool_input'+str(i+1)+')', globals(), self.__dict__)
                    current_name='model_pool'+str(i+1)
            decoder_x=skip_features[-1]
            decoder_index=0
            for skip_level in range(int(self.downnum)-2,-1,-1):
                decoder_index=decoder_index+1
                m=int(self.downnum)+decoder_index
                self.__dict__['decoder_x']=decoder_x
                exec('model_res_input'+str(m)+'=decoder_x', globals(), self.__dict__)
                exec('model_conv'+str(m)+'=self.conv'+str(m)+'(model_res_input'+str(m)+')', globals(), self.__dict__)
                exec('model_pad'+str(m)+'=self.pad'+str(m)+'(model_conv'+str(m)+')', globals(), self.__dict__)
                exec('model_act'+str(m)+'=self.act'+str(m)+'(model_pad'+str(m)+')', globals(), self.__dict__)
                if self.ifnormalization=='yes':
                    exec('model_norm'+str(m)+'=self.norm'+str(m)+'(model_act'+str(m)+')', globals(), self.__dict__)
                    decoder_x=eval('model_norm'+str(m), globals(), self.__dict__)
                else:
                    decoder_x=eval('model_act'+str(m), globals(), self.__dict__)
                decoder_x=self._apply_resnet(eval('model_res_input'+str(m), globals(), self.__dict__),decoder_x,str(m))
                self.__dict__['decoder_x']=decoder_x
                exec('model_up'+str(m)+'=self.up'+str(m)+'(decoder_x)', globals(), self.__dict__)
                up_x=eval('model_up'+str(m), globals(), self.__dict__)
                skip_x=skip_features[skip_level]
                up_x=self.align_residual(up_x,skip_x)
                self.__dict__['up_x']=up_x
                self.__dict__['skip_x']=skip_x
                exec('model_conc'+str(m)+'=torch.concatenate((up_x,skip_x),axis=1)', globals(), self.__dict__)
                current_name='model_conc'+str(m)
                for n in range(int(self.covnum)):
                    layer_id=str(m)+'_'+str(n+1)
                    input_tensor=eval(current_name, globals(), self.__dict__)
                    self.__dict__['model_res_input'+layer_id]=input_tensor
                    exec('model_conv'+layer_id+'=self.conv'+layer_id+'(model_res_input'+layer_id+')', globals(), self.__dict__)
                    exec('model_pad'+layer_id+'=self.pad'+layer_id+'(model_conv'+layer_id+')', globals(), self.__dict__)
                    exec('model_act'+layer_id+'=self.act'+layer_id+'(model_pad'+layer_id+')', globals(), self.__dict__)
                    if self.ifnormalization=='yes':
                        exec('model_norm'+layer_id+'=self.norm'+layer_id+'(model_act'+layer_id+')', globals(), self.__dict__)
                        decoder_x=eval('model_norm'+layer_id, globals(), self.__dict__)
                    else:
                        decoder_x=eval('model_act'+layer_id, globals(), self.__dict__)
                    decoder_x=self._apply_resnet(input_tensor,decoder_x,layer_id)
                    self.__dict__['model_feature'+layer_id]=decoder_x
                    current_name='model_feature'+layer_id
            if if_last_act!='no':
                self.__dict__['decoder_x']=decoder_x
                exec('model_convlast=self.convlast(decoder_x)', globals(), self.__dict__)
                outputs=eval('self.actlast(model_convlast)', globals(), self.__dict__)
            else:
                outputs=self.convlast(decoder_x)
            return outputs

    def _make_model():
        return Model(trainx[0:2],trainy[0:2],timestep,unet_mode,downnum,covnum,layer_time,baselayer,cov_kernelsize,cov_strides,pool_method,pool_kernel_size,pool_strides,ifnormalization,normalization_method,trans_num,num_heads,dim_feedforward,trans_dropout_rate,mamba_class,mamba_state,mamba_conv,mamba_expand,activate,nlp_activate,da_order,ta_order,if_bottom_attention_only,if_resnet,if_bottom_res_only)

    if k_fold!=None:
        models=[]
        for i in range(k_fold):
            models.append(_make_model())
    else:
        model=_make_model()
    if if_best_mode!='no':
        if k_fold!=None:
            for i in range(k_fold):
                models[i].load_state_dict(torch.load(modelpath+'_'+str(i+1)+'.pth', map_location=devices))
        else:
            model.load_state_dict(torch.load(modelpath+'.pth', map_location=devices))
    if k_fold!=None:
        for i in range(k_fold):
            models[i].to(devices)
    else:
        model.to(devices)
    if k_fold!=None:
        opts=[]
        for i in range(k_fold):
            if optimizer == 'SGD':
                opts.append(SGD(models[i].parameters(), lr=learning_rate))
            elif optimizer == 'Adam':
                opts.append(Adam(models[i].parameters(), lr=learning_rate))
            elif optimizer == 'Nadam':
                opts.append(NAdam(models[i].parameters(), lr=learning_rate))
    else:
        if optimizer == 'SGD':
            opt = SGD(model.parameters(), lr=learning_rate)
        elif optimizer == 'Adam':
            opt = Adam(model.parameters(), lr=learning_rate)
        elif optimizer == 'Nadam':
            opt = NAdam(model.parameters(), lr=learning_rate)
    if if_best_mode!='no':
        if k_fold!=None:
            for i in range(k_fold):
                opts[i].load_state_dict(torch.load(modelpath+'_'+str(i+1)+'_opt.pth', map_location=devices))
        else:
            opt.load_state_dict(torch.load(modelpath+'_opt.pth', map_location=devices))

    if k_fold!=None:
        schedulers=[]
        for i in range(k_fold):
            if lr_decay is None:
                schedulers.append(None)
            else:
                schedulers.append(ExponentialLR(opts[i], gamma=float(lr_decay)))
    else:
        if lr_decay is None:
            scheduler=None
        else:
            scheduler=ExponentialLR(opt, gamma=float(lr_decay))

    if if_print_model=='yes':
        if k_fold!=None:
            print(models[0])
        else:
            print(model)

    def _calculate_loss_metric(current_output, current_y):
        if task_mode=='multi_classify':
            y_tensor=torch.tensor(current_y,dtype=torch.long,device=devices).squeeze(1)
            if loss_function=='NLLLoss':
                current_loss=torch.mean(loss(torch.log(current_output).contiguous(), y_tensor))
            else:
                current_loss=torch.mean(loss(current_output, y_tensor))
            if metrics in ['default','accuracy','f1','recall','precision']:
                if torchmetrics is not None and not isinstance(metric, torch.nn.Module):
                    current_metric=torch.mean(metric(torch.argmax(current_output, dim=1), y_tensor))
                elif torchmetrics is not None and metric.__class__.__module__.startswith('torchmetrics'):
                    current_metric=torch.mean(metric(torch.argmax(current_output, dim=1), y_tensor))
                else:
                    current_metric=torch.mean(torch.tensor(accuracy_score(y_tensor.detach().cpu().numpy().reshape(-1), torch.argmax(current_output, dim=1).detach().cpu().numpy().reshape(-1)),dtype=torch.float32,device=devices))
            elif metrics=='NLLLoss':
                current_metric=torch.mean(metric(torch.log(current_output).contiguous(), y_tensor))
            else:
                current_metric=torch.mean(metric(current_output, y_tensor))
        else:
            y_tensor=torch.tensor(current_y,dtype=torch.float32,device=devices)
            current_loss=torch.mean(loss(current_output, y_tensor))
            try:
                current_metric=torch.mean(metric(current_output, y_tensor))
            except Exception:
                current_metric=current_loss
        return current_loss,current_metric

    if ifrandom_split!='just_model':
        trainx=np.nan_to_num(trainx,nan=0)
        testx=np.nan_to_num(testx,nan=0)
        if ifrandom_split!='all_test':
            if k_fold!=None:
                kf = KFold(n_splits=k_fold, shuffle=True, random_state=25)
                for fold_no, (train_idx, val_idx) in enumerate(kf.split(trainx, trainy)):
                    X_train_fold, y_train_fold = trainx[train_idx], trainy[train_idx]
                    X_val_fold, y_val_fold = trainx[val_idx], trainy[val_idx]
                    if fold_no==0:
                        train_loss=np.zeros((k_fold,int(np.ceil(X_train_fold.shape[0]/batch_size))))
                        train_metric=np.zeros((k_fold,int(np.ceil(X_train_fold.shape[0]/batch_size))))
                        test_loss=np.zeros((k_fold,int(np.ceil(X_val_fold.shape[0]/batch_size))))
                        test_metric=np.zeros((k_fold,int(np.ceil(X_val_fold.shape[0]/batch_size))))
                    if if_early_stopping!=None:
                        early_stopping = EarlyStopping(patience=if_early_stopping)
                    for i in range(epochs):
                        models[fold_no].train()
                        start = datetime.datetime.now()
                        for j in range(int(np.ceil(X_train_fold.shape[0]/batch_size))):
                            if j == int(X_train_fold.shape[0]/batch_size):
                                batch_x=X_train_fold[j*batch_size:]
                                batch_y=y_train_fold[j*batch_size:]
                            else:
                                batch_x=X_train_fold[j*batch_size:(j+1)*batch_size]
                                batch_y=y_train_fold[j*batch_size:(j+1)*batch_size]
                            train_output = models[fold_no](torch.tensor(batch_x,dtype=torch.float32,device=devices))
                            train_losss,train_metrics=_calculate_loss_metric(train_output,batch_y)
                            train_losss = add_regularization_loss(train_losss, models[fold_no])
                            train_loss[fold_no,j]=np.array(train_losss.item())
                            train_metric[fold_no,j]=np.array(train_metrics.item())
                            opts[fold_no].zero_grad()
                            train_losss.backward()
                            opts[fold_no].step()
                        models[fold_no].eval()
                        with torch.no_grad():
                            for k in range(int(np.ceil(X_val_fold.shape[0]/batch_size))):
                                if k == int(X_val_fold.shape[0]/batch_size):
                                    batch_x=X_val_fold[k*batch_size:]
                                    batch_y=y_val_fold[k*batch_size:]
                                else:
                                    batch_x=X_val_fold[k*batch_size:(k+1)*batch_size]
                                    batch_y=y_val_fold[k*batch_size:(k+1)*batch_size]
                                test_output = models[fold_no](torch.tensor(batch_x,dtype=torch.float32,device=devices))
                                test_losss,test_metrics=_calculate_loss_metric(test_output,batch_y)
                                test_loss[fold_no,k]=np.array(test_losss.item())
                                test_metric[fold_no,k]=np.array(test_metrics.item())
                        if if_early_stopping!=None:
                            early_stopping(np.nanmean(test_loss[fold_no,:]), models[fold_no])
                            if early_stopping.early_stop:
                                early_stopping.load_best_checkpoint(models[fold_no])
                                break
                        if schedulers[fold_no] is not None:
                            schedulers[fold_no].step()
                        end = datetime.datetime.now()
                        print('第',i+1,'次训练loss:',np.nanmean(train_loss[fold_no,:]),'，metric:',np.nanmean(train_metric[fold_no,:]),'  第',i+1,'次测试loss:',np.nanmean(test_loss[fold_no,:]),'，metric：',np.nanmean(test_metric[fold_no,:]),'训练用时:',end - start)
            else:
                if valid_size!=None:
                    if ifrandom_split=='yes':
                        trainy,validy,trainx,validx = train_test_split(trainy,trainx,test_size=valid_size/(1-test_size),random_state=25)
                    else:
                        index=int((1-valid_size/(1-test_size))*trainy.shape[0])
                        validy=trainy[index:]
                        trainy=trainy[:index]
                        validx=trainx[index:]
                        trainx=trainx[:index]
                else:
                    validy=testy
                    validx=testx
                if if_early_stopping!=None:
                    early_stopping = EarlyStopping(patience=if_early_stopping)
                for i in range(epochs):
                    model.train()
                    start = datetime.datetime.now()
                    train_loss=np.zeros((int(np.ceil(trainx.shape[0]/batch_size))))
                    train_metric=np.zeros((int(np.ceil(trainx.shape[0]/batch_size))))
                    test_loss=np.zeros((int(np.ceil(validx.shape[0]/batch_size))))
                    test_metric=np.zeros((int(np.ceil(validx.shape[0]/batch_size))))
                    for j in range(int(np.ceil(trainx.shape[0]/batch_size))):
                        if j == int(trainx.shape[0]/batch_size):
                            batch_x=trainx[j*batch_size:]
                            batch_y=trainy[j*batch_size:]
                        else:
                            batch_x=trainx[j*batch_size:(j+1)*batch_size]
                            batch_y=trainy[j*batch_size:(j+1)*batch_size]
                        train_output = model(torch.tensor(batch_x,dtype=torch.float32,device=devices))
                        train_losss,train_metrics=_calculate_loss_metric(train_output,batch_y)
                        train_losss = add_regularization_loss(train_losss, model)
                        train_loss[j]=np.array(train_losss.item())
                        train_metric[j]=np.array(train_metrics.item())
                        opt.zero_grad()
                        train_losss.backward()
                        opt.step()
                    model.eval()
                    with torch.no_grad():
                        for k in range(int(np.ceil(validx.shape[0]/batch_size))):
                            if k == int(validx.shape[0]/batch_size):
                                batch_x=validx[k*batch_size:]
                                batch_y=validy[k*batch_size:]
                            else:
                                batch_x=validx[k*batch_size:(k+1)*batch_size]
                                batch_y=validy[k*batch_size:(k+1)*batch_size]
                            test_output = model(torch.tensor(batch_x,dtype=torch.float32,device=devices))
                            test_losss,test_metrics=_calculate_loss_metric(test_output,batch_y)
                            test_loss[k]=np.array(test_losss.item())
                            test_metric[k]=np.array(test_metrics.item())
                    if if_early_stopping!=None:
                        early_stopping(np.nanmean(test_loss), model)
                        if early_stopping.early_stop:
                            early_stopping.load_best_checkpoint(model)
                            break
                    if scheduler is not None:
                        scheduler.step()
                    end = datetime.datetime.now()
                    print('第',i+1,'次训练loss:',np.nanmean(train_loss),'，metric:',np.nanmean(train_metric),'  第',i+1,'次测试loss:',np.nanmean(test_loss),'，metric：',np.nanmean(test_metric),'训练用时:',end - start)
        if k_fold!=None:
            if task_mode=='multi_classify':
                predicty=np.zeros((k_fold,testy.shape[0],int(np.max(vy))+1,testy.shape[2],testy.shape[3]))
            else:
                predicty=np.zeros((k_fold,testy.shape[0],testy.shape[1],testy.shape[2],testy.shape[3]))
        else:
            if task_mode=='multi_classify':
                predicty=np.zeros((testy.shape[0],int(np.max(vy))+1,testy.shape[2],testy.shape[3]))
            else:
                predicty=np.zeros((testy.shape[0],testy.shape[1],testy.shape[2],testy.shape[3]))
        if k_fold!=None:
            for i in range(k_fold):
                models[i].eval()
                with torch.no_grad():
                    for k in range(int(np.ceil(testx.shape[0]/batch_size))):
                        if k == int(testx.shape[0]/batch_size):
                            predicty_batch = models[i](torch.tensor(testx[k*batch_size:],dtype=torch.float32,device=devices))
                            predicty[i,k*batch_size:]=predicty_batch.cpu().detach().numpy()
                        else:
                            predicty_batch = models[i](torch.tensor(testx[k*batch_size:(k+1)*batch_size],dtype=torch.float32,device=devices))
                            predicty[i,k*batch_size:(k+1)*batch_size]=predicty_batch.cpu().detach().numpy()
            predicty=np.nanmean(predicty,axis=0)
        else:
            model.eval()
            with torch.no_grad():
                for k in range(int(np.ceil(testx.shape[0]/batch_size))):
                    if k == int(testx.shape[0]/batch_size):
                        predicty_batch = model(torch.tensor(testx[k*batch_size:],dtype=torch.float32,device=devices))
                        predicty[k*batch_size:]=predicty_batch.cpu().detach().numpy()
                    else:
                        predicty_batch = model(torch.tensor(testx[k*batch_size:(k+1)*batch_size],dtype=torch.float32,device=devices))
                        predicty[k*batch_size:(k+1)*batch_size]=predicty_batch.cpu().detach().numpy()
        predicty = np.nan_to_num(predicty,nan=0)
        r=np.zeros((testy.shape[2],testy.shape[3],testy.shape[1]))
        p=np.zeros((testy.shape[2],testy.shape[3],testy.shape[1]))
        f1score=np.zeros((testy.shape[2],testy.shape[3],testy.shape[1]))
        accuracyscore=np.zeros((testy.shape[2],testy.shape[3],testy.shape[1]))
        recallscore=np.zeros((testy.shape[2],testy.shape[3],testy.shape[1]))
        precisionscore=np.zeros((testy.shape[2],testy.shape[3],testy.shape[1]))
        for i in range(testy.shape[2]):
            for j in range(testy.shape[3]):
                for k in range(testy.shape[1]):
                    if task_mode=='regression':
                        try:
                            r[i,j,k],p[i,j,k] = pearsonr(predicty[:,k,i,j],testy[:,k,i,j])
                        except Exception:
                            r[i,j,k],p[i,j,k] = 0,0
                        r[i,j,k]=np.nan_to_num(r[i,j,k],nan=0)
                    elif task_mode=='binary_classify':
                        predicty[:,k,i,j]=[int(round(predicty[l,k,i,j],0)) for l in range(predicty.shape[0])]
                        if metrics=='Recall':
                            r[i,j,k]=recall_score(testy[:,k,i,j], predicty[:,k,i,j])
                        elif metrics=='Precision':
                            r[i,j,k]=precision_score(testy[:,k,i,j], predicty[:,k,i,j])
                        else:
                            r[i,j,k]=accuracy_score(testy[:,k,i,j], predicty[:,k,i,j])
                        p[i,j,k]=0
                        f1score[i,j,k]=f1_score(testy[:,k,i,j],predicty[:,k,i,j])
                        accuracyscore[i,j,k]=accuracy_score(testy[:,k,i,j],predicty[:,k,i,j])
                        recallscore[i,j,k]=recall_score(testy[:,k,i,j],predicty[:,k,i,j])
                        precisionscore[i,j,k]=precision_score(testy[:,k,i,j],predicty[:,k,i,j])
                    elif task_mode=='multi_classify':
                        r[i,j,k]=accuracy_score(testy[:,k,i,j], np.argmax(predicty[:,:,i,j],axis=1))
                        p[i,j,k]=0
        if ifmute == 'no':
            if task_mode=='regression':
                print('相关系数',np.nanmean(r))
            elif task_mode=='binary_classify':
                print('召回率+精确率',np.nanmean(f1score),'准确率',np.nanmean(accuracyscore),'召回率',np.nanmean(recallscore),'精确率',np.nanmean(precisionscore))
            elif task_mode=='multi_classify':
                print('准确率',np.nanmean(r))

        should_run_shap_explainer = False
        def _compute_shap_values_for_model_for_channel(model_instance, current_shap_data, x_sample_shape, output_channel_idx, devices):
            if shap is None:
                raise ImportError('shap is not installed.')
            wrapped_model = ShapWrapper(model_instance, x_sample_shape, output_channel_idx=output_channel_idx).to(devices)
            wrapped_model.eval()
            background_data = current_shap_data
            explainer = shap.GradientExplainer(wrapped_model, background_data)
            raw_shap_values = explainer.shap_values(current_shap_data)
            if isinstance(raw_shap_values, list):
                raw_shap_values = raw_shap_values[0]
            shap_x_reshaped = raw_shap_values.reshape(raw_shap_values.shape[0], *x_sample_shape)
            return np.abs(shap_x_reshaped)
        if ifheatmap == 'yes' or ifweight in ['yes', 'shap']:
            should_run_shap_explainer = True
        if should_run_shap_explainer:
            current_testx_data = testx
            if testx.shape[0] >= 100:
                index = np.random.randint(0, testx.shape[0], size=100)
                current_testx_data = testx[index,]
            x_sample_shape = current_testx_data.shape[1:]
            flat_current_testx_tensor = torch.tensor(current_testx_data, dtype=torch.float32, device=devices).flatten(start_dim=1)
            if task_mode=='multi_classify':
                num_model_outputs = int(np.max(vy))+1
            else:
                num_model_outputs = testy.shape[1]
            shap_values_per_output_channel = []
            for output_ch_idx in range(num_model_outputs):
                if k_fold!=None:
                    fold_shap_results_for_channel = []
                    for model_k_fold in models:
                        model_k_fold.eval()
                        fold_shap_results_for_channel.append(_compute_shap_values_for_model_for_channel(model_k_fold, flat_current_testx_tensor, x_sample_shape, output_ch_idx, devices))
                    shap_values_per_output_channel.append(np.nanmean(np.stack(fold_shap_results_for_channel, axis=0), axis=0))
                else:
                    model.eval()
                    shap_values_per_output_channel.append(_compute_shap_values_for_model_for_channel(model, flat_current_testx_tensor, x_sample_shape, output_ch_idx, devices))
            shap_values_aggregated = np.stack(shap_values_per_output_channel, axis=0)
            if ifheatmap=='yes':
                if testx.ndim==5:
                    heatmap_per_output_channel = np.nanmean(shap_values_aggregated, axis=(1,2))
                    heatmap_output_list = []
                    for i in range(num_model_outputs):
                        current_heatmap_slice = heatmap_per_output_channel[i,:,:,:]
                        current_heatmap_slice = current_heatmap_slice.transpose(1,2,0)
                        heatmap_output_list.append(current_heatmap_slice)
                    if len(heatmap_output_list) > 0 and all(h.shape == heatmap_output_list[0].shape for h in heatmap_output_list):
                        heatmap = np.stack(heatmap_output_list, axis=0)
                    else:
                        heatmap = heatmap_output_list
                else:
                    heatmap_per_output_channel = np.nanmean(shap_values_aggregated, axis=1)
                    heatmap_output_list = []
                    for i in range(num_model_outputs):
                        current_heatmap_slice = heatmap_per_output_channel[i,:,:,:]
                        current_heatmap_slice = current_heatmap_slice.transpose(1,2,0)
                        heatmap_output_list.append(current_heatmap_slice)
                    if len(heatmap_output_list) > 0 and all(h.shape == heatmap_output_list[0].shape for h in heatmap_output_list):
                        heatmap = np.stack(heatmap_output_list, axis=0)
                    else:
                        heatmap = heatmap_output_list
            if ifweight in ['yes','shap']:
                if testx.ndim==5:
                    weights_aggregated_per_channel_feature = np.nanmean(shap_values_aggregated, axis=(1,2,4,5))
                    input_channel_for_weights = trainx.shape[2]
                else:
                    weights_aggregated_per_channel_feature = np.nanmean(shap_values_aggregated, axis=(1,3,4))
                    input_channel_for_weights = trainx.shape[1]
                weights_final_output = np.zeros((num_model_outputs, input_channel_for_weights))
                for i in range(num_model_outputs):
                    current_output_channel_weights = weights_aggregated_per_channel_feature[i,:]
                    sum_for_norm = np.nansum(current_output_channel_weights)
                    if sum_for_norm != 0:
                        weights_final_output[i,:] = (current_output_channel_weights / sum_for_norm) * 100
                    else:
                        weights_final_output[i,:] = 0.0
                    if ifmute == 'no':
                        for j in range(input_channel_for_weights):
                            print('预报因子',j+1,'对预报值',i+1,'的贡献：',np.array(weights_final_output[i,j]),'％')
                        print('\n')
                weights = weights_final_output
        elif ifweight == 'oob':
            input_feature_axis=2 if testx.ndim==5 else 1
            input_channel_count=testx.shape[input_feature_axis]
            weights = np.zeros((testy.shape[1],input_channel_count))
            weight_more=np.zeros((testy.shape[1],input_channel_count))
            for i in tqdm(range(testy.shape[1])):
                for j in range(input_channel_count):
                    testx_new=copy.deepcopy(testx)
                    weight_current_feature_output_pair = []
                    for k in range(10):
                        per=np.random.permutation(testx.shape[0])
                        if testx.ndim==5:
                            testx_shuffle=testx[per,:,j,:,:]
                            testx_new[:,:,j,:,:]=testx_shuffle
                        else:
                            testx_shuffle=testx[per,j,:,:]
                            testx_new[:,j,:,:]=testx_shuffle
                        predicty_new_agg = None
                        if k_fold!=None:
                            predicty_new_folds = []
                            for o in range(k_fold):
                                predicty_new_batch_list = []
                                for n in range(int(np.ceil(testx.shape[0]/batch_size))):
                                    if n == int(testx.shape[0]/batch_size):
                                        current_batch_x = torch.tensor(testx_new[n*batch_size:],dtype=torch.float32,device=devices)
                                    else:
                                        current_batch_x = torch.tensor(testx_new[n*batch_size:(n+1)*batch_size],dtype=torch.float32,device=devices)
                                    predicty_new_batch = models[o](current_batch_x)
                                    predicty_new_batch_list.append(predicty_new_batch.cpu().detach().numpy())
                                predicty_new_folds.append(np.concatenate(predicty_new_batch_list, axis=0))
                            predicty_new_agg = np.nanmean(np.stack(predicty_new_folds, axis=0), axis=0)
                        else:
                            predicty_new_batch_list = []
                            for n in range(int(np.ceil(testx.shape[0]/batch_size))):
                                if n == int(testx.shape[0]/batch_size):
                                    current_batch_x = torch.tensor(testx_new[n*batch_size:],dtype=torch.float32,device=devices)
                                else:
                                    current_batch_x = torch.tensor(testx_new[n*batch_size:(n+1)*batch_size],dtype=torch.float32,device=devices)
                                predicty_new_batch = model(current_batch_x)
                                predicty_new_batch_list.append(predicty_new_batch.cpu().detach().numpy())
                            predicty_new_agg = np.concatenate(predicty_new_batch_list, axis=0)
                        for l in range(testy.shape[2]):
                            for m in range(testy.shape[3]):
                                if task_mode=='regression':
                                    current_mse_original = sklearn.metrics.mean_squared_error(testy[:,i,l,m],predicty[:,i,l,m])
                                    current_mse_shuffled = sklearn.metrics.mean_squared_error(testy[:,i,l,m],predicty_new_agg[:,i,l,m])
                                    weight_current_feature_output_pair.append(current_mse_shuffled - current_mse_original)
                                elif task_mode=='multi_classify':
                                    current_logloss_original = sklearn.metrics.log_loss(testy[:,i,l,m],predicty[:,:,l,m])
                                    current_logloss_shuffled = sklearn.metrics.log_loss(testy[:,i,l,m],predicty_new_agg[:,:,l,m])
                                    weight_current_feature_output_pair.append(current_logloss_shuffled - current_logloss_original)
                                else:
                                    current_logloss_original = sklearn.metrics.log_loss(testy[:,i,l,m],predicty[:,i,l,m])
                                    current_logloss_shuffled = sklearn.metrics.log_loss(testy[:,i,l,m],predicty_new_agg[:,i,l,m])
                                    weight_current_feature_output_pair.append(current_logloss_shuffled - current_logloss_original)
                    weight_more[i,j]=np.nanmean(weight_current_feature_output_pair)
            for i in range(testy.shape[1]):
                current_output_channel_weights_oob = weight_more[i,:]
                sum_for_norm = np.nansum(current_output_channel_weights_oob)
                if sum_for_norm != 0:
                    weights[i,:] = (current_output_channel_weights_oob / sum_for_norm) * 100
                else:
                    weights[i,:] = 0.0
                if ifmute == 'no':
                    for j in range(input_channel_count):
                        print('预报因子',j+1,'对预报值',i+1,'的贡献：',np.array(weights[i,j]),'％')
                    print('\n')

    if ifsave=='yes':
        if savepath is None:
            raise ValueError('savepath must not be None when ifsave=="yes".')
        if k_fold!=None:
            for i in range(k_fold):
                torch.save(models[i].state_dict(),savepath+'_'+str(i+1)+'.pth')
                torch.save(opts[i].state_dict(),savepath+'_'+str(i+1)+'_opt.pth')
        else:
            torch.save(model.state_dict(),savepath+'.pth')
            torch.save(opt.state_dict(),savepath+'_opt.pth')
    if k_fold!=None:
        return models,predicty,testy,r,p,heatmap,weights
    else:
        return model,predicty,testy,r,p,heatmap,weights
