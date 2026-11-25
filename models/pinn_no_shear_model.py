def Auto_DA_TransUnet_pytorch(vy,vx,timestep,downnum,covnum,test_size=0.2,valid_size=0.1,k_fold=None,task_mode='regression',if_best_mode='no',modelpath=None,ifrandom_split='yes',if_bottom_attention_only='no',if_resnet='no',layer_time=2,baselayer=64,cov_kernelsize=3,cov_strides=1,pool_method='maxpooling',pool_kernel_size=2,pool_strides=2,ifnormalization='no',normalization_method=None,da_attention_order='space_first',trans_num=2,num_heads=2,dim_feedforward=256,trans_dropout_rate=0.0,activate='tanh',nlp_activate='relu',if_last_act='no',if_print_model='yes',loss_function='default',optimizer='SGD',metrics='default',learning_rate=0.01,epochs=2000,batch_size=20,if_early_stopping=None,ifheatmap='no',ifweight='yes',ifmute='no',ifsave='no',savepath=None,device='cpu',pinn_vy_mean=None,pinn_vy_std=None,pinn_physics_weight=0.01,pinn_lat=None,pinn_lon=None,pinn_pressure_levels=None,pinn_w_external_scale=-0.1,pinn_lambda_continuity=611.7192928848946,pinn_lambda_vorticity=142.40845542406902,pinn_lambda_shear=0.0):
    from torch.nn import Module,BatchNorm1d,BatchNorm2d,LayerNorm,Conv2d,MaxPool2d,AvgPool2d,Dropout,LeakyReLU,ReLU,PReLU,Sigmoid,Tanh,ELU,Softmax,Linear,Flatten,ZeroPad2d,Upsample,TransformerEncoderLayer,TransformerEncoder
    from torch.optim import Adam,NAdam,SGD
    import torch
    from torch import nn
    import torchmetrics
    import math
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import KFold
    import numpy as np
    from scipy.stats import pearsonr
    import os
    from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score
    import sklearn
    import copy
    import shap
    import datetime
    import warnings
    from tqdm import tqdm
    warnings.filterwarnings('ignore')

    class PINNLoss(nn.Module):
        def __init__(self,vy_mean,vy_std,lambda_continuity=611.7192928848946,lambda_vorticity=142.40845542406902,lambda_shear=0.0,lat=None,lon=None,pressure_levels=None,w_external_scale=-0.1):
            super().__init__()
            if vy_mean is None or vy_std is None:
                raise ValueError(
                    '使用 PINNLoss 时必须传入 pinn_vy_mean 和 pinn_vy_std。'
                    '它们必须是函数外对完整、未标准化 vy 做 z-score 时实际使用的均值和标准差。'
                    '仅凭已经标准化后的 vy 无法恢复物理量。'
                )
            self.vy_mean_np=np.asarray(vy_mean,dtype=np.float32)
            self.vy_std_np=np.asarray(vy_std,dtype=np.float32)
            self.lambda_continuity=float(lambda_continuity)
            self.lambda_vorticity=float(lambda_vorticity)
            self.lambda_shear=float(lambda_shear)
            for _name,_value in [
                ('pinn_lambda_continuity',self.lambda_continuity),
                ('pinn_lambda_vorticity',self.lambda_vorticity),
                ('pinn_lambda_shear',self.lambda_shear),
            ]:
                if (not np.isfinite(_value)) or _value<0:
                    raise ValueError(_name+' 必须为有限且 >= 0 的数值。')
            self.lat_np=None if lat is None else np.asarray(lat,dtype=np.float32).reshape(-1)
            self.lon_np=None if lon is None else np.asarray(lon,dtype=np.float32).reshape(-1)
            self.pressure_levels_np=None if pressure_levels is None else np.asarray(pressure_levels,dtype=np.float32).reshape(-1)
            self.w_external_scale=float(w_external_scale)
            if (not np.isfinite(self.w_external_scale)) or abs(self.w_external_scale)<1e-12:
                raise ValueError('pinn_w_external_scale 不能为 0 或非有限数。')
            self.earth_radius=6371000.0
            self._cache_key=None
            self._mean_tensor=None
            self._std_tensor=None
            self.last_components=None

        def _stat_to_chw(self,stat,c,h,w,device,dtype,is_std=False):
            arr=np.asarray(stat,dtype=np.float32)
            arr=np.squeeze(arr)
            if arr.ndim==3:
                if arr.shape==(c,h,w):
                    chw=arr
                elif arr.shape==(h,w,c):
                    chw=np.transpose(arr,(2,0,1))
                elif arr.size==c:
                    chw=arr.reshape(c,1,1)*np.ones((c,h,w),dtype=np.float32)
                else:
                    raise ValueError('pinn_vy_mean/std 形状与输出不匹配: '+str(arr.shape))
            elif arr.ndim==1:
                if arr.size==c:
                    chw=arr.reshape(c,1,1)*np.ones((c,h,w),dtype=np.float32)
                elif arr.size==1:
                    chw=np.full((c,h,w),float(arr[0]),dtype=np.float32)
                else:
                    raise ValueError('一维 pinn_vy_mean/std 长度必须等于输出通道数。')
            elif arr.ndim==0:
                chw=np.full((c,h,w),float(arr),dtype=np.float32)
            else:
                raise ValueError('pinn_vy_mean/std 仅支持 scalar、(C,)、(H,W,C) 或 (C,H,W)。当前: '+str(arr.shape))
            if is_std:
                chw=np.where(np.isfinite(chw)&(np.abs(chw)>1e-12),chw,1.0)
            else:
                chw=np.where(np.isfinite(chw),chw,0.0)
            return torch.as_tensor(chw,dtype=dtype,device=device).unsqueeze(0)

        def _get_mean_std(self,y_pred):
            _,c,h,w=y_pred.shape
            key=(c,h,w,str(y_pred.device),str(y_pred.dtype))
            if self._cache_key!=key:
                self._mean_tensor=self._stat_to_chw(self.vy_mean_np,c,h,w,y_pred.device,y_pred.dtype,is_std=False)
                self._std_tensor=self._stat_to_chw(self.vy_std_np,c,h,w,y_pred.device,y_pred.dtype,is_std=True)
                self._cache_key=key
            return self._mean_tensor,self._std_tensor

        def _get_coordinates(self,h,w,nlevel,device,dtype):
            if self.lat_np is None:
                lat_np=np.linspace(36.0,24.0,h,dtype=np.float32)
            else:
                lat_np=self.lat_np
                if lat_np.size!=h:
                    raise ValueError('pinn_lat 长度与模型纬度维不一致。')
            if self.lon_np is None:
                lon_np=np.linspace(108.0,126.0,w,dtype=np.float32)
            else:
                lon_np=self.lon_np
                if lon_np.size!=w:
                    raise ValueError('pinn_lon 长度与模型经度维不一致。')
            if self.pressure_levels_np is None:
                if nlevel==3:
                    p_np=np.array([850.0,925.0,1000.0],dtype=np.float32)
                else:
                    raise ValueError('默认 PINN loss 仅对应 850/925/1000 hPa 三层；其他层请传 pinn_pressure_levels。')
            else:
                p_np=self.pressure_levels_np
                if p_np.size!=nlevel:
                    raise ValueError('pinn_pressure_levels 长度与每个风分量的层数不一致。')
            if np.nanmax(np.abs(p_np))<2000.0:
                p_np=p_np*100.0
            lat=torch.as_tensor(lat_np,dtype=dtype,device=device)
            lon=torch.as_tensor(lon_np,dtype=dtype,device=device)
            pressure=torch.as_tensor(p_np,dtype=dtype,device=device)
            return lat,lon,pressure

        def _ddx(self,field,lat,lon):
            if field.shape[-1]<2:
                raise ValueError('经度维至少需要 2 个格点。')
            lat_rad=torch.deg2rad(lat)
            lon_rad=torch.deg2rad(lon)
            coslat=torch.cos(lat_rad)
            coslat=torch.where(torch.abs(coslat)<1e-8,torch.full_like(coslat,1e-8),coslat)
            out=torch.empty_like(field)
            dx0=(self.earth_radius*coslat*(lon_rad[1]-lon_rad[0])).view(1,1,-1)
            dx1=(self.earth_radius*coslat*(lon_rad[-1]-lon_rad[-2])).view(1,1,-1)
            out[:,:,:,0]=(field[:,:,:,1]-field[:,:,:,0])/dx0
            out[:,:,:,-1]=(field[:,:,:,-1]-field[:,:,:,-2])/dx1
            if field.shape[-1]>2:
                dxc=(self.earth_radius*coslat.view(-1,1)*(lon_rad[2:]-lon_rad[:-2]).view(1,-1)).view(1,1,field.shape[-2],field.shape[-1]-2)
                out[:,:,:,1:-1]=(field[:,:,:,2:]-field[:,:,:,:-2])/dxc
            return out

        def _ddy(self,field,lat):
            if field.shape[-2]<2:
                raise ValueError('纬度维至少需要 2 个格点。')
            y=self.earth_radius*torch.deg2rad(lat)
            out=torch.empty_like(field)
            out[:,:,0,:]=(field[:,:,1,:]-field[:,:,0,:])/(y[1]-y[0])
            out[:,:,-1,:]=(field[:,:,-1,:]-field[:,:,-2,:])/(y[-1]-y[-2])
            if field.shape[-2]>2:
                dyc=(y[2:]-y[:-2]).view(1,1,-1,1)
                out[:,:,1:-1,:]=(field[:,:,2:,:]-field[:,:,:-2,:])/dyc
            return out

        def _ddp(self,field,pressure):
            if field.shape[1]<2:
                raise ValueError('压力层至少需要 2 层。')
            out=torch.empty_like(field)
            out[:,0,:,:]=(field[:,1,:,:]-field[:,0,:,:])/(pressure[1]-pressure[0])
            out[:,-1,:,:]=(field[:,-1,:,:]-field[:,-2,:,:])/(pressure[-1]-pressure[-2])
            if field.shape[1]>2:
                dpc=(pressure[2:]-pressure[:-2]).view(1,-1,1,1)
                out[:,1:-1,:,:]=(field[:,2:,:,:]-field[:,:-2,:,:])/dpc
            return out

        def forward(self,y_pred,y_true):
            if y_pred.ndim!=4 or y_true.ndim!=4:
                raise ValueError('PINNLoss 要求 y_pred/y_true 形状为 (B,C,H,W)。')
            if y_pred.shape!=y_true.shape:
                raise ValueError('y_pred 与 y_true 形状必须完全一致。')
            y_pred=y_pred.float()
            y_true=y_true.float()

            data_loss=torch.mean((y_pred-y_true)**2)

            _,c,h,w=y_pred.shape
            if c%3!=0:
                raise ValueError('PINNLoss 假定输出按 [u...,v...,w...] 排列，因此通道数必须能被 3 整除。')
            nlevel=c//3
            if nlevel<2:
                raise ValueError('PINNLoss 至少需要两个压力层。')

            mean_t,std_t=self._get_mean_std(y_pred)
            y_pred_phys=y_pred*std_t+mean_t
            y_true_phys=y_true*std_t+mean_t

            u_pred=y_pred_phys[:,:nlevel,:,:]
            v_pred=y_pred_phys[:,nlevel:2*nlevel,:,:]
            w_pred_external=y_pred_phys[:,2*nlevel:3*nlevel,:,:]
            u_true=y_true_phys[:,:nlevel,:,:]
            v_true=y_true_phys[:,nlevel:2*nlevel,:,:]
            w_true_external=y_true_phys[:,2*nlevel:3*nlevel,:,:]

            omega_pred=w_pred_external/self.w_external_scale
            omega_true=w_true_external/self.w_external_scale

            lat,lon,pressure=self._get_coordinates(h,w,nlevel,y_pred.device,y_pred.dtype)

            du_dx_pred=self._ddx(u_pred,lat,lon)
            dv_dy_pred=self._ddy(v_pred,lat)
            domega_dp_pred=self._ddp(omega_pred,pressure)
            r_cont=du_dx_pred+dv_dy_pred+domega_dp_pred
            continuity_loss=torch.mean(r_cont**2)

            zeta_pred=self._ddx(v_pred,lat,lon)-self._ddy(u_pred,lat)
            zeta_true=self._ddx(v_true,lat,lon)-self._ddy(u_true,lat)
            vorticity_loss=torch.mean((zeta_pred-zeta_true)**2)

            su_pred=self._ddp(u_pred,pressure)
            sv_pred=self._ddp(v_pred,pressure)
            su_true=self._ddp(u_true,pressure)
            sv_true=self._ddp(v_true,pressure)
            vertical_shear_loss=torch.mean((su_pred-su_true)**2+(sv_pred-sv_true)**2)

            physics_loss=continuity_loss+vorticity_loss+vertical_shear_loss
            weighted_physics_loss=(
                self.lambda_continuity*continuity_loss
                +self.lambda_vorticity*vorticity_loss
                +self.lambda_shear*vertical_shear_loss
            )
            total_loss=data_loss+weighted_physics_loss

            self.last_components={
                'data_loss':float(data_loss.detach().cpu()),
                'continuity_loss':float(continuity_loss.detach().cpu()),
                'vorticity_loss':float(vorticity_loss.detach().cpu()),
                'vertical_shear_loss':float(vertical_shear_loss.detach().cpu()),
                'physics_loss':float(physics_loss.detach().cpu()),
                'weighted_physics_loss':float(weighted_physics_loss.detach().cpu()),
                'lambda_continuity':float(self.lambda_continuity),
                'lambda_vorticity':float(self.lambda_vorticity),
                'lambda_shear':float(self.lambda_shear),
                'total_loss':float(total_loss.detach().cpu()),
            }
            return total_loss

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
            w = cnt.sum() / (num_classes * cnt.clamp_min(1.0))  # N/(C*count_c)
            return w  # shape: (C,)
        class_weights = get_class_weights_from_labels(vy, device=devices)
    if task_mode=='regression':
        if type(loss_function) is not str:
            loss=loss_function
        else:
            if loss_function in ['default','PINNLoss','pinn','PINN','pinn_loss','PINNLOSS']:
                loss=PINNLoss(
                    vy_mean=pinn_vy_mean,
                    vy_std=pinn_vy_std,
                    lambda_continuity=pinn_lambda_continuity,
                    lambda_vorticity=pinn_lambda_vorticity,
                    lambda_shear=pinn_lambda_shear,
                    lat=pinn_lat,
                    lon=pinn_lon,
                    pressure_levels=pinn_pressure_levels,
                    w_external_scale=pinn_w_external_scale,
                )
            elif loss_function=='MSELoss':
                loss=torch.nn.MSELoss()
            elif loss_function=='L1Loss':
                loss=torch.nn.L1Loss
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
                        return (1-pearson)**1.5
                loss=loss_pearsonr()
        if type(metrics) is not str:
            metric=metrics
        else:
            if metrics=='default' or metrics=='MSELoss':
                metric=torch.nn.MSELoss()
            elif metrics=='L1Loss':
                metric=torch.nn.L1Loss
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
                        return (1-pearson)**1.5
                metric=metric_pearsonr()
    elif task_mode=='binary_classify':
        if type(loss_function) is not str:
            loss=loss_function
        else:
            if loss_function=='default' or loss_function=='BCELoss':
                w0, w1 = class_weights[0], class_weights[1]
                weight_map = torch.where(y_batch > 0.5, w1, w0)
                loss=torch.nn.BCELoss(weight=weight_map)
            elif loss_function=='BCEWithLogitsLoss':
                pos_w = (class_weights[1] / class_weights[0]).to(devices).view(1)
                loss=torch.nn.BCEWithLogitsLoss(pos_weight=pos_w)
            elif loss_function=='SoftMarginLoss':
                loss=torch.nn.SoftMarginLoss()
            elif loss_function=='MultiLabelSoftMarginLoss':
                loss=torch.nn.MultiLabelSoftMarginLoss()
        if type(metrics) is not str:
            metric=metrics
        else:
            if metrics=='default' or metrics=='f1':
                metric=torchmetrics.F1Score(task="binary").to(devices)
            elif metrics=='accuracy':
                metric=torchmetrics.Accuracy(task="binary").to(devices)
            elif metrics=='precision':
                metric=torchmetrics.Precision(task="binary").to(devices)
            elif metrics=='recall':
                metric=torchmetrics.Recall(task="binary").to(devices)
            elif metrics=='BCELoss':
                metric=torch.nn.BCELoss()
            elif metrics=='BCEWithLogitsLoss':
                metric=torch.nn.BCEWithLogitsLoss()
            elif metrics=='SoftMarginLoss':
                metric=torch.nn.SoftMarginLoss()
            elif metrics=='MultiLabelSoftMarginLoss':
                metric=torch.nn.MultiLabelSoftMarginLoss()
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
            elif loss_function=='KLDivergence':
                loss=torch.nn.KLDivergence()
            elif loss_function=='HingeEmbeddingLoss':
                loss=torch.nn.HingeEmbeddingLoss()
            elif loss_function=='MultiLabelMarginLoss':
                loss=torch.nn.MultiLabelMarginLoss()
            elif loss_function=='TripletMarginWithDistanceLoss':
                loss=torch.nn.TripletMarginWithDistanceLoss()
        if type(metrics) is not str:
            metric=metrics
        else:
            if metrics=='default' or metrics=='accuracy' or metrics=='f1' or metrics=='recall' or  metrics=='precision':
                metric=torchmetrics.Accuracy(task="multiclass",num_classes = int(np.max(vy))+1, average="macro").to(devices)
            elif metrics=='recall' :
                metric=torchmetrics.Recall(task="multiclass", num_classes=int(np.max(vy))+1, average="macro").to(devices)
            elif metrics=='precision' :
                metric=torchmetrics.Precision(task="multiclass", num_classes=int(np.max(vy))+1, average="macro").to(devices)
            elif metrics=='f1' :
                metric=torchmetrics.F1Score(task="multiclass", num_classes=int(np.max(vy))+1, average="macro").to(devices)
            elif metrics=='CrossEntropyLoss':
                metric=torch.nn.CrossEntropyLoss()
            elif metrics=='NLLLoss':
                metric=torch.nn.NLLLoss()
            elif metrics=='TripletMarginLoss':
                metric=torch.nn.TripletMarginLoss()
            elif metrics=='KLDivergence':
                metric=torch.nn.KLDivergence()
            elif metrics=='HingeEmbeddingLoss':
                metric=torch.nn.HingeEmbeddingLoss()
            elif metrics=='MultiLabelMarginLoss':
                metric=torch.nn.MultiLabelMarginLoss()
            elif metrics=='TripletMarginWithDistanceLoss':
                metric=torch.nn.TripletMarginWithDistanceLoss()
    weights=0
    heatmap = 0
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
                return torch.mean(model_output[:, self.output_channel_idx, :, :].view(model_output.size(0), -1), dim=1).unsqueeze(1)
            elif model_output.ndim == 4:
                return torch.mean(model_output.view(model_output.size(0), -1), dim=1).unsqueeze(1)
            else:
                return torch.mean(model_output.view(model_output.size(0), -1), dim=1).unsqueeze(1)
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
    vys=vy[timestep-1:,:,:,:]
    vxs=np.zeros((vx.shape[0]-timestep+1,timestep,vx.shape[1],vx.shape[2],vx.shape[3]))
    for i in range(vx.shape[0]-timestep+1):
        vxs[i,:,:,:,:]=vx[i:i+timestep,:,:,:]
    vxs=vxs.transpose(0,1,4,2,3)
    if ifrandom_split=='yes':
        trainy,testy,trainx,testx = train_test_split(vys,vxs,test_size=test_size,random_state=25)
    elif ifrandom_split=='no':
        index=int((1-test_size)*vy.shape[0])
        trainy=vys[:index]
        testy=vys[index:]
        trainx=vxs[:index,:,:,:,:]
        testx=vxs[index:,:,:,:,:]
    elif ifrandom_split=='all_train' or ifrandom_split=='all_test' or ifrandom_split=='just_model':
        trainy=vys
        testy=vys
        trainx=vxs
        testx=vxs
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

                    da_channels = int(self.in_channels)
                    da_h = int(getattr(self, 'hight' + str(i)))
                    da_w = int(getattr(self, 'weight' + str(i)))

                    da_time_dim = da_channels * da_h * da_w

                    if da_time_dim % self.num_heads != 0:
                        raise ValueError(
                            "时间 Transformer 的 d_model=C*H*W 必须能被 num_heads 整除。"
                            f" 当前层={da_layer_id}, C={da_channels}, H={da_h}, W={da_w}, "
                            f"C*H*W={da_time_dim}, num_heads={self.num_heads}"
                        )

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

            exec('model_transformer'+str(i+1)+'=model_skip'+str(i+1), globals(), self.__dict__)

            for m in range(self.downnum,2*self.downnum-1):
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
                        exec('model_conv'+str(m+1)+'=model_act'+str(m+1), globals(), self.__dict__)

                if self.ifnormalization=='yes':
                    exec('self.covup'+str(m+1)+'=model_norm'+str(self.k), globals(), self.__dict__)
                else:
                    exec('self.covup'+str(m+1)+'=model_conv'+str(self.k), globals(), self.__dict__)

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
    if k_fold!=None:
        models=[]
        for i in range(k_fold):
            models.append(Model(trainx[0:2],trainy[0:2],timestep,downnum,covnum,layer_time,baselayer,cov_kernelsize,cov_strides,pool_method,pool_kernel_size,pool_strides,ifnormalization,normalization_method,trans_num,num_heads,dim_feedforward,trans_dropout_rate,activate,nlp_activate,da_attention_order,if_bottom_attention_only,if_resnet))
    else:
        model=Model(trainx[0:2],trainy[0:2],timestep,downnum,covnum,layer_time,baselayer,cov_kernelsize,cov_strides,pool_method,pool_kernel_size,pool_strides,ifnormalization,normalization_method,trans_num,num_heads,dim_feedforward,trans_dropout_rate,activate,nlp_activate,da_attention_order,if_bottom_attention_only,if_resnet)
    if if_best_mode!='no':
        if k_fold!=None:
            for i in range(k_fold):
                models[i].load_state_dict(torch.load(modelpath+'_'+str(i+1)+'.pth'))
        else:
            model.load_state_dict(torch.load(modelpath+'.pth'))
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
                opts[i].load_state_dict(torch.load(modelpath+'_'+str(i+1)+'_opt.pth'))
        else:
            opt.load_state_dict(torch.load(modelpath+'_opt.pth'))
    if if_print_model=='yes':
        if k_fold!=None:
            print(models[0])
        else:
            print(model)
    if ifrandom_split!='just_model':
        trainx=np.nan_to_num(trainx,nan=0)
        testx=np.nan_to_num(testx,nan=0)
        if ifrandom_split!='all_test':
            if valid_size!=None or k_fold !=None:
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
                            start = datetime.datetime.now()
                            for j in range(int(np.ceil(X_train_fold.shape[0]/batch_size))):
                                if j == int(X_train_fold.shape[0]/batch_size) :
                                    train_output = models[fold_no](torch.tensor(X_train_fold[j*batch_size:],dtype=torch.float32,device=devices))
                                    if task_mode=='multi_classify':
                                        if loss_function=='default' or loss_function=='CrossEntropyLoss':
                                            train_losss = torch.mean(loss(train_output, torch.tensor(y_train_fold[j*batch_size:],dtype=torch.long,device=devices).squeeze(1)))
                                        elif loss_function=='NLLLoss':
                                            train_losss = torch.mean(loss(torch.log(train_output).contiguous(), torch.tensor(y_train_fold[j*batch_size:],dtype=torch.long,device=devices).squeeze(1)))
                                        if metrics=='default' or metrics=='accuracy' or metrics=='f1' or metrics=='recall' or  metrics=='precision':
                                            train_metrics = torch.mean(metric(torch.argmax(train_output, dim=1), torch.tensor(y_train_fold[j*batch_size:],dtype=torch.long,device=devices).squeeze(1)))
                                        elif metrics=='CrossEntropyLoss':
                                            train_metrics = torch.mean(metric(train_output, torch.tensor(y_train_fold[j*batch_size:],dtype=torch.long,device=devices).squeeze(1)))
                                        elif metrics=='NLLLoss':
                                            train_metrics = torch.mean(metric(torch.log(train_output).contiguous(), torch.tensor(y_train_fold[j*batch_size:],dtype=torch.long,device=devices).squeeze(1)))
                                    else:
                                        train_losss = torch.mean(loss(train_output, torch.tensor(y_train_fold[j*batch_size:],dtype=torch.float32,device=devices)))
                                        train_metrics = torch.mean(metric(train_output, torch.tensor(y_train_fold[j*batch_size:],dtype=torch.float32,device=devices)))
                                else:
                                    train_output = models[fold_no](torch.tensor(X_train_fold[j*batch_size:(j+1)*batch_size],dtype=torch.float32,device=devices))
                                    if task_mode=='multi_classify':
                                        if loss_function=='default' or loss_function=='CrossEntropyLoss':
                                            train_losss = torch.mean(loss(train_output, torch.tensor(y_train_fold[j*batch_size:(j+1)*batch_size],dtype=torch.long,device=devices).squeeze(1)))
                                        elif loss_function=='NLLLoss':
                                            train_losss = torch.mean(loss(torch.log(train_output).contiguous(), torch.tensor(y_train_fold[j*batch_size:(j+1)*batch_size],dtype=torch.long,device=devices).squeeze(1)))
                                        if metrics=='default' or metrics=='accuracy' or metrics=='f1' or metrics=='recall' or  metrics=='precision':
                                            train_metrics = torch.mean(metric(torch.argmax(train_output, dim=1), torch.tensor(y_train_fold[j*batch_size:(j+1)*batch_size],dtype=torch.long,device=devices).squeeze(1)))
                                        elif metrics=='CrossEntropyLoss':
                                            train_metrics = torch.mean(metric(train_output, torch.tensor(y_train_fold[j*batch_size:(j+1)*batch_size],dtype=torch.long,device=devices).squeeze(1)))
                                        elif metrics=='NLLLoss':
                                            train_metrics = torch.mean(metric(torch.log(train_output).contiguous(), torch.tensor(y_train_fold[j*batch_size:(j+1)*batch_size],dtype=torch.long,device=devices).squeeze(1)))
                                    else:
                                        train_losss = torch.mean(loss(train_output, torch.tensor(y_train_fold[j*batch_size:(j+1)*batch_size],dtype=torch.float32,device=devices)))
                                        train_metrics = torch.mean(metric(train_output, torch.tensor(y_train_fold[j*batch_size:(j+1)*batch_size],dtype=torch.float32,device=devices)))
                                train_loss[fold_no,j]=np.array(train_losss.item())
                                train_metric[fold_no,j]=np.array(train_metrics.item())
                                opts[fold_no].zero_grad()
                                train_losss.backward(retain_graph=True)
                                opts[fold_no].step()
                            with torch.no_grad():
                                for k in range(int(np.ceil(X_val_fold.shape[0]/batch_size))):
                                    if k == int(X_val_fold.shape[0]/batch_size):
                                        test_output = models[fold_no](torch.tensor(X_val_fold[k*batch_size:],dtype=torch.float32,device=devices))
                                        if task_mode=='multi_classify':
                                            if loss_function=='default' or loss_function=='CrossEntropyLoss':
                                                test_losss= torch.mean(loss(test_output,torch.tensor(y_val_fold[k*batch_size:],dtype=torch.long,device=devices).squeeze(1)))
                                            elif loss_function=='NLLLoss':
                                                test_losss= torch.mean(loss(torch.log(test_output).contiguous(),torch.tensor(y_val_fold[k*batch_size:],dtype=torch.long,device=devices).squeeze(1)))
                                            if metrics=='default' or metrics=='accuracy' or metrics=='f1' or metrics=='recall' or  metrics=='precision':
                                                test_metrics=torch.mean(metric(torch.argmax(test_output, dim=1),torch.tensor(y_val_fold[k*batch_size:],dtype=torch.long,device=devices).squeeze(1)))
                                            elif metrics=='CrossEntropyLoss':
                                                test_metrics=torch.mean(metric(test_output,torch.tensor(y_val_fold[k*batch_size:],dtype=torch.long,device=devices).squeeze(1)))
                                            elif metrics=='NLLLoss':
                                                test_metrics=torch.mean(metric(torch.log(test_output).contiguous(),torch.tensor(y_val_fold[k*batch_size:],dtype=torch.long,device=devices).squeeze(1)))
                                        else:
                                            test_losss= torch.mean(loss(test_output,torch.tensor(y_val_fold[k*batch_size:],dtype=torch.float32,device=devices)))
                                            test_metrics=torch.mean(metric(test_output,torch.tensor(y_val_fold[k*batch_size:],dtype=torch.float32,device=devices)))
                                    else:
                                        test_output = models[fold_no](torch.tensor(X_val_fold[k*batch_size:(k+1)*batch_size],dtype=torch.float32,device=devices))
                                        if task_mode=='multi_classify':
                                            if loss_function=='default' or loss_function=='CrossEntropyLoss':
                                                test_losss= torch.mean(loss(test_output,torch.tensor(y_val_fold[k*batch_size:(k+1)*batch_size],dtype=torch.long,device=devices).squeeze(1)))
                                            elif loss_function=='NLLLoss':
                                                test_losss= torch.mean(loss(torch.log(test_output).contiguous(),torch.tensor(y_val_fold[k*batch_size:(k+1)*batch_size],dtype=torch.long,device=devices).squeeze(1)))
                                            if metrics=='default' or metrics=='accuracy' or metrics=='f1' or metrics=='recall' or  metrics=='precision':
                                                test_metrics=torch.mean(metric(torch.argmax(test_output, dim=1),torch.tensor(y_val_fold[k*batch_size:(k+1)*batch_size],dtype=torch.long,device=devices).squeeze(1)))
                                            elif metrics=='CrossEntropyLoss':
                                                test_metrics=torch.mean(metric(test_output,torch.tensor(y_val_fold[k*batch_size:(k+1)*batch_size],dtype=torch.long,device=devices).squeeze(1)))
                                            elif metrics=='NLLLoss':
                                                test_metrics=torch.mean(metric(torch.log(test_output).contiguous(),torch.tensor(y_val_fold[k*batch_size:(k+1)*batch_size],dtype=torch.long,device=devices).squeeze(1)))
                                        else:
                                            test_losss= torch.mean(loss(test_output,torch.tensor(y_val_fold[k*batch_size:(k+1)*batch_size],dtype=torch.float32,device=devices)))
                                            test_metrics=torch.mean(metric(test_output,torch.tensor(y_val_fold[k*batch_size:(k+1)*batch_size],dtype=torch.float32,device=devices)))
                                    test_loss[fold_no,k]=np.array(test_losss.item())
                                    test_metric[fold_no,k]=np.array(test_metrics.item())
                            if if_early_stopping!=None:
                                early_stopping(np.nanmean(test_loss[fold_no,:]), models[fold_no])
                                if early_stopping.early_stop:
                                    early_stopping.load_best_checkpoint(models[fold_no])
                                    break
                            end = datetime.datetime.now()
                            print('第',i+1,'次训练loss:',np.nanmean(train_loss[fold_no,:]),'，metric:',np.nanmean(train_metric[fold_no,:]),'  第',i+1,'次测试loss:',np.nanmean(test_loss[fold_no,:]),'，metric：',np.nanmean(test_metric[fold_no,:]),'训练用时:',end - start)
                else:
                    if if_early_stopping!=None:
                        early_stopping = EarlyStopping(patience=if_early_stopping)
                    for i in range(epochs):
                        start = datetime.datetime.now()
                        if i==0:
                            if ifrandom_split=='yes':
                                trainy,validy,trainx,validx = train_test_split(trainy,trainx,test_size=valid_size/(1-test_size),random_state=25)
                            else:
                                index=int((1-valid_size/(1-test_size))*trainy.shape[0])
                                validy=trainy[index:]
                                trainy=trainy[:index]
                                validx=trainx[index:]
                                trainx=trainx[:index]
                        train_loss=np.zeros((int(np.ceil(trainx.shape[0]/batch_size))))
                        train_metric=np.zeros((int(np.ceil(trainx.shape[0]/batch_size))))
                        test_loss=np.zeros((int(np.ceil(validx.shape[0]/batch_size))))
                        test_metric=np.zeros((int(np.ceil(validx.shape[0]/batch_size))))
                        for j in range(int(np.ceil(trainx.shape[0]/batch_size))):
                            if j == int(trainx.shape[0]/batch_size) :
                                train_output = model(torch.tensor(trainx[j*batch_size:],dtype=torch.float32,device=devices))
                                if task_mode=='multi_classify':
                                    if loss_function=='default' or loss_function=='CrossEntropyLoss':
                                        train_losss = torch.mean(loss(train_output, torch.tensor(trainy[j*batch_size:],dtype=torch.long,device=devices).squeeze(1)))
                                    elif loss_function=='NLLLoss':
                                        train_losss = torch.mean(loss(torch.log(train_output).contiguous(), torch.tensor(trainy[j*batch_size:],dtype=torch.long,device=devices).squeeze(1)))
                                    if metrics=='default' or metrics=='accuracy' or metrics=='f1' or metrics=='recall' or  metrics=='precision':
                                        train_metrics = torch.mean(metric(torch.argmax(train_output, dim=1), torch.tensor(trainy[j*batch_size:],dtype=torch.long,device=devices).squeeze(1)))
                                    elif metrics=='CrossEntropyLoss':
                                        train_metrics = torch.mean(metric(train_output, torch.tensor(trainy[j*batch_size:],dtype=torch.long,device=devices).squeeze(1)))
                                    elif metrics=='NLLLoss':
                                        train_metrics = torch.mean(metric(torch.log(train_output).contiguous(), torch.tensor(trainy[j*batch_size:],dtype=torch.long,device=devices).squeeze(1)))
                                else:
                                    train_losss = torch.mean(loss(train_output, torch.tensor(trainy[j*batch_size:],dtype=torch.float32,device=devices)))
                                    train_metrics = torch.mean(metric(train_output, torch.tensor(trainy[j*batch_size:],dtype=torch.float32,device=devices)))
                            else:
                                train_output = model(torch.tensor(trainx[j*batch_size:(j+1)*batch_size],dtype=torch.float32,device=devices))
                                if task_mode=='multi_classify':
                                    if loss_function=='default' or loss_function=='CrossEntropyLoss':
                                        train_losss = torch.mean(loss(train_output, torch.tensor(trainy[j*batch_size:(j+1)*batch_size],dtype=torch.long,device=devices).squeeze(1)))
                                    elif loss_function=='NLLLoss':
                                        train_losss = torch.mean(loss(torch.log(train_output).contiguous(), torch.tensor(trainy[j*batch_size:(j+1)*batch_size],dtype=torch.long,device=devices).squeeze(1)))
                                    if metrics=='default' or metrics=='accuracy' or metrics=='f1' or metrics=='recall' or  metrics=='precision':
                                        train_metrics = torch.mean(metric(torch.argmax(train_output, dim=1), torch.tensor(trainy[j*batch_size:(j+1)*batch_size],dtype=torch.long,device=devices).squeeze(1)))
                                    elif metrics=='CrossEntropyLoss':
                                        train_metrics = torch.mean(metric(train_output, torch.tensor(trainy[j*batch_size:(j+1)*batch_size],dtype=torch.long,device=devices).squeeze(1)))
                                    elif metrics=='NLLLoss':
                                        train_metrics = torch.mean(metric(torch.log(train_output).contiguous(), torch.tensor(trainy[j*batch_size:(j+1)*batch_size],dtype=torch.long,device=devices).squeeze(1)))
                                else:
                                    train_losss = torch.mean(loss(train_output, torch.tensor(trainy[j*batch_size:(j+1)*batch_size],dtype=torch.float32,device=devices)))
                                    train_metrics = torch.mean(metric(train_output, torch.tensor(trainy[j*batch_size:(j+1)*batch_size],dtype=torch.float32,device=devices)))
                            train_loss[j]=np.array(train_losss.item())
                            train_metric[j]=np.array(train_metrics.item())
                            opt.zero_grad()
                            train_losss.backward(retain_graph=True)
                            opt.step()
                        with torch.no_grad():
                            for k in range(int(np.ceil(validx.shape[0]/batch_size))):
                                if k == int(validx.shape[0]/batch_size) :
                                    test_output = model(torch.tensor(validx[k*batch_size:],dtype=torch.float32,device=devices))
                                    if task_mode=='multi_classify':
                                        if loss_function=='default' or loss_function=='CrossEntropyLoss':
                                            test_losss= torch.mean(loss(test_output,torch.tensor(validy[k*batch_size:],dtype=torch.long,device=devices).squeeze(1)))
                                        elif loss_function=='NLLLoss':
                                            test_losss= torch.mean(loss(torch.log(test_output).contiguous(),torch.tensor(validy[k*batch_size:],dtype=torch.long,device=devices).squeeze(1)))
                                        if metrics=='default' or metrics=='accuracy' or metrics=='f1' or metrics=='recall' or  metrics=='precision':
                                            test_metrics=torch.mean(metric(torch.argmax(test_output, dim=1),torch.tensor(validy[k*batch_size:],dtype=torch.long,device=devices).squeeze(1)))
                                        elif metrics=='CrossEntropyLoss':
                                            test_metrics=torch.mean(metric(test_output,torch.tensor(validy[k*batch_size:],dtype=torch.long,device=devices).squeeze(1)))
                                        elif metrics=='NLLLoss':
                                            test_metrics=torch.mean(metric(torch.log(test_output).contiguous(),torch.tensor(validy[k*batch_size:],dtype=torch.long,device=devices).squeeze(1)))
                                    else:
                                        test_losss= torch.mean(loss(test_output,torch.tensor(validy[k*batch_size:],dtype=torch.float32,device=devices)))
                                        test_metrics=torch.mean(metric(test_output,torch.tensor(validy[k*batch_size:],dtype=torch.float32,device=devices)))
                                else:
                                    test_output = model(torch.tensor(validx[k*batch_size:(k+1)*batch_size],dtype=torch.float32,device=devices))
                                    if task_mode=='multi_classify':
                                        if loss_function=='default' or loss_function=='CrossEntropyLoss':
                                            test_losss= torch.mean(loss(test_output,torch.tensor(validy[k*batch_size:(k+1)*batch_size],dtype=torch.long,device=devices).squeeze(1)))
                                        elif loss_function=='NLLLoss':
                                            test_losss= torch.mean(loss(torch.log(test_output).contiguous(),torch.tensor(validy[k*batch_size:(k+1)*batch_size],dtype=torch.long,device=devices).squeeze(1)))
                                        if metrics=='default' or metrics=='accuracy' or metrics=='f1' or metrics=='recall' or  metrics=='precision':
                                            test_metrics=torch.mean(metric(torch.argmax(test_output, dim=1),torch.tensor(validy[k*batch_size:(k+1)*batch_size],dtype=torch.long,device=devices).squeeze(1)))
                                        elif metrics=='CrossEntropyLoss':
                                            test_metrics=torch.mean(metric(test_output,torch.tensor(validy[k*batch_size:(k+1)*batch_size],dtype=torch.long,device=devices).squeeze(1)))
                                        elif metrics=='NLLLoss':
                                            test_metrics=torch.mean(metric(torch.log(test_output).contiguous(),torch.tensor(validy[k*batch_size:(k+1)*batch_size],dtype=torch.long,device=devices).squeeze(1)))
                                    else:
                                        test_losss= torch.mean(loss(test_output,torch.tensor(validy[k*batch_size:(k+1)*batch_size],dtype=torch.float32,device=devices)))
                                        test_metrics=torch.mean(metric(test_output,torch.tensor(validy[k*batch_size:(k+1)*batch_size],dtype=torch.float32,device=devices)))
                                test_loss[k]=np.array(test_losss.item())
                                test_metric[k]=np.array(test_metrics.item())
                        if if_early_stopping!=None:
                            early_stopping(np.nanmean(test_loss), model)
                            if early_stopping.early_stop:
                                early_stopping.load_best_checkpoint(model)
                                break
                        end = datetime.datetime.now()
                        print('第',i+1,'次训练loss:',np.nanmean(train_loss),'，metric:',np.nanmean(train_metric),'  第',i+1,'次测试loss:',np.nanmean(test_loss),'，metric：',np.nanmean(test_metric),'训练用时:',end - start)
            else:
                if if_early_stopping!=None:
                    early_stopping = EarlyStopping(patience=if_early_stopping)
                for i in range(epochs):
                    start = datetime.datetime.now()
                    train_loss=np.zeros((int(np.ceil(trainx.shape[0]/batch_size))))
                    train_metric=np.zeros((int(np.ceil(trainx.shape[0]/batch_size))))
                    test_loss=np.zeros((int(np.ceil(testx.shape[0]/batch_size))))
                    test_metric=np.zeros((int(np.ceil(testx.shape[0]/batch_size))))
                    for j in range(int(np.ceil(trainx.shape[0]/batch_size))):
                        if j == int(trainx.shape[0]/batch_size):
                            train_output = model(torch.tensor(trainx[j*batch_size:],dtype=torch.float32,device=devices))
                            if task_mode=='multi_classify':
                                if loss_function=='default' or loss_function=='CrossEntropyLoss':
                                    train_losss = torch.mean(loss(train_output, torch.tensor(trainy[j*batch_size:],dtype=torch.long,device=devices).squeeze(1)))
                                elif loss_function=='NLLLoss':
                                    train_losss = torch.mean(loss(torch.log(train_output).contiguous(), torch.tensor(trainy[j*batch_size:],dtype=torch.long,device=devices).squeeze(1)))
                                if metrics=='default' or metrics=='accuracy' or metrics=='f1' or metrics=='recall' or  metrics=='precision':
                                    train_metrics = torch.mean(metric(torch.argmax(train_output, dim=1), torch.tensor(trainy[j*batch_size:],dtype=torch.long,device=devices).squeeze(1)))
                                elif metrics=='CrossEntropyLoss':
                                    train_metrics = torch.mean(metric(train_output, torch.tensor(trainy[j*batch_size:],dtype=torch.long,device=devices).squeeze(1)))
                                elif metrics=='NLLLoss':
                                    train_metrics = torch.mean(metric(torch.log(train_output).contiguous(), torch.tensor(trainy[j*batch_size:],dtype=torch.long,device=devices).squeeze(1)))
                            else:
                                train_losss = torch.mean(loss(train_output, torch.tensor(trainy[j*batch_size:],dtype=torch.float32,device=devices)))
                                train_metrics = torch.mean(metric(train_output, torch.tensor(trainy[j*batch_size:],dtype=torch.float32,device=devices)))
                        else:
                            train_output = model(torch.tensor(trainx[j*batch_size:(j+1)*batch_size],dtype=torch.float32,device=devices))
                            if task_mode=='multi_classify':
                                if loss_function=='default' or loss_function=='CrossEntropyLoss':
                                    train_losss = torch.mean(loss(train_output, torch.tensor(trainy[j*batch_size:(j+1)*batch_size],dtype=torch.long,device=devices).squeeze(1)))
                                elif loss_function=='NLLLoss':
                                    train_losss = torch.mean(loss(torch.log(train_output).contiguous(), torch.tensor(trainy[j*batch_size:(j+1)*batch_size],dtype=torch.long,device=devices).squeeze(1)))
                                if metrics=='default' or metrics=='accuracy' or metrics=='f1' or metrics=='recall' or  metrics=='precision':
                                    train_metrics = torch.mean(metric(torch.argmax(train_output, dim=1), torch.tensor(trainy[j*batch_size:(j+1)*batch_size],dtype=torch.long,device=devices).squeeze(1)))
                                elif metrics=='CrossEntropyLoss':
                                    train_metrics = torch.mean(metric(train_output, torch.tensor(trainy[j*batch_size:(j+1)*batch_size],dtype=torch.long,device=devices).squeeze(1)))
                                elif metrics=='NLLLoss':
                                    train_metrics = torch.mean(metric(torch.log(train_output).contiguous(), torch.tensor(trainy[j*batch_size:(j+1)*batch_size],dtype=torch.long,device=devices).squeeze(1)))
                            else:
                                train_losss = torch.mean(loss(train_output, torch.tensor(trainy[j*batch_size:(j+1)*batch_size],dtype=torch.float32,device=devices)))
                                train_metrics = torch.mean(metric(train_output, torch.tensor(trainy[j*batch_size:(j+1)*batch_size],dtype=torch.float32,device=devices)))
                        train_loss[j]=np.array(train_losss.item())
                        train_metric[j]=np.array(train_metrics.item())
                        opt.zero_grad()
                        train_losss.backward(retain_graph=True)
                        opt.step()
                    with torch.no_grad():
                        for k in range(int(np.ceil(testx.shape[0]/batch_size))):
                            if k == int(testx.shape[0]/batch_size) :
                                test_output = model(torch.tensor(testx[k*batch_size:],dtype=torch.float32,device=devices))
                                if task_mode=='multi_classify':
                                    if loss_function=='default' or loss_function=='CrossEntropyLoss':
                                        test_losss= torch.mean(loss(test_output,torch.tensor(testy[k*batch_size:],dtype=torch.long,device=devices).squeeze(1)))
                                    elif loss_function=='NLLLoss':
                                        test_losss= torch.mean(loss(torch.log(test_output).contiguous(),torch.tensor(testy[k*batch_size:],dtype=torch.long,device=devices).squeeze(1)))
                                    if metrics=='default' or metrics=='accuracy' or metrics=='f1' or metrics=='recall' or  metrics=='precision':
                                        test_metrics=torch.mean(metric(torch.argmax(test_output, dim=1),torch.tensor(testy[k*batch_size:],dtype=torch.long,device=devices).squeeze(1)))
                                    elif metrics=='CrossEntropyLoss':
                                        test_metrics=torch.mean(metric(test_output,torch.tensor(testy[k*batch_size:],dtype=torch.long,device=devices).squeeze(1)))
                                    elif metrics=='NLLLoss':
                                        test_metrics=torch.mean(metric(torch.log(test_output).contiguous(),torch.tensor(testy[k*batch_size:],dtype=torch.long,device=devices).squeeze(1)))
                                else:
                                    test_losss= torch.mean(loss(test_output,torch.tensor(testy[k*batch_size:],dtype=torch.float32,device=devices)))
                                    test_metrics=torch.mean(metric(test_output,torch.tensor(testy[k*batch_size:],dtype=torch.float32,device=devices)))
                            else:
                                test_output = model(torch.tensor(testx[k*batch_size:(k+1)*batch_size],dtype=torch.float32,device=devices))
                                if task_mode=='multi_classify':
                                    if loss_function=='default' or loss_function=='CrossEntropyLoss':
                                        test_losss= torch.mean(loss(test_output,torch.tensor(testy[k*batch_size:(k+1)*batch_size],dtype=torch.long,device=devices).squeeze(1)))
                                    elif loss_function=='NLLLoss':
                                        test_losss= torch.mean(loss(torch.log(test_output).contiguous(),torch.tensor(testy[k*batch_size:(k+1)*batch_size],dtype=torch.long,device=devices).squeeze(1)))
                                    if metrics=='default' or metrics=='accuracy' or metrics=='f1' or metrics=='recall' or  metrics=='precision':
                                        test_metrics=torch.mean(metric(torch.argmax(test_output, dim=1),torch.tensor(testy[k*batch_size:(k+1)*batch_size],dtype=torch.long,device=devices).squeeze(1)))
                                    elif metrics=='CrossEntropyLoss':
                                        test_metrics=torch.mean(metric(test_output,torch.tensor(testy[k*batch_size:(k+1)*batch_size],dtype=torch.long,device=devices).squeeze(1)))
                                    elif metrics=='NLLLoss':
                                        test_metrics=torch.mean(metric(torch.log(test_output).contiguous(),torch.tensor(testy[k*batch_size:(k+1)*batch_size],dtype=torch.long,device=devices).squeeze(1)))
                                else:
                                    test_losss= torch.mean(loss(test_output,torch.tensor(testy[k*batch_size:(k+1)*batch_size],dtype=torch.float32,device=devices)))
                                    test_metrics=torch.mean(metric(test_output,torch.tensor(testy[k*batch_size:(k+1)*batch_size],dtype=torch.float32,device=devices)))
                            test_loss[k]=np.array(test_losss.item())
                            test_metric[k]=np.array(test_metrics.item())
                    if if_early_stopping!=None:
                        early_stopping(np.nanmean(test_loss), model)
                        if early_stopping.early_stop:
                            early_stopping.load_best_checkpoint(model)
                            break
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
                for k in range(int(np.ceil(testx.shape[0]/batch_size))):
                    if k == int(testx.shape[0]/batch_size) :
                        predicty_batch = models[i](torch.tensor(testx[k*batch_size:],dtype=torch.float32,device=devices))
                        predicty[i,k*batch_size:]=predicty_batch.cpu().detach().numpy()
                    else:
                        predicty_batch = models[i](torch.tensor(testx[k*batch_size:(k+1)*batch_size],dtype=torch.float32,device=devices))
                        predicty[i,k*batch_size:(k+1)*batch_size]=predicty_batch.cpu().detach().numpy()
            predicty=np.nanmean(predicty,axis=0)
        else:
            for k in range(int(np.ceil(testx.shape[0]/batch_size))):
                if k == int(testx.shape[0]/batch_size) :
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
                        r[i,j,k],p[i,j,k] = pearsonr(predicty[:,k,i,j],testy[:,k,i,j])
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
            use_gradient_explainer = True

            wrapped_model = ShapWrapper(model_instance, x_sample_shape, output_channel_idx=output_channel_idx).to(devices)
            wrapped_model.eval()

            background_data = current_shap_data

            if use_gradient_explainer:
                explainer = shap.GradientExplainer(wrapped_model, background_data)
            else:
                explainer = shap.DeepExplainer(wrapped_model, background_data)

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
                current_testx_data = testx[index, ]

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
                        fold_shap_results_for_channel.append(_compute_shap_values_for_model_for_channel(
                            model_k_fold, flat_current_testx_tensor, x_sample_shape, output_ch_idx, devices
                        ))
                    shap_values_per_output_channel.append(np.nanmean(np.stack(fold_shap_results_for_channel, axis=0), axis=0))
                else:
                    model.eval()
                    shap_values_per_output_channel.append(_compute_shap_values_for_model_for_channel(
                        model, flat_current_testx_tensor, x_sample_shape, output_ch_idx, devices
                    ))
            shap_values_aggregated = np.stack(shap_values_per_output_channel, axis=0)

            if ifheatmap=='yes' and testx.ndim==5:
                heatmap_per_output_channel = np.nanmean(shap_values_aggregated, axis=(1, 2))

                heatmap_output_list = []
                for i in range(num_model_outputs):
                    current_heatmap_slice = heatmap_per_output_channel[i, :, :, :]
                    current_heatmap_slice = current_heatmap_slice.transpose(1, 2, 0)
                    heatmap_output_list.append(current_heatmap_slice)

                if len(heatmap_output_list) > 0 and all(h.shape == heatmap_output_list[0].shape for h in heatmap_output_list):
                     heatmap = np.stack(heatmap_output_list, axis=0)
                else:
                     heatmap = heatmap_output_list

            if ifweight in ['yes', 'shap']:
                weights_aggregated_per_channel_feature = np.nanmean(shap_values_aggregated, axis=(1, 2, 4, 5))

                input_channel_for_weights = trainx.shape[2]

                weights_final_output = np.zeros((num_model_outputs, input_channel_for_weights))

                for i in range(num_model_outputs):
                    current_output_channel_weights = weights_aggregated_per_channel_feature[i, :]
                    sum_for_norm = np.nansum(current_output_channel_weights)

                    if sum_for_norm != 0:
                        weights_final_output[i, :] = (current_output_channel_weights / sum_for_norm) * 100
                    else:
                        weights_final_output[i, :] = 0.0

                    if ifmute == 'no':
                        for j in range(input_channel_for_weights):
                            print('预报因子',j+1,'对预报值',i+1,'的贡献：',np.array(weights_final_output[i,j]),'％')
                        print('\n')
                weights = weights_final_output

        elif ifweight == 'oob':

            weights = np.zeros((testy.shape[1],testx.shape[2])) # C_out x C_in
            weight_more=np.zeros((testy.shape[1],testx.shape[2]))
            for i in tqdm(range(testy.shape[1])):
                for j in range(testx.shape[2]):
                    testx_new=copy.deepcopy(testx)
                    weight_current_feature_output_pair = []
                    for k in range(10):
                        per=np.random.permutation(testx.shape[0])
                        testx_shuffle=testx[per,:,j,:,:]
                        testx_new[:,:,j,:,:]=testx_shuffle

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
                                else: # binary_classify
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
                    for j in range(testx.shape[2]):
                        print('预报因子',j+1,'对预报值',i+1,'的贡献：',np.array(weights[i,j]),'％')
                    print('\n')
    if ifsave=='yes':
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
