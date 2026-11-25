"""Pressure-coordinate physics losses for the 1 h DA-TransUNet experiments."""
import numpy as np
import torch
from torch import nn


class PINNLoss(nn.Module):
    def __init__(self,vy_mean,vy_std,lambda_continuity=611.7192928848946,lambda_vorticity=142.40845542406902,lambda_shear=329.4372079166584,lat=None,lon=None,pressure_levels=None,w_external_scale=-0.1,global_scale=None):
        super().__init__()
        if vy_mean is None or vy_std is None:
            raise ValueError('PINNLoss requires the target mean and standard deviation fitted to the training labels.')
        self.global_scale = global_scale
        if global_scale is not None and (not np.isfinite(global_scale) or global_scale < 0):
            raise ValueError('global_scale must be finite and nonnegative.')
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

    def component_losses(self,y_pred,y_true):
        """Return differentiable data MSE, continuity, relative-vorticity and vertical-shear losses."""
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

        return data_loss, continuity_loss, vorticity_loss, vertical_shear_loss

    def forward(self,y_pred,y_true):
        data_loss,continuity_loss,vorticity_loss,vertical_shear_loss = self.component_losses(y_pred,y_true)
        physics_loss=continuity_loss+vorticity_loss+vertical_shear_loss
        if self.global_scale == 0.0:
            weighted_physics_loss = data_loss.new_zeros(())
            total_loss = data_loss
        else:
            weighted_physics_loss = (self.lambda_continuity * continuity_loss
                                     + self.lambda_vorticity * vorticity_loss
                                     + self.lambda_shear * vertical_shear_loss)
            if self.global_scale is not None:
                weighted_physics_loss = self.global_scale * weighted_physics_loss
            total_loss = data_loss + weighted_physics_loss

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
        if self.global_scale is not None:
            self.last_components['global_scale'] = self.global_scale
        return total_loss
