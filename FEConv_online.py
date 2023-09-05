import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops.deform_conv import deform_conv2d, DeformConv2d


def distortion_aware_map(img_W, img_H, k_W, k_H, s_width, s_height, intrinsics=None):
    """
    Performs Kannala-Brandt Convolution for a given calibration
    """

    if intrinsics is None:
        #Calibration file. Change to your own file if needed, ehere is an example of our article
        intrinsics = np.loadtxt('fisheye_calibration_FOV195.txt')
    
    # Separation of each element of the intrinsic calibration file
    fe_cx, fe_cy = intrinsics[0],intrinsics[1]
    fx, fy = intrinsics[2],intrinsics[3]
    camres_W, camres_H = intrinsics[4],intrinsics[5]
    fish_FOV = intrinsics[6]
    kb_intrinsics = intrinsics[7:]
    
    pad_H,pad_W = 0,2 # Padding is computed so the feature map size is divisible by 128

    def rotation_matrix(axis, theta):
        """
        Return the rotation matrix associated with counterclockwise rotation about
        the given axis by theta radians.
        """
        axis = torch.as_tensor(axis, device='cpu', dtype=torch.float)
        axis = axis / math.sqrt(torch.dot(axis, axis))
        a = math.cos(theta / 2.0)
        b, c, d = -axis * math.sin(theta / 2.0)
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
        ROT = torch.tensor([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                            [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                            [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]], device='cpu', dtype=torch.float)
        return ROT

    #Direct KB full kb_intrinsics (without deformations correction)
    def KB_dir(x):
        return x + kb_intrinsics[0]*x**3 + kb_intrinsics[1]*x**5 + kb_intrinsics[2]*x**7 + kb_intrinsics[3]*x**9
        
    #Newton method to compute inverse KB kb_intrinsics
    def KB_inv(r_u):
        prec = np.full_like(r_u,1e-8)
        x_n = r_u 
        for _ in range(100):
            x = x_n
            fun = -r_u + x + kb_intrinsics[0]*x**3 + kb_intrinsics[1]*x**5 + kb_intrinsics[2]*x**7 + kb_intrinsics[3]*x**9
            dif = 1 + 3*kb_intrinsics[0]*x**2 + 5*kb_intrinsics[1]*x**4 + 7*kb_intrinsics[2]*x**6 + 9*kb_intrinsics[3]*x**8
            x_n = x - fun/dif
            err = np.absolute(x_n-x)
            if err.all() < prec.all():
                return x_n
        return x_n


    def KB_coord(img_W,img_H,k_W,k_H,u,v,rays,h_grid,w_grid):
        scale = (camres_W+pad_W) / img_W 
        cx = fe_cx * ((img_W-(pad_W/scale))/camres_W)
        cy = fe_cy * ((img_H-(pad_H/scale))/camres_H)

        fxk = fx * ((img_W-(pad_W/scale))/camres_W)
        fyk = fy * ((img_H-(pad_H/scale))/camres_H)

        mx = (u-cx)/(fxk)
        my = (v-cy)/(fyk)
        r0_0 = math.sqrt(mx**2 + my**2)

        phi = math.atan2(my,mx)
        theta = KB_inv(r0_0)

        ROT = rotation_matrix((0,0,1),phi)
        ROT = torch.matmul(ROT,rotation_matrix((0,1,0),theta))#np.eye(3)

        rays = torch.matmul(ROT,rays)
        rays = rays.reshape(3,k_H,k_W)

        phi_k = torch.atan2(rays[1,...],rays[0,...]).T
        theta_k = torch.acos(torch.clamp(rays[2,...],-1,1)).T

        r = KB_dir(theta_k)

        u_k = (fxk * r * torch.cos(phi_k) + cx)
        v_k = (fyk * r * torch.sin(phi_k) + cy)

        roi_x = w_grid + u
        roi_y = h_grid + v
        
        offsets_x = u_k - roi_x
        offsets_y = v_k - roi_y
        offsets_x = torch.clamp(offsets_x,-u,img_W-u)
        offsets_y = torch.clamp(offsets_y,-v,img_H-v)    
        return offsets_x, offsets_y
    
    offset = torch.zeros(2*k_H*k_W,img_H,img_W, device='cpu', dtype=torch.float)

    alpha = k_W * math.radians(fish_FOV/float(img_W))  # Field of view "perspective" 
    d = float(k_W) / (2 * math.tan(alpha/2))         # Focal length "perspective"  
    h_grid,w_grid = torch.meshgrid(torch.arange(-(k_W//2),(k_W//2)+1),
                                    torch.arange(-(k_H//2),(k_H//2)+1))
    h_grid = torch.tensor(h_grid,device='cpu',dtype=torch.float)
    w_grid = torch.tensor(w_grid,device='cpu',dtype=torch.float)

    K = torch.tensor([[d,0,0],[0,d,0],[0.,0.,1.]], device='cpu', dtype=torch.float)
    inv_K = torch.inverse(K)
    rays = torch.stack([w_grid,h_grid,torch.ones(h_grid.shape, device='cpu', dtype=torch.float)],0)
    rays = torch.matmul(inv_K,rays.reshape(3,k_H*k_W))
    rays /= torch.norm(rays,dim=0,keepdim=True)
    count = 0
    for v in range(0, img_H, s_height): 
        for u in range(0, img_W, s_width): 
            offsets_x, offsets_y = KB_coord(img_W,img_H,k_W,k_H,u,v,rays,h_grid,w_grid)
            count+= 1
            offsets = torch.cat((torch.unsqueeze(offsets_y,-1),torch.unsqueeze(offsets_x,-1)),dim=-1)
            total_offsets = offsets.flatten()
            offset[:,v,u] = total_offsets
            
    offset = torch.unsqueeze(offset, 0)
    offset.requires_grad_(False)
    return offset



class FEConv2d(DeformConv2d):
    def __init__(self, in_channels, out_channels, kernel, stride=1, dilation=1, groups=1, bias=False):
        super().__init__(in_channels, out_channels, kernel, stride, 0, dilation, groups, bias)
        self.stride = stride
        self.kernel = kernel

    def forward(self, x):
        device = x.device
        bs = x.size()[0]
        ih, iw = x.size()[-2:]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride, self.stride
        offset = distortion_aware_map(iw,ih,kw,kh,sw,sh).to(device)
        offset = torch.cat([offset for _ in range(bs)],dim=0)
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        if x.shape[0] != offset.shape[0] :
            sizediff = offset.shape[0] - x.shape[0]
            offset = torch.split(offset,[x.shape[0],sizediff],dim=0)
            offset = offset[0]
        return deform_conv2d(x, offset, self.weight, self.bias, self.stride, self.padding, self.dilation) 