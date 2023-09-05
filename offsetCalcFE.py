import cv2
import math
import torch
import numpy as np
from tqdm import tqdm,trange

torch.set_printoptions(precision=4,linewidth=1024,sci_mode=False)

def offcalc(intrinsics=None):
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
    fw = camres_W+pad_W
    fh = camres_H+pad_H

    # time1=time.time()
    def print_offsets(v,u,img_H,img_W,offsets_x,offsets_y,h_grid,w_grid,count):
        img = cv2.resize(cv2.imread('camera_Fisheye_195.png'),(img_W,img_H))
        kernel = np.zeros_like(img)
        p_v = v + (np.asarray(offsets_y[0::3,0::3] + h_grid[0::3,0::3],dtype=np.int32).reshape(-1,))
        p_u = u + (np.asarray(offsets_x[0::3,0::3] + w_grid[0::3,0::3],dtype=np.int32).reshape(-1,))
        for i in range(p_u.shape[0]):
            p_v[i] = p_v[i] if p_v[i] < img_H else img_H-1
            p_v[i] = p_v[i] if p_v[i] > 0 else 0
            p_u[i] = p_u[i] if p_u[i] < img_W else img_W-1
            p_u[i] = p_u[i] if p_u[i] > 0 else 0
        kernel[p_v,p_u,:] = (0,0,255)
        kernel[v,u] = (255,0,0)
        kernel[p_v[0],p_u[0]] = (0,255,0)
        kernel = cv2.dilate(kernel,(3,3))
        out = cv2.addWeighted(kernel, 1, img, 0.75, 0)
        return np.array(out,dtype=np.uint8)

    def rotation_matrix(axis, theta):
        """
        Return the rotation matrix associated with counterclockwise rotation about
        the given axis by theta radians.
        """
        axis = torch.as_tensor(axis, device='cpu', dtype=torch.float)
        if theta == 0 or torch.linalg.norm(axis)==0:
            return torch.eye(3)
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

        vec = torch.FloatTensor([np.cos(phi)*np.sin(theta),
                                 np.sin(phi)*np.sin(theta),
                                 np.cos(theta)])

        axis = torch.cross(vec,torch.FloatTensor([0,0,1]))
        ROT = rotation_matrix(axis,-theta)
        rays = torch.matmul(ROT,rays)
        
        rays = rays.reshape(3,k_H,k_W)

        phi_k = torch.atan2(rays[1,...],rays[0,...])
        theta_k = torch.acos(torch.clamp(rays[2,...],-1,1))
    
        r = KB_dir(theta_k)

        u_k = (fxk * r * torch.cos(phi_k) + cx)
        v_k = (fyk * r * torch.sin(phi_k) + cy)

        oK_x = w_grid + u
        oK_y = h_grid + v
        
        offsets_x = u_k - oK_x
        offsets_y = v_k - oK_y
        return offsets_x, offsets_y

    
    def distortion_aware_map(img_W, img_H, k_W, k_H, s_width, s_height):
        #n=1
        offset = torch.zeros(2*k_H*k_W,img_H,img_W, device='cpu', dtype=torch.float)

        alpha = k_W/float(img_W) * math.radians(fish_FOV)  # Field of view "perspective" 
        # scale = (camres_W+pad_W) / img_W 
        # fxk = fx * ((img_W-(pad_W/scale))/camres_W)
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
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter('Video_FEconv_FE195_test.mp4',fourcc,60,(int(fw),int(fh)),True)
        for v in trange(0, img_H, s_height, leave=True): 
            for u in range(0, img_W, s_width): 
                offsets_x, offsets_y = KB_coord(img_W,img_H,k_W,k_H,u,v,rays,h_grid,w_grid)
                frame = print_offsets(v,u,img_H,img_W,offsets_x,offsets_y,h_grid,w_grid,count)
                frame = cv2.resize(frame,(int(fw),int(fh)),interpolation=cv2.INTER_LANCZOS4)
                video.write(frame)
                count+= 1
                offsets = torch.cat((torch.unsqueeze(offsets_y,-1),torch.unsqueeze(offsets_x,-1)),dim=-1)
                total_offsets = offsets.flatten()
                offset[:,v,u] = total_offsets
                
        offset = torch.unsqueeze(offset, 0)
        offset.requires_grad_(False)
        video.release()
        return offset
    '''
    List of feature maps and convolutions. 
    This list is defined as:
        - (Height,Width) of the feature maps
        - (Kernel size) of the convolution ->  WARNING only odd size kernels are tested !!!!
        - (Convolution stride)
    Feature maps resolution is defined as the original resolution of the calibrated camera
        divided by power of 2 (standard for many deep learning applications)
    '''
                #(H        ,W  )      ,(k,k),(s,s))
    paramlist =[((int(fh)  ,int(fw))  ,(7,7),(2,2)),
                ((int(fh)  ,int(fw))  ,(5,5),(2,2)),
                ((int(fh)  ,int(fw))  ,(5,5),(1,1)),
                ((int(fh)  ,int(fw))  ,(3,3),(1,1)),
                
                ((int(fh/2),int(fw/2)),(7,7),(2,2)),
                ((int(fh/2),int(fw/2)),(5,5),(2,2)),
                ((int(fh/2),int(fw/2)),(5,5),(1,1)),
                ((int(fh/2),int(fw/2)),(3,3),(1,1)),
                        
                ((int(fh/4),int(fw/4)),(7,7),(2,2)),
                ((int(fh/4),int(fw/4)),(5,5),(2,2)),
                ((int(fh/4),int(fw/4)),(5,5),(1,1)),
                ((int(fh/4),int(fw/4)),(3,3),(2,2)),
                ((int(fh/4),int(fw/4)),(3,3),(1,1)),
                        
                ((int(fh/8),int(fw/8)),(7,7),(2,2)),
                ((int(fh/8),int(fw/8)),(5,5),(2,2)),
                ((int(fh/8),int(fw/8)),(5,5),(1,1)),
                ((int(fh/8),int(fw/8)),(3,3),(2,2)),
                ((int(fh/8),int(fw/8)),(3,3),(1,1)),
                
                ((int(fh/16),int(fw/16)),(5,5),(2,2)),
                ((int(fh/16),int(fw/16)),(5,5),(1,1)),
                ((int(fh/16),int(fw/16)),(3,3),(2,2)),
                ((int(fh/16),int(fw/16)),(3,3),(1,1)),
                        
                ((int(fh/32),int(fw/32)),(5,5),(2,2)),
                ((int(fh/32),int(fw/32)),(5,5),(1,1)),
                ((int(fh/32),int(fw/32)),(3,3),(2,2)),
                ((int(fh/32),int(fw/32)),(3,3),(1,1)),
                        
                ((int(fh/64),int(fw/64)),(5,5),(2,2)),
                ((int(fh/64),int(fw/64)),(5,5),(1,1)),
                ((int(fh/64),int(fw/64)),(3,3),(2,2)),
                ((int(fh/64),int(fw/64)),(3,3),(1,1)),
                
                ((int(fh/128),int(fw/128)),(5,5),(2,2)),
                ((int(fh/128),int(fw/128)),(5,5),(1,1)),
                ((int(fh/128),int(fw/128)),(3,3),(2,2)),
                ((int(fh/128),int(fw/128)),(3,3),(1,1))] 

    offsetdict={}
    for key in tqdm(paramlist):
        offsetdict[key] = distortion_aware_map(key[0][1],key[0][0],key[1][1],key[1][0],s_width=key[2][1],s_height=key[2][0])
    torch.save(offsetdict,'./offsetFE195.pt') 

if __name__ == '__main__':
 
    offcalc()
    
