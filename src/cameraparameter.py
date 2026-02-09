import torch
import numpy as np
import math
import cv2

class CameraParameter:
    def __init__(self, P:torch.tensor, imgsize:tuple):
        if type(P) == np.ndarray:
            P = torch.from_numpy(P.astype(np.float32)).clone()
        self.rotationorder = 'XYZ'
        self.P = P
        self.Pinv = torch.pinverse(P)
        out = self.decompose_P()
        #self.out = out
        self.A = torch.from_numpy(out[0].astype(np.float32)).clone()
        self.R = torch.from_numpy(out[1].astype(np.float32)).clone()
        self.camera_pos = torch.from_numpy((out[2][0:3]/out[2][3]).ravel().astype(np.float32)).clone()
        self.xangle, self.yangle, self.zangle = self.decompose_R()
        
        #OpenCVでは右手系を採用しています！　回転角度は左手系に修正しています！
                
        #self.f = (out[0][0][0] + out[0][1][1])/2
        self.f = out[0][1][1]
        self.focullength = out[0][0][0]*24/imgsize[0]
        self.fov = math.degrees(2*math.atan(imgsize[0]/(2*out[0][1][1])))
        self.fx = out[0][0][0]
        self.fy = out[0][1][1]
        self.aspect = self.fx/self.fy
        
        self.imagesize = imgsize[:2]
        
        # 回転は X軸 -> Y軸 -> Z軸の順
        
    def update(self, grad:torch.tensor):
        #print(grad)
        self.fov = grad[0].item()
        self.f = self.calc_f(grad[0])
        self.A = self.calc_A(self.f)
        self.R = self.calc_R(grad[1:4])
        Pt = self.A @ self.calc_Rt(self.R, self.calc_t(self.R))
        self.P = Pt/Pt[2][3]
        self.Pinv = torch.pinverse(Pt/Pt[2][3])
        self.focullength = self.f.item()*24/self.imagesize[0]
        self.fx = self.f.item()*self.aspect
        self.fy = self.f.item()
        self.xangle = grad[1].item()
        self.yangle = grad[2].item()
        self.zangle = grad[3].item()
        #print(P)
        #return P
        
    def decompose_P(self) -> np.ndarray:
        out = cv2.decomposeProjectionMatrix(self.P.to('cpu').detach().numpy().copy())
        # 内部パラメータの右下を１にしたいので再度計算
        out = cv2.decomposeProjectionMatrix(self.P.to('cpu').detach().numpy().copy()/out[0][2][2])
        return out
    
    def decompose_R(self) -> tuple[float, float, float]:
        R = torch.t(self.R)
        if self.rotationorder == 'XYZ':
            yrad = torch.asin(-R[2][0])
            xrad = (torch.asin(R[2][1]/torch.cos(yrad))+torch.acos(R[2][2]/torch.cos(yrad)))/2
            zrad = (torch.acos(R[0][0]/torch.cos(yrad))+torch.asin(R[1][0]/torch.cos(yrad)))/2   
        elif (self.rotationorder == 'ZXY'):
            xrad = torch.asin(-R[1][2])
            yrad = (torch.asin(R[0][2]/torch.cos(xrad))+torch.acos(R[2][2]/torch.cos(xrad)))/2
            zrad = (torch.asin(R[1][0]/torch.cos(xrad))+torch.acos(R[1][1]/torch.cos(xrad)))/2
        else:
            self.rotationorder = 'XYZ'
            yrad = torch.asin(-R[2][0])
            xrad = (torch.asin(R[2][1]/torch.cos(yrad))+torch.acos(R[2][2]/torch.cos(yrad)))/2
            zrad = (torch.acos(R[0][0]/torch.cos(yrad))+torch.asin(R[1][0]/torch.cos(yrad)))/2
                
        return math.degrees(xrad), math.degrees(yrad), math.degrees(zrad)
        
        
    def calc_R(self, angles:torch.tensor) -> torch.tensor:
        rad = torch.full_like(angles, math.pi/180, dtype=torch.float32)*angles
        cos_sin = torch.t(torch.stack((torch.cos(rad), torch.sin(rad), torch.ones(3))))

        Rx = (torch.tensor([[0,0,1],[0,0,0],[0,0,0],[0,0,0],[1,0,0],[0,-1,0],[0,0,0],[0,1,0],[1,0,0]], dtype=torch.float32) @ cos_sin[0]).view(3, 3)
        Ry = (torch.tensor([[1,0,0],[0,0,0],[0,1,0],[0,0,0],[0,0,1],[0,0,0],[0,-1,0],[0,0,0],[1,0,0]], dtype=torch.float32) @ cos_sin[1]).view(3, 3)
        Rz = (torch.tensor([[1,0,0],[0,-1,0],[0,0,0],[0,1,0],[1,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,1]], dtype=torch.float32) @ cos_sin[2]).view(3, 3)
                
        # カメラの回転は x軸 -> y軸 -> z軸 の順
        # よって変換行列は X軸 -> Y軸 -> Z軸 の順
        #return torch.t(Rx @ Ry @ Rz)
        if self.rotationorder == 'XYZ':
            return torch.t(Rz @ Ry @ Rx) # 右手系左手系の違い...
        elif self.rotationorder == 'ZXY':
            return torch.t(Ry @ Rx @ Rz)
        else:
            self.rotationorder = 'XYZ'
            return torch.t(Rz @ Ry @ Rx)
    
    def calc_t(self, R:torch.tensor) -> torch.tensor:
        return -R @ self.camera_pos

    def calc_Rt(self, R:torch.tensor, t:torch.tensor) -> torch.tensor:
        return torch.cat((R,t.view(-1,1)),1)
    
    def calc_f(self, fov:torch.tensor) ->torch.tensor:
        f = self.imagesize[0]/(2*torch.tan(torch.deg2rad(fov/2)))
        return f

    def calc_A(self, f:torch.tensor) -> torch.tensor:
        # 内部パラメータ
        return torch.stack((f*self.aspect, self.A[0][1], self.A[0][2], self.A[1][0], f, self.A[1][2], self.A[2][0], self.A[2][1], self.A[2][2])).view(3,3)
        #return torch.stack((f, self.A[0][1], self.A[0][2], self.A[1][0], f, self.A[1][2], self.A[2][0], self.A[2][1], self.A[2][2])).view(3,3)

    def calc_P(self, grad:torch.tensor) -> torch.tensor:
        #print(grad)
        f = self.calc_f(grad[0])
        A = self.calc_A(f)
        R = self.calc_R(grad[1:4])
        Pt = A @ self.calc_Rt(R, self.calc_t(R))
        P = Pt/Pt[2][3]
        #print(P)
        return P
    

class CameraParameterforUnity:
    def __init__(self, f:float, r:np.ndarray, camera_pos:np.ndarray):
        self.f = f
        self.r = np.array(r, dtype='float64')
        self.camera_pos = np.array(camera_pos, dtype='float64')
        self.R = self.calc_R()
        self.t = self.calc_t()
        self.Rt = self.calc_Rt()
        self.A = self.calc_A()
        self.P = self.calc_P()
    
    def calc_R(self):
        theta = np.deg2rad(self.r[0])
        phi = np.deg2rad(self.r[1])
        psi = np.deg2rad(self.r[2])

        # 回転行列
        Rx = np.array([[1,0,0],[0, math.cos(theta), -math.sin(theta)],[0, math.sin(theta), math.cos(theta)]])
        Ry = np.array([[math.cos(phi), 0, math.sin(phi)],[0, 1, 0],[-math.sin(phi), 0, math.cos(phi)]])
        Rz = np.array([[math.cos(psi), -math.sin(psi), 0],[math.sin(psi), math.cos(psi), 0],[0, 0, 1]])
        
        R = (Ry @ Rx @ Rz).T # Y軸 -> X軸 -> Z軸の順
        #R = R_Z.T @ R_X.T @ R_Y.T
        return R

    def calc_t(self):
        return -self.R @ self.camera_pos

    def calc_Rt(self):
        return np.concatenate([self.R, self.t.reshape((-1,1))], axis=1)

    def calc_A(self):
        # 内部パラメータ
        A = np.array([[self.f,0,0],[0,self.f,0],[0,0,1]]) #カメラひずみ、スキュー、光軸ずれはなしとする
        return A

    def calc_P(self):
        P = self.A @ self.Rt
        P = P/P[2][3]
        return P
    
def calc_f(fov:float, length:int) -> np.ndarray:
    #焦点距離計算
    f = length/(2*math.tan(math.radians(fov/2)))
    return f