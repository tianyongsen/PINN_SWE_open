import numpy as np
import deepxde as dde
# import scipy
import torch
from typing import Sequence,Any,Callable

from . import baseclass


class Steady_State_Maxwell_3D(baseclass.BasePDE):
    def __init__(self,
                datapath=None, 
                # bcs:Sequence[dict],
                # geom:list[list,list], 
                # loss_weights:np.array
                ):
        super().__init__()
        """The 3D steady state Maxwell's equation
            Args:
                datapath: path to reference data
                ics: initial conditions, list of dicts with keys 'component', 'function', 'bc', 'type'. See add_bcs() for details.
                bcs: boundary conditions, list of dicts with keys 'component', 'function', 'bc', 'type'. See add_bcs() for details.
                geom: geometry domain
        """
        # output dim
        self.output_dim = 5   

        # domain
        axis_center=np.array( [0,0,0])
        radius=0.003
        length=0.06
        x_min=np.array([axis_center[0],-radius,-radius])  #
        x_max=np.array([axis_center[0]+length,radius,radius])
        self.bbox =[x_min[0],x_max[0],x_min[1],x_max[1],x_min[2],x_max[2]]

        self.geom=dde.geometry.Cylinder(radius,length,axis_center=(0,0,0))
        # self.sigma=torch.tensor([10,0.001])

        # PDE
        def steady_state_Maxwell_3D(inputs, outputs):
            """pde of the steady state Maxwell's equation
                inputs:  (x,y,z)
                outputs: ( Jx, Jy, Jz, V)
            """
            Jx, Jy, Jz, V, sigma = outputs[:, 0:1], outputs[:, 1:2], outputs[:, 2:3], outputs[:, 3:4],outputs[:, 4:5]
            # dJx_x= dde.grad.jacobian(outputs, inputs, i=0, j=0)
            # dJy_y= dde.grad.jacobian(outputs, inputs, i=1, j=1)
            # dJz_z= dde.grad.jacobian(outputs, inputs, i=2, j=2)
            dV_x= dde.grad.jacobian(outputs, inputs, i=3, j=0)
            dV_y= dde.grad.jacobian(outputs, inputs, i=3, j=1)
            dV_z= dde.grad.jacobian(outputs, inputs, i=3, j=2)
            dV_xx= dde.grad.hessian(outputs, inputs, component=3, i=0, j=0)
            dV_yy= dde.grad.hessian(outputs, inputs, component=3, i=1, j=1)
            dV_zz= dde.grad.hessian(outputs, inputs, component=3, i=2, j=2)
            
            with torch.no_grad():
                #小于raius的区域，电导率sigma=10,否则电导率为0.0001
                a=torch.norm(inputs[:,1:3],dim=1)<=0.003
                sigma_true=a*5.8e7+(~a)*1e-5
                
            #电导率条件
            res_sigma=sigma_true-sigma
            
                
            #Gauss's Law of Electrostatics.(静电场高斯定理)
            res_G=sigma*dV_xx+sigma*dV_yy+sigma*dV_zz
            
            #微观形式的欧姆定律
            res_O_x=Jx+sigma*dV_x 
            res_O_y=Jy+sigma*dV_y
            res_O_z=Jz+sigma*dV_z

            

            return [res_G,res_O_x, res_O_y, res_O_z,res_sigma]

        self.pde = steady_state_Maxwell_3D

        self.set_pdeloss(num=4)   #4 items in defalut, see the return values of the pde funciton

        #---------------------------------------------
        if datapath is not None:
            self.load_ref_data(datapath,t_transpose=True) 
        def boundary_x_0(x, on_boundary):
            return on_boundary and np.isclose(x[0], axis_center[0])

        def boundary_x_1(x, on_boundary):
            return on_boundary and np.isclose(x[0], axis_center[0]+length)
        
        self.count=0
        def boundary_side(x,on_boundary):
            return on_boundary and ~np.logical_or(np.isclose(x[0],axis_center[0]),
                                                  np.isclose(x[0], axis_center[0]+length)
            )
   
        
        # add boundary conditions
        bcs=[
            {'bc':boundary_x_0,
             'component':3,
             'function':lambda x: 1.0, # 电势为1
             'type':'Dirichlet'
             },
            {'bc':boundary_x_1,
             'component':3,
             'function':lambda x: 0, #电势为0
             'type':'Dirichlet'
             },
            {'bc':boundary_side,
             'component':3,
             'function':lambda x: 0, #电势梯度为0
             'type':'Neumann'
             }
        ]

        self.add_bcs(bcs)   

        # [loss_weights(pde),loss_weights(bcs)] 4+3
        self.loss_weights =[1.,1.,1.,1.,10.,10.,10.]

        # train settings
        self.training_points(mul=1)

