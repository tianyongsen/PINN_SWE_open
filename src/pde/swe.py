import numpy as np
import deepxde as dde
import scipy
import torch
from typing import Sequence,Any,Callable

#--
from . import baseclass   
# from . import z_net

GRAVITY=9.80665   #gravity acceleration
EPSILON=0.001     #small value to avoid division by zero

class SWE2D(baseclass.BaseTimePDE):

    def __init__(self, 
                datapath:str, 
                ics:Sequence[dict],
                bcs:Sequence[dict],
                z_net:Callable,
                geom:list[list,list], 
                time:list,    
                loss_weights:np.array,
                pde_form: str ="VAR",
                rain_func:Callable=None,
                mul:int=4
                ):
        super().__init__()
        """The 2D shallow water equation without S_f term (friction term)
        Args:
            datapath: path to reference data
            ics: initial conditions, list of dicts with keys 'component', 'function', 'bc', 'type'. See add_bcs() for details.
            bcs: boundary conditions, list of dicts with keys 'component', 'function', 'bc', 'type'. See add_bcs() for details.
            z_net: z_net object, see z_net.py. Provides the topography function z and its derivative respect to x.
            geom: geometry domain with [[xmin,xmax],[ymin,ymax]]
            time: time domain with [t0,t1]
            pde_form: string, see PDE_FORM below
            rain_func: function, rainfall intensity function
            mul: int, the multiply of the default training points number, default is 4.
        """
        #basic check for pde_form
        if pde_form not in ["VAR", "CONSER", 
                            "VAR_ENTROPY_STABLE", 
                            "VAR_ENTROPY_CONSERVATION",
                            "VAR_ENTROPY_DT",
                            "CONSER_ENTROPY", 
                            "VAR_RAIN",
                            'VAR_ENTROPY_RAIN',
                            "VAR_PRIMITIVE"]:
            raise ValueError("pde_form should be in pde_form list, see the class SWE2D for details")
        #basic check for rain_func and pde_form
        if rain_func is not None and pde_form not in ["VAR_RAIN","VAR_ENTROPY_RAIN"]:
            raise ValueError("pde_form should be 'VAR_RAIN' or 'VAR_ENTROPY_RAIN' when rainf_func is not None")
            
        # output dim
        self.output_dim = 3   #(h, u, v) or (h, hu, hv)
        # if pde_form in ["CONSER", "CONSER_ENTROPY"]:
        #     self.output_dim = 6  

        #topography
        self.z_net = z_net

        #rainfall intensity function
        self.rain_func = rain_func

        # domain
        self.bbox = [geom[0][0], geom[0][1], geom[1][0], geom[1][1], time[0], time[1]]
        self.geom = dde.geometry.Rectangle(self.bbox[0:4:2], self.bbox[1:4:2])
        timedomain = dde.geometry.TimeDomain(self.bbox[4], self.bbox[5])
        self.geomtime = dde.geometry.GeometryXTime(self.geom, timedomain)

        if pde_form in ["VAR", "VAR_RAIN", "VAR_PRIMITIVE"]: 
            self.set_pdeloss(num=6)   #6 items in defalut, see the return values of the pde funciton, such as swe_2d_var
        # elif pde_form in ["CONSER"]:
        #     self.set_pdeloss(num=9)   
        elif pde_form in ["VAR_ENTROPY_STABLE", "VAR_ENTROPY_RAIN","VAR_ENTROPY_CONSERVATION"]:                                       
            self.set_pdeloss(num=7)   #7 items in defalut, see the return values of thepde funciton, such as swe_2d_var
        
        #---------------------------------------------
        if datapath is not None:
            self.load_ref_data(datapath,t_transpose=True) 

        # define bounary conditions. point=(x,y,t)
        def boundary_ic(point, on_initial):  
            return on_initial and np.isclose(point[2], time[0])

        def boundary_xb_left(point, on_boundary):
            return on_boundary and np.isclose(point[0], self.bbox[0])
        def boundary_xb_right(point, on_boundary):
            return on_boundary and np.isclose(point[0], self.bbox[1])        
        def boundary_yb_bottom(point, on_boundary):
            return on_boundary and np.isclose(point[1], self.bbox[2])
        def boundary_yb_top(point, on_boundary):
            return on_boundary and np.isclose(point[1], self.bbox[3])
        
        # Uniformly add boundary conditions and initial conditions
        for ic in ics: 
            ic['bc'] = boundary_ic

        for bc in bcs:
            if bc['bc'] =='left': 
                bc['bc'] = boundary_xb_left
            elif bc['bc'] =='right':
                bc['bc'] = boundary_xb_right
            elif bc['bc'] =='bottom':
                bc['bc'] = boundary_yb_bottom
            elif bc['bc'] =='top':
                bc['bc'] = boundary_yb_top
            else:
                raise ValueError("bc['bc'] should be 'left' or 'right' or 'bottom' or 'top'")
        bcs=bcs+ics
        self.add_bcs(bcs)   
        
        self.loss_weights = loss_weights  # [loss_weights(pde),loss_weights(bcs),loss_weights(ics)] 

        # train settings
        self.training_points(mul=mul)

        # PDEs
        def swe_2d_var(x, U):
            """variables form of the swe is : (x,t)-->(h,u,v). This function is without entropy condition"""
            z,dz_dx,dz_dy=self.z_net(x[:,0:2])     #z and its derivative respect to x,y

            h,u,v=U[:,0:1],U[:,1:2],U[:,2:3]

            h_x=dde.grad.jacobian(U,x,i=0,j=0)
            h_y=dde.grad.jacobian(U,x,i=0,j=1)
            h_t=dde.grad.jacobian(U,x,i=0,j=2)
            u_x=dde.grad.jacobian(U,x,i=1,j=0)
            u_y=dde.grad.jacobian(U,x,i=1,j=1)
            u_t=dde.grad.jacobian(U,x,i=1,j=2)
            v_x=dde.grad.jacobian(U,x,i=2,j=0)
            v_y=dde.grad.jacobian(U,x,i=2,j=1)
            v_t=dde.grad.jacobian(U,x,i=2,j=2)

            #mass conservation equation or mass equation residue
            mass_equ= h_t + h_x*u  + h*u_x + h_y*v+ h*v_y   
            #momentum conservation equation or momentum equation residue
            momentum_equ_x=u*h_t+ h*u_t+ u**2*h_x + 2*h*u*u_x+u*v*h_y+ h*v*u_y+h*u*v_y +GRAVITY*h*(h_x+dz_dx)
            momentum_equ_y=v*h_t + h*v_t+ v**2*h_y + 2*h*v*v_y+u*v*h_x+ h*u*v_x+h*v*u_x +GRAVITY*h*(h_y+dz_dy)
            #non-negative condition and dry condition
            with torch.no_grad():
                heaviside_h=torch.heaviside(h-0.001,torch.tensor(0.))
            h_positive=h-heaviside_h*h 
            u_dry=u-heaviside_h*u
            v_dry=v-heaviside_h*v


            return [mass_equ,momentum_equ_x,momentum_equ_y,h_positive,u_dry,v_dry]  #6 items
        
        def swe_2d_var_entropy_stable(x, U):  
            """variables form of the swe is : (x,t)-->(h,u,v). This function is with entropy condition"""
            z,z_x,z_y=self.z_net(x[:,0:2])     #z and its derivative respect to x,y
            h,u,v=U[:,0:1],U[:,1:2],U[:,2:3]

            h_x=dde.grad.jacobian(U,x,i=0,j=0)
            h_y=dde.grad.jacobian(U,x,i=0,j=1)
            h_t=dde.grad.jacobian(U,x,i=0,j=2)
            u_x=dde.grad.jacobian(U,x,i=1,j=0)
            u_y=dde.grad.jacobian(U,x,i=1,j=1)
            u_t=dde.grad.jacobian(U,x,i=1,j=2)
            v_x=dde.grad.jacobian(U,x,i=2,j=0)
            v_y=dde.grad.jacobian(U,x,i=2,j=1)
            v_t=dde.grad.jacobian(U,x,i=2,j=2)

            #mass conservation equation or mass equation residue#temp variables
            u_squre,v_squre,uv,gh,gz=u**2,v**2,u*v,GRAVITY*h,GRAVITY*z
            hu,hv=h*u,h*v
            huv=hu*v          

            #mass conservation equation or mass equation residue
            mass_equ= h_t + (h_x*u  + h*u_x + h_y*v+ h*v_y)/100.    #你是什么傻逼，怎么会带这个。误了多少事情！！！
            #momentum conservation equation or momentum equation residue
            momentum_equ_x=u*h_t+ h*u_t+ u_squre*h_x + 2*hu*u_x+uv*h_y+ hv*u_y+hu*v_y +gh*(h_x+z_x)
            momentum_equ_y=v*h_t + h*v_t+ v_squre*h_y + 2*hv*v_y+uv*h_x+ hu*v_x+hv*u_x +gh*(h_y+z_y)
            #non-negative condition and dry condition
            with torch.no_grad():
                heaviside_h=torch.heaviside(h-0.001,torch.tensor(0.))
            h_positive=h-heaviside_h*h 
            u_dry=u-heaviside_h*u
            v_dry=v-heaviside_h*v
            #entropy condition
            entropy_condition=0.5*(u_squre+v_squre+2*gh+gz)*h_t + u*(0.5*(u_squre+v_squre)+2*gh+gz)*h_x+\
                             v*(0.5*(u_squre+v_squre)+2*gh+gz)*h_y + \
                             hu*u_t +h*(1.5*u_squre+0.5*v_squre+gh+gz)*u_x + huv*u_y + \
                             hv*v_t + huv* v_x + h*(0.5*u_squre+1.5*v_squre+gh+gz)*v_y+\
                             gh*(u*z_x+v*z_y)
            entropy_condition=torch.relu(entropy_condition)   #less than 0

            return [mass_equ,momentum_equ_x,momentum_equ_y,entropy_condition,h_positive,u_dry,v_dry]  #7 items

                
        def swe_2d_var_entropy_conservation(x, U):  
            """variables form of the swe is : (x,t)-->(h,u,v). This function is with entropy condition"""
            z,z_x,z_y=self.z_net(x[:,0:2])     #z and its derivative respect to x,y
            h,u,v=U[:,0:1],U[:,1:2],U[:,2:3]

            h_x=dde.grad.jacobian(U,x,i=0,j=0)
            h_y=dde.grad.jacobian(U,x,i=0,j=1)
            h_t=dde.grad.jacobian(U,x,i=0,j=2)
            u_x=dde.grad.jacobian(U,x,i=1,j=0)
            u_y=dde.grad.jacobian(U,x,i=1,j=1)
            u_t=dde.grad.jacobian(U,x,i=1,j=2)
            v_x=dde.grad.jacobian(U,x,i=2,j=0)
            v_y=dde.grad.jacobian(U,x,i=2,j=1)
            v_t=dde.grad.jacobian(U,x,i=2,j=2)

            #mass conservation equation or mass equation residue#temp variables
            u_squre,v_squre,uv,gh,gz=u**2,v**2,u*v,GRAVITY*h,GRAVITY*z
            hu,hv=h*u,h*v
            huv=hu*v          

            #mass conservation equation or mass equation residue
            mass_equ= h_t + (h_x*u  + h*u_x + h_y*v+ h*v_y)/100.    #你是什么傻逼，怎么会带这个。误了多少事情！！！
            #momentum conservation equation or momentum equation residue
            momentum_equ_x=u*h_t+ h*u_t+ u_squre*h_x + 2*hu*u_x+uv*h_y+ hv*u_y+hu*v_y +gh*(h_x+z_x)
            momentum_equ_y=v*h_t + h*v_t+ v_squre*h_y + 2*hv*v_y+uv*h_x+ hu*v_x+hv*u_x +gh*(h_y+z_y)
            #non-negative condition and dry condition
            with torch.no_grad():
                heaviside_h=torch.heaviside(h-0.001,torch.tensor(0.))
            h_positive=h-heaviside_h*h 
            u_dry=u-heaviside_h*u
            v_dry=v-heaviside_h*v
            #entropy condition
            entropy_condition=0.5*(u_squre+v_squre+2*gh+gz)*h_t + u*(0.5*(u_squre+v_squre)+2*gh+gz)*h_x+\
                             v*(0.5*(u_squre+v_squre)+2*gh+gz)*h_y + \
                             hu*u_t +h*(1.5*u_squre+0.5*v_squre+gh+gz)*u_x + huv*u_y + \
                             hv*v_t + huv* v_x + h*(0.5*u_squre+1.5*v_squre+gh+gz)*v_y+\
                             gh*(u*z_x+v*z_y)

            return [mass_equ,momentum_equ_x,momentum_equ_y,entropy_condition,h_positive,u_dry,v_dry]  #7 items


        def swe_2d_var_entropy_DT(x, U):  
            """variables form of the swe is : (x,t)-->(h,u,v). 
               This function is with entropy condition and dimensional transformation"""
            z,z_x,z_y=self.z_net(x[:,0:2])     #z and its derivative respect to x,y
            h,u,v=U[:,0:1],U[:,1:2],U[:,2:3]

            h_x=dde.grad.jacobian(U,x,i=0,j=0)
            h_y=dde.grad.jacobian(U,x,i=0,j=1)
            h_t=dde.grad.jacobian(U,x,i=0,j=2)
            u_x=dde.grad.jacobian(U,x,i=1,j=0)
            u_y=dde.grad.jacobian(U,x,i=1,j=1)
            u_t=dde.grad.jacobian(U,x,i=1,j=2)
            v_x=dde.grad.jacobian(U,x,i=2,j=0)
            v_y=dde.grad.jacobian(U,x,i=2,j=1)
            v_t=dde.grad.jacobian(U,x,i=2,j=2)

            #mass conservation equation or mass equation residue#temp variables
            g_cm_min=9.80665                #cm/min^2
            u_squre,v_squre,gh,gz=u**2,v**2,g_cm_min*h,g_cm_min*z
            hu,hv=h*u,h*v
            huv=hu*v       

            #--cm  and min scale
            #mass conservation equation or mass equation residue
            mass_equ= h_t +(h_x*u  + h*u_x + h_y*v+ h*v_y)/100.   #cm/min
            #momentum conservation equation or momentum equation residue
            momentum_equ_x=u*h_t+ h*u_t  + (u**2*h_x + 2*h*u*u_x+u*v*h_y+ h*v*u_y+h*u*v_y)/100.+g_cm_min*h*(h_x/100.+z_x)  #cm^2/min^2
            momentum_equ_y=v*h_t +h*v_t  + (v**2*h_y + 2*h*v*v_y+u*v*h_x+ h*u*v_x+h*v*u_x)/100. +g_cm_min*h*(h_y/100.+z_y) #cm^2/min^2

            #entropy condition
            entropy_condition=0.5*(u_squre+v_squre+gh+2*gz*100.)*h_t + u*(0.5*(u_squre+v_squre)+2*gh+gz*100.)*h_x/100.+\
                             v*(0.5*(u_squre+v_squre)+2*gh+gz*100.)*h_y/100. + \
                             hu*u_t +h*(1.5*u_squre+0.5*v_squre+gh+gz*100.)*u_x/100. + huv*u_y/100. + \
                             hv*v_t + huv* v_x/100. + h*(0.5*u_squre+1.5*v_squre+gh+gz*100.)*v_y/100.+\
                             gh*(u*z_x+v*z_y)   #cm^3/min^3
            entropy_condition=torch.relu(entropy_condition)   #less than 0

            #non-negative condition and dry condition
            with torch.no_grad():
                heaviside_h=torch.heaviside(h-0.001,torch.tensor(0.))
            h_positive=h-heaviside_h*h 
            u_dry=u-heaviside_h*u
            v_dry=v-heaviside_h*v

            return [mass_equ,momentum_equ_x,momentum_equ_y,entropy_condition,h_positive,u_dry,v_dry]  #7 items



        def swe_2d_var_rain(x, U):  
            """variables form of the swe is : (x,t)-->(h,u,v). 
               This function is with rain term.
               Note: unit (h,u,v)=(cm,cm/s,cm/s)"""
            z,dz_dx,dz_dy=self.z_net(x[:,0:2])     #z and its derivative respect to x,y
            rain=self.rain_func(x)          #rainfall intensity,cm/min
            g_cm_min=9.80665                #cm/s^2

            h,u,v=U[:,0:1],U[:,1:2],U[:,2:3]   #cm,cm/s,cm/s

            h_x=dde.grad.jacobian(U,x,i=0,j=0)
            h_y=dde.grad.jacobian(U,x,i=0,j=1)
            h_t=dde.grad.jacobian(U,x,i=0,j=2)
            u_x=dde.grad.jacobian(U,x,i=1,j=0)
            u_y=dde.grad.jacobian(U,x,i=1,j=1)
            u_t=dde.grad.jacobian(U,x,i=1,j=2)
            v_x=dde.grad.jacobian(U,x,i=2,j=0)
            v_y=dde.grad.jacobian(U,x,i=2,j=1)
            v_t=dde.grad.jacobian(U,x,i=2,j=2)

            #--cm  and min scale
            #mass conservation equation or mass equation residue
            mass_equ= h_t +(h_x*u  + h*u_x + h_y*v+ h*v_y)/100.-rain   #cm/s
            #momentum conservation equation or momentum equation residue
            momentum_equ_x=u*h_t+ h*u_t  + (u**2*h_x + 2*h*u*u_x+u*v*h_y+ h*v*u_y+h*u*v_y)/100. + g_cm_min*h*(h_x/100.+dz_dx)  #cm^2/s^2
            momentum_equ_y=v*h_t +h*v_t  + (v**2*h_y + 2*h*v*v_y+u*v*h_x+ h*u*v_x+h*v*u_x)/100. + g_cm_min*h*(h_y/100.+dz_dy) #cm^2/s^2

            #non-negative condition and dry condition
            with torch.no_grad():
                heaviside_h=torch.heaviside(h-0.001,torch.tensor(0.))
            h_positive=h-heaviside_h*h    #cm
            u_dry=u-heaviside_h*u         #cm/s
            v_dry=v-heaviside_h*v         #cm/s
            return [mass_equ,momentum_equ_x,momentum_equ_y,h_positive,u_dry,v_dry]  #6 items
        
        def swe_2d_var_entropy_rain(x, U):  
            """variables form of the swe is : (x,t)-->(h,u,v). This function is with entropy condition"""
            z,z_x,z_y=self.z_net(x[:,0:2])     #z and its derivative respect to x,y
            rain=self.rain_func(x)          #rainfall intensity,cm/min
            h,u,v=U[:,0:1],U[:,1:2],U[:,2:3]

            h_x=dde.grad.jacobian(U,x,i=0,j=0)
            h_y=dde.grad.jacobian(U,x,i=0,j=1)
            h_t=dde.grad.jacobian(U,x,i=0,j=2)
            u_x=dde.grad.jacobian(U,x,i=1,j=0)
            u_y=dde.grad.jacobian(U,x,i=1,j=1)
            u_t=dde.grad.jacobian(U,x,i=1,j=2)
            v_x=dde.grad.jacobian(U,x,i=2,j=0)
            v_y=dde.grad.jacobian(U,x,i=2,j=1)
            v_t=dde.grad.jacobian(U,x,i=2,j=2)

            #temp variables
            g_cm_min=9.80665                #cm/min^2
            u_squre,v_squre,gh,gz=u**2,v**2,g_cm_min*h,g_cm_min*z
            hu,hv=h*u,h*v
            huv=hu*v       

            #--cm  and min scale
            #mass conservation equation or mass equation residue
            mass_equ= h_t +(h_x*u  + h*u_x + h_y*v+ h*v_y)/100.-rain   #cm/min
            #momentum conservation equation or momentum equation residue
            momentum_equ_x=u*h_t+ h*u_t  + (u**2*h_x + 2*h*u*u_x+u*v*h_y+ h*v*u_y+h*u*v_y)/100.+g_cm_min*h*(h_x/100.+z_x)  #cm^2/min^2
            momentum_equ_y=v*h_t +h*v_t  + (v**2*h_y + 2*h*v*v_y+u*v*h_x+ h*u*v_x+h*v*u_x)/100. +g_cm_min*h*(h_y/100.+z_y) #cm^2/min^2

            #entropy condition
            entropy_condition=0.5*(u_squre+v_squre+gh+2*gz*100.)*h_t + u*(0.5*(u_squre+v_squre)+2*gh+gz*100.)*h_x/100.+\
                             v*(0.5*(u_squre+v_squre)+2*gh+gz*100.)*h_y/100. + \
                             hu*u_t +h*(1.5*u_squre+0.5*v_squre+gh+gz*100.)*u_x/100. + huv*u_y/100. + \
                             hv*v_t + huv* v_x/100. + h*(0.5*u_squre+1.5*v_squre+gh+gz*100.)*v_y/100.+\
                             gh*(u*z_x+v*z_y)-\
                             g_cm_min*(h+z*100.)*rain        #cm^3/min^3
            entropy_condition=torch.relu(entropy_condition)   #less than 0

            #non-negative condition and dry condition
            with torch.no_grad():
                heaviside_h=torch.heaviside(h-0.001,torch.tensor(0.))
            h_positive=h-heaviside_h*h 
            u_dry=u-heaviside_h*u
            v_dry=v-heaviside_h*v

            return [mass_equ,momentum_equ_x,momentum_equ_y,entropy_condition,h_positive,u_dry,v_dry]  #7 items

        # def swe_2d_conser(x, U):  
        #     """conservation form of the swe is : (x,t)-->(h,hu,hv,hu2,hv2,huv). 
        #     This function is without entropy condition and without rain term.
        #     remark: we have test the form of (x,t)-->(h,hu,hv),then construct the hu2, hv2 and huv, 
        #             and the results are bad. Maybe due to the division"""
        #     z,dz_dx,dz_dy=self.z_net(x[:,0:2])     #z and its derivative respect to x,y

        #     h,hu,hv,hu2,hv2,huv=U[:,0:1],U[:,1:2],U[:,2:3],U[:,3:4],U[:,4:5],U[:,5:6]

        #     h_x=dde.grad.jacobian(U,x,i=0,j=0)
        #     h_y=dde.grad.jacobian(U,x,i=0,j=1)
        #     h_t=dde.grad.jacobian(U,x,i=0,j=2)
        #     hu_x=dde.grad.jacobian(U,x,i=1,j=0)
        #     # hu_y=dde.grad.jacobian(U,x,i=1,j=1)
        #     hu_t=dde.grad.jacobian(U,x,i=1,j=2)
        #     # hv_x=dde.grad.jacobian(U,x,i=2,j=0)
        #     hv_y=dde.grad.jacobian(U,x,i=2,j=1)
        #     hv_t=dde.grad.jacobian(U,x,i=2,j=2)
        #     hu2_x=dde.grad.jacobian(U,x,i=3,j=0)
        #     hv2_y=dde.grad.jacobian(U,x,i=4,j=1)
        #     huv_x=dde.grad.jacobian(U,x,i=5,j=0)
        #     huv_y=dde.grad.jacobian(U,x,i=5,j=1)

        #     with torch.no_grad():
        #         heaviside_h=torch.heaviside(h-0.001,torch.tensor(0.))
            
        #     #mass conservation equation or mass equation residue
        #     mass_equ= h_t + hu_x  + hv_y  
        #     #momentum conservation equation or momentum equation residue
        #     momentum_equ_x=hu_t+ hu2_x+huv_y+GRAVITY*h*(h_x+dz_dx)
        #     momentum_equ_y=hv_t+huv_x+hv2_y +GRAVITY*h*(h_y+dz_dy)
        #     #non-negative condition and dry condition
        #     h_positive=h-heaviside_h*h 
        #     hu_dry=hu-heaviside_h*hu
        #     hv_dry=hv-heaviside_h*hv
        #     #identical relation
        #     hu2_equ=10*(hu2*h-hu**2)   #with more larger weight
        #     hv2_equ=10*(hv2*h-hv**2)
        #     huv_equ=10*(huv*h-hu*hv)
        #     return [mass_equ,momentum_equ_x,momentum_equ_y,h_positive,hu_dry,hv_dry,hu2_equ,hv2_equ,huv_equ]  #6 items
            
        def swe_primitive_var(x, U):
            """primitive variables form of the swe is : (x,t)-->(h,u,v). This function is without entropy condition"""
            z,dz_dx,dz_dy=self.z_net(x[:,0:2])     #z and its derivative respect to x,y

            h,u,v=U[:,0:1],U[:,1:2],U[:,2:3]

            h_x=dde.grad.jacobian(U,x,i=0,j=0)
            h_y=dde.grad.jacobian(U,x,i=0,j=1)
            h_t=dde.grad.jacobian(U,x,i=0,j=2)
            u_x=dde.grad.jacobian(U,x,i=1,j=0)
            u_y=dde.grad.jacobian(U,x,i=1,j=1)
            u_t=dde.grad.jacobian(U,x,i=1,j=2)
            v_x=dde.grad.jacobian(U,x,i=2,j=0)
            v_y=dde.grad.jacobian(U,x,i=2,j=1)
            v_t=dde.grad.jacobian(U,x,i=2,j=2)

            #mass conservation equation or mass equation residue
            mass_equ= h_t + h_x*u  + h*u_x + h_y*v+ h*v_y   
            #momentum conservation equation or momentum equation residue
            momentum_equ_x=u_t+u*u_x+v*u_y+GRAVITY*h*(h_x+dz_dx)
            momentum_equ_y=v_t+ v*v_y+u*v_x+GRAVITY*h*(h_y+dz_dy)
            #non-negative condition and dry condition
            with torch.no_grad():
                heaviside_h=torch.heaviside(h-0.001,torch.tensor(0.))
            h_positive=h-heaviside_h*h 
            u_dry=u-heaviside_h*u
            v_dry=v-heaviside_h*v


            return [mass_equ,momentum_equ_x,momentum_equ_y,h_positive,u_dry,v_dry]  #6 items
        

        PDE_FORM={  "VAR": swe_2d_var,
                    # "CONSER": swe_2d_conser,   
                    "VAR_ENTROPY_STABLE": swe_2d_var_entropy_stable,
                    "VAR_ENTROPY_CONSERVATION": swe_2d_var_entropy_stable,
                    "VAR_ENTROPY_DT": swe_2d_var_entropy_DT,
                    # "CONSER_ENTROPY":  swe_2d_conser_entropy  #not used
                    "VAR_RAIN": swe_2d_var_rain,
                    "VAR_ENTROPY_RAIN": swe_2d_var_entropy_rain,
                    "VAR_PRIMITIVE": swe_primitive_var
                 }
        self.pde = PDE_FORM[pde_form]   
