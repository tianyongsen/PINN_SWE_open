import numpy as np
import torch
from abc import ABC, abstractmethod

import cases.static_cases as static_cases

"""There are the 2D dynamic cases."""

#type of boundary conditions, see the baseclasee.py for more details
BCS_TYPE=('dirichlet','neumann','robin','periodic','pointset','operator','ic')

class Dynamic_BaseCase:
    """The base class of dynamic cases, the aims of this class are:
        1: inform the required information of the dynamic case, such as the domain, time domain...
        2: reduce some repeat work
        3: uniform some configurations
    """
    def __init__(self):
        self.datapath = None
        self.geom=[[-10,10],[-10,10]]        #the domain of (x,y),m
        self.time=[0,5]                  #the time domain, seconds
        self.pde_form = "VAR"   # see the SWE2D class for more details

                                      # finally 1-h is the initial condition of h
        self.ics=[
            {'type': 'ic', 
             'function': self.ic_func_of_h, #修改为函数
             'bc': 'initial', 
             'component': 0             #h
            },
            {'type': 'ic', 
             'function': lambda x: 0.,
             'bc': 'initial', 
             'component': 1             #u
            },
            {'type': 'ic', 
             'function': lambda x: 0.,
             'bc': 'initial', 
             'component': 2             #v
            }]
        
        self.bcs = [
            # left boundary
            {"type": "dirichlet",               # boundary condition type
             "function": lambda x:1.0,          # boundary condition function
             "bc": "left",                      # boundary  location    
             "component": 0                     # component of the variabls,(h)
             }, 
             {"type": "dirichlet",             
             "function": lambda x: 0.,       
             "bc": "left",                     
             "component": 1                   # u
             }, 
             {"type": "dirichlet",             
             "function": lambda x: 0.,       
             "bc": "left",                     
             "component": 2                   #v
             }, 

             # right boundary
             {"type": "dirichlet",             
             "function": lambda x: 1.,       
             "bc": "right",                     
             "component": 0                   
             }, 
             {"type": "dirichlet",             
             "function": lambda x: 0.,       
             "bc": "right",                     
             "component": 1                   
             }, 
             {"type": "dirichlet",             
             "function": lambda x: 0.,       
             "bc": "right",                     
             "component": 2                   
             }, 

             # bottom boundary
             {"type": "dirichlet",             
             "function": lambda x: 1.,       
             "bc": "bottom",                     
             "component": 0                   
             }, 
             {"type": "dirichlet",             
             "function": lambda x: 0.,       
             "bc": "bottom",                     
             "component": 1                   
             }, 
             {"type": "dirichlet",             
             "function": lambda x: 0.,
             "bc": "bottom",                     
             "component": 2                   
             }, 

             # top boundary
             {"type": "dirichlet",             
             "function": lambda x: 1.,       
             "bc": "top",                     
             "component": 0                   
             }, 
             {"type": "dirichlet",             
             "function": lambda x: 0.,       
             "bc": "top",                     
             "component": 1                   
             }, 
             {"type": "dirichlet",             
             "function": lambda x: 0.,       
             "bc": "top",                     
             "component": 2                   
             }            
            ]
        # [loss(pde),loss(bcs),loss(ics)] 
        self.loss_weights =np.array([10.,1.,1., 1,1,1]+            #pde loss weights
                                    [1,1,1,  1,1,1,  1,1,1,  1,1,1]+ #bcs loss weights
                                    [100,100,100])                   #ics loss weights     
        
        self.z_net= self.z_func   
        self.mul=4
    
    def __call__(self):
        self.inputs_to_SWE2D = {
            "datapath":self.datapath,
            "ics":self.ics,
            "bcs":self.bcs,
            "z_net":self.z_net, 
            "geom": self.geom,
            "time": self.time,
            "loss_weights": self.loss_weights,
            "pde_form": self.pde_form                           
            }
        return self.inputs_to_SWE2D  

        #initial condition
    
    #initial condition functions
    @abstractmethod
    def ic_func_of_h(self,points): 
        """The initial condition function of the component h
            points: shape=(n,3),the columns are x,y,t and n is the number of points
            return: type:np.array,shape=(n,1)
        """            
        pass

    @abstractmethod
    def z_func(self,points)->tuple[torch.tensor,torch.tensor,torch.tensor]:
            """The topography function of z
                points: shape=(n,2),the columns are x,y
                return: tuple[torch.tensor,torch.tensor,torch.tensor]; torch.tensor: shape=(n,1)
            """
            pass

    def plot_topograph(self):
        """Plot the topography of z"""
        x = np.linspace(self.geom[0][0], self.geom[0][1], 2000)
        y = np.linspace(self.geom[1][0], self.geom[1][1], 2000)
        X, Y = np.meshgrid(x, y)
        points=torch.tensor(np.hstack((X.reshape(-1,1),Y.reshape(-1,1))))
        Z=self.z_net(points)
        z=Z[0].detach().numpy().reshape(X.shape)
        z_x=Z[1].detach().numpy().reshape(X.shape)
        z_y=Z[2].detach().numpy().reshape(X.shape)

        import matplotlib.pyplot as plt
        fig,axs=plt.subplots(2,2,subplot_kw={"projection": "3d"})

        surf = axs[0,0].plot_surface(X, Y,z, cmap='viridis')
        axs[0,0].set_title('Topography of z')
        fig.colorbar(surf,ax=axs[0,0])
        axs[0,0].set_xlabel('x')
        axs[0,0].set_ylabel('y')
        axs[0,0].set_zlabel('z')

        surf = axs[0,1].plot_surface(X, Y,z_x,cmap='viridis')
        axs[0,1].set_title('Topography of z_x')
        fig.colorbar(surf,ax=axs[0,1])
        axs[0,1].set_xlabel('x')
        axs[0,1].set_ylabel('y')
        axs[0,1].set_zlabel('z_x')

        surf = axs[1,0].plot_surface(X, Y,z_y, cmap='viridis')
        axs[1,0].set_title('Topography of z_y')
        fig.colorbar(surf,ax=axs[1,0])
        axs[1,0].set_xlabel('x')
        axs[1,0].set_ylabel('y')
        axs[1,0].set_zlabel('z_y')
        
        # axs[0,0].set_zlim(0,1)        
        # axs[0,1].set_zlim(-1,1)
        # axs[1,0].set_zlim(-1,1)
        plt.show()

    def plot_initial_h(self):
        """Plot the initial condition of h"""
        x = np.linspace(self.geom[0][0], self.geom[0][1], 2000)
        y = np.linspace(self.geom[1][0], self.geom[1][1], 2000)
        X, Y = np.meshgrid(x, y)
        points=torch.tensor(np.hstack((X.reshape(-1,1),Y.reshape(-1,1))))
        h=self.ic_func_of_h(points)
        h=h.detach().numpy().reshape(X.shape)

        import matplotlib.pyplot as plt
        fig,axs=plt.subplots(1,1,subplot_kw={"projection": "3d"})

        surf = axs.plot_surface(X, Y,h, cmap='viridis')
        axs.set_title('Initial condition of h')
        fig.colorbar(surf,ax=axs)
        axs.set_xlabel('x')
        axs.set_ylabel('y')
        axs.set_zlabel('h')
        plt.show()

    def plot_initial_water_level(self):
        """Plot the initial condition of water level"""
        x = np.linspace(self.geom[0][0], self.geom[0][1], 2000)
        y = np.linspace(self.geom[1][0], self.geom[1][1], 2000)
        X, Y = np.meshgrid(x, y)
        points=torch.tensor(np.hstack((X.reshape(-1,1),Y.reshape(-1,1))))
        h=self.ic_func_of_h(points.detach().numpy())
        h=h.reshape(X.shape)
        z=self.z_net(points)[0].detach().numpy().reshape(X.shape)
        water_level=z+h
        import matplotlib.pyplot as plt
        fig,axs=plt.subplots(1,1,subplot_kw={"projection": "3d"})

        surf = axs.plot_surface(X, Y,water_level, cmap='viridis')
        axs.set_title('Initial condition of water level')
        fig.colorbar(surf,ax=axs)
        axs.set_xlabel('x')
        axs.set_ylabel('y')
        axs.set_zlabel('water_level')
        plt.show()


class Dynaminc_Rain_BaseCase(Dynamic_BaseCase):
    """The base class of dynamic rain cases, the aims of this class are:
        1: inform the required information of the dynamic rain case, such as the domain, time domain...
        2: reduce some repeat work
        3: uniform some configurations
    """
    def __init__(self):  
        super().__init__()
        #reset some configurations
        self.pde_form = "VAR_ENTROPY_RAIN"   # see the SWE2D class for more details
        self.time=[0,5]                  #the time domain, Note!!!:min

        self.bcs = [ 
            # left boundary
             {"type": "dirichlet",             
             "function": lambda x: 0.,       
             "bc": "left",                     
             "component": 1                   # u
             }, 
             {"type": "dirichlet",             
             "function": lambda x: 0.,       
             "bc": "left",                     
             "component": 2                   #v 
             }, 

             # right boundary
             {"type": "dirichlet",             
             "function": lambda x: 0.,       
             "bc": "right",                     
             "component": 1                   
             }, 
             {"type": "dirichlet",             
             "function": lambda x: 0.,       
             "bc": "right",                     
             "component": 2                   
             }, 

             # bottom boundary
             {"type": "dirichlet",             
             "function": lambda x: 0.,       
             "bc": "bottom",                     
             "component": 1                   
             }, 
             {"type": "dirichlet",             
             "function": lambda x: 0.,
             "bc": "bottom",                     
             "component": 2                   
             }, 

             # top boundary
             {"type": "dirichlet",             
             "function": lambda x: 0.,       
             "bc": "top",                     
             "component": 1                   
             }, 
             {"type": "dirichlet",             
             "function": lambda x: 0.,       
             "bc": "top",                     
             "component": 2                   
             }            
            ]
                # [loss(pde),loss(bcs),loss(ics)] 
        
        self.loss_weights =np.array([10.,1.,1.  ,1  ,1,1,1]+            #pde loss weights
                                    [1,1,  1,1,  1,1,  1,1]+ #bcs loss weights
                                    [100,100,100])                   #ics loss weights
        self.rain_func=self.rain_intensity
    def __call__(self):
        super().__call__()
        self.inputs_to_SWE2D['rain_func']=self.rain_func
        self.inputs_to_SWE2D['mul']=self.mul
        return self.inputs_to_SWE2D
    #add the rainfall intensity function
    def rain_intensity(self,points)->torch.tensor:
        """The default rainfall intensity (cm/min) during the time domain [0,5],
            and the accumulative rainfall depth is 0.024019249187982226 m.
            The rainfall type is the Chicago_Design_Storm,
            please see the Chicago_Design_Storm class in the utilities.py.
            """
        _temp=torch.abs(points[:,2:]-2.5)/0.5
        ## the accumulative rainfall depth (m) is  0.024019249187982226 m
        rain=73.77884395200002*(_temp*0.13+18.1)/torch.pow(_temp+18.1,1.870) / 10. #cm/min
        # rain=0.8*torch.sin(torch.pi/5*points[:,2:])   #  cm/min 2.546   easy case 
        return rain



class Dynamic_Dam_Break_Var_Case(Dynamic_BaseCase):
    """quasi-2D dam break case in the variable form without entropy"""
    def __init__(self):
        super().__init__()           
        #--reset the domain and time
        # self.geom=[[-10,10],[0,5]]        #the domain of (x,y)
        self.time=[0,1.2]                  #the time domain

        #-reset the y-direction boundary condition
        #--bottom boundary
        self.bcs[6]['type']='Neumann'    #h_y=0
        self.bcs[6]['component_x']=1
        self.bcs[6]['function']=lambda x:0.

        self.bcs[7]['type']='Neumann'    #u_y=0
        self.bcs[7]['component_x']=1
        self.bcs[7]['function']=lambda x:0.
        
        self.bcs[8]['type']='Neumann'    #v_y=0
        self.bcs[8]['component_x']=1
        self.bcs[8]['function']=lambda x:0.

        #--top boundary
        self.bcs[9]['type']='Neumann'
        self.bcs[9]['component_x']=1
        self.bcs[9]['function']=lambda x:0.

        self.bcs[10]['type']='Neumann'
        self.bcs[10]['component_x']=1
        self.bcs[10]['function']=lambda x:0.
        
        self.bcs[11]['type']='Neumann'  #The dirichlet boundary condition is ok,too.
        self.bcs[11]['component_x']=1
        self.bcs[11]['function']=lambda x:0.

        #-reset the x-direction boundary condition
        #--left boundary
        self.bcs[0]['function']=lambda x:2.  #h=2

        self.bcs[1]['type']='Neumann'        #u_x=0
        self.bcs[1]['component_x']=0
        self.bcs[1]['function']=lambda x:0.
        
        self.bcs[2]['type']='Neumann'        #v_x=0
        self.bcs[2]['component_x']=0
        self.bcs[2]['function']=lambda x:0.
        #--right boundary
        self.bcs[3]['function']=lambda x:1.  #h=1  

        self.bcs[4]['type']='Neumann'        #u_x=0
        self.bcs[4]['component_x']=0
        self.bcs[4]['function']=lambda x:0.

        self.bcs[5]['type']='Neumann'        #v_x=0
        self.bcs[5]['component_x']=0
        self.bcs[5]['function']=lambda x:0.

        #choose the pde form and reset the loss weights
        self.pde_form = "VAR"   # see the SWE2D class for more details
        self.loss_weights =np.array([1.,10.,10.   ,10,10,10]+            #pde loss weights
                                    [1,1,1,  1,1,1,  1,1,1,  1,1,1]+ #bcs loss weights
                                    [100,100,100])                   #ics loss weights    
        

    def ic_func_of_h(self,points):
        """The initial condition function of the component h
            points: shape=(n,3),the columns are x,y,t and n is the number of points
            return: type:np.array,shape=(n,1)
        """
        #if x<0,then h=2,and if x>0, then h=1.
        x=points[:,0:1]
        h=np.zeros_like(x)+1.
        h[x<0]=2.
        return h
    def z_func(self, points)->tuple[torch.tensor,torch.tensor,torch.tensor]:
        """The topography function of z
            points: shape=(n,2),the columns are x,y
            return: tuple[torch.tensor,torch.tensor,torch.tensor]; torch.tensor: shape=(n,1)  
        """
        #horizontal topography
        z=torch.zeros_like(points[:,0:1])
        z_x=z
        z_y=z
        return z,z_x,z_y


class Dynamic_Dam_Break_Var_Entropy_Case(Dynamic_Dam_Break_Var_Case):
    """quasi-2D dam break case with entropy"""
    def __init__(self):
        super().__init__()
        #reset the pde form and loss weights
        self.pde_form = "VAR_ENTROPY"   # see the SWE2D class for more details
        self.loss_weights =np.array([1.,10.,10.  ,1,   10,10,10]+            #pde loss weights
                                    [1,1,1,  1,1,1,  1,1,1,  1,1,1]+ #bcs loss weights
                                    [100,100,100])                   #ics loss weights    


class Dynamic_Dam_Break_Var_Primitive_Case(Dynamic_Dam_Break_Var_Case):
    """quasi-2D dam break case with primitive variables"""
    def __init__(self):
        super().__init__()
        #reset the pde form and loss weights
        self.pde_form = "VAR_PRIMITIVE"   # see the SWE2D class for more details
        self.loss_weights =np.array([1.,10.,10.,10,10,10]+            #pde loss weights
                                    [1,1,1,  1,1,1,  1,1,1,  1,1,1]+ #bcs loss weights
                                    [100,100,100])                   #ics loss weights    



class Dynamic_Tidal_Var_Case(Dynamic_BaseCase):
    """ 2D tidal simulation case"""
    def __init__(self):
        super().__init__()
        self.datapath = None
        self.geom=[[-2,2],[-2,2]]        #the domain of (x,y)
        self.time=[0,0.5]
        self.pde_form="VAR"

        #set the periodic boundary condition, the "function" is not used in this case.
        if True:  #jsut for block the code
            #--left boundary and x_direction
            self.bcs[0]['type']='periodic' 
            self.bcs[0]['component_x']=0
            
            self.bcs[1]['type']='periodic'
            self.bcs[1]['component_x']=0
            
            self.bcs[2]['type']='periodic'
            self.bcs[2]['component_x']=0

            #--right boundary and x_direction
            self.bcs[3]['type']='periodic'
            self.bcs[3]['component_x']=0

            self.bcs[4]['type']='periodic'
            self.bcs[4]['component_x']=0

            self.bcs[5]['type']='periodic'
            self.bcs[5]['component_x']=0

            #--bottom boundary and y_direction
            self.bcs[6]['type']='periodic'
            self.bcs[6]['component_x']=1

            self.bcs[7]['type']='periodic'
            self.bcs[7]['component_x']=1

            self.bcs[8]['type']='periodic'
            self.bcs[8]['component_x']=1

            self.bcs[9]['type']='periodic'
            self.bcs[9]['component_x']=1

            #--top boundary and y_direction
            self.bcs[10]['type']='periodic'
            self.bcs[10]['component_x']=1

            self.bcs[11]['type']='periodic'
            self.bcs[11]['component_x']=1

    
    def ic_func_of_h(self,points):
        x=points[:,0:1]
        y=points[:,1:2]
        z=1.0+0.01*np.cos(np.pi/2.*x)*np.cos(np.pi/2.*y)
        h=z
        return h       #m
    
    def z_func(self, points)->tuple[torch.tensor,torch.tensor,torch.tensor]:
        #horizontal topography
        x=points[:,0:1]
        y=points[:,1:2]
        z=1.0+0.01*torch.cos(torch.pi/2.*x)*torch.cos(torch.pi/2.*y)
        z_x=-0.01*torch.pi/2.*torch.sin(np.pi/2.*x)*torch.cos(np.pi/2.*y)
        z_y=-0.01*torch.pi/2.*torch.cos(np.pi/2.*x)*torch.sin(np.pi/2.*y)
        return z,z_x,z_y #m

class Dynamic_Tidal_Var_Rain_Case(Dynamic_Tidal_Var_Case):
    """ 2D tidal with rain simulation case"""
    def __init__(self):
        super().__init__()
        # self.geom=[[-2,2],[-2,2]]        #the domain of (x,y)
        self.time=[0,.5]                    #min
        self.pde_form="VAR_RAIN"
        self.loss_weights =np.array([10.,1.,1.    ,1,1,1]+            #pde loss weights
                                    [1,1,1,  1,1,1, 1,1,1  ,1,1,1]+ #bcs loss weights
                                    [100,100,100])                   #ics loss weights
        self.rain_func=self.rain_intensity
        self.mul=4     #double
    def __call__(self):
        super().__call__()
        self.inputs_to_SWE2D['rain_func']=self.rain_func
        self.inputs_to_SWE2D['mul']=self.mul
        return self.inputs_to_SWE2D

    def ic_func_of_h(self,points):
        x=points[:,0:1]
        y=points[:,1:2]
        h=1.0+0.01*np.cos(np.pi/2.*x)*np.cos(np.pi/2.*y)
        return h       #m

    def z_func(self, points)->tuple[torch.tensor,torch.tensor,torch.tensor]:
        return super().z_func(points)   #m
    
    def rain_intensity(self,points)->torch.tensor:
        _temp=torch.abs(points[:,2:]*10-2.5)/0.5
        ## the accumulative rainfall depth (m) is  0.024019249187982226 m
        rain=73.77884395200002*(_temp*0.13+18.1)/torch.pow(_temp+18.1,1.870) /10. #m/s
        return rain*10  #m/s


class Dynamic_Circular_Dam_Break_Var_Case(Dynamic_BaseCase):
    """2D circular dam break case"""
    def __init__(self):
        super().__init__()
        #--reset the domain and time
        self.geom=[[-10,10],[-10,10]]        #the domain of (x,y)
        self.time=[0,0.5]                  #the time domain
        
        #-reset to the periodic boundary condition and the initial conditions unchange
        self.bcs = [
            # left boundary
            {"type": "periodic",               # boundary condition type
             "bc": "left",                      # boundary  location    
             'component_x':0,                   #x,y,t
             "component": 0                    
             }, 
             {"type": "periodic",               # boundary condition type}
             "bc": "left",                      # boundary  location    
             'component_x':0,
             "component": 1                     # component of the variabls,(u)
             },
             {"type": "periodic",             
             "bc": "left",   
             'component_x':0,                  
             "component": 2                   #v
             }, 

             # right boundary
             {"type": "periodic",             
             "bc": "right",    
             'component_x':0,                  
             "component": 0                   
             }, 
             {"type": "periodic",             
             "bc": "right",                     
             "component": 1     ,
             "component_x":0              
             }, 
             {"type": "periodic",             
             "bc": "right",                     
             "component": 2 ,
             "component_x":0                 
             }, 

             # bottom boundary
             {"type": "periodic",             
             "bc": "bottom",                     
             "component": 0  ,
             "component_x":1                 
             }, 
             {"type": "periodic",             
             "bc": "bottom",                     
             "component": 1,
             "component_x":1                   
             }, 
             {"type": "periodic",             
             "bc": "bottom",                     
             "component": 2 ,       
             "component_x":1           
             }, 

             # top boundary
             {"type": "periodic",             
             "bc": "top",                     
             "component": 0     ,
             "component_x":1                 
             }, 
             {"type": "periodic",             
             "bc": "top",                     
             "component": 1     ,
             "component_x":1                 
             }, 
             {"type": "periodic",             
             "bc": "top",                     
             "component": 2  ,
             "component_x":1                 
             }            
            ]
        
    def ic_func_of_h(self,points):
            """The initial condition function of the component h
                points: shape=(n,3),the columns are x,y,t and n is the number of points
                return: type:np.array,shape=(n,1)
            """
            #if r<3,then h=2.,and if 3<r, then h=1.m
            x=points[:,0:1]
            y=points[:,1:2]
            h=1.+np.zeros_like(x)   
            h[x**2+y**2<3**2]=2.            #radius=3.
            return h
    def z_func(self, points)->tuple[torch.tensor,torch.tensor,torch.tensor]:
        """The topography function of z
            points: shape=(n,2),the columns are x,y  
            return: tuple[torch.tensor,torch.tensor,torch.tensor]; torch.tensor: shape=(n,1)  
        """
        #horizontal topography
        z=torch.zeros_like(points[:,0:1])
        z_x=z
        z_y=z
        return z,z_x,z_y


class Dyanmic_Circular_Dam_Break_Var_Entropy_Case(Dynamic_Circular_Dam_Break_Var_Case):
    """2D circular dam break case with entropy"""
    def __init__(self):
        super().__init__()
        #reset the pde form and loss weights
        self.pde_form = "VAR_ENTROPY"   # see the SWE2D class for more details
        self.loss_weights =np.array([10.,1.,1.  ,0.01,   10,10,10]+            #pde loss weights
                                    [1,1,1,  1,1,1]+ #bcs loss weights  
                                    [100,100,100])                   #ics loss weights    



class Dynamic_Circular_Dam_Break_Var_Toro_Case(Dynamic_BaseCase):
    """2D circular dam break case"""
    def __init__(self):
        super().__init__()
        #--reset the domain and time
        self.geom=[[-20,20],[-20,20]]        #the domain of (x,y)
        self.time=[0,5]                  #the time domain
        
        #-reset to the periodic boundary condition and the initial conditions unchange
        self.bcs = [
            # left boundary
            {"type": "periodic",               # boundary condition type
             "bc": "left",                      # boundary  location    
             'component_x':0,                   #x,y,t
             "component": 0                    
             }, 
             {"type": "periodic",               # boundary condition type}
             "bc": "left",                      # boundary  location    
             'component_x':0,
             "component": 1                     # component of the variabls,(u)
             },
             {"type": "periodic",             
             "bc": "left",   
             'component_x':0,                  
             "component": 2                   #v
             }, 

             # right boundary
             {"type": "periodic",             
             "bc": "right",    
             'component_x':0,                  
             "component": 0                   
             }, 
             {"type": "periodic",             
             "bc": "right",                     
             "component": 1     ,
             "component_x":0              
             }, 
             {"type": "periodic",             
             "bc": "right",                     
             "component": 2 ,
             "component_x":0                 
             }, 

             # bottom boundary
             {"type": "periodic",             
             "bc": "bottom",                     
             "component": 0  ,
             "component_x":1                 
             }, 
             {"type": "periodic",             
             "bc": "bottom",                     
             "component": 1,
             "component_x":1                   
             }, 
             {"type": "periodic",             
             "bc": "bottom",                     
             "component": 2 ,       
             "component_x":1           
             }, 

             # top boundary
             {"type": "periodic",             
             "bc": "top",                     
             "component": 0     ,
             "component_x":1                 
             }, 
             {"type": "periodic",             
             "bc": "top",                     
             "component": 1     ,
             "component_x":1                 
             }, 
             {"type": "periodic",             
             "bc": "top",                     
             "component": 2  ,
             "component_x":1                 
             }            
            ]
        
    def ic_func_of_h(self,points):
            """The initial condition function of the component h
                points: shape=(n,3),the columns are x,y,t and n is the number of points
                return: type:np.array,shape=(n,1)
            """
            #if r<3,then h=2.,and if 3<r, then h=1.m
            x=points[:,0:1]
            y=points[:,1:2]
            h=.5+np.zeros_like(x)   
            h[x**2+y**2<2.5**2]=2.5            #radius=2.5
            return h
    def z_func(self, points)->tuple[torch.tensor,torch.tensor,torch.tensor]:
        """The topography function of z
            points: shape=(n,2),the columns are x,y  
            return: tuple[torch.tensor,torch.tensor,torch.tensor]; torch.tensor: shape=(n,1)  
        """
        #horizontal topography
        z=torch.zeros_like(points[:,0:1])
        z_x=z
        z_y=z
        return z,z_x,z_y


class Dynamic_Cosine_Bump_Rain_Case(static_cases.Static_Cosine_Bump_Rain_Case):
    #abandon this case
    def __init__(self):
        super().__init__()
        #reset some parameters
        self.cosine_bump.water_level=0    
    def __call__(self):
        return super().__call__()
    
class Dynamic_Cosine_Depression_Rain_Case(static_cases.Static_Cosine_Depression_Rain_Case):
    #abandon this case
    def __init__(self):    
        super().__init__()
        #reset some parameters
        self.cosine_depression.water_level=0.
    def __call__(self):
        return super().__call__()


    

    


