import numpy as np
import torch
from abc import ABC, abstractmethod


class Static_BaseCase(ABC):
    def __init__(self):
        self.datapath = None
        self.geom=[[-10,10],[-10,10]]        #the domain of (x,y)
        self.time=[0,5]                  #the time domain
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
            # {"type": "dirichlet",               # boundary condition type
            #  "function": lambda x:1.0,          # boundary condition function
            #  "bc": "left",                      # boundary  location    
            #  "component": 0                     # component of the variabls,(h)
            #  }, 
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
            #  {"type": "dirichlet",             
            #  "function": lambda x: 1.,       
            #  "bc": "right",                     
            #  "component": 0                   
            #  }, 
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
            #  {"type": "dirichlet",             
            #  "function": lambda x: 1.,       
            #  "bc": "bottom",                     
            #  "component": 0                   
            #  }, 
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
            #  {"type": "dirichlet",             
            #  "function": lambda x: 1.,       
            #  "bc": "top",                     
            #  "component": 0                   
            #  }, 
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
        self.loss_weights =np.array([10.,1.,1., 1,  10,10,10]+            #pde loss weights
                                    [1,1,  1,1,  1,1,  1,1]+ #bcs loss weights
                                    [100,100,100])                   #ics loss weights     
        
        self.z_net= self.z_func
    
    def __call__(self):
        self.inputs_to_SWE2D = {
            "datapath":None,
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
                return: type:torch.tensor ,shape=(n,1)  
            """
            pass
    
    def plot_topograph(self):
        """Plot the topography of z  to check"""
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
       
        plt.show()


    
class Static_Flat_Case(Static_BaseCase):
    def __init__(self):
        super().__init__()
    
    def ic_func_of_h(self,points):     
        """water depth is 1.0 at all points"""
        return np.ones_like(points[:,0:1]) 

    def z_func(self,points):
        """The topography is 0 at all points"""
        z=torch.zeros_like(points[:,0:1])
        z_x=z
        z_y=z
        return z,z_x,z_y
    
#-----------------------------------------
#3 static cases
class Static_Cosine_Bump_Case(Static_BaseCase):
    def __init__(self):
        super().__init__()
        self.radius=5.              # m
        self.height=0.2             # m
        self.water_level=0.3        # m
    
    def __call__(self):
        super().__call__()
        return self.inputs_to_SWE2D
    
    def ic_func_of_h(self,points):   
        """The initial condition function of the component h
        points: shape=(n,3),the columns are x,y,t and n is the number of points
        return: (n,1)
        """            
        x,y=points[:,0:1],points[:,1:2]
        x_square_add_y_square=x**2+y**2    
        index=np.where(x_square_add_y_square<self.radius**2)            # points in the unit circle
        h=np.zeros_like(x)                                              # out of the unit circle
        h[index]=self.height/2.*(1.+np.cos(np.pi/self.radius**2*x_square_add_y_square[index]))   # inside the unit circle
        return (self.water_level-h)*100.      #cm   finally 1-h is the initial condition of h  
    
    def z_func(self,points)->tuple[torch.tensor,torch.tensor,torch.tensor]:
        """The topography function of z :unit :m
            points: shape=(n,2),the columns are x,y
            return: shape:=(n,1),(n,1),(n,1)
        """
        x,y=points[:,0:1],points[:,1:2]
        x_square_add_y_square=x**2+y**2    
        index=torch.where(x_square_add_y_square<self.radius**2)     # points in the unit circle
        z=torch.zeros_like(x)                                       # out of the unit circle is 0.
        phi=torch.pi/self.radius**2*x_square_add_y_square[index]
        z[index]=self.height/2.*(1.+torch.cos(phi))   # inside the unit circle
        z_x=torch.zeros_like(x)
        z_y=torch.zeros_like(x)
        sin_term=-self.height*torch.pi/self.radius**2*torch.sin(phi)
        z_x[index]=x[index]*sin_term
        z_y[index]=y[index]*sin_term
        return z,z_x,z_y      #m

class Static_Cosine_Depression_Case(Static_BaseCase):
    def __init__(self):
        super().__init__()
        self.radius=5.              # m
        self.height=0.2             # m
        self.water_level=0.3         # m
    def __call__(self):
        super().__call__()
        return self.inputs_to_SWE2D
    
    def ic_func_of_h(self,points):   
        """The initial condition function of the component h
        points: shape=(n,3),the columns are x,y,t and n is the number of points
        return: (n,1)
        """            
        x,y=points[:,0:1],points[:,1:2]
        x_square_add_y_square=x**2+y**2    
        index=np.where(x_square_add_y_square<self.radius**2)            # points in the unit circle
        h=np.zeros_like(x)                                              # out of the unit circle
        h[index]=self.height/2.*(1.+np.cos(np.pi/self.radius**2*x_square_add_y_square[index]))   # inside the unit circle
        return self.water_level+h   #m   
    def z_func(self,points)->tuple[torch.tensor,torch.tensor,torch.tensor]:
        """The topography function of z
            points: shape=(n,2),the columns are x,y
            return: shape:=(n,1),(n,1),(n,1)
        """
        x,y=points[:,0:1],points[:,1:2]
        x_square_add_y_square=x**2+y**2    
        index=torch.where(x_square_add_y_square<self.radius**2)     # points in the unit circle
        z=torch.zeros_like(x)                                       # out of the unit circle is 0.
        phi=torch.pi/self.radius**2*x_square_add_y_square[index]
        z[index]=-self.height/2.*(1.+torch.cos(phi))   # inside the unit circle
        z_x=torch.zeros_like(x)
        z_y=torch.zeros_like(x)
        sin_term=self.height*torch.pi/self.radius**2*torch.sin(phi)
        z_x[index]=x[index]*sin_term
        z_y[index]=y[index]*sin_term
        return z,z_x,z_y  #m


class Static_Step_Case(Static_BaseCase):
    def __init__(self):
        super().__init__()
        self.a=0.2
        self.height=0.2             # m
        self.water_level=0.3          # m
        self.radius=5.              # m
    def __call__(self):
        super().__call__()
        return self.inputs_to_SWE2D
    
    def ic_func_of_h(self,points):   
        """The initial condition function of the component h
        points: shape=(n,3),the columns are x,y,t and n is the number of points
        return: (n,1)
        """
        x,y=points[:,0:1],points[:,1:2]
        x_square_add_y_square=x**2+y**2    
        less_than_1=x_square_add_y_square<=self.radius**2
        less_than_a=x_square_add_y_square<(self.radius+self.a)**2

        z=np.zeros_like(x)                                 
        z[less_than_1]=self.height                                     

        ring_index=np.logical_and(less_than_a,~less_than_1) # points in the ring
        r_ring_index=np.sqrt(x[ring_index]**2+y[ring_index]**2)
        z[ring_index]=self.height/self.a*(self.radius+self.a-r_ring_index)
        return self.water_level-z #cm                                       
    
    def z_func(self,points)->tuple[torch.tensor,torch.tensor,torch.tensor]:
        """The topography function of z
            points: shape=(n,2),the columns are x,y
            return: shape:=(n,1),(n,1),(n,1)
        """
        x,y=points[:,0:1],points[:,1:2]
        x_square_add_y_square=x**2+y**2    
        less_than_r=x_square_add_y_square<=self.radius**2
        
        less_than_a=x_square_add_y_square<(self.radius+self.a)**2

        z=torch.zeros_like(x)                                     
        z[less_than_r]=self.height                                   

        ring_index=torch.logical_and(less_than_a,~less_than_r) # points in the ring
        r_ring_index=torch.sqrt(x[ring_index]**2+y[ring_index]**2)
        z[ring_index]=self.height/self.a*(self.radius+self.a-r_ring_index)   

        z_x=torch.zeros_like(x)
        z_x[ring_index]=-self.height/self.a*x[ring_index]/r_ring_index

        z_y=torch.zeros_like(x)              
        z_y[ring_index]=-self.height/self.a*y[ring_index]/r_ring_index
        
        return z,z_x,z_y   #m

    
    


class Static_Tidal_Case(Static_BaseCase):
    def __init__(self):
        super().__init__()
        self.height=0.2             # m
        self.water_level=0.3          # m

    def __call__(self):
        return super().__call__()
    
    def ic_func_of_h(self,points):   
        """The initial condition function of the component h
        points: shape=(n,3),the columns are x,y,t and n is the number of points
        return: (n,1)
        """     
        x,y=points[:,0:1],points[:,1:2]
        z_cm=self.height*np.cos(np.pi/self.geom[0][1]*x)*np.cos(np.pi/self.geom[1][1]*y)
        return self.water_level-z_cm
    
    def z_func(self,points)->tuple[torch.tensor,torch.tensor,torch.tensor]:
        """The topography function of z
            points: shape=(n,2),the columns are x,y
            return: shape:=(n,1),(n,1),(n,1)
        """
        x=points[:,0:1]
        y=points[:,1:2]
        z=self.height*torch.cos(torch.pi/self.geom[0][1]*x)*torch.cos(torch.pi/self.geom[1][1]*y)
        z_x=-self.height*torch.pi/self.geom[0][1]*torch.sin(torch.pi/self.geom[0][1]*x)*torch.cos(torch.pi/self.geom[1][1]*y)
        z_y=-self.height*torch.pi/self.geom[1][1]*torch.cos(torch.pi/self.geom[0][1]*x)*torch.sin(torch.pi/self.geom[1][1]*y)
        return z,z_x,z_y   #m
    



class Static_Discontinuous_Step_Case(Static_BaseCase):
    def __init__(self):
        super().__init__()
        self.a=0.2
    
    def ic_func_of_h(self,points):   
        """The initial condition function of the component h
        points: shape=(n,3),the columns are x,y,t and n is the number of points
        return: (n,1)
        """
        x,y=points[:,0:1],points[:,1:2]
        x_square_add_y_square=x**2+y**2    
        less_than_1=x_square_add_y_square<=1.
        less_than_a=x_square_add_y_square<(1+self.a)**2

        z=np.zeros_like(x)                                  # out of the a circle is 0.
        z[less_than_1]=0.2                                     # inside the unit circle is 0.2

        ring_index=np.logical_and(less_than_a,~less_than_1) # points in the ring
        r_ring_index=np.sqrt(x[ring_index]**2+y[ring_index]**2)
        z[ring_index]=0.2/self.a*(1.+self.a-r_ring_index)
        return 1.-z                 #m                      
    
    def z_func(self,points)->tuple[torch.tensor,torch.tensor,torch.tensor]:
        """The topography function of z
            points: shape=(n,2),the columns are x,y
            return: shape:=(n,1),(n,1),(n,1)
        """
        x,y=points[:,0:1],points[:,1:2]
        x_square_add_y_square=x**2+y**2    
        less_than_1=x_square_add_y_square<=1.

        # a=0.1 
        
        less_than_a=x_square_add_y_square<(1+self.a)**2

        z=torch.zeros_like(x)                                  # out of the a circle is 0.
        z[less_than_1]=0.2                                     # inside the unit circle is 0.2

        ring_index=torch.logical_and(less_than_a,~less_than_1) # points in the ring
        r_ring_index=torch.sqrt(x[ring_index]**2+y[ring_index]**2)
        z[ring_index]=0.2/self.a*(1.+self.a-r_ring_index)   

        z_x=torch.zeros_like(x)
        z_x[ring_index]=-0.2/self.a*x[ring_index]/r_ring_index

        z_y=torch.zeros_like(x)              
        z_y[ring_index]=-0.2/self.a*y[ring_index]/r_ring_index
        
        return z,z_x,z_y 



class Static_Rain_BaseCase(Static_BaseCase):
    """The base class for static rain cases."""
    def __init__(self):
        super().__init__()
        #reset some properties
        self.geom=[[-10,10],[-10,10]]   #reset the domain (m)
        self.datapath = None
        self.time=[0,5]                  #the time domain (min)
        self.pde_form = "VAR_ENTROPY_RAIN"       #see the SWE2D class for more detailssssss
        self.mul=4                       #the multiply of the training points

        self.ics=[
            {'type': 'ic', 
             'function': self.ic_func_of_h, #initial depth
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
        self.loss_weights=np.array([10.,1.,1.,  1,  10,10,10]+            #pde loss weights
                                    [1,1,  1,1,  1,1,  1,1]+            #bcs loss weights
                                    [100,100,100])                #ics loss weights    
        
        self.z_net= self.z_func  
        self.rain_func=self.rain_intensity

    def __call__(self):
        super().__call__()
        self.inputs_to_SWE2D['rain_func']=self.rain_func
        self.inputs_to_SWE2D['mul']=self.mul
        return self.inputs_to_SWE2D
    @abstractmethod
    def ic_func_of_h(self,points):     
        """initial water depth"""
        return np.zeros_like(points[:,0:1]) 

    @abstractmethod
    def z_func(self,points):
        """The topography"""
        z=torch.zeros_like(points[:,0:1])
        z_x=z
        z_y=z
        return z,z_x,z_y
    
    def rain_intensity(self,points)->torch.tensor:
        """The default rainfall intensity (cm/min) during the time domain [0,5],
            and the accumulative rainfall depth is 0.024019249187982226 m.
            The rainfall type is the Chicago_Design_Storm,
            please see the Chicago_Design_Storm class in the utilities.py.
            """
        #t=points[:,2:]
        # P=100    #return period,year
        # a=4234.323*(1+0.952* np.log(P))
        # a=a*6*1e-3   #unit: L/(s*hm^2)-->mm/min
        # b=18.1
        # n=0.870
        # T=5     #min  300s
        # r=0.5   #coefficient of the rainfall intensity
        # tp=r*T   
        # if t<tp:
        #     t_in=(tp-t)/r
        # else:
        #     t_in=(t-tp)/(1-r)
        # i=a*( t_in* (1-n)+self.b) / np.power(t_in +b,n+1) /60000.
        _temp=torch.abs(points[:,2:]-2.5)/0.5
        ## the accumulative rainfall depth (m) is  0.024019249187982226 m
        rain=73.77884395200002*(_temp*0.13+18.1)/torch.pow(_temp+18.1,1.870) / 10. #cm/min
        # rain=0.8*torch.sin(torch.pi/5*points[:,2:])   #  cm/min 2.546   easy case 
        return rain



class Static_Flat_Rain_Case(Static_Rain_BaseCase):
    def __init__(self):
        super().__init__()
        #reset some properties

    def __call__(self):
        super().__call__()
        # self.inputs_to_SWE2D['rain_func']=self.rain_func   #repeat
        # self.inputs_to_SWE2D['mul']=self.mul              #repeat
        # self.inputs_to_SWE2D['z_net']=self.z_net   #update
        # self.inputs_to_SWE2D['ics'][0]['function']=self.ic_func_of_h #update
        return self.inputs_to_SWE2D
    
    def ic_func_of_h(self,points):     
        """water depth is 1.0 at all points"""
        return np.zeros_like(points[:,0:1]) 

    def z_func(self,points):
        """The topography is 0 at all points"""
        z=torch.zeros_like(points[:,0:1])
        z_x=z
        z_y=z
        return z,z_x,z_y


class Static_Cosine_Bump_Rain_Case(Static_Rain_BaseCase):
    def __init__(self):
        super().__init__()
        self.cosine_bump=Static_Cosine_Bump_Case()

    def __call__(self):
        super().__call__()
        return self.inputs_to_SWE2D
    
    def ic_func_of_h(self,points):   
        return self.cosine_bump.ic_func_of_h(points) # cm

    def z_func(self,points)->tuple[torch.tensor,torch.tensor,torch.tensor]:
        return self.cosine_bump.z_func(points)

class Static_Cosine_Depression_Rain_Case(Static_Rain_BaseCase):
    def __init__(self):
        super().__init__()
        self.cosine_depression=Static_Cosine_Depression_Case()

    def __call__(self):
        super().__call__()
        return self.inputs_to_SWE2D
    
    def ic_func_of_h(self,points):   
        return self.cosine_depression.ic_func_of_h(points)   # cm
     
    def z_func(self,points)->tuple[torch.tensor,torch.tensor,torch.tensor]:
        return self.cosine_depression.z_func(points)


class Static_Step_Rain_Case(Static_Rain_BaseCase):
    def __init__(self):
        super().__init__()
        self.step=Static_Step_Case()
    def __call__(self):
        super().__call__()
        return self.inputs_to_SWE2D
    
    def ic_func_of_h(self,points):   
        return self.step.ic_func_of_h(points)  #cm
    
    def z_func(self,points)->tuple[torch.tensor,torch.tensor,torch.tensor]:
        return self.step.z_func(points)


class Static_Tidal_Rain_Case(Static_Rain_BaseCase):
    def __init__(self):
        super().__init__()
        self.tidal=Static_Tidal_Case()

    def __call__(self):
        return super().__call__()
    
    def ic_func_of_h(self,points):   
        return self.tidal.ic_func_of_h(points)  #  cm
    
    def z_func(self,points)->tuple[torch.tensor,torch.tensor,torch.tensor]:
        return self.tidal.z_func(points)


    