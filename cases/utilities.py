from matplotlib import pyplot as plt
import numpy as np
class Chicago_Design_Storm:
    """
    Given the Intensity-Duration-Frequency(IDF):i=a/(T+b)^n ang the Peaking-Time-Ratio r,
    We can get the Chicago_Design_Storm using the following formula: 
        i_t= a*( (t_p - t)/r * (1-n)+b) / (t_p-t)/r +b)^{n+1}  , if t<t_p 
        i_t= a*( (t - t_p)/(1-r) * (1-n) + b) / (t-t_p)/(1-r) +b)^{n+1}  , if t>t_p 
        where t_p=r*T is the peak time.
    Args:
        see the above formula.
        there is a example of Hefei storm using the following IDF: 
            Take the IDF as i=4234.323*(1+0.952* lg P)/ (t+18.1)^0.870   5=<t <=180 min 
            where i is the intensity with unit L/(s*hm^2) which is the volume of rainfall per second per hectare of area; 
            P: the return period,year; 
            t:the duration time (min).
            then a,b and n can be get.
            The T and r is user-defined, for example, T=30min, r=0.5 
    return:
        the intensity of the Chicago_Design_Storm at time t.
    Refeerences:
        https://www.bilibili.com/read/cv26311056/ 
        https://docs.bentley.com/LiveContent/web/Bentley%20StormCAD%20SS5-v1/en/GUID-A0096667-D870-426C-A77A-233BE0A41A32.html
        https://www.163.com/dy/article/HMDCPDEG0530QRMB.html
        or more academic references.
        https://doi.org/10.3390/ijerph20054245
        Chen, J.; Li, Y.; Zhang, C. The Effect of Design Rainfall Patterns on Urban Flooding Based on the Chicago Method. Int. J. Environ. Res. Public Health 2023, 20, 4245.
    """
    def __init__(self,a,b,n,T,r):
        self.a=a
        self.b=b
        self.n=n
        self.T=T       # unit: min
        self.r=r       
        self.tp=r*T
    def __call__(self,t,y=0):
        """ return the intensity of the Chicago_Design_Storm at time t."""
        t=t/60.   #s->min
        if t<self.tp:
            return self.__general_formula__((self.tp-t)/self.r)/60000.    #unit mm/min-->m/s
        else:
            return self.__general_formula__((t-self.tp)/(1-self.r))/60000. #unit mm/min-->m/s
        
    def __general_formula__(self,t):
        return self.a*( t* (1-self.n)+self.b) / np.power(t +self.b,self.n+1) 
    
    def Accumulated_rainfall(self):
        from scipy.integrate import quad
        result,error=quad(self.__call__,0,self.T*60)  
        return result   #unit: m


def construct_rain_and_plot():  
    P=100    #return period,year
    a=4234.323*(1+0.952* np.log10(P))
    a=a*6*1e-3   #unit: L/(s*hm^2)-->mm/min
    print(a)
    b=18.1
    n=0.870
    T=5      #min
    r=0.5    #r=0.5
    Ch_05=Chicago_Design_Storm(a,b,n,T,r)

    print(f"the accumulated rainfall of the Chicago_Design_Storm with r=0.5 is: \n {Ch_05.Accumulated_rainfall()}m")

    t=np.arange(0,T*60.,0.1)
    intensity_05=np.array([Ch_05(ti) for ti in t])        #r=0.5

    #绘制
    plt.plot(t,intensity_05,label='r=0.5',linewidth=2.,color=np.array([0,153,153])/255.)
    plt.xlabel('Time (s)')
    plt.ylabel('Intensity (m/s)')
    plt.show()

# construct_rain_and_plot()
import torch
time=np.arange(0,5,0.1)
def intensity(time):   #min
    _temp=np.abs(time-2.5)/0.5
    ## the accumulative rainfall depth (m) is  0.024019249187982226 m
    i=73.77884395200002*(_temp*0.13+18.1)/np.pow(_temp+18.1,1.870) / 10.
    return i

# i=66.49799611750765*(time[:,0]*0.13+18.1)/torch.pow(time[:,0]+18.1,1.870) 
from scipy.integrate import quad
accumulate_rainfall=[quad(intensity,0,t)[0] for t in time]  


# fig,ax=plt.subplots()
# ax.plot(time,intensity(time),label='r=0.5',linewidth=2.,color=np.array([0,153,153])/255.)
# ax_twin=ax.twinx()
# ax_twin.plot(time,accumulate_rainfall,label='Accumulated rainfall',linewidth=2.,color=np.array([255,102,0])/255.)
# ax.set_xlabel('Time (min)')
# ax.set_ylabel('Intensity (cm/mm)')
# ax_twin.set_ylabel('Accumulated rainfall (cm)')
# ax.legend()
# ax_twin.legend()
# # plt.xlabel()
# plt.show()

# result,error=quad(self.__call__,0,self.T*60)  