import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
bbox=[-10,10,-10,10,0,5]   # x1min, x1max, x2min, x2max, tmin, tmax
geom=[[-10,10],[-10,10]]
height=0.2
radius=5
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def load_model(pt_file='80000.pt',device='cpu',layers=6,method="laaf"):
    import os
    os.environ["DDEBACKEND"] = "pytorch"   #in the front of the deepxde. 
    import deepxde as dde
    from src.model.laaf import DNN_LAAF

    if method == "laaf":     
        net = DNN_LAAF(layers-1, 300, 3,3)     # must be identical to the saved model. 
    else:
        net=dde.nn.FNN([3] + [300]*layers + [3], "tanh", "Glorot normal")
    net = net.float()    
    state_dict = torch.load(pt_file, map_location=device,weights_only=True)
    # model = swe_case.create_model(net)               # create the PINN model
    net.load_state_dict(state_dict.get('model_state_dict'))
    net.eval()
    return net.to(device)



def test_mesh_points_and_results(net,timesplit=6):
    # Test mesh points
    xx,yy,tt=np.meshgrid(np.linspace(bbox[0],bbox[1],100),np.linspace(bbox[2],bbox[3],100),np.linspace(bbox[4],bbox[5],timesplit))
    points=np.vstack((xx.flatten(),yy.flatten(),tt.flatten())).T
    u_pred=net(torch.tensor(points).float().to(device))
    u_pred=u_pred.detach().cpu().numpy().reshape((*xx.shape,-1))
    return xx,yy,tt,points,u_pred  #numpy

def rain_intensity(t):
    _temp=np.abs(t-2.5)/0.5
    ## the accumulative rainfall depth (m) is  0.024019249187982226 m
    value=73.77884395200002*(_temp*0.13+18.1)/np.pow(_temp+18.1,1.870) / 10. #cm/min
    return value

def cum_rain_func(times):
    from scipy.integrate import quad
    cum_rain=[quad(rain_intensity,0,_t)[0] for _t in times]
    return cum_rain

def plot_4_7_cases_rain(save_path,net4,net5,net6,net7):
    """for the cases of 4,5,6,7: flat_rain,cosine_bump_rian,cosine_depression_rain,tidal_rain"""
    water_level=30  #cm
    timesplit=60
    x1,x2,tt,points,u_pred_4=test_mesh_points_and_results(net4,timesplit)
    x1,x2,tt,points,u_pred_5=test_mesh_points_and_results(net5,timesplit)
    x1,x2,tt,points,u_pred_6=test_mesh_points_and_results(net6,timesplit)
    x1,x2,tt,points,u_pred_7=test_mesh_points_and_results(net7,timesplit)
    #e.g.  x1: shape=(100,100,6)  points: shape=(60000,3)  u_pred: shape=(100,100,6,3)  

    z4=100*z_func_flat(points).reshape(x1.shape)     # e.g. shape=(60000,1)-->x1.shape=(100,100,6)
    z5=100*z_func_cos_bump(points).reshape(x1.shape)
    z6=100*z_func_cosine_depression(points).reshape(x1.shape)
    z7=100*z_func_tital(points).reshape(x1.shape)
    h4,h5,h6,h7=u_pred_4[:,:,:,0],u_pred_5[:,:,:,0],u_pred_6[:,:,:,0],u_pred_7[:,:,:,0]
    # u4,u5,u6,u7=u_pred_4[:,:,:,1],u_pred_5[:,:,:,1],u_pred_6[:,:,:,1],u_pred_7[:,:,:,1]
    # v4,v5,v6,v7=u_pred_4[:,:,:,2],u_pred_5[:,:,:,2],u_pred_6[:,:,:,2],u_pred_7[:,:,:,2]
    net_rain4=h4+z4
    net_rain5=h5+z5-water_level
    net_rain6=h6+z6-water_level
    net_rain7=h7+z7-water_level

    #average
    r1=np.mean(net_rain4, axis=(0, 1))
    r2=np.mean(net_rain5, axis=(0, 1))
    r3=np.mean(net_rain6, axis=(0, 1))
    r4=np.mean(net_rain7, axis=(0, 1))
    r=[r1,r2,r3,r4]

    #

    times=np.linspace(bbox[4],bbox[5],timesplit)
    cum_rain=cum_rain_func(times)
    error=[np.abs(r[i]-cum_rain) for i in range(4)]
    # errors_fit=[]
    # h_fit=[]
    # # for i in range(4):
    # #     # error=np.abs(r[i]-cum_rain)   #error line
    # #     #Savitzky-Golay滤波器
    # #     from scipy.signal import savgol_filter
    # #     h=savgol_filter(r[i], 51, 3)
    # #     h_fit.append()
    # #     errors_fit.append(np.abs(h_fit-cum_rain))   #error line

    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(10, 6))
    gs = gridspec.GridSpec(2, 2)  
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    axs=[ax1,ax2,ax3,ax4]
    title_list=['(a)','(b)','(c)','(d)']


    for i,ax in enumerate(axs):
        cmap = plt.get_cmap('coolwarm')
        ax.plot(times,cum_rain,label="True",color='black',linestyle='solid')
        ax.plot(times,r[i],label="PINN",color='red',linestyle='-.')

        # axs[1,1].set_aspect(80)
        ax.set_xlabel('t (min)',fontsize=11)
        ax.set_ylabel('h (cm)',fontsize=11)

        # axs[1,1].set_ylim(1.0,1.07)
        plot_error=True
        ax.legend(loc='upper left',fontsize=11)
        if plot_error:
            ax_f = ax.twinx()
            ax_f.plot(times,error[i],label="Error",color='purple',linestyle=':')
            ax_f.set_ylabel('Error (cm)',color='purple',fontsize=11)
            ax_f.tick_params(axis='y', colors='purple')
            # ax_f.set_ylim(0,1e-3)
            from matplotlib.ticker import FuncFormatter
            def sci_formatter(x, pos):
                if x >0 and x<1: 
                    return '{:.1e}'.format(x)
            formatter = FuncFormatter(sci_formatter)
            ax_f.yaxis.set_major_formatter(formatter)
           
            ax_f.legend(loc='lower right',fontsize=11)

        ax.set_title(title_list[i],fontsize=12)
    #plot the intensity of rainfall
    intensity=np.array([rain_intensity(t) for t in times])
    ax1.plot(times,intensity,label="Intensity",color='blue',linestyle='--')
    ax1.legend(loc='upper left',fontsize=11)

    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.893, top=0.95, wspace=0.62, hspace=0.43)

    #save figure
    plt.savefig(f"{save_path}/static_rain.jpg",dpi=500)
    
    plt.show()
    
def plot_4_7_cases_slices(save_path,net4,net5,net6,net7):
    """for the cases of 4,5,6,7: flat_rain,cosine_bump_rian,cosine_depression_rain,tidal_rain"""
    # water_level=30  #cm
    timesplit=6
    _,_,_,_,u_pred_4=test_mesh_points_and_results(net4,timesplit)
    _,_,_,_,u_pred_5=test_mesh_points_and_results(net5,timesplit)
    _,_,_,_,u_pred_6=test_mesh_points_and_results(net6,timesplit)
    xx,yy,tt,points,u_pred_7=test_mesh_points_and_results(net7,timesplit)
    #e.g.  x1: shape=(100,100,6)  points: shape=(60000,3)  u_pred: shape=(100,100,6,3)  
    u_pred=[u_pred_4,u_pred_5,u_pred_6,u_pred_7]
    z4=100*z_func_flat(points).reshape(xx.shape)     # e.g. shape=(60000,1)-->x1.shape=(100,100,6)
    z5=100*z_func_cos_bump(points).reshape(xx.shape)
    z6=100*z_func_cosine_depression(points).reshape(xx.shape)
    z7=100*z_func_tital(points).reshape(xx.shape)
    z=[z4,z5,z6,z7]
    name=["flat_rain","cosine_bump_rian","cosine_depression_rain","tidal_rain"]
    

    def plot_single_cases(save_path,name,bbox,xx,yy,tt,z,u_pred,timesplit=6):
        #plot
        fig = plt.figure(figsize=(10, 6))
        axs=fig.subplots(2,2,subplot_kw={"projection": "3d"})

        title_list=['(a)','(b)','(c)','(d)']
        cor_label=['cm','cm/min','cm/min']
        for i in range(3):
            plot_3dheatmap(fig,axs[i//2,i%2],bbox,xx, yy, tt, u_pred[:,:,:, i],
                            timesplit, title=title_list[i],cor_label=cor_label[i])
        
        h_final=u_pred[:,:,:,0][:,:,-1]
        plot_water_level_slice(axs[1,1],xx[:,:,-1],yy[:,:,-1],h_final+z[:,:,-1],title=title_list[-1]) 

        #绘制1*4的子图

        plt.subplots_adjust(left=0.1, bottom=0, right=0.9, top=1, wspace=0.36, hspace=0.)
        #save figure
        plt.savefig(f"{save_path}/{name}",dpi=500)

        plt.show()
    for i in range(4):
        plot_single_cases(save_path,name[i],bbox,xx,yy,tt,z[i],u_pred[i])

def test_error_4_7_cases(net4,net5,net6,net7,time_slice=[0,5]):
    water_level=[0,30,30,30]  #cm
    z_funcs=[z_func_flat,z_func_cos_bump,z_func_cosine_depression,z_func_tital]
    name=["net4","net5","net6","net7"]
    error_list=[]
    for i,net in enumerate([net4,net5,net6,net7]):
        for t in time_slice:
            U_pred,z,_,_=get_pred_of_the_time(net,t,z_funcs[i])   #e.g. u_pred: shape=(100,100,3)  z: shape=(100,100)
            h_true=(water_level[i]+cum_rain_func([t])[0]-100*z).flatten()  #e.g. shape=(10000,)
            u_x_true,v_y_true=np.zeros_like(h_true),np.zeros_like(h_true)  #u_x_true,v_y_true=0,0  
            U_pred=U_pred.reshape(-1,3) #e.g. shape=(10000,3)
            U_true=np.column_stack((h_true,u_x_true,v_y_true))
            mae=MAE(U_pred,U_true)
            rmse=RMSE(U_pred,U_true)
            print(f"case   {name[i]} time{t} mae:{mae} rmse:{rmse}")
            
            error_list.append(np.append(mae,rmse))
    error_list=np.array(error_list)
    #输出：科学计数，小数点后一位数字

    print(np.array2string(error_list,formatter={'float_kind':lambda x: "%.1e" % x}))

    
            

def get_pred_of_the_time(net,time,z_func=None):
    x=np.linspace(bbox[0],bbox[1],100)
    y=np.linspace(bbox[2],bbox[3],100)
    xx,yy=np.meshgrid(x,y)
    points=np.vstack((xx.flatten(),yy.flatten(),time+np.zeros_like(xx.flatten()))).T
    torch_points=torch.tensor(points).float().to(device)
    u_pred=net(torch_points)
    u_pred=u_pred.detach().cpu().numpy().reshape((*xx.shape,-1)) #e.g. shape=(100,100,3)
    if z_func is not None:
        z=z_func(points).reshape(xx.shape)    #e.g. shape=(10000,1)-->x1.shape=(100,100)
    else: z=None
    return u_pred,z,xx,yy #numpy

def test_error(pred,true):
    return MAE(pred,true),RMSE(pred,true)


def MAE(pred,true):
    return np.mean(np.abs(pred-true),axis=0)

def RMSE(pred,true):
    return np.sqrt(np.mean((pred-true)**2,axis=0))

    #
def plot_water_level_slice(ax,x,y,wl_pred,title,**plot_kwargs):
    X,Y=x,y

    cmap=plot_kwargs.pop('cmap','rainbow') #default cmap='rainbow'
    alpha=plot_kwargs.pop('alpha',0.8) #default alpha=0.8,control the transparency 
    

    ax.plot_surface(X,Y,wl_pred,cmap=cmap,alpha=alpha)
    ax.set_xlabel('x (m)',fontsize=11)
    ax.set_ylabel('y (m)',fontsize=11)
    ax.set_zlabel('h(cm)',labelpad=10,fontsize=11)  #lablepad: adjust the zlabel position
    ax.tick_params(axis='both', which='major', labelsize=11)

    ax.view_init(elev=10, azim=45)
    ax.set_zlim(wl_pred.min(), wl_pred.max())
    ax.set_title(title,pad=0,y=0.95,fontsize=12)

def plot_3dheatmap(fig,ax,bbox,xx, yy, zz, vals,time_split=6,
                   title='', xlabel='x (m)', ylabel='y (m)', zlabel='t (min)',cor_label='cm'):

    ax.view_init(elev=10, azim=45)
    norm = plt.Normalize(vmin=vals.min(), vmax=vals.max())  
    for i in range(time_split):
        # ax.contourf(xx[:, :, i], yy[:, :, i], vals[:, :, i], offset=zz[0][0][i], cmap="coolwarm")
        ax.contourf(xx[:, :, i], yy[:, :, i], vals[:, :, i], offset=zz[0][0][i], cmap="coolwarm", norm=norm)
    ax.set(xlim=(bbox[0], bbox[1]), ylim=(bbox[2], bbox[3]), zlim=(bbox[4], bbox[5]))
    ax.set_xlabel(xlabel,fontsize=11)
    ax.set_ylabel(ylabel,fontsize=11)
    ax.set_zlabel(zlabel,fontsize=11)
    
    
    # cor=fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap="coolwarm"), ax=ax, shrink=0.5,pad=0.06) 
    cor=fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap="coolwarm"), ax=ax, shrink=0.5,pad=0.06) 

    cor.set_label(cor_label)
    ax.set_title(title,pad=0,y=0.97,fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=11)
    ax.set_xticks(np.arange(bbox[0], bbox[1]+1,4))   #
    ax.set_yticks(np.arange(bbox[2], bbox[3]+1, 4))

def plot_1_3_cases(save_path,net1,net2,net3,time_slice=1,save=True):
    """for the cases of 1,2,3: cosine_bump,cosine_depression,tidal"""
    nets=[net1,net2,net3]
    z_funcs=[z_func_cos_bump,z_func_cosine_depression,z_func_tital]
    title=['(a)','(b)','(c)','(d)']

    def plot_slices(save_path,name,xx,yy,z,h,u,v,**plot_kwargs):
        X,Y=xx,yy
        cmap=plot_kwargs.pop('cmap','rainbow') #default cmap='rainbow'
        alpha=plot_kwargs.pop('alpha',0.5) #default alpha=0.8,control the transparency 
        #plot 1*4 2D
        fig,axs=plt.subplots(2,2,subplot_kw={"projection": "3d"})
        # axs[0,0].plot_surface(X,Y,z,cmap='gray')
        axs[0,0].plot_surface(X,Y,z+h,cmap=cmap,alpha=alpha)
        # axs[0,0].plot_surface(X, Y, Z+1.0, color='gray', alpha=0.2)
        axs[0,0].set_xlabel('x (m)',fontsize=11)
        axs[0,0].set_ylabel('y (m)',fontsize=11)
        axs[0,0].set_title(title[0],y=0.99,fontsize=12)
        axs[0,0].set_zlabel('z+h (m)',labelpad=10,fontsize=11)  #lablepad: adjust the zlabel position

        axs[0,1].plot_surface(X,Y,h,cmap=cmap,alpha=alpha)
        axs[0,1].set_xlabel('x (m)',fontsize=11)
        axs[0,1].set_ylabel('y (m)',fontsize=11)
        axs[0,1].set_zlabel('h (m)',labelpad=10,fontsize=11)
        axs[0,1].set_title(title[1],y=0.99,fontsize=12)

        axs[1,0].plot_surface(X,Y,u,cmap=cmap,alpha=alpha)
        axs[1,0].set_xlabel('x (m)',fontsize=11)
        axs[1,0].set_ylabel('y (m)',fontsize=11)
        axs[1,0].set_zlabel('u (m/s)',labelpad=10,fontsize=11)
        axs[1,0].set_title(title[2],y=0.99,fontsize=12)

        axs[1,1].plot_surface(X,Y,v,cmap=cmap,alpha=alpha)
        axs[1,1].set_xlabel('x (m)',fontsize=11)
        axs[1,1].set_ylabel('y (m)',fontsize=11)
        axs[1,1].set_zlabel('v (m/s)',labelpad=10,fontsize=11)
        axs[1,1].set_title(title[3],y=0.99,fontsize=12)

        # plt.tight_layout(10,8)
        fig.set_size_inches(10, 7)
        fig.subplots_adjust(left=0.01, right=0.94, top=0.93, bottom=0.05, wspace=0.1,hspace=0.18)
        #save figure
        if save==True:
            plt.savefig(f"{save_path}/{name}.jpg",dpi=500)

        plt.show()

    name=["cosine_bump","cosine_depression","tidal"]
    for i,net in enumerate(nets):
        U_pred,z,xx,yy=get_pred_of_the_time(net,time_slice,z_funcs[i])
        h,u,v=U_pred[:,:,0],U_pred[:,:,1],U_pred[:,:,2]
        # z=100*z
        plot_slices(save_path,name[i],xx,yy,z,h,u,v)   
        
def test_error_1_3_cases(net1,net2,net3,time_slice=[0,5]):
    water_level=[.30,.30,.30]  #m
    nets=[net1,net2,net3]
    z_funcs=[z_func_cos_bump,z_func_cosine_depression,z_func_tital]
    name=["net1","net2","net3"]
    error_list=[]
    for i,net in enumerate(nets):
        for t in time_slice:
            U_pred,z,_,_=get_pred_of_the_time(net,t,z_funcs[i])   #e.g. u_pred: shape=(100,100,3)  z: shape=(100,100)
            # z=100*z
            h_true=(water_level[i]-z).flatten()  #e.g. shape=(10000,)
            u_x_true,v_y_true=np.zeros_like(h_true),np.zeros_like(h_true)  #u_x_true,v_y_true=0,0  
            U_pred=U_pred.reshape(-1,3) #e.g. shape=(10000,3)
            U_true=np.column_stack((h_true,u_x_true,v_y_true))
            mae=MAE(U_pred,U_true)
            rmse=RMSE(U_pred,U_true)
            print(f"case   {name[i]} time{t} mae:{mae} rmse:{rmse}")
            
            error_list.append(np.append(mae,rmse))
    error_list=np.array(error_list)

    print(np.array2string(error_list,formatter={'float_kind':lambda x: "%.1e" % x}))

def plot_static_tidal_cases(save_path,net3_en,net3,time_slice=1,save=True):
    """for the cases of 1,2,3: cosine_bump,cosine_depression,tidal"""
    # nets=[net3,net3_en]
    z_funcs=z_func_tital
    title=['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)']

    # name=["cosine_bump","cosine_depression","tidal"]
    U_pred,z,xx,yy=get_pred_of_the_time(net3_en,time_slice,z_funcs)
    U_pred_en,z_en,xx_en,yy_en=get_pred_of_the_time(net3,time_slice,z_funcs)
    h,u,v=U_pred[:,:,0],U_pred[:,:,1],U_pred[:,:,2]
    h_en,u_en,v_en=U_pred_en[:,:,0],U_pred_en[:,:,1],U_pred_en[:,:,2]

    X,Y=xx,yy
    cmap='rainbow'
    alpha=0.5
    #plot 4*2 2D
    fig,axs=plt.subplots(4,2,subplot_kw={"projection": "3d"})
    # axs[0,0].plot_surface(X,Y,z,cmap='gray')
    axs[0,0].plot_surface(X,Y,z+h,cmap=cmap,alpha=alpha)
    # axs[0,0].plot_surface(X, Y, Z+1.0, color='gray', alpha=0.2)
    axs[0,0].set_xlabel('x (m)',fontsize=11)
    axs[0,0].set_ylabel('y (m)',fontsize=11)
    axs[0,0].set_title(title[0],y=0.99,fontsize=12)
    axs[0,0].set_zlabel('z+h (m)',labelpad=10,fontsize=11)  #lablepad: adjust the zlabel position

    axs[0,1].plot_surface(X,Y,h,cmap=cmap,alpha=alpha)
    axs[0,1].set_xlabel('x (m)',fontsize=11)
    axs[0,1].set_ylabel('y (m)',fontsize=11)
    axs[0,1].set_zlabel('h (m)',labelpad=10,fontsize=11)
    axs[0,1].set_title(title[1],y=0.99,fontsize=12)

    axs[1,0].plot_surface(X,Y,u,cmap=cmap,alpha=alpha)
    axs[1,0].set_xlabel('x (m)',fontsize=11)
    axs[1,0].set_ylabel('y (m)',fontsize=11)
    axs[1,0].set_zlabel('u (m/s)',labelpad=10,fontsize=11)
    axs[1,0].set_title(title[2],y=0.99,fontsize=12)

    axs[1,1].plot_surface(X,Y,v,cmap=cmap,alpha=alpha)
    axs[1,1].set_xlabel('x (m)',fontsize=11)
    axs[1,1].set_ylabel('y (m)',fontsize=11)
    axs[1,1].set_zlabel('v (m/s)',labelpad=10,fontsize=11)
    axs[1,1].set_title(title[3],y=0.99,fontsize=12)

    #net3_en
    axs[2,0].plot_surface(X,Y,z_en+h_en,cmap=cmap,alpha=alpha)
    axs[2,0].set_xlabel('x (m)',fontsize=11)
    axs[2,0].set_ylabel('y (m)',fontsize=11)    
    axs[2,0].set_zlabel('z+h (m)',labelpad=10,fontsize=11)  #lablepad: adjust the zlabel position
    axs[2,0].set_title(title[4],y=0.99,fontsize=12)

    axs[2,1].plot_surface(X,Y,h_en,cmap=cmap,alpha=alpha)
    axs[2,1].set_xlabel('x (m)',fontsize=11)
    axs[2,1].set_ylabel('y (m)',fontsize=11)
    axs[2,1].set_zlabel('h (m)',labelpad=10,fontsize=11)
    axs[2,1].set_title(title[5],y=0.99,fontsize=12)

    axs[3,0].plot_surface(X,Y,u_en,cmap=cmap,alpha=alpha)
    axs[3,0].set_xlabel('x (m)',fontsize=11)
    axs[3,0].set_ylabel('y (m)',fontsize=11)
    axs[3,0].set_zlabel('u (m/s)',labelpad=10,fontsize=11)
    axs[3,0].set_title(title[6],y=0.99,fontsize=12)

    axs[3,1].plot_surface(X,Y,v_en,cmap=cmap,alpha=alpha)
    axs[3,1].set_xlabel('x (m)',fontsize=11)
    axs[3,1].set_ylabel('y (m)',fontsize=11)
    axs[3,1].set_zlabel('v (m/s)',labelpad=10,fontsize=11)
    axs[3,1].set_title(title[7],y=0.99,fontsize=12)


    # plt.tight_layout(10,8)
    fig.set_size_inches(10, 14)
    fig.subplots_adjust(left=0.01, right=0.94, top=0.93, bottom=0.05, wspace=0.1,hspace=0.18)
    #save figure
    name="static_tidal"
    if save:
        plt.savefig(f"{save_path}/{name}.jpg",dpi=500)
    plt.show()
    


def z_func_flat(points): 
    z=np.zeros_like(points[:,0:1])
    return z

def z_func_cos_bump(points):
    x,y=points[:,0:1],points[:,1:2]
    x_square_add_y_square=x**2+y**2    
    index=np.where(x_square_add_y_square<radius**2)     # points in the unit circle
    z=np.zeros_like(x)                                       # out of the unit circle is 0.
    phi=np.pi/radius**2*x_square_add_y_square[index]
    z[index]=height/2.*(1.+np.cos(phi))   # inside the unit circle
    return z

def z_func_cosine_depression(points):
    x,y=points[:,0:1],points[:,1:2]
    x_square_add_y_square=x**2+y**2    
    index=np.where(x_square_add_y_square<radius**2)     # points in the unit circle
    z=np.zeros_like(x)                                       # out of the unit circle is 0.
    phi=np.pi/radius**2*x_square_add_y_square[index]
    z[index]=-height/2.*(1.+np.cos(phi))   # inside the unit circle
    return z

def z_func_tital(points):
    x=points[:,0:1]
    y=points[:,1:2]
    z=height*np.cos(np.pi/geom[0][1]*x)*np.cos(np.pi/geom[1][1]*y)
    return z

def HLL_solution_for_dam_break(x,time_slice=[0,1,2]):
    """ """
    from ref.finite_volume_2d import FV_slover,Dam_Break
    dx=(bbox[1]-bbox[0])/(x.shape[0])  #  e.g. 0.2 
    report_interval=time_slice[1]-time_slice[0]    # the default is mutiples 
    damB=Dam_Break(bbox=[[bbox[0],bbox[1]],[bbox[2],bbox[3]]],
                   dx_dy=(dx,dx),  t_end=time_slice[-1],
                   report_interval=report_interval,
                   init_value=[2,1])
    damB_FV=FV_slover(**damB(),device=device)
    X,Y,resutls,report_times=damB_FV()
    #plot the solution
    # damB.plot_results(resutls,report_times)
    if time_slice[0]!=0:   
        resutls=resutls[1:]
    resutls=np.array(resutls)        #e.g. shape=(times,100,100,3)
    resutls=np.mean(resutls,axis=1)  #e.g. shape=(times,100,3)
    hs_hll= resutls[:,:,0].T            #e.g. shape=(times,100)
    us_hll= resutls[:,:,1].T/hs_hll   #hu-->u
    vs_hll= resutls[:,:,2].T/hs_hll   #hv-->v
    x=X[0,:]
    return hs_hll,us_hll,vs_hll,x
    
    
def Riemann_solution(x,time_slice=[0,1,2]):
    """solve the Riemann problem for the given time slice and position x"""
    from ref.riemann_problems import RiemannProblem,create_valdata_from_analytical
    dim_vals=[2,0,1,0]    # 
    prb=RiemannProblem(code="custom",dim_vals=dim_vals, L=bbox[1]-bbox[0], xdisc=0, tmax=time_slice[-1])
    hs_true,us_true,vs_true=[],[],[]
    hs_true=np.zeros((x.shape[0],len(time_slice)))
    us_true=np.zeros_like(hs_true)
    vs_true=np.zeros_like(hs_true)
    for i,t in enumerate(time_slice):
        x, t, h_true, u_true, _=create_valdata_from_analytical(prb,t,n_pts=x.shape[0])  
        hs_true[:,i:i+1],us_true[:,i:i+1]=h_true,u_true

    return hs_true,us_true,vs_true,x[:,0]  #

def plot_8_case_huv(save_path,net8,time_slice=[0,1,2],save=True): 
    """for the cases of 8 """
    times = len(time_slice)
    dx,num=0.2,100
    x = np.linspace(bbox[0]+dx/2.,bbox[1]-dx/2.,num)   #dx=0.1 num=200 
    y=np.linspace(bbox[2]+dx/2.,bbox[3]-dx/2.,num)     #dx=0.1 num=200 
    t=np.array(time_slice)
    xx,yy,tt=np.meshgrid(x,y,t)
    points=np.vstack((xx.flatten(),yy.flatten(),tt.flatten())).T
    torch_points=torch.tensor(points).float().to(device)

    U_pred=net8(torch_points)
    U_pred=U_pred.detach().cpu().numpy().reshape((*xx.shape,-1)) #e.g. shape=(100,100,3,3)
    hs,us,vs=U_pred[:,:,:,0],U_pred[:,:,:,1],U_pred[:,:,:,2]  #e.g. shape=(100,100,3)
    # #mean over y axis
    hs=np.mean(hs,axis=0) #e.g. shape=(100,3)
    us=np.mean(us,axis=0)
    vs=np.mean(vs,axis=0)

    #solve the true solutin
    hs_true,us_true,vs_true,_xr=Riemann_solution(x,time_slice) 
    #HLL solution
    hs_hll,us_hll,vs_hll,_x_hll=HLL_solution_for_dam_break(x,time_slice) 

    if ~np.isclose(x,_xr).all() or ~np.isclose(x,_x_hll).all():
        raise ValueError("x is not the same")    
    
    hs_error=np.abs(hs-hs_true)
    us_error=np.abs(us-us_true)
    vs_error=np.abs(vs-vs_true)


    title=["(a)","(b)", "(c)"]

    fig, axs = plt.subplots(times,3, sharex=True, figsize=(12.6, 7.8),layout='constrained')
        
    for i in range(0, times):
        axs[i,0].plot(x, hs_true[:,i],'k-', label="Exact")
        axs[i,0].plot(x, hs[:,i], "r-.", label="PINN")
        axs[i,0].plot(x, hs_hll[:,i], "b--", label="HLL")
        axs[i,0].set_ylabel("h (m)")

        # add the second axis and plot the error
        ax_twin_1=axs[i,0].twinx()
        ax_twin_1.plot(x, hs_error[:,i],linestyle=':',color='purple', label="Error")
        ax_twin_1.set_ylabel("error (m)",color='purple')
        ax_twin_1.tick_params(axis='y', colors='purple')
        if i==0:    
            ax_twin_1.legend(loc='lower right')

        axs[i,1].plot(x, us_true[:,i],'k-', label="Exact")
        axs[i,1].plot(x, us[:,i], "r-.", label="PINN")
        axs[i,1].plot(x, us_hll[:,i], "b--", label="HLL")
        axs[i,1].set_ylabel("u (m/s)")

        #add the second axis and plot the error
        ax_twin_2=axs[i,1].twinx()
        ax_twin_2.plot(x, us_error[:,i],linestyle=':',color='purple', label="Error")
        ax_twin_2.set_ylabel("error (m/s)",color='purple')
        ax_twin_2.tick_params(axis='y', colors='purple')
        
        
        axs[i,2].plot(x, vs_true[:,i],'k-', label="Exact")
        axs[i,2].plot(x, vs[:,i], "r-.", label="PINN")
        axs[i,2].plot(x, vs_hll[:,i], "b--", label="HLL")
        axs[i,2].set_ylabel("v (m/s)")

        #add the second axis and plot the error
        # ax_twin_3=axs[i,2].twinx()
        # ax_twin_3.plot(x, vs_error[:,i],linestyle=':',color='purple', label="Error")
        # ax_twin_3.set_ylabel("error (m/s)",color='purple')
        # ax_twin_3.tick_params(axis='y', colors='purple')



    axs[-1,0].set_xlabel("x (m)")
    axs[-1,1].set_xlabel("x (m)")
    axs[-1,2].set_xlabel("x (m)")
    axs[0,0].legend()
    # axs[0,0].twinx().legend("Error",loc='lower right')
    axs[0,0].set_title(title[0])
    axs[0,1].set_title(title[1])
    axs[0,2].set_title(title[2])

    
    fig.tight_layout()
    #save figure
    if save:
        plt.savefig(f"{save_path}/case_8.jpg",dpi=500)
    plt.show()

def plot_8_case_muti(save_path,net8_pri,net8_en,net8_vc,time_slice=[0,1,2],save=True): 
    """for the cases of 8 """
    times = len(time_slice)
    dx,num=0.08,250
    x = np.linspace(bbox[0]+dx/2.,bbox[1]-dx/2.,num)   #e.g. dx=0.1 num=200 
    y=np.linspace(bbox[2]+dx/2.,bbox[3]-dx/2.,num)     #e.g. dx=0.1 num=200 
    t=np.array(time_slice)
    xx,yy,tt=np.meshgrid(x,y,t)
    points=np.vstack((xx.flatten(),yy.flatten(),tt.flatten())).T
    torch_points=torch.tensor(points).float().to(device)

    U_pred_1=net8_pri(torch_points)
    U_pred_2=net8_en(torch_points)  
    U_pred_3=net8_vc(torch_points)

    U_pred_1=U_pred_1.detach().cpu().numpy().reshape((*xx.shape,-1)) #e.g. shape=(100,100,3,3)
    hs_1,us_1,vs_1=U_pred_1[:,:,:,0],U_pred_1[:,:,:,1],U_pred_1[:,:,:,2]  #e.g. shape=(100,100,3)

    U_pred_2=U_pred_2.detach().cpu().numpy().reshape((*xx.shape,-1)) #e.g. shape=(100,100,3,3)
    hs_2,us_2,vs_2=U_pred_2[:,:,:,0],U_pred_2[:,:,:,1],U_pred_2[:,:,:,2]  #e.g. shape=(100,100,3)
    U_pred_3=U_pred_3.detach().cpu().numpy().reshape((*xx.shape,-1)) #e.g. shape=(100,100,3,3)
    hs_3,us_3,vs_3=U_pred_3[:,:,:,0],U_pred_3[:,:,:,1],U_pred_3[:,:,:,2]  #e.g. shape=(100,100,3)
    # #mean over y axis
    hs_1=np.mean(hs_1,axis=0) #e.g. shape=(100,3)
    us_1=np.mean(us_1,axis=0)
    vs_1=np.mean(vs_1,axis=0)
    hs_2=np.mean(hs_2,axis=0) #e.g. shape=(100,3)
    us_2=np.mean(us_2,axis=0)
    vs_2=np.mean(vs_2,axis=0)
    hs_3=np.mean(hs_3,axis=0) #e.g. shape=(100,3)
    us_3=np.mean(us_3,axis=0)
    vs_3=np.mean(vs_3,axis=0)

    #solve the true solutin
    hs_true,us_true,vs_true,_xr=Riemann_solution(x,time_slice) 
    #HLL solution
    hs_hll,us_hll,vs_hll,_x_hll=HLL_solution_for_dam_break(x,time_slice) 

    if ~np.isclose(x,_xr).all() or ~np.isclose(x,_x_hll).all():
        raise ValueError("x is not the same")    
    
    hs_error_1=np.abs(hs_1-hs_true)
    us_error_1=np.abs(us_1-us_true)
    vs_error_1=np.abs(vs_1-vs_true)
    hs_error_2=np.abs(hs_2-hs_true)
    us_error_2=np.abs(us_2-us_true)
    vs_error_2=np.abs(vs_2-vs_true)
    hs_error_3=np.abs(hs_3-hs_true)
    us_error_3=np.abs(us_3-us_true)
    vs_error_3=np.abs(vs_3-vs_true)


    title=["(a)","(b)", "(c)"]

    fig, axs = plt.subplots(times,3, sharex=True, figsize=(12.6, 7.8),layout='constrained')
        
    for i in range(0, times):
        axs[i,0].plot(x, hs_true[:,i],'k-', label="Exact")
        axs[i,0].plot(x, hs_1[:,i], "g-.", label="PINN_v")
        axs[i,0].plot(x, hs_2[:,i], "c-.", label="PINN_es")
        axs[i,0].plot(x, hs_3[:,i], "r--", label="PINN_vc")
        axs[i,0].plot(x, hs_hll[:,i], "b--", label="HLL")
        axs[i,0].set_ylabel("h (m)")

        # add the second axis and plot the error
        ax_twin_1=axs[i,0].twinx()
        ax_twin_1.plot(x, hs_error_3[:,i],linestyle=':',color='purple', label="Error")
        ax_twin_1.set_ylabel("error (m)",color='purple')
        ax_twin_1.tick_params(axis='y', colors='purple')
        if i==0:    
            ax_twin_1.legend(loc='lower right')

        axs[i,1].plot(x, us_true[:,i],'k-', label="Exact")
        axs[i,1].plot(x, us_1[:,i], "g-.", label="PINN")
        axs[i,1].plot(x, us_2[:,i], "c-.", label="PINN_entropy")
        axs[i,1].plot(x, us_3[:,i], "r-.", label="PINN_var_con")
        axs[i,1].plot(x, us_hll[:,i], "b--", label="HLL")
        axs[i,1].set_ylabel("u (m/s)")

        #add the second axis and plot the error
        ax_twin_2=axs[i,1].twinx()
        ax_twin_2.plot(x, us_error_3[:,i],linestyle=':',color='purple', label="Error")
        ax_twin_2.set_ylabel("error (m/s)",color='purple')
        ax_twin_2.tick_params(axis='y', colors='purple')
        
        
        axs[i,2].plot(x, vs_true[:,i],'k-', label="Exact")
        axs[i,2].plot(x, vs_1[:,i], "g-.", label="PINN")
        axs[i,2].plot(x, vs_2[:,i], "c-.", label="PINN_entropy")
        axs[i,2].plot(x, vs_3[:,i], "r-.", label="PINN_var_con")
        axs[i,2].plot(x, vs_hll[:,i], "b--", label="HLL")
        axs[i,2].set_ylabel("v (m/s)")

        #add the second axis and plot the error
        # ax_twin_3=axs[i,2].twinx()
        # ax_twin_3.plot(x, vs_error[:,i],linestyle=':',color='purple', label="Error")
        # ax_twin_3.set_ylabel("error (m/s)",color='purple')
        # ax_twin_3.tick_params(axis='y', colors='purple')



    axs[-1,0].set_xlabel("x (m)")
    axs[-1,1].set_xlabel("x (m)")
    axs[-1,2].set_xlabel("x (m)")
    axs[0,0].legend()
    # axs[0,0].twinx().legend("Error",loc='lower right')
    axs[0,0].set_title(title[0])
    axs[0,1].set_title(title[1])
    axs[0,2].set_title(title[2])

    #write the text in axs[0,0],axs[1,0],axs[2,0]
    axs[0,0].text(0.1, 0.5, "t=0s", transform=axs[0,0].transAxes, fontsize=12,
        verticalalignment='center', horizontalalignment='left')
    axs[1,0].text(0.1, 0.5, "t=0.6s", transform=axs[1,0].transAxes, fontsize=12,
        verticalalignment='center', horizontalalignment='left')
    axs[2,0].text(0.1, 0.5, "t=1.2s", transform=axs[2,0].transAxes, fontsize=12,
        verticalalignment='center', horizontalalignment='left')

    
    fig.tight_layout()
    #save figure
    if save:
        plt.savefig(f"{save_path}/case_8.jpg",dpi=500)
    plt.show()


def curve_at_time(data,time=1.2):
    data_1s=np.where(np.abs(data[:,2]-time)<1e-6)[0]
    if data_1s.shape[0]==0:
        raise ValueError('time not found in the data')
    x=data[data_1s,0]
    y=data[data_1s,1]
    h=data[data_1s,3]
    u=data[data_1s,4]
    v=data[data_1s,5]
    
    #sort the data by x
    x_sort_index=np.argsort(x)
    x=x[x_sort_index]
    y=y[x_sort_index]
    h=h[x_sort_index]    
    u=u[x_sort_index]
    v=v[x_sort_index]
    z=z[x_sort_index]

    #去重
    x_new=[]
    h_new,u_new,v_new=[],[],[]
    _temp,_u_temp,_v_temp,count=0.,0.,0.,0.
    for i in range(len(x)-1):
        if x[i]==x[i+1]:
            _temp+=h[i]
            _u_temp+=u[i]
            _v_temp+=v[i]
            count+=1
        else:
            x_new.append(time[i])
            h_new.append(_temp/count)
            u_new.append(_u_temp/count)
            v_new.append(_v_temp/count)
            _temp,count=h[i+1],1
            _u_temp=u[i+1]
            _v_temp=v[i+1]
    else:
        _temp+=h[-1]
        _u_temp+=u[-1]
        _v_temp+=v[-1]
        x_new.append(time[-1])
        h_new.append(_temp/(count+1)) 
        u_new.append(_u_temp/(count+1))
        v_new.append(_v_temp/(count+1))
    x=np.array(x_new)
    h=np.array(h_new)
    u=np.array(u_new)
    v=np.array(v_new)
    return x,h,u,v


def HLL_solution_for_circular_dam_break(x,time_slice=[0,1,2]):
    """ """
    from ref.finite_volume_2d import FV_slover,Circular_Dam_Break,FV_slover_Well_Balance
    dx=(bbox[1]-bbox[0])/(x.shape[0])  #  e.g. 0.2 
    report_interval=time_slice[1]-time_slice[0]    # the default is mutiples 
    damB=Circular_Dam_Break(bbox=[[bbox[0],bbox[1]],[bbox[2],bbox[3]]],
                   dx_dy=(dx,dx),  
                   t_end=time_slice[-1],
                   report_interval=report_interval,
                   init_value=[2,1])
    damB_FV=FV_slover(**damB(),device=device)
    X,Y,resutls,report_times=damB_FV()
    #plot the solution
    # damB.plot_results(resutls,report_times)
    if time_slice[0]!=0:   
        resutls=resutls[1:]
    resutls=np.array(resutls)        #shape=(times,100,100,3)
    hs_hll= resutls[:,:,:,0]       #shape=(times,100,100)
    us_hll_= resutls[:,:,:,1]/hs_hll   #hu-->u
    vs_hll_= resutls[:,:,:,2]/hs_hll   #hv-->v
    hs_hll=hs_hll.transpose((1,2,0))   #shape=(100,100,times)
    us_hll=us_hll_.transpose((1,2,0))
    vs_hll=-vs_hll_.transpose((1,2,0))

    x,y=X[0,:],Y[:,0]
    return hs_hll,us_hll,vs_hll,x,y
    


def plot_circulatr_dam_break(save_path,net9,time_slice=[0,0.25,.5]):
    """plot the solution of circular dam break"""
    times = len(time_slice)
    dx,num=0.04,500
    x = np.linspace(bbox[0]+dx/2.,bbox[1]-dx/2.,num)   #dx=0.1 num=200 
    y=np.linspace(bbox[2]+dx/2.,bbox[3]-dx/2.,num)     #dx=0.1 num=200
    t=np.array(time_slice)
    xx,yy,tt=np.meshgrid(x,y,t)
    points=np.vstack((xx.flatten(),yy.flatten(),tt.flatten())).T
    torch_points=torch.tensor(points).float().to(device)

    U_pred=net9(torch_points)
    U_pred=U_pred.detach().cpu().numpy().reshape((*xx.shape,-1)) #e.g. shape=(100,100,3,3)
    hs,us,vs=U_pred[:,:,:,0],U_pred[:,:,:,1],U_pred[:,:,:,2]  #e.g. shape=(100,100,3)

    if True:
    #HLL solution
        hs_hll,us_hll,vs_hll,_x_hll,_y_hll=HLL_solution_for_circular_dam_break(x,time_slice)   #

        if ~np.isclose(x,_x_hll).all() and ~np.isclose(y,_y_hll).all():
            raise ValueError("x or y is not the same")
        
        
        hs_error=np.abs(hs-hs_hll)  
        us_error=np.abs(us-us_hll)
        vs_error=np.abs(vs-vs_hll)

        rmse_h= np.sqrt(np.mean((hs-hs_hll)**2,axis=(0,1)))
        mae_h=np.mean(np.abs(hs-hs_hll),axis=(0,1))
        rmse_u= np.sqrt(np.mean((us-us_hll)**2,axis=(0,1)))
        mae_u=np.mean(np.abs(us-us_hll),axis=(0,1))
        rmse_v= np.sqrt(np.mean((vs-vs_hll)**2,axis=(0,1)))
        mae_v=np.mean(np.abs(vs-vs_hll),axis=(0,1))
        print(f"RMSE={rmse_h}, MAE={mae_h}")
        print(f"RMSE={rmse_u}, MAE={mae_u}")
        print(f"RMSE={rmse_v}, MAE={mae_v}")




    def plot_2_3_surface(xx,yy,times,hs,us,vs):
        title=["(a) t=0s","(b) t=0.4s","(c) t=0.8s"]
                        
        import matplotlib.gridspec as gridspec

        fig = plt.figure(figsize=(10, 12))
        gs = gridspec.GridSpec(3, 2) 
     
        ax1 = fig.add_subplot(gs[0, 0], projection='3d')
        ax2 = fig.add_subplot(gs[1, 0],projection='3d')
        ax3 = fig.add_subplot(gs[2, 0],projection='3d')
        
        ax4 = fig.add_subplot(gs[0, 1])
        ax5 = fig.add_subplot(gs[1, 1])
        ax6 = fig.add_subplot(gs[2, 1])
        axs=np.array([[ax1,ax4],[ax2,ax5],[ax3,ax6]])
        for i in range(0, times):
            # rstride,cstride=int(1/dx),int(1/dx)
            # axs[i,0].plot_surface(xx,yy,hs[:,:,i],edgecolor='royalblue', lw=0.5, rstride=rstride, cstride=cstride,alpha=0.3)
            axs[i,0].plot_surface(xx,yy,hs[:,:,i],cmap='Blues',alpha=1)
            axs[i,0].set_zlabel("h (m)",fontsize=11)
            axs[i,0].set_xlabel("x (m)",fontsize=11)
            axs[i,0].set_ylabel("y (m)",fontsize=11)

            a=0.5
            xx_v=xx[::int(a/dx),::int(a/dx)]
            yy_v=yy[::int(a/dx),::int(a/dx)]
            us_v=us[::int(a/dx),::int(a/dx),i]
            vs_v=vs[::int(a/dx),::int(a/dx),i]
            M_v = np.hypot(us_v, vs_v)
            index_x=np.logical_and(xx_v<7 , xx_v>-7)
            index_y=np.logical_and(yy_v<7 , yy_v>-7)
            index=np.logical_and(index_x,index_y)
  
            Q = axs[i,1].quiver(xx_v[index], yy_v[index],us_v[index],vs_v[index],
                                M_v[index],
                                units='x', pivot='mid')
            M_v=M_v[index]
            sm = plt.cm.ScalarMappable(cmap=plt.get_cmap('viridis'), norm=plt.Normalize(vmin=M_v.min(), vmax=M_v.max()))
            sm.set_array([])  # 只需要颜色映射，不需要实际的数据

            # 添加颜色条
            cbar = plt.colorbar(sm, ax=axs[i,1], orientation='vertical',fraction=0.1, shrink=0.8, pad=0.04)
            cbar.set_label('(m/s)',fontsize=11)

            # qk = axs[i,1].quiverkey(Q, 0.9, 0.9, 1, r'$1 \frac{m}{s}$', labelpos='E',
            #        coordinates='figure')
            axs[i,1].set_aspect('equal')
            axs[i,1].set_xlabel("x (m)",fontsize=11)
            axs[i,1].set_ylabel("y (m)",fontsize=11)

        # axs[-1,0].set_xlabel("x (m)")
        # axs[-1,1].set_xlabel("x (m)")
        # axs[-1,2].set_xlabel("x (m)")
        
        # axs[-1,0].set_ylabel("y (m)")
        # axs[-1,1].set_ylabel("y (m)")
        # axs[-1,2].set_ylabel("y (m)")
        
        axs[0,0].set_title(title[0],fontsize=12)
        axs[1,0].set_title(title[1],fontsize=12)
        axs[2,0].set_title(title[2],fontsize=12)
        
        fig.tight_layout()
        # plt.subplots_adjust(right=0.94,)
        # save figure
        plt.savefig(f"{save_path}/case_9_1.jpg",dpi=500)
        plt.show()       
    
    def plot_error_final_time(xx,yy,hs_error,us_error,vs_error):
        import matplotlib.pyplot as plt
        fig,axs=plt.subplots(1,2,figsize=(12,6))
        cs1 = axs[0].contourf(xx, yy, hs_error[:,:,1],cmap='Blues')
        cs2 = axs[1].contourf(xx, yy, hs_error[:,:,2],cmap='Blues')
        axs[0].set_aspect('equal')
        axs[0].set_xlabel("x (m)",fontsize=11)
        axs[0].set_ylabel("y (m)",fontsize=11)
        axs[0].set_title("Distance (t=0.4s)",fontsize=12)
        axs[1].set_aspect('equal')
        axs[1].set_xlabel("x (m)",fontsize=11)
        axs[1].set_ylabel("y (m)",fontsize=11)
        axs[1].set_title("Distance (t=0.8s)",fontsize=12)
        plt.colorbar(cs1, ax=axs[0],fraction=0.05,label='(m)',shrink=0.7,location='right')
        plt.colorbar(cs2, ax=axs[1],fraction=0.05,label='(m)',shrink=0.7,location='right')
        plt.subplots_adjust(wspace=0.4)

        #save
        plt.savefig(f"{save_path}/case_9_2.jpg",dpi=500)
        plt.show()
        

    plot_2_3_surface(xx[:,:,0],yy[:,:,0],times,hs,us,vs)
    # plot_3_3_surface(xx[:,:,0],yy[:,:,0],times,hs_hll,us_hll,vs_hll)

    plot_error_final_time(xx[:,:,0],yy[:,:,0],hs_error,us_error,vs_error) 

def plot_circulatr_dam_break_one_figure(save_path,net9,time_slice=[0,0.25,.5],save=True):
    """plot the solution of circular dam break"""
    times = len(time_slice)
    dx,num=0.04,500
    # dx,num=0.1,200
    x = np.linspace(bbox[0]+dx/2.,bbox[1]-dx/2.,num)   #dx=0.1 num=200 
    y=np.linspace(bbox[2]+dx/2.,bbox[3]-dx/2.,num)     #dx=0.1 num=200
    t=np.array(time_slice)
    xx,yy,tt=np.meshgrid(x,y,t)
    points=np.vstack((xx.flatten(),yy.flatten(),tt.flatten())).T
    torch_points=torch.tensor(points).float().to(device)

    U_pred=net9(torch_points)
    U_pred=U_pred.detach().cpu().numpy().reshape((*xx.shape,-1)) #e.g. shape=(100,100,3,3)
    hs,us,vs=U_pred[:,:,:,0],U_pred[:,:,:,1],U_pred[:,:,:,2]  #e.g. shape=(100,100,3)

    if True:
    #HLL solution
        hs_hll,us_hll,vs_hll,_x_hll,_y_hll=HLL_solution_for_circular_dam_break(x,time_slice)   #

        if ~np.isclose(x,_x_hll).all() and ~np.isclose(y,_y_hll).all():
            raise ValueError("x or y is not the same")
        
        
        hs_error=np.abs(hs-hs_hll)  
        us_error=np.abs(us-us_hll)
        vs_error=np.abs(vs-vs_hll)

        rmse_h= np.sqrt(np.mean((hs-hs_hll)**2,axis=(0,1)))
        mae_h=np.mean(np.abs(hs-hs_hll),axis=(0,1))
        rmse_u= np.sqrt(np.mean((us-us_hll)**2,axis=(0,1)))
        mae_u=np.mean(np.abs(us-us_hll),axis=(0,1))
        rmse_v= np.sqrt(np.mean((vs-vs_hll)**2,axis=(0,1)))
        mae_v=np.mean(np.abs(vs-vs_hll),axis=(0,1))
        print(f"RMSE={rmse_h}, MAE={mae_h}")
        print(f"RMSE={rmse_u}, MAE={mae_u}")
        print(f"RMSE={rmse_v}, MAE={mae_v}")




    def plot_surface(xx,yy,times,hs,us,vs):
        title=["(a) t=0s","(b) t=0.4s","(c) t=0.8s"]
                        
        import matplotlib.gridspec as gridspec

        fig = plt.figure(figsize=(10, 16))
        gs = gridspec.GridSpec(4, 2) 
     
        ax1 = fig.add_subplot(gs[0, 0], projection='3d')
        ax2 = fig.add_subplot(gs[1, 0],projection='3d')
        ax3 = fig.add_subplot(gs[2, 0],projection='3d')
        
        ax4 = fig.add_subplot(gs[0, 1])
        ax5 = fig.add_subplot(gs[1, 1])
        ax6 = fig.add_subplot(gs[2, 1])

        ax7 = fig.add_subplot(gs[3, 0])
        ax8 = fig.add_subplot(gs[3, 1])

        axs=np.array([[ax1,ax4],[ax2,ax5],[ax3,ax6],[ax7,ax8]])
        for i in range(0, times):
            # rstride,cstride=int(1/dx),int(1/dx)
            # axs[i,0].plot_surface(xx,yy,hs[:,:,i],edgecolor='royalblue', lw=0.5, rstride=rstride, cstride=cstride,alpha=0.3)
            axs[i,0].plot_surface(xx,yy,hs[:,:,i],cmap='Blues',alpha=1)
            axs[i,0].set_zlabel("h (m)",fontsize=11)
            axs[i,0].set_xlabel("x (m)",fontsize=11)
            axs[i,0].set_ylabel("y (m)",fontsize=11)

            a=0.5
            xx_v=xx[::int(a/dx),::int(a/dx)]
            yy_v=yy[::int(a/dx),::int(a/dx)]
            us_v=us[::int(a/dx),::int(a/dx),i]
            vs_v=vs[::int(a/dx),::int(a/dx),i]
            M_v = np.hypot(us_v, vs_v)
            index_x=np.logical_and(xx_v<7 , xx_v>-7)
            index_y=np.logical_and(yy_v<7 , yy_v>-7)
            index=np.logical_and(index_x,index_y)
  
            Q = axs[i,1].quiver(xx_v[index], yy_v[index],us_v[index],vs_v[index],
                                M_v[index],
                                units='x', pivot='mid')
            M_v=M_v[index]
            sm = plt.cm.ScalarMappable(cmap=plt.get_cmap('viridis'), norm=plt.Normalize(vmin=M_v.min(), vmax=M_v.max()))
            sm.set_array([])  

            cbar = plt.colorbar(sm, ax=axs[i,1], orientation='vertical',fraction=0.1, shrink=0.8, pad=0.04)
            cbar.set_label('(m/s)',fontsize=11)

            # qk = axs[i,1].quiverkey(Q, 0.9, 0.9, 1, r'$1 \frac{m}{s}$', labelpos='E',
            #        coordinates='figure')
            axs[i,1].set_aspect('equal')
            axs[i,1].set_xlabel("x (m)",fontsize=11)
            axs[i,1].set_ylabel("y (m)",fontsize=11)

    
        
        axs[0,0].set_title(title[0],fontsize=12)
        axs[1,0].set_title(title[1],fontsize=12)
        axs[2,0].set_title(title[2],fontsize=12)

        #for error
        cs1=ax7.contourf(xx, yy, hs_error[:,:,1],cmap='Blues')
        ax7.set_aspect('equal')
        ax7.set_xlabel("x (m)",fontsize=11)
        ax7.set_ylabel("y (m)",fontsize=11)
        ax7.set_title("(d) Error (t=0.4s)",fontsize=12)

        cs2=ax8.contourf(xx, yy, hs_error[:,:,2],cmap='Blues')
        ax8.set_aspect('equal')
        ax8.set_xlabel("x (m)",fontsize=11)
        ax8.set_ylabel("y (m)",fontsize=11)
        ax8.set_title("(e) Error (t=0.8s)",fontsize=12)

        plt.colorbar(cs1, ax=axs[3,0],fraction=0.05,label='(m)',shrink=1.0,location='right',pad=0.02)
        plt.colorbar(cs2, ax=axs[3,1],fraction=0.05,label='(m)',shrink=1.0,location='right',pad=0.02)
        
        fig.tight_layout()
        # plt.subplots_adjust(right=0.94,)
        # save figure
        plt.savefig(f"{save_path}/case_9_3.jpg",dpi=500)
        plt.show()       
    
        

    plot_surface(xx[:,:,0],yy[:,:,0],times,hs,us,vs)



def ES1_solution_for_tidal(x,time_slice=[0,0.25,0.5],bbox_tidal=[-2,2,-2,2]):
    """ """
    from ref.finite_volume_2d import FV_slover_Well_Balance,Tidal_Case
    dx=(bbox_tidal[1]-bbox_tidal[0])/(x.shape[0])  #  e.g. 0.2 
    report_interval=time_slice[1]-time_slice[0]    # the default is mutiples 
    tidal=Tidal_Case(bbox=[[bbox_tidal[0],bbox_tidal[1]],[bbox_tidal[2],bbox_tidal[3]]],
                   dx_dy=(dx,dx),  
                   t_end=time_slice[-1],
                   report_interval=report_interval
                   )
    tidal_FV=FV_slover_Well_Balance(**tidal(),device=device)
    X,Y,resutls,report_times=tidal_FV()
    #plot the solution
    # damB.plot_results(resutls,report_times)
    if time_slice[0]!=0:   
        resutls=resutls[1:]
    resutls=np.array(resutls)        #e.g. shape=(times,100,100,3)
    hs_hll= resutls[:,:,:,0]       #e.g. shape=(times,100,100)
    us_hll_= resutls[:,:,:,1]/hs_hll   #hu-->u
    vs_hll_= resutls[:,:,:,2]/hs_hll   #hv-->v
    hs_hll=hs_hll.transpose((1,2,0))   #e.g. shape=(100,100,times)
    us_hll=us_hll_.transpose((1,2,0))
    vs_hll=-vs_hll_.transpose((1,2,0))

    x,y=X[0,:],Y[:,0]
    return hs_hll,us_hll,vs_hll,x,y
    

def plot_case10(save_path,net10,time_slice=[0,0.25,.5],save=True):
    times = len(time_slice)
    bbox_tidal=[-2,2,-2,2] 
    dx,num=0.01,400
    x = np.linspace(bbox_tidal[0]+dx/2.,bbox_tidal[1]-dx/2.,num)   #dx=0.1 num=40 
    y=np.linspace(bbox_tidal[2]+dx/2.,bbox_tidal[3]-dx/2.,num)     #dx=0.1 num=40
    t=np.array(time_slice)
    xx,yy,tt=np.meshgrid(x,y,t)
    points=np.vstack((xx.flatten(),yy.flatten(),tt.flatten())).T
    torch_points=torch.tensor(points).float().to(device)
    U_pred=net10(torch_points)
    U_pred=U_pred.detach().cpu().numpy().reshape((*xx.shape,-1)) #e.g. shape=(100,100,3,3)
    hs,us,vs=U_pred[:,:,:,0],U_pred[:,:,:,1],U_pred[:,:,:,2]  #e.g. shape=(100,100,3)

    #es1 solution
    hs_es1,us_es1,vs_es1,x_es1,y_es1=ES1_solution_for_tidal(x,time_slice,bbox_tidal)   #
    if ~np.isclose(x,x_es1).all() and ~np.isclose(y,y_es1).all():
        raise ValueError("x or y is not the same")
    hs_error=np.abs(hs-hs_es1) 
    us_error=np.abs(us-us_es1)
    vs_error=np.abs(vs-vs_es1)
    #cal RMSE and MAE  
    rmse_h= np.sqrt(np.mean((hs-hs_es1)**2,axis=(0,1)))
    mae_h=np.mean(np.abs(hs-hs_es1),axis=(0,1))
    rmse_u= np.sqrt(np.mean((us-us_es1)**2,axis=(0,1)))
    mae_u=np.mean(np.abs(us-us_es1),axis=(0,1))
    rmse_v= np.sqrt(np.mean((vs-vs_es1)**2,axis=(0,1)))
    mae_v=np.mean(np.abs(vs-vs_es1),axis=(0,1))
    print(f"RMSE={rmse_h}, MAE={mae_h}")
    print(f"RMSE={rmse_u}, MAE={mae_u}")
    print(f"RMSE={rmse_v}, MAE={mae_v}")
    

    def set_colorbar_incase10(ax,cmap:str,value,label:str,sci=False,**kwargs):
        sm = plt.cm.ScalarMappable(cmap=plt.get_cmap(cmap), norm=plt.Normalize(vmin=value.min(), vmax=value.max()))
        sm.set_array([])  
        cbar = plt.colorbar(sm, ax=ax, orientation='vertical',fraction=0.098, shrink=0.95, pad=0.04)
        cbar.set_label(label,fontsize=11)
        
        from matplotlib.ticker import ScalarFormatter
        cbar.formatter = ScalarFormatter(useMathText=True)
        cbar.formatter.set_scientific(True)
        cbar.formatter.set_powerlimits((0, 0))  
        cbar.update_ticks()                   

    def plot_3_3_surface(xx,yy,times,hs,us,vs):
        title=["(a) t=0s","(b) t=0.25s","(c) t=0.5s"]
                        
        import matplotlib.gridspec as gridspec

        fig = plt.figure(figsize=(12, 10.2))
        gs = gridspec.GridSpec(3, 3) 

        ax1 = fig.add_subplot(gs[0, 0], projection='3d')
        ax2 = fig.add_subplot(gs[0, 1],projection='3d')
        ax3 = fig.add_subplot(gs[0, 2],projection='3d')
        ax4 = fig.add_subplot(gs[1, 0])
        ax5 = fig.add_subplot(gs[1, 1])
        ax6 = fig.add_subplot(gs[1, 2])
        ax7=  fig.add_subplot(gs[2, 0])
        ax8=  fig.add_subplot(gs[2, 1])
        ax9=  fig.add_subplot(gs[2, 2])
        axs=np.array([[ax1,ax2,ax3],[ax4,ax5,ax6],[ax7,ax8,ax9]])

        for i in range(0, times):  
            # rstride,cstride=int(1/dx),int(1/dx)
            # axs[i,0].plot_surface(xx,yy,hs[:,:,i],edgecolor='royalblue', lw=0.5, rstride=rstride, cstride=cstride,alpha=0.3)
            axs[0,i].plot_surface(xx,yy,hs[:,:,i],cmap='coolwarm')
            axs[0,i].set_zlabel("h (m)",fontsize=11)
            axs[0,i].set_xlabel("x (m)",fontsize=11)
            axs[0,i].set_ylabel("y (m)",fontsize=11)

            axs[1,i].contourf(xx,yy,us[:,:,i],cmap='coolwarm',levels=15)
            axs[1,i].set_aspect('equal')
            axs[1,i].set_xlabel("x (m)",fontsize=11)
            axs[1,i].set_ylabel("y (m)",fontsize=11)
            set_colorbar_incase10(axs[1,i],'coolwarm',us[:,:,i],'u (m/s)',sci=True)

            axs[2,i].contourf(xx,yy,vs[:,:,i],cmap='coolwarm',levels=15)
            axs[2,i].set_aspect('equal')
            axs[2,i].set_xlabel("x (m)",fontsize=11)
            axs[2,i].set_ylabel("y (m)",fontsize=11)  
            set_colorbar_incase10(axs[2,i],'coolwarm',vs[:,:,i],'v (m/s)',sci=True)
        
        axs[0,0].set_title(title[0],fontsize=12)
        axs[0,1].set_title(title[1],fontsize=12)
        axs[0,2].set_title(title[2],fontsize=12)
        
        fig.tight_layout()
        # plt.subplots_adjust(hspace=,)
        # save figure
        if save:
            plt.savefig(f"{save_path}/case_10_1.jpg",dpi=500)
        plt.show()       
    
    
    def plot_error_final_time(xx,yy,hs_error,us_error,vs_error):
        import matplotlib.pyplot as plt
        fig,axs=plt.subplots(1,2,figsize=(12,6))
        cs1 = axs[0].contourf(xx, yy, hs_error[:,:,1],cmap='Blues')
        cs2 = axs[1].contourf(xx, yy, hs_error[:,:,2],cmap='Blues')
        axs[0].set_aspect('equal')
        axs[0].set_xlabel("x (m)",fontsize=11)
        axs[0].set_ylabel("y (m)",fontsize=11)
        axs[0].set_title("Error (t=0.25s)",fontsize=12)
        axs[1].set_aspect('equal')
        axs[1].set_xlabel("x (m)",fontsize=11)
        axs[1].set_ylabel("y (m)",fontsize=11)
        axs[1].set_title("Error (t=0.5s)",fontsize=12)
        cbar1=plt.colorbar(cs1, ax=axs[0],fraction=0.05,label='(m)',shrink=0.75,location='right')
        cbar2=plt.colorbar(cs2, ax=axs[1],fraction=0.05,label='(m)',shrink=0.75,location='right')
        plt.subplots_adjust(wspace=0.4)

        from matplotlib.ticker import ScalarFormatter
        cbar1.formatter = ScalarFormatter(useMathText=True)
        cbar1.formatter.set_scientific(True)
        cbar1.formatter.set_powerlimits((0, 0))  
        cbar1.update_ticks()    
        cbar2.formatter = ScalarFormatter(useMathText=True)
        cbar2.formatter.set_scientific(True)
        cbar2.formatter.set_powerlimits((0, 0))  
        cbar2.update_ticks()    
        

        #save
        if save:
            plt.savefig(f"{save_path}/case_10_2.jpg",dpi=500)
        plt.show()
    
    plot_3_3_surface(xx[:,:,0],yy[:,:,0],times,hs,us,vs)
    
    plot_error_final_time(xx[:,:,0],yy[:,:,0],hs_error,us_error,vs_error) #这个

def plot_cases10_in_one_figure(save_path,net10,time_slice=[0,0.25,.5],save=True):
    times = len(time_slice)
    bbox_tidal=[-2,2,-2,2] 
    dx,num=0.01,400
    x = np.linspace(bbox_tidal[0]+dx/2.,bbox_tidal[1]-dx/2.,num)   #dx=0.1 num=40 
    y=np.linspace(bbox_tidal[2]+dx/2.,bbox_tidal[3]-dx/2.,num)     #dx=0.1 num=40
    t=np.array(time_slice)
    xx,yy,tt=np.meshgrid(x,y,t)
    points=np.vstack((xx.flatten(),yy.flatten(),tt.flatten())).T
    torch_points=torch.tensor(points).float().to(device)
    U_pred=net10(torch_points)
    U_pred=U_pred.detach().cpu().numpy().reshape((*xx.shape,-1)) #e.g. shape=(100,100,3,3)
    hs,us,vs=U_pred[:,:,:,0],U_pred[:,:,:,1],U_pred[:,:,:,2]  #e.g. shape=(100,100,3)

    #es1 solution
    hs_es1,us_es1,vs_es1,x_es1,y_es1=ES1_solution_for_tidal(x,time_slice,bbox_tidal)   #
    if ~np.isclose(x,x_es1).all() and ~np.isclose(y,y_es1).all():
        raise ValueError("x or y is not the same")
    hs_error=np.abs(hs-hs_es1) 
    us_error=np.abs(us-us_es1)
    vs_error=np.abs(vs-vs_es1)
    #cal RMSE and MAE  
    rmse_h= np.sqrt(np.mean((hs-hs_es1)**2,axis=(0,1)))
    mae_h=np.mean(np.abs(hs-hs_es1),axis=(0,1))
    rmse_u= np.sqrt(np.mean((us-us_es1)**2,axis=(0,1)))
    mae_u=np.mean(np.abs(us-us_es1),axis=(0,1))
    rmse_v= np.sqrt(np.mean((vs-vs_es1)**2,axis=(0,1)))
    mae_v=np.mean(np.abs(vs-vs_es1),axis=(0,1))
    print(f"RMSE={rmse_h}, MAE={mae_h}")
    print(f"RMSE={rmse_u}, MAE={mae_u}")
    print(f"RMSE={rmse_v}, MAE={mae_v}")
    

    def set_colorbar_incase10(ax,cmap:str,value,label:str,sci=False,**kwargs):
        sm = plt.cm.ScalarMappable(cmap=plt.get_cmap(cmap), norm=plt.Normalize(vmin=value.min(), vmax=value.max()))
        sm.set_array([])  
        cbar = plt.colorbar(sm, ax=ax, orientation='vertical',fraction=0.20, shrink=1.0, pad=0.04)
        cbar.set_label(label,fontsize=11)
        
        from matplotlib.ticker import ScalarFormatter
        cbar.formatter = ScalarFormatter(useMathText=True)
        cbar.formatter.set_scientific(True)
        cbar.formatter.set_powerlimits((0, 0))  
        cbar.update_ticks()                   

    def plot_4_3_surface(xx,yy,times,hs,us,vs,hs_error):
        title=["(a) t=0s","(b) t=0.25s","(c) t=0.5s"]
                        
        import matplotlib.gridspec as gridspec

        fig = plt.figure(figsize=(12, 12.4))
        gs = gridspec.GridSpec(4, 3) 

        ax1 = fig.add_subplot(gs[0, 0], projection='3d')
        ax2 = fig.add_subplot(gs[0, 1],projection='3d')
        ax3 = fig.add_subplot(gs[0, 2],projection='3d')
        ax4 = fig.add_subplot(gs[1, 0])
        ax5 = fig.add_subplot(gs[1, 1])
        ax6 = fig.add_subplot(gs[1, 2])
        ax7=  fig.add_subplot(gs[2, 0])
        ax8=  fig.add_subplot(gs[2, 1])
        ax9=  fig.add_subplot(gs[2, 2])
        ax10= fig.add_subplot(gs[3, 0])
        ax11= fig.add_subplot(gs[3, 1])
        ax12= fig.add_subplot(gs[3, 2])
        axs=np.array([[ax1,ax2,ax3],[ax4,ax5,ax6],[ax7,ax8,ax9],[ax10,ax11,ax12]])

        for i in range(0, times):  
            # rstride,cstride=int(1/dx),int(1/dx)
            # axs[i,0].plot_surface(xx,yy,hs[:,:,i],edgecolor='royalblue', lw=0.5, rstride=rstride, cstride=cstride,alpha=0.3)
            axs[0,i].plot_surface(xx,yy,hs[:,:,i],cmap='coolwarm')
            axs[0,i].set_zlabel("h (m)",fontsize=11)
            axs[0,i].set_xlabel("x (m)",fontsize=11)
            axs[0,i].set_ylabel("y (m)",fontsize=11)

            axs[1,i].contourf(xx,yy,us[:,:,i],cmap='coolwarm',levels=15)
            axs[1,i].set_aspect('equal')
            axs[1,i].set_xlabel("x (m)",fontsize=11)
            axs[1,i].set_ylabel("y (m)",fontsize=11)
            set_colorbar_incase10(axs[1,i],'coolwarm',us[:,:,i],'u (m/s)',sci=True)

            axs[2,i].contourf(xx,yy,vs[:,:,i],cmap='coolwarm',levels=15)
            axs[2,i].set_aspect('equal')
            axs[2,i].set_xlabel("x (m)",fontsize=11)
            axs[2,i].set_ylabel("y (m)",fontsize=11)  
            set_colorbar_incase10(axs[2,i],'coolwarm',vs[:,:,i],'v (m/s)',sci=True)
        
        axs[0,0].set_title(title[0],fontsize=12)
        axs[0,1].set_title(title[1],fontsize=12)
        axs[0,2].set_title(title[2],fontsize=12)


        #for error
        axs[3,0].contourf(xx, yy, hs_error[:,:,0],cmap='Blues')
        axs[3,1].contourf(xx, yy, hs_error[:,:,1],cmap='Blues')
        axs[3,2].contourf(xx, yy, hs_error[:,:,2],cmap='Blues')
        axs[3,0].set_aspect('equal')
        axs[3,0].set_xlabel("x (m)",fontsize=11)
        axs[3,0].set_ylabel("y (m)",fontsize=11)
        set_colorbar_incase10(axs[3,0],'Blues',hs_error[:,:,0],'error(m)',sci=True)
        axs[3,1].set_aspect('equal')
        axs[3,1].set_xlabel("x (m)",fontsize=11)
        axs[3,1].set_ylabel("y (m)",fontsize=11)
        set_colorbar_incase10(axs[3,1],'Blues',hs_error[:,:,1],'error(m)',sci=True)
        axs[3,2].set_aspect('equal')
        axs[3,2].set_xlabel("x (m)",fontsize=11)
        axs[3,2].set_ylabel("y (m)",fontsize=11)
        set_colorbar_incase10(axs[3,2],'Blues',hs_error[:,:,2],'error(m)',sci=True)
                
        fig.tight_layout()
        # plt.subplots_adjust(hspace=,)
        # save figure
        if save:
            plt.savefig(f"{save_path}/case_10_one_figure.jpg",dpi=500)
        plt.show()       

    plot_4_3_surface(xx[:,:,0],yy[:,:,0],times,hs,us,vs,hs_error)  


def plot_case10_with_water_level(save_path,name,net10,time_slice=[0,0.25,.5]):
        times = len(time_slice)
        bbox_tidal=[-2,2,-2,2] 
        dx,num=0.1,40
        x = np.linspace(bbox_tidal[0]+dx/2.,bbox_tidal[1]-dx/2.,num)   #dx=0.1 num=40 
        y=np.linspace(bbox_tidal[2]+dx/2.,bbox_tidal[3]-dx/2.,num)     #dx=0.1 num=40
        t=np.array(time_slice)
        xx,yy,tt=np.meshgrid(x,y,t)
        points=np.vstack((xx.flatten(),yy.flatten(),tt.flatten())).T
        torch_points=torch.tensor(points).float().to(device)
        U_pred=net10(torch_points)
        U_pred=U_pred.detach().cpu().numpy().reshape((*xx.shape,-1)) #e.g. shape=(100,100,3,3)
        hs,us,vs=U_pred[:,:,:,0],U_pred[:,:,:,1],U_pred[:,:,:,2]  #e.g. shape=(100,100,3)
        
        zz=1.0+0.01*np.cos(np.pi/2.*xx[:,:,0])*np.cos(np.pi/2.*yy[:,:,0])  #m

        def set_colorbar_incase10(ax,cmap:str,value,label:str,sci=False,**kwargs):
            sm = plt.cm.ScalarMappable(cmap=plt.get_cmap(cmap), norm=plt.Normalize(vmin=value.min(), vmax=value.max()))
            sm.set_array([])  
            cbar = plt.colorbar(sm, ax=ax, orientation='vertical',fraction=0.1, shrink=0.99, pad=0.04)
            cbar.set_label(label,fontsize=11)
            
            from matplotlib.ticker import ScalarFormatter
            cbar.formatter = ScalarFormatter(useMathText=True)
            cbar.formatter.set_scientific(True)
            cbar.formatter.set_powerlimits((0, 0))  
            cbar.update_ticks()
        
        def plot_3_3_surface(xx,yy,zz,times,hs,us,vs):
            title=["(a) t=0s","(b) t=0.25s","(c) t=0.5s"]
                            
            import matplotlib.gridspec as gridspec

            fig = plt.figure(figsize=(12, 12))
            gs = gridspec.GridSpec(4, 3) 

            ax1 = fig.add_subplot(gs[0, 0], projection='3d')
            ax2 = fig.add_subplot(gs[0, 1],projection='3d')
            ax3 = fig.add_subplot(gs[0, 2],projection='3d')
            ax4 = fig.add_subplot(gs[1, 0], projection='3d')
            ax5 = fig.add_subplot(gs[1, 1],projection='3d')
            ax6 = fig.add_subplot(gs[1, 2],projection='3d')
            ax7 = fig.add_subplot(gs[2, 0])
            ax8 = fig.add_subplot(gs[2, 1])
            ax9 = fig.add_subplot(gs[2, 2])
            ax10=  fig.add_subplot(gs[3, 0])
            ax11=  fig.add_subplot(gs[3, 1])
            ax12=  fig.add_subplot(gs[3, 2])
            axs=np.array([[ax1,ax2,ax3],[ax4,ax5,ax6],[ax7,ax8,ax9],[ax10,ax11,ax12]])

            for i in range(0, times):  
                # rstride,cstride=int(1/dx),int(1/dx)
                # axs[i,0].plot_surface(xx,yy,hs[:,:,i],edgecolor='royalblue', lw=0.5, rstride=rstride, cstride=cstride,alpha=0.3)
                axs[0,i].plot_surface(xx,yy,hs[:,:,i],cmap='coolwarm')
                axs[0,i].set_zlabel("h (m)",fontsize=11)
                axs[0,i].set_xlabel("x (m)",fontsize=11)
                axs[0,i].set_ylabel("y (m)",fontsize=11)

                axs[1,i].plot_surface(xx,yy,hs[:,:,i]+zz,cmap='coolwarm')
                axs[1,i].set_zlabel("h+z (m)",fontsize=11)
                axs[1,i].set_xlabel("x (m)",fontsize=11)
                axs[1,i].set_ylabel("y (m)",fontsize=11)


                axs[2,i].contourf(xx,yy,us[:,:,i],cmap='coolwarm',levels=15)
                axs[2,i].set_aspect('equal')
                axs[2,i].set_xlabel("x (m)",fontsize=11)
                axs[2,i].set_ylabel("y (m)",fontsize=11)
                set_colorbar_incase10(axs[2,i],'coolwarm',us[:,:,i],'u (m/s)',sci=True)

                axs[3,i].contourf(xx,yy,vs[:,:,i],cmap='coolwarm',levels=15)
                axs[3,i].set_aspect('equal')
                axs[3,i].set_xlabel("x (m)",fontsize=11)
                axs[3,i].set_ylabel("y (m)",fontsize=11)  
                set_colorbar_incase10(axs[3,i],'coolwarm',vs[:,:,i],'v (m/s)',sci=True)
            
            axs[0,0].set_title(title[0],fontsize=12)
            axs[0,1].set_title(title[1],fontsize=12)
            axs[0,2].set_title(title[2],fontsize=12)
            
            fig.tight_layout()
            # plt.subplots_adjust(hspace=,)
            # save figure
            plt.savefig(f"{save_path}/{name}",dpi=500)
            plt.show()       
        plot_3_3_surface(xx[:,:,0],yy[:,:,0],zz,times,hs,us,vs)
    



def ES1_solution_for_tidal_rain(x,time_slice=[0,2.5,5],bbox_tidal=[-2,2,-2,2]):
    """ """
    from ref.finite_volume_2d import FV_slover_Well_Balance,Tidal_Case_Rian
    dx=(bbox_tidal[1]-bbox_tidal[0])/(x.shape[0])  #  e.g. 0.2 
    report_interval=time_slice[1]-time_slice[0]    # the default is mutiples 
    tidal=Tidal_Case_Rian(bbox=[[bbox_tidal[0],bbox_tidal[1]],[bbox_tidal[2],bbox_tidal[3]]],
                   dx_dy=(dx,dx),  
                   t_end=time_slice[-1],
                   report_interval=report_interval
                   )
    tidal_FV=FV_slover_Well_Balance(**tidal(),device=device)
    X,Y,resutls,report_times=tidal_FV()
    #plot the solution
    # damB.plot_results(resutls,report_times)
    if time_slice[0]!=0:   
        resutls=resutls[1:]
    resutls=np.array(resutls)        #e.g. shape=(times,100,100,3)
    hs_hll= resutls[:,:,:,0]       #e.g. shape=(times,100,100)
    us_hll_= resutls[:,:,:,1]/hs_hll   #hu-->u
    vs_hll_= resutls[:,:,:,2]/hs_hll   #hv-->v
    hs_hll=hs_hll.transpose((1,2,0))   #e.g. shape=(100,100,times)
    us_hll=us_hll_.transpose((1,2,0))
    vs_hll=-vs_hll_.transpose((1,2,0))

    x,y=X[0,:],Y[:,0]
    return hs_hll,us_hll,vs_hll,x,y
def plot_case11_with_water_level(save_path,name,net11,time_slice,save=True):
        times = len(time_slice)
        bbox_tidal=[-2,2,-2,2] 
        dx,num=0.01,400
        x = np.linspace(bbox_tidal[0]+dx/2.,bbox_tidal[1]-dx/2.,num)   #dx=0.1 num=40 
        y=np.linspace(bbox_tidal[2]+dx/2.,bbox_tidal[3]-dx/2.,num)     #dx=0.1 num=40
        t=np.array(time_slice)
        xx,yy,tt=np.meshgrid(x,y,t)
        points=np.vstack((xx.flatten(),yy.flatten(),tt.flatten())).T
        torch_points=torch.tensor(points).float().to(device)
        U_pred=net11(torch_points)
        U_pred=U_pred.detach().cpu().numpy().reshape((*xx.shape,-1)) #e.g. shape=(100,100,3,3)
        hs,us,vs=U_pred[:,:,:,0],U_pred[:,:,:,1],U_pred[:,:,:,2]  #e.g. shape=(100,100,3)
        
        zz=1.0+0.01*np.cos(np.pi/2.*xx[:,:,0])*np.cos(np.pi/2.*yy[:,:,0])  #m
        # zz=zz*100  #cm

        #es1 solution
        hs_es1,us_es1,vs_es1,x_es1,y_es1=ES1_solution_for_tidal_rain(x,time_slice,bbox_tidal)   #
        if ~np.isclose(x,x_es1).all() and ~np.isclose(y,y_es1).all():
            raise ValueError("x or y is not the same")
        # hs_es1=hs_es1*100  #cm
        # us_es1=us_es1*100  #cm
        # vs_es1=vs_es1*100  #cm
        hs_error=np.abs(hs-hs_es1) 
        us_error=np.abs(us-us_es1)
        vs_error=np.abs(vs-vs_es1)

        #cal RMSE and MAE  
        rmse_h= np.sqrt(np.mean((hs-hs_es1)**2,axis=(0,1)))
        mae_h=np.mean(np.abs(hs-hs_es1),axis=(0,1))
        rmse_u= np.sqrt(np.mean((us-us_es1)**2,axis=(0,1)))
        mae_u=np.mean(np.abs(us-us_es1),axis=(0,1))
        rmse_v= np.sqrt(np.mean((vs-vs_es1)**2,axis=(0,1)))
        mae_v=np.mean(np.abs(vs-vs_es1),axis=(0,1))
        print(f"RMSE={rmse_h}, MAE={mae_h}")
        print(f"RMSE={rmse_u}, MAE={mae_u}")
        print(f"RMSE={rmse_v}, MAE={mae_v}")
        
        def set_colorbar_incase10(ax,cmap:str,value,label:str,sci=False,**kwargs):
            sm = plt.cm.ScalarMappable(cmap=plt.get_cmap(cmap), norm=plt.Normalize(vmin=value.min(), vmax=value.max()))
            sm.set_array([])  
            cbar = plt.colorbar(sm, ax=ax, orientation='vertical',fraction=0.1, shrink=0.99, pad=0.04)
            cbar.set_label(label,fontsize=11)
            
            from matplotlib.ticker import ScalarFormatter
            cbar.formatter = ScalarFormatter(useMathText=True)
            cbar.formatter.set_scientific(True)
            cbar.formatter.set_powerlimits((0, 0))  
            cbar.update_ticks()
        
        def plot_3_3_surface(xx,yy,zz,times,hs,us,vs):
            title=["(a) t=0s","(b) t=2.5s","(c) t=5s"]
                            
            import matplotlib.gridspec as gridspec

            fig = plt.figure(figsize=(12, 12))
            gs = gridspec.GridSpec(4, 3) 

            ax1 = fig.add_subplot(gs[0, 0], projection='3d')
            ax2 = fig.add_subplot(gs[0, 1],projection='3d')
            ax3 = fig.add_subplot(gs[0, 2],projection='3d')
            ax4 = fig.add_subplot(gs[1, 0], projection='3d')
            ax5 = fig.add_subplot(gs[1, 1],projection='3d')
            ax6 = fig.add_subplot(gs[1, 2],projection='3d')
            ax7 = fig.add_subplot(gs[2, 0])
            ax8 = fig.add_subplot(gs[2, 1])
            ax9 = fig.add_subplot(gs[2, 2])
            ax10=  fig.add_subplot(gs[3, 0])
            ax11=  fig.add_subplot(gs[3, 1])
            ax12=  fig.add_subplot(gs[3, 2])
            axs=np.array([[ax1,ax2,ax3],[ax4,ax5,ax6],[ax7,ax8,ax9],[ax10,ax11,ax12]])

            for i in range(0, times):  
                # rstride,cstride=int(1/dx),int(1/dx)
                # axs[i,0].plot_surface(xx,yy,hs[:,:,i],edgecolor='royalblue', lw=0.5, rstride=rstride, cstride=cstride,alpha=0.3)
                axs[0,i].plot_surface(xx,yy,hs[:,:,i],cmap='coolwarm')
                axs[0,i].set_zlabel("h (m)",fontsize=11)
                axs[0,i].set_xlabel("x (m)",fontsize=11)
                axs[0,i].set_ylabel("y (m)",fontsize=11)

                axs[1,i].plot_surface(xx,yy,hs[:,:,i]+zz,cmap='coolwarm')
                axs[1,i].set_zlabel("h+z (m)",fontsize=11)
                axs[1,i].set_xlabel("x (m)",fontsize=11)
                axs[1,i].set_ylabel("y (m)",fontsize=11)


                axs[2,i].contourf(xx,yy,us[:,:,i],cmap='coolwarm',levels=15)
                axs[2,i].set_aspect('equal')
                axs[2,i].set_xlabel("x (m)",fontsize=11)
                axs[2,i].set_ylabel("y (m)",fontsize=11)
                set_colorbar_incase10(axs[2,i],'coolwarm',us[:,:,i],'u (m/s)',sci=True)

                axs[3,i].contourf(xx,yy,vs[:,:,i],cmap='coolwarm',levels=15)
                axs[3,i].set_aspect('equal')
                axs[3,i].set_xlabel("x (m)",fontsize=11)
                axs[3,i].set_ylabel("y (m)",fontsize=11)  
                set_colorbar_incase10(axs[3,i],'coolwarm',vs[:,:,i],'v (m/s)',sci=True)
            
            axs[0,0].set_title(title[0],fontsize=12)
            axs[0,1].set_title(title[1],fontsize=12)
            axs[0,2].set_title(title[2],fontsize=12)
            
            fig.tight_layout()
            # plt.subplots_adjust(hspace=,)
            # save figure
            plt.savefig(f"{save_path}/{name}",dpi=500)
            plt.show()       
        
        
        def plot_error_final_time(xx,yy,hs_error,us_error,vs_error):
            import matplotlib.pyplot as plt
            fig,axs=plt.subplots(1,2,figsize=(12,6))
            cs1 = axs[0].contourf(xx, yy, hs_error[:,:,1],cmap='Blues')
            cs2 = axs[1].contourf(xx, yy, hs_error[:,:,2],cmap='Blues')
            axs[0].set_aspect('equal')
            axs[0].set_xlabel("x (m)",fontsize=11)
            axs[0].set_ylabel("y (m)",fontsize=11)
            axs[0].set_title("Error (t=0.25s)",fontsize=12)
            axs[1].set_aspect('equal')
            axs[1].set_xlabel("x (m)",fontsize=11)
            axs[1].set_ylabel("y (m)",fontsize=11)
            axs[1].set_title("Error (t=0.5s)",fontsize=12)
            cbar1=plt.colorbar(cs1, ax=axs[0],fraction=0.05,label='(m)',shrink=0.75,location='right')
            cbar2=plt.colorbar(cs2, ax=axs[1],fraction=0.05,label='(m)',shrink=0.75,location='right')
            plt.subplots_adjust(wspace=0.4)
            

            from matplotlib.ticker import ScalarFormatter
            cbar1.formatter = ScalarFormatter(useMathText=True)
            cbar1.formatter.set_scientific(True)
            cbar1.formatter.set_powerlimits((0, 0))  
            cbar1.update_ticks()    
            cbar2.formatter = ScalarFormatter(useMathText=True)
            cbar2.formatter.set_scientific(True)
            cbar2.formatter.set_powerlimits((0, 0))  
            cbar2.update_ticks()    

            #save
            if save:
                plt.savefig(f"{save_path}/case_11_2.jpg",dpi=500)
            plt.show()
       
        plot_3_3_surface(xx[:,:,0],yy[:,:,0],zz,times,hs,us,vs)

        plot_error_final_time(xx[:,:,0],yy[:,:,0],hs_error,us_error,vs_error) #这个

def plot_case11_in_one_figure(save_path,net11,time_slice,save=True):
        times = len(time_slice)
        bbox_tidal=[-2,2,-2,2] 
        dx,num=0.01,400
        x = np.linspace(bbox_tidal[0]+dx/2.,bbox_tidal[1]-dx/2.,num)   #dx=0.1 num=40 
        y=np.linspace(bbox_tidal[2]+dx/2.,bbox_tidal[3]-dx/2.,num)     #dx=0.1 num=40
        t=np.array(time_slice)
        xx,yy,tt=np.meshgrid(x,y,t)
        points=np.vstack((xx.flatten(),yy.flatten(),tt.flatten())).T
        torch_points=torch.tensor(points).float().to(device)
        U_pred=net11(torch_points)
        U_pred=U_pred.detach().cpu().numpy().reshape((*xx.shape,-1)) #e.g. shape=(100,100,3,3)
        hs,us,vs=U_pred[:,:,:,0],U_pred[:,:,:,1],U_pred[:,:,:,2]  #e.g. shape=(100,100,3)
        
        zz=1.0+0.01*np.cos(np.pi/2.*xx[:,:,0])*np.cos(np.pi/2.*yy[:,:,0])  #m
        # zz=zz*100  #cm

        #es1 solution
        hs_es1,us_es1,vs_es1,x_es1,y_es1=ES1_solution_for_tidal_rain(x,time_slice,bbox_tidal)   #
        if ~np.isclose(x,x_es1).all() and ~np.isclose(y,y_es1).all():
            raise ValueError("x or y is not the same")
        # hs_es1=hs_es1*100  #cm
        # us_es1=us_es1*100  #cm
        # vs_es1=vs_es1*100  #cm
        hs_error=np.abs(hs-hs_es1) 
        us_error=np.abs(us-us_es1)
        vs_error=np.abs(vs-vs_es1)

        #cal RMSE and MAE  
        rmse_h= np.sqrt(np.mean((hs-hs_es1)**2,axis=(0,1)))
        mae_h=np.mean(np.abs(hs-hs_es1),axis=(0,1))
        rmse_u= np.sqrt(np.mean((us-us_es1)**2,axis=(0,1)))
        mae_u=np.mean(np.abs(us-us_es1),axis=(0,1))
        rmse_v= np.sqrt(np.mean((vs-vs_es1)**2,axis=(0,1)))
        mae_v=np.mean(np.abs(vs-vs_es1),axis=(0,1))
        print(f"RMSE={rmse_h}, MAE={mae_h}")
        print(f"RMSE={rmse_u}, MAE={mae_u}")
        print(f"RMSE={rmse_v}, MAE={mae_v}")
        
        def set_colorbar_incase10(ax,cmap:str,value,label:str,sci=False,**kwargs):
            sm = plt.cm.ScalarMappable(cmap=plt.get_cmap(cmap), norm=plt.Normalize(vmin=value.min(), vmax=value.max()))
            sm.set_array([])  
            cbar = plt.colorbar(sm, ax=ax, orientation='vertical',fraction=0.1, shrink=0.99, pad=0.04)
            cbar.set_label(label,fontsize=11)
            
            from matplotlib.ticker import ScalarFormatter
            cbar.formatter = ScalarFormatter(useMathText=True)
            cbar.formatter.set_scientific(True)
            cbar.formatter.set_powerlimits((0, 0))  
            cbar.update_ticks()
        
        def plot_5_3_surface(xx,yy,zz,times,hs,us,vs):
            title=["(a) t=0s","(b) t=2.5s","(c) t=5s"]
                            
            import matplotlib.gridspec as gridspec

            fig = plt.figure(figsize=(12, 15))
            gs = gridspec.GridSpec(5, 3) 

            ax1 = fig.add_subplot(gs[0, 0], projection='3d')
            ax2 = fig.add_subplot(gs[0, 1],projection='3d')
            ax3 = fig.add_subplot(gs[0, 2],projection='3d')
            ax4 = fig.add_subplot(gs[1, 0], projection='3d')
            ax5 = fig.add_subplot(gs[1, 1],projection='3d')
            ax6 = fig.add_subplot(gs[1, 2],projection='3d')
            ax7 = fig.add_subplot(gs[2, 0])
            ax8 = fig.add_subplot(gs[2, 1])
            ax9 = fig.add_subplot(gs[2, 2])
            ax10=  fig.add_subplot(gs[3, 0])
            ax11=  fig.add_subplot(gs[3, 1])
            ax12=  fig.add_subplot(gs[3, 2])
            ax13=  fig.add_subplot(gs[4, 0])
            ax14=  fig.add_subplot(gs[4, 1])
            ax15=  fig.add_subplot(gs[4, 2])
            axs=np.array([[ax1,ax2,ax3],[ax4,ax5,ax6],[ax7,ax8,ax9],[ax10,ax11,ax12],[ax13,ax14,ax15]])

            for i in range(0, times):  
                # rstride,cstride=int(1/dx),int(1/dx)
                # axs[i,0].plot_surface(xx,yy,hs[:,:,i],edgecolor='royalblue', lw=0.5, rstride=rstride, cstride=cstride,alpha=0.3)
                axs[0,i].plot_surface(xx,yy,hs[:,:,i],cmap='coolwarm')
                axs[0,i].set_zlabel("h (m)",fontsize=11)
                axs[0,i].set_xlabel("x (m)",fontsize=11)
                axs[0,i].set_ylabel("y (m)",fontsize=11)

                axs[1,i].plot_surface(xx,yy,hs[:,:,i]+zz,cmap='coolwarm')
                axs[1,i].set_zlabel("h+z (m)",fontsize=11)
                axs[1,i].set_xlabel("x (m)",fontsize=11)
                axs[1,i].set_ylabel("y (m)",fontsize=11)


                axs[2,i].contourf(xx,yy,us[:,:,i],cmap='coolwarm',levels=15)
                axs[2,i].set_aspect('equal')
                axs[2,i].set_xlabel("x (m)",fontsize=11)
                axs[2,i].set_ylabel("y (m)",fontsize=11)
                set_colorbar_incase10(axs[2,i],'coolwarm',us[:,:,i],'u (m/s)',sci=True)

                axs[3,i].contourf(xx,yy,vs[:,:,i],cmap='coolwarm',levels=15)
                axs[3,i].set_aspect('equal')
                axs[3,i].set_xlabel("x (m)",fontsize=11)
                axs[3,i].set_ylabel("y (m)",fontsize=11)  
                set_colorbar_incase10(axs[3,i],'coolwarm',vs[:,:,i],'v (m/s)',sci=True)
            
                
            axs[0,0].set_title(title[0],fontsize=12)
            axs[0,1].set_title(title[1],fontsize=12)
            axs[0,2].set_title(title[2],fontsize=12)

            axs[4,0].contourf(xx, yy, hs_error[:,:,0],cmap='Blues')
            axs[4,1].contourf(xx, yy, hs_error[:,:,1],cmap='Blues')
            axs[4,2].contourf(xx, yy, hs_error[:,:,2],cmap='Blues')
            axs[4,0].set_aspect('equal')
            axs[4,1].set_aspect('equal')
            axs[4,2].set_aspect('equal')
            axs[4,0].set_xlabel("x (m)",fontsize=11)
            axs[4,1].set_xlabel("x (m)",fontsize=11)
            axs[4,2].set_xlabel("x (m)",fontsize=11)
            axs[4,0].set_ylabel("y (m)",fontsize=11)
            axs[4,1].set_ylabel("y (m)",fontsize=11)
            axs[4,2].set_ylabel("y (m)",fontsize=11)
            set_colorbar_incase10(axs[4,0],'Blues',hs_error[:,:,0],'error(m)',sci=True)
            set_colorbar_incase10(axs[4,1],'Blues',hs_error[:,:,1],'error(m)',sci=True)
            set_colorbar_incase10(axs[4,2],'Blues',hs_error[:,:,2],'error(m)',sci=True)
            
            fig.tight_layout()
            # plt.subplots_adjust(hspace=,)
            # save figure
            name="case_11_3.jpg"
            plt.savefig(f"{save_path}/{name}",dpi=500)
            plt.show()       
        
        
        plot_5_3_surface(xx[:,:,0],yy[:,:,0],zz,times,hs,us,vs)
        return


def intgrate_rain(net):
    #the run time is very slow, be patient
    from scipy.integrate import dblquad
    def h_func(x,y,t):
        points=np.vstack((x,y,t)).T
        torch_points=torch.tensor(points).float().to(device)
        U_pred=net(torch_points)
        U_pred=U_pred.detach().cpu().numpy().reshape((-1)) #e.g. shape=(100,100,3,3)
        return U_pred[0]
    
    def init_h(x,y,t):
        h=1.0+0.01*np.cos(np.pi/2.*x)*np.cos(np.pi/2.*y)
        return h
    
    def water_volume(t):
        x_min = -2
        x_max = 2
        y_min = lambda x: -2
        y_max = lambda x: 2
        result, error = dblquad(h_func, x_min, x_max, y_min, y_max,args=(t,))

        return result
    t=np.array([0,0.25,0.5])  
    volumes=np.array([water_volume(t[i]) for i in range(len(t))])
    print(f"volumes={volumes}")

        



if __name__ == "__main__":

    # Load model        
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    save_path="plot_in_paper/paper_cases"

    if False:
        D_TIDAL_RAIN_pt="plot_in_paper/paper_cases/dTR_120000.pt" #Dynamic Tidal case

        net11=load_model(D_TIDAL_RAIN_pt,device,layers=6)
        if False: 
            plot_case11_with_water_level(save_path,"case_11_water_level.jpg",net11,time_slice=[0,.25,.5])
        if False:
            intgrate_rain(net11)
        if True:
            plot_case11_in_one_figure(save_path,net11,time_slice=[0,.25,.5],save=True)



    if False:   # plot the case 10
        D_TIDAL_pt="plot_in_paper/paper_cases/dT10_100000.pt" #Dynamic Tidal case

        net10=load_model(D_TIDAL_pt,device,layers=6)
        if True: 
            plot_case10(save_path,net10,time_slice=[0,0.25,0.5],save=True)
        # plot_case10_with_water_level(save_path,"case_10_water_level.jpg",net10,time_slice=[0,0.25,0.5])

        if False:  
            plot_cases10_in_one_figure(save_path,net10,time_slice=[0,0.25,0.5],save=True)
    
    
    if  True:   # plot the case 9
        # CDB_pt='plot_in_paper/paper_cases/circular_dam_break_30000.pt'   #circular dam break
        CDB_pt='plot_in_paper/paper_cases/dCDB_5000.pt'   #circular dam break
        # CAD_pt='plot_in_paper/CDB_3q0000.pt'   #circular dam
        
        net9=load_model(CDB_pt,device,layers=5,method='laaf')
        # plot_circulatr_dam_break(save_path,net9,time_slice=[0.,0.4,0.8])  
        plot_circulatr_dam_break_one_figure(save_path,net9,time_slice=[0.,0.4,0.8],save=True)  

    if False:   # plot the case 8 
        DB_pri_pt='plot_in_paper/paper_cases/dam_break_primitive_var_5_30000.pt'  #dam break with primary flow
        DB_entropy_pt='plot_in_paper/paper_cases/dam_break_entropy_5_30000.pt'  #dam break with entropy
        DB_var_conser_pt='plot_in_paper/paper_cases/dam_break_var_con_5_30000.pt'  #dam break with variable conservation

        net8_1 = load_model(DB_pri_pt,device,layers=5)
        net8_2 = load_model(DB_entropy_pt,device,layers=5)
        net8_3 = load_model(DB_var_conser_pt,device,layers=5)

        # plot_8_case_huv(save_path,net8_3,time_slice=[0,0.6,1.2],save=False)
        plot_8_case_muti(save_path,net8_1,net8_2,net8_3,time_slice=[0,0.6,1.2],save=True)
        
    if False:      #plot the cases of 1-3
        #without entropy
        B_pt='plot_in_paper/paper_cases/sB1_50000.pt'
        D_pt='plot_in_paper/paper_cases/sD2_50000.pt'
        T_pt='plot_in_paper/paper_cases/sT3_50000.pt'
        
        #with entropy
        B_en_pt='plot_in_paper/paper_cases/sB1_entropy_50000.pt'
        D_en_pt='plot_in_paper/paper_cases/sD2_entropy_50000.pt'
        T_en_pt='plot_in_paper/paper_cases/sT3_entropy_50000.pt'
        net1 = load_model(B_pt,device,layers=6)
        net2 = load_model(D_pt,device,layers=6)
        net3 = load_model(T_pt,device,layers=6)
        net1_en=load_model(B_en_pt,device,layers=6)
        net2_en=load_model(D_en_pt,device,layers=6)
        net3_en=load_model(T_en_pt,device,layers=6)
        #with entropy
        # if True: plot_1_3_cases(save_path,net1_en,net2_en,net3_en,time_slice=5.,save=True)
        
        #without entropy
        if True: plot_1_3_cases(save_path,net1,net2,net3,time_slice=5.)
        
        test_error_1_3_cases(net1,net2,net3,time_slice=[0,5])
        test_error_1_3_cases(net1_en,net2_en,net3_en,time_slice=[0,5])

        #just fot the static tidal cases
        plot_static_tidal_cases(save_path,net3_en,net3,time_slice=5.,save=True)



    if False:      #plot the cases of 4-7
        FR_pt='plot_in_paper/paper_cases/sFR4_80000.pt'
        BR_pt='plot_in_paper/paper_cases/sBR5_80000.pt'
        DR_pt='plot_in_paper/paper_cases/sDR6_80000.pt'
        TR_pt='plot_in_paper/paper_cases/sTR7_80000.pt'

        net4 = load_model(FR_pt,device,layers=5)   #change to your model
        net5 = load_model(BR_pt,device,layers=5)
        net6 = load_model(DR_pt,device,layers=5)
        net7 = load_model(TR_pt,device,layers=5)
        #plot the figures
        if True: plot_4_7_cases_rain(save_path,net4,net5,net6,net7)
        #plot the curves
        if False: plot_4_7_cases_slices(save_path,net4,net5,net6,net7)
        #print the errors
        if False: test_error_4_7_cases(net4,net5,net6,net7,time_slice=[0,5])








    

    
    
    
    
