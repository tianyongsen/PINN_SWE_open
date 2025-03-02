import numpy as np
import torch
import time
import matplotlib.pyplot as plt




class FV_slover():
    """HLL Scheme finite volume solver for 2D shallow water equations 
        without bottom topography and frcition term ."""
    def __init__(self,
                 bbox=[[0,1],[0,1]],
                 dx_dy=(0.1,0.1),
                 t_end=10.,
                 report_interval=1.,  
                 init_func=None, #init_func(X,Y,U)
                 z_func=None,  #z_func(X,Y)
                 rain_func=None,  #rain_func(X,Y,t)
                 bcs:dir=None,
                 cfl=0.25,     #strict CFL condition
                 device='cpu'
                 ):
        self._bbox=bbox    #
        self._dx=dx_dy[0]
        self._dy=dx_dy[1]

        self._t_end=t_end
        self._report_times=[report_interval*i for i in range(int(t_end/report_interval)+1)]  #the demand timestamps for report
        if t_end not in self._report_times:
            self._report_times.append(t_end)  #add the last report time

        self._init_func=init_func
        self._bcs=bcs
        self._z_func=z_func
        self._rain_func=rain_func
        
        self._mesh_created=False

        self._device=device  #cpu or cuda

        self._results=[]    #record the results

        self._cfl=cfl
        self.g=9.81
        
    def __call__(self):
        #step1: create mesh
        self.__creat_mesh()

        #step2: init U
        self.__init_U()
        
        #step3: iteration to solve the problem
        i,t=0,0
        t_end=torch.tensor(self._t_end,dtype=torch.float32,device=self._device,requires_grad=False)
        t=torch.tensor(t,dtype=torch.float32,device=self._device,requires_grad=False)
        report_times=torch.tensor(self._report_times,dtype=torch.float32,device=self._device,requires_grad=False)
        while(t<t_end):
            if t>=report_times[i]:   #report 0
                self._report_times[i]=t.item()
                self._results.append(self.U.detach().cpu().numpy())
                i+=1
                print(f"current time is {t}")
            with torch.no_grad():
                dt=self.__step(t)
            t+=dt
        else:
            self._report_times[i]=t.item()
            self._results.append(self.U.detach().cpu().numpy())
            print(f"current time is {t}")
        
        return self._X, self._Y, self._results, self._report_times  #results=[numpy:shape=(m,n,3),numpy,numpy,......],Note: h，hu,hv

    def __creat_mesh(self):
        self._dx_num=int((self._bbox[0][1]-self._bbox[0][0])/self._dx)
        self._dy_num=int((self._bbox[1][1]-self._bbox[1][0])/self._dy)

        x = np.linspace(self._bbox[0][0]+self._dx/2.,self._bbox[0][1]-self._dx/2.,self._dx_num)
        y = np.linspace(self._bbox[1][0]+self._dy/2.,self._bbox[1][1]-+self._dx/2.,self._dy_num)
        self._X, self._Y = np.meshgrid(x,y)  #
        self._mesh_created=True
    
    def __init_U(self):
        #U=[h,hu,hv]^T
        if not self._mesh_created:
            raise ValueError("Before initializing U, please create the Mesh")
        if self._init_func is None:
            raise ValueError("Please provide an initial condition function")
        else:
            self.U=self._init_func(self._X,self._Y)
        #check the shape of U
        if self.U.shape!=(self._dy_num,self._dx_num,3):
            raise ValueError("The shape of U is not correct")
        U_shape=np.array(self.U.shape)
        self.U=torch.tensor(self.U,dtype=torch.float32,device=self._device,requires_grad=False)
        self.U_l=torch.zeros(tuple(U_shape+np.array([0,1,0])),dtype=torch.float32,device=self._device,requires_grad=False)
        self.U_r=torch.zeros(tuple(U_shape+np.array([0,1,0])),dtype=torch.float32,device=self._device,requires_grad=False)
        self.U_t=torch.zeros(tuple(U_shape+np.array([1,0,0])),dtype=torch.float32,device=self._device,requires_grad=False)
        self.U_b=torch.zeros(tuple(U_shape+np.array([1,0,0])),dtype=torch.float32,device=self._device,requires_grad=False)
        self._dx=torch.tensor(self._dx,dtype=torch.float32,device=self._device,requires_grad=False)
        self._dy=torch.tensor(self._dy,dtype=torch.float32,device=self._device,requires_grad=False)

        #initialize the bottom topography
        if self._z_func is not None:            
            z,z_x,z_y=self._z_func(self._X,self._Y)
            self.gz_x=torch.tensor(self.g*z_x,dtype=torch.float32,device=self._device,requires_grad=False)
            self.gz_y=torch.tensor(self.g*z_y,dtype=torch.float32,device=self._device,requires_grad=False)
            self.z=torch.tensor(z,dtype=torch.float32,device=self._device,requires_grad=False)
        else:
            self.gz_x=torch.zeros_like(self.U[:,:,0:1])
            self.gz_y=torch.zeros_like(self.U[:,:,0:1])
            self.z=torch.zeros_like(self.U[:,:,0:1])
        
    def __step(self,t):
        # get time step
        dt=self.__time_step(self.U)        
        
        def right_term(U):
            #step1: reconstruction for U
            self.__reconstruction(U)  #left, right, top, bottom

            #step2: flux calculation 
            flux_x_l,flux_x_r,flux_y_b,flux_y_t=self.__flux_calculation()

            #step3: source term calculation
            source=self.__source_term(U,t)

            #step4: right term calculation
            right_term=-((flux_x_r - flux_x_l)/self._dx+(flux_y_t - flux_y_b)/self._dy)+source
            return right_term
        

        #step3: step forward, 3order Runge-Kutta
        U=self.U
        U_one=U+dt*right_term(U)
        U_two=0.75*U+0.25*U_one+0.25*dt*right_term(U_one)
        self.U=(U+2*U_two+2*dt*right_term(U_two))/3.     #update U

        return dt.item()     

    def __reconstruction(self,U):
        #the one of the key step in the FV method
        #just support the one order reconstruction for now

        #reflective boundary condition
        self.U_l[:,1:,:]=U
        self.U_r[:,:-1,:]=U
        self.U_l[:,0,:]=U[:,0,:]
        self.U_r[:,-1,:]=U[:,-1,:]

        # #periodic boundary condition
        # self.U_l[:,1:,:]=U
        # self.U_r[:,:-1,:]=U
        # self.U_l[:,0,:]=U[:,-1,:]   #
        # self.U_r[:,-1,:]=U[:,0,:]

        #periodic boundary condition
        self.U_t[1:,:,:]=U
        self.U_b[:-1,:,:]=U
        self.U_t[0,:,:]=U[-1,:,:]
        self.U_b[-1,:,:]=U[0,:,:]

    def __flux_calculation(self):
        
        def HLL(U_l,U_r,U_b,U_t):
            #U=[h,hu,hv]
            hl,hr=U_l[:,:,0:1],U_r[:,:,0:1]
            hul,hur=U_l[:,:,1:2],U_r[:,:,1:2]
            hvl,hvr=U_l[:,:,2:],U_r[:,:,2:]

            hb,ht=U_b[:,:,0:1],U_t[:,:,0:1]
            hub,hut=U_b[:,:,1:2],U_t[:,:,1:2]
            hvb,hvt=U_b[:,:,2:],U_t[:,:,2:]


            zeros=torch.zeros_like(hl)
            ul=torch.where(hl>0,hul/hl,zeros)
            vl=torch.where(hl>0,hvl/hl,zeros)
            ur=torch.where(hr>0,hur/hr,zeros)
            vr=torch.where(hr>0,hvr/hr,zeros)
            zeros=torch.zeros_like(hb)
            ub=torch.where(hb>0,hub/hb,zeros)
            vb=torch.where(hb>0,hvb/hb,zeros)
            ut=torch.where(ht>0,hut/ht,zeros)
            vt=torch.where(ht>0,hvt/ht,zeros)


            F_l=torch.zeros_like(U_l)
            F_l[:,:,0:1]=hul                   # hu
            F_l[:,:,1:2]=hul*ul+0.5*self.g*hl*hl    #hu^2+0.5gh^2
            F_l[:,:,2:]=hul*vl                 #huv

            F_r=torch.zeros_like(U_r)
            F_r[:,:,0:1]=hur                   # hu
            F_r[:,:,1:2]=hur*ur+0.5*self.g*hr*hr    #hu^2+0.5gh^2
            F_r[:,:,2:]=hur*vr                 #huv

            G_b=torch.zeros_like(U_b)
            G_b[:,:,0:1]=hvb                   # hv
            G_b[:,:,1:2]=hvb*ub                #hvu
            G_b[:,:,2:]=hvb*vb+0.5*self.g*hb*hb     #hv^2+0.5gh^2

            G_t=torch.zeros_like(U_t)
            G_t[:,:,0:1]=hvt                   #hv
            G_t[:,:,1:2]=hvt*ut                #hvu
            G_t[:,:,2:]=hvt*vt+0.5*self.g*ht*ht     #hv^2+0.5gh^2

            #--calculate S--
            def cal_S():
                sqrt_ghl=torch.sqrt(self.g*hl)
                sqrt_ghr=torch.sqrt(self.g*hr)
                sqrt_ghb=torch.sqrt(self.g*hb)
                sqrt_ght=torch.sqrt(self.g*ht)
                sqrt_gh_star_i=0.5*(sqrt_ghl+sqrt_ghr)+0.25*(ul-ur)
                sqrt_gh_star_j=0.5*(sqrt_ghb+sqrt_ght)+0.25*(ub-ut)
                u_star=0.5*(ul+ur)+sqrt_ghl-sqrt_ghr
                v_star=0.5*(vb+vt)+sqrt_ghb-sqrt_ght

                Sl=torch.min(ul-sqrt_ghl,u_star-sqrt_gh_star_i)
                Sr=torch.max(ur+sqrt_ghr,u_star+sqrt_gh_star_i)

                Sb=torch.min(ub-sqrt_ghb,v_star-sqrt_gh_star_j)
                St=torch.max(ut+sqrt_ght,v_star+sqrt_gh_star_j)

                #dry conditions
                Sl=torch.where(hr<=0,ul-sqrt_ghl,Sl)      #the right is dry
                Sl=torch.where(hl<=0,ur-2*sqrt_ghr,Sl)    #the left is dry

                Sr=torch.where(hl<=0,ul+2*sqrt_ghl,Sr)    #the right is dry
                Sr=torch.where(hr<=0,ur+sqrt_ghr,Sr)      #the left is dry

                Sb=torch.where(ht<=0,vb-sqrt_ghb,Sb)      #the top is dry
                Sb=torch.where(hb<=0,vt-2*sqrt_ght,Sb)     #the bottom is dry

                St=torch.where(hb<=0,vb+2*sqrt_ghb,St)      #the top is dry
                St=torch.where(ht<=0,vt+sqrt_ght,St)        #the bottom is dry

                return Sl,Sr,Sb,St

            #-------------- 
            Sl,Sr,Sb,St=cal_S()           
            i_l=(Sl>=0)[:,:,0]
            i_r=(Sr<=0)[:,:,0]
            j_b=(Sb>=0)[:,:,0]
            j_t=(St<=0)[:,:,0]
        
            

            F_flux=torch.zeros_like(U_l)
            G_flux=torch.zeros_like(U_b)

            #for Sl>=0 ; Sb>=0
            F_flux[i_l,:]=F_l[i_l,:]
            G_flux[j_b,:]=G_b[j_b,:]

            #for Sr<=0; St<=0
            F_flux[i_r,:]=F_r[i_r,:]
            G_flux[j_t,:]=G_t[j_t,:]


            #for Sl<0 and Sr>0
            i_mid=torch.logical_and(~i_l,~i_r)
            F_flux[i_mid,:]=(Sr[i_mid,:,]*F_l[i_mid,:]-Sl[i_mid,:]*F_r[i_mid,:]+
                             Sl[i_mid,:]*Sr[i_mid,:]*(U_r[i_mid,:]-U_l[i_mid,:])) \
                            /(Sr[i_mid,:]-Sl[i_mid,:])

            j_mid=torch.logical_and(~j_b,~j_t)
            G_flux[j_mid,:]=(St[j_mid,:,]*G_b[j_mid,:]-Sb[j_mid,:]*G_t[j_mid,:]+
                             Sb[j_mid,:]*St[j_mid,:]*(U_t[j_mid,:]-U_b[j_mid,:])) \
                             /(St[j_mid,:]-Sb[j_mid,:])
            
            return F_flux,G_flux
        

        #compute the fluxes
        F_flux,G_flux=HLL(self.U_l,self.U_r,self.U_b,self.U_t)

        cell_flux_x_l=F_flux[:,:-1,:]    #view
        cell_flux_x_r=F_flux[:,1:,:]
        cell_flux_y_b=G_flux[1:,:,:]
        cell_flux_y_t=G_flux[:-1,:,:]
        return cell_flux_x_l,cell_flux_x_r,cell_flux_y_b,cell_flux_y_t

    def __source_term(self,U,t):

        if self._rain_func==None and self._z_func==None:
            return torch.zeros_like(self.U)
        
        h=U[:,:,0:1]
        if self._rain_func==None:
            rain=torch.zeros_like(h)
        else:
            rain=self._rain_func(self._X,self._Y,t)

        source_term=torch.zeros_like(self.U)
        source_term[:,:,0:1]=rain
        source_term[:,:,1:2]=-self.gz_x*h
        source_term[:,:,2:3]=-self.gz_y*h

        return source_term
    
    def __time_step(self,U:torch.tensor):
        # get dt
        wave_celerity=torch.sqrt(self.g*U[:,:,0])
        lambd1 = wave_celerity+ torch.abs(U[:,:,1])
        lambd2=wave_celerity+ torch.abs(U[:,:,2])
        mask_x = lambd1 != 0
        mask_y=  lambd2 !=0
        dt_x= torch.where(mask_x, self._cfl * self._dx / lambd1, 1e4 * torch.ones_like(lambd1))
        dt_y= torch.where(mask_y, self._cfl * self._dy / lambd2, 1e4 * torch.ones_like(lambd2))
        dt =torch.min(torch.amin(dt_x),torch.amin(dt_y))
        return dt   



class FV_slover_Well_Balance():
    """ a well-balance solver for the Shallow water equation with bottom topography and rain."""
    def __init__(self,
                 bbox=[[0,1],[0,1]],
                 dx_dy=(0.1,0.1),
                 t_end=10.,
                 report_interval=1.,  
                 init_func=None, #init_func(X,Y,U)
                 z_func=None,  #z_func(X,Y)
                 rain_func=None,  #rain_func(X,Y,t)
                 bcs="PERIODIC",
                 cfl=0.25,     #strict CFL condition 0.25
                 device='cpu'
                 ):
        BOUNDARY_TYPE={"PERIODIC":self.__periodic_bcs,"RELECTIVE":self.__reflective_bcs}   #just support the periodic and reflective boundary condition for now
        if bcs not in BOUNDARY_TYPE:
            raise ValueError(f"The boundary condition {bcs} is not supported, please choose from {BOUNDARY_TYPE}")
        self._bcs_func=BOUNDARY_TYPE[bcs]
        
        self._bbox=bbox    #
        self._dx=dx_dy[0]
        self._dy=dx_dy[1]

        self._t_end=t_end
        self._report_times=[report_interval*i for i in range(int(t_end/report_interval)+1)]  #the demand timestamps for report
        if t_end not in self._report_times:
            self._report_times.append(t_end)  #add the last report time

        self._init_func=init_func
        
        self._z_func=z_func
        self._rain_func=rain_func
        
        self._mesh_created=False

        self._device=device  #cpu or cuda

        self._results=[]    #record the results

        self._cfl=cfl
        self.g=9.81
        
    def __call__(self):
        #step1: create mesh
        self.__creat_mesh()

        #step2: init U
        self.__init_U()
        
        #step3: iteration to solve the problem
        i,t=0,0
        t_end=torch.tensor(self._t_end,dtype=torch.float32,device=self._device,requires_grad=False)
        t=torch.tensor(t,dtype=torch.float32,device=self._device,requires_grad=False)
        report_times=torch.tensor(self._report_times,dtype=torch.float32,device=self._device,requires_grad=False)
        while(t<t_end):
            if t>=report_times[i]:   #report 0
                self._report_times[i]=t.item()
                self._results.append(self.U.detach().cpu().numpy())
                i+=1
                print(f"current time is {t}")
            with torch.no_grad():
                dt=self.__step(t)
            t+=dt
        else:
            self._report_times[i]=t.item()
            self._results.append(self.U.detach().cpu().numpy())
            print(f"current time is {t}")
        
        return self._X, self._Y, self._results, self._report_times  #results=[numpy:shape=(m,n,3),numpy,numpy,......],Note: h，hu,hv

    def __creat_mesh(self):
        self._dx_num=int((self._bbox[0][1]-self._bbox[0][0])/self._dx)
        self._dy_num=int((self._bbox[1][1]-self._bbox[1][0])/self._dy)

        x = np.linspace(self._bbox[0][0]+self._dx/2.,self._bbox[0][1]-self._dx/2.,self._dx_num)
        y = np.linspace(self._bbox[1][0]+self._dy/2.,self._bbox[1][1]-+self._dx/2.,self._dy_num)
        self._X, self._Y = np.meshgrid(x,y)  #
        self._mesh_created=True
    
    def __init_U(self):
        #U=[h,hu,hv]^T
        if not self._mesh_created:
            raise ValueError("Before initializing U, please create the Mesh")
        if self._init_func is None:
            raise ValueError("Please provide an initial condition function")
        else:
            self.U=self._init_func(self._X,self._Y)
        #check the shape of U
        if self.U.shape!=(self._dy_num,self._dx_num,3):
            raise ValueError("The shape of U is not correct")
        U_shape=np.array(self.U.shape)
        self.U=torch.tensor(self.U,dtype=torch.float32,device=self._device,requires_grad=False)
        self.U_var=torch.zeros_like(self.U)

        self.U_var_l=torch.zeros(tuple(U_shape+np.array([0,1,0])),dtype=torch.float32,device=self._device,requires_grad=False)
        self.U_var_r=torch.zeros(tuple(U_shape+np.array([0,1,0])),dtype=torch.float32,device=self._device,requires_grad=False)
        self.U_var_t=torch.zeros(tuple(U_shape+np.array([1,0,0])),dtype=torch.float32,device=self._device,requires_grad=False)
        self.U_var_b=torch.zeros(tuple(U_shape+np.array([1,0,0])),dtype=torch.float32,device=self._device,requires_grad=False)
        self._dx=torch.tensor(self._dx,dtype=torch.float32,device=self._device,requires_grad=False)
        self._dy=torch.tensor(self._dy,dtype=torch.float32,device=self._device,requires_grad=False)

        #initialize the bottom topography
        if self._z_func is not None:            
            z=self._z_func(self._X,self._Y)
            z=torch.tensor(z,dtype=torch.float32,device=self._device,requires_grad=False)
        else:
            z=torch.zeros_like(self.U[:,:,0:1])
        self.z_x_diff=torch.zeros_like(self.U_var_l[:,:,0:1])
        # self.z_x_diff[:,0,:]=0.    #the first column is 0
        # self.z_x_diff[:,-1,:]=0.   #the last column is 0
        self.z_x_diff[:,1:-1,:]=z[:,1:,:]-z[:,:-1,:]  
        
        self.z_y_diff=torch.zeros_like(self.U_var_t[:,:,0:1])
        # self.z_y_diff[0,:,:]=0.    #the first row is 0
        # self.z_y_diff[-1,:,:]=0.   #the last row is 0
        self.z_y_diff[1:-1,:,:]=z[:-1,:,:]-z[1:,:,:]  
        
    def __step(self,t):
        # get time step
        dt=self.__time_step(self.U)        
        
        def right_term(U):
            #step1: reconstruction for U
            self.__reconstruction(U)  #left, right, top, bottom

            #step2: flux calculation 
            flux_x_l,flux_x_r,flux_y_b,flux_y_t=self.__flux_calculation()

            #step3: source term calculation
            source=self.__source_term(t)

            #step4: right term calculation
            right_term=-((flux_x_r - flux_x_l)/self._dx+(flux_y_t - flux_y_b)/self._dy)+source
            return right_term
        

        #step3: step forward, 3order Runge-Kutta
        U=self.U
        U_one=U+dt*right_term(U)
        U_two=0.75*U+0.25*U_one+0.25*dt*right_term(U_one)
        self.U=(U+2*U_two+2*dt*right_term(U_two))/3.     #update U

        return dt.item()     

    def __reconstruction(self,U):
        self._bcs_func(U)
        

    def __periodic_bcs(self,U):
        self.U_var[:,:,0:1]=U[:,:,0:1]
        self.U_var[:,:,1:2]=U[:,:,1:2]/U[:,:,0:1]   #not dry  u
        self.U_var[:,:,2:3]=U[:,:,2:3]/U[:,:,0:1]   #not dry  v
        # #periodic boundary condition
        self.U_var_l[:,1:,:]=self.U_var
        self.U_var_r[:,:-1,:]=self.U_var
        self.U_var_l[:,0,:]=self.U_var[:,-1,:]   #
        self.U_var_r[:,-1,:]=self.U_var[:,0,:]
        #periodic boundary condition
        self.U_var_t[1:,:,:]=self.U_var
        self.U_var_b[:-1,:,:]=self.U_var
        self.U_var_t[0,:,:]=self.U_var[-1,:,:]
        self.U_var_b[-1,:,:]=self.U_var[0,:,:]

    def __reflective_bcs(self,U):
        self.U_var[:,:,0:1]=U[:,:,0:1]
        self.U_var[:,:,1:2]=U[:,:,1:2]/U[:,:,0:1]   #not dry  u
        self.U_var[:,:,2:3]=U[:,:,2:3]/U[:,:,0:1]   #not dry  v

        #reflective boundary condition
        self.U_var_l[:,1:,:]=U
        self.U_var_r[:,:-1,:]=U
        self.U_var_l[:,0,:]=U[:,0,:]
        self.U_var_r[:,-1,:]=U[:,-1,:]

        self.U_var_t[1:,:,:]=self.U_var
        self.U_var_b[:-1,:,:]=self.U_var
        self.U_var_t[0,:,:]=self.U_var[0,:,:]
        self.U_var_b[-1,:,:]=self.U_var[-1,:,:]

    def __flux_calculation(self):
        
        def flux(U_v_l,U_v_r,U_v_b,U_v_t):
            F_flux=torch.zeros_like(U_v_l)
            G_flux=torch.zeros_like(U_v_b)

            U_bar_F=0.5*(U_v_l+U_v_r)  #(h_bar,u_bar,v_bar)
            U_bar_G=0.5*(U_v_b+U_v_t)

            h_sqaure_bar_F=(U_v_l[:,:,0:1]**2+U_v_r[:,:,0:1]**2)/2.  #(h*h)_bar
            h_sqaure_bar_G=(U_v_b[:,:,0:1]**2+U_v_t[:,:,0:1]**2)/2.  #(h*h)_bar

            F_flux[:,:,0:1]=U_bar_F[:,:,0:1]*U_bar_F[:,:,1:2] #h_bar*u_bar
            F_flux[:,:,1:2]=U_bar_F[:,:,0:1]*U_bar_F[:,:,1:2]**2+0.5*self.g*h_sqaure_bar_F #h_bar*u_bar**2+0.5gh**2
            F_flux[:,:,2:]=U_bar_F[:,:,0:1]*U_bar_F[:,:,1:2]*U_bar_F[:,:,2:] #h_bar*u_bar*v_bar
            
            G_flux[:,:,0:1]=U_bar_G[:,:,0:1]*U_bar_G[:,:,2:] #h_bar*v_bar
            G_flux[:,:,1:2]=U_bar_G[:,:,0:1]*U_bar_G[:,:,2:]*U_bar_G[:,:,1:2] #h_bar*v_bar*u_bar
            G_flux[:,:,2:]=U_bar_G[:,:,0:1]*U_bar_G[:,:,2:]**2+0.5*self.g*h_sqaure_bar_G #h_bar*v_bar**2+0.5gh**2
            
            
            #
            matrix_F=torch.zeros((*U_v_l.shape,3))   #shape=(m,n,3,3)
            sqrt_gh_bar=torch.sqrt(self.g*U_bar_F[:,:,0:1])  #=sqrt(gh_bar)
            a_n=U_bar_F[:,:,1:2]-sqrt_gh_bar  #=u_bar-sqrt(gh_bar)
            a_p=U_bar_F[:,:,1:2]+sqrt_gh_bar  #=u_bar+sqrt(gh_bar)
            abs_a_n=torch.abs(a_n)
            abs_a_p=torch.abs(a_p)
            matrix_F[:,:,0:1,0]=abs_a_n+abs_a_p
            matrix_F[:,:,0:1,1]=a_n*abs_a_n+a_p*abs_a_p
            matrix_F[:,:,0:1,2]=(abs_a_n+abs_a_p)*U_bar_F[:,:,2:]
            matrix_F[:,:,1:2,0]=matrix_F[:,:,0:1,1]
            matrix_F[:,:,1:2,1]=abs_a_n**3+abs_a_p**3
            matrix_F[:,:,1:2,2]=(a_n*abs_a_n*+a_p*abs_a_p)*U_bar_F[:,:,2:]
            matrix_F[:,:,2:,0]=matrix_F[:,:,0:1,2]
            matrix_F[:,:,2:,1]=matrix_F[:,:,1:2,2]
            matrix_F[:,:,2:,2]=(abs_a_n+abs_a_p)*U_bar_F[:,:,2:]**2+self.g*U_bar_F[:,:,0:1]*torch.abs(U_bar_F[:,:,1:2])
            
            #
            matrix_G=torch.zeros((*U_v_b.shape,3))   #shape=(m,n,3,3)
            sqrt_gh_bar=torch.sqrt(self.g*U_bar_G[:,:,0:1])  #=sqrt(gh_bar)
            a_n=U_bar_G[:,:,2:]-sqrt_gh_bar  #=v_bar-sqrt(gh_bar)
            a_p=U_bar_G[:,:,2:]+sqrt_gh_bar  #=v_bar+sqrt(gh_bar)
            abs_a_n=torch.abs(a_n)
            abs_a_p=torch.abs(a_p)
            matrix_G[:,:,0:1,0]=abs_a_n+abs_a_p
            matrix_G[:,:,0:1,1]=(abs_a_n+abs_a_p)*U_bar_G[:,:,1:2]
            matrix_G[:,:,0:1,2]=a_n*abs_a_n+a_p*abs_a_p
            matrix_G[:,:,1:2,0]=matrix_G[:,:,0:1,1]
            matrix_G[:,:,1:2,1]=(abs_a_n+abs_a_p)*U_bar_G[:,:,1:2]**2+self.g*U_bar_G[:,:,0:1]*torch.abs(U_bar_G[:,:,2:])
            matrix_G[:,:,1:2,2]=(a_n*abs_a_n*+a_p*abs_a_p)*U_bar_G[:,:,1:2]
            matrix_G[:,:,2:,0]=matrix_G[:,:,0:1,2]
            matrix_G[:,:,2:,1]=matrix_G[:,:,1:2,2]
            matrix_G[:,:,2:,2]=abs_a_n**3+abs_a_p**3

            V_F=U_v_r-U_v_l
            V_F[:,:,0:1]=self.g*((U_v_r[:,:,0:1]-U_v_l[:,:,0:1])+self.z_x_diff)-\
                        0.5*(U_v_r[:,:,1:2]**2-U_v_l[:,:,1:2]**2+U_v_r[:,:,2:]**2-U_v_l[:,:,2:]**2)
            F_flux_2= 0.5*0.5/self.g*torch.einsum('ijkl,ijl->ijk', matrix_F, V_F)
            

            V_G=U_v_t-U_v_b
            V_G[:,:,0:1]=self.g*((U_v_t[:,:,0:1]-U_v_b[:,:,0:1])+self.z_y_diff)-\
                        0.5*(U_v_t[:,:,1:2]**2-U_v_b[:,:,1:2]**2+U_v_t[:,:,2:]**2-U_v_b[:,:,2:]**2)

            G_flux_2= 0.5*0.5/self.g*torch.einsum('ijkl,ijl->ijk', matrix_G, V_G)

            return F_flux-F_flux_2,G_flux-G_flux_2
        #compute the fluxes
        F_flux,G_flux=flux(self.U_var_l,self.U_var_r,self.U_var_b,self.U_var_t)

        cell_flux_x_l=F_flux[:,:-1,:]    #view
        cell_flux_x_r=F_flux[:,1:,:]
        cell_flux_y_b=G_flux[1:,:,:]
        cell_flux_y_t=G_flux[:-1,:,:]
        return cell_flux_x_l,cell_flux_x_r,cell_flux_y_b,cell_flux_y_t

    def __source_term(self,t):
        source=torch.zeros_like(self.U)
        def bottom_source(U_var_l,U_var_r,U_var_b,U_var_t,source):
            h_bar_F=0.5*(U_var_l[:,:,0:1]+U_var_r[:,:,0:1])
            h_bar_G=0.5*(U_var_b[:,:,0:1]+U_var_t[:,:,0:1])
            source[:,:,1:2]=-0.5/self._dx*self.g*(h_bar_F[:,:-1,:]*self.z_x_diff[:,:-1,:]+h_bar_F[:,1:,:]*self.z_x_diff[:,1:,:])
            source[:,:,2:3]=-0.5/self._dy*self.g*(h_bar_G[:-1,:,:]*self.z_y_diff[:-1,:,:]+h_bar_G[1:,:,:]*self.z_y_diff[1:,:,:])


        if self._rain_func==None and self._z_func==None:  
            return source #no source term
        
        if self._rain_func!=None:  #cal rain source term
            source[:,:,0:1]=self._rain_func(self._X,self._Y,t)
        
        if self._z_func!=None:     #cal bottom source term
            bottom_source(self.U_var_l,self.U_var_r,self.U_var_b,self.U_var_t,source)

        return source
    
    def __time_step(self,U:torch.tensor):
        # get dt
        wave_celerity=torch.sqrt(self.g*U[:,:,0])
        lambd1 = wave_celerity+ torch.abs(U[:,:,1])
        lambd2=  wave_celerity+ torch.abs(U[:,:,2])
        mask_x = lambd1 != 0
        mask_y=  lambd2 !=0
        dt_x= torch.where(mask_x, self._cfl * self._dx / lambd1, 1e4 * torch.ones_like(lambd1))
        dt_y= torch.where(mask_y, self._cfl * self._dy / lambd2, 1e4 * torch.ones_like(lambd2))
        dt =torch.min(torch.amin(dt_x),torch.amin(dt_y))
        return dt   





class Dam_Break():
    def __init__(self,bbox=[[-10,10],[-10,10]],dx_dy=(0.05,0.05),
                 t_end=0.6,report_interval=0.3,init_value=[2,1],bcs=None):
        self.bbox=bbox
        self.dx_dy=dx_dy
        self.t_end=t_end
        self.report_interval=report_interval
        self.init_value=init_value
        self.init_func=self.__init_func
    def __init_func(self,X,Y):
        U=np.zeros((*X.shape,3))
        U[X<0.,:]=np.array([self.init_value[0],0.,0.])
        U[X>0.,:]=np.array([self.init_value[1],0.,0.])
        return U
    def __call__(self):
        input_dic={'bbox':self.bbox,'dx_dy':self.dx_dy,
                    't_end':self.t_end,'report_interval':self.report_interval,
                     'init_func':self.init_func,
                     'bcs':"RELECTIVE"}
        return input_dic
    def plot_results(self,results,report_times):
        for i in range(len(results)):
            h=results[i][:,:,0]
            h_x=np.mean(h,axis=0)
            plt.figure()
            x=np.linspace(self.bbox[0][0],self.bbox[0][1],h.shape[1])
            plt.plot(x,h_x)
            plt.title(f"time={report_times[i]}")
            plt.show()
        


class Dam_Break_Topography_and_Rain():
    def __init__(self,bbox=[[-10,10],[-10,10]],dx_dy=(0.05,0.05),
                 t_end=1.2,report_interval=0.3,init_value=[2,1],bcs=None):
        self.bbox=bbox
        self.dx_dy=dx_dy
        self.t_end=t_end
        self.report_interval=report_interval
        self.init_value=init_value
        self.init_func=self.__init_func
    def __call__(self):
        input_dic={'bbox':self.bbox,'dx_dy':self.dx_dy,
                    't_end':self.t_end,'report_interval':self.report_interval,
                     'init_func':self.init_func,
                     'z_func':self.__z_func,
                     'rain_func':self.__rain_func
                     }
        return input_dic
    def __init_func(self,X,Y):
        U=np.zeros((*X.shape,3))
        U[X<0.,:]=np.array([self.init_value[0],0.,0.])
        U[X>0.,:]=np.array([self.init_value[1],0.,0.])
        return U
    def __rain_func(self,X,Y,t):
        _temp=torch.abs(t-2.5)/0.5
        ## the accumulative rainfall depth (m) is  0.024019249187982226 m during 5s
        rain=73.77884395200002*(_temp*0.13+18.1)/torch.pow(_temp+18.1,1.870) / 10. #cm/s
        # rain=0.8*torch.sin(torch.pi/5*points[:,2:])   #  cm/min 2.546   easy case 
        return rain
    
    def __z_func(self,X,Y):
        z=np.zeros((*X.shape,1))
        z_x=z
        z_y=z
        return z,z_x,z_y
        


    def plot_results(self,results,report_times):
        for i in range(len(results)):
            h=results[i][:,:,0]
            h_x=np.mean(h,axis=0)
            plt.figure()
            x=np.linspace(self.bbox[0][0],self.bbox[0][1],h.shape[1])
            plt.plot(x,h_x)
            plt.title(f"time={report_times[i]}")
            plt.show()

#
class Circular_Dam_Break():
    def __init__(self,bbox=[[-10,10],[-10,10]],dx_dy=(0.02,0.02),
                 t_end=0.8,report_interval=0.1,init_value=[2,1],bcs=None):
        self.bbox=bbox
        self.dx_dy=dx_dy
        self.t_end=t_end
        self.report_interval=report_interval
        self.init_value=init_value
        # self.init_func=self.__init_func
        # self.bcs=bcs                   
    def __call__(self):
        input_dic={'bbox':self.bbox,'dx_dy':self.dx_dy,
                    't_end':self.t_end,'report_interval':self.report_interval,
                     'init_func':self.__init_func}
        return input_dic
    def __init_func(self,X,Y):
        U=np.zeros((*X.shape,3))
        in_circle=(X**2+Y**2)<=3**2   #circle radius is 2.5
        U[in_circle,:]=np.array([self.init_value[0],0.,0.])
        U[~in_circle,:]=np.array([self.init_value[1],0.,0.])
        return U
    
    def plot_results(self,X,Y,results,report_times):
        from mpl_toolkits.mplot3d import axes3d
        for i in range(len(results)):
            h=results[i][:,:,0]
            ax = plt.figure().add_subplot(projection='3d')
            # Plot the 3D surface
            ax.plot_surface(X, Y, h)
            ax.set(xlabel='X', ylabel='Y', zlabel='Z')
            ax.set_title(f"time={report_times[i]}")
            plt.show()


class Tidal_Case():
    def __init__(self,bbox=[[-2,2],[-2,2]],dx_dy=(0.05,0.05),
                 t_end=5,report_interval=0.5,bcs=None):
        self.bbox=bbox
        self.dx_dy=dx_dy
        self.t_end=t_end
        self.report_interval=report_interval
        self.init_func=self.__init_func
    def __call__(self):
        input_dic={'bbox':self.bbox,'dx_dy':self.dx_dy,
                    't_end':self.t_end,'report_interval':self.report_interval,
                     'init_func':self.init_func,
                     'z_func':self.__z_func,
                    #  'rain_func':self.__rain_func,
                     'rain_func':None   
                     }
        return input_dic
    def __init_func(self,X,Y):
        h=1.0+0.01*np.cos(np.pi/2.*X)*np.cos(np.pi/2.*Y)
        U=np.zeros((*X.shape,3))
        U[:,:,0]=h
        return U

    def __rain_func(self,X,Y,t):
        #0.5s 内完成这样的降雨源项,
        _temp=torch.abs(t*10-2.5)/0.5
        ## the accumulative rainfall depth (m) is  0.024019249187982226 m during 5s
        rain=73.77884395200002*(_temp*0.13+18.1)/torch.pow(_temp+18.1,1.870) / 10. #cm/s
        # rain=0.8*torch.sin(torch.pi/5*points[:,2:])   #  cm/min 2.546   easy case 
        return rain
    
    def __z_func(self,X,Y):
        x,y=np.expand_dims(X,axis=2),np.expand_dims(Y,axis=2)
        z=1.0+0.01*np.cos(np.pi/2.*x)*np.cos(np.pi/2.*y)
        return z


    def plot_results(self,X,Y,results,report_times):
        from mpl_toolkits.mplot3d import axes3d
        for i in range(len(results)):
            h=results[i][:,:,0]
            ax = plt.figure().add_subplot(projection='3d')
            # Plot the 3D surface
            ax.plot_surface(X, Y, h)
            ax.set(xlabel='X', ylabel='Y', zlabel='Z')
            ax.set_title(f"time={report_times[i]}")
            plt.show()

class Tidal_Case_Rian():
    def __init__(self,bbox=[[-2,2],[-2,2]],dx_dy=(0.05,0.05),
                 t_end=0.5,report_interval=0.25,bcs=None):
        self.bbox=bbox
        self.dx_dy=dx_dy
        self.t_end=t_end
        self.report_interval=report_interval
        self.init_func=self.__init_func
    def __call__(self):
        input_dic={'bbox':self.bbox,'dx_dy':self.dx_dy,
                    't_end':self.t_end,'report_interval':self.report_interval,
                     'init_func':self.init_func,
                     'z_func':self.__z_func,
                     'rain_func':self.__rain_func,
                     }
        return input_dic
    def __init_func(self,X,Y):
        h=1.0+0.01*np.cos(np.pi/2.*X)*np.cos(np.pi/2.*Y)
        U=np.zeros((*X.shape,3))
        U[:,:,0]=h
        return U

    def __rain_func(self,X,Y,t):
        #0.5s 内完成这样的降雨源项,
        _temp=torch.abs(t*10-2.5)/0.5
        ## the accumulative rainfall depth (m) is  0.024019249187982226 m during 5s
        rain=73.77884395200002*(_temp*0.13+18.1)/torch.pow(_temp+18.1,1.870)/1000 #m/s
        return rain*10
    
    def __z_func(self,X,Y):
        x,y=np.expand_dims(X,axis=2),np.expand_dims(Y,axis=2)
        z=1.0+0.01*np.cos(np.pi/2.*x)*np.cos(np.pi/2.*y)
        return z


    def plot_results(self,X,Y,results,report_times):
        from mpl_toolkits.mplot3d import axes3d
        for i in range(len(results)):
            h=results[i][:,:,0]
            ax = plt.figure().add_subplot(projection='3d')
            # Plot the 3D surface
            ax.plot_surface(X, Y, h)
            ax.set(xlabel='X', ylabel='Y', zlabel='Z')
            ax.set_title(f"time={report_times[i]}")
            plt.show()


class Lake_at_Rest():
    def __init__(self,bbox=[[0,2],[0,2]],dx_dy=(0.02,0.02),
                 t_end=0.3,report_interval=0.1,bcs=None):
        self.bbox=bbox
        self.dx_dy=dx_dy
        self.t_end=t_end
        self.report_interval=report_interval
        self.init_func=self.__init_func
    def __call__(self):
        input_dic={'bbox':self.bbox,'dx_dy':self.dx_dy,
                    't_end':self.t_end,'report_interval':self.report_interval,
                     'init_func':self.init_func,
                     'z_func':self.__z_func,
                    #  'rain_func':self.__rain_func,
                     'rain_func':None   
                     }
        return input_dic
    def __init_func(self,X,Y):
        z=0.8*np.exp(-5*(X-1.0)**2-5*(Y-1.0)**2)
        h=1-z
        U=np.zeros((*X.shape,3))
        U[:,:,0]=h
        return U

    def __rain_func(self,X,Y,t):
        #0.5s 内完成这样的降雨源项,
        _temp=torch.abs(t*10-2.5)/0.5
        ## the accumulative rainfall depth (m) is  0.024019249187982226 m during 5s
        rain=73.77884395200002*(_temp*0.13+18.1)/torch.pow(_temp+18.1,1.870) / 10. #cm/s
        # rain=0.8*torch.sin(torch.pi/5*points[:,2:])   #  cm/min 2.546   easy case 
        return rain
    
    def __z_func(self,X,Y):
        x,y=np.expand_dims(X,axis=2),np.expand_dims(Y,axis=2)
        z=0.8*np.exp(-5*(x-1.0)**2-5*(y-1.0)**2)
        return z


    def plot_results(self,X,Y,results,report_times):
        Z=0.8*np.exp(-5*(X-1.)**2-5*(Y-1.)**2)
        from mpl_toolkits.mplot3d import axes3d
        for i in range(len(results)):
            h=results[i][:,:,0]
            ax = plt.figure().add_subplot(projection='3d')
            # Plot the 3D surface
            ax.plot_surface(X, Y, h)
            ax.set(xlabel='X', ylabel='Y', zlabel='Z')
            ax.set_title(f"time={report_times[i]}")
            plt.show()

def test_tidal_case():
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device=torch.device('cpu')

    # tidal_input=Lake_at_Rest()
    # tidal_input=Tidal_Case()
    tidal_input=Tidal_Case_Rian()
    

    # tidal_FV=FV_slover(**tidal_input(),device=device)
    tidal_FV=FV_slover_Well_Balance(**tidal_input(),device=device)

    start_time=time.time()
    X,Y,resutls,report_times=tidal_FV()

    end_time=time.time()   
    print(f"total time is {end_time-start_time}")

    #save the results
    # if False:
    np.savez("tidal_case_results.npz",X=X,Y=Y,resutls=resutls,report_times=report_times)
    
    #reload the results
    with np.load("tidal_case_results.npz") as data:
        X=data['X']
        Y=data['Y']
        resutls=data['resutls']
        report_times=data['report_times']
    #plot the results

    tidal_input.plot_results(X,Y,resutls,report_times)


def test_circular_dam_break():
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device=torch.device('cpu')

    cir_damB_input=Circular_Dam_Break()
    cir_damB_FV=FV_slover(**cir_damB_input(),device=device)
    # cir_damB_FV=FV_slover(**cir_damB_input(),device=device)


    start_time=time.time()
    X,Y,resutls,report_times=cir_damB_FV()

    end_time=time.time()   
    print(f"total time is {end_time-start_time}")

    #save the results
    # if False:
    np.savez("cir_damB_results.npz",X=X,Y=Y,resutls=resutls,report_times=report_times)
    
    #reload the results
    with np.load("cir_damB_results.npz") as data:
        X=data['X']
        Y=data['Y']
        resutls=data['resutls']
        report_times=data['report_times']
    #plot the results

    cir_damB_input.plot_results(X,Y,resutls,report_times)




def test_Dam_Break_Topography_and_Rain():
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device=torch.device('cpu')

    # damB_input=Dam_Break_Topography_and_Rain()
    damB_input=Dam_Break()

    # damB_FV=FV_slover(**damB_input(),device=device)

    damB_FV=FV_slover_Well_Balance(**damB_input(),device=device)    

    start_time=time.time()
    X,Y,resutls,report_times=damB_FV()

    end_time=time.time()   
    print(f"total time is {end_time-start_time}")

    #save the results
    # if False:
    np.savez("damB_results.npz",X=X,Y=Y,resutls=resutls,report_times=report_times)
    
    #reload the results
    with np.load("damB_results.npz") as data:
        X=data['X']
        Y=data['Y']
        resutls=data['resutls']
        report_times=data['report_times']
    #plot the results

    damB_input.plot_results(resutls,report_times)


if __name__=="__main__":

    # FV_INSTANCE=FV_slover(bbox=bbox,dx_dy=dx_dy,
    #                     t_end=t_end,report_interval=report_interval,
    #                     init_func=init_func,bcs=None)
    # test_Dam_Break_Topography_and_Rain()
    test_circular_dam_break()
    # test_tidal_case()
    

