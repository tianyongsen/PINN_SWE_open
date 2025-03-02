"""
Apache License 2.0
Copyright [2024] [Susanna Dazzi]
Reference: 
    Dazzi S. (2024). Physics-Informed Neural Networks for the Augmented System of Shallow Water Equations with Topography. 
    *Water Resources Research*.
"""
#This is a independent code file for the Riemann problem,
# the outputs of which are the exact solution of Dynamic_Dam_Break_Case

from math import sqrt 
import numpy as np

# -----------------

# CLASS WITH RIEMANN PROBLEM DATA

class RiemannProblem:
    #name = "RiemannProblem"

    def __init__(self, code="custom",dim_vals=[2.,0.,1.,0.],L=12.,xdisc=0.,tmax=1.):
        self.code = code
        self.tstart = 0.
        self.type = "riemann"

        if(self.code=="custom" or self.code==""):  # user-defined values for RP            
            self.values = dim_vals      # initial values hL, uL, hR, uR (h=depth, u=velocity)
            self.L = L                  # size of the spatial domain 
            self.xdisc = xdisc          # position of the initial discontinuity
            self.tmax = tmax            # size of the temporal domain  
        else:    # values from pre-defined RP (0-8)
            self.values = [1.,0.,1.,0.]   
            self.L = 12.
            self.xdisc = 0.          
            self.tmax = 1.
            if(self.code=="RP0"):       # static case, no discontinuity
                self.L = 12.
            elif(self.code=="RP1"):
                self.values[0] = 2.     #hL
            else:
                raise ValueError ("Code not recognized: only \"RP0\" to \"RP1\" available. Provide \"custom\" and user-defined values as arguments")
       
        self.x_start = self.xdisc - self.L/2
        self.x_end = self.xdisc + self.L/2
        self.hpos_flag = False 
        
    def print_recap(self):
        print("Riemann problem:", self.code, "left state:", self.values[0], self.values[1], "right state", self.values[2], self.values[3])
    
    def nondimensionalize_values(self, scale_H = float, scale_U = float, scale_L = float, scale_T = float):
        self.values[0] = self.values[0] / scale_H
        self.values[1] = self.values[1] / scale_U
        self.values[2] = self.values[2] / scale_H
        self.values[3] = self.values[3] / scale_U
    
    def nondimensionalize_domain(self, scale_L = float, scale_T = float):
        self.L = self.L / scale_L 
        self.xdisc = self.xdisc / scale_L
        self.x_start = self.x_start / scale_L
        self.x_end = self.x_end / scale_L
        self.tmax = self.tmax / scale_T

    def check_drybed(self):
        # checks if the RP includes dry bed or produces dry bed in the star region
        if (self.values[0]<1e-10):
            self.hpos_flag = True
            print("Dry bed on the left")
        elif (self.values[2]<1e-10):
            self.hpos_flag = True
            print("Dry bed on the right")            
        else:
            celL = sqrt(9.81*self.values[0])
            celR = sqrt(9.81*self.values[2])
            dcrit = (self.values[3]-self.values[1]) - 2*(celL+celR)
            if (dcrit>=0):
                self.hpos_flag = True
                print("Dry bed is generated in the middle")  
            else:
                print("Wet bed everywhere")

# -----------------

# ANALYTICAL SOLUTION OF RP

from scipy.optimize import newton
g=9.81

def phi(hs,hLR):
    if(hs>hLR): # shock (Rankine-Hugoniot conditions)        
        return sqrt(0.5*g*(hs+hLR)/(hs*hLR)) * (hs-hLR);  
    else: # rarefaction (Riemann invariants)         
        return 2*( sqrt(g*hs) - sqrt(g*hLR) ); 

def get_values_from_rp(rp=RiemannProblem()):
    hL = rp.values[0]
    uL = rp.values[1]
    hR = rp.values[2]
    uR = rp.values[3]    
    return hL, uL, hR, uR

def func(hs,hL,hR,uL,uR):
    return phi(hs,hL) + phi(hs,hR) + uR - uL;  # equazione 5.5 libro blu Toro

def dfunc(hs,hL,hR,uL,uR):
    eps = 1e-7; # central finite difference approximation
    return (func(hs+eps,hL,hR,uL,uR)-func(hs-eps,hL,hR,uL,uR))/(2*eps) 

def star_region_solution(hL,hR,uL,uR):
    celL = sqrt(g*hL)
    celR = sqrt(g*hR)
    cel0 = 0.5*(celL+celR +0.5*(uL-uR))
    h0 = cel0**2 / g  # initial estimate of h_star (2 raref approx)
    hs = newton(func, h0, fprime=dfunc, args=(hL,hR,uL,uR, ), maxiter=200)
    us = uL - phi(hs,hL)
    return hs, us

def sol_sampling(csi,hL,hR,uL,uR,hs,us):
    if(csi<=us):
        #left of us (sampling point is to the left of the shear) => solution
        #determined by the character of the left wave        
        if(hs>hL): # left shock
            s = uL - sqrt(0.5*g*hs/hL*(hL+hs))
            if(csi<=s):
                h=hL
                u=uL
            else:
                h=hs
                u=us
        else: # left rarefaction
            head = uL - sqrt(g*hL) 
            tail = us - sqrt(g*hs)
            if(csi<=head):    # left
                h=hL 
                u=uL 
            elif(csi>=tail):  # right
                h=hs
                u=us 
            else:              # inside rarefaction
                h = ( (uL+2.*sqrt(g*hL)-csi)/3. )**2 / g 
                u = csi + sqrt(g*h) 
    else:  #  csi>us
        # on the right of us (sampling point is to the right of the shear)     
        if(hs>hR):  # right shock
            s = uR + sqrt(0.5*g*hs/hR*(hs+hR))
            if(csi<=s):
                h = hs
                u = us
            else:
                h = hR
                u = uR 
        else:  # right rarefaction
            tail = us + sqrt(g*hs) 
            head = uR + sqrt(g*hR)
            if(csi<=tail):
                h=hs
                u=us
            elif(csi>=head):
                h=hR
                u=uR
            else:
                h = ( (csi-uR+2.*sqrt(g*hR))/3. )**2 / g
                u = csi - sqrt(g*h) 
    return h, u

def sol_sampling_dry(csi,hL,hR,uL,uR):
    if (hR<1e-10):  # first case: dry bed on the right
        head = uL - sqrt(g*hL)
        tail = uL + 2.* sqrt(g*hL)
        if(csi<=head):
            h=hL
            u=uL 
        elif(csi>=tail):
            h=hR
            u=uR
        else: # inside rarefaction 
            u = (uL + 2.0*sqrt(g*hL) + 2.0*csi)/3.0
            h = ( (uL + 2.0*sqrt(g*hL) - csi)/3.0 )**2 / g
    elif (hL<1e-10): # second case: dry bed on the left  
        head = uR + sqrt(g*hR) 
        tail = uR - 2.* sqrt(g*hR)
        if(csi>=head): 
            h=hR 
            u=uR 
        elif(csi<=tail):
            h=hL
            u=uL 
        else:     
            u = (uR - 2.0*sqrt(g*hR) + 2.0*csi)/3.0
            h = ( (-uR + 2.0*sqrt(g*hR) + csi)/3.0 )**2 /g
    else:  # third case: generation of dry bed in the middle
        SHL = uL - sqrt(g*hL)
        SSL = uL + 2.0*sqrt(g*hL)
        SSR = uR - 2.0*sqrt(g*hR)
        SHR = uR + sqrt(g*hR)
        if (csi<=SHL):
            h=hL
            u=uL
        elif (csi >SHL and csi<= SSL):
            u = (uL + 2.0*sqrt(g*hL) + 2.0*csi)/3.0
            h = ((uL + 2.0*sqrt(g*hL) - csi)/3.0 )**2 /g
        elif (csi >SSL and csi<= SSR):
            h=0
            u=0
        elif (csi >SSR and csi<= SHR):
            u = (uR - 2.0*sqrt(g*hR) + 2.0*csi)/3.0
            h = ((-uR + 2.0*sqrt(g*hR) + csi)/3.0 )**2 /g
        elif (csi>SHR):
            h=hR
            u=uR 
    return h, u

def analytical_solution(rp=RiemannProblem(), ts=1., n_pts=100):
    rp.check_drybed()
    if(rp.hpos_flag):  #dry_solution
        hL, uL, hR, uR = get_values_from_rp(rp)
        dx=(rp.x_end-rp.x_start)/n_pts                   #added by <Tian Yongfu> on 2024/12/29
        xe = np.linspace(rp.x_start+dx/2., rp.x_end-dx/2.,n_pts)    #modifed by <Tian Yongfu> on 2024/12/29

        # xe=np.linspace(rp.x_start,rp.x_end,n_pts)     # original code
        he, ue = np.zeros(np.size(xe)), np.zeros(np.size(xe))
        for i in range(np.size(xe)):
            csi=(xe[i]-rp.xdisc)/ts
            he[i], ue[i] =sol_sampling_dry(csi,hL,hR,uL,uR)
    else:  #wet solution
        hL, uL, hR, uR = get_values_from_rp(rp)
        hs, us = star_region_solution(hL,hR,uL,uR)

        dx=(rp.x_end-rp.x_start)/n_pts                   #added by <Tian Yongfu> on 2024/12/29
        xe = np.linspace(rp.x_start+dx/2., rp.x_end-dx/2.,n_pts)    #modifed by <Tian Yongfu> on 2024/12/29

        # xe=np.linspace(rp.x_start,rp.x_end,n_pts)   # original code

        he, ue = np.zeros(np.size(xe)), np.zeros(np.size(xe))
        for i in range(np.size(xe)):
            csi=(xe[i]-rp.xdisc)/ts
            he[i], ue[i] =sol_sampling(csi,hL,hR,uL,uR,hs,us)
    return xe, he, ue

# -----------------

# USEFUL FUNCTIONS FOR VALIDATION

def create_valdata_ic(rp, n_pts=100):
    # create array of exact solution at t=0 for validator using a pre-defined number of points
    x0 = np.linspace(rp.x_start, rp.x_end, n_pts)
    h0 = np.full(np.shape(x0), rp.values[0])
    h0[int(np.size(h0)/2):] = rp.values[2]
    u0 = np.full(np.shape(x0), rp.values[1])
    u0[int(np.size(u0)/2):] = rp.values[3]
    x0 = np.expand_dims(x0, axis=-1)
    t0 = np.full(np.shape(x0), rp.tstart)
    h0 = np.expand_dims(h0, axis=-1)
    u0 = np.expand_dims(u0, axis=-1)
    q0 = np.multiply(h0,u0)
    return x0, t0, h0, u0, q0

def create_valdata_ic_bottom(rp, n_pts=100):
    # create array of exact solution at t=0 for validator using a pre-defined number of points
    x0, t0, h0, u0, q0 = create_valdata_ic(rp, n_pts)
    z0 = np.full(np.shape(x0), 0)
    return x0, t0, h0, u0, q0, z0

def read_val_profile_from_csv_and_nondim(filename, nd_params, time_val=1., xdisc=0):
    # reads array of exact solution at t=time_val for validator and performs non-dimensionalization
    sol = np.loadtxt(filename, delimiter=",")
    x1 = np.expand_dims(sol[:,0], axis=-1) / nd_params.L0 - xdisc
    t1 = np.full(np.shape(x1), time_val)
    h1 = np.expand_dims(sol[:,1], axis=-1) / nd_params.H0
    u1 = np.expand_dims(sol[:,2], axis=-1) / nd_params.U0
    q1 = np.multiply(h1,u1)
    return x1, t1, h1, u1, q1

def create_val_profile_bottom_flat(prb, x0, nd_params):
    z1 = np.full(np.shape(x0), 0) / nd_params.H0
    return z1

def create_valdata_from_analytical(rp, time_val=1., n_pts=100):
    if (time_val>0):
        xa, ha, ua = analytical_solution(rp, time_val, n_pts)
        xa = np.expand_dims(xa, axis=-1)
        ta = np.full(np.shape(xa), time_val)
        ha = np.expand_dims(ha, axis=-1)
        ua = np.expand_dims(ua, axis=-1)
        qa = np.multiply(ha,ua)
    else:
        xa, ta, ha, ua, qa = create_valdata_ic(rp, n_pts)
    return xa, ta, ha, ua, qa



#revised by <Tian Yongfu> on 2024/11/27
# def create_valdata_from_analytical_and_z(rp, time_val=1., n_pts=101):
#     if (time_val>0):
#         xa, ta, ha, ua, qa = create_valdata_from_analytical(rp, time_val, n_pts)
#     else:
#         xa, ta, ha, ua, qa, za = create_valdata_ic_bottom(rp, n_pts)
#     za = np.full(np.shape(xa), 0)
#     return xa, ta, ha, ua, qa, za


"""New code by <Tian Yongfu> on 2024/11/27"""
def independent_test():
    """you can test the code of this file by running this function"""
    import matplotlib.pyplot as plt

    # difine the Riemann problem with step
    h_0_left=2.0
    h_0_right=1.0

    u_0_left=0.0
    u_0_right=0.0

    Length=16          #the length of the domain
    xdiscontinuity=0.   #the position of the initial discontinuity    
    t_end=1.0            #the end time of the simulation

    #add the y-dimension  y=[0,5] for the 2D
    y=np.arange(0,5+0.02,0.02)  #251 points


    dim_vals=[h_0_left,u_0_right,h_0_right,u_0_right]

    prb=RiemannProblem(code="custom",dim_vals=dim_vals, L=Length, xdisc=xdiscontinuity, tmax=t_end)


    # solve the Riemann problem with step
    step =0.2          # interval of time steps
    npts = 501          # number of points in the solution (500+2)
    t_val_loop = np.arange(prb.tstart, prb.tmax+step, step)
    
    h_all=[]
    u_all=[]
    for t_i in t_val_loop:
        print(f"t={t_i:.2f}")
        x, t, h, u, q= create_valdata_from_analytical(prb, t_i, npts)          
        h_all.append(h)
        u_all.append(u)
        if True:
            plt.plot(x,h,label=f"h(x,t={t_i:.2f})")
            plt.plot(x,u,label=f"u(x,t={t_i:.2f})")
            plt.legend()
            plt.show()
    
    
    #plot and save the solution
    header= "Model:              SWE2d\n"+\
            "Case:               Dynamic_NonFlat_Dam_Break_Case\n"+\
            "Content:            The exact solution of the dynamic non-flat dam break case with step\n"+\
            "Dimension:          2 \n"+\
            "Points:              {}\n".format(npts) +\
            "Time steps:          {}\n".format(len(t_val_loop)) +\
            "Dependent variables: h,u,v\n"
    header+="X     Y    "+' '.join([f"h @ t={t:.4f}   u @ t={t:.4f}  v @ t={t:.4f} " for t in t_val_loop])
    print(header)

    X,Y=np.meshgrid(x,y)      
    H=[] 
    U=[]
    for h,u in zip(h_all,u_all):
        h_expanded = np.tile(h, (1, y.shape[0])).T
        u_expanded = np.tile(u, (1, y.shape[0])).T
        #绘制曲面
        if True:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(X, Y, h_expanded, rstride=1, cstride=1, cmap='rainbow', edgecolor='none')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('h')
            ax.set_title('h(x,y,t)')
            plt.show()
        H.append(h_expanded.reshape(-1))   #reshape to 1D array
        U.append(u_expanded.reshape(-1))

    #将X拉伸为1维
    X=X.reshape(-1)
    Y=Y.reshape(-1)
    V=[ np.zeros_like(U[0]) for i in range(len(U))]
    content_write=[X,Y]
    for h,u,v in zip(H,U,V):
        content_write.append(h)
        content_write.append(u)
        content_write.append(v)
    content_write=np.array(content_write).T


    np.savetxt("ref/Dynamic_Dam_Break_Case.dat", content_write, header=header,fmt='%.10f',delimiter='    ',comments='%')
    print("The solution is saved in ref/Dynamic_Dam_Break_Case.dat")

        



