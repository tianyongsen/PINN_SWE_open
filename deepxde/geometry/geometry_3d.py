import itertools

import numpy as np

from .geometry_2d import Rectangle
from .geometry_nd import Hypercube, Hypersphere


class Cuboid(Hypercube):
    """
    Args:
        xmin: Coordinate of bottom left corner.
        xmax: Coordinate of top right corner.
    """

    def __init__(self, xmin, xmax):
        super().__init__(xmin, xmax)
        dx = self.xmax - self.xmin
        self.area = 2 * np.sum(dx * np.roll(dx, 2))

    def random_boundary_points(self, n, random="pseudo"):
        pts = []
        density = n / self.area
        rect = Rectangle(self.xmin[:-1], self.xmax[:-1])
        for z in [self.xmin[-1], self.xmax[-1]]:
            u = rect.random_points(int(np.ceil(density * rect.area)), random=random)
            pts.append(np.hstack((u, np.full((len(u), 1), z))))
        rect = Rectangle(self.xmin[::2], self.xmax[::2])
        for y in [self.xmin[1], self.xmax[1]]:
            u = rect.random_points(int(np.ceil(density * rect.area)), random=random)
            pts.append(np.hstack((u[:, 0:1], np.full((len(u), 1), y), u[:, 1:])))
        rect = Rectangle(self.xmin[1:], self.xmax[1:])
        for x in [self.xmin[0], self.xmax[0]]:
            u = rect.random_points(int(np.ceil(density * rect.area)), random=random)
            pts.append(np.hstack((np.full((len(u), 1), x), u)))
        pts = np.vstack(pts)
        if len(pts) > n:
            return pts[np.random.choice(len(pts), size=n, replace=False)]
        return pts

    def uniform_boundary_points(self, n):
        h = (self.area / n) ** 0.5
        nx, ny, nz = np.ceil((self.xmax - self.xmin) / h).astype(int) + 1
        x = np.linspace(self.xmin[0], self.xmax[0], num=nx)
        y = np.linspace(self.xmin[1], self.xmax[1], num=ny)
        z = np.linspace(self.xmin[2], self.xmax[2], num=nz)

        pts = []
        for v in [self.xmin[-1], self.xmax[-1]]:
            u = list(itertools.product(x, y))
            pts.append(np.hstack((u, np.full((len(u), 1), v))))
        if nz > 2:
            for v in [self.xmin[1], self.xmax[1]]:
                u = np.array(list(itertools.product(x, z[1:-1])))
                pts.append(np.hstack((u[:, 0:1], np.full((len(u), 1), v), u[:, 1:])))
        if ny > 2 and nz > 2:
            for v in [self.xmin[0], self.xmax[0]]:
                u = list(itertools.product(y[1:-1], z[1:-1]))
                pts.append(np.hstack((np.full((len(u), 1), v), u)))
        pts = np.vstack(pts)
        if n != len(pts):
            print(
                "Warning: {} points required, but {} points sampled.".format(
                    n, len(pts)
                )
            )
        return pts


class Sphere(Hypersphere):
    """
    Args:
        center: Center of the sphere.
        radius: Radius of the sphere.
    """

    def __init__(self, center, radius):
        super().__init__(center, radius)
 

#we add
from .geometry import Geometry
from .. import config
from .sampler import sample
from scipy import stats
from sklearn import preprocessing
class Cylinder(Geometry):
    """ 3D cylinder. 圆柱轴线在x轴上, 圆柱内端圆心坐标默认为(0,0,0),还需要半径和长度参数"""
    def __init__(self, radius,length,axis_center=(0,0,0)):
            if radius <= 0 or length <= 0:
                raise ValueError("Radius and length must be positive.")
            if axis_center[1]!=0 or axis_center[2]!=0:
                raise ValueError("The axis of the cylinder must be in the x-axis.")
            
            
            self.radius = radius
            self.length = length            

            self.axis_center = np.array(axis_center,dtype=config.real(np))
            self.x_min=np.array((0,-radius,-radius),dtype=config.real(np))     #
            self.x_max=np.array((self.axis_center[0]+self.length,radius,radius),dtype=config.real(np))
            self.side_length = self.x_max - self.x_min                         
            
            super().__init__(
                3, (self.x_min, self.x_max), np.linalg.norm(self.side_length)
            )

    def inside(self, x):
        #x:shape=(n,3),np.array 
        #y=x[:,1:2] ,shape=(n,1)
        _inside=np.logical_and(x[:,0]<=self.axis_center[0]+self.length,
                               x[:,0]>=self.axis_center[0]
        )
        return np.logical_and(_inside,
            np.linalg.norm(x[:,1:3], axis=-1) <= self.radius,
        )


    def on_boundary(self, x):
        _on_boundary = np.logical_or(
        np.isclose(x[:,0],self.axis_center[0]),                             
        np.isclose(x[:,0],self.length+self.axis_center[0])                
        ) 
        #Attention: logical_or can only take two arrays
        _on_boundary = np.logical_or(_on_boundary,
            np.isclose(np.linalg.norm(x[:,1:3], axis=-1), self.radius))
         #  is close to the ring
        
        # print(np.count_nonzero(_on_boundary))
        # return np.logical_and(self.inside(x), _on_boundary)
        return _on_boundary

    # def distance2boundary_unitdirn(self, x, dirn):
    #     # https://en.wikipedia.org/wiki/Line%E2%80%93sphere_intersection
    #     xc = x - self.center
    #     ad = np.dot(xc, dirn)
    #     return (-ad + (ad ** 2 - np.sum(xc * xc, axis=-1) + self._r2) ** 0.5).astype(config.real(np))

    # def distance2boundary(self, x, dirn):
    #     return self.distance2boundary_unitdirn(x, dirn / np.linalg.norm(dirn))

    # def mindist2boundary(self, x):
    #     return np.amin(self.radius - np.linalg.norm(x - self.center, axis=-1))

    def boundary_normal(self, x):
        _n=np.zeros_like(x,dtype=config.real(np))
        _n[:,0:1]=-np.isclose(x[:,0:1],self.axis_center[0]).astype(config.real(np))+np.isclose(
            x[:,0:1],self.length+self.axis_center[0])
        
        l=np.linalg.norm(x[:,1:3], axis=-1, keepdims=True)
        _n[:,1:3]=x[:,1:3]/l*np.isclose(l, self.radius)

        # For vertices, the normal is averaged for all directions
        idx = np.count_nonzero(_n, axis=-1) == 3
        if np.any(idx):
            print(
                f"Warning: {self.__class__.__name__} boundary_normal called on vertices. "
                "You may use PDE(..., exclusions=...) to exclude the vertices."
            )
            l = np.linalg.norm(_n[idx], axis=-1, keepdims=True)
            _n[idx] /= l

        return _n

    def random_points(self, n, random="pseudo"):
        #首先在(y,z)的圆里面随机采样ni个点
        # http://mathworld.wolfram.com/DiskPointPicking.html
        y_z=self.__disk_sample(n, random)*self.radius

        #其次对x轴随机采样
        x = sample(n, 1, random)
        x=self.length * x + self.axis_center[0]
        #最后合并
        return np.hstack((x,y_z))


    def random_boundary_points(self, n, random="pseudo"):
        points=np.zeros((n,self.dim),dtype=config.real(np))

        #对0~n的数字随机打乱
        idx = np.arange(n)
        np.random.shuffle(idx)

        # 计算侧面积比例
        ratio = self.length / (self.radius+self.length)

        # 计算每个面的采样点数，这里其实应该随机计算，但是为了保证每次采样上下两面都会有点采样到,
        # 所以这里采样固定比例
        num_side = int(np.round(n*ratio))
        num_circle = n-num_side

        #对上下圆盘采样
        points[idx[num_side:num_side+int(num_circle/2)],0:1]= self.axis_center[0]   
        points[idx[num_side+int(num_circle/2):],0:1]=self.axis_center[0]+self.length
        y_z = self.__disk_sample(num_circle, random)
        points[idx[num_side:],1:3]=y_z*self.radius
        # print(10000-np.count_nonzero(points[:,0:1]))

        #圆环采样
        circle=self.__circle_sample(num_side, random)*self.radius
        #对x轴随机采样
        x = sample(num_side, 1, random)
        x=self.length * x + self.axis_center[0]

        side_points=np.hstack((x,circle))
        points[idx[:num_side],:]=side_points
        return points

    # def uniform_boundary_points(self, n):
    #     theta = np.linspace(0, 2 * np.pi, num=n, endpoint=False)
    #     X = np.vstack((np.cos(theta), np.sin(theta))).T
    #     return self.radius * X + self.center
    
    
    def __disk_sample(self, n, random="pseudo"):
        rng = sample(n, 2, random)
        r, theta = rng[:, 0], 2 * np.pi * rng[:, 1]
        y, z = np.cos(theta), np.sin(theta)
        y_z=(np.sqrt(r) * np.vstack((y, z))).T
        return y_z  #shape=(n,2)
    
    def __circle_sample(self, n, random="pseudo"):
        theta = 2 * np.pi * sample(n, 1, random)
        x = np.cos(theta)
        y = np.sin(theta)
        return np.hstack((x, y))
    
