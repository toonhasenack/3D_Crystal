import numpy as np
import matplotlib.pyplot as plt
import openpyscad as o
import pandas as pd
from tqdm import tqdm

class Crystal():
    def __init__(self, vertices, edges):
        self.vertices = vertices
        self.edges = edges
    
    def make_grid(self, N):
        self.N = N
        self.grid = np.zeros((N, N, N), dtype = np.bool_)
        index = np.multiply(N,self.vertices).astype(np.int32).T
        for i in range(len(index)):
            self.grid[index[i,0], index[i,1], index[i,2]] = 1
            
        for i in range(len(self.edges)):
            e = self.edges[i] 
            
            p1 = N*self.vertices.T[e[0]]
            p2 = N*self.vertices.T[e[1]]
            
            n = 100
            dx = (p2 - p1)/n
            line = np.zeros((n, 3), dtype = np.int32)
            for i in range(3):
                line[:,i] = np.linspace(1,n,n)
            line = np.add(np.multiply(line, dx), p1).astype(np.int32)
            
            dist = np.sum(np.square(np.diff(line.T)).T,axis=1)
            locs = np.where(dist > 0)
            
            line = line[locs]
            
            for i in range(len(line)):
                self.grid[line[i,0], line[i,1], line[i,2]] = 1
    
    def thicken(self, p):
        idx_true = np.where(self.grid)
        r = int(self.N*p)
        for i in range(-r,r):
            for j in range(-r,r):
                for k in range(-r,r):
                    if np.sqrt(i**2 + j**2 + k**2) <= r:
                        for t in range(len(idx_true[0])):
                            self.grid[idx_true[0][t] + i, idx_true[1][t] + j, idx_true[2][t] + k] = 1
    
    def hollow(self):
        bm_x = np.logical_and(np.roll(self.grid,1,axis=0), np.roll(self.grid,-1,axis=0))
        bm_y = np.logical_and(np.roll(self.grid,1,axis=1), np.roll(self.grid,-1,axis=1))
        bm_z = np.logical_and(np.roll(self.grid,1,axis=2), np.roll(self.grid,-1,axis=2))
        bm_interior = np.logical_and(np.logical_and(self.grid, bm_x), np.logical_and(bm_y, bm_z))
        self.grid[bm_interior] = False
    
    def find_cube_indices(self):
        self.cube_indices = np.array([[0,0,0,0,0,0]], dtype='int32')
        for i in range(-1,2):
            for j in range(-1,2):
                for k in range(-1,2):
                    if (i != 0) | (j !=0) | (k !=0 ):
                        for l in range(-1,2):
                            for m in range(-1,2):
                                for n in range(-1,2):
                                    not_backwards = (i != -l) | (j != -m) | (k != -n)
                                    not_still = (l != 0) | (m != 0) | (n != 0 )
                                    on_cube = (i+l >= -1) & (i+l <= 1) & (j+m >= -1) & (j+m <= 1) & (k+n >= -1) & (k+n <= 1)
                                    if not_backwards & not_still & on_cube:
                                        self.cube_indices = np.concatenate((self.cube_indices, np.array([[i,j,k,l,m,n]])))
        
        self.cube_indices = np.delete(self.cube_indices,0,axis = 0)
    
    def tile(self): 
        
        self.find_cube_indices()
        self.faces = np.array([[0, 0, 0]], dtype='int32')
        
        for index in tqdm(range(self.cube_indices.shape[0])):
            i,j,k,l,m,n = self.cube_indices[index]
            
            trans1 = np.roll(self.grid,-i,axis=0)
            trans1 = np.roll(trans1,-j,axis=1)
            trans1 = np.roll(trans1,-k,axis=2)
            bm1 = np.logical_and(self.grid, trans1)
                        
            trans2 = np.roll(trans1,-l,axis=0)
            trans2 = np.roll(trans2,-m,axis=1)
            trans2 = np.roll(trans2,-n,axis=2)
            bm2 = np.logical_and(self.grid, trans2)

            bm = np.logical_and(bm1, bm2)
            loci = np.array(np.where(bm))                                        

            p1 = np.array([loci[0], loci[1], loci[2]], dtype=np.int32).T
            p2 = np.array([loci[0]+i, loci[1]+j, loci[2]+k], dtype=np.int32).T
            p3 = np.array([loci[0]+i+l, loci[1]+j+m, loci[2]+k+n], dtype=np.int32).T

            #Check for negative values and values outside the range (0,N)
            bm_neg = np.logical_or(np.logical_or(p1 < 0, p2 < 0), p3 < 0)
            bm_outside = np.logical_or(np.logical_or(p1 > self.N, p2 > self.N), p3 > self.N)
            
            d1 = np.abs(p1 - p2)
            d2 = np.abs(p2 - p3)
            d3 = np.abs(p3 - p1)
            
            bm_direction = np.sum(d1[:,[1,2]], axis = 1) == 1
            print(bm_direction)
            
            keep_locs = np.where(np.logical_and(np.logical_not(np.logical_or(bm_neg,bm_outside)), np.logical_not(bm_direction)))[0]

            p1 = p1[keep_locs]
            p2 = p2[keep_locs]
            p3 = p3[keep_locs]
            
            points = np.array(self.points)
            
            locs = []
            for p in [p1,p2,p3]:
                eq1 = np.abs(np.subtract.outer(p[:,0], points[:,0]))
                eq2 = np.abs(np.subtract.outer(p[:,1], points[:,1]))
                eq3 = np.abs(np.subtract.outer(p[:,2], points[:,2]))
                
                eq = eq1 + eq2 + eq3
                locs.append(np.where(eq == 0)[1])
                
            locs = np.array(locs).T
            
            self.faces = np.concatenate((self.faces,locs))

        self.faces = np.delete(self.faces,0,axis = 0)
        self.faces = self.faces.tolist()
        
    def to_points(self):
        self.points = np.array(np.where(self.grid)).T.tolist()
        
    def to_SCAD(self, output):
        p = o.Polyhedron(points=self.points,faces=self.faces)
        p.write(f'{output}.scad'),