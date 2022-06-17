import numpy as np

def GetIndentCords(path,r_origin,b1,b2,ncols,nrows,xlims): 

    xlin = np.arange(ncols)
    ylin = np.arange(nrows)
    yv, xv = np.meshgrid(xlin,ylin)
    indices = list(zip(yv.flatten(),xv.flatten())) #--- square grid

    r_indenters = np.matmul(np.array([b1,b2]),np.array(indices).T).T + r_origin
    filtr = np.all([r_indenters[:,0]>xlims[0], r_indenters[:,0]<xlims[1], 
                 r_indenters[:,1]>xlims[2], r_indenters[:,1]<xlims[3]],axis=0)
    np.savetxt('%s/r_indenters.txt'%path,np.c_[xv[filtr.reshape(xv.shape)],yv[filtr.reshape(yv.shape)],r_indenters[filtr]],fmt='%s',header='loadID\tlabel\tx\ty')


if __name__ == '__main__':
    r_origin = np.array([-41,-11.0]) #--- x,y
    b1 = np.array([2.0,-22.0])
    b2 = np.array([-20.0,-2.0])
    ncols = 15
    nrows = 16
    xlims=(-276.8950,0.4619,-322.8593,0.4619)
    GetIndentCords('/home/kamran.karimi1/Project/git/DislocationAvalanches/nanoindentation/ebsd',
                    r_origin,b1,b2,ncols,nrows,xlims)
    
