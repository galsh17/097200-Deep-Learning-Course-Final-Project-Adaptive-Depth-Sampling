from sklearn.mixture import GaussianMixture
from scipy.interpolate import griddata 
from scipy.interpolate import interp2d 
import numpy as np

def img2centers(img, n_points=100, threshold=0.03):
#converts probabilistic sampling map to discrete sample points
  x = np.where(img>threshold)
  x = list( zip( x[0].ravel(),x[1].ravel()))
  clustering = GaussianMixture(n_components=n_points)
  clustering.fit(x)
  centers = clustering.means_
  centers = centers.astype(int)
  new = np.zeros_like(img)
  new[centers[:,0],centers[:,1]] = 1

  return new
        
def errorCalc(gt, recon, error_type='RMSE'):
    
    # ERRORCALC  error calculation of reconstructed depth map
    
    # gt - numpy array of depth ground-truth
    # recon - numpy array of reconstructed depth map
    # error_type - choose between {'RMSE'(default),'MAE','ARE','iMAE'}
    
    gt = gt/100
    recon = recon/100
    
    # remove zero elements
#     gt_eval = gt[np.nonzero(gt)]
#     recon_eval = recon[np.nonzero(gt)]
    # remove above 100m elements
#     gt_eval = gt_eval[gt_eval<=100]
    gt_eval = gt[gt<=100]
#     recon_eval = recon_eval[gt_eval<=100]
    recon_eval = recon[gt<=100]

    
    diff = gt_eval-recon_eval
    
    if error_type=='RMSE': # root mean square error
        return np.sqrt(np.mean(diff**2))
    
    if error_type=='MAE': # mean absolute error
        return np.mean(np.abs(diff))

    if error_type=='ARE': # absolute relative error
        return np.mean(np.abs(diff)/np.abs(gt_eval))
    
    if error_type=='iMAE': # inverse mean absolute error
        return  np.mean(np.abs(1/recon_eval-1/gt_eval))
        

def interpDepth(depth_gt, sampmap, kind='cubic'):
    
    # INTERP_2D  2D interpolation of sparse depth samples
    
    # depth_gt - numpy array of depth ground-truth
    # sampmap - numpy array with sampled pixels as 1 and non-sampled pixels as 0
    # kind - kind of interpolation (‘cubic’, 'linear', ‘nearest’)
    
    """
    h,w = sampmap.shape
    xi = np.arange(0,len(depth_gt[0,:]),1)
    yi = np.arange(0,len(depth_gt[:,0]),1)
    grid_x,grid_y = np.meshgrid(xi, yi)
    #puts sampling in the coreners to interpolate the entire image
    sampmap[0,0] = 1
    sampmap[0,-1] = 1
    sampmap[-1,-1] = 1
    sampmap[-1,0] = 1
    points = sampmap.nonzero()
    values = depth_gt[points]

    outgrid = griddata(points, values,(grid_y,grid_x), method=kind) # or method='linear', method='cubic'
    return outgrid 
    """
    h,w = sampmap.shape
    xi = np.arange(0,len(depth_gt[0,:]),1)
    yi = np.arange(0,len(depth_gt[:,0]),1)
    grid_x,grid_y = np.meshgrid(xi, yi)
    points = sampmap.nonzero()
    values = depth_gt[points]

    depth_interp = griddata(points, values,(grid_y,grid_x), method='nearest') # or method='linear', method='cubic'

    if kind=='nearest':
        return depth_interp 
    else:
        depth_lin = griddata(points, values,(grid_y,grid_x), method='linear') # or method='linear', method='cubic'
        depth_lin[np.isnan(depth_lin)] = depth_interp[np.isnan(depth_lin)]
        return depth_lin 
       
