from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy  as np
import statsmodels.api as sm
import imageio
import os
coords_id=[0,11,13,15,23,25,27,31]
rel=[[0,11],[11,23],[11,13],[13,15],[23,25],[25,27],[27,31]]

import cv2
from utils import calc,get_angle
import params
import os
LOWESS = False      #once set this to true othertime False

def remove_err(df):
    # # import pdb;pdb.set_trace()
    # cols = [2]
    # for i in range(132):
    #     if i%4:
    #         cols.append(i+5)
    # # print("cols here",df.columns)
    # print(cols)
    # df = df[cols]
    # labels = ['frame']
    # cord = {}
    # cord[0] = 'x'
    # cord[1] = 'y'
    # cord[2] = 'z'
    # for i in range(99):
    #     t = i // 3
    #     c = i % 3
    #     labels.append(cord[c]+str(t))

    # df.columns = labels
#         select = ['time','x0','y0','z0','x23','y23','z23','x24','y24','z24','x25','y25','z25','x26','y26','z26']
    # select = ['time','x0','y0', 'x24','y24','x23','y23','x26','y26','x25','y25'] #SQUATS
    # print(df.columns)

    # import pdb;pdb.set_trace()
    
    # print("top botoom",df.columns)
    select = ['frame']
    for coords in range(33):
        select.append('x'+str(coords))
        select.append('y'+str(coords))
        select.append('z'+str(coords))
        select.append('visibility'+str(coords))

    df.columns=select
    select = ['frame']
    for coords in coords_id:
        select.append('x'+str(coords))
        select.append('y'+str(coords))
        select.append('z'+str(coords))

    # select = ['time','x12','y12', 'x14','y14','x16','y16','x24','y24','x26','y26','x28','y28'] #shoulder press

    # print("SIZE = %d before 0 removal"%(len(df)))
    df = df.loc[(df[select[1:-6:3]] !=0).all(axis=1)] # Two ways to handle such data, either decrease visibility threshold or remove such rows. removing is problem if its for a long time
    df = df.loc[(df[select[2:-6:3]] !=0).all(axis=1)]

    df = smooth(df,select)
    return df
def max_min(df,select):
        mn = 10.0
        mx = -10.0

        for i in select:
            mn = min(mn, min(df[i]))
            mx = max(mx, max(df[i]))
        return mx,mn



def _get_pose_size( landmarks, torso_size_multiplier=2.5):
  
    # This approach uses only 2D landmarks to compute pose size.
    # landmarks = landmarks[:, :2]
    # import pdb;pdb.set_trace()
    # Hips center.
    left_hip = landmarks[:,coords_id.index(23)*3:coords_id.index(23)*3+2] # only x and y not z
    # right_hip = landmarks[:,24*3:24*3+2]# only x and y not z
    hips = left_hip# + right_hip) * 0.5

    # Shoulders center.
    left_shoulder = landmarks[:,coords_id.index(11)*3:coords_id.index(11)*3+2]
    # right_shoulder = landmarks[:,12*3:12*3+2]
    shoulders = left_shoulder #+ right_shoulder) * 0.5

    # Torso size as the minimum body size.
    torso_size = np.linalg.norm(shoulders - hips,axis=1)

    # Max dist to pose center.
    pose_center = hips
    total_joints = [i for i in range(99)]
    # import pdb;pdb.set_trace()
    norm_vals = landmarks[:,np.ravel([ i-1 for i in range(len(coords_id)*3) if i%3!=0])] - np.tile(pose_center,len(coords_id))

    # norm_vals = landmarks[:,[i-1 for i in range(1,99) if i%3 != 0 ]] - np.tile(pose_center,33)
    # max_dist = np.max(np.linalg.norm(landmarks[:,[i-1 for i in range(1,99) if i%3 != 0 ]] - np.tile(pose_center,33), axis=1))
    max_dist = np.max(np.linalg.norm(norm_vals.reshape(len(norm_vals),-1,2),axis=2),axis=1)
    return np.maximum(torso_size * torso_size_multiplier, max_dist)



def smooth(df,select,norm1 = False,norm2=False,norm3=False,norm4=False,norm5=False,norm6=False):  
        new_df = pd.DataFrame()
        cols_dict={}
        # remove frame column
        all = select[1:]
        if LOWESS:
            if f == 0:
                f = min( 0.03, (41 / len(df)) )
            time = [i for i in range(len(df))]
            
          
            for i in all:
                new_df[i] = sm.nonparametric.lowess(df[i].values, time,frac= f,
                                                        it=3, delta=0.0, is_sorted=True,
                                                        missing='drop', return_sorted=False)
        else:
            new_df = df[all] 

        joint_ids=coords_id
        skel = new_df.to_numpy()
        joints_data = np.zeros((len(skel),len(joint_ids)*3))
        for index,id in enumerate(joint_ids):
            joints_data[:,index*3:index*3+3] = skel[:,index*3:index*3+3] - skel[:,coords_id.index(23)*3:coords_id.index(23)*3+3]#+skel[:,24*3:24*3+3])/2  # subtracct SpineMid
       

        joint_ids = params.coords_ids 

        if norm1: # CHECK IF THIS IF CORRECT
            # import pdb;pdb.set_trace()
            print("file len",len(joints_data))
            shifted_joints = joints_data

            # import pdb;pdb.set_trace()
            x_min_data,x_max_data = shifted_joints[:,0::3].min(axis=1),shifted_joints[:,0::3].max(axis=1)
            y_min_data,y_max_data = shifted_joints[:,1::3].min(axis=1),shifted_joints[:,1::3].max(axis=1)
            z_min_data,z_max_data = shifted_joints[:,2::3].min(axis=1),shifted_joints[:,2::3].max(axis=1)

            normalised_joints3=shifted_joints.copy()
            normalised_joints3[:,0::3] = (normalised_joints3[:,0::3] - x_min_data[:,None])/(x_max_data[:,None]-x_min_data[:,None]) 
            normalised_joints3[:,1::3] = (normalised_joints3[:,1::3] - y_min_data[:,None])/(y_max_data[:,None]-y_min_data[:,None]) 
            normalised_joints3[:,2::3] = (normalised_joints3[:,2::3] - z_min_data[:,None])/(z_max_data[:,None]-z_min_data[:,None]) 
            if 0 in (x_max_data[:,None]-x_min_data[:,None]) or 0 in (y_max_data[:,None]-y_min_data[:,None]) or 0 in (z_max_data[:,None]-z_min_data[:,None]) :
                import pdb;pdb.set_trace()
            # new_df = normalised_joints3.copy()
        elif norm2:
            body_size = _get_pose_size(skel)
            # import pdb;pdb.set_trace()
            # joints_data /= body_size
            # import pdb;pdb.set_trace()
            joints_data /= np.repeat(body_size.reshape(-1,1),len(coords_id)*3,axis=1)
            normalised_joints3 = joints_data
        elif norm3:
            # import pdb;pdb.set_trace()
            joints_data = np.zeros((len(skel),len(joint_ids)*3))
            for index,id in enumerate(joint_ids):
                joints_data[:,index*3:index*3+3] = skel[:,index*3:index*3+3] 
            minx = np.min(joints_data[:,[3*i for i in range(len(joint_ids))]],axis=1)
            miny = np.min(joints_data[:,[3*i+1 for i in range(len(joint_ids))]],axis=1)
            max_coord = np.max(joints_data[:,np.ravel([i-1 for i in range(1,len(joint_ids)*3+1) if i%3!=0])],axis=1)
            joints_data[:,[3*i for i in range(len(joint_ids))]] -= np.tile(np.reshape(minx,(-1,1)),len(joint_ids))
            joints_data[:,[3*i+1 for i in range(len(joint_ids))]] -= np.tile(np.reshape(miny,(-1,1)),len(joint_ids))
            # import pdb;pdb.set_trace()

            joints_data[:,[3*i for i in range(len(joint_ids))]]  /=  np.tile(np.reshape(max_coord,(-1,1)),len(joint_ids))
            joints_data[:,[3*i+1 for i in range(len(joint_ids))]] /= np.tile(np.reshape(max_coord,(-1,1)),len(joint_ids))
            normalised_joints3 = joints_data

            if norm4:
                # import pdb;pdb.set_trace()
                joints_data[:,[3*i for i in range(len(joint_ids))]]  /=  np.tile(np.reshape(max_coord,(-1,1)),len(joint_ids))
                print(joints_data[0])
                for  i in range(len(joint_ids)):
                    joints_data[:,3*i] =joints_data[:,3*i] / (2*np.linalg.norm(joints_data[:,[3*i,3*i+1]],axis=1))
                    joints_data[:,3*i+1] =joints_data[:,3*i+1] / (2*np.linalg.norm(joints_data[:,[3*i,3*i+1]],axis=1))
                normalised_joints3 = np.nan_to_num(joints_data,0)
                import pdb;pdb.set_trace()
        elif norm5:
            # import pdb;pdb.set_trace()
            mid_joint=23
            final_joints = joints_data.copy() # hip is 0,0
            assert np.all(final_joints[:,3*coords_id.index(mid_joint)] ==0) and np.all(final_joints[:,3*coords_id.index(mid_joint)+1]) ==0  , 'data not hip normalised'
            norm_rel=[[23,25],[25,27],[27,31],[23,11],[11,13],[13,15],[11,0]]
            for sender,receiver in norm_rel:
                s= coords_id.index(sender)
                r= coords_id.index(receiver)
                dist = 4*np.linalg.norm(df[['x'+str(sender),'y'+str(sender)]].values - df[['x'+str(receiver),'y'+str(receiver)]].values, axis=1)
                # assume sender is already correct 
                final_joints[:,3*r] = (joints_data[:,3*r]-joints_data[:,3*s])/dist + joints_data[:,3*s] +  final_joints[:,3*s]  -  joints_data[:,3*s] # shape?
                final_joints[:,3*r+1] = (joints_data[:,3*r+1]-joints_data[:,3*s+1])/dist + joints_data[:,3*s+1] + final_joints[:,3*s+1]  -  joints_data[:,3*s+1] # shape?
            normalised_joints3 = final_joints    
            # np.linalg.norm(np.subtract([final_joints[:,3*0],final_joints[:,3*0+1]] , [final_joints[:,3*1],final_joints[:,3*1+1]]))
        elif norm6:
            # import pdb;pdb.set_trace()
            body_size = _get_pose_size(skel,torso_size_multiplier=1000)
            avg_body_size = np.mean(body_size)
            joints_data = joints_data*avg_body_size
            # import pdb;pdb.set_trace()
            
            joints_data /= np.repeat(body_size.reshape(-1,1),len(coords_id)*3,axis=1)
            normalised_joints3 = joints_data
        else:
            normalised_joints3 = joints_data

        column_labels = []
        for id in joint_ids:
            column_labels.extend(['x'+str(id),'y'+str(id),'z'+str(id)])
        new_df=pd.DataFrame(normalised_joints3,columns=column_labels)




        # new_df = new_df[select]
            
        return new_df
from math import sqrt, cos, sin, acos

def unit_axis_angle(a, b):
    an = sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2])
    bn = sqrt(b[0]*b[0] + b[1]*b[1] + b[2]*b[2])
    ax, ay, az = a[0]/an, a[1]/an, a[2]/an
    bx, by, bz = b[0]/bn, b[1]/bn, b[2]/bn
    nx, ny, nz = ay*bz-az*by, az*bx-ax*bz, ax*by-ay*bx
    nn = sqrt(nx*nx + ny*ny + nz*nz)
    return (nx/nn, ny/nn, nz/nn), acos(ax*bx + ay*by + az*bz)
def rotation_matrix(axis, angle):
    ax, ay, az = axis[0], axis[1], axis[2]
    s = sin(angle)
    c = cos(angle)
    u = 1 - c
    return ( ( ax*ax*u + c,    ax*ay*u - az*s, ax*az*u + ay*s ),
             ( ay*ax*u + az*s, ay*ay*u + c,    ay*az*u - ax*s ),
             ( az*ax*u - ay*s, az*ay*u + ax*s, az*az*u + c    ) )

def multiply(matrix, vector):
    return ( matrix[0][0]*vector[0] + matrix[0][1]*vector[1] + matrix[0][2]*vector[2],
             matrix[1][0]*vector[0] + matrix[1][1]*vector[1] + matrix[1][2]*vector[2],
             matrix[2][0]*vector[0] + matrix[2][1]*vector[1] + matrix[2][2]*vector[2] )


def draw_fig(df,path,name,remove_error=True):
    if remove_error:
        df = remove_err(df)
    
## n_obj=9
    # coords_id =[0,11,23,25,27,31]#,33,34]
    
    

    #[0,33],]
    # rel = [[coords_id.index(a),coords_id.index(b)] for a,b in rel2]
    # rel2.append
    plt.rcParams.update({'font.size': 16})
    plt.figure(figsize=(7, 6))
    plt.ylim([-1,0])
    plt.xlim([0,1])
    # df=pd.DataFrame(df,columns=labels)
    # import pdb;pdb.set_trace()
    Bottom2Top=True
    # if sum(df.loc[:,'y0']) < sum(df.loc[:,'y24']):
    #     Bottom2Top=True
    filenames=[]
    angle_elbow = get_angle("left shoulder", "left elbow", "left wrist", df)
    angle_knee = get_angle("left hip", "left knee", "left ankle", df)
    angle_ankle = get_angle("left knee", "left ankle", "left foot index", df)
    angle_hip = get_angle("left shoulder", "left hip", "left knee", df)

    for pos in range(0,len(df),1):
        plt.ylim([-1,2])
        plt.xlim([-1,1])
        ax=0
        ay=1
        positions_x = []
        positions_y = []
        for a,b in  rel:
            a=a
            b=b
            # a=max(0,a-4)
            # b=max(0,b-4)

            positions_x.extend([df.loc[pos,'x'+str(a)].item(),df.loc[pos,'x'+str(b)].item(),None])
            if  Bottom2Top:
                positions_y.extend([1-df.loc[pos,'y'+str(a)].item(),1-df.loc[pos,'y'+str(b)].item(),None])
            else:
                positions_y.extend([df.loc[pos,'y'+str(a)].item(),df.loc[pos,'y'+str(b)].item(),None])
            # plt.plot(positions_x,positions_y)
            # print(a,b)
            # print(positions_x,positions_y)
            # plt.show(  )

        # print(np.maximum(df.loc[pos,'rx5x15'],df.loc[pos,'rx6x16']),df.loc[pos,'rx5x6'])
        # import pdb;pdb.set_trace()
        plt.plot(positions_x,positions_y)
        plt.grid()
        if name:
            plt.title(name)

        
        plt.text(-.8, 0.4, 'shoulder-hip-knee : '+str(angle_hip[pos]), fontsize = 12) 
        plt.text(-.8, 0, 'shoulder-elbow-wrist : '+str(angle_elbow[pos]), fontsize = 12) 
        plt.text(-.8, -0.4, 'hip-knee-ankle : '+str(angle_knee[pos]), fontsize = 12)
        plt.text(-.8, -0.8, 'knee-ankle-heel : '+str(angle_ankle[pos]), fontsize = 12) 


        # import pdb;pdb.set_trace()
        plt.savefig(str(path)+"/"+str(pos)+'.png')
        filenames.append(str(path)+"/"+str(pos)+'.png')
        

        # plt.draw()
        # plt.pause(.0001)
        
        plt.clf()
    print("{} saved at {}".format(name+'.gif',path))
 
    frame = cv2.imread(str(path)+"/"+str(0)+'.png')
    height, width, layers = frame.shape

    video = cv2.VideoWriter(str(path.parent)+"/"+name+'.mp4', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 30, (width,height))

    for image in filenames:
        video.write(cv2.imread(image))
    
    video.release()
    # with imageio.get_writer(str(path.parent)+"/"+name+'.gif', mode='I',fps = 30) as writer:
    #     for filename in filenames:
    #         image = imageio.imread(filename)
    #         writer.append_data(image)
        
    # Remove files
    for filename in set(filenames):
        os.remove(filename)

if __name__ == '__main__':
    path = Path("./")
    rootdir='./'
    for p in path.rglob("*"):
        # if '.csv' in p.name and 'g1' in str(p) and 'gif' not in str(p) and 'jayesh' in str(p):
        # ['Bhuj','Sahil_new','Sabyasachi','Chitresh-2','Abhishek','Sahil','Apte','Kamal','Jayesh','chitresh','Manideep']
        
        folder = ['Devendra1','Devendra2','Devendra3','Devendra4','Devendra5','Devendra6','Devendra7']+['Bhuj','Sahil_new','Sabyasachi']+['Chitresh-2','Abhishek']+['Sahil','Apte'] +['Kamal','Jayesh','chitresh','Manideep']
        # if not any(  p.name in s for s in folder):
        #     continue

        if '.py' not in p.name and not 'stick_fig' in str(p) and os.path.isfile(p) and any([s in str(p) for s  in folder]):
            # import pdb;pdb.set_trace()
            # if 'hand' not in str(p):
            #     continue
            path = Path('Subject/stick_figs_no_norm_LOWESS'+str(LOWESS)+'/',str(p))
            if Path.exists(path):
                print(path," skipped")
                continue
            path.mkdir(parents=True, exist_ok=True)
            print(p,path)
            df = pd.read_csv(p)
            draw_fig(df,path,p.name)
    # file_list = [f for f in path.glob('**/*') if f.is_file()]
    # print(file_list)

