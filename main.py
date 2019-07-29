import io
import sys
import cv2
import copy
import numpy as np
import time
from mht import MHTTracker

DEBUGGING = True

def fakeDetections(imwd, imht, time, obstacle, p_D=0.92, m_FA=3):
    obj_wd = 35.0
    obj_ht = 35.0
    size_std = 10.0
    m_FA = m_FA
    stride = 15
    p_FA = m_FA*stride*stride/(imwd*imht) # numer of FAs per area and per scan
    p_D = p_D
    precision_D = 2.0
    margin = 3
    
    N_objects = 4
    obj_m = {1:((-margin, imht*0.5),(imwd+margin, imht*0.5)), 2:((-margin, imht*0.2),(imwd+margin, imht*0.8)),
                3:((-margin, imht*0.8),(imwd+margin, imht*0.2))}
    fp_patch = []
    fa_set = []
    tp_patch = []
    tp_set = []
    for h in range(0, imht, stride):
        for w in range(0, imwd, stride):
            if np.random.random_sample() < p_FA:
                cx, cy = w, h
                sz = np.fabs(np.random.normal([obj_wd, obj_ht], size_std)) + 2
                sz = np.around(sz)
                x1 = round(cx-sz[0]/2)
                y1 = round(cy-sz[1]/2)
                x1 = 0 if x1 < 0 else x1
                y1 = 0 if y1 < 0 else y1
                sz[0] = imwd-x1-1 if x1+sz[0] >= imwd else sz[0]
                sz[1] = imht-y1-1 if y1+sz[1] >= imht else sz[1]
                conf = np.random.random_sample()
                fa_set.append([x1, y1, sz[0], sz[1], conf, -1])
                fa_im = np.random.random_sample((int(sz[1]),int(sz[0])))*255
                fa_im = fa_im.astype(np.uint8)
                fp_patch.append(fa_im)
    
    for o in obj_m:
        s = obj_m[o][0]
        f = obj_m[o][1]
        cx = (f[0]-s[0]) * time + s[0]
        cy = (f[1]-s[1]) * time + s[1]
        cx = cx + np.random.normal(0, precision_D)
        cy = cy + np.random.normal(0, precision_D)
        sz = np.fabs(np.random.normal([obj_wd, obj_ht], size_std/2)) + 2
        sz = np.around(sz)
        if cx < 0 or cx >= imwd or cy < 0 or cy >= imht:
            continue
        if cx >= obstacle[0] and cx <= obstacle[0]+obstacle[2] and cy >= obstacle[1] and cy <= obstacle[1]+obstacle[3]:
            continue
        x1 = round( cx - sz[0]/2 )
        y1 = round( cy - sz[1]/2 )
        x1 = 0 if x1 < 0 else x1
        y1 = 0 if y1 < 0 else y1
        sz[0] = imwd-x1-1 if x1+sz[0] >= imwd else sz[0]
        sz[1] = imht-y1-1 if y1+sz[1] >= imht else sz[1]
        if np.random.random_sample() < p_D:
            conf = np.random.normal(0.8, 0.2)
            if conf > 0.99: conf = 0.995
            tp_set.append([x1, y1, sz[0], sz[1], conf, o])
        else:
            tp_set.append([x1, y1, sz[0], sz[1], -1, o])
        mu = o/N_objects
        std = 1/(N_objects*3)
        tp_im = np.random.normal(mu, std, (int(sz[1]), int(sz[0])))
        tp_im = np.clip(tp_im, a_min=0, a_max=1)*255
        tp_im = tp_im.astype(np.uint8)
        tp_patch.append(tp_im)
    
    if True:
        laps = 3
        myid = len(obj_m)+1
        radius = min([imwd,imht])/2.5
        t = time*np.pi*2*laps
        cx = radius*np.cos(-t**1.0) + imwd/2
        cy = radius*np.sin(-t**1.0) + imht/2
        sz = np.fabs(np.random.normal([obj_wd, obj_ht], size_std/2)) + 2
        sz = np.around(sz)
        if not (cx >= obstacle[0] and cx <= obstacle[0]+obstacle[2] and cy >= obstacle[1] and cy <= obstacle[1]+obstacle[3]):
            x1 = round( cx - sz[0]/2 )
            y1 = round( cy - sz[1]/2 )
            x1 = 0 if x1 < 0 else x1
            y1 = 0 if y1 < 0 else y1
            sz[0] = imwd-x1-1 if x1+sz[0] >= imwd else sz[0]
            sz[1] = imht-y1-1 if y1+sz[1] >= imht else sz[1]
            if np.random.random_sample() < p_D :
                tp_set.append([x1, y1, sz[0], sz[1], np.random.normal(0.8, 0.2), myid])
            else:
                tp_set.append([x1, y1, sz[0], sz[1], -1, myid])
            mu = myid/N_objects
            std = 1/(N_objects*3)
            tp_im = np.random.normal(mu, std, (int(sz[1]), int(sz[0])))
            tp_im = np.clip(tp_im, a_min=0, a_max=1)*255
            tp_im = tp_im.astype(np.uint8)
            tp_patch.append(tp_im)
    return tp_set, fa_set, tp_patch, fp_patch

def compute_similarity(detections, tracks, width, height):
    wd = width
    ht = height
    sim = {}
    for f1 in detections:
        im1 = detections[f1]['app']
        im1 = cv2.resize(im1, (wd, ht))
        for f2 in tracks:
            im2 = tracks[f2]['app']
            im2 = cv2.resize(im2, (wd, ht))
            sim[(f1,f2)] = np.exp(-((im1/255.0-im2/255.0)**2).sum()/(2*ht))
    return sim

if __name__ == '__main__':
    im_width = 800
    im_height = 600
    patch_wd = 50
    patch_ht = 50
    end_of_the_world = 800
    obstacle = (600, 275, 70, 50)
    obstacle_color = (200, 200, 200)
    np.random.seed(20190523)

    cv2.namedWindow('canvas')
    key = 0
    nframe = 0

    params = {'K':10, 'init_score':1.2, 'kalman_constant_noise':False, 'kalman_Q_xy':0.15, 'kalman_Q_vel':0.035, 'kalman_R':0.15,
        'P_D':0.9, 'P_FA':0.001, 'kin_null':0.2, 'distance_threshold':6, 'canonical_kin_prob':False, 'max_scale_change':2,
        'appearance_weight':0.7, 'app_null':0.2, 'max_missing':100, 'min_track_quality':0.5, 'min_track_length':20, 'max_num_leaves':8,
        'use_gurobi':True, 'min_det_conf':-1.0}
    mht = MHTTracker(params)
    while key != 27 and nframe <= end_of_the_world:
        print('\nframe: {}'.format(nframe))
        canvas = np.zeros((im_height, im_width,3), np.uint8)
        cv2.rectangle(canvas, (obstacle[0],obstacle[1]), (obstacle[0]+obstacle[2],obstacle[1]+obstacle[3]), obstacle_color, -1)
        
        tp_set, fp_set, tp_patch, fp_patch = fakeDetections(im_width, im_height, nframe/end_of_the_world, obstacle, p_D=0.8, m_FA=3)
        detections = {}

        # draw detection results
        counter = 1
        for i, fp in enumerate( fp_set ):
            l, t = int(round(fp[0])), int(round(fp[1]))
            r, b = int(round(fp[0]+fp[2])), int(round(fp[1]+fp[3]))
            for c in range(3):
                canvas[t:b,l:r,c] = fp_patch[i]
            cv2.rectangle(canvas, (l, t), (r, b), (0,0,255), 2)
            detections[counter] = {'det':fp[:-1]+[nframe, counter, 0], 'app':fp_patch[i]}
            counter+=1
        for i, tp in enumerate(tp_set):
            l, t = int(round(tp[0])), int(round(tp[1]))
            r, b = int(round(tp[0]+tp[2])), int(round(tp[1]+tp[3]))
            if tp[4] != -1:
                for c in range(3):
                    canvas[t:b,l:r,c] = tp_patch[i]
                cv2.rectangle(canvas, (l, t), (r, b), (255,0,0), 2)
                detections[counter] = {'det':tp[:-1]+[nframe, counter, 0], 'app':tp_patch[i]}
                counter+=1
            else:
                pass
                #cv2.rectangle(canvas, (l, t), (r, b), (0,255,255), 2)
        features, _ = mht.getTrackPatches()
        sim = compute_similarity(detections, features, patch_wd, patch_ht)
        tracks = mht.doTracking(nframe, detections, sim, canvas)
        for t in tracks:
            item = tracks[t]
            anode = mht.hypothesis_set[item[0]].nodes[item[1]]
            adet = anode.detection
            xy1 = anode.kalman_state.copy()
            #cv2.putText(canvas, str(t), (int(round(xy1[0,0])), int(round(xy1[1,0]))), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255,0,255), 2)
            #assert adet[5] == nframe, 'Check'
            assert anode.is_leaf, 'Check 2'
            route = anode.path
            #fileout.write('{}, {}, {}, {}\n'.format(nframe, anode.track_id, xy1[0,0], xy1[1,0]))
            det_id = anode.det_index[-1]
            assert len(anode.det_index) == anode.status[1]
            if anode.is_dummy == False and DEBUGGING:
                assert np.allclose(mht.dets_set[det_id[0]][det_id[1]]['app'], detections[det_id[1]]['app'] )
                assert mht.dets_set[det_id[0]][det_id[1]]['det'] == detections[det_id[1]]['det']
            for p in reversed(route):
                if p == anode:
                    continue
                xy2 = p.kalman_state.copy()
                if p.status[4] == 0:
                    cv2.line(canvas, (int(round(xy1[0,0])), int(round(xy1[1,0])) ), 
                        (int(round(xy2[0,0])), int(round(xy2[1,0])) ),(255,0,255), 2)
                    assert p.detection[7] == 0, 'Check 3'
                else:
                    cv2.line(canvas, (int(round(xy1[0,0])), int(round(xy1[1,0])) ),
                        (int(round(xy2[0,0])), int(round(xy2[1,0])) ), (255,255,0), 2)
                    assert p.detection[7] == 1, 'Check 3'
                xy1 = xy2

        cv2.imshow('canvas', canvas)
        key = cv2.waitKey(1)
        nframe+=1
    mht.concludeTracks()
    
    print('\nResutls:')
    for t in mht.confirmed_tracks:
        canvas2 = np.zeros((im_height, im_width,3), np.uint8)
        cv2.rectangle(canvas2, (obstacle[0],obstacle[1]), (obstacle[0]+obstacle[2],obstacle[1]+obstacle[3]), obstacle_color, -1)
        track = mht.confirmed_tracks[t]
        colorid = cv2.applyColorMap( np.array([(t * 32) % 256], dtype=np.uint8 ), cv2.COLORMAP_HSV )
        colors = (int(colorid[0,0,0]), int(colorid[0,0,1]), int(colorid[0,0,2]))
        for l in range(1,len(track)):
            t1 = track[l-1]
            t2 = track[l]
            x1, y1 = int(round(t1[1]+t1[3]/2)), int(round(t1[2]+t1[4]/2))
            x2, y2 = int(round(t2[1]+t2[3]/2)), int(round(t2[2]+t2[4]/2))
            
            dummy = t2[5]
            if dummy == 0:
                cv2.line(canvas2, (x1, y1), (x2, y2), colors, 2)
            else:
                cv2.line(canvas2, (x1, y1), (x2, y2), (int(colors[0]/2),int(colors[1]/2),int(colors[2]/2)), 1)
        x1, y1 = int(round(track[-1][1] + track[-1][3]/2)), int(round(track[-1][2] + track[-1][4]/2))
        time_end = int(round(track[-1][0]))
        cv2.putText(canvas2, str(t), (x1, y1-25), cv2.FONT_HERSHEY_COMPLEX, .8, colors, 2)
        cv2.putText(canvas2, 'end:'+str(time_end), (x1, y1), cv2.FONT_HERSHEY_COMPLEX, .5, colors, 2)
        x1, y1 = int(round(track[0][1] + track[0][3]/2)), int(round(track[0][2] + track[0][4]/2))
        time_start = int(round(track[0][0]))
        cv2.putText(canvas2, 'start:'+str(time_start), (x1, y1), cv2.FONT_HERSHEY_COMPLEX, .5, colors, 2)
        cv2.imshow('ObjectID_{}'.format(t), canvas2)
        print('\tObjectID:{}, start:{}, end:{}, length:{}'.format(t, time_start, time_end, len(track)))
    cv2.waitKey()
    print('finished.')
    