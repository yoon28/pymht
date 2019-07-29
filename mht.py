import io
import sys
import cv2
import copy
import numpy as np
from anytree import Node, RenderTree, render
from anytree.search import findall
import gurobipy as grb
import cvxpy, cvxopt
import time

DEBUGGING = True

class TrackTree():
    INIT_NODE = 1
    STATUS = {'tracking':0, 'end':1, 'purge':2}
    def __init__(self, treeNum, detection, init_score, track_id, P_init):
        self.nodes = dict()
        self.history = {'dets':[], 'estimates':[]}
        self.v_num = self.INIT_NODE # Tree is initialized with INIT_NODE
        self.root_num = self.INIT_NODE # save current root node number
        self.treeNum = treeNum
        self.valid_track = [-1, -1, -1, -1] # indicator, node num, track id, score
        self.nodes[self.v_num] = Node('{}({})'.format(self.v_num, treeNum), parent=None)
        self.nodes[self.v_num].detection = detection # (x, y, w, h, b, t, i, dummy) , b=confidence, t=frame, i=i-th detection at the frame, dummy=dummy indicator
        self.nodes[self.v_num].is_dummy = False
        self.nodes[self.v_num].det_index = [(detection[5], detection[6])]
        self.nodes[self.v_num].scores = [init_score, 0, 0, 0] # score, app_score, st_score, detection confidence
        self.nodes[self.v_num].status = [1, 1, 0, 0, 0, self.STATUS['tracking']] # [ total_length, num_obs, num_totoal_missing, num_conseq_missing, dummy_node_indicator, status ]
        self.nodes[self.v_num].kalman_state = np.array([ [detection[0]+detection[2]/2], [detection[1]+detection[3]/2], [0], [0]]) # cx, cy, vx, vy
        self.nodes[self.v_num].kalman_cov = P_init
        self.nodes[self.v_num].track_id = track_id
        self.nodes[self.v_num].v_num = self.v_num
        self.incrementVertexNum()

    def addNode(self, node_info, parent_node):
        if DEBUGGING: assert self.v_num not in self.nodes, 'fatal error: node num'
        self.nodes[self.v_num] = Node('{}({})'.format(self.v_num, self.treeNum), parent=parent_node)
        self.nodes[self.v_num].detection = node_info['detection']
        self.nodes[self.v_num].is_dummy = node_info['is_dummy']
        self.nodes[self.v_num].det_index = node_info['det_index']
        self.nodes[self.v_num].scores = node_info['scores']
        self.nodes[self.v_num].status = node_info['status']
        self.nodes[self.v_num].kalman_state = node_info['kalman_state']
        self.nodes[self.v_num].kalman_cov = node_info['kalman_cov']
        self.nodes[self.v_num].track_id = node_info['track_id']
        self.nodes[self.v_num].v_num = self.v_num
        return self.getVertexNum_and_Increment()
    
    def getParent(self, node_idx):
        p = self.nodes[node_idx].parent
        if p is None:
            return None
        else:
            return p.v_num

    def getChildren(self, node_idx):
        children = [c.v_num for c in self.nodes[node_idx].children]
        return children
    
    def getNode(self, node_idx):
        return self.nodes[node_idx]

    def findLeaves(self):
        leaves = [l.v_num for l in self.nodes[self.root_num].leaves]
        return leaves
    
    def getRoot(self):
        return self.root_num
    
    def removeBranch(self, node_idx):
        assert self.nodes[node_idx].is_leaf
        root = self.nodes[node_idx].root
        path = self.nodes[node_idx].path
        for p in reversed(path):
            if not self.nodes[p.v_num].is_leaf:
                break
            self.nodes[p.v_num].parent = None
            del self.nodes[p.v_num]
    
    def detachSubTree(self, new_root):
        if self.nodes[new_root].is_root:
            return
        
        path = self.nodes[new_root].path
        for p in path:
            if p.v_num == new_root:
                continue
            self.history['dets'].append(tuple(p.detection))
            kstate = (p.kalman_state[0,0], p.kalman_state[1,0])
            self.history['estimates'].append(kstate)

        prunedLeaves = []
        allNodesIdx = list(self.nodes.keys())
        descendIdx = { d.v_num for d in self.nodes[new_root].descendants }
        for n in allNodesIdx:
            if not(n in descendIdx) and n != new_root:
                if self.nodes[n].is_leaf:
                    prunedLeaves.append(n)
                self.nodes[n].parent = None
                del self.nodes[n]
        self.nodes[new_root].parent = None
        self.root_num = new_root
        return prunedLeaves

    def incrementVertexNum(self):
        self.v_num += 1
    
    def getVertexNum_and_Increment(self):
        self.v_num += 1
        return self.v_num - 1

class MHTTracker():
    def __init__(self, parameters):
        self.id_pool = 1 # give new ID 1) after updated with a det, 2) init new track with a det, 
        self.tree_num = 1
        self.hypothesis_set = dict()
        self.confirmed_tracks = dict()
        self.conflictList = dict()
        self.dets_set = dict()
        self.eps = 1e-10

        self.min_det_conf = parameters['min_det_conf']
        self.max_scale_change = parameters['max_scale_change']
        self.use_denom = parameters['canonical_kin_prob']
        self.use_gurobi = parameters['use_gurobi']
        self.max_num_leaves = parameters['max_num_leaves']
        self.min_track_length = parameters['min_track_length']
        self.K = parameters['K']
        self.init_score = parameters['init_score']
        self.P_D = parameters['P_D']
        self.P_FA = parameters['P_FA'] # false alarms per area
        self.d_th = parameters['distance_threshold']
        self.kin_null = parameters['kin_null']
        self.max_missing = parameters['max_missing']
        self.w_appearance = parameters['appearance_weight']
        self.app_null = parameters['app_null']
        self.w_motion = 1 - self.w_appearance
        self.min_track_quality = parameters['min_track_quality']
        self.kalman_const_noise = parameters['kalman_constant_noise']
        self.kalman_cov_xy = parameters['kalman_Q_xy'] # covariance is a function of width of an object
        self.kalman_cov_vel = parameters['kalman_Q_vel'] # covariance is a function of width of an object
        self.kalman_R = parameters['kalman_R'] # observation noise
        self.kalman_F = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]])
        self.kalman_H = np.array([[1,0,0,0],[0,1,0,0]])
        #self.noise_R = np.diag([self.kalman_R**2, self.kalman_R**2])
        if self.kalman_const_noise:
            self.noise_Q = np.diag([self.kalman_cov_xy**2, self.kalman_cov_xy**2, 
                self.kalman_cov_vel**2, self.kalman_cov_vel**2])
    
    def getKalmanNoiseR(self, size=None):
        if self.kalman_const_noise:
            return np.diag([self.kalman_R**2, self.kalman_R**2])
        if size > 0 and self.kalman_const_noise==False:
            return np.diag([ (self.kalman_R*size)**2, (self.kalman_R*size)**2 ])
        assert False

    def getKalmanNoiseQ(self, size=None, init_P=False):
        if init_P and self.kalman_const_noise:
            return np.diag([self.kalman_cov_xy**2, self.kalman_cov_xy**2, 
                self.kalman_cov_xy**2, self.kalman_cov_xy**2])
        if init_P and size > 0 and self.kalman_const_noise==False:
            return np.diag([ (self.kalman_cov_xy*size)**2, (self.kalman_cov_xy*size)**2,
                (self.kalman_cov_xy*size)**2, (self.kalman_cov_xy*size)**2 ])
        if self.kalman_const_noise and init_P==False:
            return self.noise_Q
        if size > 0 and self.kalman_const_noise==False and init_P==False:
            return np.diag([ (self.kalman_cov_xy*size)**2, (self.kalman_cov_xy*size)**2,
                (self.kalman_cov_vel*size)**2, (self.kalman_cov_vel*size)**2 ])
        assert False

    def incrementID(self):
        self.id_pool += 1
    
    def getID_and_Increment(self):
        self.id_pool += 1
        return self.id_pool - 1
    
    def incrementTreeNum(self):
        self.tree_num += 1
    
    def getTreeNum_and_Increment(self):
        self.tree_num += 1
        return self.tree_num - 1
    
    def multivariateNormalProb(self, x, mean, cov, ln_output=False, inv_cov=None):
        d_2 = x.shape[0] * 0.5
        if self.use_denom:
            denom = np.power(2*np.pi, d_2)*np.sqrt(np.abs(np.linalg.det(cov)))
        else:
            denom = 1
        if inv_cov is None:
            exponent = (x-mean).T @ np.linalg.inv(cov) @ (x-mean)
        else:
            exponent = (x-mean).T @ inv_cov @ (x-mean)
        assert exponent >= 0, 'covariance is not PSD 1'

        if ln_output:
            prob = -0.5*exponent - np.log(denom)
        else:
            prob = np.exp(-0.5*exponent) / (denom+self.eps)
        return prob.item()

    def copyNodeInfo(self, node):
        info = {'kalman_state':node.kalman_state.copy(), 'kalman_cov':node.kalman_cov.copy(),
                'detection':list(node.detection), 'det_index':list(node.det_index), 'is_dummy':node.is_dummy,
                'scores':list(node.scores), 'status':list(node.status),
                'track_id':node.track_id, 'v_num':node.v_num}
        return info

    def kalman_predict(self, X, P, size):
        Q = self.getKalmanNoiseQ(size=size)
        X_pred = self.kalman_F @ X
        P_pred = self.kalman_F @ P @ self.kalman_F.T + Q
        return X_pred, P_pred

    def addDummyNode(self, current_node, atree, nframe):
        info = self.copyNodeInfo(current_node)
        info['is_dummy'] = True
        if info['status'][3] > self.max_missing:
            current_node.status[5] = TrackTree.STATUS['end'] # status
            return -1
        info['status'][0] += 1 # total length
        info['status'][2] += 1 # total_missing
        info['status'][3] += 1 # conseq_missing
        info['status'][4] = 1 # dummy_indicator
        info['detection'][5] = nframe # frame
        #info['detection'][6] = -1 # dummy_indicator
        info['detection'][7] = 1 # dummy_indicator
        
        # state, cov, scores
        size = info['detection'][2]
        info['kalman_state'] = self.kalman_F @ info['kalman_state']
        Q = self.getKalmanNoiseQ(size=size)
        R = self.getKalmanNoiseR(size=size) # self.noise_R
        P_pred = self.kalman_F @ info['kalman_cov'] @ self.kalman_F.T + Q
        IS = np.linalg.inv(R + self.kalman_H @ P_pred @ self.kalman_H.T)
        K = P_pred @ self.kalman_H.T @ IS
        IKH = np.eye(K.shape[0]) - K @ self.kalman_H
        info['kalman_cov'] = IKH @ P_pred @ IKH.T + K @ R @ K.T
        
        # info['kalman_cov'] = Q*2.0 # P_pred.copy()
        # info['kalman_state'], info['kalman_cov'] = self.kalman_predict(info['kalman_state'], info['kalman_cov'], info['detection'][2])
        
        info['scores'] = [current_node.scores[0] + np.log(1-self.P_D), 0, np.log(1-self.P_D), current_node.scores[3]]
        # if info['scores'][0] < 0:
        #     info['scores'][0] = 0
        v_num = atree.addNode(info, current_node)
        return v_num

    def updateNodeWithDetection(self, adet, current_node, atree,
                    X_predict, P_predict, Kalman_innovation, Kalman_gain, Kalman_S, Kalman_IS, Noise_R, app_score=None):
        new_info = self.copyNodeInfo(current_node)

        new_info['status'][0] += 1 # total length
        new_info['status'][1] += 1 # num dets
        new_info['status'][3] = 0 # reset conseq missings
        new_info['status'][4] = 0 # indicator dummy
        det_bbox = list(adet)

        if max([ det_bbox[3] / current_node.detection[3], 
            current_node.detection[3] / det_bbox[3] ]) > self.max_scale_change: # scale gating
            del new_info
            return -1, -1
        if app_score <= 0:
            del new_info
            return -1, -1
        #det_bbox[2] = current_node.detection[2] * 0.5 + det_bbox[2] * 0.5
        #det_bbox[3] = current_node.detection[3] * 0.5 + det_bbox[3] * 0.5
        new_info['detection'] = det_bbox
        new_info['det_index'].append((adet[5], adet[6])) # detection index
        new_info['is_dummy'] = False
        
        X_new = X_predict + Kalman_gain @ Kalman_innovation
        IKH = np.eye(Kalman_gain.shape[0]) - Kalman_gain @ self.kalman_H
        P_new = IKH @ P_predict @ IKH.T + Kalman_gain @ Noise_R @ Kalman_gain.T
        # P_new = (np.eye(P_predict.shape[0]) - Kalman_gain @ self.kalman_H) @ P_predict
        new_info['kalman_state'] = X_new
        new_info['kalman_cov'] = P_new
        #lnlh_kinematic = self.multivariateNormalProb(Kalman_innovation, np.zeros(Kalman_innovation.shape), Kalman_S, True, Kalman_IS) - np.log(self.P_FA)
        motion_term = self.multivariateNormalProb(Kalman_innovation, np.zeros(Kalman_innovation.shape), cov=Kalman_S, 
            ln_output=True, inv_cov=Kalman_IS) - np.log(self.kin_null)
        #print(self.multivariateNormalProb(np.zeros((2,1)), np.zeros(Kalman_innovation.shape), Kalman_S, True, Kalman_IS))
        if app_score == None:
            appearance_term = np.log(self.P_D/self.P_FA)
        else:
            appearance_term = np.log(app_score) - np.log(self.app_null) # np.log(self.P_D/self.P_FA) #
        llh = self.w_appearance*appearance_term + self.w_motion*motion_term

        if llh <= 0: # no update
            del new_info
            return -1, -1
        sc = current_node.scores[0] + llh
        det_score = current_node.scores[3] + adet[4]
        new_info['scores'] = [sc, self.w_appearance*appearance_term, self.w_motion*motion_term, det_score]

        new_id = self.getID_and_Increment() # return new id
        new_info['track_id'] = new_id
        v_num = atree.addNode(new_info, current_node)
        return v_num, new_id

    def compDistAll(self, x, y, x0, y0, ivcov):
        a, b, c, d = ivcov[0,0], ivcov[0,1], ivcov[1,0], ivcov[1,1]
        xx = x-x0
        yy = y-y0
        return a*((xx)**2) + d*((yy)**2) + (b+c)*(xx*yy)

    def updateTrackTrees(self, nframe, dets, app_scores=None, canvas=None):
        det_usage = { d:[] for d in dets } # usage list of each detection

        # loop over track trees
        for ti in self.hypothesis_set:
            atree = self.hypothesis_set[ti]
            leaves = atree.findLeaves()
            
            for l in leaves:
                leaf = atree.nodes[l]

                if leaf.status[5] == TrackTree.STATUS['purge'] or leaf.status[5] == TrackTree.STATUS['end']:
                    continue

                # add a dummy node
                if self.addDummyNode(leaf, atree, nframe) == -1:
                    continue
                
                # compute the prediction step of kalman filter
                size = leaf.detection[2] # width
                X_prior = leaf.kalman_state
                P_prior = leaf.kalman_cov
                X_predict, P_predict = self.kalman_predict(X_prior, P_prior, size)

                # kalman correction
                R = self.getKalmanNoiseR(size=size) # self.noise_R
                S = (self.kalman_H @ P_predict @ self.kalman_H.T) + R
                IS = np.linalg.inv(S)
                K = P_predict @ self.kalman_H.T @ IS

                # vis
                if canvas is not None: # nframe > 0:
                    t_xy = X_predict[:2] # X_prior[:2]
                    im_wd = canvas.shape[1]
                    im_ht = canvas.shape[0]
                    strides = 17
                    for yy in range(0,im_ht,strides):
                        for xx in range(0,im_wd,strides):
                            xyxy = np.array([[xx], [yy]])
                            dt = (xyxy-t_xy).T @ IS @ (xyxy-t_xy)
                            dt = dt.item()
                            if dt < self.d_th:
                                cv2.circle(canvas, (xx, yy), 2, (0,0,255), -1)
                    cv2.circle(canvas, ( int(np.round(t_xy[0,0])), int(np.round(t_xy[1,0])) ), 2, (0,255,0), -2)
                    #cv2.imshow('cvs', canvas)

                for d in dets:
                    det_xy = np.array([[dets[d]['det'][0]+dets[d]['det'][2]/2], [dets[d]['det'][1]+dets[d]['det'][3]/2]])
                    Y = det_xy - self.kalman_H @ X_predict
                    distance = Y.T @ IS @ Y
                    distance = distance.item()
                    if DEBUGGING: assert distance >= 0, 'Fatal error: covariance is not PSD 2'
                    
                    if distance < self.d_th: # gating
                        adet = dets[d]['det']  # (x, y, w, h, b, t, i, dummy) , b=confidence, t=frame, i=i-th detection at the frame
                        if app_scores != None:
                            v_num, new_id = self.updateNodeWithDetection(adet, leaf, atree, X_predict, P_predict, Y, K, S, IS, R, app_scores[(adet[6], leaf.det_index[-1])])
                        else:
                            v_num, new_id = self.updateNodeWithDetection(adet, leaf, atree, X_predict, P_predict, Y, K, S, IS, R)
                        if v_num != -1 and new_id != -1:
                            det_usage[d].append((ti, v_num, new_id))
        
        # init new tracks
        for d in dets:
            adet = list(dets[d]['det'])  # (x, y, w, h, b, t, i, dummy) , b=confidence, t=frame, i=i-th detection at the frame
            P_init = self.getKalmanNoiseQ(size=adet[2], init_P=False) # np.zeros((4,4))
            atree = TrackTree(self.tree_num, adet, self.init_score, self.id_pool, P_init)
            det_usage[d].append((self.tree_num, TrackTree.INIT_NODE, self.id_pool)) # tree_num, node_num, id
            if DEBUGGING: assert self.tree_num not in self.hypothesis_set, 'Fatal error: tree'
            self.hypothesis_set[self.tree_num] = atree
            self.incrementID()
            self.incrementTreeNum()

        if DEBUGGING:
            valid_tracks = 0
            totLeaves = 0 # check code
            max_leaves = [-1,-1]
            for ti in self.hypothesis_set:
                if self.hypothesis_set[ti].valid_track[0] == 1:
                    valid_tracks += 1
                leaves = self.hypothesis_set[ti].findLeaves()
                totLeaves += len(leaves)
                if max_leaves[0] < len(leaves):
                    max_leaves[0] = len(leaves)
                    max_leaves[1] = ti
            #if max_leaves[0] != -1 and max_leaves[1] != -1:
            #    print(RenderTree(self.hypothesis_set[max_leaves[1]].nodes[self.hypothesis_set[max_leaves[1]].findRoot().v_num]).by_attr())
            print('\nNUM ({}): {}/{}/{}'.format(valid_tracks,len(self.hypothesis_set), max_leaves[0], totLeaves))
        return det_usage
    
    def makeConflictList(self, det_usage):
        n_tree = len(self.hypothesis_set)
        if n_tree == 0:
            self.conflictList.clear()
            return
        
        conflictPrev = copy.deepcopy(self.conflictList)
        self.conflictList.clear()
        self.conflictList = {t:None for t in self.hypothesis_set}
        
        for t in self.hypothesis_set:
            atree = self.hypothesis_set[t]
            leaves = atree.findLeaves()
            self.conflictList[t] = dict()
            for l in leaves:
                leaf = atree.getNode(l)
                cflcts = []

                # conflicts from the parent node
                # if leaf.status[5] != TrackTree.STATUS['end']:
                #     parent = atree.getParent(l)
                # else:
                #     parent = l
                if leaf.status[5] == TrackTree.STATUS['tracking']:
                    parent = atree.getParent(l)
                else:
                    parent = l
                    if DEBUGGING: assert atree.nodes[l].is_leaf

                if DEBUGGING: assert leaf.status[5] != TrackTree.STATUS['purge']
                if parent is None: # root node
                    cflcts.append((t, l, leaf.track_id)) # tree_num, node_num, track_id
                else:
                    cflctParent = conflictPrev[t][parent]
                    for acflct in cflctParent:
                        if DEBUGGING: assert self.hypothesis_set[acflct[0]].nodes[acflct[1]].track_id == acflct[2], 'Fatal error: ID does not match'

                        # if self.hypothesis_set[acflct[0]].nodes[acflct[1]].status[5] == TrackTree.STATUS['end']:
                        #     cflcts.append((acflct[0], acflct[1], self.hypothesis_set[acflct[0]].nodes[acflct[1]].track_id))
                        #     assert self.hypothesis_set[acflct[0]].nodes[acflct[1]].is_leaf, 'Fatal error: not a leaf'
                        #     continue

                        cflct_ch = self.hypothesis_set[acflct[0]].getChildren(acflct[1])
                        if len(cflct_ch) > 0:
                            for c in cflct_ch:
                                if DEBUGGING: assert self.hypothesis_set[acflct[0]].nodes[c].is_leaf, 'Fatal error: not a leaf'
                                if DEBUGGING: assert self.hypothesis_set[acflct[0]].nodes[c].status[5] != TrackTree.STATUS['purge']
                                cflcts.append((acflct[0], c, self.hypothesis_set[acflct[0]].nodes[c].track_id))
                        else:
                            cflcts.append((acflct[0], acflct[1], self.hypothesis_set[acflct[0]].nodes[acflct[1]].track_id))
        
                # check the confliction of current detections
                if leaf.status[4] == 0: # not a dummy node
                    det_i = leaf.detection[6]
                    cflcts = cflcts + det_usage[det_i]
                    if DEBUGGING: assert leaf.detection[7] == 0
                self.conflictList[t][l] = set(cflcts)

    def clustering(self):
        if len(self.hypothesis_set) == 0:
            return dict()

        conflictTreeList = {t:None for t in self.hypothesis_set}
        for t in self.hypothesis_set:
            leaves = self.hypothesis_set[t].findLeaves()
            cflTrees = set()
            for l in leaves:
                for k in self.conflictList[t][l]:
                    cflTrees.add(k[0])
            conflictTreeList[t] = list(cflTrees)
        
        conflictTrees = {t:[] for t in self.hypothesis_set}
        it = [k for k in sorted(self.hypothesis_set)]
        for i in range(len(it)):
            t = it[i]
            conflictTrees[t].append(t)
            conflicts_1 = conflictTreeList[t]
            for j in range(i+1, len(it)):
                t2 = it[j]
                conflicts_2 = conflictTreeList[t2]
                for f in conflicts_2:
                    if f in conflicts_1:
                        conflictTrees[t].append(t2)
                        break
            assert len(conflictTrees[t]) == len(set(conflictTrees[t])), 'error check'
        
        category = 0
        tree_category = {t:0 for t in self.hypothesis_set}
        while len(it) > 0:
            category += 1
            i = it[0]
            cflTrees = conflictTrees[i]
            for j in cflTrees:
                if i == j: continue
                conflictTrees[i] = conflictTrees[i] + conflictTrees[j]
                conflictTrees[j] = []
            
            conflictTrees[i] = list(set(conflictTrees[i]))
            it = sorted(list(set(it) - set(cflTrees)))

            cat_cfl = []
            for k in conflictTrees[i]:
                if tree_category[k] != 0:
                    cat_cfl.append(tree_category[k])
            
            if len(cat_cfl) == 0:
                for k in conflictTrees[i]:
                    tree_category[k] = category
            else:
                for k in tree_category:
                    if tree_category[k] in cat_cfl:
                        tree_category[k] = category
                for k in conflictTrees[i]:
                    tree_category[k] = category
        
        clusters = {k:[] for k in set(tree_category.values())}
        for c in clusters:
            trees = [k for k in tree_category if tree_category[k]==c]
            for t in trees:
                leaves = self.hypothesis_set[t].findLeaves()
                for l in leaves:
                    clusters[c].append((t, l, self.hypothesis_set[t].nodes[l].track_id)) # tree_no, vertex_no, track_id
        
        if DEBUGGING: # sanity check
            cl = list(clusters.keys())
            for i in range(len(cl)):
                ci = cl[i]
                list_ci = clusters[ci]
                for j in range(i+1, len(cl)):
                    cj = cl[j]
                    if ci == cj: continue
                    list_cj = clusters[cj]
                    intersect = set(list_ci) & set(list_cj)
                    assert len(intersect) == 0, 'error chk 2'
                    
        return clusters
    
    def compBestHypoSet(self, clusters):
        best_set = {k:None for k in clusters}
        
        for c in clusters:
            tracks = clusters[c]
            n_tracks = len(tracks)

            edges = np.zeros((n_tracks, n_tracks))
            weights = np.zeros(n_tracks)
            min_score = 1e12
            for l in range(n_tracks):
                atrack = tracks[l]
                score = list(self.hypothesis_set[atrack[0]].nodes[atrack[1]].scores)
                status = list(self.hypothesis_set[atrack[0]].nodes[atrack[1]].status)
                cfls = self.conflictList[atrack[0]][atrack[1]]
                # cnt = 0
                for l2 in range(l+1, n_tracks):
                    if tracks[l2] in cfls:
                        edges[l, l2] = 1
                #        cnt += 1
                # assert cnt == len(cfls), 'Fatal error: check clusters'
                
                if score[0] < min_score:
                    min_score = score[0]

                track_len = status[0] - status[3]
                if status[1]/track_len > self.min_track_quality and track_len > self.min_track_length:
                    if self.hypothesis_set[atrack[0]].valid_track[0] == 1 and status[5] == TrackTree.STATUS['end']:
                        for _ in range(status[3]): # compensate for tails
                            score[0] -= np.log(1-self.P_D)
                #        if score[0] <= 0:
                #            score[0] = self.init_score
                # if status[1]/track_len > self.min_track_quality and track_len > self.min_track_length:
                #     score[0] += 0.1
                weights[l] = score[0]
            
            #weights = weights - (min_score - 0.1)
            best_scores = []
            if self.use_gurobi:
                gb = grb.Model('bestset')
                x = {}
                for l in range(n_tracks):
                    x[l] = gb.addVar(obj=weights[l], vtype=grb.GRB.BINARY)
                for l in range(n_tracks):
                    for j in range(l+1, n_tracks):
                        if edges[l,j] == 1:
                            gb.addConstr(x[l]+x[j], '<=', 1)
                gb.ModelSense = grb.GRB.MAXIMIZE
                gb.Params.OutputFlag = 0
                gb.update()
                gb.optimize()
                if gb.status == grb.GRB.OPTIMAL:
                    best_set[c] = [tracks[l] for l in x if x[l].x == 1]
                    best_scores = [weights[l] for l in x if x[l].x == 1]
                else:
                    assert False, 'Fatal error: check gurobi solutions'
            else:
                xx = cvxpy.Variable(shape=n_tracks, boolean=True)
                maximize = weights * xx
                constraints = [0<=xx, xx<=1] # a meaningless constrain
                for l in range(n_tracks):
                    for j in range(l+1, n_tracks):
                        if edges[l,j] == 1:
                            constraints.append(xx[l]+xx[j]<=1)
                problem = cvxpy.Problem(cvxpy.Maximize(maximize), constraints)
                problem.solve(solver=cvxpy.ECOS_BB, verbose=False)
                best_set[c] = [tracks[l] for l in range(n_tracks) if xx[l].value >= 0.98]
                best_scores = [weights[l] for l in range(n_tracks) if xx[l].value >= 0.98]
                # result = [tracks[l] for l in range(n_tracks) if xx[l].value >= 0.98]
                # assert len(result) == len(best_set[c])
                # for r in result:
                #     assert r in best_set[c]
            for bi, b in enumerate(best_set[c]):
                status = self.hypothesis_set[b[0]].nodes[b[1]].status
                track_len = status[0]-status[3]
                if status[1]/track_len > self.min_track_quality and track_len > self.min_track_length:
                    self.hypothesis_set[b[0]].valid_track[0] = 1
                    self.hypothesis_set[b[0]].valid_track[1] = b[1] # node num
                    self.hypothesis_set[b[0]].valid_track[2] = b[2] # track id
                    self.hypothesis_set[b[0]].valid_track[3] = best_scores[bi] # score

        if DEBUGGING: # satiny check
            for c in best_set:
                bests = best_set[c]
                for b in bests:
                    cfls = self.conflictList[b[0]][b[1]]

                    for c2 in best_set:
                        bests2 = best_set[c2]
                        for b2 in bests2:
                            if c==c2 and b == b2: continue
                            assert not(b2 in cfls), 'Fatal error: check'
        return best_set

    def treePruning(self, clusters, best_set):
        # K-Depth pruning
        # delete tracks: too many false positives,
        # terminate tracks: track termination (conseq missing)
        bestTracks = dict()
        surviveTracks = set()
        currentTracks = dict()
        for c in clusters:
            for b in best_set[c]:
                bestTracks[b[0]] = (b[1], b[2])

        deathNote = []
        confirmed = []
        treeNums = sorted(self.hypothesis_set.keys())
        for i in range(len(treeNums)):
            t = treeNums[i]

            if t in bestTracks:
                best_node = bestTracks[t]
                # depth pruning
                new_root = best_node[0]
                sel_node = best_node[0]
                sel_id = best_node[1]
                for _ in range(self.K):
                    new_root = self.hypothesis_set[t].getParent(new_root)
                    if new_root == None:
                        break
                if new_root == None or self.hypothesis_set[t].nodes[new_root].is_root:
                    new_root = self.hypothesis_set[t].getRoot()
                else:
                    prunedLeaves = self.hypothesis_set[t].detachSubTree(new_root)
                    for p in prunedLeaves: 
                        del self.conflictList[t][p]
                if self.hypothesis_set[t].valid_track[0] == 1:
                    self.hypothesis_set[t].valid_track = [1, sel_node, sel_id, self.hypothesis_set[t].nodes[sel_node].scores[0]]
            elif self.hypothesis_set[t].valid_track[0] == 1:
                leaves = self.hypothesis_set[t].findLeaves()
                scores = [self.hypothesis_set[t].nodes[l].scores[0] for l in leaves]
                pairs = zip(leaves, scores)
                sortleaf = sorted(pairs, key=lambda x:x[1], reverse=True)
                sel_node = -1
                find_node = self.hypothesis_set[t].valid_track[1]
                for si in range(len(sortleaf)):
                    route = self.hypothesis_set[t].nodes[sortleaf[si][0]].path
                    for r in reversed(route):
                        if r.v_num == find_node:
                            sel_node = sortleaf[si][0]
                            break
                if DEBUGGING: assert sel_node != -1
                sel_id = self.hypothesis_set[t].nodes[sel_node].track_id
                for l in leaves:
                    if l == sel_node:
                        continue
                    self.hypothesis_set[t].removeBranch(l)
                    del self.conflictList[t][l]
                status = self.hypothesis_set[t].nodes[sel_node].status
                track_len = status[0]-status[3]
                if status[1]/track_len > self.min_track_quality and track_len > self.min_track_length:
                    self.hypothesis_set[t].valid_track = [1, sel_node, sel_id, self.hypothesis_set[t].nodes[sel_node].scores[0]]
            else:
                del self.hypothesis_set[t] 
                del self.conflictList[t]
                continue
            
            # record survived track
            leaves = self.hypothesis_set[t].findLeaves()
            tempset = [] # a note for survied leaves 
            # n_bad = 0 # bad tracks among finished tracks
            # n_good = 0 # good tracks among finished tracks
            # n_tracking = 0 # under tracking
            for l in leaves:
                anode = self.hypothesis_set[t].getNode(l)
                if anode.status[5] != TrackTree.STATUS['purge']:
                    tempset.append((t, l, anode.track_id))

            b_status = self.hypothesis_set[t].nodes[sel_node].status
            if b_status[5] == TrackTree.STATUS['end']:
                track_len = b_status[0]-b_status[3]
                quality = b_status[1] / track_len
                if quality > self.min_track_quality and track_len > self.min_track_length:
                    confirmed.append((t, sel_node, sel_id))
                else:
                    deathNote.append(t)
            else:
                surviveTracks = surviveTracks.union(tempset)
                currentTracks[sel_id] = (t, sel_node)
            if DEBUGGING: assert b_status != TrackTree.STATUS['purge']

        for d in deathNote:
            del self.hypothesis_set[d]
            del self.conflictList[d]

        for c in confirmed: # confirming
            self.saveConfirmedTrack(c[0], c[1], c[2])
            del self.hypothesis_set[c[0]]
            del self.conflictList[c[0]]

        # update conflict list
        for t in self.hypothesis_set:
            leaves = self.hypothesis_set[t].findLeaves()
            for l in leaves:
                newcfls = set()
                for c in self.conflictList[t][l]:
                    if c in surviveTracks:
                        newcfls.add(c)
                self.conflictList[t][l] = newcfls

        return currentTracks

    def branchMerging(self, currentTracks):
        
        if DEBUGGING: 
            curr_trees = { currentTracks[t][0] for t in currentTracks }
            assert set(self.hypothesis_set.keys()) == curr_trees

        survived = []
        for t in currentTracks:
            best = currentTracks[t]
            if DEBUGGING: self.hypothesis_set[best[0]].nodes[best[1]].track_id == t
            bestscore = self.hypothesis_set[best[0]].nodes[best[1]].scores[0]
            depth = self.hypothesis_set[best[0]].nodes[best[1]].depth
            leaves = self.hypothesis_set[best[0]].findLeaves()
            scores = [self.hypothesis_set[best[0]].nodes[l].scores[0] for l in leaves]
            pairs = zip(leaves, scores)
            sortleaf = sorted(pairs, key=lambda x:x[1], reverse=True)
            idx = sortleaf.index((best[1],bestscore))
            sortleaf[0], sortleaf[idx] = sortleaf[idx], sortleaf[0]

            if depth < self.K and False:
                for i, l in enumerate(sortleaf):
                    if i < self.max_num_leaves:
                        survived.append((best[0], l[0], self.hypothesis_set[best[0]].nodes[l[0]].track_id))
                    else:
                        self.hypothesis_set[best[0]].removeBranch(l[0])
                        del self.conflictList[best[0]][l[0]]
                continue

            #survived.append((best[0], best[1], t)) # best track
            #best_dets = set(self.hypothesis_set[best[0]].nodes[best[1]].det_index)
            while len(sortleaf) > 0:
                item = sortleaf.pop(0)
                dets = set(self.hypothesis_set[best[0]].nodes[item[0]].det_index)
                loop = list(sortleaf)
                for i, l in enumerate(loop):
                    det_i = set(self.hypothesis_set[best[0]].nodes[l[0]].det_index)
                    if det_i.issubset(dets):
                        L_D = self.hypothesis_set[best[0]].nodes[l[0]].scores[0]
                        L_S = self.hypothesis_set[best[0]].nodes[item[0]].scores[0]
                        self.hypothesis_set[best[0]].nodes[item[0]].scores[0] = L_S + np.log(1+np.exp(-(L_S-L_D)))
                        self.hypothesis_set[best[0]].removeBranch(l[0])
                        del self.conflictList[best[0]][l[0]]
                        sortleaf.remove((l[0], l[1]))
            
            bestscore = self.hypothesis_set[best[0]].nodes[best[1]].scores[0]
            leaves = self.hypothesis_set[best[0]].findLeaves()
            scores = [self.hypothesis_set[best[0]].nodes[l].scores[0] for l in leaves]
            pairs = zip(leaves, scores)
            sortleaf = sorted(pairs, key=lambda x:x[1], reverse=True)
            idx = sortleaf.index((best[1],bestscore))
            sortleaf[0], sortleaf[idx] = sortleaf[idx], sortleaf[0]
            for i, l in enumerate(sortleaf):
                if i < self.max_num_leaves:
                    survived.append((best[0], l[0], self.hypothesis_set[best[0]].nodes[l[0]].track_id))
                else:
                    self.hypothesis_set[best[0]].removeBranch(l[0])
                    del self.conflictList[best[0]][l[0]]

            # for i, l in enumerate(sortleaf):
            #     if l[0] == best[1]:
            #         continue
            #     if i < self.max_num_leaves:
            #         dets = set(self.hypothesis_set[best[0]].nodes[l[0]].det_index)
            #         if dets.issubset(best_dets):
            #             self.hypothesis_set[best[0]].removeBranch(l[0])
            #             del self.conflictList[best[0]][l[0]]
            #         else:
            #             survived.append((best[0], l[0], self.hypothesis_set[best[0]].nodes[l[0]].track_id))
            #     else:
            #         self.hypothesis_set[best[0]].removeBranch(l[0])
            #         del self.conflictList[best[0]][l[0]]

        for t in self.hypothesis_set:
            leaves = self.hypothesis_set[t].findLeaves()
            if DEBUGGING: assert len(leaves) <= self.max_num_leaves+1
            for l in leaves:
                newcfls = set()
                for c in self.conflictList[t][l]:
                    if c in survived:
                        newcfls.add(c)
                self.conflictList[t][l] = newcfls

    def saveConfirmedTrack(self, treeNo, v_num, trackID):
        status = self.hypothesis_set[treeNo].nodes[v_num].status
        route = self.hypothesis_set[treeNo].nodes[v_num].path
        score = self.hypothesis_set[treeNo].nodes[v_num].scores
        det_score = score[3]
        trajectory = []
        n_dummies = 0
        if DEBUGGING: assert len(self.hypothesis_set[treeNo].history['dets']) == len(self.hypothesis_set[treeNo].history['estimates'])
        history_len = len(self.hypothesis_set[treeNo].history['dets'])
        for h in range(history_len):
            adet = self.hypothesis_set[treeNo].history['dets'][h]
            estimate = self.hypothesis_set[treeNo].history['estimates'][h]
            wd, ht, frame, d_i, dummy = adet[2], adet[3], adet[5], adet[6], adet[7]
            cx, cy = estimate[0], estimate[1]
            track = [frame, cx-wd/2, cy-ht/2, wd, ht, dummy, 0]
            trajectory.append(track)
            if dummy == 1:n_dummies += 1
        for r in route:
            adet = r.detection
            wd, ht, frame, d_i, dummy = adet[2], adet[3], adet[5], adet[6], adet[7]
            cx, cy = r.kalman_state[0].item(), r.kalman_state[1].item()
            track = [frame, cx-wd/2, cy-ht/2, wd, ht, dummy, 0]
            trajectory.append(track)
            if DEBUGGING: assert dummy == int(r.is_dummy)
            if dummy == 1:n_dummies += 1
        tails = 0
        for t in reversed(trajectory):
            if t[5] == 0: break # dummy indicator: not a dummy
            tails += 1
        #trajectory = trajectory[:(len(trajectory)-tails)]
        track_len = len(trajectory)-tails
        for i in range(track_len, track_len+tails):
            trajectory[i][6] = 1 # marking tails
        if DEBUGGING:
            assert tails == status[3]
            assert track_len == (status[0]-status[3])
            assert n_dummies == status[2]
            assert not(trackID in self.confirmed_tracks), 'Fatal error: track ID conflict'
            assert (n_dummies-tails+status[1]) == track_len
            for i in range(1, track_len):
                assert trajectory[i][0] - trajectory[i-1][0] == 1
        if status[1]/track_len > self.min_track_quality and track_len > self.min_track_length and det_score/status[1] > self.min_det_conf:
            self.confirmed_tracks[trackID] = trajectory

    def rand_string(self, param, que):
        leaves = self.hypothesis_set[param].findLeaves()
        que.put({param:leaves})

    def doTracking(self, nframe, detections, app_scores=None, canvas=None): # update MHT
        
        # detections: (x, y, w, h, b, t, i, dummy), b=confidence, t=frame, i=i-th detection at the frame
        if DEBUGGING:start = time.time()
        det_usage = self.updateTrackTrees(nframe, detections, app_scores=app_scores, canvas=canvas)
        if DEBUGGING:print('\nupdateTree: %f' % ((time.time()-start)*1000))

        if DEBUGGING:start = time.time()
        self.makeConflictList(det_usage)
        if DEBUGGING:print('conflict: %f' % ((time.time()-start)*1000))

        if DEBUGGING:start = time.time()
        clusters = self.clustering()
        if DEBUGGING:print('clustering: %f' % ((time.time()-start)*1000))

        if DEBUGGING:start = time.time()
        best_set = self.compBestHypoSet(clusters)
        if DEBUGGING:print('bestset: %f' % ((time.time()-start)*1000))

        if DEBUGGING:start = time.time()
        currentTracks = self.treePruning(clusters, best_set)
        if DEBUGGING:print('pruning: %f' % ((time.time()-start)*1000))

        if DEBUGGING:start = time.time()
        self.branchMerging(currentTracks)
        if DEBUGGING:print('merging: %f' % ((time.time()-start)*1000))

        # save detections
        if DEBUGGING: assert nframe not in self.dets_set
        self.dets_set[nframe] = detections
        return currentTracks
    
    def getTrackPatches(self):
        features = {}
        feat_list = []
        counter = 1
        for t in self.hypothesis_set:
            leaves = self.hypothesis_set[t].findLeaves()
            for l in leaves:
                det_i = self.hypothesis_set[t].nodes[l].det_index[-1]
                if det_i not in features:
                    features[det_i] = {'app':self.dets_set[det_i[0]][det_i[1]]['app'], 'used':[(t,l)],'det':self.dets_set[det_i[0]][det_i[1]]['det']}
                    feat_list.append(det_i)
                else:
                    features[det_i]['used'].append((t,l))
        return features, feat_list
    
    def concludeTracks(self): # conclude MHT
        if DEBUGGING:
            allcfls = set()
            assert len(self.conflictList) == len(self.hypothesis_set)
            for t in self.conflictList:
                assert t in self.hypothesis_set
                leaves = self.hypothesis_set[t].findLeaves()
                assert len(self.conflictList[t]) == len(leaves)
                for l in self.conflictList[t]:
                    assert l in self.hypothesis_set[t].nodes
                    allcfls = allcfls.union(self.conflictList[t][l])
            for t in self.hypothesis_set:
                leaves = self.hypothesis_set[t].findLeaves()
                for l in leaves:
                    track_id = self.hypothesis_set[t].nodes[l].track_id
                    item = {(t, l, track_id)}
                    assert len(allcfls.intersection(item)) == 1
                    allcfls.difference_update(item)
            assert len(allcfls) == 0
        clusters = self.clustering()
        best_set = self.compBestHypoSet(clusters)
        for c in best_set:
            for b in best_set[c]:
                self.saveConfirmedTrack(b[0], b[1], b[2])
        
        new_id = 0
        results = []
        for t in self.confirmed_tracks:
            new_id += 1
            tracks = self.confirmed_tracks[t]
            for a in tracks:
                if a[6] == 1: # tails
                    continue
                dummy = 1 if a[5] == 1 else -1 # 1 == dummy bbox, -1 == normal bbox
                trj = (a[0], new_id, round(a[1]), round(a[2]), round(a[3]), round(a[4]), 1, -1, -1, dummy)
                results.append(trj)
        results.sort(key=lambda x:(x[0], x[1]))
        return results
