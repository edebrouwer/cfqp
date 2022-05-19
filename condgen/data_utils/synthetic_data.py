import numpy as np
from numpy.polynomial.polynomial import polyval
from sklearn.preprocessing import OneHotEncoder
from numpy.random import choice 
import random
from numpy.random import default_rng

def get_subtype(data):
    d0 = (data[:,0]<0)*2.
    d1 = (data[:,1]<0)*1.
    return np.array(d0+d1)

def get_ys(subtype):
    ys = []
    for s in subtype:
        if s==0:
            ys.append(3)
        elif s==1:
            ys.append(10)
        elif s==2:
            ys.append(12)
        elif s==3:
            ys.append(5)
        else:
            raise ValueError('Not expecting subtype: '+str(s))
    return np.array(ys)

def get_ys_noisy(subtype):
    ys = []
    for s in subtype:
        if s==0:
            ys.append(3)
        elif s==1:
            ys.append(7)
        elif s==2:
            ys.append(9)
        elif s==3:
            ys.append(5)
        else:
            raise ValueError('Not expecting subtype: '+str(s))
    return np.array(ys)

def make_one_hot(X_sub, n_values='auto'):
    assert X_sub.ndim==2,'expecting 2d array'
#     enc = OneHotEncoder(n_values = n_values, categories='auto') # nvalues deprecated 
    enc = OneHotEncoder(categories='auto')
    enc.fit(X_sub)
    subtype_one_hot = enc.transform(X_sub).toarray()
    return subtype_one_hot

def get_longitudinal_data_nosub(samples, lrandom=False, add_feats=0):
    Xvals = np.arange(20)
    if not lrandom: 
        up = [-2, 0.0002, 0.1]; down = [4, -0.1, -0.1]
    else: 
        a  = np.random.uniform(0.05, 0.15)
        bu = np.random.uniform(0.0001, 0.01)
        bd = np.random.uniform(-0.3, -0.05)
        cu = np.random.uniform(-5, -1)
        cd = np.random.uniform(1,5) 
        up = [cu, bu, a]; down = [cd, bd, -a]

    datalist = []

    for _ in samples: 
        fxn = []
        for i in range(2+add_feats):
            a  = np.random.uniform(0.05, 0.15)
            bu = np.random.uniform(0.0001, 0.01)
            bd = np.random.uniform(-0.3, -0.05)
            cu = np.random.uniform(-5, -1)
            cd = np.random.uniform(1,5) 
            up = [cu, bu, a]; down = [cd, bd, -a]

            # fxn.append(up)
            ty = np.random.uniform(0,1)
            if ty < 0.5: 
                fxn.append(up)
            else: 
                fxn.append(down)
        dims = [np.array([polyval(x, fxn[num]) for x in Xvals]) for num in range(len(fxn))]
        tot  = np.concatenate([dim[...,None] for dim in dims], axis=-1)
        datalist.append(tot)
    data = np.array(datalist)
    data = 0.5*data+0.5*np.random.randn(*data.shape)
    return data

def get_longitudinal_data(stypelist, lrandom=False, add_feats=0):
    Xvals = np.arange(20)
    if not lrandom: 
        up = [-2, 0.0002, 0.1]; down = [4, -0.1, -0.1]
    else: 
        a  = np.random.uniform(0.05, 0.15)
        bu = np.random.uniform(0.0001, 0.01)
        bd = np.random.uniform(-0.3, -0.05)
        cu = np.random.uniform(-5, -1)
        cd = np.random.uniform(1,5) 
        up = [cu, bu, a]; down = [cd, bd, -a]
    subtype_fxn = {}
    subtype_fxn[0] = [down, down]
    subtype_fxn[1] = [down, up]
    subtype_fxn[2] = [up, down]
    subtype_fxn[3] = [up, up]

    for k in subtype_fxn.keys(): 
        add = []
        for num in range(add_feats): 
            # a  = np.random.uniform(0.05, 0.15)
            # bu = np.random.uniform(0.0001, 0.01)
            # bd = np.random.uniform(-0.3, -0.05)
            # cu = np.random.uniform(1,5)
            # du = np.random.uniform(5,10)
            # dd = np.random.uniform(-10,-5)
            # up = [6, cu, bu, a]; down = [-6, cu, bd, -a]
            a  = np.random.uniform(0.05, 0.15)
            bu = np.random.uniform(0.0001, 0.01)
            bd = np.random.uniform(-0.3, -0.05)
            cu = np.random.uniform(-5, -1)
            cd = np.random.uniform(1,5) 
            up = [cu, bu, a]; down = [cd, bd, -a]
            if k == 0:
                add.append(up)
            elif k == 1: 
                if num % 2 == 0: 
                    add.append(down)
                else: 
                    add.append(up)
            elif k == 2: 
                if num % 2 == 0: 
                    add.append(up)
                else: 
                    add.append(down)
            else: 
                add.append(down)

        subtype_fxn[k] = subtype_fxn[k] + add 

    datalist  = []
    for s in stypelist:
        fxn   = subtype_fxn[s]
        if add_feats == 0: 
            dim0  = np.array([polyval(x, fxn[0]) for x in Xvals])
            dim1  = np.array([polyval(x, fxn[1]) for x in Xvals])
            both  = np.concatenate([dim0[...,None], dim1[...,None]], axis=-1)
            datalist.append(both)
        else: 
            dims = [np.array([polyval(x, fxn[num]) for x in Xvals]) for num in range(len(fxn))]
            tot  = np.concatenate([dim[...,None] for dim in dims], axis=-1)
            datalist.append(tot)

    data = np.array(datalist)
    data = 0.5*data+0.5*np.random.randn(*data.shape)

    return data

def generate_synthetic(nsamples, noisy_ys= False, lrandom=False, add_feats=0, sub=True):
    # Generate Data
    baseline_data   = np.random.randn(nsamples,2)
    subtype         = get_subtype(baseline_data)
    subtype_oh      = make_one_hot(subtype[:,None])
    if noisy_ys:
        ys              = get_ys_noisy(subtype)
    else:
        ys              = get_ys(subtype)
    if sub: 
        ldata           = get_longitudinal_data(subtype, lrandom, add_feats)
    else: 
        ldata           = get_longitudinal_data_nosub(subtype, lrandom, add_feats)
    return baseline_data, ys, subtype, subtype_oh, ldata

def preds_to_category(preds):
    category = []
    for p in preds:
        if p>-0.5 and p<=0.5:
            category.append(0)
        elif p>0.5 and p<=1.5:
            category.append(1)
        elif p>1.5 and p<=2.5:
            category.append(2)
        elif p>2.5 and p<=3.5:
            category.append(3)
        else:
            category.append(-1)
    return np.array(category)

""" Function to return data to feed into models """
def load_synthetic_data(fold_span = range(5), nsamples = {'train':500, 'valid':300, 'test':200}):
    np.random.seed(0)
    dset = {}
    for fold in fold_span:
        dset[fold] = {}
        for k in ['train','valid','test']:
            b, ys, subtype, subtype_oh, ldata = generate_synthetic(nsamples[k])
            dset[fold][k]      = {}
            dset[fold][k]['b'] = b
            dset[fold][k]['x'] = ldata
            dset[fold][k]['a'] = np.cumsum(np.ones_like(ldata[...,0])[...,None], axis=1)/5.
            dset[fold][k]['m'] = np.ones_like(ldata[...,0])
            ys_rshp = ys.reshape(-1,1)
            ys_rshp = ys_rshp+0.1*np.random.randn(*ys_rshp.shape)
            ys_seq  = ys_rshp - (np.cumsum(np.ones_like(ldata[...,0]), 1)-1)
            m_ys_seq= (ys_seq>0)
            ys_seq[~m_ys_seq] = 0.
            m_ys_seq= m_ys_seq*1.
            dset[fold][k]['ys_seq']     = ys_seq
            dset[fold][k]['m_ys_seq']   = m_ys_seq
            dset[fold][k]['ce']         = np.zeros((nsamples[k],1))
            dset[fold][k]['subtype']    = subtype
            dset[fold][k]['subtype_oh'] = subtype_oh
    return dset

""" Function to return data to feed into models """
def load_synthetic_data_noisy(fold_span = range(5), nsamples = {'train':500, 'valid':300, 'test':200},
                       distractor_dims_b = 0, sigma_ys = 0.1):
    np.random.seed(0)
    dset = {}
    for fold in fold_span:
        dset[fold] = {}
        for k in ['train','valid','test']:
            b, ys, subtype, subtype_oh, ldata = generate_synthetic(nsamples[k], noisy_ys= True)
            dset[fold][k]      = {}
            if distractor_dims_b == 0:
                dset[fold][k]['b'] = b
            elif distractor_dims_b > 0:
                dset[fold][k]['b'] = np.concatenate((b, np.random.randn(b.shape[0], distractor_dims_b)), -1)
            else:
                raise ValueError('Bad setting for distractor_dims_b')
                
            dset[fold][k]['x'] = ldata
            dset[fold][k]['a'] = np.cumsum(np.ones_like(ldata[...,0])[...,None], axis=1)/5.
            dset[fold][k]['m'] = np.ones_like(ldata[...,0])
            ys_rshp = ys.reshape(-1,1)
            ys_rshp = ys_rshp+sigma_ys*np.random.randn(*ys_rshp.shape)
            ys_seq  = ys_rshp - (np.cumsum(np.ones_like(ldata[...,0]), 1)-1)
            m_ys_seq= (ys_seq>0)
            ys_seq[~m_ys_seq] = 0.
            m_ys_seq= m_ys_seq*1.
            dset[fold][k]['ys_seq']     = ys_seq
            dset[fold][k]['m_ys_seq']   = m_ys_seq
            dset[fold][k]['ce']         = np.zeros((nsamples[k],1))
            dset[fold][k]['subtype']    = subtype
            dset[fold][k]['subtype_oh'] = subtype_oh
    return dset

def set_trt(ldata, subtype = None, confounding = False): 
#     a = np.zeros_like(ldata[...,0]) 
    a = np.zeros_like(ldata)

    
    min_time = 4
    if confounding: #time of treatment is confounded with the subtype
        t_span = a.shape[1]-min_time
        t_third = t_span //3
        t_remain = t_span - 2*t_third

        p0 = np.linspace(0,0.2,num = t_span)[::-1]
        p1 = np.concatenate((np.ones(t_third)*0.1,np.ones(t_remain)*0.3,np.ones(t_third)*0.1))
        p2 = np.concatenate((np.ones(t_third)*0.3,np.ones(t_remain)*0.1,np.ones(t_third)*0.3))
        p3 = np.linspace(0,0.2,num = t_span)
        
        ps = np.stack((p0,p1,p2,p3))
        ps = np.stack([ps[int(i)] for i in subtype])

        rng = default_rng()
        times_vec = rng.binomial(n = 1, p = ps)
        t = times_vec.argmax(1) + min_time
    else:
        t = np.random.randint(low=4, high=a.shape[1], size=a.shape[0])

    for i in range(a.shape[0]):
        if t[i] == a.shape[1]-1 or t[i] == a.shape[1]-2: 
            t[i] -= 3
        a[i,t[i],1] = 1. 
#     return a[...,None]
    return a

def apply_trt(ldata, a, subtype): 
    alpha_1s = [10,5,-5,-10] # positive for subtype 1, negative for subtype 2; alpha2 - 0.2, alpha_3 - 0.4
    gamma = 2; b = 3
    alpha_2s = [0.6, 0.6, 0.6, 0.6]
    alpha_3s = [0.8, 0.8, 0.8, 0.8]
    z = [1,1,-1,-1]
    
    for i in range(ldata.shape[0]): 
        int_t = 0
        for t in range(ldata.shape[1]): 
            if t == 0: 
                continue
            if a[i,t-1,1] == 1.: 
                if t+gamma < ldata.shape[1]:
                    gamma_i = t+gamma
                    real_t1 = np.arange(0,gamma); real_t2 = np.arange(gamma,gamma+(ldata.shape[1]-gamma_i))
                else: 
                    gamma_i = ldata.shape[1]-1
                    real_t1 = np.arange(0,gamma-1); real_t2 = np.arange(gamma-1,gamma-1+(ldata.shape[1]-gamma_i))
                alpha_1 = alpha_1s[int(subtype[i])]; alpha_2= alpha_2s[int(subtype[i])]
                alpha_3 = alpha_3s[int(subtype[i])]
                base_0  = -alpha_1 / (1 + np.exp(alpha_2 * gamma_i / 2.))
#                 alpha_0 = (alpha_1 + 2*base_0 - ldata[i,gamma_i:,:]) / (1 + np.exp(-alpha_3 * gamma_i / 2.))
                alpha_0 = (alpha_1 + 2*base_0 - b) / (1 + np.exp(-alpha_3 * gamma_i / 2.))
                ldata[i,t:gamma_i,:] += (base_0 + alpha_1 / (1 + np.exp(-alpha_2 * (real_t1[:,None] - gamma / 2.))))
                ldata[i,gamma_i:,:]  += (b + alpha_0 / (1 + np.exp(alpha_3 * (real_t2[:,None] - 3*gamma / 2.))))
                int_t = t
        a[i,int_t:,1] = 1.; a[i,int_t-1:,0] = 1.
        time_val      = (np.cumsum(a[i,int_t-1:,0]))*0.1
        a[i,int_t-1:,0] = time_val 
    return ldata, a

def set_trt_line(ldata, num_trt=1, subtype = None): 
    a = np.zeros((ldata.shape[0], ldata.shape[1],num_trt+4))
    t = np.random.randint(low=4, high=a.shape[1], size=a.shape[0])


    # t = np.random.randint(low=4, high=10, size=a.shape[0])
    for i in range(a.shape[0]):
        if t[i] == a.shape[1]-1: 
            t[i] -= 3
#         l = choice(np.arange(2,5), 1, p=[0.6, 0.3, 0.1])
        l = num_trt+1 # fix the line of therapy as first line for now
        # randomly select how many of the num_trt should be given
        if num_trt == 1: 
            trt_idxs = np.array([1])
        else: 
            num_select = np.random.randint(low=2, high=num_trt+1)
            # then, pick which indices should be turned "on"
            trt_idxs   = np.random.randint(low=1, high=num_trt+1, size=num_select)
        a[i,t[i],trt_idxs] = 1.; a[i,t[i],l] = 1.
        # a[i,t[i],1] = 1.; a[i,t[i],l] = 1.
    return a

def apply_trt_line_complex(ldata, a, subtype): 
    alpha_1s = np.zeros((2,4))
    alpha_1s[:,0] = np.array([10,10]); alpha_1s[:,1] = np.array([5,-5])
    alpha_1s[:,2] = np.array([-5,5]); alpha_1s[:,3]  = np.array([-10,-10])
    alpha_2 = 0.6
    alpha_3s = [0.6,0.8,0.9]; gamma = 2
    s = [1, 1, -1, -1]; b = [3, 4.5, 6]
    for i in range(ldata.shape[0]): 
        int_t = 0; l = 0
        for t in range(ldata.shape[1]): 
            if t == 0: 
                continue
            if a[i,t-1,1] == 1.:
                if t+gamma < ldata.shape[1]:
                    gamma_i = t+gamma
                    real_t1 = np.arange(0,gamma); real_t2 = np.arange(gamma,gamma+(ldata.shape[1]-gamma_i))
                else: 
                    gamma_i = ldata.shape[1]-1
                    real_t1 = np.arange(0,gamma-1); real_t2 = np.arange(gamma-1,gamma-1+(ldata.shape[1]-gamma_i))
                l = np.where(a[i,t-1,2:] == 1.)[0]
                alpha_1 = alpha_1s[:,int(subtype[i])]
                alpha_3 = alpha_3s[int(l)]
                base_0  = -alpha_1 / (1 + np.exp(alpha_2 * gamma_i / 2.))
                alpha_0 = (alpha_1 + 2*base_0 - b[int(l)]) / (1 + np.exp(-alpha_3 * gamma_i / 2.))
                ldata[i,t:gamma_i,:] += (base_0 + alpha_1 / (1 + np.exp(-alpha_2 * (real_t1[:,None] - gamma_i / 2.))))
                ldata[i,gamma_i:,:]  += (b[int(l)] + alpha_0 / (1 + np.exp(alpha_3 * (real_t2[:,None] - 3*gamma_i / 2.))))
                int_t = t
        a[i,int_t:,1] = 1.; a[i,int_t-1:,0] = 1.
        time_val      = (np.cumsum(a[i,int_t-1:,0]))*0.1
        a[i,int_t-1:,0] = time_val 
        a[i,int_t:,2:]  = np.repeat(a[i,int_t-1,2:][None,...], a[i,int_t:,0].shape[0],axis=0)
    return ldata, a

# def apply_trt_line(ldata, a, subtype, num_trt): 
#     alpha_1s = [10,5,-5,-10]; alpha_2 = 0.6
#     alpha_3s = [0.6,0.8,0.9]; gamma = 2
#     b = [3, 4.5, 6]
#     s = [1, 1, -1, -1]
#     params = {}
#     params[1] = [alpha_1s, alpha_3s, alpha_2, gamma, b]

#     for trt_idx in range(2,num_trt):
#         l = np.random.randint(low=5,high=15)
#         alpha_1s = [l,int(0.5*l),-int(0.5*l),-l]
#         alpha_2  = np.random.uniform(low=0.,high=1.); alpha_3 = np.random.uniform(low=0.,high=.7)
#         alpha_3s = [alpha_3, alpha_3+0.2, alpha_3+0.3]
#         gamma = np.random.randint(low=2,high=5)
#         params[trt_idx] = [alpha_1s, alpha_3s, alpha_2, gamma, b]
    
#     for i in range(ldata.shape[0]): 
#         int_t = 0; l = 0
#         for t in range(ldata.shape[1]): 
#             if t == 0: 
#                 continue
#             if a[i,t-1,1] == 1.:
#                 if t+gamma < ldata.shape[1]:
#                     gamma_i = t+gamma
#                     real_t1 = np.arange(0,gamma); real_t2 = np.arange(gamma,gamma+(ldata.shape[1]-gamma_i))
#                 else: 
#                     gamma_i = ldata.shape[1]-1
#                     real_t1 = np.arange(0,gamma-1); real_t2 = np.arange(gamma-1,gamma-1+(ldata.shape[1]-gamma_i))
#                 l = np.where(a[i,t-1,2:] == 1.)[0]
#                 alpha_1 = alpha_1s[int(subtype[i])]
#                 alpha_3 = alpha_3s[int(l)]
#                 base_0  = -alpha_1 / (1 + np.exp(alpha_2 * gamma_i / 2.))
#                 alpha_0 = (alpha_1 + 2*base_0 - b[int(l)]) / (1 + np.exp(-alpha_3 * gamma_i / 2.))
#                 ldata[i,t:gamma_i,:] += (base_0 + alpha_1 / (1 + np.exp(-alpha_2 * (real_t1[:,None] - gamma_i / 2.))))
#                 ldata[i,gamma_i:,:]  += (b[int(l)] + alpha_0 / (1 + np.exp(alpha_3 * (real_t2[:,None] - 3*gamma_i / 2.))))
#                 int_t = t
#         a[i,int_t:,1] = 1.; a[i,int_t-1:,0] = 1.
#         time_val      = (np.cumsum(a[i,int_t-1:,0]))*0.1
#         a[i,int_t-1:,0] = time_val 
#         a[i,int_t:,2:]  = np.repeat(a[i,int_t-1,2:][None,...], a[i,int_t:,0].shape[0],axis=0)
#     return ldata, a

def apply_trt_line(ldata, a, subtype, num_trt=1): 
    # import pdb; pdb.set_trace()
    alpha_1s = [10,5,-5,-10]; alpha_2 = 0.6
    alpha_3s = [0.6,0.8,0.9]; gamma = 2
    b = [3, 4.5, 6]
    s = [1, 1, -1, -1]
    params = {}
    params[1] = [alpha_1s, alpha_3s, alpha_2, gamma, b]

     
    for trt_idx in range(2,num_trt+1):
        
        #times_vec = rng.binomial(n = 1, p = p, size=(N,t_span,num_treats))
        l = np.random.randint(low=5,high=15)
        alpha_1s = [l,int(0.5*l),-int(0.5*l),-l]
        alpha_2  = np.random.uniform(low=0.,high=1.); alpha_3 = np.random.uniform(low=0.,high=.7)
        alpha_3s = [alpha_3, alpha_3+0.2, alpha_3+0.3]
        gamma = np.random.randint(low=2,high=5)
        params[trt_idx] = [alpha_1s, alpha_3s, alpha_2, gamma, b]
    
    for i in range(ldata.shape[0]): 
        int_t = 0; l = 0
        trt_select = []
        for t in range(ldata.shape[1]): 
            if t == 0: 
                continue
            for k in range(1,num_trt+1): 
                alpha_1s, alpha_3s, alpha_2, gamma, b = params[k]
                if a[i,t-1,k] == 1.:
                    if t+gamma < ldata.shape[1]:
                        gamma_i = t+gamma
                        real_t1 = np.arange(0,gamma); real_t2 = np.arange(gamma,gamma+(ldata.shape[1]-gamma_i))
                    else: 
                        gamma_i = ldata.shape[1]-1
                        real_t1 = np.arange(0,gamma-1); real_t2 = np.arange(gamma-1,gamma-1+(ldata.shape[1]-gamma_i))
                    l = np.where(a[i,t-1,num_trt+1:] == 1.)[0]
                    alpha_1 = alpha_1s[int(subtype[i])]
                    alpha_3 = alpha_3s[int(l)]
                    base_0  = -alpha_1 / (1 + np.exp(alpha_2 * gamma_i / 2.))
                    alpha_0 = (alpha_1 + 2*base_0 - b[int(l)]) / (1 + np.exp(-alpha_3 * gamma_i / 2.))
                    ldata[i,t:gamma_i,:] += (base_0 + alpha_1 / (1 + np.exp(-alpha_2 * (real_t1[:,None] - gamma_i / 2.))))
                    ldata[i,gamma_i:,:]  += (b[int(l)] + alpha_0 / (1 + np.exp(alpha_3 * (real_t2[:,None] - 3*gamma_i / 2.))))
                    trt_select.append(k)
                    int_t = t
        # import pdb; pdb.set_trace()
        a[i,int_t:,trt_select] = 1.; a[i,int_t-1:,0] = 1.
        time_val      = (np.cumsum(a[i,int_t-1:,0]))*0.1
        a[i,int_t-1:,0] = time_val 
        a[i,int_t:,num_trt+1:]  = np.repeat(a[i,int_t-1,num_trt+1:][None,...], a[i,int_t:,0].shape[0],axis=0)
    return ldata, a

""" Function to return data to feed into models """
def load_synthetic_data_trt(fold_span = range(5), nsamples = {'train':500, 'valid':300, 'test':200}, \
                        distractor_dims_b = 0, sigma_ys = 0.1, include_line=True, lrandom=False, \
                        seed=0, multiplier= 1., alpha_1_complex = False, per_missing=0., add_feats=0, num_trt=1, sub=False, confounding = False):
    np.random.seed(seed)
    random.seed(seed)
    dset = {}
    for fold in fold_span:
        dset[fold] = {}
        for k in ['train','valid','test']:
            b, ys, subtype, subtype_oh, ldata = generate_synthetic(nsamples[k], noisy_ys=True, lrandom=lrandom, add_feats=add_feats, sub=sub)
            dset[fold][k]      = {}
            if distractor_dims_b == 0:
                dset[fold][k]['b'] = b
            elif distractor_dims_b > 0:
                dset[fold][k]['b'] = np.concatenate((b, np.random.randn(b.shape[0], distractor_dims_b)), -1)
            else:
                raise ValueError('Bad setting for distractor_dims_b')
                
            dset[fold][k]['x_orig'] = ldata
            if include_line: 
                dset[fold][k]['a'] = set_trt_line(np.copy(ldata), num_trt=num_trt)
                if alpha_1_complex: 
                    new_x, new_a = apply_trt_line_complex(np.copy(ldata), np.copy(dset[fold][k]['a']), subtype)
                else: 
                    new_x, new_a = apply_trt_line(np.copy(ldata), np.copy(dset[fold][k]['a']), subtype, num_trt=num_trt)
            else: 
                dset[fold][k]['a'] = set_trt(np.copy(ldata), subtype = subtype, confounding = confounding)
                new_x, new_a = apply_trt(np.copy(ldata), np.copy(dset[fold][k]['a']), subtype)
            dset[fold][k]['x'] = new_x*multiplier
            # use different trt representation for now
            dset[fold][k]['a'] = new_a
            # dset[fold][k]['m'] = np.ones_like(ldata[...,0])
            dset[fold][k]['m'] = np.ones_like(ldata)
            if per_missing > 0.:
                x, m = add_missingness(dset[fold][k]['x'], dset[fold][k]['m'], per_missing=per_missing)
                dset[fold][k]['x'] = x; dset[fold][k]['m'] = m
            ys_rshp = ys.reshape(-1,1)
            ys_rshp = ys_rshp+sigma_ys*np.random.randn(*ys_rshp.shape)
            ys_seq  = ys_rshp - (np.cumsum(np.ones_like(ldata[...,0]), 1)-1)
            m_ys_seq= (ys_seq>0)
            ys_seq[~m_ys_seq] = 0.
            m_ys_seq= m_ys_seq*1.
            dset[fold][k]['ys_seq']     = ys_seq
            dset[fold][k]['m_ys_seq']   = m_ys_seq
            dset[fold][k]['ce']         = np.zeros((nsamples[k],1))
            dset[fold][k]['subtype']    = subtype
            dset[fold][k]['subtype_oh'] = subtype_oh
            dset[fold][k]["event"] = np.random.randint(2,size=(nsamples[k],1)) #random event
    return dset

def add_missingness(x, m, per_missing=0.6):
    possible_idxs = []
    for i in range(1,m.shape[0]): 
        for j in range(1,m.shape[1]): 
            possible_idxs.append((i,j))
    missing_idxs = random.sample(possible_idxs, int(per_missing*m.shape[0]*m.shape[1]))
    missing_arrs = list(zip(*missing_idxs)) 

    new_m = np.ones_like(m)
    new_m[missing_arrs[0],missing_arrs[1]] = 0. 

    # forward fill 
    new_x = np.copy(x)
    new_x[np.where(1-new_m)] = np.nan
    for feat in range(new_x.shape[2]): 
        x_feat = new_x[...,feat]
        mask = np.isnan(x_feat)
        idx = np.where(~mask,np.arange(mask.shape[1]),0)
        np.maximum.accumulate(idx,axis=1, out=idx)
        out = x_feat[np.arange(idx.shape[0])[:,None], idx]
        new_x[...,feat] = out
        # new_x[mask] = new_x[np.nonzero(mask)[0], idx[mask]]

    return new_x, new_m 

if __name__=='__main__':
    import pdb; pdb.set_trace()
    dset = load_synthetic_data_trt(fold_span = [1])
