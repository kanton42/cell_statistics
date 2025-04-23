from ctypes.wintypes import FLOAT
from numba import njit, prange, set_num_threads
import numpy as np
from scipy.signal import periodogram
from scipy import special #fro gamma function
import tidynamics
import matplotlib.pyplot as plt
import matplotlib.cm as cm
# from correlation_mkl import *
import mpltex
from joblib import Parallel, delayed


# effective kernel extraction
def kernel_1dcc(vacf, dt):
    #also works for delta peaks
    G = np.zeros(len(vacf)-1) #G0 is 1/2 term G1 3/2 and so on
    gamma = np.zeros(len(G))
    G[0] = 2 * (vacf[0] - vacf[1]) / (dt * (vacf[0] + vacf[1]))
    #G[1]=2*(vacf[0]-vacf[2])/(dt*(vacf[0]+vacf[1]))-G[0]*((vacf[2]+vacf[1])/(vacf[1]+vacf[0]))
    for n in range(1,len(G)-1):
        summ = np.sum(G[n-1::-1] * (vacf[1:n+1] + vacf[2:n+2])) / (vacf[0] + vacf[1])
        G[n] = 2 * (vacf[0] - vacf[n+1]) / (dt * (vacf[0] + vacf[1])) - summ

    gamma[0] = 2 * G[0] / dt #use antisymmetry of G (G_1/2-G_-1/2 = 2*G_1/2)
    gamma[1:] = np.diff(G) / dt #analytic exact midpoint derivative of half stepped G
    # for kernels not including delta peak, higher accuracy when using np.gradient(G,dt) for gamma[0] as well
    return gamma[:-1]

###########################################################################################
# combine_ functions are used to apply a function to every row of a masked matrix individually


def combine_fun2_output(func, xall, yall, lmodified=0, lmod_y=0, out=4):
    #combine masked arrays to masked output matrix, where out determines how many outputs the function has to have
    ntrj=len(xall[:,0])
    lm=len(xall[0,:])
    modified=np.ma.empty((out,ntrj,lm-lmodified))
    modified.mask=True
    for i in range(ntrj):
        x=xall[i,:].compressed()
        #x=x[x.mask==False]
        #x=np.array(x)
        y=yall[i,:].compressed()
        #y=y[y.mask==False]
        #y=np.array(y)
        if lmod_y!=0:
            y=y[:-lmod_y]
        funxy=func(x,y)
        for k in range(out):
            modified[k,i,:len(funxy[k])] = funxy[k]
 
    return modified

def combine_fun(func,xall,lmodified=0):
    #combine masked arrays applying func to every row individually
    #lmodified tells how much shorter data is after applying func
    ntrj = len(xall[:,0])
    lm = len(xall[0,:])
    modified = np.ma.empty((ntrj,lm-lmodified))
    modified.mask = True
    for i in range(ntrj):
        x = xall[i,:].compressed()
        funx = func(x)
        modified[i,:len(funx)] = funx
        
    return modified


def combine_fun_output(func, xall, lmodified=0, outputs=2):
    #combine masked arrays applying func to every row individually
    #lmodified tells how much shorter data is after applying func
    # outputs has to agree to how many values the function func gives as output
    ntrj = len(xall[:,0])
    lm = len(xall[0,:])
    # modified = np.ma.empty((ntrj,lm-lmodified))
    # modified.mask = True
    fun_output_list = np.ma.empty((outputs, ntrj,lm-lmodified))
    fun_output_list.mask = True
    for i in range(ntrj):
        x = xall[i,:].compressed()
        funx = func(x)
        # for k in range(outputs):
        #     fun_output_list[k, i,:len(x)-lmodified] = funx[k]
        fun_output_list[:, i,:len(funx[0])] = funx
        
    return fun_output_list

def combine_fun2(func, xall, yall, lmodified=0, lmod_y=0):
    #combine masked arrays xall,yall applying func to every row individually
    #lmodified tells how much shorter data is after applying func
    ntrj=len(xall[:,0])
    lm=len(xall[0,:])
    modified=np.ma.empty((ntrj, lm-lmodified))
    modified.mask=True
    for i in range(ntrj):
        x = xall[i,:].compressed()
        try:
            y = yall[i,:].compressed()
        except:
            #AttributeError
            y = yall[i,:] # if yall is not masked, use the full array
        if lmod_y != 0:
            y = y[:-lmod_y]
        funxy = func(x,y)

        modified[i,:len(funxy)] = funxy
 
    return modified

def combine_fun_multiple(func, input_list, lmodified=0, lmod_list=[], out_1d=False):
    #combine masked arrays in a list [xall,yall,...] applying func to every row individually
    #lmodified tells how much shorter data is after applying func
    # arguments in the list have to be given in the order that the function func takes them

    ntrj=len(input_list[0][:,0])
    lm=len(input_list[0][0,:])
    if out_1d:
        modified=np.ma.empty((ntrj))
    else:
        modified=np.ma.empty((ntrj, lm-lmodified))
    modified.mask=True
    
    for i in range(ntrj):
        input_list_i = []
        for list_ind in range(len(input_list)):
            append_i = input_list[list_ind][i,:].compressed()
            if len(lmod_list) > 0:
                append_i = append_i[:-lmod_list[list_ind]]
            input_list_i.append(append_i) # append the input_list for every index i, then give it to func
            
        funxy = func(*input_list_i)
        if out_1d:
            modified[i] = funxy
        else:
            modified[i,:len(funxy)] = funxy
 
    return modified

###################################################################################################
# definition of correlation and MSD

def correlation(a, b=None, subtract_mean=False):
    # correlation of a,b or autocorrelation if b=None via Fourier transformation
    #using less memory by defining less arrays in between
    meana = int(subtract_mean) * np.mean(a)
    add_zeros1 = 2**int(np.ceil((np.log(len(a)) / np.log(2)))) - len(a)
    add_zeros2 = 2 * add_zeros1 + len(a)
    data_a = np.append(a, np.zeros(add_zeros2))
    fra = np.fft.fft(data_a)
    data_a = 0 #empty memory
    if b is None:
        sf = np.conj(fra)*fra
        fra = 0 #empty memory
    else:
        meanb = int(subtract_mean)*np.mean(b)
        add_zeros1 = 2**int(np.ceil((np.log(len(b)) / np.log(2)))) - len(b)
        add_zeros2 = 2 * add_zeros1 + len(b)
        data_b = np.append(b, np.zeros(add_zeros2))
        frb = np.fft.fft(data_b)
        data_b = 0 #empty memory
        sf = np.conj(fra)*frb
        frb = 0 #empty memory
    #res = np.fft.ifft(sf)
    return np.real(np.fft.ifft(sf)[:len(a)])/np.array(range(len(a),0,-1))

def msd_fft(x):
    mu = np.mean(x)
    N = len(x)
    D = np.square(x - mu) #.sum(axis=1) 
    D = np.append(D, 0) 
    pos_corr = correlation(x - mu)
    Q = 2 * np.sum(D)
    running_av_sq = np.zeros(N)
    for m in range(N):
        Q = Q - D[m-1] - D[N-m]
        running_av_sq[m] = Q / (N - m)
    return running_av_sq - 2 * pos_corr

def msd_fast(x,y=[],z=[], less_memory=False):
    if less_memory:
        msd_fun = msd_fft # a bit slower but uses less memory
    else:
        msd_fun = tidynamics.msd
    if len(z)>0:
        msd=msd_fun(x)+msd_fun(y)+msd_fun(z)
        
    elif len(y)>0:
        msd=msd_fun(x)+msd_fun(y)
    else:
        msd=msd_fun(x)
    return msd
    
def alpha_msd(msd, dt):
    """
    Compute the logarithmic derivative of the MSD
    """
    t = np.arange(len(msd)) * dt
    return np.diff(np.log(msd[1:])) / np.diff(np.log(t[1:]))
 
 
##########################################################################################
# extracting correlation functions, kernels, etc. from trajectory data

def analysis(x,dt,y=[],ori=[],scale=False, tskip=10):  #,save_name=None
    # create dictionary that contains most important functions from trajectory analysis
    results_dict = {'cvvx': [], 'cvvy': [], 'cvvxy': [], 'vx': [], 'vy': [], 'vxscale': [], 'vyscale': [], 'msd': [], 'kernelx': [], 'kernely': [],
    'frx': [], 'fry': [], 'frx_corr': [], 'fry_corr': [], 'speeds': [],  'msadv': [], 'msad_ori': [], 'angles_difv': [],
    'total_anglev': [], 'angles_difv_corr': [], 'ori_change': [], 'ori_change_corr': [], 'ori_v': []}

    vxall = combine_fun(lambda x:np.diff(x)/dt,x,lmodified=1)
    vxacf = combine_fun(correlation,vxall)
    kernelx = combine_fun(lambda x:kernel_1dcc(x,dt),vxacf,lmodified=2)
    msdx = combine_fun(msd_fast,x)
    random_forcex = 0
    corr_rand_forcex = 0
    if scale:
        scale_vxall = combine_fun(scale_v, vxall)
    if len(y) > 0:
        
        vyall = combine_fun(lambda x:np.diff(x)/dt,y,lmodified=1)
        speed_all = combine_fun2(lambda x,y:np.sqrt(x**2+y**2),vxall,vyall)
        vyacf = combine_fun(correlation,vyall)
        msd = combine_fun2(lambda x,y: msd_fast(x,y),x,y)
        kernely = combine_fun(lambda x:kernel_1dcc(x,dt),vyacf,lmodified=2)
        random_forcey = 0 # combine_fun2(lambda x,y: random_force_gamma_ch(x,dt,y)[0], y, kernely)
        corr_vxy = combine_fun2(lambda x,y: correlation(x, y), vxall, vyall)
        arr = combine_fun2_output(lambda x,y: calc_msad(x,y), x,y, out=3)
        msadv_nobo, angles_difv, tot_av = arr[0,:,:], arr[1,:,:], arr[2,:,:]
        corr_rand_forcey = 0 # combine_fun(correlation, random_forcey)
        corr_angles_difv=combine_fun(correlation, angles_difv)

        ori_average_V = combine_fun2(lambda x,y: v_osc_av(x, y, tskip*dt, dt)[2], vxall, vyall)

        if scale:
            scale_vyall=combine_fun(scale_v,vyall)
        if len(ori) > 0:
            diff_ori=combine_fun(lambda x: np.diff(x), ori, lmodified=1)
            corr_diff_ori=combine_fun(correlation, diff_ori)
            msad_ori=combine_fun(msad_difori_nobo, diff_ori)
            if scale:
                results_dict.update({'cvvx':vxacf, 'cvvy':vyacf, 'cvvxy':corr_vxy, 'vx':vxall, 'vy':vyall, 'vxscale':scale_vxall, 'vyscale':scale_vyall, 'msd':msd, 'kernelx':kernelx, 'kernely':kernely,
    'frx':random_forcex, 'fry':random_forcey, 'frx_corr':corr_rand_forcex, 'fry_corr':corr_rand_forcey, 'speeds':speed_all,  'msadv':msadv_nobo, 'msad_ori':msad_ori, 'angles_difv':angles_difv,
    'total_anglev':tot_av, 'angles_difv_corr':corr_angles_difv, 'ori_change':diff_ori, 'ori_change_corr':corr_diff_ori, 'ori_v':ori_average_V})

            else:
                results_dict.update({'cvvx':vxacf, 'cvvy':vyacf, 'cvvxy':corr_vxy, 'vx':vxall, 'vy':vyall, 'msd':msd, 'kernelx':kernelx, 'kernely':kernely,
    'frx':random_forcex, 'fry':random_forcey, 'frx_corr':corr_rand_forcex, 'fry_corr':corr_rand_forcey, 'speeds':speed_all,  'msadv':msadv_nobo, 'msad_ori':msad_ori, 'angles_difv':angles_difv,
    'total_anglev':tot_av, 'angles_difv_corr':corr_angles_difv, 'ori_change':diff_ori, 'ori_change_corr':corr_diff_ori, 'ori_v':ori_average_V})
        
        if scale:
            results_dict.update({'cvvx':vxacf, 'cvvy':vyacf, 'cvvxy':corr_vxy, 'vx':vxall, 'vy':vyall, 'vxscale':scale_vxall, 'vyscale':scale_vyall, 'msd':msd, 'kernelx':kernelx, 'kernely':kernely,
    'frx':random_forcex, 'fry':random_forcey, 'frx_corr':corr_rand_forcex, 'fry_corr':corr_rand_forcey, 'speeds':speed_all,  'msadv':msadv_nobo, 'angles_difv':angles_difv,
    'total_anglev':tot_av, 'angles_difv_corr':corr_angles_difv, 'ori_v':ori_average_V})


        else:
            results_dict.update({'cvvx':vxacf, 'cvvy':vyacf, 'cvvxy':corr_vxy, 'vx':vxall, 'vy':vyall, 'msd':msd, 'kernelx':kernelx, 'kernely':kernely,
    'frx':random_forcex, 'fry':random_forcey, 'frx_corr':corr_rand_forcex, 'fry_corr':corr_rand_forcey, 'speeds':speed_all,  'msadv':msadv_nobo, 'angles_difv':angles_difv,
    'total_anglev':tot_av, 'angles_difv_corr':corr_angles_difv, 'ori_v':ori_average_V})


    else:
        if scale:
            results_dict.update({'cvvx':vxacf, 'vx':vxall, 'vxscale':scale_vxall, 'msd':msdx, 'kernelx':kernelx,
    'frx':random_forcex, 'frx_corr':corr_rand_forcex})

        else:
            results_dict.update({'cvvx':vxacf, 'vx':vxall, 'msd':msdx, 'kernelx':kernelx,
    'frx':random_forcex, 'frx_corr':corr_rand_forcex})


    return results_dict



def analysis_short(x, dt, y=[], trunc_idx=100):  #,save_name=None
    # create dictionary that contains most important functions from trajectory analysis
    results_dict = {'cvvx': [], 'cvvy': [], 'cvvxy': [], 'vx': [], 'vy': [], 'msd': [], 'kernelx': [], 'kernely': [],
    'kernel_av': []}

    vxall = combine_fun(lambda x: np.diff(x) / dt, x, lmodified=1)
    vxacf = combine_fun(correlation, vxall)
    kernelx = combine_fun(lambda x: kernel_1dcc(x, dt), vxacf[:, :trunc_idx], lmodified=2)
    msdx = combine_fun(msd_fast,x)

    if len(y) > 0:
        
        vyall = combine_fun(lambda x: np.diff(x) / dt, y, lmodified=1)
        vyacf = combine_fun(correlation, vyall)
        msd = combine_fun2(lambda x,y: msd_fast(x,y), x, y)
        kernely = combine_fun(lambda x: kernel_1dcc(x,dt), vyacf[:, :trunc_idx], lmodified=2)
        corr_vxy = combine_fun2(lambda x,y: correlation(x, y), vxall, vyall)

        vacf_avxy = 0.5 * (vyacf + vxacf)
        kern_avxy = combine_fun(lambda x: kernel_1dcc(x, dt), vacf_avxy[:, :trunc_idx], lmodified=2)
        
        results_dict.update({'cvvx':vxacf, 'cvvy':vyacf, 'cvvxy':corr_vxy, 'vx':vxall, 'vy':vyall, 'msd':msd, 'kernelx':kernelx, 'kernely':kernely,
        'kernel_av':kern_avxy})

    else:
        results_dict.update({'cvvx':vxacf, 'vx':vxall, 'msd':msdx, 'kernelx':kernelx,
        })

    return results_dict

#######################################################################################
# some distribution functions for plots
def Gauss(x, mu, sig):
    return np.exp(-(x-mu)**2/(2*sig**2))/(sig*np.sqrt(2*np.pi))

def log_normal(x, log_med, sig):
    return np.exp(-(np.log(x) - log_med)**2 / (2 * sig**2)) / (x * sig * np.sqrt(2 * np.pi))

def laplace_distr(x, mu, b):
    return (1/(2*b)) * np.exp(-np.abs(x-mu)/b)
    
#################################################################################################
    
def weighted_av(xall,lmod=0):
    # average over masked matrix containing correlation functions, where each value is weighted by the corresponding trajectory length
    weightedx=np.zeros(len(xall[0,:]))
    lx=np.zeros(len(xall[:,0]))
    add=0
    for i in range(len(xall[:,0])):
        x = xall[i,:].compressed()
        lx[i] = len(x)
        for j in range(int(lx[i])):
            weightedx[j] += (x[j]*(lx[i]-j))
    
    for k in range(len(weightedx)):
        ind_ended = np.where(lx-k<0)[0]
        if len(ind_ended)>0:
            add = 0
            for l in range(len(ind_ended)):
                add+=(k-1-lx[ind_ended[l]])

        divide = np.sum(lx-k) + add
        if divide != 0:
            weightedx[k] /= divide
    
    return weightedx


def scale_v(vd):
    return (vd-np.mean(vd))/np.std(vd)

def cut_fun(x,cut_ind=20,cut_end=-1):
    return x[cut_ind:cut_end]

def msd_prw(tm,B,t):
    #td=1/D
    #tm=tau_m*td
    return 2*B*tm*(t-tm*(1-np.exp(-t/tm)))


############################################################
# noisy MSD, VACF for persistent random walk

def msd_prw_noise(tm, B, sig_loc, t):
    #td=1/D
    #tm=tau_m*td
    msd_theo = msd_prw(tm,B,t)
    msd_noise = np.zeros(len(msd_theo))
    msd_noise[1:] = msd_theo[1:] + 2 * sig_loc**2
    msd_noise[0] = msd_theo[0]
    return msd_noise


def vacf_noise_prw(tm, B, sig_loc, t, msd=msd_prw):
    msd_theo = msd(tm,B,t)
    msd_noise = np.zeros(len(msd_theo))
    msd_noise[1:] = msd_theo[1:] + 2*sig_loc**2
    msd_noise[0] = msd_theo[0]
    dt = t[1]-t[0]
    vacf_noisy = (np.roll(msd_noise,-1)[:-1]-2*msd_noise[:-1]+np.roll(msd_noise,1)[:-1])/(2*dt**2)
    vacf_noisy[0] = msd_noise[1]/dt**2

    inf_residues = np.where(vacf_noisy>1e140)[0] #check if residues (vacf**2) might lead to overflows
    if len(inf_residues)>0:
        #for indx in inf_residues:
        #    vacf_noisy[indx] = 1e140
        vacf_noisy = np.ones(len(vacf_noisy))*1e140
    
    return vacf_noisy

# MSD without noise with input B, tau or D, tau

def msd_prw_bt(tm,B,t):
    #td=1/D
    #tm=tau_m*td
    return 2*B*tm*(t-tm*(1-np.exp(-t/tm)))

def msd_prw_dt(tm,D,t):
    #td=1/D
    #tm=tau_m*td
    return 2*D*(t-tm*(1-np.exp(-t/tm)))
    
##########################################################################

def load_masked_matrix(lengths, compressed_data):
    #load masked matrix, which is saved as compressed array and array of lengths of each row in the matrix
    masked_array = np.ma.empty((len(lengths), max(lengths)))
    masked_array.mask = True
    sum_lengths = 0
    for j in range(len(lengths)):
        masked_array[j,:lengths[j]] = compressed_data[sum_lengths:sum_lengths + lengths[j]]
        sum_lengths += lengths[j]
    return masked_array


def lens_rows_masked_matrix(matrix, vacf=True):
    ntrj = len(matrix[:, 0])
    len_vector = np.zeros(ntrj)

    for i in range(ntrj):
        fun = matrix[i,:].compressed()
        len_vector[i] = len(fun)

    if vacf:
        len_vector += 1

    return len_vector

#######################################################################
# plotting functions
    
@mpltex.aps_decorator
def plot_single_cell(xall, dt, yall=[], cutoff=0, only_mean=False, cmean='k', msd=False, weight=False, xlabel='', ylabel='', title='', xlim=[0.01, 10], ylim=[0.01, 100], dpi=300, mean_legend=False, alpha=1):
    fig, ax = plt.subplots(1, 1, dpi=dpi)
    # x = xall
    if cutoff == 0:
        cutoff = len(xall[0,:])

    colors = [cm.jet(l) for l in np.linspace(0, 1, len(xall[:,0]))]
    j=0
    if not only_mean:
        for col in colors:
            xj = xall[j,:].compressed()
            t = np.linspace(0, dt*len(xj[:cutoff]), len(xj[:cutoff]))

            if msd:
                ax.plot(t[1:], xj[1:cutoff], c=col, alpha=alpha)
            else:
                ax.plot(t, xj[:cutoff], c=col, alpha=alpha)
            if len(yall) > 0:
                yj=yall[j,:].compressed()
                if len(yj[:cutoff]) != len(xj[:cutoff]):
                    print('lengths of x,y not equal')
                #t = np.linspace(0, dt*len(yj[:cutoff]), len(yj[:cutoff]))
                if msd:
                    ax.plot(t[1:], yj[1:cutoff], c=col, alpha=alpha)
                else:
                    ax.plot(t, yj[:cutoff], c=col, alpha=alpha)
                
            j+=1
    if len(yall) > 0:
        if weight:
            wav = 0.5*(weighted_av(xall)[:cutoff]+weighted_av(yall)[:cutoff])
        else:
            wav = 0.5*(np.mean(xall[:, :cutoff], axis=0) + np.mean(yall[:, :cutoff], axis=0))
        t = np.linspace(0, dt*len(wav), len(wav))
        if msd:
            print(np.any(np.isnan(wav)), np.any(np.isinf(wav)))
            ax.plot(t[1:], wav[1:], c=cmean, lw=3, label='mean')
        else:
            ax.plot(t, wav, c=cmean, lw=3, label='mean')
        
        
    else:
        if weight:
            wav = weighted_av(xall)[:cutoff]
        else:
            wav = np.mean(xall[:, :cutoff], axis=0)

        t = np.linspace(0, dt*len(wav), len(wav))
        if msd:
            ax.plot(t[1:], wav[1:], c=cmean, lw=3)
        else:
            ax.plot(t, wav, c=cmean, lw=3)
    
    if xlim[0] == xlim[1]:
        xlim[0] = np.min(t)
        xlim[1] = np.max(t)
    if ylim[0] == ylim[1]:
        ylim[0] = np.min(wav)/1.5
        ylim[1] = np.max(wav)*1.5

    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if mean_legend:
        ax.legend(loc=0)


@mpltex.acs_decorator
def plot_single_cell_axin(xall, dt, ax, yall=[], cutoff=0, only_mean=False, cmean='k', msd=False, weight=False, xlabel='', ylabel='', title='',
                          xlim=[0.01, 10], ylim=[0.01, 100], mean_legend=False, alpha=1, xlabelpad=None, ylabelpad=None, av_cut=0):

    if cutoff == 0:
        cutoff = len(xall[0,:])

    colors = [cm.jet(l) for l in np.linspace(0, 1, len(xall[:,0]))]
    j=0
    if not only_mean:
        for col in colors:
            xj = xall[j,:].compressed()
            t = np.linspace(0, dt*len(xj[:cutoff]), len(xj[:cutoff]))

            if msd:
                ax.plot(t[1:], xj[1:cutoff], c=col, alpha=alpha)
            else:
                ax.plot(t, xj[:cutoff], c=col, alpha=alpha)
            if len(yall) > 0:
                yj=yall[j,:].compressed()
                if len(yj[:cutoff]) != len(xj[:cutoff]):
                    print('lengths of x,y not equal')
                #t = np.linspace(0, dt*len(yj[:cutoff]), len(yj[:cutoff]))
                if msd:
                    ax.plot(t[1:], yj[1:cutoff], c=col, alpha=alpha)
                else:
                    ax.plot(t, yj[:cutoff], c=col, alpha=alpha)
                
            j+=1
    if len(yall) > 0:
        if weight:
            wav = 0.5*(weighted_av(xall)[:cutoff]+weighted_av(yall)[:cutoff])
        else:
            wav = 0.5*(np.mean(xall[:, :cutoff], axis=0) + np.mean(yall[:, :cutoff], axis=0))
        t = np.linspace(0, dt*len(wav), len(wav))
        if msd:
            print(np.any(np.isnan(wav)), np.any(np.isinf(wav)))
            if av_cut > 0:
                ax.plot(t[1:av_cut], wav[1:av_cut], c=cmean, lw=3)
            else:
                ax.plot(t[1:], wav[1:], c=cmean, lw=3)
        else:
            if av_cut > 0:
                ax.plot(t[:av_cut], wav[:av_cut], c=cmean, lw=3)
            else:
                ax.plot(t, wav, c=cmean, lw=3)
        
        
    else:
        if weight:
            wav = weighted_av(xall)[:cutoff]
        else:
            wav = np.mean(xall[:, :cutoff], axis=0)
            
        t = np.linspace(0, dt*len(wav), len(wav))
        if msd:
            if av_cut > 0:
                ax.plot(t[1:av_cut], wav[1:av_cut], c=cmean, lw=3)
            else:
                ax.plot(t[1:], wav[1:], c=cmean, lw=3)
        else:
            if av_cut > 0:
                ax.plot(t[:av_cut], wav[:av_cut], c=cmean, lw=3)
            else:
                ax.plot(t, wav, c=cmean, lw=3)
    
    if xlim[0] == xlim[1]:
        xlim[0] = np.min(t)
        xlim[1] = np.max(t)
    if ylim[0] == ylim[1]:
        ylim[0] = np.min(wav)/1.5
        ylim[1] = np.max(wav)*1.5

    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])
    ax.set_xlabel(xlabel, labelpad=xlabelpad)
    ax.set_ylabel(ylabel, labelpad=ylabelpad)
    ax.set_title(title)
    if mean_legend:
        ax.legend(loc=0)
    if msd:
        ax.loglog()


@mpltex.aps_decorator
def plot_single_cell_xy_multiple(xlist, ylist, cutoff=0, only_mean=False, cmean=['k'], weight=False, labels=[], xlabel='', ylabel='', title='', xlim=[0, 0], ylim=[0, 0], dpi=600, alpha=1):
    fig, ax = plt.subplots(1, 1, dpi=dpi)
    for i in range(len(xlist)):
        xall = xlist[i]
        yall = ylist[i]
        plot_xy(ax, xall, yall, cutoff=cutoff, only_mean=only_mean, cmean=cmean[i], weight=weight, label=labels[i], alpha=alpha)
    if xlim[0] == xlim[1]:
        xlim[0] = np.min(xall.compressed())
        xlim[1] = np.max(xall.compressed())
    if ylim[0] == ylim[1]:
        ylim[0] = np.min(yall.compressed())
        ylim[1] = np.max(yall.compressed())
        
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

def plot_xy(ax, x, y, cutoff=0, only_mean=False, cmean='k', weight=False, label='', alpha=1):
    if cutoff == 0:
        cutoff = len(x[0,:])
    colors = [cm.jet(l) for l in np.linspace(0, 1, len(x[:,0]))]
    j = 0
    lens = np.zeros(len(x[:,0]))
    if not only_mean:
        for col in colors:
            xj = x[j,:].compressed()
            yj = y[j,:].compressed()
            lens[j] = len(xj)

            ax.plot(xj[:cutoff], yj[:cutoff],c=col, alpha=alpha)
            j += 1
    else:
        for col in colors:
            xj = x[j,:].compressed()
            lens[j] = len(xj)
            j += 1
    
    if weight:
        wav = weighted_av(y)[:cutoff]
    else:
        wav = np.mean(y[:, :cutoff], axis=0)
    x_max_len = x[np.argmax(lens), :cutoff].compressed()
    #print(np.max(lens), lens[np.argmax(lens)], len(wav), len(x_max_len), lens[:4], wav[-1])
    ax.plot(x_max_len, wav[:int(np.max(lens))], lw=3, c=cmean, label=label)



@mpltex.aps_decorator
def plot_single_cell_multiple(xlist, dt, ylist=[], cutoff=0, only_mean=False, cmean=['k'], weight=False, msd=False, labels=[], xlabel='', ylabel='', title='', xlim=[0, 0], ylim=[0, 0], dpi=300, alpha=1):
    fig, ax = plt.subplots(1, 1, dpi=dpi)
    for i in range(len(xlist)):
        xall = xlist[i]
        if len(ylist) > 0:
            yall = ylist[i]
        else:
            yall = []
        plot_x(ax, dt, xall, yall=yall, cutoff=cutoff, only_mean=only_mean, cmean=cmean[i], weight=weight, msd=msd, label=labels[i], alpha=alpha)
    if xlim[0] == xlim[1]:
        xlim[0] = np.min(xall.compressed())
        xlim[1] = np.max(xall.compressed())
    if ylim[0] == ylim[1]:
        ylim[0] = np.min(yall.compressed())
        ylim[1] = np.max(yall.compressed())
        
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def plot_x(ax, dt, x, yall=[], cutoff=0, only_mean=False, cmean='k', weight=False, msd=False, label='', alpha=1):
    if cutoff==0:
        cutoff=len(x[0,:])

    colors = [cm.jet(l) for l in np.linspace(0, 1, len(x[:,0]))]
    j=0
    if not only_mean:
        for col in colors:
            xj=x[j,:].compressed()
            t = np.linspace(0, dt*len(xj[:cutoff]), len(xj[:cutoff]))

            if msd:
                ax.plot(t[1:], xj[1:cutoff], c=col, alpha=alpha)
            else:
                ax.plot(t, xj[:cutoff], c=col, alpha=alpha)
            if len(yall) > 0:
                yj=yall[j,:].compressed()
                if len(yj[:cutoff]) != len(xj[:cutoff]):
                    print('lengths of x,y not equal')
                #t = np.linspace(0, dt*len(yj[:cutoff]), len(yj[:cutoff]))
                if msd:
                    ax.plot(t[1:], yj[1:cutoff], c=col, alpha=alpha)
                else:
                    ax.plot(t, yj[:cutoff], c=col, alpha=alpha)
            j+=1

    if len(yall) > 0:
        if weight:
            wav = 0.5*(weighted_av(x)[:cutoff]+weighted_av(yall)[:cutoff])
        else:
            wav = 0.5*(np.mean(x[:, :cutoff], axis=0) + np.mean(yall[:, :cutoff], axis=0))
        t = np.linspace(0, dt*len(wav), len(wav))
        if msd:
            ax.plot(t[1:], wav[1:], c=cmean, lw=3, label=label)
        else:
            ax.plot(t, wav, c=cmean, lw=3, label=label)
        
        
    else:
        if weight:
            wav = weighted_av(x)[:cutoff]
        else:
            wav = np.mean(x[:, :cutoff], axis=0)
        t = np.linspace(0, dt*len(wav), len(wav))
        if msd:
            ax.plot(t[1:], wav[1:], c=cmean, lw=3)
        else:
            ax.plot(t, wav, c=cmean, lw=3)

@mpltex.aps_decorator
def plot_single_cell_xy(xall, yall, cutoff=0, only_mean=False, cmean='k', weight=False, xlabel='', ylabel='', title='', xlim=[0, 0], ylim=[0, 0], dpi=300, alpha=1):
    fig, ax = plt.subplots(1, 1, dpi=dpi)
    if cutoff == 0:
        cutoff = len(xall[0,:])

    colors = [cm.jet(l) for l in np.linspace(0, 1, len(xall[:,0]))]
    j = 0
    lens = np.zeros(len(xall[:,0]))
    if not only_mean:
        for col in colors:
            xj = xall[j,:].compressed()
            yj = yall[j,:].compressed()
            lens[j] = len(xj)

            ax.plot(xj[:cutoff], yj[:cutoff], c=col, alpha=alpha)
            j += 1
    else:
        for col in colors:
            xj = xall[j,:].compressed()
            lens[j] = len(xj)
            j += 1
    
    if weight:
        wav = weighted_av(yall)[:cutoff]
    else:
        wav = np.mean(yall[:, :cutoff], axis=0)
    x_max_len = xall[np.argmax(lens), :cutoff].compressed()
    #print(np.max(lens), lens[np.argmax(lens)], len(wav), len(x_max_len), lens[:4], wav[-1])
    ax.plot(x_max_len, wav[:int(np.max(lens))], lw=3, c=cmean)

    if xlim[0] == xlim[1]:
        xlim[0] = np.min(x_max_len)
        xlim[1] = np.max(x_max_len)
    if ylim[0] == ylim[1]:
        ylim[0] = np.min(wav)/1.5
        ylim[1] = np.max(wav)*1.5
        
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)




@mpltex.aps_decorator
def histo(xall, binn, ranging, yall=[], scatter=True, only_mean=False, cmean='k', rescale=False, xlabel='', ylabel='', title='', xlim=[0, 0], ylim=[0, 0], dpi=300):
    #do scatter plot of histogram of each individual row of xall (also yall if given)
    #or only of the overall normalized histogram (including all rows and columns) if only_mean is True
    fig, ax = plt.subplots(1, 1, dpi=dpi)
    xa=xall#np.copy(xall)
    ya=yall#np.copy(yall)
    ntrj=len(xa[:,0])
    lm=len(xa[0,:])
    vals=np.zeros((ntrj,binn))
    binpos=np.zeros((ntrj,binn))
    cols=[cm.jet(l) for l in np.linspace(0, 1, len(xa[:,0]))]
    if not only_mean:
        for i in range(ntrj):
            #x=xall[i,:]
            #x=np.array(x[x.mask==False])
            x=xa[i,:].compressed()
            if rescale:
                x = (x - np.mean(x)) / np.std(x)
            histval,histbin=np.histogram(x,bins=binn,range=ranging,density=True)
            vals[i,:]=histval
            binpos[i,:]=0.5*(histbin[:-1]+np.roll(histbin,-1)[:-1])
            if len(yall) > 0:
                y=ya[i,:].compressed()
                if rescale:
                    y = (y - np.mean(y)) / np.std(y)
                histvaly,histbiny=np.histogram(y,bins=binn,range=ranging,density=True)
                valsy=histvaly
                binposy=0.5*(histbiny[:-1]+np.roll(histbiny,-1)[:-1])
            if scatter:
                ax.scatter(binpos[i,:],vals[i,:],c=[cols[i]])
                if len(yall) > 0:
                    ax.scatter(binposy,valsy,c=[cols[i]])
        
    if not scatter:  
        return vals,binpos
    else:
        if len(yall) > 0:
            if rescale:
                xa_rescaled = combine_fun(scale_v, xa)
                ya_rescaled = combine_fun(scale_v, ya)
                histval,histbin=np.histogram(np.append(xa_rescaled.compressed(),ya_rescaled.compressed()),bins=binn, range=ranging, density=True)
            else:
                histval,histbin=np.histogram(np.append(xa.compressed(),ya.compressed()),bins=binn, range=ranging, density=True)
        else:
            if rescale:
                xa_rescaled = combine_fun(scale_v, xa)
                histval,histbin=np.histogram(xa_rescaled.compressed(),bins=binn, range=ranging, density=True)
            else:
                histval,histbin=np.histogram(xa.compressed(),bins=binn, range=ranging, density=True)

        binpos=0.5*(histbin[:-1]+np.roll(histbin,-1)[:-1])
        ax.plot(binpos, histval, c=cmean, lw=2)
        if rescale:
            ax.plot(binpos, Gauss(binpos, 0, 1), c='cyan', lw=2, ls='--')
        # else:
        #     curve_fit(Gauss, mean...)

    if xlim[0] == xlim[1]:
        xlim[0] = np.min(binpos)
        xlim[1] = np.max(binpos)
    if ylim[0] == ylim[1]:
        ylim[0] = np.min(histval)/1.1
        ylim[1] = np.max(histval)*1.1
        
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)



@mpltex.aps_decorator
def histo_ax(ax, xall, binn, ranging, yall=[], scatter=True, only_mean=False, cmean='k', rescale=False, xlabel='', ylabel='', title='', xlim=[0, 0], ylim=[0, 0]):
    #do scatter plot of histogram of each individual row of xall (also yall if given)
    #or only of the overall normalized histogram (including all rows and columns) if only_mean is True

    xa=xall#np.copy(xall)
    ya=yall#np.copy(yall)
    ntrj=len(xa[:,0])
    lm=len(xa[0,:])
    vals=np.zeros((ntrj,binn))
    binpos=np.zeros((ntrj,binn))
    cols=[cm.jet(l) for l in np.linspace(0, 1, len(xa[:,0]))]
    if not only_mean:
        for i in range(ntrj):
            #x=xall[i,:]
            #x=np.array(x[x.mask==False])
            x=xa[i,:].compressed()
            if rescale:
                x = (x - np.mean(x)) / np.std(x)
            histval,histbin=np.histogram(x,bins=binn,range=ranging,density=True)
            vals[i,:]=histval
            binpos[i,:]=0.5*(histbin[:-1]+np.roll(histbin,-1)[:-1])
            if len(yall) > 0:
                y=ya[i,:].compressed()
                if rescale:
                    y = (y - np.mean(y)) / np.std(y)
                histvaly,histbiny=np.histogram(y,bins=binn,range=ranging,density=True)
                valsy=histvaly
                binposy=0.5*(histbiny[:-1]+np.roll(histbiny,-1)[:-1])
            if scatter:
                ax.scatter(binpos[i,:],vals[i,:],c=[cols[i]])
                if len(yall) > 0:
                    ax.scatter(binposy,valsy,c=[cols[i]])
        
    if not scatter:  
        return vals,binpos
    else:
        if len(yall) > 0:
            if rescale:
                xa_rescaled = combine_fun(scale_v, xa)
                ya_rescaled = combine_fun(scale_v, ya)
                histval,histbin=np.histogram(np.append(xa_rescaled.compressed(),ya_rescaled.compressed()),bins=binn, range=ranging, density=True)
            else:
                histval,histbin=np.histogram(np.append(xa.compressed(),ya.compressed()),bins=binn, range=ranging, density=True)
        else:
            if rescale:
                xa_rescaled = combine_fun(scale_v, xa)
                histval,histbin=np.histogram(xa_rescaled.compressed(),bins=binn, range=ranging, density=True)
            else:
                histval,histbin=np.histogram(xa.compressed(),bins=binn, range=ranging, density=True)

        binpos=0.5*(histbin[:-1]+np.roll(histbin,-1)[:-1])
        ax.plot(binpos, histval, c=cmean, lw=2)
        if rescale:
            ax.plot(binpos, Gauss(binpos, 0, 1), c='cyan', lw=2, ls='--')
        # else:
        #     curve_fit(Gauss, mean...)

    if xlim[0] == xlim[1]:
        xlim[0] = np.min(binpos)
        xlim[1] = np.max(binpos)
    if ylim[0] == ylim[1]:
        ylim[0] = np.min(histval)/1.1
        ylim[1] = np.max(histval)*1.1
        
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)




@mpltex.aps_decorator
def plot_vel_rescaled(vels_list, bin_number=20, dpi=300, xlabel=r'$(v_{ind}-\overline{v_{ind}})/\sigma_{ind}$', ylabel=r'probability', title='', xlim=[-5, 5], ylim=[0.01, 1], yscale=''):

    fig, ax = plt.subplots(1, 1, dpi=dpi)


    if len(vels_list) == 1:
        vels_list = [combine_fun(scale_v, vels_list[0])]
        v_scale_all = vels_list[0].compressed()
    elif len(vels_list) == 2:
        vels_list = [combine_fun(scale_v, vels_list[0]), combine_fun(scale_v, vels_list[0])]
        v_scale_all = np.append(vels_list[0].compressed(), vels_list[1].compressed())
    
    vals_scale_all, bins_scale_all = np.histogram(v_scale_all, bins=bin_number, density=True)
    mean_bins_scale_all = 0.5 * (bins_scale_all + np.roll(bins_scale_all, -1))[:-1]


    for j in range(len(vels_list[0][:, 0])):
        if len(vels_list) == 1:
            vx = vels_list[0][j, :].compressed()
        elif len(vels_list) == 2:
            vx = np.append(vels_list[0][j, :].compressed(), vels_list[1][j, :].compressed())
        
        val_prob_vxyw_scale, bins_vxyw_scale = np.histogram((vx-np.mean(vx))/np.std(vx), bins=bin_number, density=True) # [j, :]
        mean_bins_scale = 0.5 * (bins_vxyw_scale + np.roll(bins_vxyw_scale, -1))[:-1]

        ax.scatter(mean_bins_scale, val_prob_vxyw_scale, c='cyan')

    
    ax.scatter(mean_bins_scale_all, vals_scale_all, c='k')
    ax.plot(mean_bins_scale_all, np.exp(-mean_bins_scale_all**2/2)/np.sqrt(2*np.pi), c='orange',lw=2, ls='dashed')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    if yscale != '':
        ax.set_yscale(yscale)

    plt.show()

@mpltex.acs_decorator
def hist_lengths(path_lens, path_distr, dt, bins=20, range=0, t_unit=r'[$\rm{min}$]', min_vals=[0.25, 0.25, 0.01], max_vals=[4, 4, 1], filt=True):
    plt.figure(dpi=300)
    length_vec = np.load(path_lens)
    distr = np.load(path_distr)
    if filt:
        cond_err = np.where(np.logical_and(np.logical_and(np.logical_and(np.logical_and(distr[:,2] > 1.5 * min_vals[-1], distr[:,2] < 0.99 * max_vals[-1]),
                                np.logical_not(np.isnan(distr[:, 1]))), distr[:, 1] < 1e50),
                                np.logical_and(np.logical_not(np.isnan(distr[:, 0])), distr[:, 0] < 1e50)))[0]
        length_vec = length_vec[cond_err]
    if range == 0:
        range = np.array([np.min(length_vec), np.max(length_vec)]) * dt
    else:
        range = np.array(range) * dt

    plt.hist(length_vec * dt, bins=bins, range=range)
    plt.xlabel(r'$\rm{trajectory}$ ' + r'$\rm{length}$ ' + t_unit, fontsize=20)
    plt.ylabel(r'$\rm{occurrences}$', fontsize=20)
    plt.show()

def scatter_prw(optis, opt_av, tmmax=1e5, Dmax=1000, show_mean=True, yunit='min', xunit='$\mu m^2/min$', log_ax='loglog', marker='o', fontsize=14):
    plt.rc('font', size=fontsize)
    plt.scatter(optis[:, 1], optis[:, 0], marker=marker)
    if show_mean:
        # plt.scatter(np.mean(optis[:,0]),np.mean(optis[:,1]),c='k')
        plt.scatter(np.mean(optis[:, 1]), np.mean(
            optis[:, 0]), edgecolor='orange', fc='none', s=200, lw=2)
    plt.scatter(opt_av[1], opt_av[0], edgecolor='r', fc='none', s=200, lw=2)

    plt.ylabel(r'$\tau$'+' ['+yunit+']')
    plt.xlabel('D'+' ['+xunit+']')
    plt.ylim(0.5*np.min(optis[:, 0]), 1.3*tmmax)
    plt.xlim(0.5*np.min(optis[:, 1]), 1.3*Dmax)
    if log_ax == 'loglog':
        plt.loglog()
    elif log_ax == 'logx':
        plt.semilogx()
    elif log_ax == 'logy':
        plt.semilogy()


def prob_distr(vx,bins):
    n=len(vx[:,0])
    probv=np.zeros((n,bins))
    bin_vals=np.zeros((n,bins+1))
    for i in range(n):
        x=vx[i,:]
        x=x.compressed()
        probv[i,:],bin_vals[i,:]=np.histogram(x,bins=bins,density=True)
    
    return probv,bin_vals

def scatter_prob(prob,bin_vals):
    n=len(prob[:,0])
    colors = [cm.jet(x) for x in np.linspace(0, 1, n)]
    bin_val=np.zeros((n,len(prob[0,:])))
    for j in range(n):
        bin_val[j,:]=(bin_vals[j,:-1]+np.roll(bin_vals[j,:],-1)[:-1])/2
        plt.scatter(bin_val[j,:],prob[j,:],color=colors[j])
        
        
def histo_old(xall,binn,scatter=True):
    ntrj=len(xall[:,0])
    lm=len(xall[0,:])
    vals=np.zeros((ntrj,binn))
    binpos=np.zeros((ntrj,binn))
    for i in range(ntrj):
        x=xall[i,:]
        x=np.array(x[x.mask==False])
        histval,histbin=np.histogram(x,bins=binn,density=True)
        vals[i,:]=histval
        binpos[i,:]=0.5*(histbin[:-1]+np.roll(histbin,-1)[:-1])
        if scatter:
            plt.scatter(binpos[i,:],vals[i,:])
    if not scatter:  
        return vals,binpos
