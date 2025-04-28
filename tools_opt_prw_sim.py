import numpy as np
from scipy.optimize import least_squares, curve_fit

from scipy import io
from numba import njit
from motility import *
from chi2_tests import covariance_diff_perm_test, multivariate_DD_test, dist_perm_test, multivariate_test_mean_cov_gauss, multivariate_phantasy_test
from joblib import Parallel, delayed
from correlation_functions_expos import msd_osc_sincos, msd_osc_analytic, vacf_osc_sincos, vacf_osc_analytic
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import mpltex

import glob

########################################
# euler integrators and according functions to produce trajectories with or without memory

@njit
def fm(t,y, tau_m,tau_g, h):    
    return np.array([y[1],(1/tau_m)*(y[2]),-(1/tau_g)*(-np.random.normal(0.0, np.sqrt(2/h)) + y[1] + y[2])])

@njit
def f(t,y,tau_m,tau_g,h):
    return np.array([(y[1]),(1/tau_m)*(-y[1]+np.random.normal(0.0,np.sqrt(2.0/h)))])

@njit
def msd_prw(tm, B, t):
    #td=1/D
    #tm=tau_m*td
    return 2 * B * tm * (t-tm * (1 - np.exp(-t/tm)))

@njit
def msd_diffusion(D, t):
    return 2 * D * t

@njit
def msd_diffusion_noise(D, sig_loc, t):
    msd = 2 * D * t + 2 * sig_loc**2
    msd[0] = 0
    return msd
    
@njit
def euler(param, h, tmax, tau_m, tau_g, f):
    t=0
    i=0
    y=np.zeros((param, int(tmax/h)))
    v0=np.random.normal(0.0,np.sqrt(1/tau_m))
    y[1,0]=v0
    while i<(tmax/h)-1:
        y[:,i+1] = y[:,i] + h * f(t,y[:,i], tau_m, tau_g, h)
        t += h
        i += 1
    return y[0,:]

@njit
def euler_t(h, tmax, B, tau_m, B_rate=0):
    t = 0
    y = np.zeros((2, int(tmax/h)))
    v0 = np.random.normal(0.0, np.sqrt(B))
    y[1,0] = v0

    for i in range(int((tmax/h)-1)):
        Bt = B * np.exp(B_rate * t)
        y[:,i+1] = y[:,i] + h * f_B(y[:,i], Bt, tau_m, h)
        t += h
    return y[0,:]

@njit
def f_B(y, B, tau_m, h):
    return np.array([(y[1]), -y[1] / tau_m + np.random.normal(0.0, np.sqrt(2 * B / h / tau_m))])



@njit
def euler_diffusion(D, h, tmax):
    y = np.zeros(int(tmax / h))
    for i in range(len(y) - 1):
        y[i+1] = y[i] + np.random.normal(0.0, np.sqrt(2 * D * h))
    return y


######################################################################################################################
@njit
def f_osc(param_list, y, h):
    # param list [a, b, f, tau] has to be an array for numba to work
    # for kernel of form a * delta + b * exp(-t/tau) * (cos(ft) + sin(ft) / (f * tau))
    # units of B
    bracket = 1 / param_list[3]**2 + param_list[2]**2
    return np.array([y[1],  -param_list[0] * y[1] + param_list[1] * (y[2] - y[0]) + np.sqrt(param_list[0]) * np.random.normal(0, np.sqrt(2/h)),
                     y[3], -2 * y[3] / param_list[3] + bracket * (y[0] - y[2]) + np.sqrt(2 * bracket / (param_list[3] * param_list[1])) * np.random.normal(0, np.sqrt(2/h)) ])


@njit
def euler_mem(param_list, h, tmax, f, dim=4):

    y = np.zeros((dim, int(tmax/h)))
    v0 = np.random.normal(0, 1) #np.sqrt(param_list[-2])
    y[1,0] = v0

    for i in range(int(tmax/h) - 1):
        y[:, i+1] = y[:, i] + h * f(param_list, y[:, i], h)

    return y[0, :]

def test_euler_mem(params = np.array([100, 5000, 60, 1, 1]), h=0.0001, tmax=1000, idx=10000):
    # parameter B(=params[4]) has to be one here! velocity is scaled with respect to B (=msqv)
    pos = euler_mem(params, h, tmax, f_osc)
    vels = np.diff(pos) / h
    vacf = correlation(vels)

    t = np.arange(len(vacf)) * h
    kernel = kernel_1dcc(vacf[:int(idx+2)], h)
    
    plt.plot(t[1:len(kernel[:idx])], kernel[1:idx])
    plt.plot(t[1:], params[1] * np.exp(-t[1:]/params[3]) * (np.cos(params[2] * t[1:]) + np.sin(params[2] * t[1:]) / (params[2] * params[3])))
    print('ratio kernel at 0 compared to theory', kernel[0]/params[0]/2 * h)
    plt.xlabel(r't')
    plt.ylabel(r'$\Gamma(t)$')
    plt.show()

    plt.plot(t, vacf[:len(t)])
    plt.plot(t, vacf_osc_sincos(*params, t), c='r', ls='--')
    plt.xlabel(r't')
    plt.ylabel(r'$C_{vv}(t)$')
    plt.show()

    plt.hist(vels, bins=200, density=True)
    plt.plot(np.arange(-5*np.sqrt(params[4]), 5*np.sqrt(params[4]), 0.2), Gauss(np.arange(-5*np.sqrt(params[4]), 5*np.sqrt(params[4]), 0.2), 0, np.sqrt(params[4])))
    plt.xlabel('v')
    plt.ylabel('p(v)')
    plt.show()

##########################################################################################################################

# optimization for persistent random walk (PRW)

def vacf_noise_prw(tm, B, sig_loc, t, msd=msd_prw):
    # theoretical prediction for VACF from persistent random walk model at discrete time points t for Gaussian localization noise of width sig_loc
    msd_theo = msd(tm,B,t)
    msd_noise = np.zeros(len(msd_theo))
    msd_noise[1:] = msd_theo[1:] + 2 * sig_loc**2
    msd_noise[0] = msd_theo[0]
    dt = t[1] - t[0]
    vacf_noisy = (np.roll(msd_noise,-1)[:-1] - 2 * msd_noise[:-1] + np.roll(msd_noise,1)[:-1]) / (2*dt**2)
    vacf_noisy[0] = msd_noise[1] / dt**2

    inf_residues = np.where(vacf_noisy > 1e140)[0] #check if residues (vacf**2) might lead to overflows
    if len(inf_residues) > 0:
        #for indx in inf_residues:
        #    vacf_noisy[indx] = 1e140
        vacf_noisy = np.ones(len(vacf_noisy))*1e140
    
    return vacf_noisy

def sigma_vacf_prw(args, vacf_exp, t, msd=msd_prw):
    # residual array for optimization
    #args of the form [tau_m,D,sig_loc]
    residual_vec = vacf_noise_prw(args[0],args[1],args[2],t,msd=msd) - vacf_exp
    if np.any(np.isinf(residual_vec)) or np.any(np.isnan(residual_vec)):
        residual_vec = np.ones(len(residual_vec)) * 1e140
    return residual_vec

def optimize_vacf_prw(vacf_exp, dt=0.5, tm_max=1e5, tm_min=0.001, Bmax=1000, loc_errmax=10, msd_fun=msd_prw, errmin=0.001):
    # fit a single VACF by PRW model with localization noise
    t = np.arange(0, len(vacf_exp) + 1) * dt

    tm0 = np.random.rand() * (tm_max - tm_min) + tm_min #number between min and max
    err0 = np.random.rand() * (loc_errmax - errmin) + errmin
    if err0 < errmin:
        err0 += errmin # security if something went wrong before
    B0 = np.random.rand() * Bmax
    if B0 < 0.01:
        B0 += 0.01
    if tm0 < tm_min:
        tm0 += tm_min
    x0 = np.array([tm0, B0, err0])
    lb = np.array([tm_min, 0.01, errmin])
    upb = np.array([tm_max, Bmax, loc_errmax])
    
    res = least_squares(lambda x: sigma_vacf_prw(x,vacf_exp,t,msd=msd_fun), x0,
                               bounds=(lb,upb))

    return res.x, res.cost

def optimize_all_cells_prw(vacf_all, av=10, dt=0.5, tm_max=1e5, tm_min=0.001, Bmax=10000, loc_errmax=10, msd_func=msd_prw, index=10, errmin=0.001, threads=1):
    # fit multiple VACFs each by a PRW model with localization noise
    ntrj=len(vacf_all[:,0])
    opt_params=np.zeros((ntrj,3))
    av_params=np.zeros((ntrj,3))
    cost_ar=np.zeros(ntrj)
    av_cost_ar=np.zeros(ntrj)
    for i in tqdm(range(ntrj)):
        vacf=vacf_all[i,:].compressed()
        vacf = vacf[:index]  #vacf[:int(len(vacf)/2)]

        res_cost_list = Parallel(n_jobs=threads)(delayed(optimize_vacf_prw)(vacf, dt=dt, tm_max=tm_max, tm_min=tm_min, Bmax=Bmax, loc_errmax=loc_errmax, msd_fun=msd_func, errmin=errmin) for j in range(av))

        results = np.array(res_cost_list, dtype=object)[:,0]
        costs = np.array(res_cost_list, dtype=object)[:,1]    
        avres = np.mean(results, axis=0)
        
        avcost = np.mean(costs)
        best_index = np.argmin(costs)
        costb = costs[best_index]
        resopt = results[best_index]
        
        av_cost_ar[i]=avcost
        cost_ar[i]=costb
        opt_params[i,:]=resopt
        av_params[i,:]=avres
        
    return opt_params, cost_ar, av_params, av_cost_ar


###################################################################################################################
# simulate different variances for PRW model

def sim_var_prw(exp_distr, scaling, n_realization, n_steps, dt, path_out, name, min_vals, max_vals, av=100, index_vacf=10, threads=16, mean=False, dim=1, small_sim_step=False):
    cond_err = filt_distr(exp_distr, min_vals[-1], max_vals[-1], max_val=[1e50, 1e50], min_factor=1.5, max_factor=0.99) # filter non-physical fit params
    # takes exp_distr parameters and scales the covariance according to scaling then simulate PRW model using a Gaussian parameter distribution with the scaled covariance
    distr_exp = exp_distr[cond_err, :]
    mean_data = np.mean(distr_exp, axis=0)
    mean_str = ''
    if not mean:
        mean_data = np.median(distr_exp, axis=0) # if not mean, median is used as central point for underlying Gaussian distr
        mean_str = '_med'

    cov_data = np.cov(distr_exp.T)

    ltrjt = np.ones(len(distr_exp[:,0])) * n_steps[cond_err]
    simulation_step = 0.05 * dt

    scale_str = str(int(100 * scaling)) + 'percent'
    if scaling == 1:
        scale_str = 'full'
        if small_sim_step:
            simulation_step = 0.1 * np.min(distr_exp[:, 0]) # use minimal tau_m * 0.1 as simulation step to ensure no infs or nans in simulation
        xall_list = Parallel(n_jobs=threads)(delayed(cre_trjs_prw)(ltrjt, 1 / (distr_exp[:, 0] * distr_exp[:, 1]), distr_exp[:, 0], dt, distr_exp[:, 2], h=simulation_step) for j in range(n_realization))
        if dim == 2:
            yall_list = Parallel(n_jobs=threads)(delayed(cre_trjs_prw)(ltrjt, 1 / (distr_exp[:, 0] * distr_exp[:, 1]), distr_exp[:, 0], dt, distr_exp[:, 2], h=simulation_step) for j in range(n_realization))


    elif scaling == 0:
        tm_use = mean_data[0] * np.ones(len(distr_exp[:, 0]))
        B_use = mean_data[1] * np.ones(len(distr_exp[:, 0]))
        err_use = mean_data[2] * np.ones(len(distr_exp[:, 0]))
        td_use = 1 / (tm_use * B_use)
        if small_sim_step:
            simulation_step = 0.1 * np.min(tm_use) # use minimal tau_m * 0.1 as simulation step to ensure no infs or nans in simulation
        xall_list = Parallel(n_jobs=threads)(delayed(cre_trjs_prw)(ltrjt, td_use, tm_use, dt, err_use, h=simulation_step) for j in range(n_realization))
        if dim ==2:
            yall_list = Parallel(n_jobs=threads)(delayed(cre_trjs_prw)(ltrjt, td_use, tm_use, dt, err_use, h=simulation_step) for j in range(n_realization))

    else:
        sim_data = np.abs(np.random.multivariate_normal(mean_data, scaling * cov_data, (len(distr_exp[:,0]), n_realization)))
        if small_sim_step:
            simulation_step = 0.1 * np.min(sim_data[:, 0]) # use minimal tau_m * 0.1 as simulation step to ensure no infs or nans in simulation
        xall_list = Parallel(n_jobs=threads)(delayed(cre_trjs_prw)(ltrjt, 1 / (sim_data[:, j][:, 0] * sim_data[:, j][:, 1]), sim_data[:, j][:, 0], dt, sim_data[:, j][:, 2], h=simulation_step) for j in range(n_realization))
        if dim == 2:
            yall_list = Parallel(n_jobs=threads)(delayed(cre_trjs_prw)(ltrjt, 1 / (sim_data[:, j][:, 0] * sim_data[:, j][:, 1]), sim_data[:, j][:, 0], dt, sim_data[:, j][:, 2], h=simulation_step) for j in range(n_realization))
        for k in range(10):
            np.save(path_out + '/' + name + 'origdistr' + str(int(k)) + mean_str + '_idx' + str(int(index_vacf)) + '_var' + scale_str + '_av' + str(int(av)), sim_data[:, k])
        
    max_exp = np.max(distr_exp, axis=0)
    for i in range(n_realization):
        if dim ==2:
            vxall = combine_fun(lambda x: np.diff(x) / dt, xall_list[i][0], lmodified=1)
            vyall = combine_fun(lambda x: np.diff(x) / dt, yall_list[i][0], lmodified=1)
            vxacf_all = combine_fun(correlation, vxall)
            vyacf_all = combine_fun(correlation, vyall)
            vacf_all = 0.5 * (vxacf_all + vyacf_all)
        elif dim == 1:
            vall = combine_fun(lambda x: np.diff(x) / dt, xall_list[i][0], lmodified=1)
            vacf_all = combine_fun(correlation, vall)
        if np.any(np.isnan(vacf_all.compressed())) or np.any(np.isinf(vacf_all.compressed())):
            if scaling == 0:
                print('simulated trajectory leads to inf or nan VACF', i, simulation_step)
            elif scaling == 1:
                print('simulated trajectory leads to inf or nan VACF for params taum, B, sig_loc: ', distr_exp[np.where(np.isnan(vacf_all).any(axis=1))[0], :], i, simulation_step)
            else:
                print('simulated trajectory leads to inf or nan VACF for params taum, B, sig_loc: ', sim_data[:, i][np.where(np.isnan(vacf_all).any(axis=1))[0], :], i, simulation_step)

        optis, costs, avres, avcosts = optimize_all_cells_prw_ind(vacf_all, max_vals, min_vals, av=av, dt=dt, msd_func=msd_prw, index=index_vacf, threads=threads)
        cond_err_optis = filt_distr(optis, min_vals[-1], max_vals[-1], max_val=max_exp[:-1], min_factor=1.5, max_factor=0.99)
        distr_optis = optis[cond_err_optis, :]
        
        np.save(path_out + '/' + name + str(int(i)) + mean_str + '_idx' + str(int(index_vacf)) + '_var' + scale_str + '_av' + str(int(av)), distr_optis) # + '_steps' + str(int(n_steps))
        print(i)

        

###################################################################################################################
# simulate scaled distribution around median/mean

def sim_scale_prw(exp_distr, scaling, n_realization, n_steps, dt, path_out, name, min_vals, max_vals, av=100, index_vacf=10, threads=16, mean=False, dim=1, small_sim_step=False):
    #scale whole distribution instead of assuming Gauss
    cond_err = filt_distr(exp_distr, min_vals[-1], max_vals[-1], max_val=[1e50, 1e50], min_factor=1.5, max_factor=0.99)# filter non-physical fit params
    distr_exp = exp_distr[cond_err, :]
    mean_data = np.mean(distr_exp, axis=0)
    mean_str = ''
    if not mean:
        mean_data = np.median(distr_exp, axis=0) # if not mean, median is used as central point for underlying Gaussian distr
        mean_str = '_med'

    cov_data = np.cov(distr_exp.T)

    ltrjt = np.ones(len(distr_exp[:,0])) * n_steps[cond_err]
    simulation_step = 0.05 * dt

    scale_str = str(int(100 * scaling)) + 'percent'
    if scaling == 1:
        scale_str = 'full'
        if small_sim_step:
            simulation_step = 0.1 * np.min(distr_exp[:, 0]) # use minimal tau_m * 0.1 as simulation step to ensure no infs or nans in simulation
        xall_list = Parallel(n_jobs=threads)(delayed(cre_trjs_prw)(ltrjt, 1 / (distr_exp[:, 0] * distr_exp[:, 1]), distr_exp[:, 0], dt, distr_exp[:, 2], h=simulation_step) for j in range(n_realization))
        if dim == 2:
            yall_list = Parallel(n_jobs=threads)(delayed(cre_trjs_prw)(ltrjt, 1 / (distr_exp[:, 0] * distr_exp[:, 1]), distr_exp[:, 0], dt, distr_exp[:, 2], h=simulation_step) for j in range(n_realization))

    elif scaling == 0:
        tm_use = mean_data[0] * np.ones(len(distr_exp[:, 0]))
        B_use = mean_data[1] * np.ones(len(distr_exp[:, 0]))
        err_use = mean_data[2] * np.ones(len(distr_exp[:, 0]))
        td_use = 1 / (tm_use * B_use)
        if small_sim_step:
            simulation_step = 0.1 * np.min(tm_use) # use minimal tau_m * 0.1 as simulation step to ensure no infs or nans in simulation
        xall_list = Parallel(n_jobs=threads)(delayed(cre_trjs_prw)(ltrjt, td_use, tm_use, dt, err_use, h=simulation_step) for j in range(n_realization))
        if dim == 2:
            yall_list = Parallel(n_jobs=threads)(delayed(cre_trjs_prw)(ltrjt, td_use, tm_use, dt, err_use, h=simulation_step) for j in range(n_realization))
        

    else:
        sim_data = distr_exp * np.sqrt(scaling) + (1 - np.sqrt(scaling)) * mean_data
        if small_sim_step:
            simulation_step = 0.1 * np.min(sim_data[:, 0]) # use minimal tau_m * 0.1 as simulation step to ensure no infs or nans in simulation
        xall_list = Parallel(n_jobs=threads)(delayed(cre_trjs_prw)(ltrjt, 1 / (sim_data[:, 0] * sim_data[:, 1]), sim_data[:, 0], dt, sim_data[:, 2], h=simulation_step) for j in range(n_realization))
        if dim == 2:
            yall_list = Parallel(n_jobs=threads)(delayed(cre_trjs_prw)(ltrjt, 1 / (sim_data[:, 0] * sim_data[:, 1]), sim_data[:, 0], dt, sim_data[:, 2], h=simulation_step) for j in range(n_realization))
    
        
    max_exp = np.max(distr_exp, axis=0)
    for i in range(n_realization):
        if dim ==2:
            vxall = combine_fun(lambda x: np.diff(x) / dt, xall_list[i][0], lmodified=1)
            vyall = combine_fun(lambda x: np.diff(x) / dt, yall_list[i][0], lmodified=1)
            vxacf_all = combine_fun(correlation, vxall)
            vyacf_all = combine_fun(correlation, vyall)
            vacf_all = 0.5 * (vxacf_all + vyacf_all)
        elif dim == 1:
            vall = combine_fun(lambda x: np.diff(x) / dt, xall_list[i][0], lmodified=1)
            vacf_all = combine_fun(correlation, vall)

        if np.any(np.isnan(vacf_all.compressed())) or np.any(np.isinf(vacf_all.compressed())):
            print('simulated trajectory leads to inf or nan VACF')

        optis, costs, avres, avcosts = optimize_all_cells_prw_ind(vacf_all, max_vals, min_vals, av=av, dt=dt, msd_func=msd_prw, index=index_vacf, threads=threads)
        cond_err_optis = filt_distr(optis, min_vals[-1], max_vals[-1], max_val=max_exp[:-1], min_factor=1.5, max_factor=0.99)
        distr_optis = optis[cond_err_optis, :]
        
        np.save(path_out + '/' + name + str(int(i)) + mean_str + '_idx' + str(int(index_vacf)) + '_var' + scale_str + '_av' + str(int(av)), distr_optis) # + '_steps' + str(int(n_steps))
        print(i)
        

def lens_rows_masked_matrix(matrix, vacf=True):
    ntrj = len(matrix[:, 0])
    len_vector = np.zeros(ntrj)

    for i in range(ntrj):
        fun = matrix[i,:].compressed()
        len_vector[i] = len(fun)

    if vacf:
        len_vector += 1

    return len_vector

#####################################################################################################################
# test how often realization is accepted to give same distribution as experiment
# if number of rejections is below n_real * alpha (which is the expected number of rejection by chance for the same covariances),
# accept this scaling of covariance as true parameter covariance

def cov_test_scaling(exp_distr, scaling, n_realization, path_out, name, min_vals=[0.25, 0.25, 0.01], max_vals=[4, 4, 2.1],
                        index_vacf=10, mean=False, n_perm=10000, alpha=0.05, av=1000, test='', filt_err=True, compare_dim=-1, n_steps=100):
    if filt_err:
        cond_err = filt_distr(exp_distr, min_vals[-1], max_vals[-1], max_val=[1e50, 1e50], min_factor=1.5, max_factor=0.99)
        # filter non-physical fit params
    else:
        cond_err = filt_distr(exp_distr, min_vals[-1], max_vals[-1], max_val=[1e50, 1e50], min_factor=0.9, max_factor=1.1)
        # adjust min and max factor such that errors are not filtered, but only nans and infs
    distr_exp = exp_distr[cond_err, :]

    if compare_dim == -1:
        compare_dim = slice(0, len(distr_exp[0, :])) # if one wants to test only certain parameters (for instance without localization error)

    max_exp = np.max(distr_exp, axis=0)
    mean_str = ''
    if not mean:
        mean_str = '_med'

    scale_str = str(int(100 * scaling)) + 'percent'
    if scaling == 1:
        scale_str = 'full'
        # to clarify that full distribution is used and not drawn from Gaussian with same covariance!

    p_sim_var = np.zeros(n_realization)
    for i in range(n_realization):
        try:
            # distr_sim = np.load(f'{path_out}/{name}{int(i)}{mean_str}_idx{int(index_vacf)}_var{scale_str}.npy')
            try:
                distr_sim = np.load(path_out + '/' + name + str(int(i)) + mean_str + '_idx' + str(int(index_vacf)) + '_var' + scale_str + '.npy')
            except:
                distr_sim = np.load(path_out + '/' + name + str(int(i)) + mean_str + '_idx' + str(int(index_vacf)) + '_var' + scale_str + '_av' + str(int(av)) + '.npy')
        except:
            distr_sim = np.load(path_out + '/' + name + str(int(i)) + mean_str + '_idx' + str(int(index_vacf)) + '_steps' + str(int(n_steps)) + '_var' + scale_str + '.npy')

        if filt_err:
            cond_err_sim = filt_distr(distr_sim, min_vals[-1], max_vals[-1], max_val=max_exp[:-1] * 1e3, min_factor=1.5, max_factor=0.99)
            # filter non-physical fit params
        else:
            cond_err_sim = filt_distr(distr_sim, min_vals[-1], max_vals[-1], max_val=max_exp[:-1] * 1e3, min_factor=0.9, max_factor=1.1)
            # adjust min and max factor such that errors are not filtered, but only nans and too large values compared to experiment
        distr_sim = distr_sim[cond_err_sim, :]


        if test == 'cov':
            p_sim_var[i] = covariance_diff_perm_test(distr_exp[:, compare_dim], distr_sim[:, compare_dim], n_perm, off_diag=True)
        elif test == 'DD' or test == 'dd':
            p_sim_var[i] = multivariate_DD_test(distr_exp[:, compare_dim], distr_sim[:, compare_dim], n_perm=n_perm)
        elif test == 'dist' or test == 'mean' or test == 'median':
            p_sim_var[i] = dist_perm_test(distr_exp[:, compare_dim], distr_sim[:, compare_dim], n_perm, test=test)
        elif test == 'gauss':
            p_sim_var[i] = multivariate_test_mean_cov_gauss(distr_exp[:, compare_dim], distr_sim[:, compare_dim], alpha=alpha)
        elif test == 'interdist' or test == 'inter':
            p_sim_var[i] = multivariate_phantasy_test(distr_exp[:, compare_dim], distr_sim[:, compare_dim], n_perm)
        else:
            p_sim_var[i] = covariance_diff_perm_test(distr_exp[:, compare_dim], distr_sim[:, compare_dim], n_perm, off_diag=False)


    n_rejected = len(np.where(p_sim_var < alpha)[0])
    if n_rejected < int(alpha * n_realization + 1):
        print('covariance scale ' + scale_str + ' is accepted at alpha level', alpha)
        print('rate of rejection: ', n_rejected / n_realization)
    else:
        print('covariance scale ' + scale_str + ' rejected at alpha level', alpha)
        print('rate of rejection: ', n_rejected / n_realization)
    np.save(path_out + '/' + test + name + 'pvals' + mean_str + '_idx' + str(int(index_vacf)) + '_var' + scale_str + '_nreal' + str(int(n_realization)), p_sim_var)

    if len(max_vals) == 2:
        plot_distr_sim(distr_exp, distr_sim, max_vals, min_vals, name + '_var' + scale_str, labels=[r'$D$', r'$\sigma_{\rm{loc}}$'], units=[r' [$\mu m^2/s$]', r' [$\mu m$]'], filt_err=filt_err)
    elif len(max_vals) == 3:
        plot_distr_sim(distr_exp, distr_sim, max_vals, min_vals, name + '_var' + scale_str, filt_err=filt_err)

    # if scaling != 1 and scaling != 0:
    #     plot_distr_sim(distr_exp, sim_data, max_vals, min_vals, name + '_var' + scale_str +' orig distr', labels=[r'$D$', r'$\sigma_{\rm{loc}}$'], units=[r' [$\mu m^2/s$]', r' [$\mu m$]'], filt_err=filt_err)


def filt_distr(distr, min_err, max_err, max_val=[1e50, 1e50], min_factor=1.5, max_factor=0.99):
    '''filter distribution for nan values, inf values and localization errors that are too large or too small'''
    cond_err = np.where(np.logical_not(np.isnan(distr).any(axis=1)) & np.all(distr[:, :len(max_val)] < max_val, axis=1) & (distr[:, -1] > min_factor * min_err) & (distr[:, -1] < max_factor * max_err))[0]

    return cond_err

import re

def load_pvals(path, name, mean_str='_med', test='', idx=10, n_real=100, alpha=0.05, hist=False):
    '''load all p-values of simulation realizations in path containing name
        printed values <= alpha mean the distributions are accepted to be drawn from the same distribution as the experimental distribution'''
    pval_list = []
    file_list = []
    match_list = []
    if test == '':
        path_str = path + '/' + test + name + 'pvals' + mean_str + '_idx' + str(int(idx))+ '*_nreal' + str(int(n_real)) + '*.npy'

    else:
        path_str = path + '/*' + test + name + 'pvals' + mean_str + '_idx' + str(int(idx))+ '*_nreal' + str(int(n_real)) + '*.npy'

    # if no pvals found, try to load pvals with * in case something is missing in name string
    if len(glob.glob(path_str)) == 0:
        print('name is not precise enough, trying to load pvals with * in name string')
        path_str = path + '/*' + test + name + '*pvals' + mean_str + '_idx' + str(int(idx))+ '*_nreal' + str(int(n_real)) + '*.npy'
    
    for file in glob.glob(path_str):
        pvals = np.load(file)

        if hist:
            print(file[len(path) + 1 : -4])
            plt.hist(pvals)
            plt.show()

        pval_realizations = len(np.where(pvals < alpha)[0]) / len(pvals)
        match_obj = re.search(r'var(\d+)', file)

        if match_obj:
            match_list.append(int(match_obj.group(1)))
        elif 'full' in file:
            match_list.append(100) #n_real???

        pval_list.append(pval_realizations)
        file_list.append(file[len(path) + 1 : -4])
        # print(file[len(path) + 1 : -4] + ' :', pval_realizations)

    pval_list = np.array(pval_list)
    file_list = np.array(file_list)
    match_list = np.array(match_list)
    index_sort = np.argsort(match_list)

    for i in range(len(index_sort)):
        print(file_list[index_sort[i]] + ' :', pval_list[index_sort[i]])

    
    
####################################################################################################################
#optimization for prw with individual cell bounds for B and tau_m


def optimize_vacf_prw_ind(vacf_exp, max_vals, min_vals, dt=0.5, msd_fun=msd_prw):
    #max_vals and min_vals for taum and B in terms of observed quantity (e.g. 2 means maximally twice the observed value)

    if np.any(np.abs(vacf_exp) > 1e100) or np.any(np.isnan(vacf_exp)):
        return np.nan * np.ones(3), np.nan

    # t = np.arange(0, dt * (len(vacf_exp) + 0.5), dt)
    t = np.arange(0, len(vacf_exp) + 1) * dt

    gamma0 =  2 * (2 * (vacf_exp[0] - vacf_exp[1]) / (dt * (vacf_exp[0] + vacf_exp[1]))) / dt
    tm_estimate = (2 / (dt * gamma0)) # estimate tm via kernel peak height gamma0 = 2 / (dt * tm)
    B_estimate = (vacf_exp[0]) # estimate mean squared velocity by observed value

    tmmin = min_vals[0] * tm_estimate
    if min_vals[0] > 1:
        tmmin = tm_estimate / min_vals[0]
    tmmax = max_vals[0] * tm_estimate
    Bmin = min_vals[1] * B_estimate
    if min_vals[1] > 1:
        Bmin = B_estimate / min_vals[1]
    Bmax = max_vals[1] * B_estimate
    errmin = min_vals[-1]
    errmax = max_vals[-1]

    err0 = np.random.rand() * (errmax - errmin) + errmin
    tm0 = np.random.rand() * (tmmax - tmmin) + tmmin #number between min and max
    B0 = np.random.rand() * (Bmax - Bmin) + Bmin

    x0 = np.array([tm0, B0, err0])

    if np.any(np.isnan(x0)) or np.any(np.isinf(x0)) or np.any(x0 < 0) or np.any(x0 > 1e100):
        return np.nan * np.ones(3), np.nan

    lb = np.array([tmmin, Bmin, errmin])
    upb = np.array([tmmax, Bmax, errmax])
    
    res = least_squares(lambda x: sigma_vacf_prw(x,vacf_exp, t, msd=msd_fun), x0,
                               bounds=(lb,upb))

    return res.x, res.cost

def optimize_all_cells_prw_ind(vacf_all, max_vals, min_vals, av=10, dt=0.5, msd_func=msd_prw, index=10, threads=1, tqdm_bar=False):
    ntrj=len(vacf_all[:,0])
    opt_params=np.zeros((ntrj,3))
    av_params=np.zeros((ntrj,3))
    cost_ar=np.zeros(ntrj)
    av_cost_ar=np.zeros(ntrj)
    if tqdm_bar:
        for i in tqdm(range(ntrj)):
            vacf=vacf_all[i,:].compressed()
            vacf = vacf[:index]

            res_cost_list = Parallel(n_jobs=threads)(delayed(optimize_vacf_prw_ind)(vacf, max_vals, min_vals, dt=dt, msd_fun=msd_func) for j in range(av))

            results = np.array(res_cost_list, dtype=object)[:,0]
            costs = np.array(res_cost_list, dtype=object)[:,1]    
            avres = np.mean(results, axis=0)
            
            avcost = np.mean(costs)
            best_index = np.argmin(costs)
            costb = costs[best_index]
            resopt = results[best_index]
            
            av_cost_ar[i]=avcost
            cost_ar[i]=costb
            opt_params[i,:]=resopt
            av_params[i,:]=avres
    else:
        for i in range(ntrj):
            vacf=vacf_all[i,:].compressed()
            vacf = vacf[:index]

            res_cost_list = Parallel(n_jobs=threads)(delayed(optimize_vacf_prw_ind)(vacf, max_vals, min_vals, dt=dt, msd_fun=msd_func) for j in range(av))

            results = np.array(res_cost_list, dtype=object)[:,0]
            costs = np.array(res_cost_list, dtype=object)[:,1]    
            avres = np.mean(results, axis=0)
            
            avcost = np.mean(costs)
            best_index = np.argmin(costs)
            costb = costs[best_index]
            resopt = results[best_index]
            
            av_cost_ar[i]=avcost
            cost_ar[i]=costb
            opt_params[i,:]=resopt
            av_params[i,:]=avres

        
    return opt_params, cost_ar, av_params, av_cost_ar
    



################################################################################################
#optimization for pure diffusion with individual bounds
# C_MSD = 2*D*t + (1 - delta_0i) sig_loc^2
# Cvv_0 = (2*D*dt + sig_loc^2) / dt^2
# Cvv_1 = - sig_loc^2 / (2*dt^2)

def vacf_noise_diffusion(D, sig_loc, t, msd=msd_diffusion):
    msd_theo = msd(D, t)
    msd_noise = np.zeros(len(msd_theo))
    msd_noise[1:] = msd_theo[1:] + 2 * sig_loc**2
    msd_noise[0] = msd_theo[0]
    dt = t[1] - t[0]
    vacf_noisy = (np.roll(msd_noise,-1)[:-1] - 2 * msd_noise[:-1] + np.roll(msd_noise,1)[:-1]) / (2*dt**2)
    vacf_noisy[0] = msd_noise[1] / dt**2

    inf_residues = np.where(vacf_noisy > 1e140)[0] #check if residues (vacf**2) might lead to overflows
    if len(inf_residues) > 0:
        #for indx in inf_residues:
        #    vacf_noisy[indx] = 1e140
        vacf_noisy = np.ones(len(vacf_noisy))*1e140
    
    return vacf_noisy

def sigma_vacf_diffusion(args, vacf_exp, t, msd=msd_diffusion):
    #args of the form [D, sig_loc]
    residual_vec = vacf_noise_diffusion(args[0], args[1], t, msd=msd) - vacf_exp
    if np.any(np.isinf(residual_vec)) or np.any(np.isnan(residual_vec)):
        residual_vec = np.ones(len(residual_vec)) * 1e140
    return residual_vec



def optimize_vacf_diffusion_ind(vacf_exp, max_vals, min_vals, dt=0.5, msd_fun=msd_diffusion):
    #max_vals and min_vals for D in terms of observed quantity (e.g. 2 means maximally twice the estimated value)

    if np.any(np.abs(vacf_exp) > 1e100) or np.any(np.isnan(vacf_exp)):
        return np.nan * np.ones(2), np.nan

    t = np.arange(0, len(vacf_exp) + 1) * dt

    Dest = vacf_exp[0] * dt / 2 # estimation of D for zero noise

    Dmin = Dest * min_vals[0]
    if min_vals[0] > 1:
        Dmin = Dest / min_vals[0]
    Dmax = Dest * max_vals[0]
    errmin = min_vals[-1]
    errmax = max_vals[-1]

    err0 = np.random.rand() * (errmax - errmin) + errmin
    D0 = np.random.rand() * (Dmax - Dmin) + Dmin #number between min and max

    x0 = np.array([D0, err0])

    if np.any(np.isnan(x0)) or np.any(np.isinf(x0)) or np.any(x0 < 0) or np.any(x0 > 1e100):
        return np.nan * np.ones(2), np.nan

    lb = np.array([Dmin, errmin])
    upb = np.array([Dmax, errmax])
    
    res = least_squares(lambda x: sigma_vacf_diffusion(x, vacf_exp, t, msd=msd_fun), x0,
                               bounds=(lb,upb))

    return res.x, res.cost

def optimize_all_cells_diffusion_ind(vacf_all, max_vals, min_vals, av=10, dt=0.5, msd_func=msd_diffusion, index=10, threads=1, tqdm_bar=False):
    ntrj = len(vacf_all[:, 0])
    opt_params=np.zeros((ntrj, 2))
    av_params=np.zeros((ntrj, 2))
    cost_ar=np.zeros(ntrj)
    av_cost_ar=np.zeros(ntrj)
    if tqdm_bar:
        for i in tqdm(range(ntrj)):
            vacf = vacf_all[i,:].compressed()
            vacf = vacf[:index]  #vacf[:int(len(vacf)/2)]

            res_cost_list = Parallel(n_jobs=threads)(delayed(optimize_vacf_diffusion_ind)(vacf, max_vals, min_vals, dt=dt, msd_fun=msd_func) for j in range(av))

            results = np.array(res_cost_list, dtype=object)[:,0]
            costs = np.array(res_cost_list, dtype=object)[:,1]    
            avres = np.mean(results, axis=0)
            
            avcost = np.mean(costs)
            best_index = np.argmin(costs)
            costb = costs[best_index]
            resopt = results[best_index]
            
            av_cost_ar[i] = avcost
            cost_ar[i] = costb
            opt_params[i,:] = resopt
            av_params[i,:] = avres
    else:
        for i in range(ntrj):
            vacf = vacf_all[i,:].compressed()
            vacf = vacf[:index]  #vacf[:int(len(vacf)/2)]

            res_cost_list = Parallel(n_jobs=threads)(delayed(optimize_vacf_diffusion_ind)(vacf, max_vals, min_vals, dt=dt, msd_fun=msd_func) for j in range(av))

            results = np.array(res_cost_list, dtype=object)[:,0]
            costs = np.array(res_cost_list, dtype=object)[:,1]    
            avres = np.mean(results, axis=0)
            
            avcost = np.mean(costs)
            best_index = np.argmin(costs)
            costb = costs[best_index]
            resopt = results[best_index]
            
            av_cost_ar[i] = avcost
            cost_ar[i] = costb
            opt_params[i,:] = resopt
            av_params[i,:] = avres

    return opt_params, cost_ar, av_params, av_cost_ar

#################################################################################################


def sigma_msd_diffusion(args, msd_exp, t):
    #args of the form [D, sig_loc]
    residual_vec = msd_diffusion_noise(args[0], args[1], t) - msd_exp
    if np.any(np.isinf(residual_vec)) or np.any(np.isnan(residual_vec)):
        residual_vec = np.ones(len(residual_vec)) * 1e140
    return residual_vec


def optimize_msd_diffusion_ind(msd_exp, max_vals, min_vals, dt=0.5, abs_minmax=False):
    #max_vals and min_vals for D in terms of observed quantity (e.g. 2 means maximally twice the estimated value)

    if np.any(np.abs(msd_exp) > 1e100) or np.any(np.isnan(msd_exp)):
        return np.nan * np.ones(2), np.nan

    t = np.arange(0, len(msd_exp)) * dt

    if abs_minmax:
        Dmin = min_vals[0]
        Dmax = max_vals[0]
    else:
        Dest = msd_exp[1] / (dt * 2) # estimation of D for zero noise
        Dmin = Dest * min_vals[0]
        if min_vals[0] > 1:
            Dmin = Dest / min_vals[0]
        Dmax = Dest * max_vals[0]
    errmin = min_vals[-1]
    errmax = max_vals[-1]

    err0 = np.random.rand() * (errmax - errmin) + errmin
    D0 = np.random.rand() * (Dmax - Dmin) + Dmin #number between min and max

    x0 = np.array([D0, err0])

    if np.any(np.isnan(x0)) or np.any(np.isinf(x0)) or np.any(x0 < 0) or np.any(x0 > 1e100):
        return np.nan * np.ones(2), np.nan

    lb = np.array([Dmin, errmin])
    upb = np.array([Dmax, errmax])
    
    res = least_squares(lambda x: sigma_msd_diffusion(x, msd_exp, t), x0,
                               bounds=(lb,upb))

    return res.x, res.cost

def optimize_all_cells_diffusion_msd_ind(msd_all, max_vals, min_vals, dim, av=10, dt=0.5, index=10, threads=1, abs_minmax=False, tqdm_bar=False):
    ntrj = len(msd_all[:, 0])
    opt_params = np.zeros((ntrj, 2))
    av_params = np.zeros((ntrj, 2))
    cost_ar = np.zeros(ntrj)
    av_cost_ar = np.zeros(ntrj)
    if tqdm_bar:
        for i in tqdm(range(ntrj)):
            msd = msd_all[i,:].compressed() / dim # 2D * dim * t is MSD
            msd = msd[:index]  #msd[:int(len(msd)/2)]

            res_cost_list = Parallel(n_jobs=threads)(delayed(optimize_msd_diffusion_ind)(msd, max_vals, min_vals, dt=dt, abs_minmax=abs_minmax) for j in range(av))

            results = np.array(res_cost_list, dtype=object)[:,0]
            costs = np.array(res_cost_list, dtype=object)[:,1]    
            avres = np.mean(results, axis=0)
            
            avcost = np.mean(costs)
            best_index = np.argmin(costs)
            costb = costs[best_index]
            resopt = results[best_index]
            
            av_cost_ar[i] = avcost
            cost_ar[i] = costb
            opt_params[i,:] = resopt
            av_params[i,:] = avres
    else:
        for i in range(ntrj):
            msd = msd_all[i,:].compressed() / dim # 2D * dim * t is MSD
            msd = msd[:index]  #msd[:int(len(msd)/2)]

            res_cost_list = Parallel(n_jobs=threads)(delayed(optimize_msd_diffusion_ind)(msd, max_vals, min_vals, dt=dt, abs_minmax=abs_minmax) for j in range(av))

            results = np.array(res_cost_list, dtype=object)[:,0]
            costs = np.array(res_cost_list, dtype=object)[:,1]    
            avres = np.mean(results, axis=0)
            
            avcost = np.mean(costs)
            best_index = np.argmin(costs)
            costb = costs[best_index]
            resopt = results[best_index]
            
            av_cost_ar[i] = avcost
            cost_ar[i] = costb
            opt_params[i,:] = resopt
            av_params[i,:] = avres

    return opt_params, cost_ar, av_params, av_cost_ar


###################################################################################################################


def sim_var_diffusion(exp_distr, scaling, n_realization, n_steps, dt, path_out, name, min_vals, max_vals, dim=1, av=100, index_vacf=10, threads=16, mean=False, filt_err=False, msd_fit=False):
    if filt_err:
        cond_err = filt_distr(exp_distr, min_vals[-1], max_vals[-1], max_val=[1e50], min_factor=1.5, max_factor=0.99)
    else:
        cond_err = filt_distr(exp_distr, min_vals[-1], max_vals[-1], max_val=[1e50], min_factor=0.9, max_factor=1.1)
        # adjust min and max factor such that errors are not filtered, but only nans and infs
    
    distr_exp = exp_distr[cond_err, :]
    mean_data = np.mean(distr_exp, axis=0)
    mean_str = ''
    if not mean:
        mean_data = np.median(distr_exp, axis=0) # if not mean, median is used as central point for underlying Gaussian distr
        mean_str = '_med'

    cov_data = np.cov(distr_exp.T)

    ltrjt = np.ones(len(distr_exp[:,0])) * n_steps[cond_err]
    simulation_step = 0.05 * dt

    scale_str = str(int(100 * scaling)) + 'percent'
    if scaling == 1:
        scale_str = 'full'
        xall_list = Parallel(n_jobs=threads)(delayed(cre_trjs_diffusion)(ltrjt, distr_exp[:, 0], dt, distr_exp[:, 1], h=simulation_step) for j in range(n_realization))
        if dim == 2:
            yall_list = Parallel(n_jobs=threads)(delayed(cre_trjs_diffusion)(ltrjt, distr_exp[:, 0], dt, distr_exp[:, 1], h=simulation_step) for j in range(n_realization))

    elif scaling == 0:
        err_use = mean_data[1] * np.ones(len(distr_exp[:, 0]))
        D_use =  mean_data[0] * np.ones(len(distr_exp[:, 0]))
        xall_list = Parallel(n_jobs=threads)(delayed(cre_trjs_diffusion)(ltrjt, D_use, dt, err_use, h=simulation_step) for j in range(n_realization))
        if dim == 2:
            yall_list = Parallel(n_jobs=threads)(delayed(cre_trjs_diffusion)(ltrjt, D_use, dt, err_use, h=simulation_step) for j in range(n_realization))

    else:
        sim_data = np.abs(np.random.multivariate_normal(mean_data, scaling * cov_data, (len(distr_exp[:,0]), n_realization)))
        xall_list = Parallel(n_jobs=threads)(delayed(cre_trjs_diffusion)(ltrjt, sim_data[:, j][:, 0], dt, sim_data[:, j][:, 1], h=simulation_step) for j in range(n_realization))
        if dim == 2:
            yall_list = Parallel(n_jobs=threads)(delayed(cre_trjs_diffusion)(ltrjt, sim_data[:, j][:, 0], dt, sim_data[:, j][:, 1], h=simulation_step) for j in range(n_realization))
        for k in range(10):
            np.save(path_out + '/' + name + 'origdistr' + str(int(k)) + mean_str + '_idx' + str(int(index_vacf)) + '_var' + scale_str + '_av' + str(int(av)), sim_data[:, k])
        
    max_exp = np.max(exp_distr, axis=0)
    for i in range(n_realization):
        if msd_fit:
            msd = combine_fun(msd_fast, xall_list[i][0])
            if dim == 2:
                msd += combine_fun(msd_fast, yall_list[i][0])
            if np.any(np.isnan(msd.compressed())) or np.any(np.isinf(msd.compressed())):
                print('simulated trajectory leads to inf or nan MSD')

            optis, costs, avres, avcosts = optimize_all_cells_diffusion_msd_ind(msd, max_vals, min_vals, dim, av=av, dt=dt, index=index_vacf, threads=threads)
        else:
            if dim ==2:
                vxall = combine_fun(lambda x: np.diff(x) / dt, xall_list[i][0], lmodified=1)
                vyall = combine_fun(lambda x: np.diff(x) / dt, yall_list[i][0], lmodified=1)
                vxacf_all = combine_fun(correlation, vxall)
                vyacf_all = combine_fun(correlation, vyall)
                vacf = 0.5 * (vxacf_all + vyacf_all)
            elif dim == 1:
                vall = combine_fun(lambda x: np.diff(x) / dt, xall_list[i][0], lmodified=1)
                vacf = combine_fun(correlation, vall)
            if np.any(np.isnan(vacf.compressed())) or np.any(np.isinf(vacf.compressed())):
                print('simulated trajectory leads to inf or nan vacf')

            optis, costs, avres, avcosts = optimize_all_cells_diffusion_ind(vacf, max_vals, min_vals, av=av, dt=dt, index=index_vacf, threads=threads)

        if filt_err:
            cond_distr_err = filt_distr(optis, min_vals[-1], max_vals[-1], max_val=max_exp[:-1] * 1e3, min_factor=1.5, max_factor=0.99)
        else:
            cond_distr_err = filt_distr(optis, min_vals[-1], max_vals[-1], max_val=max_exp[:-1] * 1e3, min_factor=0.9, max_factor=1.1)
            # adjust min and max factor such that errors are not filtered, but only nans and too large values compared to experiment
        distr_optis = optis[cond_distr_err, :]
        
        np.save(path_out + '/' + name + str(int(i)) + mean_str + '_idx' + str(int(index_vacf)) + '_var' + scale_str + '_av' + str(int(av)), distr_optis) # + '_steps' + str(int(n_steps))
        #print(i)

        

###################################################################################################################
# simulate scaled distribution around median/mean for diffusion


def sim_scale_diffusion(exp_distr, scaling, n_realization, n_steps, dt, path_out, name, min_vals, max_vals, dim=1, av=100, index_vacf=10, threads=16, mean=False, filt_err=False):
    if filt_err:
        cond_err = filt_distr(exp_distr, min_vals[-1], max_vals[-1], max_val=[1e50], min_factor=1.5, max_factor=0.99)
    else:
        cond_err = filt_distr(exp_distr, min_vals[-1], max_vals[-1], max_val=[1e50], min_factor=0.9, max_factor=1.1)
        # adjust min and max factor such that errors are not filtered, but only nans and infs
    distr_exp = exp_distr[cond_err, :]
    mean_data = np.mean(distr_exp, axis=0)
    mean_str = ''
    if not mean:
        mean_data = np.median(distr_exp, axis=0) # if not mean, median is used as central point for underlying Gaussian distr
        mean_str = '_med'

    ltrjt = np.ones(len(distr_exp[:,0])) * n_steps[cond_err]
    simulation_step = 0.05 * dt

    scale_str = str(int(100 * scaling)) + 'percent'
    if scaling == 1:
        scale_str = 'full'
        xall_list = Parallel(n_jobs=threads)(delayed(cre_trjs_diffusion)(ltrjt, distr_exp[:, 0], dt, distr_exp[:, 1], h=simulation_step) for j in range(n_realization))

    elif scaling == 0:
        err_use = mean_data[1] * np.ones(len(distr_exp[:, 0]))
        D_use = mean_data[0] * np.ones(len(distr_exp[:, 0]))
        xall_list = Parallel(n_jobs=threads)(delayed(cre_trjs_diffusion)(ltrjt, D_use, dt, err_use, h=simulation_step) for j in range(n_realization))

    else:
        sim_data = distr_exp * np.sqrt(scaling) + (1 - np.sqrt(scaling)) * mean_data
        xall_list = Parallel(n_jobs=threads)(delayed(cre_trjs_diffusion)(ltrjt, sim_data[:, 0], dt, sim_data[:, 1], h=simulation_step) for j in range(n_realization))
        
    max_exp = np.max(exp_distr, axis=0)
    for i in range(n_realization):
        msd = combine_fun(msd_fast, xall_list[i][0])
        if np.any(np.isnan(msd.compressed())) or np.any(np.isinf(msd.compressed())):
            print('simulated trajectory leads to inf or nan MSD')

        optis, costs, avres, avcosts = optimize_all_cells_diffusion_msd_ind(msd, max_vals, min_vals, dim, av=av, dt=dt, index=index_vacf, threads=threads)

        if filt_err:
            cond_distr_err = filt_distr(optis, min_vals[-1], max_vals[-1], max_val=max_exp[:-1] * 1e3, min_factor=1.5, max_factor=0.99)
        else:
            cond_distr_err = filt_distr(optis, min_vals[-1], max_vals[-1], max_val=max_exp[:-1] * 1e3, min_factor=0.9, max_factor=1.1)
            # adjust min and max factor such that errors are not filtered, but only nans and too large values compared to experiment
        distr_optis = optis[cond_distr_err, :]
        
        np.save(path_out + '/' + name + str(int(i)) + mean_str + '_idx' + str(int(index_vacf)) + '_var' + scale_str + '_av' + str(int(av)), distr_optis) # + '_steps' + str(int(n_steps))
        #print(i)
        


################################################################################################
#optimization for any memory function

def msd_noise_mem(params, t, msd=msd_osc_analytic):
    msd_theo = msd(*params[:-1], t)
    msd_noise = msd_theo + 2 * params[-1]**2
    msd_noise[0] = msd_theo[0]
    
    return msd_noise

def vacf_noise_mem(params, t, msd=msd_osc_sincos):
    msd_theo = msd(*params[:-1],t)
    msd_noise = np.zeros(len(msd_theo))
    msd_noise[1:] = msd_theo[1:] + 2 * params[-1]**2
    msd_noise[0] = msd_theo[0]
    dt = t[1] - t[0]
    vacf_noisy = (np.roll(msd_noise,-1)[:-1] - 2*msd_noise[:-1] + np.roll(msd_noise,1)[:-1]) / (2*dt**2) #compatible with forward difference velocities
    vacf_noisy[0] = msd_noise[1]/dt**2

    inf_residues=np.where(vacf_noisy>1e140)[0] #check if residues (vacf**2) might lead to overflows
    if len(inf_residues)>0:
        vacf_noisy=np.ones(len(vacf_noisy))*1e140
    
    return vacf_noisy

def sigma_vacf_mem(args, vacf_exp, t, msd=msd_osc_sincos):
    #args of the form [..., B, sig_loc]
    residual_vec = vacf_noise_mem(args, t, msd=msd) - vacf_exp
    if np.any(np.isinf(residual_vec)) or np.any(np.isnan(residual_vec)):
        residual_vec = np.ones(len(residual_vec)) * 1e140
    return residual_vec

def optimize_vacf_mem(vacf_exp, max_vals, min_vals, dt=0.5, msd_fun=msd_osc_sincos, B_param=True, rel_ampl=0):

    if np.any(np.abs(vacf_exp) > 1e100) or np.any(np.isnan(vacf_exp)):
        print('VACF contains nans or infs')
        return np.nan * np.ones(len(max_vals)), np.nan

    if B_param:
        max_vals[-2] = max_vals[-2] * vacf_exp[0]
        min_vals[-2] = min_vals[-2] * vacf_exp[0]
        if rel_ampl > 0:
            for ind_rel in range(1, rel_ampl + 1):
                max_vals[-2 - ind_rel] = max_vals[-2 - ind_rel] * vacf_exp[0] # all amplitudes are bounded individually and relatively to the data point C_vv(0)
                min_vals[-2 - ind_rel] = min_vals[-2 - ind_rel] * vacf_exp[0]
        # last parameter is localization noise and previous parameter is square velocity, for which bounds are individually extimated by experimental B
    t = np.arange(0, len(vacf_exp) + 1) * dt

    lb = np.array(min_vals)
    upb = np.array(max_vals)
    x0 = np.random.rand(len(upb)) * (upb - lb) + lb
    
    res = least_squares(lambda x: sigma_vacf_mem(x, vacf_exp, t, msd=msd_fun), x0,
                               bounds=(lb, upb))

    return res.x, res.cost

def optimize_all_cells_mem(vacf_all, max_vals, min_vals, av=10, dt=0.5, msd_func=msd_osc_sincos, index=10, threads=1, B_param=True, rel_ampl=0):
    ntrj = len(vacf_all[:,0])
    opt_params = np.zeros((ntrj, len(max_vals)))
    av_params = np.zeros((ntrj, len(max_vals)))
    cost_ar = np.zeros(ntrj)
    av_cost_ar = np.zeros(ntrj)
    for i in tqdm(range(ntrj)):
        vacf = vacf_all[i,:].compressed()
        vacf = vacf[:index]  #vacf[:int(len(vacf)/2)]

        res_cost_list = Parallel(n_jobs=threads)(delayed(optimize_vacf_mem)(vacf, max_vals, min_vals, dt=dt, msd_fun=msd_func, B_param=B_param, rel_ampl=rel_ampl) for j in range(av))

        results = np.array(res_cost_list, dtype=object)[:,0]
        costs = np.array(res_cost_list, dtype=object)[:,1]    
        avres = np.mean(results, axis=0)
        
        avcost = np.mean(costs)
        best_index = np.argmin(costs)
        costb = costs[best_index]
        resopt = results[best_index]
        
        av_cost_ar[i] = avcost
        cost_ar[i] = costb
        opt_params[i,:] = resopt
        av_params[i,:] = avres

        
    return opt_params, cost_ar, av_params, av_cost_ar



############################################################################################################################################



def cre_trjs_diffusion(ltrjs, D, dt, loc_err, h=0.01):
    '''take arguments D,dt in [min], ltrjs is an array containing lengths of the trjs in steps of dt
    loc_err is the width of the location noise error (gaussian). D, loc_err should be arrays of len ltrjs or float,
    in the latter case all trjs are simulated with the same given value.
    returns masked matrix of simulated trjs with timestep dt and location noise loc_err
    '''

    skip = dt / h
    #print('take every jth value of simulation with j:',skip)

    ntrj = len(ltrjs)
    skip = np.ones(ntrj) * skip
    Ds = np.ones(ntrj) * D
    loc_errs = np.ones(ntrj) * loc_err
    
    lmax = np.max(ltrjs)
    xall = np.ma.empty( ( ntrj, int(lmax) ) )
    xall.mask = True
    for i in range(ntrj):
        hu = h
        skipi = skip[i]
        if skip[i] < 1:
            hu = dt # if time step is smaller than h, use every step and dt as simulation hu
            skipi = 1
        elif skip[i] % 1 != 0:
            skipi = round(skip[i] - 0.5)
            #print('changed timestep from '+ str(dt)+' to:', skip*h*taud)
        skipi = int(skipi)
        
        tmax = ltrjs[i] * skipi * hu
        x = euler_diffusion(Ds[i], hu, tmax)[::skipi]
        x += np.random.normal(0, loc_errs[i], len(x))
        xall[i,:len(x)] = x
        ltrjs[i] = len(x)

    return xall, ltrjs

def test_cre_trj_diff(D, dt, h=0.01, steps=100000):
    x, ls = cre_trjs_diffusion([steps], D, dt, 0, h=h)
    msd = msd_fast(x.compressed())
    t = np.arange(steps) * dt
    plt.plot(t[1:], msd[1:])
    plt.plot(t[1:], 2*D*t[1:], ls='--')
    plt.loglog()
    plt.show()


############################################################################################################################################


def cre_trjs_prw(ltrjs, taud, taum, dt, loc_err, h=0.01):
    '''take arguments taud,taum,dt in [min], ltrjs is an array containing lengths of the trjs in steps of dt
    loc_err is the width of the location noise error (gaussian). taud, taum, loc_err should be arrays of len ltrjs or float,
    in which case all trjs are simulated with the same given value
    returns masked matrix of simulated trjs with timestep dt and location noise loc_err
    '''
    tau_m = taum/taud
    skip = (dt/taud)/h
    
    #print('take every jth value of simulation with j:',skip)

    ntrj = len(ltrjs)
    #if type(skip) == float:
    skip = np.ones(ntrj) * skip
    tauds = np.ones(ntrj) * taud
    loc_errs = np.ones(ntrj) * loc_err
    tau_ms = np.ones(ntrj) * tau_m
    
    lmax = np.max(ltrjs)
    xall = np.ma.empty( ( ntrj, int(lmax) ) )#+1
    xall.mask = True
    for i in range(ntrj):
        hu = h
        skipi = skip[i]
        if skip[i] < 1:
            hu = dt/tauds[i]
            skipi = 1
        elif skip[i] % 1 != 0:
        #print('time step and tau_D cannot be chosen together because simulation data is not available at times \n\
        #n*dt/tau_D')
            skipi = round(skip[i] - 0.5)
            #print('changed timestep from '+ str(dt)+' to:', skip*h*taud)
        skipi = int(skipi)
        
        tmax = ltrjs[i] * skipi * hu
        x = euler(2, hu, tmax , tau_ms[i], 0, f)[::skipi] #tmax+dt
        x += np.random.normal(0, loc_errs[i], len(x))
        xall[i,:len(x)] = x
        ltrjs[i] = len(x)
    #np.save('trj1d_tm'+str(int(tau_m))+'_dt'+str(int(dt))+...,arr)   
    return xall, ltrjs

############################################################################################################################################



def cre_trjs_osc_sincos(ltrjs, a, b, freq, tau, B, loc_err, dt, h=0.01):
    '''take kernel parameters a, b, freq, tau, and time step of experiment as input
    ltrjs is an array containing lengths of the trjs in steps of dt
    loc_err is the width of the location noise error (gaussian). taud, taum, loc_err should be arrays of len ltrjs or float,
    in which case all trjs are simulated with the same given value
    returns masked matrix of simulated trjs with timestep dt and location noise loc_err
    '''
    tau_s = 1 # / np.sqrt(B) #instead just multiply position by sqrt(B), which computes faster and yields same results
    skip = (dt / tau_s) / h
    
    #print('take every jth value of simulation with j:',skip)

    ntrj = len(ltrjs)
    skip = np.ones(ntrj) * skip
    Bs = np.ones(ntrj) * B
    loc_errs = np.ones(ntrj) * loc_err
    ass = np.ones(ntrj) * a * tau_s
    bs = np.ones(ntrj) * b * tau_s**2
    freqs = np.ones(ntrj) * freq * tau_s
    taus = np.ones(ntrj) * tau / tau_s
    tau_s_arr = np.ones(ntrj) * tau_s

    lmax = np.max(ltrjs)
    xall = np.ma.empty( ( ntrj, int(lmax) ) )
    xall.mask = True
    
    for i in range(ntrj):
        hu = h
        skipi = skip[i]
        if skip[i] < 1:
            hu = dt / tau_s_arr[i]
            skipi = 1
        elif skip[i] % 1 != 0:
            skipi = round(skip[i] - 0.5)
            #print('changed timestep from '+ str(dt)+' to:', skip*h*taud)
        skipi = int(skipi)
        tmax = ltrjs[i] * skipi * hu
        x = euler_mem(np.array([ass[i], bs[i], freqs[i], taus[i]]), hu, tmax, f_osc)[::skipi] * np.sqrt(Bs[i]) #tmax+dt
        x += np.random.normal(0, loc_errs[i], len(x))
        xall[i,:len(x)] = x
        ltrjs[i] = len(x)
    #np.save('trj1d_tm'+str(int(tau_m))+'_dt'+str(int(dt))+...,arr)   
    return xall, ltrjs


#############################################################################################################################

# direct kernel fit for chlamy data

def function2(x, dp, p1, p2, p3):
    delta = np.zeros(len(x))
    delta[0] = dp
    return p1*np.exp(-x*p2)*np.cos(p3*x) + delta

def function_embedded(x, dp, p1, p2, p3):
    delta = np.zeros(len(x))
    delta[0] = dp
    return p1*np.exp(-x*p2)*np.cos(p3*x) + delta + p1*np.exp(-x*p2)*np.sin(p3*x)*p2/p3


def kernel_fit(vacf, dt, min_vals=np.array([0.001, 0.01, 0.1, 25]), max_vals=np.array([0.99, 0.999, 15, 250]), fix_x0=2, kernel_fun=function_embedded):
    kernel = kernel_1dcc(vacf, dt)
    t = np.arange(len(kernel)) * dt
    k0 = kernel[0]

    lb = np.array([k0, k0, 1, 1]) * min_vals  # np.array([0.001*k0, 0.01*k0, 0.1, 25])
    ub = np.array([k0, k0, 1, 1]) * max_vals  # np.array([0.99*k0, 0.999*k0, 15, 250])
    p0 = np.random.rand(len(lb)) * (ub - lb) + lb
    if fix_x0 == 0:
        p0 = np.array([0.6666 * k0, 0.3333 * k0, 5, 100])

    opt_fitp, cov_fit = curve_fit(kernel_fun, t, kernel, p0, bounds=(lb,ub), maxfev=100000)
    cost = np.sum((kernel - kernel_fun(t, *opt_fitp))**2) # + 10 * (k0 - kernel_fun(np.zeros(1), *opt_fitp))**2

    return opt_fitp, cost

def kernel_fit_all(vacf_all, dt, min_vals=np.array([0.001, 0.01, 0.1, 25]), max_vals=np.array([0.99, 0.999, 15, 250]), av=10, kernel_fun=function_embedded, index=100, threads=1):
    ntrj = len(vacf_all[:,0])
    opt_params = np.zeros((ntrj, len(max_vals)))
    av_params = np.zeros((ntrj, len(max_vals)))
    cost_ar = np.zeros(ntrj)
    av_cost_ar = np.zeros(ntrj)
    for i in tqdm(range(ntrj)):
        vacf = vacf_all[i,:].compressed()
        vacf = vacf[:index]

        res_cost_list = Parallel(n_jobs=threads)(delayed(kernel_fit)(vacf, dt, min_vals=min_vals, max_vals=max_vals, fix_x0=j, kernel_fun=kernel_fun) for j in range(av))
        # zeroth optimization from fixed starting point, known to work well (fix_x0=j)

        results = np.array(res_cost_list, dtype=object)[:,0]
        costs = np.array(res_cost_list, dtype=object)[:,1]    
        avres = np.mean(results, axis=0)
        
        avcost = np.mean(costs)
        best_index = np.argmin(costs)
        costb = costs[best_index]
        resopt = results[best_index]
        
        av_cost_ar[i] = avcost
        cost_ar[i] = costb
        opt_params[i,:] = resopt
        av_params[i,:] = avres

        
    return opt_params, cost_ar, av_params, av_cost_ar



######################################################################################################################

def sim_var_osc_sincos(exp_distr, scaling, n_realization, n_steps, dt, path_out, name, min_vals, max_vals, av=100, index_vacf=10, threads=16, mean=False):
    exp_distr[:, 0] *= dt/2
    exp_distr[:, 2] = 1 / exp_distr[:, 2]

    mean_data = np.mean(exp_distr, axis=0)
    mean_str = ''
    if not mean:
        mean_data = np.median(exp_distr, axis=0) # if not mean, median is used as central point for underlying Gaussian distr
        mean_str = '_med'

    cov_data = np.cov(exp_distr.T)

    ltrjt = np.ones(len(exp_distr[:,0])) * n_steps
    simulation_step = 0.005 * dt

    scale_str = str(int(100 * scaling)) + 'percent'
    if scaling == 1:
        scale_str = 'full'
        xall_list = Parallel(n_jobs=threads)(delayed(cre_trjs_osc_sincos)(ltrjt, exp_distr[:, 0], exp_distr[:, 1], exp_distr[:, 3], exp_distr[:, 2], exp_distr[:, 4], exp_distr[:, 5], dt, h=simulation_step) for j in range(n_realization))

    elif scaling == 0:
        a_use = mean_data[0] * np.ones(len(exp_distr[:, 0]))
        b_use = mean_data[1] * np.ones(len(exp_distr[:, 0]))
        tau_use = mean_data[2] * np.ones(len(exp_distr[:, 0]))
        f_use = mean_data[3] * np.ones(len(exp_distr[:, 0]))
        B_use = mean_data[4] * np.ones(len(exp_distr[:, 0]))
        err_use = mean_data[5] * np.ones(len(exp_distr[:, 0]))

        xall_list = Parallel(n_jobs=threads)(delayed(cre_trjs_osc_sincos)(ltrjt, a_use, b_use, f_use, tau_use, B_use, err_use, dt, h=simulation_step) for j in range(n_realization))

    else:
        sim_data = np.abs(np.random.multivariate_normal(mean_data, scaling * cov_data, (len(exp_distr[:,0]), n_realization)))
        xall_list = Parallel(n_jobs=threads)(delayed(cre_trjs_osc_sincos)(ltrjt, sim_data[:, j][:, 0], sim_data[:, j][:, 1], sim_data[:, j][:, 3], sim_data[:, j][:, 2], sim_data[:, j][:, 4], sim_data[:, j][:, 5], dt, h=simulation_step) for j in range(n_realization))
        for k in range(10):
            np.save(path_out + '/sincos' + name + 'origdistr' + str(int(k)) + mean_str + '_idx' + str(int(index_vacf)) + '_var' + scale_str + '_av' + str(int(av)), sim_data[:, k])

    for i in range(n_realization):
        vall = combine_fun(lambda x: np.gradient(x)[1:-1] / dt, xall_list[i][0], lmodified=2)
        vacf_all = combine_fun(correlation, vall)
        if np.any(np.isnan(vacf_all.compressed())) or np.any(np.isinf(vacf_all.compressed())):
            print('simulated trajectory leads to inf or nan VACF')
        B_all = vacf_all[:, 0].compressed()
        optis, costs, avres, avcosts = kernel_fit_all(vacf_all, dt, max_vals=max_vals, min_vals=min_vals, av=av, kernel_fun=function_embedded, index=index_vacf, threads=threads)
        opt_all = np.c_[optis, B_all]
        
        np.save(path_out + '/' + name + str(int(i)) + mean_str + '_idx' + str(int(index_vacf)) + '_var' + scale_str + '_av' + str(int(av)), opt_all) # + '_steps' + str(int(n_steps))
        print(i)


def smooth_neighbors(x):
    return (x[1:] + x[:-1]) / 2




######################################################################################################################################################################################
# optimization for sincos kernel

def msd_noise_osc(a, b, f, tg, B, sig_loc, t, msd=msd_osc_analytic):
    msd_theo = msd(a, b, f, tg, B, t)
    msd_noise = msd_theo + 2 * sig_loc**2
    msd_noise[0] = msd_theo[0]
    
    return msd_noise
    

def vacf_noise_osc(a, b, f, tg, B, sig_loc, t, msd=msd_osc_sincos):
    msd_theo = msd(a, b, f, tg, B, t)
    msd_noise = msd_theo + 2 * sig_loc**2 # add noise to msd
    msd_noise[0] = msd_theo[0] # set first value to original msd value, which is not affected by noise = 0
    dt = t[1] - t[0] # constant time step

    # if nans or infs appear, set vacf to 1e140, so that no overflow occurs, but optimization has high cost for that case
    if np.any(np.isnan(msd_noise)) or np.any(np.isinf(msd_noise)):
        vacf_noisy = np.ones(len(msd_noise)-1)*1e140
    else:
        vacf_noisy = (np.roll(msd_noise,-1)[:-1] - 2 * msd_noise[:-1] + np.roll(msd_noise,1)[:-1]) / (2 * dt**2)
        vacf_noisy[0] = msd_noise[1] / dt**2
    
        inf_residues = np.where(vacf_noisy>1e140)[0] #check if residues (vacf**2) might lead to overflows
        if len(inf_residues) > 0:
            vacf_noisy = np.ones(len(vacf_noisy))*1e140
    return vacf_noisy

def sigma_vacf_osc(args, vacf_exp, t, msd=msd_osc_sincos):
    #args of the form [a, b, f, tg, B, sig_loc]
    residual_vec = vacf_noise_osc(args[0], args[1], args[2], args[3], args[4], args[5], t, msd=msd) - vacf_exp
    return residual_vec


def optimize_vacf_osc(vacf_exp, dt, msd_fun=msd_osc_sincos, min_vals=np.array([10, 1000, 15, 0.005, 0.25, 0]), max_vals=np.array([400, 500000, 250, 15, 400, 0.05])):
    t = np.arange(len(vacf_exp) + 1) * dt
    x0 = np.random.rand(len(max_vals)) * (np.array(max_vals) - np.array(min_vals)) + np.array(min_vals)
    
    res=least_squares(lambda x: sigma_vacf_osc(x, vacf_exp, t, msd=msd_fun), x0,
                               bounds=(min_vals, max_vals))
    
    return res.x, res.cost

def optimize_all_cells_osc_av(vacf_all, dt, av=10, msd_fun=msd_osc_sincos, threads=4, min_vals=np.array([0.01, 0.01, 15, 0.01, 0.1, 0]), max_vals=np.array([1, 1, 250, 15, 3, 0.05]), index=100, time_estimate_cvv=False):
    # ntrj = len(vacf_all[:,0])
    ntrj = vacf_all.shape[0]
    opt_params = np.zeros((ntrj, 6))
    av_params = np.zeros((ntrj, 6))
    cost_ar = np.zeros(ntrj)
    av_cost_ar = np.zeros(ntrj)
    lb = np.copy(min_vals)
    ub = np.copy(max_vals)

    for i in tqdm(range(ntrj)):
        vacf = vacf_all[i, :]
        vacf = vacf.compressed()
        if time_estimate_cvv:
            tau_est = dt * (2 + np.where(np.abs(vacf[2:]) < np.abs(vacf[2]/np.exp(1)))[0][0]) # first time that vacf is below vacf[2]/2.7 (first value not influenced by sigma_loc) is used to estimate decay time
            # using absolute value to cover case of negative values of vacf with oscillation
            ub[3] = max_vals[3] * tau_est
            lb[3] = min_vals[3] * tau_est
        vacf = vacf[:int(index)]
        costb = np.inf
        avres = np.zeros(6)
        avcost = 0
        
        gamma0 = 4 * (vacf[0] - vacf[1]) / (dt**2 * (vacf[0] + vacf[1])) # from discrete delta kernel 2 * (2 * (vacf_exp[0] - vacf_exp[1]) / (dt * (vacf_exp[0] + vacf_exp[1]))) / dt
        bmax = gamma0
        amax = gamma0 * dt / 2 # from discrete delta kernel

        lb[0] = amax / 100 #at least 1% of initial peak belonging to a
        lb[1] = bmax / 100 #at least 1% of initial peak belonging to b
        lb[-2] = vacf[0] / max_vals[-2] # velocity estimated by experimental B
        ub[0] = amax
        ub[1] = bmax
        ub[-2] = max_vals[-2] * vacf[0] # velocity estimated by experimental B

        res_cost_list = Parallel(n_jobs=threads)(delayed(optimize_vacf_osc)(vacf, dt, max_vals=ub, min_vals=lb, msd_fun=msd_fun) for j in range(av))
        # res_cost_list = [optimize_vacf_osc(vacf, dt, max_vals=ub, min_vals=lb, msd_fun=msd_fun) for j in range(av)] # is not faster...

        results = np.array(res_cost_list, dtype=object)[:,0]
        costs = np.array(res_cost_list, dtype=object)[:,1]    
        avres = np.mean(results, axis=0)
        
        avcost = np.mean(costs)
        best_index = np.argmin(costs)
        costb = costs[best_index]
        resopt = results[best_index]
        
        av_cost_ar[i] = avcost
        cost_ar[i] = costb
        opt_params[i,:] = resopt
        av_params[i,:] = avres
        
    return opt_params, cost_ar, av_params, av_cost_ar


###########################################################################################
# using mid point rule and vacf localization noise fit


def vacf_noise_osc_mid(a, b, f, tg, B, sig_loc, t, msd=msd_osc_analytic):
    msd_theo = msd(a, b, f, tg, B, t)
    msd_noise = msd_theo + 2 * sig_loc**2
    msd_noise[0] = msd_theo[0]
    dt = t[1] - t[0]
    if np.any(np.isnan(msd_noise)) or np.any(np.isinf(msd_noise)):
        vacf_noisy = np.ones(len(msd_noise) - 2) * 1e140

    else:
        vacf_noisy = (np.roll(msd_noise,-2)[:-2] - 2 * msd_noise[:-2] + np.roll(msd_noise,2)[:-2]) / (8 * dt**2)
        vacf_noisy[0] = msd_noise[2] / (4 * dt**2) # assuming symmetric msd function
        vacf_noisy[1] = (msd_noise[3] - msd_noise[1]) / (8 * dt**2) # assuming symmetric msd function
    
        inf_residues = np.where(vacf_noisy > 1e140)[0] #check if residues (vacf**2) might lead to overflows
        if len(inf_residues) > 0:
            vacf_noisy = np.ones(len(vacf_noisy)) * 1e140
    return vacf_noisy


def sigma_vacf_osc_mid(args, vacf_exp, t, msd=msd_osc_analytic):
    #args of the form [a, b, f, tg, B, sig_loc]
    residual_vec = vacf_noise_osc_mid(args[0], args[1], args[2], args[3], args[4], args[5], t, msd=msd) - vacf_exp
    return residual_vec


def optimize_vacf_osc_mid(vacf_exp, dt, msd_fun=msd_osc_analytic, min_vals=np.array([10, 1000, 15, 0.005, 0.25, 0]), max_vals=np.array([400, 500000, 250, 15, 400, 0.05])):
    t = np.arange(len(vacf_exp) + 2) * dt
    x0 = np.random.rand(len(max_vals)) * (np.array(max_vals) - np.array(min_vals)) + np.array(min_vals)
    
    res=least_squares(lambda x: sigma_vacf_osc_mid(x, vacf_exp, t, msd=msd_fun), x0,
                               bounds=(min_vals, max_vals))
    
    return res.x, res.cost


def optimize_all_cells_osc_av_mid(vacf_all, dt, av=10, msd_fun=msd_osc_analytic, threads=4, min_vals=np.array([0.1, 100, 15, 0.01, 0.1, 0]), max_vals=np.array([1, 1, 250, 15, 3, 0.05]), index=100):
    ntrj = len(vacf_all[:,0])
    opt_params = np.zeros((ntrj, 6))
    av_params = np.zeros((ntrj, 6))
    cost_ar = np.zeros(ntrj)
    av_cost_ar = np.zeros(ntrj)
    lb = np.copy(min_vals)
    ub = np.copy(max_vals)

    for i in tqdm(range(ntrj)):
        vacf = vacf_all[i, :]
        vacf = vacf.compressed()
        vacf = vacf[:int(index)]
        costb = np.inf
        avres = np.zeros(6)
        avcost = 0
        
        gamma0 = 4 * (vacf[0] - vacf[1]) / (dt**2 * (vacf[0] + vacf[1]))
        bmax = gamma0
        amax = gamma0 * dt / 2 # from discrete delta kernel

        lb[0] = amax / 100 #at least 1% of initial peak belonging to a
        lb[1] = bmax / 100 #at least 1% of initial peak belonging to b
        lb[-2] = vacf[0] / max_vals[-2]
        ub[0] = amax
        ub[1] = bmax
        ub[-2] = max_vals[-2] * vacf[0]

        res_cost_list = Parallel(n_jobs=threads)(delayed(optimize_vacf_osc_mid)(vacf, dt, max_vals=ub, min_vals=lb, msd_fun=msd_fun) for j in range(av))

        results = np.array(res_cost_list, dtype=object)[:,0]
        costs = np.array(res_cost_list, dtype=object)[:,1]    
        avres = np.mean(results, axis=0)
        
        avcost = np.mean(costs)
        best_index = np.argmin(costs)
        costb = costs[best_index]
        resopt = results[best_index]
        
        av_cost_ar[i] = avcost
        cost_ar[i] = costb
        opt_params[i,:] = resopt
        av_params[i,:] = avres
        
    return opt_params, cost_ar, av_params, av_cost_ar


def sim_var_osc_sincos_locfit(distr_exp, scaling, n_realization, n_steps, dt, path_out, name, min_vals, max_vals, av=100, index_vacf=100, threads=16, mean=False, der='s0', dim=1):
    cond_err = filt_distr(distr_exp, min_vals[-1], max_vals[-1], max_val=[1e50, 1e50], min_factor=0.9, max_factor=1.1)
    # filter non-physical fit params
    # changed to min_factor=0.9, max_factor=1.1 error, because all tracks should be included for chlamy
    exp_distr = distr_exp[cond_err, :]

    mean_data = np.mean(exp_distr, axis=0)
    mean_str = ''
    if not mean:
        mean_data = np.median(exp_distr, axis=0) # if not mean, median is used as central point for underlying Gaussian distr
        mean_str = '_med'

    cov_data = np.cov(exp_distr.T)

    ltrjt = np.ones(len(exp_distr[:,0])) * n_steps[cond_err]
    simulation_step = 0.005 * dt

    scale_str = str(int(100 * scaling)) + 'percent'
    if scaling == 1:
        scale_str = 'full'
        xall_list = Parallel(n_jobs=threads)(delayed(cre_trjs_osc_sincos)(ltrjt, exp_distr[:, 0], exp_distr[:, 1], exp_distr[:, 2], exp_distr[:, 3], exp_distr[:, 4], exp_distr[:, 5], dt, h=simulation_step) for j in range(n_realization))
        if dim == 2:
            yall_list = Parallel(n_jobs=threads)(delayed(cre_trjs_osc_sincos)(ltrjt, exp_distr[:, 0], exp_distr[:, 1], exp_distr[:, 2], exp_distr[:, 3], exp_distr[:, 4], exp_distr[:, 5], dt, h=simulation_step) for j in range(n_realization))


    elif scaling == 0:
        a_use = mean_data[0] * np.ones(len(exp_distr[:, 0]))
        b_use = mean_data[1] * np.ones(len(exp_distr[:, 0]))
        tau_use = mean_data[3] * np.ones(len(exp_distr[:, 0]))
        f_use = mean_data[2] * np.ones(len(exp_distr[:, 0]))
        B_use = mean_data[4] * np.ones(len(exp_distr[:, 0]))
        err_use = mean_data[5] * np.ones(len(exp_distr[:, 0]))

        xall_list = Parallel(n_jobs=threads)(delayed(cre_trjs_osc_sincos)(ltrjt, a_use, b_use, f_use, tau_use, B_use, err_use, dt, h=simulation_step) for j in range(n_realization))
        if dim == 2:
            yall_list = Parallel(n_jobs=threads)(delayed(cre_trjs_osc_sincos)(ltrjt, a_use, b_use, f_use, tau_use, B_use, err_use, dt, h=simulation_step) for j in range(n_realization))


    else:
        sim_data = np.abs(np.random.multivariate_normal(mean_data, scaling * cov_data, (len(exp_distr[:,0]), n_realization)))
        xall_list = Parallel(n_jobs=threads)(delayed(cre_trjs_osc_sincos)(ltrjt, sim_data[:, j][:, 0], sim_data[:, j][:, 1], sim_data[:, j][:, 2], sim_data[:, j][:, 3], sim_data[:, j][:, 4], sim_data[:, j][:, 5], dt, h=simulation_step) for j in range(n_realization))
        if dim == 2:
            yall_list = Parallel(n_jobs=threads)(delayed(cre_trjs_osc_sincos)(ltrjt, sim_data[:, j][:, 0], sim_data[:, j][:, 1], sim_data[:, j][:, 2], sim_data[:, j][:, 3], sim_data[:, j][:, 4], sim_data[:, j][:, 5], dt, h=simulation_step) for j in range(n_realization))

        for k in range(10):
            np.save(path_out + '/locfitsincos' + name + 'origdistr' + str(int(k)) + mean_str + '_idx' + str(int(index_vacf)) + '_var' + scale_str + '_av' + str(int(av)), sim_data[:, k])

    for i in range(n_realization):
        if dim ==2:
            if der == 's0':
                vxall = combine_fun(lambda x: np.diff(x) / dt, xall_list[i][0], lmodified=1)
                vyall = combine_fun(lambda x: np.diff(x) / dt, yall_list[i][0], lmodified=1)
            else:
                vxall = combine_fun(lambda x: np.gradient(x)[1:-1] / dt, xall_list[i][0], lmodified=2)
                vyall = combine_fun(lambda x: np.gradient(x)[1:-1] / dt, yall_list[i][0], lmodified=2)

            vxacf_all = combine_fun(correlation, vxall)
            vyacf_all = combine_fun(correlation, vyall)
            vacf_all = 0.5 * (vxacf_all + vyacf_all)
            
        elif dim == 1:
            if der == 's0':
                vall = combine_fun(lambda x: np.diff(x) / dt, xall_list[i][0], lmodified=1)
            else:
                vall = combine_fun(lambda x: np.gradient(x)[1:-1] / dt, xall_list[i][0], lmodified=2)
            vacf_all = combine_fun(correlation, vall)
        
        if np.any(np.isnan(vacf_all.compressed())) or np.any(np.isinf(vacf_all.compressed())):
            print('simulated trajectory leads to inf or nan VACF')
        if der == 's0':
            optis, costs, avres, avcosts = optimize_all_cells_osc_av(vacf_all, dt, max_vals=max_vals, min_vals=min_vals, av=av, msd_fun=msd_osc_analytic, index=index_vacf, threads=threads)
        else:
            optis, costs, avres, avcosts = optimize_all_cells_osc_av_mid(vacf_all, dt, max_vals=max_vals, min_vals=min_vals, av=av, msd_fun=msd_osc_analytic, index=index_vacf, threads=threads)

        np.save(path_out + '/locfit' + name + str(int(i)) + mean_str + '_idx' + str(int(index_vacf)) + '_var' + scale_str + '_av' + str(int(av)), optis) # + '_steps' + str(int(n_steps))
        print(i)
        
########################################################################################################################################################################


def calc_dict_from_data(path, lens_name, min_steps, dt, dx, filter_agree=0, filter_corr=0, salmonella=False, return_pos=False):
    # load data
    dictionary = io.loadmat(path)
    data = dictionary['data'] # 726,932 times 15 matrix
    # 1st column is particle ID (just counting rows)
    # 2 frame number (containing information about time point)
    # 3 x-pos with pixel res
    # 4 y-pos with pixel res
    # 5 intensity of maximum
    # 6 track ID (used to filter trajectories)
    # 7-9 not important
    # 10 x-pos via gauss fit
    # 11 y-pos via gauss fit
    # 12 peak intensity of particle
    # 13 radius of point spread function (in pixel)
    # 14 original subpixel x-pos in case of salmonella data
    # 15 original subpixel y-pos in case of salmonella data

    # determine lengths of trjs
    try:
        lens = np.load(lens_name)
    except:
        lens = np.zeros(int(np.max(data[:, 0])))
        for i in range(int(np.max(data[:, 0]))):
            lens[i] = len(np.where(data[:, 5] == i)[0])
        np.save(lens_name, lens)
    
    max_len = int(np.max(lens))

    # extract position arrays from data
    indx_lens = np.where(lens > min_steps)[0]
    n_trj = len(indx_lens)

    trjx_matrix = np.ma.empty((n_trj, max_len))
    trjx_matrix.mask = True
    trjy_matrix = np.ma.empty((n_trj, max_len))
    trjy_matrix.mask = True

    trjx_matrix_orig = np.ma.empty((n_trj, max_len))
    trjx_matrix_orig.mask = True
    trjy_matrix_orig = np.ma.empty((n_trj, max_len))
    trjy_matrix_orig.mask = True

    trjxp_matrix = np.ma.empty((n_trj, max_len))
    trjxp_matrix.mask = True
    trjyp_matrix = np.ma.empty((n_trj, max_len))
    trjyp_matrix.mask = True

    trjxycorr_matrix = np.ma.empty((n_trj, max_len))
    trjxycorr_matrix.mask = True

    for i in range(n_trj):
        indx_data = np.where(data[:, 5] == indx_lens[i])[0] # indices where length is larger than min_steps

        trjx_matrix[i, :int(lens[indx_lens[i]])] = data[indx_data, 9]
        trjy_matrix[i, :int(lens[indx_lens[i]])] = data[indx_data, 10] # subpixel resolution data
        if salmonella:
            trjx_matrix_orig[i, :int(lens[indx_lens[i]])] = data[indx_data, 13]
            trjy_matrix_orig[i, :int(lens[indx_lens[i]])] = data[indx_data, 14] # subpixel resolution data without correction for convection

        trjxycorr_matrix[i, :int(lens[indx_lens[i]])] = data[indx_data, 11]

        trjxp_matrix[i, :int(lens[indx_lens[i]])] = data[indx_data, 2]
        trjyp_matrix[i, :int(lens[indx_lens[i]])] = data[indx_data, 3] #pixel resolution data

    if filter_agree > 0:
        sum_devx = np.zeros(n_trj)
        sum_devy = np.zeros(n_trj)
        if salmonella:
            for i in range(n_trj):
                sum_devx[i] = np.sum((trjxp_matrix[i,:].compressed() - trjx_matrix_orig[i,:].compressed())**2)
                sum_devy[i] = np.sum((trjyp_matrix[i,:].compressed() - trjy_matrix_orig[i,:].compressed())**2)
        else:
            for i in range(n_trj):
                sum_devx[i] = np.sum((trjxp_matrix[i,:].compressed() - trjx_matrix[i,:].compressed())**2)
                sum_devy[i] = np.sum((trjyp_matrix[i,:].compressed() - trjy_matrix[i,:].compressed())**2)

        indx_agree = np.where(np.logical_and(sum_devx/lens[indx_lens] < filter_agree, sum_devy/lens[indx_lens] < filter_agree))[0]
        # less than filter_agree pixel per point difference between Gauss fit and pixel resolution position for x and y
        lens_final = lens[indx_lens][indx_agree]
        xpos_final = trjx_matrix[indx_agree, :] * dx #convert from pixel to mu m
        ypos_final = trjy_matrix[indx_agree, :] * dx #convert from pixel to mu m

    elif filter_corr > 0:
        med_xycorr_sub = np.zeros(len(trjxycorr_matrix[:, 0]))
        # mean_xycorr_sub = np.zeros(len(trjxycorr_matrix[:, 0]))
        for i in range(len(trjxycorr_matrix[:, 0])):
            med_xycorr_sub[i] = np.median(trjxycorr_matrix[i, :].compressed())
            # mean_xycorr_sub[i] = np.mean(trjxycorr_matrix[i, :].compressed())

        indx_agree = np.where(med_xycorr_sub > filter_corr)[0]

        lens_final = lens[indx_lens][indx_agree]
        xpos_final = trjx_matrix[indx_agree, :] * dx #convert from pixel to mu m
        ypos_final = trjy_matrix[indx_agree, :] * dx #convert from pixel to mu m

        # xpos_final_pixel = trjxp_matrix[indx_agree, :] * dx #convert from pixel to mu m
        # ypos_final_pixel = trjyp_matrix[indx_agree, :] * dx #convert from pixel to mu m

    else:
        lens_final = lens[indx_lens]
        xpos_final = trjx_matrix * dx #convert from pixel to mu m
        ypos_final = trjy_matrix * dx #convert from pixel to mu m
        
    # filter out nan values
    indx_notnan = np.where(np.logical_and(np.logical_not(np.isnan(xpos_final).any(axis=1)), np.logical_not(np.isnan(ypos_final).any(axis=1))))[0]

    xpos_final = xpos_final[indx_notnan, :]
    ypos_final = ypos_final[indx_notnan, :]

    max_len_filt = int(np.max(lens_final[indx_notnan]))
    xpos_final = xpos_final[:, :max_len_filt]
    ypos_final = ypos_final[:, :max_len_filt]

    # compute correlation functions etc. in beads_dict_short
    beads_dict_short = analysis_short(xpos_final, dt, ypos_final)

    if return_pos:
        return beads_dict_short, xpos_final, ypos_final
    else:
        return beads_dict_short

def log_smoothing(data, n): #pref, log_fac):
    # function tested by plotting noisy data and smoothed data
    # n is the length of the data after smoothing
    log_inds = np.unique(np.int64((np.logspace(0, np.log10(len(data)), n))))
    mid_points = np.int64((log_inds + np.append(0, log_inds[:-1])) / 2) # indices at which smooth function values are evaluated
    #print(len(mid_points))
    smooth_data = np.zeros(len(log_inds))
    ind_before = 0
    for i in range(len(log_inds)):
        smooth_data[i] = np.sum(data[ind_before : log_inds[i]]) / (log_inds[i] - ind_before) #average over exponentially increasing window
        ind_before = log_inds[i]

    return mid_points, smooth_data

    
############################################################################################################################################

#plotting functions
@mpltex.acs_decorator    
def plot_fit_vacf(opt_params, vacf_matrix, dt, idx=100, vacf_fun=vacf_noise_osc, msd_fun=msd_osc_analytic, n_plot='all', plot_kern=False, dpi=300, save_name='', save_i=None, c_plot=['r']):
    """Plot fit of VACF with noise model"""
    line_styles = ['-', '--', '-.', ':'][:len(opt_params)]
    if len(c_plot) != len(opt_params):
        c_plot = [cm.jet(l) for l in np.linspace(0, 1, len(opt_params))]
    t = np.arange(idx + 5) * dt # time up to chosen index and some more, because kernel and vacf_noise functions can shorten array
    if n_plot == 'all':
        n_plot = range(len(opt_params[0][:, 0]))
    for j in n_plot:
        fig, ax = plt.subplots(1, 1, dpi=dpi)
        ax.scatter(t[:idx], vacf_matrix[j, :idx])
        for i in range(len(opt_params)):
            vacf_noise = vacf_fun(*opt_params[i][j,:], t, msd=msd_fun)
            ax.plot(t[:idx], vacf_noise[:idx], c=c_plot[i], ls=line_styles[i])

        ax.set_xlabel(r'$t$ $[s]$')
        ax.set_ylabel(r'$C_{vv}(t)$ $[\mu m^2/s^{2}]$')
        print(j, opt_params[i][j,:])

        if save_i == j:
            plt.savefig('vacf_fit' + save_name + '.png', bbox_inches='tight')
        plt.show()
        
        if plot_kern:
            fig, ax = plt.subplots(1, 1, dpi=dpi)
            for i in range(len(opt_params)):
                vacf_noise = vacf_fun(*opt_params[i][j,:], t, msd=msd_fun)
                kerneljx = kernel_1dcc(vacf_noise, dt)
                ax.plot(t[:idx], kerneljx[:idx], c=c_plot[i])

            kernel_av = kernel_1dcc(vacf_matrix[j, :idx], dt)
            ax.scatter(t[:idx], kernel_av[:idx])
            
            ax.set_xlabel(r'$t$ $[s]$')
            ax.set_ylabel(r'$\Gamma(t)$ $[s^{-2}]$')
            plt.show()


@mpltex.acs_decorator
def plot_distr_sim(exp_distr, sim_distr, max_vals, min_vals, name, units=[r' [$\rm{min}$]', r' [$\mu m^2/\rm{min}^2$]'], labels=[r'$\tau$', r'$B$'], filt_err=True):
    fig, ax = plt.subplots(1, 1, dpi=300)
    if filt_err:
        cond_exp = filt_distr(exp_distr, min_vals[-1], max_vals[-1], min_factor=1.5, max_factor=0.99)
        cond_sim = filt_distr(sim_distr, min_vals[-1], max_vals[-1], min_factor=1.5, max_factor=0.99)
    else:
        cond_exp = filt_distr(exp_distr, min_vals[-1], max_vals[-1], min_factor=0.9, max_factor=1.1)
        cond_sim = filt_distr(sim_distr, min_vals[-1], max_vals[-1], min_factor=0.9, max_factor=1.1)

    exp_distr = exp_distr[cond_exp, :]
    sim_distr = sim_distr[cond_sim, :]
    ax.scatter(exp_distr[:, 0], exp_distr[:, 1], label='exp')
    ax.scatter(sim_distr[:, 0], sim_distr[:, 1], marker='x', label='sim')
    ax.set_xlabel(labels[0] + units[0])
    ax.set_ylabel(labels[1] + units[1])
    ax.set_title(name)

    plt.show()




