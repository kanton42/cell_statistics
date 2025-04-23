from scipy.stats import chi2
import numpy as np

from scipy.spatial.distance import pdist, cdist, squareform
from scipy.stats import cramervonmises_2samp
from joblib import Parallel, delayed


def var_test(x, va0, direction="two-tailed", alpha=0.05):
    # test if variance equals given variance va0 at alpha level
    # direction determins whether alternative hypothesis says var_exp > va0 ('upper'), vaer_exp < va0 ('lower'), or both possibilities (every other string)
    n = len(x)
    varx = np.var(x)
    Q = (n - 1) * varx / va0
    print(varx / va0)
    if direction == "lower":
        q = chi2.ppf(alpha, n - 1)
        konf = (n - 1) * varx / q
        print('confidence interval:', konf)
        if Q <= q:

            return "H_0 rejected"
        else:
            return "H_0 not rejected"
    elif direction == "upper":
        q = chi2.ppf(1 - alpha, n - 1)
        konf = (n - 1) * varx / q
        print('confidence interval:', konf)
        if Q >= q:
            return "H_0 rejected"
        else:
            return "H_0 not rejected"
    else:
        q1 = chi2.ppf(alpha / 2, n - 1)
        q2 = chi2.ppf(1 - (alpha / 2), n - 1)
        konf1 = (n - 1) * varx / q1
        konf2 = (n - 1) * varx / q2
        print('confidence interval:', konf1, konf2)
        if Q <= q1 or Q >= q2:
            return "H_0 rejected"
        else:
            return "H_0 not rejected"



def test_cov(cov_theo, data_in, alpha=0.05):
    # data has to be in format [[tau1, tau2, ..., taum], [B1, B2, ..., Bm], [if more than 2 params same with others]] (no 1d data accepted)
    # function tests whether covariance of data is equal to cov_theo with probability 1-alpha under assumption of H0
    # chance of alpha that measured covariance stems from theoretical prediction when test result says it does not
    if len(data_in[:,0]) != len(cov_theo[0,:]):
        data = np.copy(data_in.T)
    else:
        data = np.copy(data_in)

    N = len(data[0]) # N measurements of params lead to mu_data, cov_data
    cov_data = np.cov(data) # since numpy definition divides sum of squared distances to mean by N-1 by default
    dim_data = len (cov_data[0])
    p = dim_data # number of different parameters in observation vector
    dof = 0.5 * p * (p + 1)
    B_matr = cov_data

    inv_cov_theo = np.linalg.inv(cov_theo)
    # theoretical covariance matrix has to be invertible
    matrix_product = np.matmul(B_matr, inv_cov_theo)
    trace = np.trace(matrix_product)
    det = np.linalg.det(matrix_product)
        
    #lambda_crit = np.exp(p*(N-1)/2) * det**((N-1)/2) * np.exp(-(N-1)/2 * trace)
    #compare = -2*np.log(lambda_crit)
    compare = -p*(N-1) - (N-1)*np.log(det) + (N-1)*trace # easier to apply log directly to avoid overflow in exp for large N
    #table of correct c to compare to -2ln(l) are listed in Davidson 1971 e.g. c = 11.9015, 1% , p=2, N=20,  c = 8.1283, 5% p=2, N=20
    c = chi2.ppf(1-alpha, dof)
    
    if compare >= c:
        return 'H is rejected', compare, c 
    elif compare < c:
        return 'H is accepted', compare, c 
    else:
        return 'error', compare, c 





def multivariate_test_mean_cov(mu_theo, cov_theo, data_in, alpha=0.05):
    # data has to be in format [[tau1, tau2, ..., taum], [B1, B2, ..., Bm], [if more than 2 params same with others]] (no 1d data accepted)
    # function tests whether mean and covariance of data is stemming from mu_theo and cov_theo with probability 1-alpha under assumption of H0
    # chance of alpha that measured mean and variance are stemming from theoretical prediction when test result says it does not

    if len(data_in[:,0]) != len(mu_theo):
        data = np.copy(data_in.T)
    else:
        data = np.copy(data_in)
    
    inv_cov_theo = np.linalg.inv(cov_theo)
    # theoretical covariance matrix has to be invertible
    mu_data = np.mean(data, axis=1)
    cov_data = np.cov(data) # numpy divides by (N-1) per default
    dim_data = len(cov_data[0])
    
    N = len(data[0]) # N measurements of params lead to mu_data, cov_data
    p = dim_data # number of different parameters in observation vector
    dof = 0.5 * p * (p + 1) + p
    
    diff_of_mean = mu_data - mu_theo
    B_matr = (N-1) * cov_data # according to Davis 1971
    #B_matr = N * np.outer(diff_of_mean, diff_of_mean.T) + (N-1) * cov_data # according to Anderson book
    matrix_product = np.matmul(B_matr, inv_cov_theo)
    
    #lambda_crit = (np.e/N)**(0.5*p*N) * np.linalg.det(matrix_product)**(0.5*N) * np.exp(-0.5 * (np.trace(matrix_product + N * np.outer(diff_of_mean, np.matmul(inv_cov_theo, diff_of_mean)))))
    #lambda_crit = -2*np.log(lambda_crit) # old version used to calculate first lambda than log, but is inefficient for large N
    
    # values of correction factors for -2ln(l) in Davis 1971
    lambda_crit = - p*N * (1-np.log(N)) - N * np.log(np.linalg.det(matrix_product)) + np.trace(matrix_product + N * np.outer(diff_of_mean, np.matmul(inv_cov_theo, diff_of_mean)))
    # probability ratio criterion, where -2ln(lambda) is asymptotically chi2 distributed # easier to apply log directly to avoid overflow in exp for large N
    
    c = chi2.ppf(1-alpha, dof) # very good approximation vor large N (>100)
    
    if lambda_crit >= c:
        return 'H is rejected', lambda_crit, c
    elif lambda_crit < c:
        return 'H is accepted', lambda_crit, c
    else:
        return 'error', lambda_crit, c
    


def multivariate_test_mean_cov_gauss(data_exp, data_in, alpha=0.05):
    # data has to be in format [[tau1, tau2, ..., taum], [B1, B2, ..., Bm], [if more than 2 params same with others]] (no 1d data accepted)
    # function tests whether mean and covariance of data are stemming from mu_theo and cov_theo with probability 1-alpha under assumption of H0
    # chance of alpha that measured mean and variance are stemming from theoretical prediction when test result says it does not

    mu_theo = np.mean(data_exp, axis=0)
    cov_theo = np.cov(data_exp)

    if len(np.mean(data_in, axis=0)) != len(mu_theo):
        data = np.copy(data_in.T)
    else:
        data = np.copy(data_in)

    inv_cov_theo = np.linalg.inv(cov_theo)
    # theoretical covariance matrix has to be invertible
    mu_data = np.mean(data, axis=0)
    cov_data = np.cov(data) # numpy divides by (N-1) per default
    dim_data = len(cov_data[0])
    
    N = len(data[0]) # N measurements of params lead to mu_data, cov_data
    p = dim_data # number of different parameters in observation vector
    dof = 0.5 * p * (p + 1) + p
    
    diff_of_mean = mu_data - mu_theo
    B_matr = (N-1) * cov_data # according to Davis 1971
    #B_matr = N * np.outer(diff_of_mean, diff_of_mean.T) + (N-1) * cov_data # according to Anderson book
    matrix_product = np.matmul(B_matr, inv_cov_theo)
    
    #lambda_crit = (np.e/N)**(0.5*p*N) * np.linalg.det(matrix_product)**(0.5*N) * np.exp(-0.5 * (np.trace(matrix_product + N * np.outer(diff_of_mean, np.matmul(inv_cov_theo, diff_of_mean)))))
    #lambda_crit = -2*np.log(lambda_crit) # old version used to calculate first lambda than log, but is inefficient for large N
    
    # values of correction factors for -2ln(l) in Davis 1971
    lambda_crit = - p*N * (1-np.log(N)) - N * np.log(np.linalg.det(matrix_product)) + np.trace(matrix_product + N * np.outer(diff_of_mean, np.matmul(inv_cov_theo, diff_of_mean)))
    # probability ratio criterion, where -2ln(lambda) is asymptotically chi2 distributed # easier to apply log directly to avoid overflow in exp for large N
    
    c = chi2.ppf(1-alpha, dof) # very good approximation vor large N (>100)
    
    if lambda_crit >= c:
        return 1 # H is rejected
    elif lambda_crit < c:
        return 0 # H is accepted


###########
#check that test leads to right statistics

def check_test_cov(nt, cov_theo, alpha, N):
    count1 = 0
    for j in range(nt):
        data1 = np.random.multivariate_normal(np.array([0,0]), cov_theo, N).T # N random multivariate normal numbers
        if test_cov(cov_theo, data1, alpha=alpha)[0] == 'H is rejected':
            count1 += 1
    # count all rejections, then divide by total number of tested distributions to obtain probability of rejection by chance (even though covariances are the same)
    # one could call output "true alpha" since it gives the actually observed rejection chance
    return(count1/nt)


def check_test_mean_cov(nt, mu_theo, cov_theo, alpha, N):
    count1 = 0
    for j in range(nt):
        data1 = np.random.multivariate_normal(mu_theo, cov_theo, N).T # N random multivariate normal numbers
        if multivariate_test_mean_cov(mu_theo, cov_theo, data1, alpha=alpha)[0] == 'H is rejected':
            count1 += 1
    # count all rejections, then divide by total number of tested distributions to obtain probability of rejection by chance (even though covariances are the same)
    # one could call output "true alpha" since it gives the actually observed rejection chance
    return(count1/nt)

def check_test_var(ntest, var, alpha, nsample):
    count=0
    for i in range(ntest):
        rand=np.random.normal(0, var, nsample)
        if var_test(rand, var, alpha = alpha) == 'H_0 not rejected':
            1==1
        else:
            count+=1
            
    return count/ntest


###########################################################################################################
# test to compare two sample distributions (compare entire distributions, not just first or second moment)

def multivariate_DD_test(obs1, obs2, n_perm=1000):
    '''Perform a multivariate DD test (based on Montero-Manso 2018) to determine whether two samples are drawn from the same distribution.
    Arguments:
    obs1 -- array-like object representing the first set of observations (n x d matrix) d is dimension e.g. [[a1, b1, c1], [a2, b2, c2]]
    obs2 -- array-like object representing the second set of observations (m x d matrix)
    n_perm -- number of random permutations used to approximate p_value

    Returns:
    p_value -- p-value (how likely is it to reject H0 by chance even though it is true)
    '''

    if obs1.shape[1] > obs1.shape[0]:
        obs1 = obs1.T
    if obs2.shape[1] > obs2.shape[0]:
        obs2 = obs2.T

    # compute intra distances in each sample
    intra_distances = pdist(obs1)
    intra_distances = np.append(intra_distances, pdist(obs2)) # n(n-1)/2 + m(m-1)/2 entries

    # compute inter distances between samples
    inter_distances = cdist(obs1, obs2).flatten() # n * m entries

    # testing if obs1 and obs2 come from the same distribution is equivalent to
    # testing if inter_distances and intra_distances stem from the same distribution
    res = cramervonmises_2samp(inter_distances, intra_distances)
    statistic_orig = res.statistic

    all_distances = np.append(inter_distances, intra_distances)
    rng = np.random.default_rng()
    pvalue_sum = 0

    # never possible to go through all possible permutations for actual data,
    # thus approximate by large number of random permutations n_perm

    for iter in range(n_perm):
        # permute randomly
        rng.shuffle(all_distances)
        # assign randomly shuffled distances to new subsets
        new_inter_dist = all_distances[:len(inter_distances)]
        new_intra_dist = all_distances[len(inter_distances):]

        # compare permuted test statistic (Cramer- von Mises two sample) to original test statistic
        res_perm = cramervonmises_2samp(new_inter_dist, new_intra_dist)
        if res_perm.statistic > statistic_orig:
            pvalue_sum += 1

    p_value = pvalue_sum / n_perm
    # if p_value < alpha reject H0, i.e. distributions are not equal

    return p_value


def multivariate_phantasy_test(obs1, obs2, n_perm=1000):
    '''Perform a multivariate test to determine whether two samples are drawn from the same distribution.
    Arguments:
    obs1 -- array-like object representing the first set of observations (n x d matrix) d is dimension e.g. [[a1, b1, c1], [a2, b2, c2]]
    obs2 -- array-like object representing the second set of observations (m x d matrix)
    n_perm -- number of random permutations used to approximate p_value

    Returns:
    p_value -- p-value (how likely is it to reject H0 by chance even though it is true)
    '''

    if obs1.shape[1] > obs1.shape[0]:
        obs1 = obs1.T
    if obs2.shape[1] > obs2.shape[0]:
        obs2 = obs2.T

    # compute intra distances in each sample
    intra_distances = pdist(obs1)
    intra_distances = np.append(intra_distances, pdist(obs2)) # n(n-1)/2 + m(m-1)/2 entries

    # compute inter distances between samples
    inter_distances = cdist(obs1, obs2).flatten() # n * m entries

    # testing if obs1 and obs2 come from the same distribution is equivalent to
    # testing if inter_distances and intra_distances stem from the same distribution
    statistic_orig = np.abs(np.mean(inter_distances) - np.mean(intra_distances))

    all_distances = np.append(inter_distances, intra_distances)
    rng = np.random.default_rng()
    pvalue_sum = 0

    # never possible to go through all possible permutations for actual data,
    # thus approximate by large number of random permutations n_perm

    for iter in range(n_perm):
        # permute randomly
        rng.shuffle(all_distances)
        # assign randomly shuffled distances to new subsets
        new_inter_dist = all_distances[:len(inter_distances)]
        new_intra_dist = all_distances[len(inter_distances):]

        # compare permuted test statistic to original test statistic
        statistic_perm = np.abs(np.mean(new_inter_dist) - np.mean(new_intra_dist))
        if statistic_perm > statistic_orig:
            pvalue_sum += 1

    p_value = pvalue_sum / n_perm
    # if p_value < alpha reject H0, i.e. distributions are not equal

    return p_value


###########################################################################################################



def dist_perm_test(obs1, obs2, n_perm, test='mean'):
    if np.any(np.isinf(obs1)) or np.any(np.isinf(obs2)):
        print('inf Error: inf included in distribution. Cannot compare covariances.')
        return 'inf Error: inf included in distribution. Cannot compare covariances.'
    if np.any(np.isnan(obs1)) or np.any(np.isnan(obs2)):
        print('nan Error: nan included in distribution. Cannot compare covariances.')
        return 'nan Error: nan included in distribution. Cannot compare covariances.'
    if len(obs1[:, 0]) == 0:
        print('Error: empty distribution. Cannot compare covariances.')    

    if obs1.shape[1] > obs1.shape[0]:
        obs1 = obs1.T
        obs2 = obs2.T

    dist_1 = pdist(obs1)
    dist_2 = pdist(obs2)

    if test == 'mean':
        statistic = np.abs(np.mean(dist_1) - np.mean(dist_2))
    else:
        statistic = np.abs(np.median(dist_1) - np.median(dist_2))
    # elif test == 'cov' or test == 'var' or test == 'std':
    #     statistic = np.abs(np.std(dist_1) - np.std(dist_2))
    # else:
    #     statistic = np.abs(np.std(dist_1) - np.std(dist_2)) + np.abs(np.median(dist_1) - np.median(dist_2))

    all_obs = np.append(obs1, obs2, axis=0)

    rng = np.random.default_rng()
    pvalue_sum = 0

    for iter in range(n_perm):
        # permute randomly
        rng.shuffle(all_obs)
        # assign randomly shuffled distances to new subsets

        new_dist_1 = pdist(all_obs[:len(obs1[:, 0]), :])
        new_dist_2 = pdist(all_obs[len(obs1[:, 0]):, :])

        # compare permuted test statistic (made up) to original test statistic
        if test == 'mean':
            statistic_perm = np.abs(np.mean(new_dist_1) - np.mean(new_dist_2))
        else:
            statistic_perm = np.abs(np.median(new_dist_1) - np.median(new_dist_2))
        # elif test == 'cov' or test == 'var' or test == 'std':
        #     statistic_perm = np.abs(np.std(new_dist_1) - np.std(new_dist_2))
        # else:
        #     statistic_perm = np.abs(np.std(new_dist_1) - np.std(new_dist_2)) + np.abs(np.median(new_dist_1) - np.median(new_dist_2))

        if statistic_perm > statistic:
            pvalue_sum += 1

    # how often does permutation lead to a larger test statistic than the original one (more different cov means higher p-value)
    p_value = pvalue_sum / n_perm

    return p_value


########################################################################################################
# test based on permutation, that solely compares covariance matrices

def covariance_diff_perm_test(obs1, obs2, n_perm, off_diag=False):
    if np.any(np.isinf(obs1)) or np.any(np.isinf(obs2)):
        print('inf Error: inf included in distribution. Cannot compare covariances.')
        return 'inf Error: inf included in distribution. Cannot compare covariances.'
    if np.any(np.isnan(obs1)) or np.any(np.isnan(obs2)):
        print('nan Error: nan included in distribution. Cannot compare covariances.')
        return 'nan Error: nan included in distribution. Cannot compare covariances.'
    if len(obs1[:, 0]) == 0:
        print('Error: empty distribution. Cannot compare covariances.')    

    cov1 = np.cov(obs1.T)
    cov2 = np.cov(obs2.T)

    if off_diag:
        statistic_obs = np.sum((cov1 - cov2)**2)
    else:
        statistic_obs = np.sum((cov1 - cov2)**2) - np.sum(np.triu(cov1 - cov2, 1)**2)

    all_obs = np.append(obs1, obs2, axis=0)
    rng = np.random.default_rng()
    pvalue_sum = 0

    for iter in range(n_perm):
        # permute randomly
        rng.shuffle(all_obs)
        # assign randomly shuffled distances to new subsets
        new_cov1 = np.cov(all_obs[:len(obs1[:, 0]), :].T)
        new_cov2 = np.cov(all_obs[len(obs1[:, 0]):, :].T)

        # compare permuted test statistic (made up) to original test statistic
        statistic_perm = np.sum((new_cov1 - new_cov2)**2)
        if not off_diag:
            # subtract triangular part of covariance matrix, such that every unique element is only counted once
            statistic_perm -= np.sum(np.triu(new_cov1 - new_cov2, 1)**2)

        if statistic_perm > statistic_obs:
            pvalue_sum += 1

    # how often does permutation lead to a larger test statistic than the original one (more different cov means higher p-value)
    p_value = pvalue_sum / n_perm

    return p_value
    
###################################################################################################
# checking that multivariate two sample test give correct level of rejection for distributions with same covariance

def check_multivariate_perm_test(n_test, N, alpha=0.05, threads=6, n_perm=1000, mean=np.array([20, 100]), var=[1, 1], distr='gauss', test_fun=multivariate_DD_test, args=[]):

    if distr == 'gauss':
        observations1 = np.random.multivariate_normal(mean, var[0] * np.eye(len(mean)) , (N, n_test)) #+ np.array([[0, 0.1], [0.1, 0.1]])
        observations2 = np.random.multivariate_normal(mean, var[1] * np.eye(len(mean)), (N, n_test))
    else:
        observations1 = var[0] * np.random.rand(N * n_test * len(mean)).reshape(N, n_test, len(mean)) * mean[0]
        observations2 = var[1] * np.random.rand(N * n_test * len(mean)).reshape(N, n_test, len(mean)) * mean[0]

    p_values = Parallel(n_jobs=threads)(delayed(test_fun)(observations1[:, j], observations2[:, j], n_perm, *args) for j in range(n_test))
    p_values = np.array(p_values)

    print('alpha level:', alpha, 'ratio of rejection by chance:', len(np.where(p_values < alpha)[0]) / n_test)

    return p_values



#########################################################################################################################################
# Chat-GPT code (not tested):

# from scipy.stats import f


# def multivariate_two_sample_test(obs1, obs2):
#     """
#     Perform a multivariate two-sample test (Hotelling's T-squared test) to test whether two matrices of observations are
#     drawn from the same distribution.

#     Arguments:
#     obs1 -- numpy array representing the first matrix of observations (shape: m x n1)
#     obs2 -- numpy array representing the second matrix of observations (shape: m x n2)

#     Returns:
#     stat -- test statistic (Hotelling's T-squared)
#     p_value -- p-value associated with the test statistic
#     """

#     # Compute sample means and sample covariance matrices
#     mean1 = np.mean(obs1, axis=1, keepdims=True)
#     mean2 = np.mean(obs2, axis=1, keepdims=True)
#     cov1 = np.cov(obs1)
#     cov2 = np.cov(obs2)

#     # Compute the pooled covariance matrix
#     n1 = obs1.shape[1]
#     n2 = obs2.shape[1]
#     pooled_cov = ((n1 - 1) * cov1 + (n2 - 1) * cov2) / (n1 + n2 - 2)

#     # Compute the test statistic (Hotelling's T-squared)
#     diff = mean1 - mean2
#     inv_pooled_cov = np.linalg.inv(pooled_cov)
#     stat = (n1 * n2 / (n1 + n2)) * np.matmul(np.matmul(diff.T, inv_pooled_cov), diff)

#     # Compute the degrees of freedom
#     num_features = obs1.shape[0]
#     df1 = num_features
#     df2 = n1 + n2 - num_features - 1

#     # Compute the p-value using the F-distribution
#     p_value = 1 - f.cdf(stat, df1, df2)

#     return stat, p_value


# # Example usage
# observations1 = np.array([[1.2, 1.5, 2.3, 0.8, 1.9],
#                           [0.9, 1.4, 2.1, 0.5, 1.7]])
# observations2 = np.array([[1.0, 1.3, 2.1, 0.7, 1.8],
#                           [1.1, 1.6, 2.2, 0.6, 1.5]])

# test_statistic, p_value = multivariate_two_sample_test(observations1, observations2)

# print("Test Statistic (Hotelling's T-squared):", test_statistic)
# print("P-value:", p_value)


# from scipy.stats import ks_2samp

# def two_sample_ks_test(obs1, obs2):
#     """
#     Perform a two-sample Kolmogorov-Smirnov (KS) test to test whether two samples are drawn from the same distribution in 1D.

#     Arguments:
#     obs1 -- array-like object representing the first set of observations
#     obs2 -- array-like object representing the second set of observations

#     Returns:
#     stat -- test statistic
#     p_value -- p-value associated with the test statistic
#     """

#     # Perform the KS test
#     res = ks_2samp(obs1, obs2)
#     stat, p_value = res.statistic, res.pvalue

#     return stat, p_value


# # Example usage
# observations1 = np.array([1.2, 1.5, 2.3, 0.8, 1.9])
# observations2 = np.array([0.9, 1.4, 2.1, 0.5, 1.7])

# test_statistic, p_value = two_sample_ks_test(observations1, observations2)

# print("Test Statistic:", test_statistic)
# print("P-value:", p_value)




# from scipy.stats import cramervonmises_2samp

# def two_sample_cramer_von_mises_test(obs1, obs2):
#     """
#     Perform a two-sample Cramér-von Mises test in 1D.

#     Arguments:
#     obs1 -- array-like object representing the first set of observations
#     obs2 -- array-like object representing the second set of observations

#     Returns:
#     stat -- test statistic
#     p_value -- p-value associated with the test statistic
#     """

#     # Perform the test
#     res = cramervonmises_2samp(obs1, obs2)
#     stat, p_value = res.statistic, res.pvalue

#     return stat, p_value


# # Example usage
# observations1 = np.array([1.2, 1.5, 2.3, 0.8, 1.9])
# observations2 = np.array([0.9, 1.4, 2.1, 0.5, 1.7])

# test_statistic, p_value = two_sample_cramer_von_mises_test(observations1, observations2)

# print("Test Statistic:", test_statistic)
# print("P-value:", p_value)