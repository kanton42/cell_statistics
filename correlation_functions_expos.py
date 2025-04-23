from sympy import nroots, roots, symbols, fraction, cancel, Poly, degree
from sympy.abc import x
import numpy as np



def roots_osc_sincos(a, b, f, tau):
    k1 = -2 * (1 + f**2 * tau**2) * (a + 2 * b * tau + a * f**2 * tau**2)
    k2 = 4 * a * tau**2 * (-1 + f**2 * tau**2)
    k3 = -2 * a * tau**4

    c1 = (a + 2 * b * tau + a * f**2 * tau**2)**2
    c2 = (1 + 2 * (a**2 - 3 * b + f**2) * tau**2 + (-2 * a**2 * f**2 + (b + f**2)**2) * tau**4)
    c3 = tau**2 * (2 + tau**2 * (a**2 - 2 * (b + f**2)))
    c4 = tau**4

    nroot = nroots( c4 *x**3 + c3*x**2 + c2*x + c1, n=9, maxsteps=1000)

    return np.complex64(nroot), np.array([k1, k2, k3]), np.array([c1, c2, c3, c4])



def msd_osc_sincos(a, b, f, tau, B, t):
    # due to sin(f)/f term limit of f-->0 does not yield simple exponential behavior in the limit, but larger friction in long time limit
    # kernel of the form: a * delta(t) + b * (cos(ft) + sin(ft)/(f*tau)) * exp(-t/tau)
    r_i, k_i, c_i = roots_osc_sincos(a, b, f, tau)

    summe = 0
    for i in range(len(r_i)):
        summe += (np.exp(-np.sqrt(-r_i[i]) * t) - 1) / (np.sqrt(-r_i[i]) * np.prod((r_i[i] - np.roll(r_i, -i-1)[:-1]))) * (k_i[0]/r_i[i] + k_i[1] + k_i[2] * r_i[i])

    msd = B/c_i[-1] * (k_i[0] * t / np.prod(r_i) + summe)
    return np.real(msd)


def vacf_osc_sincos(a, b, f, tau, B, t):
    # due to sin(f)/f term limit of f-->0 does not yield simple exponential behavior in the limit, but larger friction in long time limit
    # kernel of the form: a * delta(t) + b * (cos(ft) + sin(ft)/(f*tau)) * exp(-t/tau)
    r_i, k_i, c_i = roots_osc_sincos(a, b, f, tau)

    summe = 0
    for i in range(len(r_i)):
        summe += (np.exp(-np.sqrt(-r_i[i]) * t)) * np.sqrt(-r_i[i]) / (np.prod((r_i[i] - np.roll(r_i, -i-1)[:-1]))) * (k_i[0]/r_i[i] + k_i[1] + k_i[2] * r_i[i])

    vacf = B/c_i[-1] * summe / 2
    return np.real(vacf)




def roots_osc_sincos_harm(a, b, f, tau, K):
    k1 = -2 * (1 + f**2 * tau**2) * (a + 2 * b * tau + a * f**2 * tau**2)
    k2 = 4 * a * tau**2 * (-1 + f**2 * tau**2)
    k3 = -2 * a * tau**4
    c0 = (K + f**2 * K * tau**2)**2
    c1 = 4 * b**2 * tau**2 + 4 * a * b * tau * (1 + f**2 * tau**2) + (a + a * f**2 * tau**2)**2 + 2 * K * (-1 + (3 * b - 2 * f**2 + K) * tau**2 - f**2 * (b + f**2 + K) * tau**4)
    c2 = 1 + 2 * (a**2 - 3 * b + f**2 - 2 * K) * tau**2 + (-2 * a**2 * f**2 + (b + f**2)**2 + 2 * (b + 2 * f**2) * K + K**2) * tau**4
    c3 = 2 + (a**2 - 2 * (b + f**2 + K)) * tau**2
    c4 = tau**4

    nroot = nroots(c0 + x*(c4*x**3 + c3*x**2 + c2*x + c1), n=15, maxsteps=1000)

    return np.complex64(nroot), np.array([k1, k2, k3]), np.array([c0, c1, c2, c3, c4])

def msd_osc_sincos_harm(a, b, f, tau,  K, B, t):
    # due to sin(f)/f term limit of f-->0 does not yield simple exponential behavior in the limit, but larger friction in long time limit
    # kernel of the form: a * delta(t) + b * (cos(ft) + sin(ft)/(f*tau)) * exp(-t/tau)
    r_i, k_i, c_i = roots_osc_sincos_harm(a, b, f, tau, K)

    summe = 0
    for i in range(len(r_i)):
        summe += (np.exp(-np.sqrt(-r_i[i]) * t) - 1) / (np.sqrt(-r_i[i]) * np.prod((r_i[i] - np.roll(r_i, -i-1)[:-1]))) * (k_i[0] + k_i[1] * r_i[i] + k_i[2] * r_i[i]**2)
    msd = B/c_i[-1] * summe
    return np.real(msd)


def vacf_osc_sincos_harm(a, b, f, tau,  K, B, t):
    # due to sin(f)/f term limit of f-->0 does not yield simple exponential behavior in the limit, but larger friction in long time limit
    # kernel of the form: a * delta(t) + b * (cos(ft) + sin(ft)/(f*tau)) * exp(-t/tau)
    r_i, k_i, c_i = roots_osc_sincos_harm(a, b, f, tau, K)

    summe = 0
    for i in range(len(r_i)):
        summe += (np.exp(-np.sqrt(-r_i[i]) * t)) * np.sqrt(-r_i[i]) / (np.prod((r_i[i] - np.roll(r_i, -i-1)[:-1]))) * (k_i[0] + k_i[1] * r_i[i] + k_i[2] * r_i[i]**2)
    vacf = B/c_i[-1] * summe / 2
    return np.real(vacf)


def pacf_osc_sincos_harm(a, b, f, tau,  K, B, t):
    # due to sin(f)/f term limit of f-->0 does not yield simple exponential behavior in the limit, but larger friction in long time limit
    r_i, k_i, c_i = roots_osc_sincos_harm(a, b, f, tau, K)

    summe = 0
    for i in range(len(r_i)):
        summe += (np.exp(-np.sqrt(-r_i[i]) * t)) / (np.sqrt(-r_i[i]) * np.prod((r_i[i] - np.roll(r_i, -i-1)[:-1]))) * (k_i[0] + k_i[1] * r_i[i] + k_i[2] * r_i[i]**2)
    pacf = -B/c_i[-1] * summe / 2
    return np.real(pacf)


def msd_osc_cos(a, b, f, tau, B, t):
    # tested that same result is obtained when roots are numerically approximated instead of using analytic expression for roots
    # kernel of the form: a * delta(t) + b * cos(ft)  * exp(-t/tau)
    a = np.array(a, dtype=np.float128)
    b = np.array(b, dtype=np.float128)#float(b)
    f = np.array(f, dtype=np.float128)#float(f)
    tau = np.array(tau, dtype=np.float128)#float(tau)
    B = np.array(B, dtype=np.float128)#float(B)
    k1 = -2* (1 + f**2 * tau**2) * (a + b*tau + a * f**2 * tau**2)
    k2 = -2*tau**2 * (b*tau + 2*a * (1 - f**2 * tau**2))
    k3 = -2 * a * tau**4
    c1 = (a + b*tau + a * f**2 * tau**2)**2
    c2 = 1 + 2*tau**2 * (a**2 - b + f**2) + 2*a*b*tau**3 + tau**4 * ((b + f**2)**2 - 2*a**2 * f**2)
    c3 = tau**2 * (2 + tau**2 * (a**2 - 2*(b + f**2)) )
    #print(c1,c2,c3,k1,k2,k3)
    
    nroot = nroots( tau**4 *x**3 + c3*x**2 + c2*x + c1, n=9, maxsteps=1000)
    nroot = np.complex128(nroot)

    # r1, r3, r5 = roots(c1, c2, c3, tau)
    # r1, r3, r5 = np.sqrt(nroot[0]), np.sqrt(nroot[2], np.sqrt(nroot[4]))
    r1sq, r3sq, r5sq = nroot[0], nroot[1], nroot[2]
    r_i = np.array([r1sq, r3sq, r5sq])
    c_i = np.array([c1, c2, c3])
    k_i = np.array([k1, k2, k3])

    summe = 0
    for i in range(len(r_i)):
        summe += (np.exp(-np.sqrt(-r_i[i]) * t) - 1) / (np.sqrt(-r_i[i]) * np.prod((r_i[i] - np.roll(r_i, -i-1)[:-1]))) * (k_i[0]/r_i[i] + k_i[1] + k_i[2] * r_i[i])

    msd = B/tau**4 * (k_i[0] * t / np.prod(r_i) + summe)
    return np.real(msd)

def poles_osc_analytic(a, b, f, tau, B):
    a = np.array(a, dtype=np.float128)
    b = np.array(b, dtype=np.float128)#float(b)
    f = np.array(f, dtype=np.float128)#float(f)
    tau = np.array(tau, dtype=np.float128)#float(tau)
    B = np.array(B, dtype=np.float128)#float(B)
    k1 = -2* (1 + f**2 * tau**2) * (a + b*tau + a * f**2 * tau**2)
    k2 = -2*tau**2 * (b*tau + 2*a * (1 - f**2 * tau**2))
    k3 = -2 * a * tau**4
    c1 = (a + b*tau + a * f**2 * tau**2)**2
    c2 = 1 + 2*tau**2 * (a**2 - b + f**2) + 2*a*b*tau**3 + tau**4 * ((b + f**2)**2 - 2*a**2 * f**2)
    c3 = tau**2 * (2 + tau**2 * (a**2 - 2*(b + f**2)) )
    #print(c1,c2,c3,k1,k2,k3)
    return roots_osc(c1, c2, c3, tau)




def vacf_osc_analytic(a, b, f, tau, B, t):
# kernel of the form: a * delta(t) + b * cos(ft) * exp(-t/tau)
    a = np.array(a, dtype=np.float128)
    b = np.array(b, dtype=np.float128)#float(b)
    f = np.array(f, dtype=np.float128)#float(f)
    tau = np.array(tau, dtype=np.float128)#float(tau)
    B = np.array(B, dtype=np.float128)#float(B)
    k1 = -2* (1 + f**2 * tau**2) * (a + b*tau + a * f**2 * tau**2)
    k2 = -2*tau**2 * (b*tau + 2*a * (1 - f**2 * tau**2))
    k3 = -2 * a * tau**4
    c1 = (a + b*tau + a * f**2 * tau**2)**2
    c2 = 1 + 2*tau**2 * (a**2 - b + f**2) + 2*a*b*tau**3 + tau**4 * ((b + f**2)**2 - 2*a**2 * f**2)
    c3 = tau**2 * (2 + tau**2 * (a**2 - 2*(b + f**2)) )
    #print(c1,c2,c3,k1,k2,k3)
    r1, r3, r5 = roots_osc(c1, c2, c3, tau)
    #print(r1,r3,r5) #roots are same as in mathematica
    diff_r13 = r1**2 - r3**2
    diff_r15 = r1**2 - r5**2
    diff_r35 = r3**2 - r5**2

    I11 = -r1**2 * np.exp(-t*np.sqrt(-r1**2)) / (np.sqrt(-r1**2) * r1**2 * diff_r13 * diff_r15)
    I21 = I11 * r1**2 
    I31 = I21 * r1**2

    I12 = -r3**2 * np.exp(-np.sqrt(-r3**2)*t) / (-np.sqrt(-r3**2) * r3**2 * diff_r13 * diff_r35)
    I22 = I12 * r3**2 
    I32 = I22 * r3**2

    I13 = -r5**2 * np.exp(-np.sqrt(-r5**2)*t) / (np.sqrt(-r5**2) * r5**2 * diff_r15 * diff_r35)
    I23 = I13 * r5**2 
    I33 = I23 * r5**2

    I1 = k1 * (I11 + I12 + I13)
    I2 = k2 * (I21 + I22 + I23)
    I3 = k3* (I31 + I32 + I33)
    #print(B/tau**4, np.real(I1 + I2 + I3))
    vacf = B/tau**4 * np.real(I1 + I2 + I3)/2
    return vacf


def pacf_osc_cos(a, b, f, tau, B, t):
    # test whether same result is obtained when roots are numerically approximated instead of using analytic expression for roots
    a = np.array(a, dtype=np.float128)
    b = np.array(b, dtype=np.float128)#float(b)
    f = np.array(f, dtype=np.float128)#float(f)
    tau = np.array(tau, dtype=np.float128)#float(tau)
    B = np.array(B, dtype=np.float128)#float(B)
    k1 = -2* (1 + f**2 * tau**2) * (a + b*tau + a * f**2 * tau**2)
    k2 = -2*tau**2 * (b*tau + 2*a * (1 - f**2 * tau**2))
    k3 = -2 * a * tau**4
    c1 = (a + b*tau + a * f**2 * tau**2)**2
    c2 = 1 + 2*tau**2 * (a**2 - b + f**2) + 2*a*b*tau**3 + tau**4 * ((b + f**2)**2 - 2*a**2 * f**2)
    c3 = tau**2 * (2 + tau**2 * (a**2 - 2*(b + f**2)) )
    #print(c1,c2,c3,k1,k2,k3)
    
    nroot = nroots( tau**4 *x**3 + c3*x**2 + c2*x + c1, n=9, maxsteps=1000)
    nroot = np.complex128(nroot)

    # r1, r3, r5 = roots(c1, c2, c3, tau)
    # r1, r3, r5 = np.sqrt(nroot[0]), np.sqrt(nroot[2], np.sqrt(nroot[4]))
    r1sq, r3sq, r5sq = nroot[0], nroot[1], nroot[2]
    r_i = np.array([r1sq, r3sq, r5sq])
    c_i = np.array([c1, c2, c3])
    k_i = np.array([k1, k2, k3])

    summe = 0
    for i in range(len(r_i)):
        summe += (np.exp(-np.sqrt(-r_i[i]) * t)) / (np.sqrt(-r_i[i]) * np.prod((r_i[i] - np.roll(r_i, -i-1)[:-1]))) * (k_i[0]/r_i[i] + k_i[1] + k_i[2] * r_i[i])

    pacf = -B/tau**4 * (summe) / 2
    return np.real(pacf)




def mysqrt(x):
    return np.sqrt((1+0j)*x)

def roots_osc(c1, c2, c3, tau):
    aux0=(4.*(((3.*(c2*(tau**4.)))-(c3**2))**3.))+((((-2.*(c3**3.))+((\
    9.*(c2*(c3*(tau**4.))))+(-27.*(c1*(tau**8.)))))**2));
    aux1=(-2.*(c3**3.))+((9.*(c2*(c3*(tau**4.))))+((-27.*(c1*(\
tau**8.)))+(mysqrt(aux0))));
    aux2=(4.*(((3.*(c2*(tau**4.)))-(c3**2))**3.))+((((-2.*(c3**3.))\
+((9.*(c2*(c3*(tau**4.))))+(-27.*(c1*(tau**8.)))))**2));
    aux3=(-2.*(c3**3.))+((9.*(c2*(c3*(tau**4.))))+((-27.*(c1*(\
tau**8.)))+(mysqrt(aux2))));
    aux4=((1/3)*((2.**(1/3))*((c3**2)*((tau**-4.)*(aux1**(-1/3))))))+((1/3)*((2.**(-1/3))*((tau**-4.)*(aux3**(1/3))\
)));
    #if np.any(np.isnan(aux4)) or np.any(np.isinf(aux4)):
    #    print(c1, c2, c3, tau, aux1, aux3)
    aux5=(4.*(((3.*(c2*(tau**4.)))-(c3**2))**3.))+((((-2.*(c3**3.))\
+((9.*(c2*(c3*(tau**4.))))+(-27.*(c1*(tau**8.)))))**2));
    aux6=(-2.*(c3**3.))+((9.*(c2*(c3*(tau**4.))))+((-27.*(c1*(\
tau**8.)))+(mysqrt(aux5))));
    aux7=(((-1/3)*(c3*(tau**-4.)))+aux4)-((2.**(1/3))*(c2*(\
aux6**(-1/3))));
    root1=(-mysqrt(aux7))

    ##

    aux0=(4.*(((3.*(c2*(tau**4.)))-(c3**2))**3.))+((((-2.*(c3**3.))+((\
9.*(c2*(c3*(tau**4.))))+(-27.*(c1*(tau**8.)))))**2));
    aux1=(-2.*(c3**3.))+((9.*(c2*(c3*(tau**4.))))+((-27.*(c1*(\
tau**8.)))+(mysqrt(aux0))));
    aux2=(4.*(((3.*(c2*(tau**4.)))-(c3**2))**3.))+((((-2.*(c3**3.))\
+((9.*(c2*(c3*(tau**4.))))+(-27.*(c1*(tau**8.)))))**2));
    aux3=(-2.*(c3**3.))+((9.*(c2*(c3*(tau**4.))))+((-27.*(c1*(\
tau**8.)))+(mysqrt(aux2))));
    aux4=(4.*(((3.*(c2*(tau**4.)))-(c3**2))**3.))+((((-2.*(c3**3.))\
+((9.*(c2*(c3*(tau**4.))))+(-27.*(c1*(tau**8.)))))**2));
    aux5=(-2.*(c3**3.))+((9.*(c2*(c3*(tau**4.))))+((-27.*(c1*(\
tau**8.)))+(mysqrt(aux4))));
    aux6=(4.*(((3.*(c2*(tau**4.)))-(c3**2))**3.))+((((-2.*(c3**3.))\
+((9.*(c2*(c3*(tau**4.))))+(-27.*(c1*(tau**8.)))))**2));
    aux7=(-2.*(c3**3.))+((9.*(c2*(c3*(tau**4.))))+((-27.*(c1*(\
tau**8.)))+(mysqrt(aux6))));
    aux8=(0.+1.j)*((2.**(-2/3))*((3.**-0.5)*((c3**2)*((tau**-4.)\
*(aux7**(-1/3))))));
    aux9=(4.*(((3.*(c2*(tau**4.)))-(c3**2))**3.))+((((-2.*(c3**3.))\
+((9.*(c2*(c3*(tau**4.))))+(-27.*(c1*(tau**8.)))))**2));
    aux10=(-2.*(c3**3.))+((9.*(c2*(c3*(tau**4.))))+((-27.*(c1*(\
tau**8.)))+(mysqrt(aux9))));
    aux11=(4.*(((3.*(c2*(tau**4.)))-(c3**2))**3.))+((((-2.*(c3**3.)\
)+((9.*(c2*(c3*(tau**4.))))+(-27.*(c1*(tau**8.)))))**2));
    aux12=(-2.*(c3**3.))+((9.*(c2*(c3*(tau**4.))))+((-27.*(c1*(\
tau**8.)))+(mysqrt(aux11))));
    aux13=(-(1/6)*((2.**(-1/3))*((tau**-4.)*(aux10**(1/3))\
)))+((0.-0.5j)*((2.**(-1/3))*((3.**-0.5)*((tau**-4.)*(aux12**(1/3))))));
    aux14=((-1/3)*((2.**(-2/3))*((c3**2)*((tau**-4.)*(aux5**(-1/3))))))+(aux8+aux13);
    aux15=((0.-1.j)*((2.**(-2/3))*((mysqrt(3.))*(c2*(aux3**(-1/3))))))+aux14;
    aux16=((-1/3)*(c3*(tau**-4.)))+(((2.**(-2/3))*(c2*(aux1**(-1/3))))+aux15);
    root3=(-mysqrt(aux16))

    ##

    aux0=(4.*(((3.*(c2*(tau**4.)))-(c3**2))**3.))+((((-2.*(c3**3.))+((\
    9.*(c2*(c3*(tau**4.))))+(-27.*(c1*(tau**8.)))))**2));
    aux1=(-2.*(c3**3.))+((9.*(c2*(c3*(tau**4.))))+((-27.*(c1*(\
tau**8.)))+(mysqrt(aux0))));
    aux2=(4.*(((3.*(c2*(tau**4.)))-(c3**2))**3.))+((((-2.*(c3**3.))\
+((9.*(c2*(c3*(tau**4.))))+(-27.*(c1*(tau**8.)))))**2));
    aux3=(-2.*(c3**3.))+((9.*(c2*(c3*(tau**4.))))+((-27.*(c1*(\
tau**8.)))+(mysqrt(aux2))));
    aux4=(4.*(((3.*(c2*(tau**4.)))-(c3**2))**3.))+((((-2.*(c3**3.))\
+((9.*(c2*(c3*(tau**4.))))+(-27.*(c1*(tau**8.)))))**2));
    
    aux5=(-2.*(c3**3.))+((9.*(c2*(c3*(tau**4.))))+((-27.*(c1*(\
tau**8.)))+(mysqrt(aux4))));
    aux6=(4.*(((3.*(c2*(tau**4.)))-(c3**2))**3.))+((((-2.*(c3**3.))\
+((9.*(c2*(c3*(tau**4.))))+(-27.*(c1*(tau**8.)))))**2));
    aux7=(-2.*(c3**3.))+((9.*(c2*(c3*(tau**4.))))+((-27.*(c1*(\
tau**8.)))+(mysqrt(aux6))));
    aux8=(0.-1.j)*((2.**(-2/3))*((3.**-0.5)*((c3**2)*((tau**-4.\
)*(aux7**(-1/3))))));
    aux9=(4.*(((3.*(c2*(tau**4.)))-(c3**2))**3.))+((((-2.*(c3**3.))\
+((9.*(c2*(c3*(tau**4.))))+(-27.*(c1*(tau**8.)))))**2));
    aux10=(-2.*(c3**3.))+((9.*(c2*(c3*(tau**4.))))+((-27.*(c1*(\
tau**8.)))+(mysqrt(aux9))));
    aux11=(4.*(((3.*(c2*(tau**4.)))-(c3**2))**3.))+((((-2.*(c3**3.)\
)+((9.*(c2*(c3*(tau**4.))))+(-27.*(c1*(tau**8.)))))**2));
    aux12=(-2.*(c3**3.))+((9.*(c2*(c3*(tau**4.))))+((-27.*(c1*(\
tau**8.)))+(mysqrt(aux11))));
    aux13=((-1/6)*((2.**(-1/3))*((tau**-4.)*(aux10**(1/3))\
)))+((0.+0.5j)*((2.**(-1/3))*((3.**-0.5)*((tau**-4.)*(aux12**(1/3))))));
    aux14=((-1/3)*((2.**(-2/3))*((c3**2)*((tau**-4.)*(aux5**-\
(1/3))))))+(aux8+aux13);
    aux15=((0.+1.j)*((2.**(-2/3))*((mysqrt(3.))*(c2*(aux3**(-1/3))))))+aux14;
    aux16=((-1/3)*(c3*(tau**-4.)))+(((2.**(-2/3))*(c2*(aux1**\
(-1/3))))+aux15);
    root5=(-mysqrt(aux16))

    return root1, root3, root5




####################################################################
#delta plus decaying oscilation a d(t) + b e**-t/tau cos(f t) MSD

def msd_osc_analytic(a, b, f, tau, B, t):
    a = np.array(a, dtype=np.float128)
    b = np.array(b, dtype=np.float128)#float(b)
    f = np.array(f, dtype=np.float128)#float(f)
    tau = np.array(tau, dtype=np.float128)#float(tau)
    B = np.array(B, dtype=np.float128)#float(B)
    k1 = -2* (1 + f**2 * tau**2) * (a + b*tau + a * f**2 * tau**2)
    k2 = -2*tau**2 * (b*tau + 2*a * (1 - f**2 * tau**2))
    k3 = -2 * a * tau**4
    c1 = (a + b*tau + a * f**2 * tau**2)**2
    c2 = 1 + 2*tau**2 * (a**2 - b + f**2) + 2*a*b*tau**3 + tau**4 * ((b + f**2)**2 - 2*a**2 * f**2)
    c3 = tau**2 * (2 + tau**2 * (a**2 - 2*(b + f**2)) )
    #print(c1,c2,c3,k1,k2,k3)
    r1, r3, r5 = roots_osc(c1, c2, c3, tau)
    #print(r1,r3,r5) #roots are same as in mathematica
    diff_r13 = r1**2 - r3**2
    diff_r15 = r1**2 - r5**2
    diff_r35 = r3**2 - r5**2
    
    I11 = (np.exp(-t*np.sqrt(-r1**2)) - 1) / (np.sqrt(-r1**2) * r1**2 * diff_r13 * diff_r15)
    I21 = I11 * r1**2 
    I31 = I21 * r1**2

    I12 = (np.exp(-np.sqrt(-r3**2)*t) - 1) / (-np.sqrt(-r3**2) * r3**2 * diff_r13 * diff_r35)
    I22 = I12 * r3**2 
    I32 = I22 * r3**2

    I13 = (np.exp(-np.sqrt(-r5**2)*t) - 1) / (np.sqrt(-r5**2) * r5**2 * diff_r15 * diff_r35)
    I23 = I13 * r5**2 
    I33 = I23 * r5**2

    I1 = k1 * (t / (r1**2 * r3**2 * r5**2) + I11 + I12 + I13)
    I2 = k2 * (I21 + I22 + I23)
    I3 = k3* (I31 + I32 + I33)
    #print(B/tau**4, np.real(I1 + I2 + I3))
    msd = B/tau**4 * np.real(I1 + I2 + I3)
    return msd


#####################################################################
