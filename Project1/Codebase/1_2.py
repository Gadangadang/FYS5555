import numpy as np
import matplotlib.pyplot as plt
import plot_set


N = 100

#Masses of particles in Gev
m_m = 0.1057 
m_b = 4.85
m_z = 91.1876 

#Charge
e = 0.31345


#Energy in Gev
E_cm = 10
E = E_cm/2
s = E_cm**2

#Constants
sin2_tw = 0.231
sin_tw = np.sqrt(sin2_tw)
tw = np.arcsin(sin_tw)
g_z = e/(sin_tw*np.cos(tw))
WidthZ = 2.43631


c_g = 8*e**4/(9*s**2)
c_z = 8*g_z**4*1/(s-m_z**2)**2#np.real( 1/( (s-m_z**2-1j*m_z*WidthZ)*(s-m_z**2+1j*m_z*WidthZ) ) )
c_gz =  8*e**2*g_z**2 * 1/((s-m_z**2)*3*s)#np.real(1/(3*s*(s-m_z**2+1j*m_z*WidthZ)))


#Axial coupling constants
g_ap = -0.04*0.5
g_bp = -0.5*0.5

g_a = -0.35*0.5
g_b = -0.5*0.5



g_tot = g_ap*g_bp*g_a*g_b 

Omega =  (g_ap**2 + g_bp**2)
Omega_t =  (g_ap**2 - g_bp**2)

Omega_p =  (g_a**2 + g_b**2)
Omega_tp =  (g_a**2 - g_b**2)


#Momentum
p = np.sqrt(E**2-m_m**2)
pp = np.sqrt(E**2-m_b**2)


cos_theta = np.cos(np.linspace(0,np.pi,N))


def Xi(cos_theta):
    return 2*(E**4 + (p*pp)**2 * cos_theta**2) + m_m**2*(pp**2+E**2) + m_b**2*(E**2+ p**2) + 2*m_m**2*m_b**2

def M_g(cos_theta):
    return c_g*Xi(cos_theta) 

def M_z(cos_theta):
    return c_z*(2*Omega*Omega_p*(E**4 + (p*pp)**2*cos_theta**2) + Omega_t*Omega_p*m_m**2*(E**2 + pp**2) \
            + Omega*Omega_tp*m_b**2*(E**2 + p**2) + 2* Omega_t*Omega_p*m_m**2*m_b**2 + 16*g_tot*E**2*p*pp*cos_theta) 

def M_gz(cos_theta):
    return 2*c_gz*(g_ap*g_a*Xi(cos_theta) + g_bp*g_b*4*p*pp*cos_theta*E**2)


def calcCrossSection(*args):
    crossSection = np.zeros(N)
    for arg in args:
        crossSection += arg(cos_theta)
    return 3/(32*np.pi*s)*1/(2.56810e-9) * pp/p * crossSection

plt.figure(num=0, dpi=80, facecolor='w', edgecolor='k')

# plt.savefig("../article/figures/figure.pdf", bbox_inches="tight")

plt.plot(cos_theta, calcCrossSection(M_g,M_z,M_gz), label = r"$M_{tot}$")
plt.plot(cos_theta, calcCrossSection(M_g), label = r"$M_{\gamma}$")
#plt.plot(cos_theta, calcCrossSection(M_z), label = r"$M_{z}$")
#plt.plot(cos_theta, calcCrossSection(M_gz), label = r"$2M_{\gamma,z}$")
plt.legend(fontsize = 13)
plt.xlabel(r"$\cos(\theta)$", fontsize=14)
plt.ylabel(r"$Cross section$: $\sigma$  [fb/rad]", fontsize=14)
plt.title(f"Center-of-mass energy= {E_cm} for " + r"$\mu^-,\mu^+ \rightarrow b,\bar{b}$", fontsize = 14)
plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
#plt.savefig("10_gamma.pdf")
plt.show()