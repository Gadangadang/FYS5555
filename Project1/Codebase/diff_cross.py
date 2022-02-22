from plot_set import *
import numpy as np
import scipy as sp


class DiffCrossection:
    def __init__(self, theta_interval, energy_centermass, particle):

        particle_data = {
            "electron": [0.000511, 0.5*(-0.5), 0.5*(-0.04), -1],
            "bottom": [4.85, 0.5*(-0.5), 0.5*(-0.35), -1/3],
            "charm" : [1.275, 0.5*0.5, 0.5*0.19, 2/3], 
            "muon" : [0.1057, 0.5*(-0.5), 0.5*(-0.04), -1]
        }

        # General constants
        self.costheta = np.cos(theta_interval)
        self.m_mu = 0.1057  # GeV
        self.mZ = 91.1876  # GeV
        self.m_other = particle_data[particle][0]
        Q = (particle_data[particle][3])
        

        self.sinthetaW = 0.48076
        self.thetaw = np.arcsin(self.sinthetaW)
        self.e = 0.31345
        self.costhetaW = np.cos(self.thetaw)

        self.g = self.e / self.sinthetaW
        self.gz = self.g / np.cos(self.thetaw)

        # Z boson vertex constants
        self.g_Amu = particle_data["muon"][1]
        self.g_Vmu = particle_data["muon"][2]
        self.g_A_other = particle_data[particle][1]
        self.g_V_other = particle_data[particle][2]  # self.gz * 0.5 * (-0.5 - 2 * (-1 / 3) * self.sinthetaW**2)

        self.four_gs = self.g_Amu * self.g_Vmu * self.g_A_other * self.g_V_other

        self.A = self.g_Amu**2 + self.g_Vmu**2
        self.Atilde = self.g_Amu**2 - self.g_Vmu**2
        self.B = self.g_A_other**2 + self.g_V_other**2
        self.Btilde = self.g_A_other**2 - self.g_V_other**2

        # Energy and com split
        self.Ecm = energy_centermass  # GeV
        self.E2 = (self.Ecm / 2) ** 2
        self.E = self.Ecm / 2
        self.s = self.Ecm**2  # GeV

        # Amplitude constants
        self.m1const = 8 * self.e**4 * Q**2 / (self.s**2)
        self.m2const = 8 * self.gz**4 / (self.s - self.mZ**2) ** 2
        self.m1m2const = -(
            8
            * self.e**2
            * self.g**2
            * Q
            / (self.costhetaW**2 * self.s * (self.s - self.mZ**2))
        )

        comp_data = np.loadtxt("../datasets/comphep_200GeV.txt", skiprows=3)
        self.comphep_degree = comp_data[:, 0]
        self.comphep_diff = comp_data[:, 1]

    def compute_momentum_products(self):
        # Momentum dot products and def of momentum
        self.p = np.sqrt(self.E2 - self.m_mu**2)
        self.p_ = np.sqrt(self.E2 - self.m_other**2)
        self.pp_kk_ = (self.E**2 - self.p * self.p_ * self.costheta) ** 2
        self.pk_p_k = (self.E**2 + self.p * self.p_ * self.costheta) ** 2
        self.pk = self.E2 + self.p**2
        self.p_k_ = self.E2 + self.p_**2

        self.cross_const = (
            1 / (2.56810e-9) / (32 * np.pi * self.s) * self.p_ / self.p * 3
        )

    def M1squared(self):
        return (
            self.cross_const
            * self.m1const
            * (
                self.pp_kk_
                + self.pk_p_k
                + self.m_other**2 * self.pk
                + self.m_mu**2 * self.p_k_
                + 2 * self.m_other**2 * self.m_mu**2
            )
        )

    def M2squared(self):
        return (
            self.cross_const
            * self.m2const
            * (
                self.A * self.B * (self.pp_kk_ + self.pk_p_k)
                - self.Atilde * self.B * self.m_other**2 * self.pk
                - self.A * self.Btilde * self.m_mu**2 * self.p_k_
                + 2 * self.m_other**2 * self.m_mu**2 * self.Atilde * self.Btilde
                - 4 * self.four_gs * (self.pp_kk_ - self.pk_p_k)
            )
        )

    def M1M2(self):
        return (
            self.cross_const
            * self.m1m2const
            * (
                self.g_Vmu
                * self.g_V_other
                * (
                    self.pk_p_k
                    + self.pp_kk_
                    + self.m_other**2 * self.pk
                    + self.m_mu**2 * self.p_k_
                    + 2 * self.m_other**2 * self.m_mu**2
                )
                - self.g_Amu * self.g_A_other * (self.pp_kk_ - self.pk_p_k)
            )
        )

    def diff_cross(self):
        return self.M1squared() + self.M2squared() + 2 * self.M1M2()

    def plot_cross(self):
        plt.plot(
            self.costheta,
            self.diff_cross(),
            label=r"Analytical $d\sigma/d cos(\theta)$ for  $M_{tot}^2$",
        )
        plt.plot(
            self.comphep_degree,
            self.comphep_diff,
            "r--",
            label=r"CompHEP $d\sigma/d cos(\theta)$ for$M_{tot}^2$",
        )

        plt.xlabel(r"cos($\theta$) ")
        plt.ylabel(r"$\frac{d\sigma}{d \cos{(\theta)}}$")
        plt.legend()
        plt.title(
            r" $\frac{d\sigma}{d \cos{(\theta)}}$ for $\mu^+ \mu^- \to b \bar{b}$ with $\sqrt{s}$ = "
            + str(np.sqrt(self.s))
            + " GeV"
        )
        plt.savefig("../Figures/cross_sec_sqrts_200GeV.pdf")
        plt.show()

    def plot_m2(self):
        plt.plot(
            self.costheta,
            self.diff_cross(),
            label=r"$d\sigma/d cos(\theta)$ for $M_{tot}^2$",
        )
        plt.plot(
            self.costheta,
            self.cross_const * self.M1squared(),
            label=r"$d\sigma/d cos(\theta)$ for $M_{\gamma}^2$",
        )
        plt.plot(
            self.costheta,
            self.cross_const * self.M2squared(),
            label=r"$d\sigma/d cos(\theta)$ for $M_{Z}^2$",
        )
        plt.plot(
            self.costheta,
            self.cross_const * self.M1M2(),
            label=r"$d\sigma/d cos(\theta)$ for $M_{\gamma,Z}^2$",
        )

        plt.xlabel(r"cos($\theta$) ")
        plt.ylabel(r"$\frac{d\sigma}{d \cos{(\theta)}}$")
        plt.legend()
        plt.title(
            r" $\frac{d\sigma}{d \cos{(\theta)}}$ for $\mu^+ \mu^- \to b \bar{b}$ with $\sqrt{s}$ = "
            + str(np.sqrt(self.s))
            + " GeV"
        )
        plt.savefig("../Figures/m2_sqrts_200GeV.pdf")
        plt.show()


def intergrate_crossec(start, stop, energy_range, particle):

    particle_data = {
        "bottom": ["bottom_asymmetry.txt", r"$b\bar{b}$"],
        "charm" : ["charm_asymmetry.txt", r"$c\bar{c}$"],
        "muon" : ["muon_asymmetry.txt", r"$\mu^+\mu^-$"],
        "electron" : ["electron_asymmetry.txt", r"$e^+e^-$"]
    }

    N = 1000
    asymmetry = np.zeros_like(energy_range)
    asymmetry_gamma = np.zeros_like(energy_range)
    asymmetry_z = np.zeros_like(energy_range)
    new_costheta = np.linspace(start, stop, N)

    for index, energy_cm in enumerate(energy_range):

        # sigma for angle less than pi/2
        dcs1 = DiffCrossection(new_costheta, energy_cm, particle)
        dcs1.compute_momentum_products()
        diff_cross_sec1 = dcs1.diff_cross()
        diff_cross_gamma1 = dcs1.M1squared()
        diff_cross_z1 = dcs1.M2squared()

        # sigma for angle larger than pi/2
        dcs2 = DiffCrossection(new_costheta, energy_cm, particle)
        dcs2.compute_momentum_products()
        diff_cross_sec2 = dcs2.diff_cross()
        diff_cross_gamma2 = dcs2.M1squared()
        diff_cross_z2 = dcs2.M2squared()

        tot_cross1 = sp.integrate.simps(diff_cross_sec1[: int(N / 2)])
        tot_cross2 = sp.integrate.simps(diff_cross_sec2[int(N / 2) :])

        gamma_cross1 = sp.integrate.simps(diff_cross_gamma1[: int(N / 2)])
        gamma_cross2 = sp.integrate.simps(diff_cross_gamma2[int(N / 2) :])

        z_cross1 = sp.integrate.simps(diff_cross_z1[: int(N / 2)])
        z_cross2 = sp.integrate.simps(diff_cross_z2[int(N / 2) :])

        asymmetry[index] = (tot_cross1 - tot_cross2) / (tot_cross1 + tot_cross2)
        asymmetry_gamma[index] = (gamma_cross1 - gamma_cross2) / (
            gamma_cross1 + gamma_cross2
        )
        asymmetry_z[index] = (z_cross1 - z_cross2) / (z_cross1 + z_cross2)

    asym_comphep = np.loadtxt("../datasets/"+particle_data[particle][0], skiprows=3)

    plt.plot(energy_range, asymmetry, label="Total asymmetry")
    plt.plot(asym_comphep[:, 0], asym_comphep[:, 1], "r--", label="CompHEP asymmetry")
    plt.plot(energy_range, asymmetry_gamma, label=r"$\gamma$ asymmetry")
    plt.plot(energy_range, asymmetry_z, label="Z asymmetry")
    plt.legend()
    plt.xlabel(r"COM energy $\sqrt{s}$")
    plt.ylabel("Asymmetry")
    plt.title(r"Asymmetry for $\mu^+\mu^- \to$ " + particle_data[particle][1] + " as function of $\sqrt{s}$")
    plt.savefig("../Figures/asymmetry_comp_"+particle_data[particle][1]+".pdf")
    plt.show()


if __name__ == "__main__":
    names = ["bottom", "charm", "electron"]
    for name in names:
        intergrate_crossec(0, np.pi, np.linspace(10, 200, 191), name)
    