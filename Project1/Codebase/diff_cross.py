from plot_set import *
import numpy as np


class DiffCrossection:
    def __init__(self, theta_interval, energy_centermass):

        # General constants
        self.costheta = np.cos(theta_interval)
        self.m_mu = 0.1057  # GeV
        self.mZ = 91.1876  # GeV
        self.m_b = 4.85  # GeV

        self.sinthetaW = 0.48076
        self.thetaw = np.arcsin(self.sinthetaW)
        self.e = 0.31345
        self.costhetaW = np.cos(self.thetaw)

        self.g = self.e / self.sinthetaW
        self.gz = self.g / np.cos(self.thetaw)

        # Z boson vertex constants
        self.g_Amu = -0.5 * 0.5  # self.gz * 0.5 * (-0.5)
        self.g_Vmu = (
            -0.04 * 0.5
        )  # self.gz * 0.5 * (-0.5 - 2 * (-1) * self.sinthetaW**2)
        self.g_Ab = -0.5 * 0.5  # self.gz * 0.5 * (-0.5)
        self.g_Vb = (
            -0.35 * 0.5
        )  # self.gz * 0.5 * (-0.5 - 2 * (-1 / 3) * self.sinthetaW**2)

        self.four_gs = self.g_Amu * self.g_Vmu * self.g_Ab * self.g_Vb

        self.A = self.g_Amu**2 + self.g_Vmu**2
        self.Atilde = self.g_Amu**2 - self.g_Vmu**2
        self.B = self.g_Ab**2 + self.g_Vb**2
        self.Btilde = self.g_Ab**2 - self.g_Vb**2

        # Energy and com split
        self.Ecm = energy_centermass  # GeV
        self.E2 = (self.Ecm / 2) ** 2
        self.E = self.Ecm / 2
        self.s = self.Ecm**2  # GeV

        # Amplitude constants
        self.m1const = 8 * self.e**4 / (9 * self.s**2)
        self.m2const = 8 * self.gz**4 / (self.s - self.mZ**2) ** 2
        self.m1m2const = (
            8
            * self.e**2
            * self.g**2
            / (3 * self.costhetaW**2 * self.s * (self.s - self.mZ**2))
        )

        # Momentum dot products and def of momentum
        self.p = np.sqrt(self.E2 - self.m_mu**2)
        self.p_ = np.sqrt(self.E2 - self.m_b**2)
        self.pp_kk_ = (self.E**2 - self.p * self.p_ * self.costheta) ** 2
        self.pk_p_k = (self.E**2 + self.p * self.p_ * self.costheta) ** 2
        self.pk = self.E2 + self.p**2
        self.p_k_ = self.E2 + self.p_**2

        comp_data = np.loadtxt("comphep_200GeV.txt", skiprows=3)
        self.comphep_degree = comp_data[:, 0]
        self.comphep_diff = comp_data[:, 1]

        self.cross_const = (
            1 / (2.56810e-9) / (32 * np.pi * self.s) * self.p_ / self.p * 3
        )

    def M1squared(self):
        return self.m1const * (
            self.pp_kk_
            + self.pk_p_k
            + self.m_b**2 * self.pk
            + self.m_mu**2 * self.p_k_
            + 2 * self.m_b**2 * self.m_mu**2
        )

    def M2squared(self):
        return self.m2const * (
            self.A * self.B * (self.pp_kk_ + self.pk_p_k)
            - self.Atilde * self.B * self.m_b**2 * self.pk
            - self.A * self.Btilde * self.m_mu**2 * self.p_k_
            + 2 * self.m_b**2 * self.m_mu**2 * self.Atilde * self.Btilde
            - 4 * self.four_gs * (self.pp_kk_ - self.pk_p_k)
        )

    def M1M2(self):
        return self.m1m2const * (
            self.g_Vmu
            * self.g_Vb
            * (
                self.pk_p_k
                + self.pp_kk_
                + self.m_b**2 * self.pk
                + self.m_mu**2 * self.p_k_
                + 2 * self.m_b**2 * self.m_mu**2
            )
            - self.g_Amu * self.g_Ab * (self.pp_kk_ - self.pk_p_k)
        )

    def diff_cross(self):
        return self.cross_const * (
            self.M1squared() + self.M2squared() + 2 * self.M1M2()
        )

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


if __name__ == "__main__":

    theta_interval = np.linspace(0, np.pi, 1001)
    dcs = DiffCrossection(theta_interval, 200)
    dcs.plot_cross()
    dcs.plot_m2()
