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
        self.g_Amu = self.gz * 0.5 * (-0.5)
        self.g_Vmu = self.gz * 0.5 * (-0.5 - 2 * (-1) * self.sinthetaW**2)
        self.g_Ab = self.gz * 0.5 * (-0.5)
        self.g_Vb = self.gz * 0.5 * (-0.5 - 2 * (-1 / 3) * self.sinthetaW**2)

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
            self.g_Ab * self.g_Vb * self.pk_p_k
            + self.g_Amu * self.g_Vmu * self.pp_kk_
            + self.g_Vb * self.m_b**2 * self.pk
            + self.g_Vmu * self.m_mu**2 * self.p_k_
            + 2 * self.m_b**2 * self.m_mu**2 * self.g_Vb * self.g_Vmu
            - self.g_Amu * self.g_Ab * (self.pp_kk_ - self.pk_p_k)
        )

    def diff_cross(self):
        return (
            1
            / (2.56810e-9)
            / (32 * np.pi * self.s)
            * self.p_
            / self.p
            * 3
            * (self.M1squared() + self.M2squared() + 2 * self.M1M2())
        )

    def plot_cross(self):
        plt.plot(self.costheta, self.diff_cross(), label="Analytical")
        plt.plot(self.comphep_degree, self.comphep_diff, "r--", label="CompHEP")
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


if __name__ == "__main__":

    theta_interval = np.linspace(-np.pi, np.pi, 1001)
    dcs = DiffCrossection(theta_interval, 200)
    dcs.plot_cross()
