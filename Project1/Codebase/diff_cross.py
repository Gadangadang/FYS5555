import numpy as np
import matplotlib.pyplot as plt


class DiffCrossection:
    def __init__(self, theta_interval):

        self.costheta = np.cos(theta_interval)
        self.m_mu = 0.1057  # GeV
        self.mZ = 91.1876  # GeV
        self.m_b = 4.85  # GeV

        self.sinthetaW = 0.48076
        self.thetaw = np.arcsin(self.sinthetaW)
        self.e = 0.31345
        self.costhetaW = np.cos(self.thetaw)

        self.gz = self.e / (self.sinthetaW * np.cos(self.thetaw))
        self.g = self.e / self.sinthetaW

        self.g_Amu = self.gz * 0.5 * (-0.5)
        self.g_Vmu = self.gz * 0.5 * (-0.5 - 2 * (-1) * self.sinthetaW ** 2)
        self.g_Ab = self.gz * 0.5 * (-0.5)
        self.g_Vb = self.gz * 0.5 * (-0.5 - 2 * (-1 / 3) * self.sinthetaW ** 2)

        self.four_gs = self.g_Amu * self.g_Vmu * self.g_Ab * self.g_Vb

        self.A = self.g_Amu ** 2 + self.g_Vmu ** 2
        self.B = self.g_Ab ** 2 + self.g_Vb ** 2
        self.Ecm = 200  # GeV
        self.E2 = (self.Ecm / 2) ** 2
        self.E = self.Ecm / 2
        self.s = self.Ecm ** 2  # GeV

        self.m1const = self.e ** 4 / (9 * self.s ** 2)
        self.m2const = self.g ** 2 / (self.costhetaW**2 * (self.s - self.mZ ** 2))
        self.m1m2const = (
            self.e ** 2
            * self.g ** 2
            / (3 * self.costhetaW**2 * self.s * (self.s - self.mZ ** 2))
        )

        self.p = np.sqrt(self.E2 - self.m_mu ** 2)
        self.p_ = np.sqrt(self.E2 - self.m_b ** 2)
        self.pp_kk_ = (self.E ** 2 - self.p * self.p_ * self.costheta) ** 2
        self.pk_p_k = (self.E ** 2 + self.p * self.p_ * self.costheta) ** 2
        self.pk = self.E2 + self.p ** 2
        self.p_k_ = self.E2 + self.p_ ** 2

    def M1squared(self):
        return (
            self.m1const
            * 8
            * (
                self.pp_kk_
                + self.pk_p_k
                + self.m_b ** 2 * self.pk
                + self.m_mu ** 2 * self.p_k_
                + 2 * self.m_b * self.m_mu
            )
        )

    def M2squared(self):
        return (
            self.m2const
            * 8
            * (
                self.A * self.B * (self.pp_kk_ + self.pk_p_k)
                + self.B * self.m_b ** 2 * self.pk
                + self.A * self.m_mu ** 2 * self.p_k_
                + 2 * self.m_b ** 2 * self.m_mu ** 2 * self.A * self.B
                - 4 * self.four_gs * (self.pp_kk_ - self.pk_p_k)
            )
        )

    def M1M2(self):
        return (
            self.m1m2const
            * 8
            * (
                self.g_Ab * self.g_Vb * self.pk_p_k
                + self.g_Amu * self.g_Vmu * self.pp_kk_
                + self.g_Vb * self.m_b ** 2 * self.pk
                + self.g_Vmu * self.m_mu ** 2 * self.p_k_
                + 2 * self.m_b ** 2 * self.m_mu ** 2 * self.g_Vb * self.g_Vmu
                - self.g_Amu * self.g_Ab * (self.pp_kk_ - self.pk_p_k)
            )
        )

    def diff_cross(self):
        return (
            1
            / (32 * np.pi * self.s)
            * self.p_
            / self.p
            * 3
            * (self.M1squared() + self.M2squared() + 2 * self.M1M2())
        )

    def plot_cross(self):
        plt.plot(self.costheta, self.diff_cross(), label=r"d$\sigma$/cos($\theta$) ")
        plt.xlabel(r"cos($\theta$) ")
        plt.ylabel(r"$d\sigma/d cos(\theta)$")
        plt.legend()
        plt.savefig("../Figures/cross_sec_sqrts_200GeV.pdf")
        plt.show()


if __name__ == "__main__":

    theta_interval = np.linspace(-np.pi, np.pi, 1001)
    dcs = DiffCrossection(theta_interval)
    dcs.plot_cross()
