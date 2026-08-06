# -*- coding: utf-8 -*-
"""
USGS adjustments to ground-motion models for Guam and the Northern Mariana Islands.


"""
import pathlib

import numpy as np

from openquake.baselib.general import CallableDict
from openquake.hazardlib.gsim.base import CoeffsTable
from openquake.hazardlib import const
from openquake.hazardlib.imt import PGA

from openquake.hazardlib.gsim import (
    abrahamson_gulerce_2020,
    abrahamson_gulerce_2022,
    kuehn_2020,
    parker_2020,
    )


_depth_scaling = CallableDict()


@_depth_scaling.add(const.TRT.SUBDUCTION_INTERFACE)
def _depth_scaling_1(trt, C, z):
    """
    Depth scaling is for slab.
    """
    return 0


@_depth_scaling.add(const.TRT.SUBDUCTION_INTRASLAB)
def _depth_scaling_2(trt, C, z):
    """Compute Guam-specific depth term for intraslab.
    
    Args:
        trt: Tectonic regime.
        C: Coefficients.
        z: Hypocenter depth (Parker et al.) or ZTOR (Abrahamson and Gulerce; Kuehn et al.)
    """
    # mid-depth
    res = C["d"] - C["m"] * (C["db2"] - z)

    # deep
    res[z >= C["db2"]] = C["d"]

    # shallow
    res[z < C["db1"]] = C["d"] - C["m"] * (C["db2"] - C["db1"])

    return res


def abrahamson_adjust_depth(C, trt, ctx):
    mean = _depth_scaling(trt, C, ctx.ztor)
    mean -= abrahamson_gulerce_2020.get_rupture_depth_scaling_term(C, trt, ctx)
    return mean


def kuehn_adjust_depth(C, trt, ctx):
    mean = _depth_scaling(trt, C, ctx.ztor)
    mean -= kuehn_2020.get_depth_term(C, trt, ctx.ztor)
    return mean


def _load_coeffs(name):
    fname = pathlib.Path(__file__).parent / name
    with open(fname, encoding="utf-8") as f:
        return CoeffsTable(sa_damping=5, table=f.read())


class AbrahamsonGulerce2022SInter_USGSGuamStage1(abrahamson_gulerce_2022.AbrahamsonGulerce2022SInter):
    """
    Abrahamson and Gulerce (2020) subduction interface ground-motion model with 
    USGS adjustments for Guam and the Northern Mariana Islands.
    """
    COEFFS = _load_coeffs("AbrahamsonGulerce2022_USGSGuamStage1_coeffs.csv")


class AbrahamsonGulerce2022SSlab_USGSGuamStage1(abrahamson_gulerce_2022.AbrahamsonGulerce2022SSlab):
    """
    Abrahamson and Gulerce (2020) subduction intraslab ground-motion model with 
    USGS adjustments for Guam and the Northern Mariana Islands.
    """
    COEFFS = _load_coeffs("AbrahamsonGulerce2022_USGSGuamStage1_coeffs.csv")

    def compute(self, ctx: np.recarray, imts, mean, sig, tau, phi):
        """
        See :meth:`superclass method
        <.base.GroundShakingIntensityModel.compute>`
        for spec of input and result values.
        """
        trt = self.DEFINED_FOR_TECTONIC_REGION_TYPE
        C_PGA = self.COEFFS[PGA()]
        pga1000 = abrahamson_gulerce_2020.get_acceleration_on_reference_rock(C_PGA, trt,
                                                     self.region, ctx,
                                                     self.apply_usa_adjustment)
        pga1000 += abrahamson_adjust_depth(C_PGA, trt, ctx)
        pga1000 = np.exp(pga1000)

        for m, imt in enumerate(imts):
            C = self.COEFFS[imt]
            mean[m] = abrahamson_gulerce_2020.get_mean_acceleration(C, trt, self.region, ctx, pga1000,
                                            self.apply_usa_adjustment)
            mean[m] += abrahamson_adjust_depth(C, trt, ctx)
            if self.sigma_mu_epsilon:
                # Apply an epistmic adjustment factor
                mean[m] += (self.sigma_mu_epsilon *
                            abrahamson_gulerce_2020.get_epistemic_adjustment(C, ctx.rrup))
            # Get the standard deviations
            tau_m, phi_m = abrahamson_gulerce_2020.get_tau_phi(C, C_PGA, self.region, imt.period,
                                       ctx.rrup, ctx.vs30, pga1000,
                                       self.ergodic)
            tau[m] = tau_m
            phi[m] = phi_m
        sig += np.sqrt(tau ** 2.0 + phi ** 2.0)


class AbrahamsonGulerce2022SInter_USGSGuamStage2(AbrahamsonGulerce2022SInter_USGSGuamStage1):
    """
    Abrahamson and Gulerce (2020) subduction interface ground-motion model with 
    USGS adjustments for Guam and the Northern Mariana Islands.
    """
    COEFFS = _load_coeffs("AbrahamsonGulerce2022_USGSGuamStage2_coeffs.csv")


class AbrahamsonGulerce2022SSlab_USGSGuamStage2(AbrahamsonGulerce2022SSlab_USGSGuamStage1):
    """
    Abrahamson and Gulerce (2020) subduction intraslab ground-motion model with 
    USGS adjustments for Guam and the Northern Mariana Islands.
    """
    COEFFS = _load_coeffs("AbrahamsonGulerce2022_USGSGuamStage2_coeffs.csv")


class KuehnEtAl2020SInter_USGSGuamStage1(kuehn_2020.KuehnEtAl2020SInter):
    """
    Kuehn et al. (2020) subduction interface ground-motion model with 
    USGS adjustments for Guam and the Northern Mariana Islands.
    """
    COEFFS = _load_coeffs("KuehnEtAl2020_USGSGuamStage1_coeffs.csv")


class KuehnEtAl2020SSlab_USGSGuamStage1(kuehn_2020.KuehnEtAl2020SSlab):
    """
    Kuehn et al. (2020) subduction intraslab ground-motion model with 
    USGS adjustments for Guam and the Northern Mariana Islands.
    """
    COEFFS = _load_coeffs("KuehnEtAl2020_USGSGuamStage1_coeffs.csv")

    def compute(self, ctx: np.recarray, imts, mean, sig, tau, phi):
        """
        See :meth:`superclass method
        <.base.GroundShakingIntensityModel.compute>`
        for spec of input and result values.
        """
        trt = self.DEFINED_FOR_TECTONIC_REGION_TYPE
        if self.m_b:
            # Take the user define magnitude scaling breakpoint
            m_b = self.m_b
        else:
            # Take the global m_b for the tectonic region type and region
            m_b = kuehn_2020.REGION_TERMS_IF[self.region]["mb"] \
                if trt == const.TRT.SUBDUCTION_INTERFACE else \
                kuehn_2020.REGION_TERMS_SLAB[self.region]["mb"]
        C_PGA = self.COEFFS[PGA()]

        # Get PGA on rock
        pga1100 = np.exp(kuehn_2020.get_mean_values(
            C_PGA, self.region, trt, m_b, ctx, None))
        # For PGA and SA ( T <= 0.1 ) we need to define PGA on soil to
        # ensure that SA ( T ) does not fall below PGA on soil
        pga_soil = None
        for imt in imts:
            if ("PGA" in imt.string) or (("SA" in imt.string) and
                                         (imt.period <= 0.1)):
                pga_soil = kuehn_2020.get_mean_values(C_PGA, self.region, trt, m_b,
                                           ctx, pga1100)
                pga_soil += kuehn_adjust_depth(C_PGA, trt, ctx)
                break

        for m, imt in enumerate(imts):
            # Get coefficients for imt
            C = self.COEFFS[imt]
            m_break = m_b + C["dm_b"] if (
                trt == const.TRT.SUBDUCTION_INTERFACE and
                self.region in ("JPN", "SAM")) else m_b
            if imt.string == "PGA":
                mean[m] = pga_soil
            elif "SA" in imt.string and imt.period <= 0.1:
                # If Sa (T) < PGA for T <= 0.1 then set mean Sa(T) to mean PGA
                mean[m] = kuehn_2020.get_mean_values(C, self.region, trt, m_break,
                                          ctx, pga1100)
                mean[m] += kuehn_adjust_depth(C, trt, ctx)
                idx = mean[m] < pga_soil
                mean[m][idx] = pga_soil[idx]
            else:
                # For PGV and Sa (T > 0.1 s)
                mean[m] = kuehn_2020.get_mean_values(C, self.region, trt, m_break,
                                          ctx, pga1100)
                mean[m] += kuehn_adjust_depth(C, trt, ctx)
            # Apply the sigma mu adjustment if necessary
            if self.sigma_mu_epsilon:
                sigma_mu_adjust = kuehn_2020.get_sigma_mu_adjustment(
                    self.sigma_mu_model, imt, ctx.mag, ctx.rrup)
                mean[m] += self.sigma_mu_epsilon * sigma_mu_adjust
            # Get standard deviations
            tau[m] = C["tau"]
            phi[m] = C["phi"]
            sig[m] = np.sqrt(C["tau"] ** 2.0 + C["phi"] ** 2.0)


class KuehnEtAl2020SInter_USGSGuamStage2(KuehnEtAl2020SInter_USGSGuamStage1):
    """
    Kuehn et al. (2020) subduction interface ground-motion model with 
    USGS adjustments for Guam and the Northern Mariana Islands.
    """
    COEFFS = _load_coeffs("KuehnEtAl2020_USGSGuamStage2_coeffs.csv")


class KuehnEtAl2020SSlab_USGSGuamStage2(KuehnEtAl2020SSlab_USGSGuamStage1):
    """
    Kuehn et al. (2020) subduction intraslab ground-motion model with 
    USGS adjustments for Guam and the Northern Mariana Islands.
    """
    COEFFS = _load_coeffs("KuehnEtAl2020_USGSGuamStage2_coeffs.csv")


class ParkerEtAl2020SInter_USGSGuamStage1(parker_2020.ParkerEtAl2020SInter):
    """
    Parker et al. (2020) subduction interface ground-motion model with 
    USGS adjustments for Guam and the Northern Mariana Islands.
    """
    COEFFS = _load_coeffs("ParkerEtAl2020_USGSGuamStage1_coeffs.csv")


class ParkerEtAl2020SSlab_USGSGuamStage1(parker_2020.ParkerEtAl2020SSlab):
    """
    Parker et al. (2020) subduction intraslab ground-motion model with 
    USGS adjustments for Guam and the Northern Mariana Islands.
    """
    COEFFS = _load_coeffs("ParkerEtAl2020_USGSGuamStage1_coeffs.csv")

    def compute(self, ctx: np.recarray, imts, mean, sig, tau, phi):
        """
        See :meth:`superclass method
        <.base.GroundShakingIntensityModel.compute>`
        for spec of input and result values.
        """
        trt = self.DEFINED_FOR_TECTONIC_REGION_TYPE
        C_PGA = self.COEFFS[PGA()]
        for m, imt in enumerate(imts):
            C = self.COEFFS[imt]

            # Regional Mb factor
            if self.saturation_region in self.MB_REGIONS:
                m_b = self.MB_REGIONS[self.saturation_region]
            else:
                m_b = self.MB_REGIONS["default"]
            c0, c0_pga = parker_2020._c0(
                trt, self.region, self.saturation_region, C, C_PGA)
            fm, fm_pga = parker_2020._magnitude_scaling(
                self.SUFFIX, C, C_PGA, ctx.mag, m_b)
            fp, fp_pga = parker_2020._path_term(
                trt, self.region, self.basin, self.SUFFIX,
                C, C_PGA, ctx.mag, ctx.rrup, m_b)
            fd = _depth_scaling(trt, C, ctx.hypo_depth)
            fd_pga = _depth_scaling(trt, C_PGA, ctx.hypo_depth)
            fb = parker_2020._basin_term(self.region, self.basin, C, ctx)
            flin = parker_2020._linear_amplification(self.region, C, ctx.vs30)
            fnl = parker_2020._non_linear_term(C, imt, ctx.vs30, fp_pga, fm_pga, c0_pga,
                                   fd_pga)

            # The output is the desired median model prediction in LN units
            # Take the exponential to get PGA, PSA in g or the PGV in cm/s
            mean[m] = fp + fnl + fb + flin + fm + c0 + fd

            sig[m], tau[m], phi[m] = parker_2020.get_stddevs(C, ctx.rrup, ctx.vs30)


class ParkerEtAl2020SInter_USGSGuamStage2(ParkerEtAl2020SInter_USGSGuamStage1):
    """
    Parker et al. (2020) subduction interface ground-motion model with 
    USGS adjustments for Guam and the Northern Mariana Islands.
    """
    COEFFS = _load_coeffs("ParkerEtAl2020_USGSGuamStage2_coeffs.csv")


class ParkerEtAl2020SSlab_USGSGuamStage2(ParkerEtAl2020SSlab_USGSGuamStage1):
    """
    Parker et al. (2020) subduction intraslab ground-motion model with 
    USGS adjustments for Guam and the Northern Mariana Islands.
    """
    COEFFS = _load_coeffs("ParkerEtAl2020_USGSGuamStage2_coeffs.csv")
