# -*- coding: utf-8 -*-
"""
USGS adjustments to ground-motion models for Puerto Rico and the Virgin Islands.

The adjustments correspond to the fixed effect computed using linear mixed effects with
grouping for the event and site terms.
"""
import pathlib
import abc

import numpy

from openquake.hazardlib.gsim.base import CoeffsTable, add_alias
from openquake.hazardlib.imt import PGA
from openquake.hazardlib import const

from openquake.hazardlib.gsim import (
    abrahamson_2014,
    boore_2014,
    campbell_bozorgnia_2014,
    chiou_youngs_2014,
    abrahamson_gulerce_2020,
    kuehn_2020,
    parker_2020,
    )


class AbrahamsonEtAl2014_NoSiteResp(abrahamson_2014.AbrahamsonEtAl2014):
    """
    Implements the active crustal Abrahamson et al. (2014) GMM without site response.
    """
    def __init__(self, sigma_mu_epsilon=0.0, **kwargs):
        super().__init__(sigma_mu_epsilon=sigma_mu_epsilon, **kwargs)
        self.REQIURES_SITES_PARAMETERS = {}

    def compute(self, ctx: numpy.recarray, imts, mean, sig, tau, phi):
        """
        """
        for m, imt in enumerate(imts):
            C = self.COEFFS[imt]

            # get the mean value
            mean[m] = abrahamson_2014.get_mean_nosite(C, imt, ctx)
            mean[m] += (self.sigma_mu_epsilon*abrahamson_2014.get_epistemic_sigma(ctx))

            # compute median sa on rock (vs30=1180m/s). Used for site response
            # term calculation
            sa1180 = numpy.exp(abrahamson_2014._get_sa_at_1180(self.region, C, imt, ctx))

            # get standard deviations
            ctx = ctx.copy()
            ctx.vs30 = 1180
            sig[m], tau[m], phi[m] = abrahamson_2014._get_stddevs(
                self.region, C, imt, ctx, sa1180)


class BooreEtAl2014_NoSiteResp(boore_2014.BooreEtAl2014):
    """
    Implements the active crustal Boore et al. (2014) GMM without site response.
    """
    def __init__(self, region='nobasin', sof=True, sigma_mu_epsilon=0.0, **kwargs):
        super().__init__(region=region, sof=sof, sigma_mu_epsilon=sigma_mu_epsilon, **kwargs)
        self.REQIURES_SITES_PARAMETERS = {}

    def compute(self, ctx: numpy.recarray, imts, mean, sig, tau, phi):
        """
        """
        for m, imt in enumerate(imts):
            C = self.COEFFS[imt]
            mean[m] = (
                boore_2014._get_magnitude_scaling_term(self.sof, C, ctx) +
                boore_2014._get_path_scaling(self.kind, self.region, C, ctx)
                )
            if self.sigma_mu_epsilon:
                mean[m] += (self.sigma_mu_epsilon*boore_2014.get_epistemic_sigma(ctx))
            sig[m], tau[m], phi[m] = boore_2014._get_stddevs(self.kind, C, ctx)


class CampbellBozorgnia2014_NoSiteResp(campbell_bozorgnia_2014.CampbellBozorgnia2014):
    """
    Implements the active crustal Campbell and Bozorgnia (2014) GMM without site response.
    """
    def __init__(self, sigma_mu_epsilon=0.0, **kwargs):
        super().__init__(rsigma_mu_epsilon=sigma_mu_epsilon, **kwargs)
        self.REQIURES_SITES_PARAMETERS = {}

    def compute(self, ctx: numpy.recarray, imts, mean, sig, tau, phi):
        """
        """
        if (self.estimate_ztor or self.estimate_width or
                self.estimate_hypo_depth):
            ctx = ctx.copy()
            campbell_bozorgnia_2014._update_ctx(self, ctx)

        C_PGA = self.COEFFS[PGA()]
        for m, imt in enumerate(imts):
            C = self.COEFFS[imt]

            mean[m] = campbell_bozorgnia_2014.get_mean_nosite(C, ctx)
            mean[m] += (self.sigma_mu_epsilon*campbell_bozorgnia_2014.get_epistemic_sigma(ctx))
            if imt.string[:2] == "SA" and imt.period < 0.25:
                # According to Campbell & Bozorgnia (2013) [NGA West 2 Report]
                # If Sa (T) < PGA for T < 0.25 then set mean Sa(T) to mean PGA
                # Get PGA on soil
                pga = campbell_bozorgnia_2014.get_mean_nosite(C_PGA, ctx)
                idx = mean[m] <= pga
                mean[m, idx] = pga[idx]
                mean[m] += (self.sigma_mu_epsilon*campbell_bozorgnia_2014.get_epistemic_sigma(ctx))

            # Get stddevs for PGA on basement rock
            tau_lnpga_b = campbell_bozorgnia_2014._get_taulny(C_PGA, ctx.mag)
            phi_lnpga_b = numpy.sqrt(campbell_bozorgnia_2014._get_philny(C_PGA, ctx.mag) ** 2. - campbell_bozorgnia_2014.CONSTS["philnAF"] ** 2.)

            # Get tau_lny on the basement rock
            tau_lnyb = campbell_bozorgnia_2014._get_taulny(C, ctx.mag)
            # Get phi_lny on the basement rock
            phi_lnyb = numpy.sqrt(campbell_bozorgnia_2014._get_philny(C, ctx.mag) ** 2. - campbell_bozorgnia_2014.CONSTS["philnAF"] ** 2.)
            # Get site scaling term
            alpha = 0.0
            # Evaluate tau according to equation 29
            t = numpy.sqrt(tau_lnyb**2 + alpha**2 * tau_lnpga_b**2 + 2.0 * alpha * C["rholny"] * tau_lnyb * tau_lnpga_b)

            # Evaluate phi according to equation 30
            p = numpy.sqrt(
                phi_lnyb**2 + campbell_bozorgnia_2014.CONSTS["philnAF"]**2 + alpha**2 * phi_lnpga_b**2
                + 2.0 * alpha * C["rholny"] * phi_lnyb * phi_lnpga_b)
            sig[m] = numpy.sqrt(t**2 + p**2)
            tau[m] = t
            phi[m] = p


class ChiouYoungs2014_NoSiteResp(chiou_youngs_2014.ChiouYoungs2014):
    """
    Implements active crustal GMM developed by Brian S.-J. Chiou and Robert R. Youngs without site response.
    """

    def __init__(self, sigma_mu_epsilon=0.0, **kwargs):
        super().__init__(sigma_mu_epsilon=sigma_mu_epsilon, **kwargs)
        self.REQIURES_SITES_PARAMETERS = {"vs30", "vs30measured"}
    
    def compute(self, ctx: numpy.recarray, imts, mean, sig, tau, phi):
        """
        See :meth:`superclass method
        <.base.GroundShakingIntensityModel.compute>`
        for spec of input and result values.
        """
        name = self.__class__.__name__

        # reference to page 1144, PSA might need PGA value
        pga_mean, pga_sig, pga_tau, pga_phi = chiou_youngs_2014.get_mean_stddevs_nosite(name, self.COEFFS[PGA()], ctx)
        for m, imt in enumerate(imts):
            if repr(imt) == "PGA":
                mean[m] = pga_mean
                mean[m] += (self.sigma_mu_epsilon*chiou_youngs_2014.get_epistemic_sigma(ctx))
                sig[m], tau[m], phi[m] = pga_sig, pga_tau, pga_phi
            else:
                imt_mean, imt_sig, imt_tau, imt_phi = \
                    chiou_youngs_2014.get_mean_stddevs_nosite(name, self.COEFFS[imt], ctx)
                # reference to page 1144
                # Predicted PSA value at T ≤ 0.3s should be set equal to the value of PGA
                # when it falls below the predicted PGA
                mean[m] = numpy.where(imt_mean < pga_mean, pga_mean, imt_mean) \
                    if repr(imt).startswith("SA") and imt.period <= 0.3 \
                    else imt_mean

                mean[m] += (self.sigma_mu_epsilon*chiou_youngs_2014.get_epistemic_sigma(ctx))

                sig[m], tau[m], phi[m] = imt_sig, imt_tau, imt_phi


    
class AbrahamsonGulerce2020SInter_NoSiteResp(abrahamson_gulerce_2020.AbrahamsonGulerce2020SInter):
    """
    Implements the 2020 Subduction ground motion model of Abrahamson &
    Gulerce (2020) GMM without site response.
    """
    def __init__(self, region="GLO", ergodic=True, apply_usa_adjustment=False,
                 sigma_mu_epsilon=0.0, **kwargs):
        super().__init__(region=region, ergodic=ergodic, apply_usa_adjustment=apply_usa_adjustment, sigma_mu_epsilon=sigma_mu_epsilon, **kwargs)
        self.REQIURES_SITES_PARAMETERS = {}

    def compute(self, ctx: numpy.recarray, imts, mean, sig, tau, phi):
        """
        See :meth:`superclass method
        <.base.GroundShakingIntensityModel.compute>`
        for spec of input and result values.
        """
        trt = self.DEFINED_FOR_TECTONIC_REGION_TYPE
        C_PGA = self.COEFFS[PGA()]
        pga1000 = numpy.exp(abrahamson_gulerce_2020.get_mean_acceleration_nosite(C_PGA, trt, self.region, ctx, self.apply_usa_adjustment))

        for m, imt in enumerate(imts):
            C = self.COEFFS[imt]
            mean[m] = abrahamson_gulerce_2020.get_mean_acceleration_nosite(C, trt, self.region, ctx, self.apply_usa_adjustment)
            if self.sigma_mu_epsilon:
                # Apply an epistmic adjustment factor
                mean[m] += (self.sigma_mu_epsilon *
                            get_epistemic_adjustment(C, ctx.rrup))
            # Get the standard deviations
            tau_m, phi_m = abrahamson_gulerce_2020.get_tau_phi(C, C_PGA, self.region, imt.period,
                                       ctx.rrup, ctx.vs30, pga1000,
                                       self.ergodic)
            tau[m] = tau_m
            phi[m] = phi_m
        sig += numpy.sqrt(tau ** 2.0 + phi ** 2.0)


class AbrahamsonGulerce2020SSlab_NoSiteResp(AbrahamsonGulerce2020SInter_NoSiteResp):
    """
    Implements the 2020 Subduction ground motion model of Abrahamson &
    Gulerce (2020) GMM without site response.
    """
    #: Required rupture parameters are magnitude and top-of-rupture depth
    REQUIRES_RUPTURE_PARAMETERS = {'mag', 'ztor'}

    #: Supported tectonic region type is subduction inslab
    DEFINED_FOR_TECTONIC_REGION_TYPE = const.TRT.SUBDUCTION_INTRASLAB


for region in abrahamson_gulerce_2020.SUPPORTED_REGIONS[1:]:
    add_alias("AbrahamsonGulerce2020SInter" + abrahamson_gulerce_2020.REGION_ALIASES[region] + "_NoSiteResp",
              AbrahamsonGulerce2020SInter_NoSiteResp,
              region=region)
    add_alias("AbrahamsonGulerce2020SSlab" + abrahamson_gulerce_2020.REGION_ALIASES[region] + "_NoSiteResp",
              AbrahamsonGulerce2020SSlab_NoSiteResp,
              region=region)


class KuehnEtAl2020SInter_NoSiteResp(kuehn_2020.KuehnEtAl2020SInter):
    """
    Implements the 2020 Subduction ground motion model of Kuehn et al. (2020) GMM without site response.
    """
    def __init__(self, region="GLO", m_b=None, sigma_mu_epsilon=0.0, **kwargs):
        super().__init__(region=region, m_b=m_b, sigma_mu_epsilon=sigma_mu_epsilon, **kwargs)
        self.REQIURES_SITES_PARAMETERS = {}

    def compute(self, ctx: numpy.recarray, imts, mean, sig, tau, phi):
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

        # For PGA and SA ( T <= 0.1 ) we need to define PGA on soil to
        # ensure that SA ( T ) does not fall below PGA on soil
        pga = None
        for imt in imts:
            if ("PGA" in imt.string) or (("SA" in imt.string) and (imt.period <= 0.1)):
                pga = kuehn_2020.get_mean_values_nosite(C_PGA, self.region, trt, m_b, ctx)
                break

        for m, imt in enumerate(imts):
            # Get coefficients for imt
            C = self.COEFFS[imt]
            m_break = m_b + C["dm_b"] if (
                trt == const.TRT.SUBDUCTION_INTERFACE and
                self.region in ("JPN", "SAM")) else m_b
            if imt.string == "PGA":
                mean[m] = pga
            elif "SA" in imt.string and imt.period <= 0.1:
                # If Sa (T) < PGA for T <= 0.1 then set mean Sa(T) to mean PGA
                mean[m] = kuehn_2020.get_mean_values_nosite(C, self.region, trt, m_break, ctx)
                idx = mean[m] < pga
                mean[m][idx] = pga[idx]
            else:
                # For PGV and Sa (T > 0.1 s)
                mean[m] = kuehn_2020.get_mean_values_nosite(C, self.region, trt, m_break, ctx)
            # Apply the sigma mu adjustment if necessary
            if self.sigma_mu_epsilon:
                sigma_mu_adjust = kuehn_2020.get_sigma_mu_adjustment(
                    self.sigma_mu_model, imt, ctx.mag, ctx.rrup)
                mean[m] += self.sigma_mu_epsilon * sigma_mu_adjust
            # Get standard deviations
            tau[m] = C["tau"]
            phi[m] = C["phi"]
            sig[m] = numpy.sqrt(C["tau"] ** 2.0 + C["phi"] ** 2.0)



class KuehnEtAl2020SSlab_NoSiteResp(KuehnEtAl2020SInter_NoSiteResp):
    """
    Implements the 2020 Subduction ground motion model of Kuehn et al. (2020) GMM without site response.
    """
    #: Supported tectonic region type is subduction inslab
    DEFINED_FOR_TECTONIC_REGION_TYPE = const.TRT.SUBDUCTION_INTRASLAB


for region in kuehn_2020.SUPPORTED_REGIONS[1:]:
    add_alias("KuehnEtAl2020SInter" + kuehn_2020.REGION_ALIASES[region] + "_NoSiteResp",
              KuehnEtAl2020SInter_NoSiteResp,
              region=region)
    add_alias("KuehnEtAl2020SSlab" + kuehn_2020.REGION_ALIASES[region] + "_NoSiteResp",
              KuehnEtAl2020SSlab_NoSiteResp,
              region=region)


class ParkerEtAl2020SInter_NoSiteResp(parker_2020.ParkerEtAl2020SInter):
    """
    Implements the 2020 Subduction ground motion model of Parker et al. (2020) GMM without site response.
    """
    def __init__(self, region=None, saturation_region=None, basin=None, **kwargs):
        super().__init__(region=region, saturation_region=saturation_region, basin=basin, **kwargs)
        self.REQIURES_SITES_PARAMETERS = {"vs30", "vs30measured"}

    def compute(self, ctx: numpy.recarray, imts, mean, sig, tau, phi):
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
            fd = parker_2020._depth_scaling(trt, C, ctx)

            # The output is the desired median model prediction in LN units
            # Take the exponential to get PGA, PSA in g or the PGV in cm/s
            mean[m] = fp + fm + c0 + fd

            sig[m], tau[m], phi[m] = parker_2020.get_stddevs(C, ctx.rrup, ctx.vs30)


class ParkerEtAl2020SSlab_NoSiteResp(ParkerEtAl2020SInter_NoSiteResp):
    """
    Modifications for subduction slab.
    """
    DEFINED_FOR_TECTONIC_REGION_TYPE = const.TRT.SUBDUCTION_INTRASLAB

    # slab also requires hypo_depth
    REQUIRES_RUPTURE_PARAMETERS = {'mag', 'hypo_depth'}

    # constant table suffix
    SUFFIX = "slab"

    MB_REGIONS = {"Aleutian": 7.98, "AK": 7.2, "Cascadia": 7.2,
                  "CAM_S": 7.6, "CAM_N": 7.4, "JP_Pac": 7.65, "JP_Phi": 7.55,
                  "SA_N": 7.3, "SA_S": 7.25, "TW_W": 7.7, "TW_E": 7.7,
                  "default": 7.6}


add_alias('ParkerEtAl2020SSlabCAMN_NoSiteResp', ParkerEtAl2020SSlab_NoSiteResp,
          region="CAM", saturation_region="CAM_N")

        
class AbrahamsonEtAl2014_USGSPRVI(abrahamson_2014.AbrahamsonEtAl2014):
    """
    Abrahamson et al. (2014) active crustal ground-motion model with 
    USGS adjustments for Puerto Rico and the Virgin Islands.
    """
    filename = pathlib.Path(__file__).parent / "AbrahamsonEtAl2014_USGSPRVI_coeffs.csv"
    with open(filename) as f:
        COEFFS = CoeffsTable(sa_damping=5, table=f.read())


class BooreEtAl2014_USGSPRVI(boore_2014.BooreEtAl2014):
    """
    Boore et al. (2014) active crustal ground-motion model with 
    USGS adjustments for Puerto Rico and the Virgin Islands.
    """
    filename = pathlib.Path(__file__).parent / "BooreEtAl2014_USGSPRVI_coeffs.csv"
    with open(filename) as f:
        COEFFS = CoeffsTable(sa_damping=5, table=f.read())


class CampbellBozorgnia2014_USGSPRVI(campbell_bozorgnia_2014.CampbellBozorgnia2014):
    """
    Campbell and Bozorgnia (2014) active crustal ground-motion model with 
    USGS adjustments for Puerto Rico and the Virgin Islands.
    """
    filename = pathlib.Path(__file__).parent / "CampbellBozorgnia2014_USGSPRVI_coeffs.csv"
    with open(filename) as f:
        COEFFS = CoeffsTable(sa_damping=5, table=f.read())


class ChiouYoungs2014_USGSPRVI(chiou_youngs_2014.ChiouYoungs2014):
    """
    Chiou and Youngs (2014) active crustal ground-motion model with 
    USGS adjustments for Puerto Rico and the Virgin Islands.
    """
    filename = pathlib.Path(__file__).parent / "ChiouYoungs2014_USGSPRVI_coeffs.csv"
    with open(filename) as f:
        COEFFS = CoeffsTable(sa_damping=5, table=f.read())


class AbrahamsonGulerce2020SInter_USGSPRVI(abrahamson_gulerce_2020.AbrahamsonGulerce2020SInter):
    """
    Abrahamson and Gulerce (2020) subduction interface ground-motion model with 
    USGS adjustments for Puerto Rico and the Virgin Islands.
    """
    filename = pathlib.Path(__file__).parent / "AbrahamsonGulerce2020_USGSPRVI_coeffs.csv"
    with open(filename) as f:
        COEFFS = CoeffsTable(sa_damping=5, table=f.read())



class AbrahamsonGulerce2020SSlab_USGSPRVI(AbrahamsonGulerce2020SInter_USGSPRVI):
    """
    Implements the 2020 subduction intraslab ground motion model of Abrahamson &
    Gulerce (2020) GMM without site response.
    """
    #: Required rupture parameters are magnitude and top-of-rupture depth
    REQUIRES_RUPTURE_PARAMETERS = {'mag', 'ztor'}

    #: Supported tectonic region type is subduction inslab
    DEFINED_FOR_TECTONIC_REGION_TYPE = const.TRT.SUBDUCTION_INTRASLAB


class KuehnEtAl2020SInter_USGSPRVI(kuehn_2020.KuehnEtAl2020SInter):
    """
    Kuehn et al. (2020) subduction interface ground-motion model with 
    USGS adjustments for Puerto Rico and the Virgin Islands.
    """
    filename = pathlib.Path(__file__).parent / "KuehnEtAl2020_USGSPRVI_coeffs.csv"
    with open(filename) as f:
        COEFFS = CoeffsTable(sa_damping=5, table=f.read())


class KuehnEtAl2020SSlab_USGSPRVI(KuehnEtAl2020SInter_USGSPRVI):
    """
    Implements the 2020 subduction intraslab ground motion model of Kuehn
    et al. (2020) GMM without site response.
    """
    #: Supported tectonic region type is subduction inslab
    DEFINED_FOR_TECTONIC_REGION_TYPE = const.TRT.SUBDUCTION_INTRASLAB


class ParkerEtAl2020SInter_USGSPRVI(parker_2020.ParkerEtAl2020SInter):
    """
    Parker et al. (2020) subduction interface ground-motion model with 
    USGS adjustments for Puerto Rico and the Virgin Islands.
    """
    filename = pathlib.Path(__file__).parent / "ParkerEtAl2020_USGSPRVI_coeffs.csv"
    with open(filename) as f:
        COEFFS = CoeffsTable(sa_damping=5, table=f.read())
    

class ParkerEtAl2020SSlab_USGSPRVI(ParkerEtAl2020SInter_USGSPRVI):
    """
    Modifications for subduction slab.
    """
    DEFINED_FOR_TECTONIC_REGION_TYPE = const.TRT.SUBDUCTION_INTRASLAB

    # slab also requires hypo_depth
    REQUIRES_RUPTURE_PARAMETERS = {'mag', 'hypo_depth'}

    # constant table suffix
    SUFFIX = "slab"

    MB_REGIONS = {"Aleutian": 7.98, "AK": 7.2, "Cascadia": 7.2,
                  "CAM_S": 7.6, "CAM_N": 7.4, "JP_Pac": 7.65, "JP_Phi": 7.55,
                  "SA_N": 7.3, "SA_S": 7.25, "TW_W": 7.7, "TW_E": 7.7,
                  "default": 7.6}



class AdjustedGMM:
    """
    Add USGS period-independent adjustment to ground-motion models for Puerto Rico and the Virgin Islands.
    """

    ACTIVE_CRUSTAL_MEAN_ADJUSTMENT = -0.3
    SUBDUCTION_MEAN_ADJUSTMENT = -0.4

class AbrahamsonEtAl2014_USGSPRVIAdj(AbrahamsonEtAl2014_USGSPRVI):
    """
    Abrahamson et al. (2014) active crustal ground-motion model with 
    USGS adjustments for Puerto Rico and the Virgin Islands.
    """
    
    def __init__(self, sigma_mu_epsilon = 0.0, **kwargs):
        super().__init__(sigma_mu_epsilon=sigma_mu_epsilon, **kwargs)
        self.mean_adjustment = AdjustedGMM.ACTIVE_CRUSTAL_MEAN_ADJUSTMENT

    def compute(self, ctx: numpy.recarray, imts, mean, sig, tau, phi):
        """
        Compute GMM values.
        """
        super().compute(ctx, imts, mean, sig, tau, phi)
        mean += self.mean_adjustment


class BooreEtAl2014_USGSPRVIAdj(BooreEtAl2014_USGSPRVI):
    """
    Boore et al. (2014) active crustal ground-motion model with 
    USGS adjustments for Puerto Rico and the Virgin Islands.
    """
    
    def __init__(self, sigma_mu_epsilon = 0.0, **kwargs):
        super().__init__(sigma_mu_epsilon=sigma_mu_epsilon, **kwargs)
        self.mean_adjustment = AdjustedGMM.ACTIVE_CRUSTAL_MEAN_ADJUSTMENT

    def compute(self, ctx: numpy.recarray, imts, mean, sig, tau, phi):
        """
        Compute GMM values.
        """
        super().compute(ctx, imts, mean, sig, tau, phi)
        mean += self.mean_adjustment
    

class CampbellBozorgnia2014_USGSPRVIAdj(CampbellBozorgnia2014_USGSPRVI):
    """
    Campbell and Bozorgnia (2014) active crustal ground-motion model with 
    USGS adjustments for Puerto Rico and the Virgin Islands.
    """
    
    def __init__(self, sigma_mu_epsilon = 0.0, **kwargs):
        super().__init__(sigma_mu_epsilon=sigma_mu_epsilon, **kwargs)
        self.mean_adjustment = AdjustedGMM.ACTIVE_CRUSTAL_MEAN_ADJUSTMENT

    def compute(self, ctx: numpy.recarray, imts, mean, sig, tau, phi):
        """
        Compute GMM values.
        """
        super().compute(ctx, imts, mean, sig, tau, phi)
        mean += self.mean_adjustment
    

class ChiouYoungs2014_USGSPRVIAdj(ChiouYoungs2014_USGSPRVI):
    """
    Chiou and Youngs (2014) active crustal ground-motion model with 
    USGS adjustments for Puerto Rico and the Virgin Islands.
    """
    
    def __init__(self, sigma_mu_epsilon = 0.0, **kwargs):
        super().__init__(sigma_mu_epsilon=sigma_mu_epsilon, **kwargs)
        self.mean_adjustment = AdjustedGMM.ACTIVE_CRUSTAL_MEAN_ADJUSTMENT

    def compute(self, ctx: numpy.recarray, imts, mean, sig, tau, phi):
        """
        Compute GMM values.
        """
        super().compute(ctx, imts, mean, sig, tau, phi)
        mean += self.mean_adjustment
    

class AbrahamsonGulerce2020SInter_USGSPRVIAdj(AbrahamsonGulerce2020SInter_USGSPRVI):
    """
    Abrahamson and Gulerce (2020) subduction interface ground-motion model with 
    USGS adjustments for Puerto Rico and the Virgin Islands.
    """
    
    def __init__(self, sigma_mu_epsilon = 0.0, **kwargs):
        super().__init__(sigma_mu_epsilon=sigma_mu_epsilon, **kwargs)
        self.mean_adjustment = AdjustedGMM.SUBDUCTION_MEAN_ADJUSTMENT

    def compute(self, ctx: numpy.recarray, imts, mean, sig, tau, phi):
        """
        Compute GMM values.
        """
        super().compute(ctx, imts, mean, sig, tau, phi)
        mean += self.mean_adjustment


class AbrahamsonGulerce2020SSlab_USGSPRVIAdj(AbrahamsonGulerce2020SSlab_USGSPRVI):
    """
    Abrahamson and Gulerce (2020) subduction intraslab ground-motion model with 
    USGS adjustments for Puerto Rico and the Virgin Islands.
    """
    
    def __init__(self, sigma_mu_epsilon = 0.0, **kwargs):
        super().__init__(sigma_mu_epsilon=sigma_mu_epsilon, **kwargs)
        self.mean_adjustment = AdjustedGMM.SUBDUCTION_MEAN_ADJUSTMENT

    def compute(self, ctx: numpy.recarray, imts, mean, sig, tau, phi):
        """
        Compute GMM values.
        """
        super().compute(ctx, imts, mean, sig, tau, phi)
        mean += self.mean_adjustment


class KuehnEtAl2020SInter_USGSPRVIAdj(KuehnEtAl2020SInter_USGSPRVI):
    """
    Kuehn et al. (2020) subduction interface ground-motion model with 
    USGS adjustments for Puerto Rico and the Virgin Islands.
    """
    
    def __init__(self, sigma_mu_epsilon = 0.0, **kwargs):
        super().__init__(sigma_mu_epsilon=sigma_mu_epsilon, **kwargs)
        self.mean_adjustment = AdjustedGMM.SUBDUCTION_MEAN_ADJUSTMENT

    def compute(self, ctx: numpy.recarray, imts, mean, sig, tau, phi):
        """
        Compute GMM values.
        """
        super().compute(ctx, imts, mean, sig, tau, phi)
        mean += self.mean_adjustment


class KuehnEtAl2020SSlab_USGSPRVIAdj(KuehnEtAl2020SSlab_USGSPRVI):
    """
    Kuehn et al. (2020) subduction intraslab ground-motion model with 
    USGS adjustments for Puerto Rico and the Virgin Islands.
    """
    
    def __init__(self, sigma_mu_epsilon = 0.0, **kwargs):
        super().__init__(sigma_mu_epsilon=sigma_mu_epsilon, **kwargs)
        self.mean_adjustment = AdjustedGMM.SUBDUCTION_MEAN_ADJUSTMENT

    def compute(self, ctx: numpy.recarray, imts, mean, sig, tau, phi):
        """
        Compute GMM values.
        """
        super().compute(ctx, imts, mean, sig, tau, phi)
        mean += self.mean_adjustment


class ParkerEtAl2020SInter_USGSPRVIAdj(ParkerEtAl2020SInter_USGSPRVI):
    """
    Parker et al. (2020) subduction interface ground-motion model with 
    USGS adjustments for Puerto Rico and the Virgin Islands.
    """
    
    def __init__(self, sigma_mu_epsilon = 0.0, **kwargs):
        super().__init__(sigma_mu_epsilon=sigma_mu_epsilon, **kwargs)
        self.mean_adjustment = AdjustedGMM.SUBDUCTION_MEAN_ADJUSTMENT

    def compute(self, ctx: numpy.recarray, imts, mean, sig, tau, phi):
        """
        Compute GMM values.
        """
        super().compute(ctx, imts, mean, sig, tau, phi)
        mean += self.mean_adjustment


class ParkerEtAl2020SSlab_USGSPRVIAdj(ParkerEtAl2020SSlab_USGSPRVI):
    """
    Parker et al. (2020) subduction intraslab ground-motion model with 
    USGS adjustments for Puerto Rico and the Virgin Islands.
    """
    
    def __init__(self, sigma_mu_epsilon = 0.0, **kwargs):
        super().__init__(sigma_mu_epsilon=sigma_mu_epsilon, **kwargs)
        self.mean_adjustment = AdjustedGMM.SUBDUCTION_MEAN_ADJUSTMENT

    def compute(self, ctx: numpy.recarray, imts, mean, sig, tau, phi):
        """
        Compute GMM values.
        """
        super().compute(ctx, imts, mean, sig, tau, phi)
        mean += self.mean_adjustment
