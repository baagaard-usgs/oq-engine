# -*- coding: utf-8 -*-
"""
USGS adjustments to ground-motion models for Puerto Rico and the Virgin Islands.

The adjustments correspond to the fixed effect computed using linear mixed effects with
grouping for the event and site terms.
"""
import numpy
import abc

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

    COEFFS = CoeffsTable(sa_damping=5, table="""\
    IMT         m1        vlin           b           c          c4          a1          a2          a3          a4          a5          a6          a7          a8         a10         a11         a12         a13         a14         a15         a17         a43         a44         a45         a46         a25         a28         a29         a31         a36         a37         a38         a39         a40         a41         a42         s1e         s2e          s3          s4         s1m         s2m          s5          s6
    pgv   6.750000  330.000000   -2.020000 2400.000000    4.500000    5.975000   -0.919000    0.275000   -0.100000   -0.410000    2.366000    0.000000   -0.094000    1.691036    0.000000   -0.100000    0.250000    0.220000    0.300000   -0.000500    0.280000    0.150000    0.090000    0.070000   -0.000100    0.000500   -0.003700   -0.146200    0.377000    0.212000    0.157000    0.000000    0.095000   -0.038000    0.065000    0.662000    0.510000    0.380000    0.380000    0.660000    0.510000    0.580000    0.530000
    pga   6.750000  660.000000   -1.470000    2.400000    4.500000    0.587000   -0.790000    0.275000   -0.100000   -0.410000    2.154000    0.000000   -0.015000    2.108038    0.000000   -0.100000    0.600000   -0.300000    1.100000   -0.007200    0.100000    0.050000    0.000000   -0.050000   -0.001500    0.002500   -0.003400   -0.150300    0.265000    0.337000    0.188000    0.000000    0.088000   -0.196000    0.044000    0.754000    0.520000    0.470000    0.360000    0.741000    0.501000    0.540000    0.630000
  0.010   6.750000  660.000000   -1.470000    2.400000    4.500000    0.587000   -0.790000    0.275000   -0.100000   -0.410000    2.154000    0.000000   -0.015000    1.735000    0.000000   -0.100000    0.600000   -0.300000    1.100000   -0.007200    0.100000    0.050000    0.000000   -0.050000   -0.001500    0.002500   -0.003400   -0.150300    0.265000    0.337000    0.188000    0.000000    0.088000   -0.196000    0.044000    0.754000    0.520000    0.470000    0.360000    0.741000    0.501000    0.540000    0.630000
  0.020   6.750000  680.000000   -1.460000    2.400000    4.500000    0.598000   -0.790000    0.275000   -0.100000   -0.410000    2.146000    0.000000   -0.015000    1.718000    0.000000   -0.100000    0.600000   -0.300000    1.100000   -0.007300    0.100000    0.050000    0.000000   -0.050000   -0.001500    0.002400   -0.003300   -0.147900    0.255000    0.328000    0.184000    0.000000    0.088000   -0.194000    0.061000    0.760000    0.520000    0.470000    0.360000    0.747000    0.501000    0.540000    0.630000
  0.030   6.750000  770.000000   -1.390000    2.400000    4.500000    0.602000   -0.790000    0.275000   -0.100000   -0.410000    2.157000    0.000000   -0.015000    1.615000    0.000000   -0.100000    0.600000   -0.300000    1.100000   -0.007500    0.100000    0.050000    0.000000   -0.050000   -0.001600    0.002300   -0.003400   -0.144700    0.249000    0.320000    0.180000    0.000000    0.093000   -0.175000    0.162000    0.781000    0.520000    0.470000    0.360000    0.769000    0.501000    0.550000    0.630000
  0.050   6.750000  915.000000   -1.220000    2.400000    4.500000    0.707000   -0.790000    0.275000   -0.100000   -0.410000    2.085000    0.000000   -0.015000    1.548745    0.000000   -0.100000    0.600000   -0.300000    1.100000   -0.008000    0.100000    0.050000    0.000000   -0.050000   -0.002000    0.002700   -0.003300   -0.132600    0.202000    0.289000    0.167000    0.000000    0.133000   -0.090000    0.451000    0.810000    0.530000    0.470000    0.360000    0.798000    0.512000    0.560000    0.650000
  0.075   6.750000  960.000000   -1.150000    2.400000    4.500000    0.973000   -0.790000    0.275000   -0.100000   -0.410000    2.029000    0.000000   -0.015000    1.436491    0.000000   -0.100000    0.600000   -0.300000    1.100000   -0.008900    0.100000    0.050000    0.000000   -0.050000   -0.002700    0.003200   -0.002900   -0.135300    0.126000    0.275000    0.173000    0.000000    0.186000    0.090000    0.506000    0.810000    0.540000    0.470000    0.360000    0.798000    0.522000    0.570000    0.690000
  0.100   6.750000  910.000000   -1.230000    2.400000    4.500000    1.169000   -0.790000    0.275000   -0.100000   -0.410000    2.041000    0.000000   -0.015000    1.496934    0.000000   -0.100000    0.600000   -0.300000    1.100000   -0.009500    0.100000    0.050000    0.000000   -0.050000   -0.003300    0.003600   -0.002500   -0.112800    0.022000    0.256000    0.189000    0.000000    0.160000    0.006000    0.335000    0.810000    0.550000    0.470000    0.360000    0.795000    0.527000    0.570000    0.700000
  0.150   6.750000  740.000000   -1.590000    2.400000    4.500000    1.442000   -0.790000    0.275000   -0.100000   -0.410000    2.121000    0.000000   -0.022000    1.836462    0.000000   -0.100000    0.600000   -0.300000    1.100000   -0.009500    0.100000    0.050000    0.000000   -0.050000   -0.003500    0.003300   -0.002500    0.038300   -0.136000    0.162000    0.108000    0.000000    0.068000   -0.156000   -0.084000    0.801000    0.560000    0.470000    0.360000    0.773000    0.519000    0.580000    0.700000
  0.200   6.750000  590.000000   -2.010000    2.400000    4.500000    1.637000   -0.790000    0.275000   -0.100000   -0.410000    2.224000    0.000000   -0.030000    2.354954    0.000000   -0.100000    0.600000   -0.300000    1.100000   -0.008600    0.100000    0.050000    0.000000   -0.030000   -0.003300    0.002700   -0.003100    0.077500   -0.078000    0.224000    0.115000    0.000000    0.048000   -0.274000   -0.178000    0.789000    0.565000    0.470000    0.360000    0.753000    0.514000    0.590000    0.700000
  0.250   6.750000  495.000000   -2.410000    2.400000    4.500000    1.701000   -0.790000    0.275000   -0.100000   -0.410000    2.312000    0.000000   -0.038000    2.753642    0.000000   -0.100000    0.600000   -0.240000    1.100000   -0.007400    0.100000    0.050000    0.000000    0.000000   -0.002900    0.002400   -0.003600    0.074100    0.037000    0.248000    0.122000    0.000000    0.055000   -0.248000   -0.187000    0.770000    0.570000    0.470000    0.360000    0.729000    0.513000    0.610000    0.700000
  0.300   6.750000  430.000000   -2.760000    2.400000    4.500000    1.712000   -0.790000    0.275000   -0.100000   -0.410000    2.338000    0.000000   -0.045000    3.014200    0.000000   -0.100000    0.600000   -0.190000    1.030000   -0.006400    0.100000    0.050000    0.030000    0.030000   -0.002700    0.002000   -0.003900    0.254800   -0.091000    0.203000    0.096000    0.000000    0.073000   -0.203000   -0.159000    0.740000    0.580000    0.470000    0.360000    0.693000    0.519000    0.630000    0.700000
  0.400   6.750000  360.000000   -3.280000    2.400000    4.500000    1.662000   -0.790000    0.275000   -0.100000   -0.410000    2.469000    0.000000   -0.055000    3.493741    0.000000   -0.100000    0.580000   -0.110000    0.920000   -0.004300    0.100000    0.070000    0.060000    0.060000   -0.002300    0.001000   -0.004800    0.213600    0.129000    0.232000    0.123000    0.000000    0.143000   -0.154000   -0.023000    0.699000    0.590000    0.470000    0.360000    0.644000    0.524000    0.660000    0.700000
  0.500   6.750000  340.000000   -3.600000    2.400000    4.500000    1.571000   -0.790000    0.275000   -0.100000   -0.410000    2.559000    0.000000   -0.065000    3.817007    0.000000   -0.100000    0.560000   -0.040000    0.840000   -0.003200    0.100000    0.100000    0.100000    0.090000   -0.002000    0.000800   -0.005000    0.154200    0.310000    0.252000    0.134000    0.000000    0.160000   -0.159000   -0.029000    0.676000    0.600000    0.470000    0.360000    0.616000    0.532000    0.690000    0.700000
  0.750   6.750000  330.000000   -3.800000    2.400000    4.500000    1.299000   -0.790000    0.275000   -0.100000   -0.410000    2.682000    0.000000   -0.095000    4.161542    0.000000   -0.100000    0.530000    0.070000    0.680000   -0.002500    0.140000    0.140000    0.140000    0.130000   -0.001000    0.000700   -0.004100    0.078700    0.505000    0.208000    0.129000    0.000000    0.158000   -0.141000    0.061000    0.631000    0.615000    0.470000    0.360000    0.566000    0.548000    0.730000    0.690000
  1.000   6.750000  330.000000   -3.500000    2.400000    4.500000    1.043000   -0.790000    0.275000   -0.100000   -0.410000    2.763000    0.000000   -0.110000    3.721063    0.000000   -0.100000    0.500000    0.150000    0.570000   -0.002500    0.170000    0.170000    0.170000    0.140000   -0.000500    0.000700   -0.003200    0.047600    0.358000    0.208000    0.152000    0.000000    0.145000   -0.144000    0.062000    0.609000    0.630000    0.470000    0.360000    0.541000    0.565000    0.770000    0.680000
  1.500   6.750000  330.000000   -2.400000    2.400000    4.500000    0.665000   -0.790000    0.275000   -0.100000   -0.410000    2.836000    0.000000   -0.124000    1.827234    0.000000   -0.100000    0.420000    0.270000    0.420000   -0.002200    0.220000    0.210000    0.200000    0.160000   -0.000400    0.000600   -0.002000   -0.016300    0.131000    0.108000    0.118000    0.000000    0.131000   -0.126000    0.037000    0.578000    0.640000    0.470000    0.360000    0.506000    0.576000    0.800000    0.660000
  2.000   6.750000  330.000000   -1.000000    2.400000    4.500000    0.329000   -0.790000    0.275000   -0.100000   -0.410000    2.897000    0.000000   -0.138000   -0.318038    0.000000   -0.100000    0.350000    0.350000    0.310000   -0.001900    0.260000    0.250000    0.220000    0.160000   -0.000200    0.000300   -0.001700   -0.120300    0.123000    0.068000    0.119000    0.000000    0.083000   -0.075000   -0.143000    0.555000    0.650000    0.470000    0.360000    0.480000    0.587000    0.800000    0.620000
  3.000   6.820000  330.000000    0.000000    2.400000    4.500000   -0.060000   -0.790000    0.275000   -0.100000   -0.410000    2.906000    0.000000   -0.172000   -1.633661    0.000000   -0.100000    0.200000    0.460000    0.160000   -0.001500    0.340000    0.300000    0.230000    0.160000    0.000000    0.000000   -0.002000   -0.271900    0.109000   -0.023000    0.093000    0.000000    0.070000   -0.021000   -0.028000    0.548000    0.640000    0.470000    0.360000    0.472000    0.576000    0.800000    0.550000
  4.000   6.920000  330.000000    0.000000    2.400000    4.500000   -0.299000   -0.790000    0.275000   -0.100000   -0.410000    2.889000    0.000000   -0.197000   -1.633661    0.000000   -0.100000    0.000000    0.540000    0.050000   -0.001000    0.410000    0.320000    0.230000    0.140000    0.000000    0.000000   -0.002000   -0.295800    0.135000    0.028000    0.084000    0.000000    0.101000    0.072000   -0.097000    0.527000    0.630000    0.470000    0.360000    0.447000    0.565000    0.760000    0.520000
  5.000   7.000000  330.000000    0.000000    2.400000    4.500000   -0.562000   -0.765000    0.275000   -0.100000   -0.410000    2.898000    0.000000   -0.218000   -1.633661    0.000000   -0.100000    0.000000    0.610000   -0.040000   -0.001000    0.510000    0.320000    0.220000    0.130000    0.000000    0.000000   -0.002000   -0.271800    0.189000    0.031000    0.058000    0.000000    0.095000    0.205000    0.015000    0.505000    0.630000    0.470000    0.360000    0.425000    0.568000    0.720000    0.500000
  7.500   7.150000  330.000000    0.000000    2.400000    4.500000   -1.303000   -0.634000    0.275000   -0.100000   -0.410000    2.870000    0.000000   -0.255000   -1.633661    0.000000   -0.200000    0.000000    0.720000   -0.190000   -0.001000    0.490000    0.280000    0.170000    0.090000    0.000000    0.000000   -0.002000   -0.140000    0.150000   -0.070000    0.000000    0.000000    0.151000    0.329000    0.299000    0.457000    0.630000    0.470000    0.360000    0.378000    0.575000    0.670000    0.500000
 10.000   7.250000  330.000000    0.000000    2.400000    4.500000   -1.928000   -0.529000    0.275000   -0.100000   -0.410000    2.843000    0.000000   -0.285000   -1.633661    0.000000   -0.200000    0.000000    0.800000   -0.300000   -0.001000    0.420000    0.220000    0.140000    0.080000    0.000000    0.000000   -0.002000   -0.021600    0.092000   -0.159000   -0.050000    0.000000    0.124000    0.301000    0.243000    0.429000    0.630000    0.470000    0.360000    0.359000    0.585000    0.640000    0.500000
""")


class BooreEtAl2014_USGSPRVI(boore_2014.BooreEtAl2014):
    """
    Boore et al. (2014) active crustal ground-motion model with 
    USGS adjustments for Puerto Rico and the Virgin Islands.
    """

    COEFFS = CoeffsTable(sa_damping=5, table="""\
    IMT         e0          e1          e2          e3          e4          e5          e6          Mh          c1          c2          c3           h         Dc3           c          Vc          f4          f5          f6          f7          R1          R2         DfR         DfV          f1          f2        tau1        tau2
    pgv   5.037000    5.078000    4.849000    5.033000    1.073000   -0.153600    0.225200    6.200000   -1.243000    0.148900   -0.003440    5.300000    0.000000   -0.349728 3693.589041   -0.100000   -0.008440   -9.900000   -9.900000  105.000000  272.000000    0.082000    0.080000    0.644000    0.552000    0.401000    0.346000
    pga   0.447300    0.485600    0.245900    0.453900    1.431000    0.050530   -0.166200    5.500000   -1.134000    0.191700   -0.008088    4.500000    0.000000   -0.410626 2471.206720   -0.150000   -0.007010   -9.900000   -9.900000  110.000000  270.000000    0.100000    0.070000    0.695000    0.495000    0.398000    0.348000
  0.010   0.453400    0.491600    0.251900    0.459900    1.421000    0.049320   -0.165900    5.500000   -1.134000    0.191600   -0.008088    4.500000    0.000000   -0.603720 1500.200000   -0.148330   -0.007010   -9.900000   -9.900000  111.670000  270.000000    0.096000    0.070000    0.698000    0.499000    0.402000    0.345000
  0.020   0.485980    0.523590    0.297070    0.488750    1.433100    0.053388   -0.165610    5.500000   -1.139400    0.189620   -0.008074    4.500000    0.000000   -0.573880 1500.360000   -0.147100   -0.007280   -9.900000   -9.900000  113.100000  270.000000    0.092000    0.030000    0.702000    0.502000    0.409000    0.346000
  0.030   0.569160    0.609200    0.403910    0.557830    1.426100    0.061444   -0.166900    5.500000   -1.142100    0.188420   -0.008336    4.490000    0.000000   -0.534140 1502.950000   -0.154850   -0.007350   -9.900000   -9.900000  112.130000  270.000000    0.081000    0.029000    0.721000    0.514000    0.445000    0.364000
  0.050   0.754360    0.799050    0.606520    0.727260    1.397400    0.067357   -0.180820    5.500000   -1.115900    0.187090   -0.009819    4.200000    0.000000   -0.573541 1150.549107   -0.192000   -0.006470   -9.900000   -9.900000   97.930000  270.000000    0.063000    0.030000    0.753000    0.532000    0.503000    0.426000
  0.075   0.964470    1.007700    0.776780    0.956300    1.417400    0.073549   -0.196650    5.500000   -1.083100    0.182250   -0.010580    4.040000    0.000000   -0.560944  887.324000   -0.235000   -0.005730   -9.900000   -9.900000   85.990000  270.040000    0.064000    0.022000    0.745000    0.542000    0.474000    0.466000
  0.100   1.126800    1.166900    0.887100    1.145400    1.429300    0.055231   -0.198380    5.540000   -1.065200    0.172030   -0.010200    4.130000    0.000000   -0.541360  849.701810   -0.249160   -0.005600   -9.900000   -9.900000   79.590000  270.090000    0.087000    0.014000    0.728000    0.541000    0.415000    0.458000
  0.150   1.309500    1.348100    1.064800    1.332400    1.284400   -0.042065   -0.182340    5.740000   -1.053200    0.154010   -0.008977    4.390000    0.000000   -0.644475  980.085643   -0.257130   -0.005850   -9.900000   -9.900000   81.330000  270.160000    0.120000    0.015000    0.720000    0.537000    0.354000    0.388000
  0.200   1.325500    1.359000    1.122000    1.341400    1.134900   -0.110960   -0.158520    5.920000   -1.060700    0.144890   -0.007717    4.610000    0.000000   -0.728422  912.689141   -0.246580   -0.006140   -9.900000   -9.900000   90.910000  270.000000    0.136000    0.045000    0.711000    0.539000    0.344000    0.309000
  0.250   1.276600    1.301700    1.082800    1.305200    1.016600   -0.162130   -0.127840    6.050000   -1.077300    0.139250   -0.006517    4.780000    0.000000   -0.791379  887.339258   -0.235740   -0.006440   -9.900000   -9.900000   97.040000  269.450000    0.141000    0.055000    0.698000    0.547000    0.350000    0.266000
  0.300   1.221700    1.240100    1.024600    1.265300    0.956760   -0.195900   -0.092855    6.140000   -1.094800    0.133880   -0.005475    4.930000    0.000000   -0.750531 1034.504980   -0.219120   -0.006700   -9.900000   -9.900000  103.150000  268.590000    0.138000    0.050000    0.675000    0.561000    0.363000    0.229000
  0.400   1.104600    1.121400    0.897650    1.155200    0.967660   -0.226080   -0.023189    6.200000   -1.124300    0.125120   -0.004053    5.160000    0.000000   -0.583279 2393.703999   -0.195820   -0.007130   -9.900000   -9.900000  106.020000  266.540000    0.122000    0.049000    0.643000    0.580000    0.381000    0.210000
  0.500   0.969910    0.991060    0.761500    1.012000    1.038400   -0.235220    0.029119    6.200000   -1.145900    0.120150   -0.003220    5.340000    0.000000   -0.324727 1723.946608   -0.175000   -0.007440   -9.900000   -9.900000  105.540000  265.000000    0.109000    0.060000    0.615000    0.599000    0.410000    0.224000
  0.750   0.669030    0.697370    0.475230    0.691730    1.287100   -0.215910    0.108290    6.200000   -1.177700    0.110540   -0.001931    5.600000    0.000000   -0.102238 1905.724343   -0.138660   -0.008120    0.092259    0.059024  108.390000  266.510000    0.100000    0.070000    0.581000    0.622000    0.457000    0.266000
  1.000   0.393200    0.421800    0.207000    0.412400    1.500400   -0.189830    0.178950    6.200000   -1.193000    0.102480   -0.001210    5.740000    0.000000   -0.086406 1899.063339   -0.105210   -0.008440    0.366950    0.207890  116.390000  270.000000    0.098000    0.020000    0.553000    0.625000    0.498000    0.298000
  1.500  -0.149540   -0.118660   -0.313800   -0.143700    1.762200   -0.146700    0.338960    6.200000   -1.206300    0.096445   -0.000365    6.180000    0.000000   -0.086406 1899.063339   -0.062000   -0.007710    0.637890    0.309440  125.380000  262.410000    0.104000    0.010000    0.532000    0.619000    0.525000    0.315000
  2.000  -0.586690   -0.550030   -0.714660   -0.606580    1.915200   -0.112370    0.447880    6.200000   -1.215900    0.096361    0.000000    6.540000    0.000000   -0.086406 1899.063339   -0.036136   -0.004790    0.871380    0.382450  130.370000  240.140000    0.105000    0.008000    0.526000    0.618000    0.532000    0.329000
  3.000  -1.189800   -1.142000   -1.230000   -1.266400    2.132300   -0.043320    0.626940    6.200000   -1.217900    0.097638    0.000000    6.930000    0.000000   -0.086406 1899.063339   -0.013577   -0.001830    1.134800    0.515850  130.360000  195.000000    0.088000    0.000000    0.534000    0.619000    0.537000    0.344000
  4.000  -1.638800   -1.574800   -1.667300   -1.751600    2.204000   -0.014642    0.763030    6.200000   -1.216200    0.102180   -0.000052    7.320000    0.000000   -0.086406 1899.063339   -0.003212   -0.001520    1.271100    0.629390  129.490000  199.450000    0.070000    0.000000    0.536000    0.616000    0.543000    0.349000
  5.000  -1.966000   -1.888200   -2.024500   -2.092800    2.229900   -0.014855    0.873140    6.200000   -1.218900    0.103530    0.000000    7.780000    0.000000   -0.086406 1899.063339   -0.000255   -0.001440    1.328900    0.738060  130.220000  230.000000    0.061000    0.000000    0.528000    0.622000    0.532000    0.335000
  7.500  -2.586500   -2.487400   -2.817600   -2.685400    2.118700   -0.081606    1.012100    6.200000   -1.254300    0.125070    0.000000    9.480000    0.000000   -0.086406 1899.063339   -0.000055   -0.001370    1.328800    0.809000  130.720000  250.390000    0.058000    0.000000    0.512000    0.634000    0.511000    0.270000
 10.000  -3.070200   -2.953700   -3.377600   -3.172600    1.883700   -0.150960    1.065100    6.200000   -1.325300    0.151830    0.000000    9.660000    0.000000   -0.086406 1899.063339    0.000000   -0.001360    1.182900    0.703000  130.000000  210.000000    0.060000    0.000000    0.510000    0.604000    0.487000    0.239000
""")


class CampbellBozorgnia2014_USGSPRVI(campbell_bozorgnia_2014.CampbellBozorgnia2014):
    """
    Campbell and Bozorgnia (2014) active crustal ground-motion model with 
    USGS adjustments for Puerto Rico and the Virgin Islands.
    """

    COEFFS = CoeffsTable(sa_damping=5, table="""\
    IMT         c0          c1          c2          c3          c4          c5          c6          c7          c9         c10         c11         c12         c13         c14         c15         c16         c17         c18         c19         c20        Dc20          a2          h1          h2          h3          h5          h6          k1          k2          k3        phi1        phi2        tau1        tau2        phiC      rholny
    pgv  -2.895000    1.510000    0.270000   -1.299000   -0.453000   -2.466000    0.204000    5.837000   -0.168000    0.305000    1.232085    2.602000    2.457000    0.106000    0.332000    0.585000    0.051700    0.032700    0.006130   -0.001700    0.000000    0.596000    0.117000    1.616000   -0.733000   -0.128000   -0.756000  400.000000   -1.955000    1.929000    0.655000    0.494000    0.317000    0.297000    0.190000    0.684000
    pga  -4.416000    0.984000    0.537000   -1.499000   -0.496000   -2.773000    0.248000    6.768000   -0.212000    0.720000    0.957867    2.186000    1.420000   -0.006400   -0.202000    0.393000    0.097700    0.033300    0.007570   -0.005500    0.000000    0.167000    0.241000    1.474000   -0.715000   -0.337000   -0.270000  865.000000   -1.186000    1.839000    0.734000    0.492000    0.409000    0.322000    0.166000    1.000000
  0.010  -4.365000    0.977000    0.533000   -1.485000   -0.499000   -2.773000    0.248000    6.753000   -0.214000    0.720000    1.094000    2.191000    1.416000   -0.007000   -0.207000    0.390000    0.098100    0.033400    0.007550   -0.005500    0.000000    0.168000    0.242000    1.471000   -0.714000   -0.336000   -0.270000  865.000000   -1.186000    1.839000    0.734000    0.492000    0.404000    0.325000    0.166000    1.000000
  0.020  -4.348000    0.976000    0.549000   -1.488000   -0.501000   -2.772000    0.247000    6.502000   -0.208000    0.730000    1.149000    2.189000    1.453000   -0.016700   -0.199000    0.387000    0.100900    0.032700    0.007590   -0.005500    0.000000    0.166000    0.244000    1.467000   -0.711000   -0.339000   -0.263000  865.000000   -1.219000    1.840000    0.738000    0.496000    0.417000    0.326000    0.166000    0.998000
  0.030  -4.024000    0.931000    0.628000   -1.494000   -0.517000   -2.782000    0.246000    6.291000   -0.213000    0.759000    1.290000    2.164000    1.476000   -0.042200   -0.202000    0.378000    0.109500    0.033100    0.007900   -0.005700    0.000000    0.167000    0.246000    1.467000   -0.713000   -0.338000   -0.259000  908.000000   -1.273000    1.841000    0.747000    0.503000    0.446000    0.344000    0.165000    0.986000
  0.050  -3.479000    0.887000    0.674000   -1.388000   -0.615000   -2.791000    0.240000    6.317000   -0.244000    0.826000    1.256414    2.138000    1.549000   -0.066300   -0.339000    0.295000    0.122600    0.027000    0.008030   -0.006300    0.000000    0.173000    0.251000    1.449000   -0.701000   -0.338000   -0.263000 1054.000000   -1.346000    1.843000    0.777000    0.520000    0.508000    0.377000    0.162000    0.938000
  0.075  -3.293000    0.902000    0.726000   -1.469000   -0.596000   -2.745000    0.227000    6.861000   -0.266000    0.815000    1.290439    2.446000    1.772000   -0.079400   -0.404000    0.322000    0.116500    0.028800    0.008110   -0.007000    0.000000    0.198000    0.260000    1.435000   -0.695000   -0.347000   -0.219000 1086.000000   -1.471000    1.845000    0.782000    0.535000    0.504000    0.418000    0.158000    0.887000
  0.100  -3.666000    0.993000    0.698000   -1.572000   -0.536000   -2.633000    0.210000    7.294000   -0.229000    0.831000    1.349152    2.969000    1.916000   -0.029400   -0.416000    0.384000    0.099800    0.032500    0.007440   -0.007300    0.000000    0.174000    0.259000    1.449000   -0.708000   -0.391000   -0.201000 1032.000000   -1.624000    1.847000    0.769000    0.543000    0.445000    0.426000    0.170000    0.870000
  0.150  -4.866000    1.267000    0.510000   -1.669000   -0.490000   -2.458000    0.183000    8.031000   -0.211000    0.749000    1.567701    3.544000    2.161000    0.064200   -0.407000    0.417000    0.076000    0.038800    0.007160   -0.006900    0.000000    0.198000    0.254000    1.461000   -0.715000   -0.449000   -0.099000  878.000000   -1.931000    1.852000    0.769000    0.543000    0.382000    0.387000    0.180000    0.876000
  0.200  -5.411000    1.366000    0.447000   -1.750000   -0.451000   -2.421000    0.182000    8.385000   -0.163000    0.764000    1.719237    3.707000    2.465000    0.096800   -0.311000    0.404000    0.057100    0.043700    0.006880   -0.006000    0.000000    0.204000    0.237000    1.484000   -0.721000   -0.393000   -0.198000  748.000000   -2.188000    1.856000    0.761000    0.552000    0.339000    0.338000    0.186000    0.870000
  0.250  -5.962000    1.458000    0.274000   -1.711000   -0.404000   -2.392000    0.189000    7.534000   -0.150000    0.716000    1.944150    3.343000    2.766000    0.144100   -0.172000    0.466000    0.043700    0.046300    0.005560   -0.005500    0.000000    0.185000    0.206000    1.581000   -0.787000   -0.339000   -0.210000  654.000000   -2.381000    1.861000    0.744000    0.545000    0.340000    0.316000    0.191000    0.850000
  0.300  -6.403000    1.528000    0.193000   -1.770000   -0.321000   -2.376000    0.195000    6.990000   -0.131000    0.737000    2.151170    3.334000    3.011000    0.159700   -0.084000    0.528000    0.032300    0.050800    0.004580   -0.004900    0.000000    0.164000    0.210000    1.586000   -0.795000   -0.447000   -0.121000  587.000000   -2.518000    1.865000    0.727000    0.568000    0.340000    0.300000    0.198000    0.819000
  0.400  -7.566000    1.739000   -0.020000   -1.594000   -0.426000   -2.303000    0.185000    7.012000   -0.159000    0.738000    2.287169    3.544000    3.203000    0.141000    0.085000    0.540000    0.020900    0.043200    0.004010   -0.003700    0.000000    0.160000    0.226000    1.544000   -0.770000   -0.525000   -0.086000  503.000000   -2.657000    1.874000    0.690000    0.593000    0.356000    0.264000    0.206000    0.743000
  0.500  -8.379000    1.872000   -0.121000   -1.577000   -0.440000   -2.296000    0.186000    6.902000   -0.153000    0.718000    2.263771    3.016000    3.333000    0.147400    0.233000    0.638000    0.009200    0.040500    0.003880   -0.002700    0.000000    0.184000    0.217000    1.554000   -0.770000   -0.407000   -0.281000  457.000000   -2.669000    1.883000    0.663000    0.611000    0.379000    0.263000    0.208000    0.684000
  0.750  -9.841000    2.021000   -0.042000   -1.757000   -0.443000   -2.232000    0.186000    5.522000   -0.090000    0.795000    1.738420    2.616000    3.054000    0.176400    0.411000    0.776000   -0.008200    0.042000    0.004200   -0.001600    0.000000    0.216000    0.154000    1.626000   -0.780000   -0.371000   -0.285000  410.000000   -2.401000    1.906000    0.606000    0.633000    0.430000    0.326000    0.221000    0.562000
  1.000 -11.011000    2.180000   -0.069000   -1.707000   -0.527000   -2.158000    0.169000    5.650000   -0.105000    0.556000    1.122445    2.470000    2.562000    0.259300    0.479000    0.771000   -0.013100    0.042600    0.004090   -0.000600    0.000000    0.596000    0.117000    1.616000   -0.733000   -0.128000   -0.756000  400.000000   -1.955000    1.929000    0.579000    0.628000    0.470000    0.353000    0.225000    0.467000
  1.500 -12.469000    2.270000    0.047000   -1.621000   -0.630000   -2.063000    0.158000    5.795000   -0.058000    0.480000    1.122445    2.108000    1.453000    0.288100    0.566000    0.748000   -0.018700    0.038000    0.004240    0.000000    0.000000    0.596000    0.117000    1.616000   -0.733000   -0.128000   -0.756000  400.000000   -1.025000    1.974000    0.541000    0.603000    0.497000    0.399000    0.222000    0.364000
  2.000 -12.969000    2.271000    0.149000   -1.512000   -0.768000   -2.104000    0.158000    6.632000   -0.028000    0.401000    1.122445    1.327000    0.657000    0.311200    0.562000    0.763000   -0.025800    0.025200    0.004480    0.000000    0.000000    0.596000    0.117000    1.616000   -0.733000   -0.128000   -0.756000  400.000000   -0.299000    2.019000    0.529000    0.588000    0.499000    0.400000    0.226000    0.298000
  3.000 -13.306000    2.150000    0.368000   -1.315000   -0.890000   -2.051000    0.148000    6.759000    0.000000    0.206000    1.122445    0.601000    0.367000    0.347800    0.534000    0.686000   -0.031100    0.023600    0.003450    0.000000    0.000000    0.596000    0.117000    1.616000   -0.733000   -0.128000   -0.756000  400.000000    0.000000    2.110000    0.527000    0.578000    0.500000    0.417000    0.229000    0.234000
  4.000 -14.020000    2.132000    0.726000   -1.506000   -0.885000   -1.986000    0.135000    7.978000    0.000000    0.105000    1.122445    0.568000    0.306000    0.374700    0.522000    0.691000   -0.041300    0.010200    0.006030    0.000000    0.000000    0.596000    0.117000    1.616000   -0.733000   -0.128000   -0.756000  400.000000    0.000000    2.200000    0.521000    0.559000    0.543000    0.393000    0.237000    0.202000
  5.000 -14.558000    2.116000    1.027000   -1.721000   -0.878000   -2.021000    0.135000    8.538000    0.000000    0.000000    1.122445    0.356000    0.268000    0.338200    0.477000    0.670000   -0.028100    0.003400    0.008050    0.000000    0.000000    0.596000    0.117000    1.616000   -0.733000   -0.128000   -0.756000  400.000000    0.000000    2.291000    0.502000    0.551000    0.534000    0.421000    0.237000    0.184000
  7.500 -15.509000    2.223000    0.169000   -0.756000   -1.077000   -2.179000    0.165000    8.468000    0.000000    0.000000    1.122445    0.075000    0.374000    0.375400    0.321000    0.757000   -0.020500    0.005000    0.002800    0.000000    0.000000    0.596000    0.117000    1.616000   -0.733000   -0.128000   -0.756000  400.000000    0.000000    2.517000    0.457000    0.546000    0.523000    0.438000    0.271000    0.176000
 10.000 -15.975000    2.132000    0.367000   -0.800000   -1.282000   -2.244000    0.180000    6.564000    0.000000    0.000000    1.122445   -0.027000    0.297000    0.350600    0.174000    0.621000    0.000900    0.009900    0.004580    0.000000    0.000000    0.596000    0.117000    1.616000   -0.733000   -0.128000   -0.756000  400.000000    0.000000    2.744000    0.441000    0.543000    0.466000    0.438000    0.290000    0.154000
""")


class ChiouYoungs2014_USGSPRVI(chiou_youngs_2014.ChiouYoungs2014):
    """
    Chiou and Youngs (2014) active crustal ground-motion model with 
    USGS adjustments for Puerto Rico and the Virgin Islands.
    """

    COEFFS = CoeffsTable(sa_damping=5, table="""\
    IMT         c1         c1a         c1b         c1c         c1d          cn          cm          c2          c3          c4         c4a         crb          c5         chm          c6          c7         c7b          c8         c8a         c8b          c9         c9a         c9b         c11        c11b         cg1         cg2         cg3        phi1        phi2        phi3        phi4        phi5        phi6       gjpit         gwn      phi1jp      phi5jp      phi6jp        tau1        tau2        sig1        sig2        sig3      sig2jp
    pgv   2.354900    0.165000   -0.062600   -0.165000    0.062600    3.302400    5.423000    1.060000    2.315200   -2.100000   -0.500000   50.000000    5.809600    3.051400    0.440700    0.032400    0.009700    0.215400    0.269500    5.000000    0.307900    0.100000    6.500000    0.000000   -0.383400   -0.001852   -0.007403    4.343900   -0.486706   -0.069900   -0.008444    5.410000    0.020200  300.000000    2.230600    0.335000   -0.796600    0.948800  800.000000    0.389400    0.257800    0.478500    0.362900    0.750400    0.391800
    pga  -1.506500    0.165000   -0.255000   -0.165000    0.255000   16.087500    4.999300    1.060000    1.963600   -2.100000   -0.500000   50.000000    6.455100    3.095600    0.490800    0.035200    0.046200    0.000000    0.269500    0.483300    0.922800    0.120200    6.860700    0.000000   -0.453600   -0.007146   -0.006758    4.254200   -0.442757   -0.141700   -0.007010    0.102151    0.000000  300.000000    1.581700    0.759400   -0.684600    0.459000  800.000000    0.400000    0.260000    0.491200    0.376200    0.800000    0.452800
  0.010  -1.506500    0.165000   -0.255000   -0.165000    0.255000   16.087500    4.999300    1.060000    1.963600   -2.100000   -0.500000   50.000000    6.455100    3.095600    0.490800    0.035200    0.046200    0.000000    0.269500    0.483300    0.922800    0.120200    6.860700    0.000000   -0.453600   -0.007146   -0.006758    4.254200   -0.521000   -0.141700   -0.007010    0.102151    0.000000  300.000000    1.581700    0.759400   -0.684600    0.459000  800.000000    0.400000    0.260000    0.491200    0.376200    0.800000    0.452800
  0.020  -1.479800    0.165000   -0.255000   -0.165000    0.255000   15.711800    4.999300    1.060000    1.963600   -2.100000   -0.500000   50.000000    6.455100    3.096300    0.492500    0.035200    0.047200    0.000000    0.269500    1.214400    0.929600    0.121700    6.869700    0.000000   -0.453600   -0.007249   -0.006758    4.238600   -0.505500   -0.136400   -0.007279    0.108360    0.000000  300.000000    1.574000    0.760600   -0.668100    0.458000  800.000000    0.402600    0.263700    0.490400    0.376200    0.800000    0.455100
  0.030  -1.297200    0.165000   -0.255000   -0.165000    0.255000   15.881900    4.999300    1.060000    1.963600   -2.100000   -0.500000   50.000000    6.455100    3.097400    0.499200    0.035200    0.053300    0.000000    0.269500    1.642100    0.939600    0.119400    6.911300    0.000000   -0.453600   -0.007869   -0.006758    4.251900   -0.436800   -0.140300   -0.007354    0.119888    0.000000  300.000000    1.554400    0.764200   -0.631400    0.462000  800.000000    0.406300    0.268900    0.498800    0.384900    0.800000    0.457100
  0.050  -0.929200    0.165000   -0.255000   -0.165000    0.255000   17.645300    4.999300    1.060000    1.963600   -2.100000   -0.500000   50.000000    6.455100    3.101100    0.504800    0.035200    0.063900    0.000000    0.269500    2.181000    0.979400    0.117600    7.095900    0.000000   -0.453600   -0.008743   -0.006758    4.357800   -0.328750   -0.186200   -0.006467    0.148927    0.000000  300.000000    1.539100    0.773900   -0.545700    0.436000  800.000000    0.412400    0.277700    0.509600    0.395700    0.800000    0.471600
  0.075  -0.658000    0.165000   -0.254000   -0.165000    0.254000   20.177200    5.003100    1.060000    1.963600   -2.100000   -0.500000   50.000000    6.455100    3.109400    0.504800    0.035200    0.063000    0.000000    0.269500    2.608700    1.026000    0.117100    7.329800    0.000000   -0.453600   -0.009537   -0.006190    4.545500   -0.404121   -0.253800   -0.005734    0.190596    0.000000  300.000000    1.480400    0.795600   -0.468500    0.383000  800.000000    0.417900    0.285500    0.517900    0.404300    0.800000    0.502200
  0.100  -0.561300    0.165000   -0.253000   -0.165000    0.253000   19.999200    5.017200    1.060000    1.963600   -2.100000   -0.500000   50.000000    6.830500    3.238100    0.504800    0.035200    0.053200    0.000000    0.269500    2.912200    1.017700    0.114600    7.258800    0.000000   -0.453600   -0.009830   -0.005332    4.760300   -0.481007   -0.294300   -0.005604    0.230662    0.000000  300.000000    1.409400    0.793200   -0.498500    0.375000  800.000000    0.421900    0.291300    0.523600    0.410400    0.800000    0.523000
  0.150  -0.546200    0.165000   -0.250000   -0.165000    0.250000   16.624600    5.054700    1.060000    2.036200   -2.100000   -0.500000   50.000000    7.362100    3.430000    0.504500    0.035200    0.034500    0.000000    0.269500    3.339900    0.980100    0.110600    7.210900    0.000000   -0.453600   -0.009896   -0.003806    5.064400   -0.628393   -0.311300   -0.005845    0.266468    0.000000  300.000000    1.324100    0.743700   -0.645100    0.379000  800.000000    0.427500    0.299300    0.530800    0.419100    0.800000    0.530400
  0.200  -0.679800    0.165000   -0.244900   -0.165000    0.244900   13.701200    5.093900    1.060000    2.152100   -2.100000   -0.500000   50.000000    7.497200    3.514600    0.501600    0.035200    0.020200    0.000000    0.269500    3.643400    0.945900    0.120800    7.298800    0.000000   -0.444000   -0.009505   -0.002690    5.188000   -0.748519   -0.292700   -0.006141    0.255253    0.000000  300.000000    1.293100    0.692200   -0.765300    0.384000  800.000000    0.431300    0.304700    0.535100    0.425200    0.800000    0.531200
  0.250  -0.866300    0.165000   -0.238200   -0.165000    0.238200   11.266700    5.131500    1.060000    2.257400   -2.100000   -0.500000   50.000000    7.541600    3.574600    0.497100    0.035200    0.009000    0.000000    0.269500    3.878700    0.919600    0.120800    7.369100    0.000000   -0.353900   -0.008918   -0.002128    5.216400   -0.802441   -0.266200   -0.006439    0.231541    0.000000  300.000000    1.315000    0.657900   -0.846900    0.393000  800.000000    0.434100    0.308700    0.537700    0.429900    0.799900    0.530900
  0.300  -1.051400    0.165000   -0.231300   -0.165000    0.231300    9.190800    5.167000    1.060000    2.344000   -2.100000   -0.500000   50.000000    7.560000    3.623200    0.491900    0.035200   -0.000400    0.000000    0.269500    4.071100    0.882900    0.117500    6.878900    0.000000   -0.268800   -0.008251   -0.001812    5.195400   -0.739228   -0.240500   -0.006704    0.207277    0.001000  300.000000    1.351400    0.636200   -0.899900    0.408000  800.000000    0.436300    0.311900    0.539500    0.433800    0.799700    0.530700
  0.400  -1.379400    0.165000   -0.214600   -0.165000    0.214600    6.545900    5.231700    1.060000    2.470900   -2.100000   -0.500000   50.000000    7.573500    3.694500    0.480700    0.035200   -0.015500    0.000000    0.269500    4.374500    0.830200    0.106000    6.533400    0.000000   -0.179300   -0.007267   -0.001274    5.089900   -0.685160   -0.197500   -0.007125    0.165464    0.004000  300.000000    1.405100    0.604900   -0.961800    0.462000  800.000000    0.439600    0.316500    0.542200    0.439900    0.798800    0.531000
  0.500  -1.650800    0.165000   -0.197200   -0.165000    0.197200    5.230500    5.289300    1.060000    2.556700   -2.100000   -0.500000   50.000000    7.577800    3.740100    0.470700    0.035200   -0.027800    0.099100    0.269500    4.609900    0.788400    0.106100    6.526000    0.000000   -0.142800   -0.006492   -0.001074    4.785400   -0.547805   -0.163300   -0.007435    0.133828    0.010000  300.000000    1.440200    0.550700   -0.994500    0.524000  800.000000    0.441900    0.319900    0.543300    0.444600    0.796600    0.531300
  0.750  -2.151100    0.165000   -0.162000   -0.165000    0.162000    3.789600    5.410900    1.060000    2.681200   -2.100000   -0.500000   50.000000    7.580800    3.794100    0.457500    0.035200   -0.047700    0.198200    0.269500    5.037600    0.675400    0.100000    6.500000    0.000000   -0.113800   -0.005147   -0.001115    4.330400   -0.431431   -0.102800   -0.008120    0.085153    0.034000  300.000000    1.528000    0.358200   -1.022500    0.658000  800.000000    0.445900    0.325500    0.529400    0.453300    0.779200    0.530900
  1.000  -2.536500    0.165000   -0.140000   -0.165000    0.140000    3.302400    5.510600    1.060000    2.747400   -2.100000   -0.500000   50.000000    7.581400    3.814400    0.452200    0.035200   -0.055900    0.215400    0.269500    5.341100    0.619600    0.100000    6.500000    0.000000   -0.106200   -0.004277   -0.001197    4.166700   -0.421486   -0.069900   -0.008444    0.058595    0.067000  300.000000    1.652300    0.200300   -1.000200    0.780000  800.000000    0.448400    0.329100    0.510500    0.459400    0.750400    0.530200
  1.500  -3.068600    0.165000   -0.118400   -0.165000    0.118400    2.849800    5.670500    1.060000    2.816100   -2.100000   -0.500000   50.000000    7.581700    3.828400    0.450100    0.035200   -0.063000    0.215400    0.269500    5.768800    0.510100    0.100000    6.500000    0.000000   -0.102000   -0.002979   -0.001675    4.002900   -0.421486   -0.042500   -0.007707    0.031787    0.143000  300.000000    1.887200    0.035600   -0.924500    0.960000  800.000000    0.451500    0.333500    0.478300    0.468000    0.713600    0.527600
  2.000  -3.414800    0.164500   -0.110000   -0.164500    0.110000    2.541700    5.798100    1.060000    2.851400   -2.100000   -0.500000   50.000000    7.581800    3.833000    0.450000    0.035200   -0.066500    0.215400    0.269500    6.072300    0.391700    0.100000    6.500000    0.000000   -0.100900   -0.002301   -0.002349    3.894900   -0.421486   -0.030200   -0.004792    0.019716    0.203000  300.000000    2.134800    0.000000   -0.862600    1.110000  800.000000    0.453400    0.336300    0.468100    0.468100    0.703500    0.516700
  3.000  -3.901300    0.116800   -0.104000   -0.116800    0.104000    2.148800    5.998300    1.060000    2.887500   -2.100000   -0.500000   50.000000    7.581800    3.836100    0.450000    0.016000   -0.051600    0.215400    0.269500    6.500000    0.124400    0.100000    6.500000    0.000000   -0.100300   -0.001344   -0.003306    3.792800   -0.421486   -0.012900   -0.001828    0.009643    0.277000  300.000000    3.575200    0.000000   -0.788200    1.291000  800.000000    0.455800    0.339800    0.461700    0.461700    0.700600    0.491700
  4.000  -4.246600    0.073200   -0.102000   -0.073200    0.102000    1.895700    6.155200    1.060000    2.905800   -2.100000   -0.500000   50.000000    7.581800    3.836900    0.450000    0.006200   -0.044800    0.215400    0.269500    6.803500    0.008600    0.100000    6.500000    0.000000   -0.100100   -0.001084   -0.003566    3.744300   -0.421486   -0.001600   -0.001523    0.005379    0.309000  300.000000    3.864600    0.000000   -0.719500    1.387000  800.000000    0.457400    0.341900    0.457100    0.457100    0.700100    0.468200
  5.000  -4.514300    0.048400   -0.101000   -0.048400    0.101000    1.722800    6.285600    1.060000    2.916900   -2.100000   -0.500000   50.000000    7.581800    3.837600    0.450000    0.002900   -0.042400    0.215400    0.269500    7.038900    0.000000    0.100000    6.500000    0.000000   -0.100100   -0.001010   -0.003640    3.709000   -0.421486    0.000000   -0.001440    0.003223    0.321000  300.000000    3.729200    0.000000   -0.656000    1.433000  800.000000    0.458400    0.343500    0.453500    0.453500    0.700000    0.451700
  7.500  -5.000900    0.022000   -0.101000   -0.022000    0.101000    1.573700    6.542800    1.060000    2.932000   -2.100000   -0.500000   50.000000    7.581800    3.838000    0.450000    0.000700   -0.034800    0.215400    0.269500    7.466600    0.000000    0.100000    6.500000    0.000000   -0.100000   -0.000964   -0.003686    3.663200   -0.421486    0.000000   -0.001369    0.001134    0.329000  300.000000    2.376300    0.000000   -0.520200    1.460000  800.000000    0.460100    0.345900    0.447100    0.447100    0.700000    0.416700
 10.000  -5.346100    0.012400   -0.100000   -0.012400    0.100000    1.526500    6.741500    1.060000    2.939600   -2.100000   -0.500000   50.000000    7.581800    3.838000    0.450000    0.000300   -0.025300    0.215400    0.269500    7.770000    0.000000    0.100000    6.500000    0.000000   -0.100000   -0.000950   -0.003700    3.623000   -0.421486    0.000000   -0.001361    0.000515    0.330000  300.000000    1.767900    0.000000   -0.406800    1.464000  800.000000    0.461200    0.347400    0.442600    0.442600    0.700000    0.375500
""")


class AbrahamsonGulerce2020SInter_USGSPRVI(abrahamson_gulerce_2020.AbrahamsonGulerce2020SInter):
    """
    Abrahamson and Gulerce (2020) subduction interface ground-motion model with 
    USGS adjustments for Puerto Rico and the Virgin Islands.
    """

    COEFFS = CoeffsTable(sa_damping=5, table="""\
    IMT        c1i        vlin           b          a1          a2          a6          a7          a8         a10         a11         a12         a13         a14         a16         a17         a18         a19         a20         a21         a22         a23         a24         a25         a26         a27         a28         a29         a30         a31         a32         a33         a34         a35         a36         a37         a39         a41  USA-AK_Adj     CAS_Adj          d1          d2        rhoW        rhoB  phi_s2s_g1  phi_s2s_g2  phi_s2s_g3          e1          e2          e3
    pga   8.200000  865.100000   -1.186000    4.596000   -1.450000   -0.004300    3.210000    0.044000    3.210000    0.007000    1.273292    0.000000   -0.460000    0.090000    0.000000   -0.200000    0.000000    0.000000    0.040000    0.040000    0.000000    0.001500    0.000700    0.003600   -0.000400    0.002500    0.000600    0.003300    3.778300    3.346800    3.802500    5.036100    4.627200    4.804400    3.566900    0.000000   -0.029000    0.487000    0.828000    0.325000    0.137000    1.000000    1.000000    0.396000    0.396000    0.545000    0.550000   -0.270000    0.050000
  0.010   8.200000  865.100000   -1.186000    4.596000   -1.450000   -0.004300    3.210000    0.044000    3.210000    0.007000    0.900000    0.000000   -0.460000    0.090000    0.000000   -0.200000    0.000000    0.000000    0.040000    0.040000    0.000000    0.001500    0.000700    0.003600   -0.000400    0.002500    0.000600    0.003300    3.778300    3.346800    3.802500    5.036100    4.627200    4.804400    3.566900    0.000000   -0.029000    0.487000    0.828000    0.325000    0.137000    1.000000    1.000000    0.396000    0.396000    0.545000    0.550000   -0.270000    0.050000
  0.020   8.200000  865.100000   -1.219000    4.678000   -1.450000   -0.004300    3.210000    0.044000    3.210000    0.007000    1.008000    0.000000   -0.460000    0.090000    0.000000   -0.200000    0.000000    0.000000    0.040000    0.040000    0.000000    0.001500    0.000600    0.003600   -0.000500    0.002500    0.000500    0.003300    3.828100    3.440100    3.905300    5.137500    4.695800    4.894300    3.642500    0.000000   -0.024000    0.519000    0.825000    0.325000    0.137000    0.990000    0.990000    0.396000    0.396000    0.545000    0.550000   -0.270000    0.050000
  0.030   8.200000  907.800000   -1.273000    4.773000   -1.450000   -0.004400    3.210000    0.044000    3.210000    0.007000    1.127000    0.000000   -0.460000    0.090000    0.000000   -0.200000    0.000000    0.000000    0.040000    0.040000    0.000000    0.001500    0.000600    0.003700   -0.000700    0.002500    0.000500    0.003400    3.893300    3.508700    4.018900    5.269900    4.780900    5.002800    3.706300    0.000000   -0.034000    0.543000    0.834000    0.325000    0.137000    0.990000    0.990000    0.396000    0.396000    0.545000    0.550000   -0.270000    0.050000
  0.050   8.200000 1053.500000   -1.346000    5.029000   -1.450000   -0.004600    3.210000    0.044000    3.210000    0.007000    1.006922    0.000000   -0.460000    0.090000    0.000000   -0.200000    0.000000    0.000000    0.040000    0.040000    0.000000    0.001100    0.000600    0.003900   -0.000900    0.002600    0.000400    0.003600    4.286700    3.655300    4.295200    5.615700    5.021100    5.281900    3.918400    0.000000   -0.061000    0.435000    0.895000    0.325000    0.137000    0.970000    0.985000    0.396000    0.467000    0.644000    0.560000   -0.270000    0.050000
  0.075   8.200000 1085.700000   -1.471000    5.334000   -1.450000   -0.004700    3.210000    0.044000    3.210000    0.007000    1.591214    0.000000   -0.460000    0.090000    0.000000   -0.200000    0.000000    0.000000    0.060000    0.060000    0.000000    0.001100    0.000400    0.003900   -0.000900    0.002600    0.000300    0.003700    4.594000    3.979900    4.546400    6.020400    5.347400    5.612300    4.220700    0.000000   -0.076000    0.410000    0.863000    0.325000    0.137000    0.950000    0.980000    0.396000    0.516000    0.713000    0.580000   -0.270000    0.050000
  0.100   8.200000 1032.500000   -1.624000    5.455000   -1.450000   -0.004800    3.210000    0.044000    3.210000    0.007000    1.815783    0.000000   -0.460000    0.090000    0.000000   -0.200000    0.000000    0.000000    0.100000    0.100000    0.000000    0.001200    0.000300    0.003900   -0.000800    0.002600    0.000300    0.003800    4.707700    4.131200    4.613800    6.162500    5.506500    5.766800    4.353600    0.000000   -0.049000    0.397000    0.842000    0.325000    0.137000    0.920000    0.970000    0.396000    0.516000    0.713000    0.590000   -0.270000    0.050000
  0.150   8.200000  877.600000   -1.931000    5.376000   -1.425000   -0.004700    3.210000    0.044000    3.210000    0.007000    1.884416    0.000000   -0.460000    0.090000    0.000000   -0.186000    0.000000   -0.055000    0.135000    0.135000    0.069000    0.001300   -0.000200    0.003700   -0.000900    0.002200    0.000100    0.003700    4.606500    4.273700    4.529000    5.961400    5.518000    5.731300    4.366400    0.000000   -0.026000    0.428000    0.737000    0.325000    0.137000    0.900000    0.960000    0.396000    0.516000    0.647000    0.590000   -0.270000    0.050000
  0.200   8.200000  748.200000   -2.188000    4.936000   -1.335000   -0.004500    3.210000    0.043000    3.210000    0.006200    2.211014    0.000000   -0.460000    0.084000    0.000000   -0.150000    0.000000   -0.105000    0.170000    0.170000    0.140000    0.001300   -0.000700    0.003100   -0.001000    0.001800   -0.000100    0.003500    4.186600    3.965000    4.165600    5.392000    5.166800    5.294300    4.016900    0.000000   -0.011000    0.442000    0.746000    0.325000    0.137000    0.870000    0.940000    0.396000    0.516000    0.596000    0.570000   -0.270000    0.050000
  0.250   8.200000  654.300000   -2.381000    4.636000   -1.275000   -0.004300    3.210000    0.042000    3.210000    0.005600    2.393009    0.000000   -0.460000    0.080000    0.000000   -0.140000    0.000000   -0.134000    0.170000    0.170000    0.164000    0.001300   -0.000900    0.002700   -0.001100    0.001600   -0.000300    0.003300    3.851500    3.682100    3.914700    5.011700    4.874400    5.005800    3.759000    0.101000   -0.009000    0.494000    0.796000    0.325000    0.137000    0.840000    0.930000    0.396000    0.501000    0.539000    0.530000   -0.224000    0.043000
  0.300   8.200000  587.100000   -2.518000    4.423000   -1.231000   -0.004200    3.210000    0.041000    3.210000    0.005100    2.586175   -0.002000   -0.460000    0.078000    0.000000   -0.120000    0.000000   -0.150000    0.170000    0.170000    0.190000    0.001400   -0.001000    0.002000   -0.000900    0.001400   -0.000200    0.003200    3.578300    3.541500    3.784600    4.705700    4.654400    4.758800    3.591400    0.184000    0.005000    0.565000    0.782000    0.325000    0.137000    0.820000    0.910000    0.396000    0.488000    0.488000    0.490000   -0.186000    0.037000
  0.400   8.200000  503.000000   -2.657000    4.124000   -1.165000   -0.004000    3.210000    0.040000    3.210000    0.004300    2.548691   -0.007000   -0.470000    0.075000    0.000000   -0.100000    0.000000   -0.150000    0.170000    0.170000    0.206000    0.001500   -0.001000    0.001300   -0.000700    0.001100    0.000000    0.003000    3.249300    3.325600    3.570200    4.289600    4.366000    4.378900    3.370400    0.315000    0.040000    0.625000    0.768000    0.325000    0.137000    0.740000    0.860000    0.396000    0.468000    0.468000    0.425000   -0.126000    0.028000
  0.500   8.200000  456.600000   -2.669000    3.838000   -1.115000   -0.003700    3.210000    0.039000    3.210000    0.003700    2.323753   -0.011000   -0.480000    0.072000    0.000000   -0.080000    0.000000   -0.150000    0.170000    0.170000    0.220000    0.001500   -0.001100    0.000900   -0.000700    0.000800    0.000200    0.002700    2.981800    3.133400    3.355200    3.932200    4.077900    4.039400    3.156400    0.416000    0.097000    0.634000    0.728000    0.325000    0.137000    0.660000    0.800000    0.396000    0.451000    0.451000    0.375000   -0.079000    0.022000
  0.750   8.150000  410.500000   -2.401000    3.152000   -1.020000   -0.003200    3.210000    0.037000    3.210000    0.002700    1.490987   -0.021000   -0.500000    0.067000    0.000000   -0.047000    0.000000   -0.150000    0.170000    0.170000    0.217000    0.001400   -0.001100    0.000300   -0.000700    0.000400    0.000200    0.002200    2.478000    2.538000    2.657200    3.178500    3.439100    3.293000    2.655600    0.600000    0.197000    0.497000    0.685000    0.325000    0.137000    0.500000    0.730000    0.396000    0.420000    0.420000    0.300000    0.005000    0.009000
  1.000   8.100000  400.000000   -1.955000    2.544000   -0.950000   -0.002900    3.210000    0.035000    3.210000    0.001900    0.831327   -0.028000   -0.510000    0.063000    0.000000   -0.035000    0.000000   -0.150000    0.170000    0.170000    0.185000    0.001300   -0.000800    0.000100   -0.000800    0.000200    0.000100    0.001900    1.925200    1.962600    2.145900    2.572200    2.805600    2.647500    2.066700    0.731000    0.269000    0.469000    0.642000    0.325000    0.137000    0.410000    0.690000    0.396000    0.396000    0.396000    0.240000    0.065000    0.000000
  1.500   8.050000  400.000000   -1.025000    1.636000   -0.860000   -0.002600    3.210000    0.034000    3.210000    0.000800   -0.206390   -0.041000   -0.520000    0.059000    0.000000   -0.018000    0.000000   -0.130000    0.170000    0.170000    0.083000    0.001400   -0.000400   -0.000100   -0.000800    0.000100    0.000000    0.001600    0.992400    1.356800    1.349900    1.649900    1.854600    1.684200    1.331600    0.748000    0.347000    0.509000    0.325000    0.312000    0.113000    0.330000    0.620000    0.379000    0.379000    0.379000    0.230000    0.065000    0.000000
  2.000   8.000000  400.000000   -0.299000    1.076000   -0.820000   -0.002400    3.210000    0.032000    3.210000    0.000000   -0.884968   -0.050000   -0.530000    0.059000    0.000000   -0.010000    0.000000   -0.110000    0.170000    0.170000    0.045000    0.001500    0.000200    0.000000   -0.000700    0.000200    0.000000    0.001400    0.467600    0.818000    0.814800    1.065800    1.302000    1.100200    0.760700    0.761000    0.384000    0.478000    0.257000    0.302000    0.096000    0.300000    0.560000    0.366000    0.366000    0.366000    0.230000    0.065000    0.000000
  3.000   7.900000  400.000000    0.000000    0.424000   -0.793000   -0.002100    3.130000    0.030000    3.130000    0.000000   -0.843900   -0.065000   -0.540000    0.059000    0.000000    0.000000    0.000000   -0.085000    0.170000    0.170000    0.035000    0.001400    0.000700    0.000300   -0.000700    0.000400   -0.000200    0.001100   -0.139100    0.104600    0.104600    0.388200    0.595800    0.412600    0.168800    0.778000    0.404000    0.470000    0.296000    0.289000    0.072000    0.250000    0.495000    0.348000    0.348000    0.348000    0.240000    0.065000    0.000000
  4.000   7.850000  400.000000    0.000000    0.093000   -0.793000   -0.002000    2.985000    0.029000    2.985000    0.000000   -0.741945   -0.077000   -0.540000    0.050000    0.000000    0.000000    0.000000   -0.073000    0.170000    0.170000    0.053000    0.001400    0.001000    0.000700   -0.000600    0.000600   -0.000200    0.001000   -0.303000   -0.159700   -0.232400    0.016400    0.352200    0.009700   -0.032300    0.790000    0.397000    0.336000    0.232000    0.280000    0.055000    0.220000    0.430000    0.335000    0.335000    0.335000    0.270000    0.065000    0.000000
  5.000   7.800000  400.000000    0.000000   -0.145000   -0.793000   -0.002000    2.818000    0.028000    2.818000    0.000000   -0.534875   -0.088000   -0.540000    0.043000    0.000000    0.000000    0.000000   -0.065000    0.170000    0.170000    0.072000    0.001400    0.001300    0.001400   -0.000400    0.000800   -0.000100    0.001000   -0.409400   -0.206300   -0.572200   -0.280200    0.187400   -0.271500   -0.151600    0.799000    0.378000    0.228000    0.034000    0.273000    0.041000    0.190000    0.400000    0.324000    0.324000    0.324000    0.300000    0.065000    0.000000
  7.500   7.800000  400.000000    0.000000   -0.556000   -0.793000   -0.002000    2.515000    0.026000    2.515000    0.000000    0.121813   -0.110000   -0.540000    0.032000    0.000000    0.000000    0.000000   -0.055000    0.170000    0.170000    0.115000    0.001400    0.001700    0.001500   -0.000200    0.001400    0.000100    0.001000   -0.620900   -0.422300   -1.177300   -0.756600   -0.331600   -0.682200   -0.333800    0.817000    0.333000    0.051000   -0.178000    0.259000    0.017000    0.140000    0.320000    0.301000    0.301000    0.301000    0.350000    0.065000    0.000000
 10.000   7.800000  400.000000    0.000000   -0.860000   -0.793000   -0.002000    2.300000    0.025000    2.300000    0.000000    0.723583   -0.127000   -0.540000    0.024000    0.000000    0.000000    0.000000   -0.045000    0.170000    0.170000    0.151000    0.001400    0.001700    0.001500   -0.000100    0.001700    0.000200    0.001000   -0.622100   -0.590900   -1.407000   -1.087000   -0.678300   -0.917300   -0.544100    0.829000    0.281000   -0.251000   -0.313000    0.250000    0.000000    0.100000    0.280000    0.286000    0.286000    0.286000    0.350000    0.065000    0.000000
 """)


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
    COEFFS = CoeffsTable(sa_damping=5, table="""\
    IMT       dm_b   mu_c_1_if mu_c_1_slab      c_2_if    c_2_slab         c_3      c_4_if    c_4_slab         c_5      mu_c_6     mu_c_6b      mu_c_7      c_9_if    c_9_slab       c_6xc c_1_if_reg_Al c_1_if_reg_Ca c_1_if_reg_CAM c_1_if_reg_Ja c_1_if_reg_NZ c_1_if_reg_SA c_1_if_reg_Tw c_1_slab_reg_Al c_1_slab_reg_Ca c_1_slab_reg_CAM c_1_slab_reg_Ja c_1_slab_reg_NZ c_1_slab_reg_SA c_1_slab_reg_Tw  c_7_reg_Al  c_7_reg_Ca c_7_reg_CAM  c_7_reg_Ja  c_7_reg_NZ  c_7_reg_SA  c_7_reg_Tw c_6_x1_reg_Al c_6_x1_reg_Ca c_6_x1_reg_CAM c_6_x1_reg_Ja c_6_x1_reg_NZ c_6_x1_reg_SA c_6_x1_reg_Tw c_6_x2_reg_Al c_6_x2_reg_Ca c_6_x2_reg_CAM c_6_x2_reg_Ja c_6_x2_reg_NZ c_6_x2_reg_SA c_6_x2_reg_Tw c_6_x3_reg_Al c_6_x3_reg_Ca c_6_x3_reg_CAM c_6_x3_reg_Ja c_6_x3_reg_NZ c_6_x3_reg_SA c_6_x3_reg_Tw c_6_1_reg_Al c_6_1_reg_Ca c_6_1_reg_CAM c_6_1_reg_Ja c_6_1_reg_NZ c_6_1_reg_SA c_6_1_reg_Tw c_6_2_reg_Al c_6_2_reg_Ca c_6_2_reg_CAM c_6_2_reg_Ja c_6_2_reg_NZ c_6_2_reg_SA c_6_2_reg_Tw c_6_3_reg_Al c_6_3_reg_Ca c_6_3_reg_CAM c_6_3_reg_Ja c_6_3_reg_NZ c_6_3_reg_SA c_6_3_reg_Tw     dz_b_if   dz_b_slab     c_nft_1     c_nft_2     c_11_Ca     c_12_Ca theta_11_Sea     c_11_Ja     c_12_Ja     c_11_NZ     c_12_NZ     c_11_Tw     c_12_Tw          k1          k2         phi         tau
    pgv  -0.200000    6.433543    9.435733   -2.217436   -2.669506    0.127884    1.014483    1.307578    0.123716   -0.001348   -0.002342    1.129366    0.016151    0.015521   -0.335150    6.371803    6.433418    6.411284    6.505276    6.460452    6.439995    6.434487    9.528993    9.441571    9.408429    9.561717    9.442427    9.366311    9.337388    1.331446    1.558877    1.851774    1.539524    1.786085    2.098709    1.672486   -0.002803   -0.002615   -0.002561   -0.004159   -0.002270   -0.001328   -0.002372   -0.002712   -0.002440   -0.000595   -0.001180   -0.002238   -0.001283   -0.002401   -0.002115   -0.002364   -0.002266   -0.000385   -0.002540   -0.002591   -0.002308   -0.002247   -0.002348   -0.002551   -0.002221   -0.002570   -0.002929   -0.002376   -0.001592   -0.001310   -0.000983   -0.001961   -0.000550   -0.001124   -0.001855   -0.002654   -0.002517   -0.002447   -0.004161   -0.002220   -0.002095   -0.002444   12.566095  -13.359066    0.936379    0.210943   -0.012278    0.118442    0.119977   -0.016288    0.101240    0.007889   -0.011142    0.011641    0.078625  400.000000   -1.955000    0.511486    0.450985
    pga   0.000000    3.715307    4.789080   -2.460896   -2.438099    0.103931    0.952187    1.105012    0.128151   -0.002635   -0.005254    1.467510    0.025481    0.022385   -0.604166    3.356030    3.600305    3.696421    4.008987    3.943338    4.051210    3.446871    4.548715    4.486160    5.041801    5.360941    4.822427    5.079649    4.391530    0.560135    0.662527    1.147447    0.869726    1.005783    1.131304    0.933688   -0.005574   -0.005541   -0.006286   -0.008737   -0.005012   -0.003629   -0.004813   -0.006505   -0.006748   -0.000543   -0.002042   -0.004534   -0.002734   -0.005830   -0.005413   -0.005270   -0.005316   -0.005587   -0.005359   -0.005335   -0.005177   -0.004745   -0.004892   -0.005721   -0.005544   -0.005702   -0.007038   -0.005131   -0.002900   -0.003789   -0.001379   -0.002812   -0.001984   -0.002095   -0.003047   -0.005178   -0.004983   -0.005920   -0.008307   -0.005361   -0.005848   -0.004710   16.816058  -15.841253    0.894589    0.198851   -0.033754    0.020801   -0.126369   -0.000553   -0.027347   -0.029315   -0.083453    0.013177    0.047719  865.000000   -1.186000    0.595755    0.488745
  0.010   0.000000    3.575653    4.559572   -2.428917   -2.395028    0.104318    0.945374    1.108607    0.129146   -0.002699   -0.005284    0.892438    0.025058    0.021964   -0.591998    3.195018    3.480335    3.521664    3.786328    3.810325    3.935535    3.311994    4.307772    4.309174    4.809797    5.048204    4.653637    4.840731    4.203531    0.571021    0.667083    1.109681    0.870923    1.012432    1.147077    0.910860   -0.005973   -0.005746   -0.006377   -0.008752   -0.005025   -0.003597   -0.004965   -0.006405   -0.006703   -0.000430   -0.002050   -0.004560   -0.002729   -0.005885   -0.005306   -0.005312   -0.005427   -0.005492   -0.005333   -0.005296   -0.005372   -0.004754   -0.005063   -0.005768   -0.005460   -0.005706   -0.006884   -0.005365   -0.002918   -0.003919   -0.001325   -0.002916   -0.001939   -0.002072   -0.003128   -0.005217   -0.005212   -0.005897   -0.008358   -0.005520   -0.005771   -0.004758   16.823813  -16.125061    0.896424    0.199092   -0.034374    0.020404   -0.126723    0.000253   -0.028776   -0.028300   -0.083199    0.012099    0.046885  865.000000   -1.186000    0.599596    0.494230
  0.020   0.000000    3.716489    4.808207   -2.459748   -2.443224    0.104711    0.958824    1.116236    0.132407   -0.002614   -0.005324    0.930261    0.027178    0.023277   -0.619772    3.381819    3.601502    3.684087    3.966415    3.912658    4.057270    3.436771    4.585991    4.492272    5.138580    5.356058    4.860414    5.154176    4.388375    0.607154    0.707218    1.161897    0.912944    1.056041    1.208209    0.953422   -0.005949   -0.005743   -0.006273   -0.008641   -0.005121   -0.003612   -0.004936   -0.006443   -0.006944   -0.000439   -0.001984   -0.004702   -0.002811   -0.006108   -0.005316   -0.005377   -0.005471   -0.005435   -0.005328   -0.005394   -0.005463   -0.004839   -0.005007   -0.005958   -0.005698   -0.005746   -0.007151   -0.005306   -0.002925   -0.003788   -0.001329   -0.002889   -0.001895   -0.002034   -0.002916   -0.005238   -0.005291   -0.005955   -0.008231   -0.005457   -0.005810   -0.004865   16.662189  -16.284032    0.897168    0.199151   -0.034662    0.018233   -0.128641   -0.000199   -0.030457   -0.028718   -0.082966    0.011978    0.047268  865.000000   -1.219000    0.592415    0.505458
  0.030   0.000000    4.019204    5.167198   -2.513834   -2.496445    0.104531    0.973096    1.109268    0.134843   -0.002548   -0.005442    1.012753    0.029344    0.024539   -0.654993    3.700438    3.881528    4.009647    4.346256    4.214684    4.354312    3.698231    4.947996    4.780066    5.547288    5.823779    5.176631    5.577609    4.666984    0.707423    0.782766    1.247419    1.011507    1.141593    1.276473    1.033921   -0.005875   -0.005766   -0.006266   -0.008825   -0.005277   -0.003726   -0.004932   -0.006591   -0.007211   -0.000515   -0.001952   -0.004851   -0.002909   -0.006340   -0.005433   -0.005498   -0.005577   -0.005582   -0.005449   -0.005570   -0.005546   -0.004967   -0.005036   -0.006164   -0.006002   -0.005867   -0.007533   -0.005298   -0.002937   -0.003683   -0.001376   -0.002837   -0.001911   -0.002046   -0.002688   -0.005300   -0.005346   -0.006091   -0.008232   -0.005537   -0.005971   -0.004967   17.181661  -16.314490    0.896405    0.198910   -0.032014    0.014046   -0.132861    0.001563   -0.036268   -0.032177   -0.083557    0.012910    0.047962  908.000000   -1.273000    0.604430    0.511668
  0.050   0.000000    4.544769    5.640303   -2.589450   -2.541067    0.103376    0.991235    1.090513    0.137301   -0.002550   -0.005676    1.085337    0.031911    0.025760   -0.692566    4.217633    4.387813    4.549221    4.964188    4.772027    4.888842    4.161202    5.404896    5.191630    6.041135    6.424630    5.618999    6.114167    5.054938    0.969922    0.959354    1.368368    1.257664    1.296724    1.367020    1.187023   -0.005885   -0.005848   -0.006436   -0.009375   -0.005584   -0.003981   -0.005016   -0.006858   -0.007564   -0.000688   -0.001973   -0.005075   -0.003081   -0.006676   -0.005678   -0.005733   -0.005815   -0.006002   -0.005712   -0.005880   -0.005674   -0.005218   -0.005196   -0.006476   -0.006429   -0.006119   -0.008093   -0.005396   -0.003006   -0.003680   -0.001487   -0.002794   -0.002021   -0.002131   -0.002454   -0.005498   -0.005496   -0.006320   -0.008430   -0.005797   -0.006278   -0.005125   18.320745  -16.300176    0.893480    0.198210   -0.028009   -0.016394   -0.179041    0.009631   -0.057726   -0.044734   -0.081383    0.010570    0.032748 1054.000000   -1.346000    0.638019    0.515793
  0.075   0.000000    4.920943    5.848238   -2.619896   -2.525556    0.101546    1.000331    1.078532    0.137915   -0.002673   -0.005933    1.642738    0.032823    0.025792   -0.695463    4.577888    4.768797    4.916525    5.361936    5.180893    5.274643    4.512720    5.599347    5.409688    6.214869    6.654076    5.835590    6.322656    5.254963    1.228310    1.183367    1.494059    1.490283    1.462358    1.469249    1.357870   -0.006040   -0.005971   -0.006747   -0.009926   -0.005871   -0.004229   -0.005202   -0.007084   -0.007762   -0.000874   -0.002086   -0.005267   -0.003241   -0.006915   -0.005932   -0.005995   -0.006062   -0.006452   -0.006006   -0.006149   -0.005850   -0.005463   -0.005449   -0.006707   -0.006712   -0.006362   -0.008416   -0.005620   -0.003105   -0.003884   -0.001609   -0.002831   -0.002194   -0.002263   -0.002470   -0.005739   -0.005721   -0.006547   -0.008744   -0.006114   -0.006531   -0.005310   18.923699  -16.240071    0.889434    0.197375   -0.033582   -0.060272   -0.261658    0.013763   -0.076256   -0.043511   -0.097524    0.013509    0.040802 1086.000000   -1.471000    0.666120    0.514873
  0.100   0.000000    5.073490    5.830854   -2.608323   -2.477879    0.099788    1.001913    1.078729    0.137331   -0.002821   -0.006073    1.944628    0.032448    0.025081   -0.676483    4.726850    4.937742    5.053960    5.488464    5.346632    5.425393    4.676007    5.582709    5.437211    6.146774    6.587999    5.839812    6.268244    5.277205    1.387116    1.382118    1.643536    1.630929    1.625128    1.606936    1.520788   -0.006227   -0.006075   -0.007025   -0.010221   -0.006029   -0.004379   -0.005355   -0.007161   -0.007794   -0.001024   -0.002206   -0.005363   -0.003348   -0.007003   -0.006108   -0.006154   -0.006218   -0.006751   -0.006176   -0.006295   -0.005944   -0.005625   -0.005647   -0.006780   -0.006798   -0.006486   -0.008488   -0.005784   -0.003186   -0.004139   -0.001703   -0.002886   -0.002347   -0.002367   -0.002642   -0.005917   -0.005859   -0.006677   -0.009001   -0.006306   -0.006643   -0.005425   18.758649  -16.147277    0.885757    0.196698   -0.025228   -0.077067   -0.276774    0.010235   -0.068037   -0.047969   -0.124030    0.010252    0.045775 1032.000000   -1.624000    0.678369    0.511975
  0.150   0.000000    5.025991    5.531067   -2.531738   -2.361283    0.096886    0.997174    1.099428    0.134973   -0.003085   -0.006177    2.153108    0.030226    0.022996   -0.620266    4.700169    4.927519    4.981229    5.350477    5.289008    5.350817    4.689842    5.308804    5.242301    5.751217    6.135933    5.579031    5.866442    5.092472    1.550575    1.682451    1.973241    1.786976    1.924312    1.945485    1.816594   -0.006508   -0.006200   -0.007406   -0.010371   -0.006160   -0.004489   -0.005541   -0.007131   -0.007655   -0.001237   -0.002384   -0.005422   -0.003465   -0.007001   -0.006223   -0.006283   -0.006335   -0.007034   -0.006317   -0.006365   -0.006022   -0.005773   -0.005872   -0.006752   -0.006729   -0.006541   -0.008283   -0.005971   -0.003281   -0.004590   -0.001826   -0.003045   -0.002575   -0.002507   -0.003112   -0.006151   -0.006031   -0.006786   -0.009308   -0.006455   -0.006640   -0.005557   17.286575  -15.877666    0.879903    0.195765   -0.032857   -0.051386   -0.246680   -0.009498   -0.050344   -0.039382   -0.118888    0.015652    0.040261  878.000000   -1.931000    0.677740    0.505515
  0.200   0.000000    4.769661    5.132678   -2.434299   -2.252963    0.094745    0.990942    1.131547    0.132355   -0.003273   -0.006152    2.443119    0.027520    0.020848   -0.563794    4.481015    4.702099    4.710481    5.005995    5.002107    5.054294    4.504351    4.947768    4.932355    5.279849    5.591939    5.203980    5.371817    4.803260    1.651087    1.889591    2.261113    1.896749    2.157453    2.274056    2.057558   -0.006651   -0.006196   -0.007598   -0.010205   -0.006160   -0.004465   -0.005625   -0.006969   -0.007424   -0.001376   -0.002505   -0.005392   -0.003503   -0.006876   -0.006216   -0.006276   -0.006319   -0.007089   -0.006304   -0.006298   -0.006017   -0.005815   -0.005951   -0.006597   -0.006540   -0.006456   -0.007916   -0.006039   -0.003330   -0.004916   -0.001892   -0.003173   -0.002722   -0.002565   -0.003563   -0.006264   -0.006081   -0.006773   -0.009406   -0.006447   -0.006484   -0.005593   15.321677  -15.526638    0.875691    0.195220   -0.034588   -0.016656   -0.174198   -0.013802   -0.035031   -0.025001   -0.089642    0.014436    0.042213  748.000000   -2.188000    0.664654    0.500076
  0.250   0.000000    4.441348    4.737977   -2.338216   -2.162150    0.093177    0.986669    1.166600    0.129945   -0.003413   -0.006065    2.512998    0.024917    0.018907   -0.513766    4.192964    4.396801    4.375634    4.606384    4.638720    4.683924    4.239805    4.590721    4.604524    4.832426    5.079309    4.820203    4.898130    4.497810    1.742163    2.041945    2.476116    1.995613    2.324779    2.542924    2.242567   -0.006688   -0.006126   -0.007674   -0.009923   -0.006086   -0.004392   -0.005641   -0.006760   -0.007178   -0.001477   -0.002617   -0.005338   -0.003512   -0.006707   -0.006158   -0.006199   -0.006240   -0.007029   -0.006216   -0.006172   -0.005958   -0.005804   -0.005942   -0.006413   -0.006337   -0.006327   -0.007514   -0.006006   -0.003360   -0.005140   -0.001937   -0.003288   -0.002818   -0.002591   -0.003945   -0.006273   -0.006060   -0.006693   -0.009368   -0.006349   -0.006279   -0.005597   13.374937  -15.126413    0.872636    0.194916   -0.031240    0.035524   -0.087996   -0.008879   -0.022592   -0.022149   -0.118750    0.021984    0.050554  654.000000   -2.381000    0.649528    0.495846
  0.300   0.000000    4.095960    4.374731   -2.250205   -2.088032    0.092032    0.985056    1.201258    0.127846   -0.003508   -0.005954    2.633840    0.022575    0.017207   -0.470368    3.885682    4.068026    4.028588    4.206322    4.259728    4.299053    3.947874    4.261667    4.290405    4.431418    4.624839    4.459910    4.472629    4.204567    1.826956    2.156155    2.621751    2.081212    2.438066    2.742513    2.376440   -0.006654   -0.006032   -0.007688   -0.009600   -0.005988   -0.004291   -0.005632   -0.006534   -0.006943   -0.001559   -0.002684   -0.005270   -0.003502   -0.006534   -0.006056   -0.006096   -0.006135   -0.006912   -0.006086   -0.006020   -0.005874   -0.005759   -0.005905   -0.006212   -0.006134   -0.006177   -0.007117   -0.005957   -0.003363   -0.005276   -0.001960   -0.003360   -0.002882   -0.002588   -0.004250   -0.006235   -0.005989   -0.006589   -0.009257   -0.006233   -0.006069   -0.005578   11.613497  -14.699405    0.870399    0.194766   -0.027222    0.051800   -0.046549   -0.010823   -0.005372   -0.010104   -0.084008    0.019670    0.053315  587.000000   -2.518000    0.635787    0.492609
  0.400   0.000000    3.433449    3.757220   -2.102935   -1.979317    0.090584    0.989150    1.265049    0.124505   -0.003620   -0.005710    2.547019    0.018723    0.014446   -0.399165    3.287036    3.425683    3.370174    3.472405    3.541018    3.570140    3.364051    3.699186    3.734440    3.767458    3.885418    3.835827    3.769548    3.682761    1.949962    2.287970    2.748279    2.186512    2.540737    2.955972    2.511423   -0.006485   -0.005821   -0.007570   -0.008950   -0.005756   -0.004090   -0.005518   -0.006102   -0.006504   -0.001669   -0.002778   -0.005106   -0.003457   -0.006192   -0.005843   -0.005840   -0.005895   -0.006622   -0.005827   -0.005734   -0.005692   -0.005603   -0.005732   -0.005835   -0.005769   -0.005857   -0.006400   -0.005763   -0.003363   -0.005410   -0.001984   -0.003454   -0.002936   -0.002556   -0.004668   -0.006101   -0.005823   -0.006352   -0.008915   -0.005927   -0.005656   -0.005470    8.768200  -13.818273    0.867543    0.194726   -0.045715    0.110659    0.023018   -0.013931    0.030155    0.007677   -0.045938    0.016889    0.051926  503.000000   -2.657000    0.615315    0.488215
  0.500   0.000000    2.846435    3.265244   -1.989991   -1.907871    0.089848    1.000680    1.320126    0.122092   -0.003628   -0.005454    2.379608    0.015823    0.012345   -0.342272    2.747332    2.848400    2.791079    2.847101    2.912944    2.933256    2.827443    3.246231    3.274078    3.251794    3.324407    3.330662    3.227339    3.246846    1.984421    2.302688    2.718123    2.187921    2.519785    2.980301    2.502296   -0.006235   -0.005580   -0.007357   -0.008356   -0.005503   -0.003889   -0.005353   -0.005720   -0.006120   -0.001736   -0.002829   -0.004930   -0.003384   -0.005877   -0.005591   -0.005573   -0.005633   -0.006313   -0.005552   -0.005429   -0.005466   -0.005416   -0.005519   -0.005510   -0.005469   -0.005559   -0.005797   -0.005533   -0.003303   -0.005393   -0.001976   -0.003482   -0.002924   -0.002492   -0.004878   -0.005888   -0.005601   -0.006063   -0.008492   -0.005621   -0.005298   -0.005319    6.728119  -12.947979    0.866017    0.194876   -0.031600    0.129282    0.099229   -0.019981    0.074733    0.016126   -0.019815    0.018284    0.065895  457.000000   -2.669000    0.603263    0.485607
  0.750   0.000000    1.701640    2.398837   -1.811385   -1.818966    0.089518    1.047321    1.426015    0.118614   -0.003470   -0.004892    1.717941    0.011352    0.008821   -0.233915    1.669392    1.708937    1.666433    1.673875    1.712003    1.715470    1.739913    2.431124    2.429917    2.368039    2.394375    2.430572    2.313242    2.434938    1.732147    1.969236    2.279571    1.837323    2.140887    2.590645    2.089380   -0.005576   -0.005020   -0.006700   -0.007178   -0.004932   -0.003436   -0.004928   -0.004963   -0.005355   -0.001822   -0.002874   -0.004510   -0.003184   -0.005217   -0.005027   -0.004974   -0.005049   -0.005618   -0.004933   -0.004801   -0.004962   -0.004947   -0.005005   -0.004836   -0.004887   -0.004918   -0.004647   -0.004990   -0.003076   -0.005049   -0.001916   -0.003411   -0.002787   -0.002311   -0.004934   -0.005332   -0.005063   -0.005419   -0.007421   -0.004987   -0.004591   -0.004903    3.964760  -10.953227    0.864983    0.195520   -0.032779    0.184908    0.209579   -0.024187    0.141359    0.012059    0.004011    0.006109    0.085523  410.000000   -2.401000    0.594773    0.482859
  1.000   0.000000    0.890492    1.829616   -1.719993   -1.791481    0.090096    1.103814    1.500926    0.117157   -0.003218   -0.004405    1.131651    0.009168    0.006645   -0.150852    0.883999    0.894850    0.869768    0.868523    0.880915    0.873662    0.941449    1.877900    1.855767    1.801357    1.819634    1.837888    1.742139    1.870986    1.245808    1.400361    1.664982    1.264854    1.572666    1.980834    1.460665   -0.004962   -0.004537   -0.006054   -0.006320   -0.004447   -0.003084   -0.004507   -0.004400   -0.004769   -0.001847   -0.002815   -0.004135   -0.002973   -0.004689   -0.004537   -0.004462   -0.004545   -0.005065   -0.004429   -0.004296   -0.004506   -0.004518   -0.004532   -0.004303   -0.004448   -0.004405   -0.003842   -0.004520   -0.002836   -0.004565   -0.001838   -0.003230   -0.002607   -0.002133   -0.004674   -0.004796   -0.004592   -0.004850   -0.006464   -0.004457   -0.004080   -0.004491    2.998396   -9.255852    0.865596    0.196227   -0.029058    0.173916    0.236757   -0.002671    0.188750    0.030448    0.044832    0.015145    0.110850  400.000000   -1.955000    0.598970    0.482329
  1.500  -0.147628   -0.175097    1.090154   -1.652688   -1.797804    0.091897    1.215947    1.600756    0.116773   -0.002748   -0.003684    0.079370    0.007815    0.004020   -0.023750   -0.176119   -0.177633   -0.180840   -0.172629   -0.186117   -0.203168   -0.135477    1.133013    1.097427    1.074441    1.099620    1.077222    1.029526    1.110182    0.310304    0.356550    0.592746    0.224216    0.544469    0.913347    0.346959   -0.004028   -0.003785   -0.004995   -0.005172   -0.003700   -0.002567   -0.003845   -0.003638   -0.003923   -0.001845   -0.002740   -0.003545   -0.002612   -0.003903   -0.003790   -0.003709   -0.003780   -0.004274   -0.003690   -0.003552   -0.003798   -0.003828   -0.003795   -0.003585   -0.003829   -0.003643   -0.002796   -0.003790   -0.002409   -0.003667   -0.001701   -0.002862   -0.002268   -0.001875   -0.003939   -0.003950   -0.003827   -0.003965   -0.004955   -0.003703   -0.003409   -0.003795    2.958590   -6.600575    0.868036    0.197459   -0.023876    0.201715    0.286400    0.004608    0.249536    0.043018    0.094099    0.022357    0.151252  400.000000   -1.025000    0.611546    0.482850
  2.000  -0.252372   -0.854227    0.586593   -1.646035   -1.825030    0.093684    1.313705    1.666320    0.117651   -0.002340   -0.003172   -0.637043    0.008070    0.002398    0.069323   -0.866431   -0.859883   -0.854613   -0.836519   -0.854159   -0.872871   -0.832154    0.612013    0.582327    0.579302    0.610200    0.570044    0.552350    0.585691   -0.281251   -0.262818   -0.063796   -0.399937   -0.082811    0.274633   -0.315466   -0.003377   -0.003258   -0.004207   -0.004419   -0.003164   -0.002182   -0.003356   -0.003136   -0.003329   -0.001810   -0.002643   -0.003099   -0.002329   -0.003348   -0.003257   -0.003176   -0.003249   -0.003711   -0.003171   -0.003040   -0.003283   -0.003309   -0.003265   -0.003100   -0.003379   -0.003121   -0.002151   -0.003287   -0.002064   -0.002921   -0.001592   -0.002515   -0.001982   -0.001685   -0.003235   -0.003330   -0.003271   -0.003335   -0.003869   -0.003190   -0.002980   -0.003263    3.467766   -4.646263    0.870534    0.198405   -0.039218    0.230982    0.282358   -0.014419    0.280580    0.046788    0.112241    0.013478    0.168365  400.000000   -0.299000    0.617348    0.483536
  3.000  -0.400000   -1.707597   -0.146184   -1.676629   -1.875876    0.096502    1.463120    1.752684    0.120191   -0.001757   -0.002495   -0.689732    0.009820    0.000279    0.186616   -1.739858   -1.712326   -1.706495   -1.684797   -1.689675   -1.703079   -1.706174   -0.149235   -0.154870   -0.148164   -0.120412   -0.154813   -0.151733   -0.166119   -0.721297   -0.638106   -0.544893   -0.802764   -0.498045   -0.174359   -0.749483   -0.002572   -0.002532   -0.003144   -0.003444   -0.002447   -0.001633   -0.002667   -0.002509   -0.002538   -0.001694   -0.002464   -0.002467   -0.001880   -0.002609   -0.002556   -0.002474   -0.002535   -0.003000   -0.002489   -0.002362   -0.002587   -0.002609   -0.002541   -0.002463   -0.002735   -0.002425   -0.001389   -0.002607   -0.001585   -0.001910   -0.001408   -0.001985   -0.001552   -0.001416   -0.002186   -0.002520   -0.002547   -0.002511   -0.002522   -0.002513   -0.002436   -0.002518    3.927270   -1.994229    0.874446    0.199694    0.000421    0.210506    0.286273    0.017050    0.305661    0.058672    0.124638    0.025344    0.157175  400.000000    0.000000    0.609291    0.484022
  4.000  -0.400000   -2.257680   -0.711213   -1.709830   -1.908284    0.098438    1.566046    1.811761    0.122612   -0.001377   -0.002044   -0.669863    0.011724   -0.001183    0.242311   -2.296566   -2.258376   -2.257421   -2.243242   -2.233628   -2.239112   -2.261116   -0.727215   -0.714678   -0.713875   -0.696792   -0.710637   -0.708760   -0.732028   -0.765949   -0.592150   -0.580628   -0.780816   -0.483198   -0.205314   -0.741057   -0.002111   -0.002046   -0.002479   -0.002803   -0.001988   -0.001226   -0.002211   -0.002115   -0.002029   -0.001585   -0.002315   -0.002036   -0.001554   -0.002134   -0.002100   -0.002026   -0.002068   -0.002528   -0.002046   -0.001926   -0.002128   -0.002159   -0.002068   -0.002047   -0.002283   -0.001984   -0.000954   -0.002166   -0.001279   -0.001296   -0.001271   -0.001631   -0.001265   -0.001220   -0.001523   -0.002018   -0.002083   -0.002008   -0.001761   -0.002083   -0.002080   -0.002035    3.370656   -0.305375    0.877064    0.200494    0.018832    0.244947    0.448002    0.052368    0.307164    0.042134    0.136546    0.030811    0.132087  400.000000    0.000000    0.587788    0.483600
  5.000  -0.400000   -2.663616   -1.182577   -1.732636   -1.925829    0.099774    1.637883    1.857447    0.124654   -0.001124   -0.001734   -0.634281    0.013387   -0.002311    0.259027   -2.699286   -2.660213   -2.663635   -2.659024   -2.640906   -2.638955   -2.664492   -1.200297   -1.179600   -1.186649   -1.179522   -1.176416   -1.181004   -1.197658   -0.724174   -0.487937   -0.519162   -0.687582   -0.401493   -0.159049   -0.659580   -0.001823   -0.001708   -0.002043   -0.002345   -0.001673   -0.000896   -0.001892   -0.001845   -0.001680   -0.001485   -0.002174   -0.001731   -0.001301   -0.001805   -0.001787   -0.001718   -0.001752   -0.002184   -0.001737   -0.001625   -0.001805   -0.001854   -0.001741   -0.001745   -0.001952   -0.001682   -0.000682   -0.001861   -0.001074   -0.000913   -0.001168   -0.001413   -0.001062   -0.001081   -0.001110   -0.001691   -0.001770   -0.001683   -0.001312   -0.001777   -0.001824   -0.001705    2.129627    0.840704    0.878804    0.201026    0.016383    0.261637    0.521325    0.010242    0.292740    0.044755    0.159273    0.012982    0.135285  400.000000    0.000000    0.562102    0.482730
  7.500  -0.400000   -3.368097   -2.100674   -1.750019   -1.933397    0.101594    1.738644    1.941358    0.128328   -0.000826   -0.001280   -0.404429    0.016399   -0.004287    0.209134   -3.376621   -3.357721   -3.365219   -3.375555   -3.360428   -3.344668   -3.355595   -2.098557   -2.089046   -2.103672   -2.106138   -2.094424   -2.109619   -2.099843   -0.631117   -0.364709   -0.410008   -0.549708   -0.332415   -0.113042   -0.551929   -0.001460   -0.001222   -0.001463   -0.001598   -0.001234   -0.000341   -0.001417   -0.001444   -0.001182   -0.001287   -0.001935   -0.001277   -0.000902   -0.001338   -0.001344   -0.001281   -0.001288   -0.001593   -0.001284   -0.001207   -0.001341   -0.001427   -0.001261   -0.001293   -0.001426   -0.001252   -0.000347   -0.001401   -0.000868   -0.000495   -0.001003   -0.001219   -0.000774   -0.000875   -0.000666   -0.001263   -0.001354   -0.001270   -0.000881   -0.001335   -0.001401   -0.001262   -2.254304    2.472428    0.881005    0.201732    0.012890    0.220665    0.485046   -0.005722    0.248740    0.037142    0.126677    0.013252    0.125747  400.000000    0.000000    0.500922    0.480025
 10.000  -0.400000   -3.837779   -2.768999   -1.735295   -1.918144    0.102312    1.780438    2.000913    0.130643   -0.000728   -0.001042   -0.323410    0.018282   -0.005541    0.101221   -3.812918   -3.824348   -3.829129   -3.838503   -3.845875   -3.821734   -3.816493   -2.735575   -2.759945   -2.764280   -2.756891   -2.773092   -2.784359   -2.761835   -0.569527   -0.362123   -0.387004   -0.505966   -0.387988   -0.143450   -0.531306   -0.001304   -0.000969   -0.001212   -0.001164   -0.001032    0.000029   -0.001172   -0.001228   -0.000956   -0.001137   -0.001776   -0.001050   -0.000652   -0.001120   -0.001123   -0.001069   -0.001052   -0.001234   -0.001047   -0.000998   -0.001106   -0.001238   -0.001011   -0.001036   -0.001158   -0.001051   -0.000235   -0.001149   -0.000830   -0.000390   -0.000899   -0.001253   -0.000635   -0.000737   -0.000611   -0.001084   -0.001166   -0.001122   -0.000846   -0.001086   -0.001146   -0.001075   -6.949083    3.239206    0.881692    0.202004    0.015757    0.186337    0.402734   -0.012161    0.214443    0.020092    0.109094   -0.002629    0.112043  400.000000    0.000000    0.453378    0.477483
""")


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

    COEFFS = CoeffsTable(sa_damping=5, table="""\
    IMT         c0       AK_c0 Aleutian_c0 Cascadia_c0    CAM_N_c0    CAM_S_c0   JP_Pac_c0   JP_Phi_c0     SA_N_c0     SA_S_c0     TW_E_c0     TW_W_c0      c0slab   AK_c0slab Aleutian_c0slab Cascadia_c0slab  CAM_c0slab   JP_c0slab SA_N_c0slab SA_S_c0slab   TW_c0slab          c1      c1slab          b4          a0       AK_a0      CAM_a0       JP_a0       SA_a0       TW_a0      a0slab   AK_a0slab Cascadia_a0slab  CAM_a0slab   JP_a0slab   SA_a0slab   TW_a0slab          c4          c5          c6      c4slab      c5slab      c6slab           d           m          db          V2       JP_s1       TW_s1          s2       AK_s2 Cascadia_s2       JP_s2       SA_s2       TW_s2          f4      f4slab          f5        J_e1        J_e2        J_e3        C_e1        C_e2        C_e3    del_None del_Seattle         Tau       phi21       phi22       phi2V          VM   phi2S2S,0          a1    phi2SS,1    phi2SS,2          a2
    pgv   8.097000    9.283796    8.374796    7.728000    7.046900    7.046900    8.772126    7.579126    8.528671    8.679671    7.559846    7.559846   13.194000   12.790000   13.600000   12.874000   12.810000   13.248000   12.754000   12.927000   13.516000   -1.661000   -2.422000    0.100000   -0.003950   -0.004040   -0.001530   -0.002390   -0.000311   -0.005140   -0.001900   -0.002380   -0.001090   -0.001920   -0.002150   -0.001920   -0.003660    1.336000   -0.039000    1.844000    1.840000   -0.050000    0.800000    0.269300    0.025200   67.000000 3868.617587   -0.738000   -0.454000   -0.305487   -1.031000   -0.671000   -0.738000   -0.681000   -0.590000   -0.317630   -0.317630   -0.005200   -0.137000    0.137000    0.091000    0.000000    0.115000    0.068000   -0.115000    0.000000    0.477000    0.348000    0.288000   -0.179000  423.000000    0.142000    0.047000    0.153000    0.166000    0.011000
    pga   4.082000    4.458796    3.652796    3.856000    2.875900    2.875900    5.373126    4.309126    5.064671    5.198671    3.032846    3.032846    9.907000    9.404000    9.912000    9.600000    9.580000   10.145000    9.254000    9.991000   10.071000   -1.662000   -2.543000    0.100000   -0.006570   -0.005410   -0.003870   -0.008620   -0.003970   -0.007870   -0.002550   -0.002270   -0.003540   -0.002380   -0.003350   -0.002380   -0.003620    1.246000   -0.021000    1.128000    1.840000   -0.050000    0.400000    0.300400    0.031400   67.000000  920.512384   -0.586000   -0.440000   -0.839905   -0.785000   -0.572000   -0.586000   -0.333000   -0.440000   -0.441690   -0.441690   -0.005200    0.000000    0.000000    1.000000    0.000000    0.000000    1.000000    0.000000    0.000000    0.480000    0.396000    0.565000   -0.180000  423.000000    0.221000    0.093000    0.149000    0.327000    0.068000
  0.010   3.714000    4.094796    3.288796    3.488000    2.564900    2.564900    5.022126    3.901126    4.673671    4.807671    2.636846    2.636846    9.962000    9.451000    9.954000    9.802000    9.612000   10.162000    9.293000    9.994000   10.174000   -1.587000   -2.554000    0.100000   -0.006570   -0.005410   -0.003870   -0.008620   -0.003970   -0.007870   -0.002550   -0.002190   -0.004010   -0.002170   -0.003110   -0.002170   -0.003550    1.246000   -0.021000    1.128000    1.840000   -0.050000    0.400000    0.283900    0.029600   67.000000 1300.000000   -0.604000   -0.440000   -0.498000   -0.803000   -0.571000   -0.604000   -0.333000   -0.440000   -0.485900   -0.485900   -0.005200    0.000000    0.000000    1.000000    0.000000    0.000000    1.000000    0.000000    0.000000    0.476000    0.397000    0.560000   -0.180000  423.000000    0.223000    0.098000    0.148000    0.294000    0.071000
  0.020   3.762000    4.132796    3.338796    3.536000    2.636900    2.636900    5.066126    3.935126    4.694671    4.827671    2.698846    2.698846   10.099000    9.587000   10.086000    9.933000    9.771000   10.306000    9.403000   10.152000   10.273000   -1.593000   -2.566000    0.100000   -0.006570   -0.005410   -0.003870   -0.008620   -0.003970   -0.007870   -0.002550   -0.002190   -0.004010   -0.002170   -0.003110   -0.002170   -0.003550    1.227000   -0.021000    1.128000    1.884000   -0.050000    0.415000    0.285400    0.029800   67.000000 1225.000000   -0.593000   -0.458000   -0.478000   -0.785000   -0.575000   -0.593000   -0.345000   -0.458000   -0.485900   -0.485900   -0.005180    0.000000    0.000000    1.000000    0.000000    0.000000    1.000000    0.000000    0.000000    0.482000    0.401000    0.563000   -0.181000  423.000000    0.227000    0.105000    0.149000    0.294000    0.073000
  0.030   4.014000    4.386796    3.535796    3.788000    2.890900    2.890900    5.317126    4.278126    4.935671    5.066671    2.926846    2.926846   10.311000    9.808000   10.302000   10.133000    9.993000   10.498000    9.592000   10.459000   10.451000   -1.630000   -2.594000    0.100000   -0.006570   -0.005410   -0.003870   -0.008620   -0.003970   -0.007870   -0.002550   -0.002190   -0.004010   -0.002170   -0.003110   -0.002170   -0.003550    1.215000   -0.021000    1.128000    1.884000   -0.050000    0.445000    0.293200    0.030600   67.000000 1200.000000   -0.539000   -0.455000   -0.446000   -0.690000   -0.565000   -0.561000   -0.380000   -0.464000   -0.490800   -0.490800   -0.005110    0.000000    0.000000    1.000000    0.000000    0.000000    1.000000    0.000000    0.000000    0.500000    0.413000    0.589000   -0.188000  423.000000    0.239000    0.145000    0.153000    0.313000    0.077000
  0.050   4.456000    4.745796    3.959796    4.230000    3.287900    3.287900    5.843126    4.816126    5.457671    5.586671    3.236846    3.236846   10.824000   10.379000   10.862000   10.634000   10.563000   10.981000   10.027000   11.102000   10.860000   -1.687000   -2.649000    0.100000   -0.006570   -0.005410   -0.003870   -0.008620   -0.003970   -0.007870   -0.002550   -0.002190   -0.004010   -0.002170   -0.003110   -0.002170   -0.003550    1.201000   -0.021000    1.128000    1.884000   -0.050000    0.475000    0.304800    0.031600   67.000000  605.000118   -0.403000   -0.452000   -1.942634   -0.594000   -0.519000   -0.461000   -0.427000   -0.468000   -0.498230   -0.498230   -0.004970    0.000000    0.000000    1.000000    0.100000   -0.100000   -0.063000   -0.050000    0.000000    0.528000    0.473000    0.653000   -0.230000  423.000000    0.285000    0.200000    0.167000    0.330000    0.077000
  0.075   4.742000    4.972796    4.231796    4.516000    3.560900    3.560900    6.146126    5.126126    5.788671    5.917671    3.446846    3.446846   11.084000   10.650000   11.184000   10.888000   10.785000   11.250000   10.265000   11.424000   11.093000   -1.715000   -2.650000    0.100000   -0.006570   -0.005410   -0.003870   -0.008620   -0.003970   -0.007870   -0.002550   -0.002190   -0.004010   -0.002170   -0.003110   -0.002170   -0.003550    1.190000   -0.021000    1.128000    1.884000   -0.050000    0.490000    0.299200    0.032100   67.000000  538.000008   -0.325000   -0.456000   -1.267990   -0.586000   -0.497000   -0.452000   -0.458000   -0.473000   -0.497240   -0.497240   -0.004890    0.050000   -0.043000   -0.025000    0.300000   -0.340000   -0.200000   -0.075000    0.078000    0.530000    0.529000    0.722000   -0.262000  423.000000    0.339000    0.205000    0.184000    0.299000    0.063000
  0.100   4.952000    5.160796    4.471796    4.726000    3.788900    3.788900    6.346126    5.333126    5.998671    6.126671    3.643846    3.643846   11.232000   10.816000   11.304000   11.030000   10.841000   11.466000   10.467000   11.490000   11.283000   -1.737000   -2.647000    0.100000   -0.006570   -0.005410   -0.003870   -0.008620   -0.003970   -0.007870   -0.002550   -0.002190   -0.004010   -0.002170   -0.003110   -0.002170   -0.003550    1.182000   -0.021000    1.128000    1.884000   -0.050000    0.505000    0.285400    0.032000   67.000000  834.126307   -0.264000   -0.468000   -1.088713   -0.629000   -0.486000   -0.498000   -0.490000   -0.482000   -0.494710   -0.494710   -0.004780    0.100000   -0.085000   -0.050000    0.333000   -0.377000   -0.222000   -0.081000    0.075000    0.524000    0.517000    0.712000   -0.239000  423.000000    0.347000    0.185000    0.176000    0.310000    0.061000
  0.150   5.080000    5.285796    4.665796    4.848000    3.945900    3.945900    6.425126    5.420126    6.103671    6.230671    3.798846    3.798846   11.311000   10.883000   11.402000   11.103000   10.809000   11.619000   10.566000   11.320000   11.503000   -1.745000   -2.634000    0.100000   -0.006570   -0.005410   -0.003870   -0.008620   -0.003970   -0.007870   -0.002550   -0.002190   -0.004010   -0.002170   -0.003110   -0.002170   -0.003550    1.171000   -0.021000    1.162000    1.884000   -0.060000    0.520000    0.281400    0.032500   67.000000  558.601030   -0.250000   -0.484000   -1.347175   -0.729000   -0.499000   -0.568000   -0.536000   -0.499000   -0.485830   -0.485830   -0.004600    0.164000   -0.139000   -0.082000    0.290000   -0.290000   -0.193000   -0.091000    0.064000    0.510000    0.457000    0.644000   -0.185000  423.000000    0.313000    0.123000    0.164000    0.307000    0.076000
  0.200   5.035000    5.277796    4.661796    4.798000    3.943900    3.943900    6.288126    5.289126    6.013671    6.140671    3.827846    3.827846   11.055000   10.633000   11.183000   10.841000   10.519000   11.351000   10.330000   10.927000   11.320000   -1.732000   -2.583000    0.100000   -0.006570   -0.005410   -0.003870   -0.008620   -0.003970   -0.007870   -0.002550   -0.002190   -0.004010   -0.002170   -0.003110   -0.002170   -0.003550    1.163000   -0.021000    1.187000    1.884000   -0.068000    0.535000    0.291000    0.030600   67.000000  605.000012   -0.288000   -0.498000   -1.341523   -0.867000   -0.533000   -0.667000   -0.584000   -0.522000   -0.473830   -0.473830   -0.004340    0.164000   -0.139000   -0.082000    0.177000   -0.192000   -0.148000   -0.092000    0.075000    0.501000    0.432000    0.640000   -0.138000  423.000000    0.277000    0.110000    0.163000    0.301000    0.070000
  0.250   4.859000    5.154796    4.503796    4.618000    3.800900    3.800900    5.972126    4.979126    5.849671    5.974671    3.765846    3.765846   10.803000   10.322000   10.965000   10.583000   10.268000   11.063000   10.124000   10.555000   11.147000   -1.696000   -2.539000    0.100000   -0.006570   -0.005410   -0.003870   -0.008620   -0.003970   -0.007870   -0.002550   -0.002190   -0.004010   -0.002170   -0.003110   -0.002170   -0.003550    1.156000   -0.021000    1.204000    1.884000   -0.075000    0.550000    0.275800    0.030600   67.000000  604.999930   -0.360000   -0.511000   -1.387303   -1.011000   -0.592000   -0.781000   -0.654000   -0.555000   -0.476960   -0.476960   -0.004020    0.080000   -0.080000   -0.053000    0.100000   -0.035000   -0.054000    0.000000    0.000000    0.492000    0.450000    0.633000   -0.185000  423.000000    0.260000    0.119000    0.169000    0.233000    0.077000
  0.300   4.583000    4.910796    4.276796    4.340000    3.491900    3.491900    5.582126    4.592126    5.603671    5.728671    3.602846    3.602846   10.669000   10.116000   10.870000   10.443000   10.134000   10.878000   10.077000   10.328000   11.079000   -1.643000   -2.528000    0.100000   -0.006570   -0.005410   -0.003870   -0.008620   -0.003970   -0.007870   -0.002550   -0.002190   -0.004010   -0.002170   -0.003110   -0.002170   -0.003550    1.151000   -0.021000    1.215000    1.884000   -0.082000    0.565000    0.271900    0.032300   67.000000  589.790784   -0.455000   -0.514000   -1.344229   -1.133000   -0.681000   -0.867000   -0.725000   -0.596000   -0.484500   -0.484500   -0.003700    0.000000    0.000000    1.000000    0.000000    0.000000    1.000000    0.000000    0.000000    0.492000    0.436000    0.584000   -0.158000  423.000000    0.254000    0.092000    0.159000    0.220000    0.065000
  0.400   4.180000    4.548796    3.919796    3.935000    3.128900    3.128900    5.091126    4.089126    5.151671    5.277671    3.343846    3.343846   10.116000    9.561000   10.411000    9.884000    9.598000   10.296000    9.539000    9.639000   10.547000   -1.580000   -2.452000    0.100000   -0.006570   -0.005410   -0.003870   -0.008620   -0.003970   -0.007870   -0.002550   -0.002190   -0.004010   -0.002170   -0.003110   -0.002170   -0.003550    1.143000   -0.022000    1.227000    1.884000   -0.091000    0.580000    0.253900    0.030200   67.000000  587.334695   -0.617000   -0.510000   -1.110239   -1.238000   -0.772000   -0.947000   -0.801000   -0.643000   -0.481050   -0.481050   -0.003420   -0.130000    0.113000    0.087000    0.000000    0.050000    0.200000    0.000000    0.000000    0.492000    0.433000    0.556000   -0.190000  423.000000    0.230000    0.044000    0.158000    0.222000    0.064000
  0.500   3.752000    4.168796    3.486796    3.505000    2.640900    2.640900    4.680126    3.571126    4.719671    4.848671    3.028846    3.028846    9.579000    8.973000    9.901000    9.341000    9.097000    9.711000    9.030000    9.030000   10.049000   -1.519000   -2.384000    0.100000   -0.006570   -0.005410   -0.003870   -0.008620   -0.003970   -0.007870   -0.002550   -0.002190   -0.004010   -0.002170   -0.003110   -0.002170   -0.003550    1.143000   -0.023000    1.234000    1.884000   -0.100000    0.595000    0.248200    0.029500   67.000000  574.717064   -0.757000   -0.506000   -0.867564   -1.321000   -0.838000   -1.003000   -0.863000   -0.689000   -0.464920   -0.464920   -0.003220   -0.200000    0.176000    0.118000    0.000000    0.100000    0.200000    0.000000    0.000000    0.492000    0.428000    0.510000   -0.186000  423.000000    0.225000    0.038000    0.160000    0.243000    0.044000
  0.750   3.085000    3.510796    2.710796    2.837000    1.987900    1.987900    3.906126    2.844126    3.995671    4.129671    2.499846    2.499846    8.837000    8.246000    9.335000    8.593000    8.324000    8.934000    8.258000    8.258000    9.327000   -1.440000   -2.338000    0.100000   -0.006350   -0.004780   -0.003420   -0.007630   -0.003510   -0.006800   -0.002110   -0.001890   -0.003470   -0.001880   -0.002690   -0.001880   -0.003070    1.217000   -0.026000    1.240000    1.884000   -0.115000    0.610000    0.222700    0.026600   67.000000  604.999725   -0.966000   -0.500000   -0.484668   -1.383000   -0.922000   -1.052000   -0.942000   -0.745000   -0.434390   -0.434390   -0.003120   -0.401000    0.284000    0.167000    0.000000    0.200000    0.125000   -0.200000    0.012000    0.492000    0.448000    0.471000   -0.177000  422.000000    0.218000    0.040000    0.175000    0.241000    0.040000
  1.000   2.644000    3.067796    2.238796    2.396000    1.553900    1.553900    3.481126    2.371126    3.512671    3.653671    2.140846    2.140846    8.067000    7.507000    8.680000    7.817000    7.557000    8.164000    7.467000    7.417000    8.504000   -1.419000   -2.267000    0.100000   -0.005800   -0.004150   -0.002970   -0.006630   -0.003050   -0.006050   -0.001870   -0.001680   -0.003090   -0.001670   -0.002390   -0.001670   -0.002730    1.270000   -0.028000    1.240000    1.884000   -0.134000    0.625000    0.196900    0.023100   67.000000  604.989699   -0.986000   -0.490000   -0.365585   -1.414000   -0.932000   -1.028000   -0.960000   -0.777000   -0.384840   -0.384840   -0.003100   -0.488000    0.346000    0.203000    0.000000    0.245000    0.153000   -0.245000    0.037000    0.492000    0.430000    0.430000   -0.166000  422.000000    0.227000    0.015000    0.195000    0.195000    0.043000
  1.500   2.046000    2.513796    1.451796    1.799000    0.990900    0.990900    2.870126    1.779126    2.875671    3.023671    1.645846    1.645846    6.829000    6.213000    7.581000    6.573000    6.350000    6.896000    6.220000    6.180000    7.204000   -1.400000   -2.166000    0.100000   -0.005050   -0.003420   -0.002450   -0.005460   -0.002520   -0.004980   -0.001540   -0.001390   -0.002540   -0.001380   -0.001970   -0.001380   -0.002250    1.344000   -0.031000    1.237000    1.884000   -0.154000    0.640000    0.145200    0.011800   67.000000  604.998656   -0.966000   -0.486000   -0.236900   -1.430000   -0.814000   -0.971000   -0.942000   -0.790000   -0.323180   -0.323180   -0.003100   -0.578000    0.480000    0.240000    0.000000    0.320000    0.200000   -0.320000    0.064000    0.492000    0.406000    0.406000   -0.111000  422.000000    0.244000   -0.047000    0.204000    0.204000   -0.034000
  2.000   1.556000    2.061796    0.906796    1.310000    0.534900    0.534900    2.507126    1.293126    2.327671    2.481671    1.217846    1.217846    5.871000    5.206000    6.671000    5.609000    5.434000    5.935000    5.261000    5.161000    6.227000   -1.391000   -2.077000    0.100000   -0.004290   -0.002900   -0.002080   -0.004630   -0.002140   -0.004230   -0.001310   -0.001180   -0.002160   -0.001170   -0.001670   -0.001170   -0.001910    1.396000   -0.034000    1.232000    1.884000   -0.154000    0.655000    0.060000    0.007000   67.000000  599.831889   -0.901000   -0.475000   -0.139993   -1.421000   -0.725000   -0.901000   -0.891000   -0.765000   -0.265770   -0.265770   -0.003100   -0.645000    0.579000    0.254000    0.000000    0.370000    0.239000   -0.280000    0.140000    0.492000    0.393000    0.393000    0.000000  422.000000    0.231000   -0.036000    0.196000    0.196000   -0.036000
  3.000   0.920000    1.456796    0.099796    0.675000   -0.087100   -0.087100    1.969126    0.607126    1.766671    1.932671    0.596846    0.596846    4.830000    4.206000    5.667000    4.556000    4.441000    4.849000    4.176000    4.076000    5.157000   -1.416000   -2.012000    0.100000   -0.003210   -0.002170   -0.001560   -0.003470   -0.001600   -0.003160   -0.000979   -0.000880   -0.001610   -0.000873   -0.001250   -0.000873   -0.001430    1.470000   -0.038000    1.223000    1.949000   -0.154000    0.685000    0.000000    0.000000    0.000000  599.831889   -0.751000   -0.428000   -0.139993   -1.343000   -0.570000   -0.751000   -0.787000   -0.675000    0.000000   -0.178070   -0.003100   -0.772000    0.635000    0.265000    0.000000    0.430000    0.287000   -0.355000    0.165000    0.492000    0.367000    0.367000    0.000000  419.000000    0.199000   -0.030000    0.177000    0.177000   -0.011000
  4.000   0.595000    1.207796   -0.356204    0.352000   -0.353100   -0.353100    1.675126    0.303126    1.524671    1.698671    0.268846    0.268846    4.173000    3.517000    4.970000    3.893000    3.849000    4.074000    3.495000    3.445000    4.550000   -1.452000   -1.989000    0.100000   -0.002440   -0.001650   -0.001180   -0.002640   -0.001220   -0.002410   -0.000745   -0.000670   -0.001230   -0.000664   -0.000952   -0.000664   -0.001090    1.523000   -0.044000    1.216000    2.031000   -0.154000    0.700000    0.000000    0.000000    0.000000  599.831889   -0.680000   -0.396000   -0.139993   -1.297000   -0.489000   -0.680000   -0.706000   -0.613000    0.000000   -0.137290   -0.003100   -0.699000    0.709000    0.259000    0.000000    0.440000    0.303000   -0.417000    0.163000    0.492000    0.330000    0.330000    0.000000  416.000000    0.191000   -0.042000    0.158000    0.158000    0.033000
  5.000   0.465000    1.131796   -0.601204    0.223000   -0.491100   -0.491100    1.601126    0.183126    1.483671    1.665671    0.014846    0.014846    3.833000    3.142000    4.592000    3.547000    3.502000    3.814000    3.038000    3.038000    4.229000   -1.504000   -1.998000    0.100000   -0.001600   -0.001250   -0.000895   -0.002000   -0.000919   -0.001820   -0.000564   -0.000507   -0.000929   -0.000503   -0.000720   -0.000503   -0.000822    1.564000   -0.048000    1.210000    2.131000   -0.154000    0.715000    0.000000    0.000000    0.000000  599.831889   -0.592000   -0.353000   -0.139993   -1.233000   -0.421000   -0.592000   -0.621000   -0.536000    0.000000   -0.077330   -0.003100   -0.642000    0.630000    0.215000    0.000000    0.450000    0.321000   -0.450000    0.132000    0.492000    0.298000    0.298000    0.000000  415.000000    0.181000    0.005000    0.132000    0.132000    0.014000
  7.500   0.078000    0.758796   -1.137204   -0.162000   -0.837100   -0.837100    1.270126   -0.143874    1.175671    1.366671   -0.446154   -0.446154    3.132000    2.391000    3.650000    2.840000    2.821000    3.152000    2.368000    2.368000    3.554000   -1.569000   -2.019000    0.100000   -0.000766   -0.000519   -0.000371   -0.000828   -0.000382   -0.000755   -0.000234   -0.000210   -0.000385   -0.000209   -0.000299   -0.000209   -0.000341    1.638000   -0.059000    1.200000    2.185000   -0.154000    0.730000    0.000000    0.000000    0.000000  599.831889   -0.494000   -0.311000   -0.139993   -1.147000   -0.357000   -0.520000   -0.520000   -0.444000    0.000000   -0.054430   -0.003100   -0.524000    0.306000    0.175000    0.000000    0.406000    0.312000   -0.350000    0.150000    0.492000    0.254000    0.254000    0.000000  419.000000    0.181000   -0.016000    0.113000    0.113000    0.016000
 10.000   0.046000    0.708796   -1.290204   -0.193000   -0.864100   -0.864100    1.364126   -0.195874    1.271671    1.462671   -0.473154   -0.473154    2.720000    2.031000    2.950000    2.422000    2.408000    2.791000    1.939000    1.939000    3.166000   -1.676000   -2.047000    0.100000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    0.000000    1.690000   -0.067000    1.194000    2.350000   -0.154000    0.745000    0.000000    0.000000    0.000000  599.831889   -0.395000   -0.261000   -0.139993   -1.060000   -0.302000   -0.395000   -0.420000   -0.352000    0.000000   -0.033130   -0.003100   -0.327000    0.182000    0.121000    0.000000    0.345000    0.265000   -0.331000    0.117000    0.492000    0.231000    0.231000    0.000000  427.000000    0.181000    0.040000    0.110000    0.110000    0.017000
 """)
    

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

    @staticmethod    
    def compute(gmm, imts, mean, sig, tau, phi):
        """
        Adjust GMM values.
        """
        gmm.super().compute(ctx, imts, mean, sig, tau, phi)
        mean += gmm.mean_adjustment


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
        AdjustedGMM.compute(self, imts, mean, sig, tau, phi)


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
        AdjustedGMM.compute(self, imts, mean, sig, tau, phi)
    

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
        AdjustedGMM.compute(self, imts, mean, sig, tau, phi)
    

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
        AdjustedGMM.compute(self, imts, mean, sig, tau, phi)
    

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
        AdjustedGMM.compute(self, imts, mean, sig, tau, phi)


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
        AdjustedGMM.compute(self, imts, mean, sig, tau, phi)


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
        AdjustedGMM.compute(self, imts, mean, sig, tau, phi)


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
        AdjustedGMM.compute(self, imts, mean, sig, tau, phi)


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
        AdjustedGMM.compute(self, imts, mean, sig, tau, phi)


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
        AdjustedGMM.compute(self, imts, mean, sig, tau, phi)
