# -*- coding: utf-8 -*-
"""
USGS adjustments to ground-motion models for Puerto Rico and the Virgin Islands.

The adjustments correspond to the fixed effect computed using linear mixed effects with
grouping for the event and site terms.

If bias_adjustment='IMT', then use IMT-dependent adjustment.
If bias_adjustment='Mean", then use IMT-dependent adjustment (average bias over IMTs).


Module exports :class:`AbrahamsonEtAl2014_USGSPRVI_AdjMean`
               :class:`AbrahamsonEtAl2014_USGSPRVI_AdjIMT`
"""
import numpy as np
import abc

from openquake.hazardlib.gsim.base import CoeffsTable, add_alias

from openquake.hazardlib.gsim import (
    abrahamson_2014,
    boore_2014,
    campbell_bozorgnia_2014,
    chiou_youngs_2014,
    abrahamson_gulerce_2020,
    kuehn_2020,
    parker_2020,
    )


ACTIVE_CRUSTAL_MEAN_ADJUSTMENT = -0.370
COEFFS_ACTIVE_CRUSTAL_ADJUSTMENTS = CoeffsTable(sa_damping=5, table="""\
IMT   bias
pgv     -0.450
pga     -0.327
0.010   -0.327
0.020   -0.300
0.030   -0.273
0.050   -0.246
0.075   -0.155
0.100   -0.183
0.150   -0.138
0.200   -0.128
0.250   -0.192
0.300   -0.286
0.400   -0.467
0.500   -0.597
0.750   -0.679
1.000   -0.664
1.500   -0.592
2.000   -0.488
3.000   -0.325
4.000   -0.176
5.000   -0.003
7.500    0.255
10.000   0.424
""")

SUBDUCTION_INTERFACE_MEAN_ADJUSTMENT = -1.092
COEFFS_SUBDUCTION_INTERFACE_ADJUSTMENTS = CoeffsTable(sa_damping=5, table="""\
IMT   bias
pgv     -1.168
pga     -0.885
0.010   -0.885
0.020   -0.802
0.030   -0.720
0.050   -0.637
0.075   -0.597
0.100   -0.590
0.150   -0.557
0.200   -0.792
0.250   -0.834
0.300   -0.967
0.400   -1.246
0.500   -1.425
0.750   -1.653
1.000   -1.722
1.500   -1.669
2.000   -1.512
3.000   -1.258
4.000   -1.120
5.000   -0.973
7.500   -0.756
10.000  -0.612
""")

SUBDUCTION_INTRASLAB_MEAN_ADJUSTMENT = -0.404
COEFFS_SUBDUCTION_INTRASLAB_ADJUSTMENTS = CoeffsTable(sa_damping=5, table="""\
IMT   bias
pgv     -0.461
pga     -0.370
0.010   -0.370
0.020   -0.236
0.030   -0.101
0.050    0.034
0.075   -0.049
0.100   -0.252
0.150   -0.030
0.200   -0.228
0.250   -0.226
0.300   -0.296
0.400   -0.422
0.500   -0.564
0.750   -0.755
1.000   -0.829
1.500   -0.850
2.000   -0.781
3.000   -0.471
4.000   -0.288
5.000   -0.053
7.500    0.289
10.000   0.640
""")


class AdjustedGMM:
    """
    Add USGS adjustments to ground-motion models for Puerto Rico and the Virgin Islands.
    """

    @staticmethod    
    def add_adjustment(gmm, imts, mean, sig, tau, phi):
        """
        Adjust GMM values.
        """
        if gmm.bias_adjustment == "Mean":
            mean += gmm.mean_adjustment
        else:
            for m, imt in enumerate(imts):
                mean[m] +=gmm.imt_adjustments[imt]['bias']



class AbrahamsonEtAl2014_USGSPRVI(abrahamson_2014.AbrahamsonEtAl2014):
    """
    Abrahamson et al. (2014) active crustal ground-motion model with 
    USGS adjustments for Puerto Rico and the Virgin Islands.
    """
    
    def __init__(self, sigma_mu_epsilon = 0.0, **kwargs):
        super().__init__(sigma_mu_epsilon=sigma_mu_epsilon, **kwargs)
        self.bias_adjustment = kwargs.get('bias_adjustment')
        assert self.bias_adjustment in ("Mean", "IMT")
        self.mean_adjustment = ACTIVE_CRUSTAL_MEAN_ADJUSTMENT
        self.imt_adjustments = COEFFS_ACTIVE_CRUSTAL_ADJUSTMENTS

    def compute(self, ctx: np.recarray, imts, mean, sig, tau, phi):
        """
        Compute GMM values.
        """
        super().compute(ctx, imts, mean, sig, tau, phi)
        AdjustedGMM.add_adjustment(self, imts, mean, sig, tau, phi)


class BooreEtAl2014_USGSPRVI(boore_2014.BooreEtAl2014):
    """
    Boore et al. (2014) active crustal ground-motion model with 
    USGS adjustments for Puerto Rico and the Virgin Islands.
    """
    
    def __init__(self, sigma_mu_epsilon = 0.0, **kwargs):
        super().__init__(sigma_mu_epsilon=sigma_mu_epsilon, **kwargs)
        self.bias_adjustment = kwargs.get('bias_adjustment')
        assert self.bias_adjustment in ("Mean", "IMT")
        self.mean_adjustment = ACTIVE_CRUSTAL_MEAN_ADJUSTMENT
        self.imt_adjustments = COEFFS_ACTIVE_CRUSTAL_ADJUSTMENTS

    def compute(self, ctx: np.recarray, imts, mean, sig, tau, phi):
        """
        Compute GMM values.
        """
        super().compute(ctx, imts, mean, sig, tau, phi)
        AdjustedGMM.add_adjustment(self, imts, mean, sig, tau, phi)
    

class CampbellBozorgnia2014_USGSPRVI(campbell_bozorgnia_2014.CampbellBozorgnia2014):
    """
    Campbell and Bozorgnia (2014) active crustal ground-motion model with 
    USGS adjustments for Puerto Rico and the Virgin Islands.
    """
    
    def __init__(self, sigma_mu_epsilon = 0.0, **kwargs):
        super().__init__(sigma_mu_epsilon=sigma_mu_epsilon, **kwargs)
        self.bias_adjustment = kwargs.get('bias_adjustment')
        assert self.bias_adjustment in ("Mean", "IMT")
        self.mean_adjustment = ACTIVE_CRUSTAL_MEAN_ADJUSTMENT
        self.imt_adjustments = COEFFS_ACTIVE_CRUSTAL_ADJUSTMENTS

    def compute(self, ctx: np.recarray, imts, mean, sig, tau, phi):
        """
        Compute GMM values.
        """
        super().compute(ctx, imts, mean, sig, tau, phi)
        AdjustedGMM.add_adjustment(self, imts, mean, sig, tau, phi)
    

class ChiouYoungs2014_USGSPRVI(chiou_youngs_2014.ChiouYoungs2014):
    """
    Chiou and Youngs (2014) active crustal ground-motion model with 
    USGS adjustments for Puerto Rico and the Virgin Islands.
    """
    
    def __init__(self, sigma_mu_epsilon = 0.0, **kwargs):
        super().__init__(sigma_mu_epsilon=sigma_mu_epsilon, **kwargs)
        self.bias_adjustment = kwargs.get('bias_adjustment')
        assert self.bias_adjustment in ("Mean", "IMT")
        self.mean_adjustment = ACTIVE_CRUSTAL_MEAN_ADJUSTMENT
        self.imt_adjustments = COEFFS_ACTIVE_CRUSTAL_ADJUSTMENTS

    def compute(self, ctx: np.recarray, imts, mean, sig, tau, phi):
        """
        Compute GMM values.
        """
        super().compute(ctx, imts, mean, sig, tau, phi)
        AdjustedGMM.add_adjustment(self, imts, mean, sig, tau, phi)
    

class AbrahamsonGulerce2020SInter_USGSPRVI(abrahamson_gulerce_2020.AbrahamsonGulerce2020SInter):
    """
    Abrahamson and Gulerce (2020) subduction interface ground-motion model with 
    USGS adjustments for Puerto Rico and the Virgin Islands.
    """
    
    def __init__(self, sigma_mu_epsilon = 0.0, **kwargs):
        super().__init__(sigma_mu_epsilon=sigma_mu_epsilon, **kwargs)
        self.bias_adjustment = kwargs.get('bias_adjustment')
        assert self.bias_adjustment in ("Mean", "IMT")
        self.mean_adjustment = SUBDUCTION_INTERFACE_MEAN_ADJUSTMENT
        self.imt_adjustments = COEFFS_SUBDUCTION_INTERFACE_ADJUSTMENTS

    def compute(self, ctx: np.recarray, imts, mean, sig, tau, phi):
        """
        Compute GMM values.
        """
        super().compute(ctx, imts, mean, sig, tau, phi)
        AdjustedGMM.add_adjustment(self, imts, mean, sig, tau, phi)


class KuehnEtAl2020SInter_USGSPRVI(kuehn_2020.KuehnEtAl2020SInter):
    """
    Kuehn et al. (2020) subduction interface ground-motion model with 
    USGS adjustments for Puerto Rico and the Virgin Islands.
    """
    
    def __init__(self, sigma_mu_epsilon = 0.0, **kwargs):
        super().__init__(sigma_mu_epsilon=sigma_mu_epsilon, **kwargs)
        self.bias_adjustment = kwargs.get('bias_adjustment')
        assert self.bias_adjustment in ("Mean", "IMT")
        self.mean_adjustment = SUBDUCTION_INTERFACE_MEAN_ADJUSTMENT
        self.imt_adjustments = COEFFS_SUBDUCTION_INTERFACE_ADJUSTMENTS

    def compute(self, ctx: np.recarray, imts, mean, sig, tau, phi):
        """
        Compute GMM values.
        """
        super().compute(ctx, imts, mean, sig, tau, phi)
        AdjustedGMM.add_adjustment(self, imts, mean, sig, tau, phi)


class ParkerEtAl2020SInter_USGSPRVI(parker_2020.ParkerEtAl2020SInter):
    """
    Parker et al. (2020) subduction interface ground-motion model with 
    USGS adjustments for Puerto Rico and the Virgin Islands.
    """
    
    def __init__(self, sigma_mu_epsilon = 0.0, **kwargs):
        super().__init__(sigma_mu_epsilon=sigma_mu_epsilon, **kwargs)
        self.bias_adjustment = kwargs.get('bias_adjustment')
        assert self.bias_adjustment in ("Mean", "IMT")
        self.mean_adjustment = SUBDUCTION_INTERFACE_MEAN_ADJUSTMENT
        self.imt_adjustments = COEFFS_SUBDUCTION_INTERFACE_ADJUSTMENTS

    def compute(self, ctx: np.recarray, imts, mean, sig, tau, phi):
        """
        Compute GMM values.
        """
        super().compute(ctx, imts, mean, sig, tau, phi)
        AdjustedGMM.add_adjustment(self, imts, mean, sig, tau, phi)


class AbrahamsonGulerce2020SSlab_USGSPRVI(abrahamson_gulerce_2020.AbrahamsonGulerce2020SSlab):
    """
    Abrahamson and Gulerce (2020) subduction intraslab ground-motion model with 
    USGS adjustments for Puerto Rico and the Virgin Islands.
    """
    
    def __init__(self, sigma_mu_epsilon = 0.0, **kwargs):
        super().__init__(sigma_mu_epsilon=sigma_mu_epsilon, **kwargs)
        self.bias_adjustment = kwargs.get('bias_adjustment')
        assert self.bias_adjustment in ("Mean", "IMT")
        self.mean_adjustment = SUBDUCTION_INTRASLAB_MEAN_ADJUSTMENT
        self.imt_adjustments = COEFFS_SUBDUCTION_INTRASLAB_ADJUSTMENTS

    def compute(self, ctx: np.recarray, imts, mean, sig, tau, phi):
        """
        Compute GMM values.
        """
        super().compute(ctx, imts, mean, sig, tau, phi)
        AdjustedGMM.add_adjustment(self, imts, mean, sig, tau, phi)


class KuehnEtAl2020SSlab_USGSPRVI(kuehn_2020.KuehnEtAl2020SSlab):
    """
    Kuehn et al. (2020) subduction intraslab ground-motion model with 
    USGS adjustments for Puerto Rico and the Virgin Islands.
    """
    
    def __init__(self, sigma_mu_epsilon = 0.0, **kwargs):
        super().__init__(sigma_mu_epsilon=sigma_mu_epsilon, **kwargs)
        self.bias_adjustment = kwargs.get('bias_adjustment')
        assert self.bias_adjustment in ("Mean", "IMT")
        self.mean_adjustment = SUBDUCTION_INTRASLAB_MEAN_ADJUSTMENT
        self.imt_adjustments = COEFFS_SUBDUCTION_INTRASLAB_ADJUSTMENTS

    def compute(self, ctx: np.recarray, imts, mean, sig, tau, phi):
        """
        Compute GMM values.
        """
        super().compute(ctx, imts, mean, sig, tau, phi)
        AdjustedGMM.add_adjustment(self, imts, mean, sig, tau, phi)


class ParkerEtAl2020SSlab_USGSPRVI(parker_2020.ParkerEtAl2020SSlab):
    """
    Parker et al. (2020) subduction intraslab ground-motion model with 
    USGS adjustments for Puerto Rico and the Virgin Islands.
    """
    
    def __init__(self, sigma_mu_epsilon = 0.0, **kwargs):
        super().__init__(sigma_mu_epsilon=sigma_mu_epsilon, **kwargs)
        self.bias_adjustment = kwargs.get('bias_adjustment')
        assert self.bias_adjustment in ("Mean", "IMT")
        self.mean_adjustment = SUBDUCTION_INTRASLAB_MEAN_ADJUSTMENT
        self.imt_adjustments = COEFFS_SUBDUCTION_INTRASLAB_ADJUSTMENTS

    def compute(self, ctx: np.recarray, imts, mean, sig, tau, phi):
        """
        Compute GMM values.
        """
        super().compute(ctx, imts, mean, sig, tau, phi)
        AdjustedGMM.add_adjustment(self, imts, mean, sig, tau, phi)


# Create aliases for adjusted models                
for adjustment in ('Mean', 'IMT'):
    add_alias('AbrahamsonEtAl2014_USGSPRVI_Adj' + adjustment, AbrahamsonEtAl2014_USGSPRVI, bias_adjustment=adjustment)
    add_alias('BooreEtAl2014_USGSPRVI_Adj' + adjustment, BooreEtAl2014_USGSPRVI, bias_adjustment=adjustment)
    add_alias('CampbellBozorgnia2014_USGSPRVI_Adj' + adjustment, CampbellBozorgnia2014_USGSPRVI, bias_adjustment=adjustment)
    add_alias('ChiouYoungs2014_USGSPRVI_Adj' + adjustment, ChiouYoungs2014_USGSPRVI, bias_adjustment=adjustment)
    add_alias('AbrahamsonGulerce2020SInter_USGSPRVI_Adj' + adjustment, AbrahamsonGulerce2020SInter_USGSPRVI, bias_adjustment=adjustment)
    add_alias('KuehnEtAl2020SInter_USGSPRVI_Adj' + adjustment, KuehnEtAl2020SInter_USGSPRVI, bias_adjustment=adjustment)
    add_alias('ParkerEtAl2020SInter_USGSPRVI_Adj' + adjustment, ParkerEtAl2020SInter_USGSPRVI, bias_adjustment=adjustment)
    add_alias('AbrahamsonGulerce2020SSlab_USGSPRVI_Adj' + adjustment, AbrahamsonGulerce2020SSlab_USGSPRVI, bias_adjustment=adjustment)
    add_alias('KuehnEtAl2020SSlab_USGSPRVI_Adj' + adjustment, KuehnEtAl2020SSlab_USGSPRVI, bias_adjustment=adjustment)
    add_alias('ParkerEtAl2020SSlab_USGSPRVI_Adj' + adjustment, ParkerEtAl2020SSlab_USGSPRVI, bias_adjustment=adjustment)
