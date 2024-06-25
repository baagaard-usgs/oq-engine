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

ACTIVE_CRUSTAL_MEAN_ADJUSTMENT = -0.371
COEFFS_ACTIVE_CRUSTAL_ADJUSTMENTS = CoeffsTable(sa_damping=5, table="""\
IMT   bias
pgv     -0.452
pga     -0.329
0.010   -0.329
0.020   -0.301
0.030   -0.274
0.050   -0.246
0.075   -0.155
0.100   -0.188
0.150   -0.142
0.200   -0.132
0.250   -0.195
0.300   -0.289
0.400   -0.470
0.500   -0.598
0.750   -0.679
1.000   -0.661
1.500   -0.588
2.000   -0.483
3.000   -0.319
4.000   -0.171
5.000    0.001
7.500    0.258
10.000   0.426
""")

SUBDUCTION_INTERFACE_MEAN_ADJUSTMENT = -1.098
COEFFS_SUBDUCTION_INTERFACE_ADJUSTMENTS = CoeffsTable(sa_damping=5, table="""\
IMT   bias
pgv     -1.189
pga     -0.888
0.010   -0.888
0.020   -0.804
0.030   -0.720
0.050   -0.637
0.075   -0.621
0.100   -0.601
0.150   -0.575
0.200   -0.839
0.250   -0.852
0.300   -0.971
0.400   -1.250
0.500   -1.424
0.750   -1.636
1.000   -1.692
1.500   -1.658
2.000   -1.512
3.000   -1.256
4.000   -1.123
5.000   -0.985
7.500   -0.776
10.000  -0.639
""")

SUBDUCTION_INTRASLAB_MEAN_ADJUSTMENT = -0.405
COEFFS_SUBDUCTION_INTRASLAB_ADJUSTMENTS = CoeffsTable(sa_damping=5, table="""\
IMT   bias
pgv     -0.476
pga     -0.376
0.010   -0.376
0.020   -0.239
0.030   -0.103
0.050    0.034
0.075   -0.044
0.100   -0.234
0.150   -0.016
0.200   -0.241
0.250   -0.234
0.300   -0.302
0.400   -0.428
0.500   -0.570
0.750   -0.757
1.000   -0.837
1.500   -0.849
2.000   -0.780
3.000   -0.466
4.000   -0.282
5.000   -0.056
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
