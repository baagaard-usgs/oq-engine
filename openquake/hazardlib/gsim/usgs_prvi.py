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

ACTIVE_CRUSTAL_MEAN_ADJUSTMENT = -0.461
COEFFS_ACTIVE_CRUSTAL_ADJUSTMENTS = CoeffsTable(sa_damping=5, table="""\
IMT   bias
pgv     -0.590
pga     -0.470
0.010   -0.470
0.020   -0.396
0.030   -0.322
0.050   -0.247
0.075   -0.192
0.100   -0.281
0.150   -0.243
0.200   -0.260
0.250   -0.318
0.300   -0.410
0.400   -0.595
0.500   -0.732
0.750   -0.777
1.000   -0.735
1.500   -0.653
2.000   -0.544
3.000   -0.350
4.000   -0.203
5.000   -0.007
7.500    0.278
10.000   0.453
""")

SUBDUCTION_INTERFACE_MEAN_ADJUSTMENT = -0.948
COEFFS_SUBDUCTION_INTERFACE_ADJUSTMENTS = CoeffsTable(sa_damping=5, table="""\
IMT   bias
pgv     -1.088
pga     -0.769
0.010   -0.769
0.020   -0.552
0.030   -0.335
0.050   -0.118
0.075   -0.539
0.100   -0.567
0.150   -0.479
0.200   -0.663
0.250   -0.700
0.300   -0.816
0.400   -1.081
0.500   -1.260
0.750   -1.539
1.000   -1.608
1.500   -1.527
2.000   -1.422
3.000   -1.102
4.000   -0.938
5.000   -0.822
7.500   -0.659
10.000  -0.502
""")

SUBDUCTION_INTRASLAB_MEAN_ADJUSTMENT = -0.498
COEFFS_SUBDUCTION_INTRASLAB_ADJUSTMENTS = CoeffsTable(sa_damping=5, table="""\
IMT   bias
pgv     -0.552
pga     -0.461
0.010   -0.461
0.020   -0.343
0.030   -0.226
0.050   -0.108
0.075   -0.211
0.100   -0.320
0.150   -0.135
0.200   -0.320
0.250   -0.297
0.300   -0.373
0.400   -0.497
0.500   -0.634
0.750   -0.838
1.000   -0.905
1.500   -0.928
2.000   -0.903
3.000   -0.579
4.000   -0.405
5.000   -0.209
7.500    0.142
10.000   0.508
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
