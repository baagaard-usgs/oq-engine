Classical Hazard QA Test, Case 1
================================

============== ===================
checksum32     1,984,592,463      
date           2018-02-25T06:43:26
engine_version 2.10.0-git1f7c0c0  
============== ===================

num_sites = 1, num_levels = 6

Parameters
----------
=============================== ==================
calculation_mode                'classical'       
number_of_logic_tree_samples    0                 
maximum_distance                {'default': 200.0}
investigation_time              1.0               
ses_per_logic_tree_path         1                 
truncation_level                2.0               
rupture_mesh_spacing            1.0               
complex_fault_mesh_spacing      1.0               
width_of_mfd_bin                1.0               
area_source_discretization      None              
ground_motion_correlation_model None              
minimum_intensity               {}                
random_seed                     1066              
master_seed                     0                 
ses_seed                        42                
=============================== ==================

Input files
-----------
=============== ============================================
Name            File                                        
=============== ============================================
gsim_logic_tree `gsim_logic_tree.xml <gsim_logic_tree.xml>`_
job_ini         `job.ini <job.ini>`_                        
source          `source_model.xml <source_model.xml>`_      
source_model    `source_model.xml <source_model.xml>`_      
=============== ============================================

Composite source model
----------------------
========= ====== =============== ================
smlt_path weight gsim_logic_tree num_realizations
========= ====== =============== ================
b1        1.000  trivial(1)      1/1             
========= ====== =============== ================

Required parameters per tectonic region type
--------------------------------------------
====== ================ ========= ========== ==========
grp_id gsims            distances siteparams ruptparams
====== ================ ========= ========== ==========
0      SadighEtAl1997() rrup      vs30       mag rake  
====== ================ ========= ========== ==========

Realizations per (TRT, GSIM)
----------------------------

::

  <RlzsAssoc(size=1, rlzs=1)
  0,SadighEtAl1997(): [0]>

Number of ruptures per tectonic region type
-------------------------------------------
================ ====== ==================== ============ ============
source_model     grp_id trt                  eff_ruptures tot_ruptures
================ ====== ==================== ============ ============
source_model.xml 0      Active Shallow Crust 1.000        1           
================ ====== ==================== ============ ============

Informational data
------------------
======================= =========================================================================
count_ruptures.received max_per_task 822 B, tot 822 B                                            
count_ruptures.sent     sources 1.15 KB, srcfilter 722 B, param 513 B, monitor 330 B, gsims 120 B
hazard.input_weight     0.1                                                                      
hazard.n_imts           2                                                                        
hazard.n_levels         6                                                                        
hazard.n_realizations   1                                                                        
hazard.n_sites          1                                                                        
hazard.n_sources        1                                                                        
hazard.output_weight    6.0                                                                      
hostname                tstation.gem.lan                                                         
require_epsilons        False                                                                    
======================= =========================================================================

Slowest sources
---------------
========= ============ ============ ========= ========= =========
source_id source_class num_ruptures calc_time num_sites num_split
========= ============ ============ ========= ========= =========
1         PointSource  1            1.340E-04 2         1        
========= ============ ============ ========= ========= =========

Computation times by source typology
------------------------------------
============ ========= ======
source_class calc_time counts
============ ========= ======
PointSource  1.340E-04 1     
============ ========= ======

Duplicated sources
------------------
There are no duplicated sources

Information about the tasks
---------------------------
================== ===== ====== ===== ===== =========
operation-duration mean  stddev min   max   num_tasks
count_ruptures     0.002 NaN    0.002 0.002 1        
================== ===== ====== ===== ===== =========

Slowest operations
------------------
============================== ========= ========= ======
operation                      time_sec  memory_mb counts
============================== ========= ========= ======
store source_info              0.004     0.0       1     
managing sources               0.002     0.0       1     
total count_ruptures           0.002     0.0       1     
reading composite source model 0.001     0.0       1     
reading site collection        6.270E-05 0.0       1     
saving probability maps        3.195E-05 0.0       1     
aggregate curves               2.170E-05 0.0       1     
============================== ========= ========= ======