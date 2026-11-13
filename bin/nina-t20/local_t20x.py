import numpy as np

from ocelot import *
from ocelot.cpbd.elements import Drift, Quadrupole, SBend, Marker, Monitor

from math import pi
#Initial Twiss parameters
tws0 = Twiss()
tws0.beta_x = 11.3792
tws0.beta_y = 42.8892
tws0.alpha_x = 0.7046
tws0.alpha_y = -2.1571
tws0.E = 14.0

# Drifts
d_17 = Drift(l=2.000043000000005, eid='D_17')
d_18 = Drift(l=0.500001999999995, eid='D_18')
d_20 = Drift(l=4.75, eid='D_20')
d_21 = Drift(l=3.025000000000091, eid='D_21')
d_22 = Drift(l=0.21044999999980973, eid='D_22')
d_23 = Drift(l=0.15544999999974607, eid='D_23')
d_24 = Drift(l=1.9980000000000473, eid='D_24')
d_25 = Drift(l=0.2104500000000371, eid='D_25')
d_26 = Drift(l=0.15545000000020082, eid='D_26')
d_27 = Drift(l=2.52599999999984, eid='D_27')
d_30 = Drift(l=0.9400019999998221, eid='D_30')
d_31 = Drift(l=0.5000059999997575, eid='D_31')
d_32 = Drift(l=0.5000039999999899, eid='D_32')
d_33 = Drift(l=0.40000300000019706, eid='D_33')
d_36 = Drift(l=0.15500199999996767, eid='D_36')
d_39 = Drift(l=0.8520010000002003, eid='D_39')
d_42 = Drift(l=2.324999999999818, eid='D_42')
d_45 = Drift(l=4.6083999999996195, eid='D_45')
d_46 = Drift(l=0.21045000000049185, eid='D_46')
d_48 = Drift(l=1.5250019999998585, eid='D_48')
d_49 = Drift(l=0.5000050000003284, eid='D_49')
d_51 = Drift(l=0.277002999999695, eid='D_51')
d_54 = Drift(l=2.467999999999847, eid='D_54')
d_57 = Drift(l=0.5994500000001608, eid='D_57')
d_58 = Drift(l=0.48545100000001185, eid='D_58')
d_59 = Drift(l=1.099999999999909, eid='D_59')
d_60 = Drift(l=0.2699999999999818, eid='D_60')
d_61 = Drift(l=0.9600000000000364, eid='D_61')
d_63 = Drift(l=0.9000009999999747, eid='D_63')

# Quadrupoles
ql_2070_t20 = Quadrupole(l=1.0291, k1=0.3056555719998057, eid='QL.2070.T20')
ql_2074_t20 = Quadrupole(l=1.0291, k1=-0.30713706419978626, eid='QL.2074.T20')
ql_2078_t20 = Quadrupole(l=1.0291, k1=0.37053283619959193, eid='QL.2078.T20')
ql_2085_t20 = Quadrupole(l=1.0291, k1=-0.23124472729958218, eid='QL.2085.T20')
ql_2092_t20 = Quadrupole(l=1.0291, k1=0.3603954642998737, eid='QL.2092.T20')
ql_2096_t20 = Quadrupole(l=1.0291, k1=-0.3110947717996308, eid='QL.2096.T20')
ql_2102_t20 = Quadrupole(l=1.0291, k1=0.2421705056000389, eid='QL.2102.T20')
ql_2109_t20 = Quadrupole(l=1.0291, k1=-0.18471154050043728, eid='QL.2109.T20')
ql_2114_t20 = Quadrupole(l=1.0291, k1=0.2014881282003693, eid='QL.2114.T20')
ql_2116_t20 = Quadrupole(l=1.0291, k1=-0.061285903509862995, eid='QL.2116.T20')

# SBends
# initial tilt for  bz_2055_t20 = -0.193652074rad.  Set tilt to 0 and reduce kick by cos(tilt) to
# retain just the horizontal bending.
bz_2055_t20 = SBend(l=1.0, angle=0.005500006932*np.cos(-0.193652074), e1=0.002750003, e2=0.002750003, tilt=0, eid='BZ.2055.T20')
bz_2058_t20 = SBend(l=1.0, angle=0.005500006932, e1=0.002750003, e2=0.002750003, eid='BZ.2058.T20')
bz_2060_t20 = SBend(l=1.0, angle=0.005500006932, e1=0.002750003, e2=0.002750003, eid='BZ.2060.T20')
bz_2061_t20 = SBend(l=1.0, angle=0.005500006932, e1=0.002750003, e2=0.002750003, eid='BZ.2061.T20')
bd_2080_t20 = SBend(l=1.0, angle=0.0107798589, e1=0.005389929, e2=0.005389929, eid='BD.2080.T20')
bd_2082_t20 = SBend(l=1.0, angle=0.0107798589, e1=0.005389929, e2=0.005389929, eid='BD.2082.T20')
bd_2083_t20 = SBend(l=1.0, angle=0.0107798589, e1=0.005389929, e2=0.005389929, eid='BD.2083.T20')
bd_2087_t20 = SBend(l=1.0, angle=0.0105626766, e1=0.005281338, e2=0.005281338, eid='BD.2087.T20')
bd_2088_t20 = SBend(l=1.0, angle=0.0105626766, e1=0.005281338, e2=0.005281338, eid='BD.2088.T20')
bd_2090_t20 = SBend(l=1.0, angle=0.0105626766, e1=0.005281338, e2=0.005281338, eid='BD.2090.T20')
bd_2105_t20 = SBend(l=1.0, angle=0.0103184545, e1=0.005159227, e2=0.005159227, eid='BD.2105.T20')
bd_2106_t20 = SBend(l=1.0, angle=0.0103184545, e1=0.005159227, e2=0.005159227, eid='BD.2106.T20')
bd_2108_t20 = SBend(l=1.0, angle=0.0103184545, e1=0.005159227, e2=0.005159227, eid='BD.2108.T20')
# Set vertical bend to 0.0 for horizontal model:
bd_2117_t20 = SBend(l=1.0, angle=0, e1=0.002345482, e2=0.002345482, tilt=1.570796327, eid='BD.2117.T20')
# Set vertical bend to 0.0 for horizontal model:
bd_2122_t20 = SBend(l=1.0, angle=0, e1=-0.002345482, e2=-0.002345482, tilt=1.570796327, eid='BD.2122.T20')


# ------------------ BEGIN DEFINITION OF QK ("XY") QUADRUPOLE --------------------------
# Copy/pasted from Nina's MAD script with MAD syntax transformed into corresponding
# Python (OCELOT) syntax by hand.  Original MAD8 definitions left, commented
# For clarity as to where the Python comes from.

#--- Quad QK.2.T20.TL is sliced in N slices: Exception
#      1) vertically focusing quadrupole, k1 < 0
#      2) trajectory is shifted in both planes at the entrance => hor and ver dipole

#--- Horizontal trajectory at entrance and exit of QK.2.TL

#xE1_QK_T20 = -0.0053973151221;   # from Matlab code
xE1_QK_T20 = -5.397213228E-03    # from MAD8, used
xE2_QK_T20 = -0.0061799810553    # from Matlab code

angleX_T20 = -(xE2_QK_T20 - xE1_QK_T20);


#--- Vertical trajectory at entrance and exit of QK.2.TL

#yE1_QK_T20 = +0.0022480661411;   # from Matlab code
yE1_QK_T20 = +2.248663716E-03;    # from MAD8, used
yE2_QK_T20 = +0.0;                # from Matlab code

angleY_T20 = -(yE2_QK_T20 - yE1_QK_T20);    # sign (+) to deflect down (to negative Y)


#--- QK.2.TL is splitted in 200 parts: 10 hor and 190 ver slices

z3_slice = 200                  # total number of slices
x3_slice = 10                   # number of X-slices, for angle
y3_slice = z3_slice - x3_slice;  # number of Y-slices, for angle


# xQK.2.TL.T20: SBEND, L = L_QK_eff/z3_slice, ANGLE = angleX_T20/x3_slice,
#                      E1 = 0, E2 = 0,
#                      K1 = -KFscale*(0.5321/1.0552);

KFscale = 0.1906949*0.5/0.5321    #  0.17919084758504
l_qk_eff  = 1.0552
xqk_2_tl_t20 = SBend(l = l_qk_eff/z3_slice, angle = angleX_T20/x3_slice,
                     e1 = 0, e2 = 0,
                     k1 =-KFscale*(0.5321/1.0552),
                     eid="xQK.2.TL.T20")

# yQK.2.TL.T20: SBEND, L = l_qk_eff/z3_slice, ANGLE = angleY_T20/y3_slice,
#                      E1 = 0, E2 = 0,
#                      K1 = +KFscale*(0.5321/1.0552),
#                      TILT = pi/2;

yqk_2_tl_t20 = SBend(l = l_qk_eff/z3_slice, angle = 0.0,  # Set vertical bend to 0.0 for horizontal model
                     e1 = 0, e2 = 0,
                     k1 = +KFscale*(0.5321/1.0552),
                     tilt = pi/2,
                     eid="yQK.2.TL.T20"
                     )
# QK_T20_slice20: Line = (19 * yQK.2.TL.T20, 1 * xQK.2.TL.T20);
qk_t20_slice20 = [19 * [yqk_2_tl_t20], xqk_2_tl_t20]
# QK.2.T20.TL: Line = (10 * QK_T20_slice20);
qk_2_t20_tl = 10 * qk_t20_slice20

# ------------------ END DEFINITION OF QK ("XY") QUADRUPOLE --------------------------

# Hcors
cmx_2071_t20 = Hcor(l=0.3, eid='CMX.2071.T20')
cmx_2079_t20 = Hcor(l=0.3, eid='CMX.2079.T20')
cmx_2092_t20 = Hcor(l=0.3, eid='CMX.2092.T20')
cmx_2103_t20 = Hcor(l=0.3, eid='CMX.2103.T20')
cmx_2114_t20 = Hcor(l=0.3, eid='CMX.2114.T20')
cmx_2121_t20 = Hcor(l=0.3, eid='CMX.2121.T20')

# Vcors
cmy_2075_t20 = Vcor(l=0.3, eid='CMY.2075.T20')
cmy_2086_t20 = Vcor(l=0.3, eid='CMY.2086.T20')
cmy_2097_t20 = Vcor(l=0.3, eid='CMY.2097.T20')
cmy_2110_t20 = Vcor(l=0.3, eid='CMY.2110.T20')
cmy_2119_t20 = Vcor(l=0.3, eid='CMY.2119.T20')

# Monitors
bpma_2069_t20 = Monitor(eid='BPMA.2069.T20')
bpma_2073_t20 = Monitor(eid='BPMA.2073.T20')
bpma_2077_t20 = Monitor(eid='BPMA.2077.T20')
bpma_2084_t20 = Monitor(eid='BPMA.2084.T20')
bpma_2091_t20 = Monitor(eid='BPMA.2091.T20')
bpma_2095_t20 = Monitor(eid='BPMA.2095.T20')
bpma_2101_t20 = Monitor(eid='BPMA.2101.T20')
bpma_2109_t20 = Monitor(eid='BPMA.2109.T20')
bpma_2113_t20 = Monitor(eid='BPMA.2113.T20')
bpma_2119_t20 = Monitor(eid='BPMA.2119.T20')
bpma_2120_t20 = Monitor(eid='BPMA.2120.T20')

# Markers
stsec_2055_t20 = Marker(eid='STSEC.2055.T20')
tora_2066_t20 = Marker(eid='TORA.2066.T20')
ensec_2123_t20 = Marker(eid='ENSEC.2123.T20')

# QK.2.TL placement comment (Hand placed)
# From Nina script: QK quadrupole starts 0.4724 metres after the BZ.2055.T20 (aka BZ.1.T20).
# This therefore requires the subsection (bz_2055_t20, d_17,) to be split.  d_17 is about 2m
# long so there is enough space to place the quad.

d_17_pre_qk = Drift(l=0.4724, eid="D_17_pre_QK")
d_17_post_qk = Drift(l=d_17.l - d_17_pre_qk.l - l_qk_eff, eid="D_17_post_QK")
# Redefine this drift to be a sequence including the QK quad
d_17 = [d_17_pre_qk, qk_2_t20_tl, d_17_post_qk]

# Lattice
cell = (stsec_2055_t20, bz_2055_t20, d_17, bz_2058_t20, d_18, bz_2060_t20, d_18, bz_2061_t20, d_20,
tora_2066_t20, d_21, bpma_2069_t20, d_22, ql_2070_t20, d_23, cmx_2071_t20, d_24, bpma_2073_t20, d_25,
ql_2074_t20, d_26, cmy_2075_t20, d_27, bpma_2077_t20, d_25, ql_2078_t20, d_26, cmx_2079_t20, d_30,
bd_2080_t20, d_31, bd_2082_t20, d_32, bd_2083_t20, d_33, bpma_2084_t20, d_25, ql_2085_t20, d_26,
cmy_2086_t20, d_36, bd_2087_t20, d_32, bd_2088_t20, d_31, bd_2090_t20, d_39, bpma_2091_t20, d_25,
ql_2092_t20, d_26, cmx_2092_t20, d_42, bpma_2095_t20, d_25, ql_2096_t20, d_26, cmy_2097_t20, d_45,
bpma_2101_t20, d_46, ql_2102_t20, d_23, cmx_2103_t20, d_48, bd_2105_t20, d_49, bd_2106_t20, d_32,
bd_2108_t20, d_51, bpma_2109_t20, d_46, ql_2109_t20, d_23, cmy_2110_t20, d_54, bpma_2113_t20, d_46,
ql_2114_t20, d_23, cmx_2114_t20, d_57, ql_2116_t20, d_58, bd_2117_t20, d_59, bpma_2119_t20, d_60,
cmy_2119_t20, d_61, bpma_2120_t20, d_60, cmx_2121_t20, d_63, bd_2122_t20, ensec_2123_t20)

# power supplies

#
ql_2070_t20.ps_id = 'QL.1.T20'
ql_2074_t20.ps_id = 'QL.2.T20'
ql_2078_t20.ps_id = 'QL.3.T20'
ql_2085_t20.ps_id = 'QL.4.T20'
ql_2092_t20.ps_id = 'QL.5.T20'
ql_2096_t20.ps_id = 'QL.6.T20'
ql_2102_t20.ps_id = 'QL.7.T20'
ql_2109_t20.ps_id = 'QL.8.T20'
ql_2114_t20.ps_id = 'QL.9.T20'
ql_2116_t20.ps_id = 'QL.10.T20'

#

#

#

#
bz_2055_t20.ps_id = 'BZ.1.T20'
bz_2058_t20.ps_id = 'BZ.2.T20'
bz_2060_t20.ps_id = 'BZ.2.T20'
bz_2061_t20.ps_id = 'BZ.2.T20'
bd_2080_t20.ps_id = 'BD.1.T20'
bd_2082_t20.ps_id = 'BD.1.T20'
bd_2083_t20.ps_id = 'BD.1.T20'
bd_2087_t20.ps_id = 'BD.2.T20'
bd_2088_t20.ps_id = 'BD.2.T20'
bd_2090_t20.ps_id = 'BD.2.T20'
bd_2105_t20.ps_id = 'BD.3.T20'
bd_2106_t20.ps_id = 'BD.3.T20'
bd_2108_t20.ps_id = 'BD.3.T20'
bd_2117_t20.ps_id = 'BD.4.T20'
bd_2122_t20.ps_id = 'BD.5.T20'
