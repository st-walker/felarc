from ocelot import *

#Initial Twiss parameters
tws0 = Twiss()
tws0.beta_x = 22.5476
tws0.beta_y = 22.6465
tws0.alpha_x = -1.462
tws0.alpha_y = 1.3978
tws0.E = 14.0

# Drifts
d_0 = Drift(l=0.15999999999985448, eid='D_0')
d_1 = Drift(l=0.07000000000016371, eid='D_1')
d_5 = Drift(l=0.20499999999992724, eid='D_5')
d_6 = Drift(l=0.9006999999999152, eid='D_6')
d_7 = Drift(l=0.15430000000014843, eid='D_7')
d_8 = Drift(l=0.20895100000007005, eid='D_8')
d_9 = Drift(l=0.1539510000000064, eid='D_9')
d_10 = Drift(l=0.20000000000027285, eid='D_10')
d_11 = Drift(l=7.78469999999993, eid='D_11')
d_12 = Drift(l=0.09999999999990905, eid='D_12')
d_13 = Drift(l=1.4345999999998185, eid='D_13')
d_14 = Drift(l=0.34999999999990905, eid='D_14')
vcv100_2053_tl = Drift(l=0.085, eid='VCV100.2053.TL')
d_15 = Drift(l=0.585700000000088, eid='D_15')
vcb100_2054_tl = Drift(l=0.18, eid='VCB100.2054.TL')
vcabsa_2054_tl = Drift(l=0.61, eid='VCABSA.2054.TL')
d_16 = Drift(l=0.25, eid='D_16')

# Quadrupoles
# qf_2042_tl = Quadrupole(l=0.5321, k1=0.17919084760007517, eid='QF.2042.TL')
qf_2042_tl = SBend(l=0.5321,
                   angle=0, # Set vertical bend to 0.0 for horizontal model
                   k1=-0.1791908475850404,
                   e1=0.0007548342024,
                   e2=0.0011902429655,
                   tilt=1.570796326794897,
                   eid='QF.1.T20.TL')

# RBends
kl_2035_tl = RBend(l=0.9300000055196924,
                   angle=0, # Set vertical bend to 0.0 for horizontal model.
                   e1=-0.0001887085517200141,
                   e2=-0.0001887085517200141,
                   tilt=1.570796326794897,
                   fintx=-1.0,
                   eid='KL.20.TL')
kl_2036_tl = RBend(l=0.9300000055196924,
                   angle=0, # Set vertical bend to 0.0 for horizontal model.
                   e1=-0.0001887085517200141,
                   e2=-0.0001887085517200141,
                   tilt=1.570796326794897,
                   fintx=-1.0,
                   eid='KL.20.TL')
# kl_2035_tl = RBend(l=0.93, tilt=1.570796327, eid='KL.2035.TL')
# kl_2036_tl = RBend(l=0.93, tilt=1.570796327, eid='KL.2036.TL')
kl_2037_tl = RBend(l=0.93, tilt=1.570796327, eid='KL.2037.TL')
kl_2038_tl = RBend(l=0.93, tilt=1.570796327, eid='KL.2038.TL')
kl_2039_tl = RBend(l=0.93, tilt=1.570796327, eid='KL.2039.TL')

# Hcors
chx_2042_tl = Hcor(l=0.2, eid='CHX.2042.TL')
cnx_2051_tl = Hcor(l=0.3, eid='CNX.2051.TL')

# Vcors
cmy_2034_tl = Vcor(l=0.3, eid='CMY.2034.TL')
cmy_2040_tl = Vcor(l=0.3, eid='CMY.2040.TL')
chy_2042_tl = Vcor(l=0.2, eid='CHY.2042.TL')
cny_2051_tl = Vcor(l=0.3, eid='CNY.2051.TL')

# Monitors
bpma_2041_tl = Monitor(eid='BPMA.2041.TL')
bpmd_2052_tl = Monitor(eid='BPMD.2052.TL')

# Markers
stsub_2034_tl = Marker(eid='STSUB.2034.TL')
tora_2041_tl = Marker(eid='TORA.2041.TL')
ensub_2055_tl = Marker(eid='ENSUB.2055.TL')

# Lattice
cell = (stsub_2034_tl, cmy_2034_tl, d_0, kl_2035_tl, d_1, kl_2036_tl, d_1, kl_2037_tl, d_1,
kl_2038_tl, d_1, kl_2039_tl, d_5, cmy_2040_tl, d_6, tora_2041_tl, d_7, bpma_2041_tl, d_8,
qf_2042_tl, d_9, chx_2042_tl, d_10, chy_2042_tl, d_11, cnx_2051_tl, d_12, cny_2051_tl, d_13,
bpmd_2052_tl, d_14, vcv100_2053_tl, d_15, vcb100_2054_tl, vcabsa_2054_tl, d_16, ensub_2055_tl)

# power supplies

#
qf_2042_tl.ps_id = 'QF.1.TL'

#

#

#

#
kl_2035_tl.ps_id = 'KL.2035.TL'
kl_2036_tl.ps_id = 'KL.2036.TL'
kl_2037_tl.ps_id = 'KL.2037.TL'
kl_2038_tl.ps_id = 'KL.2038.TL'
kl_2039_tl.ps_id = 'KL.2039.TL'
