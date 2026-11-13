from ocelot import * 

# Drifts
d1 = Drift(l=0.817, eid='d1')
d2 = Drift(l=0.331, eid='d2')
d3 = Drift(l=0.102, eid='d3')
d4 = Drift(l=1.315, eid='d4')
d5 = Drift(l=0.125, eid='d5')
d6 = Drift(l=0.139, eid='d6')
d7 = Drift(l=0.37, eid='d7')

# Quadrupoles
q3 = Quadrupole(l=0.2, k1=7.48, eid='q3')
q1 = Quadrupole(l=0.2, k1=-8.889, eid='q1')
q2 = Quadrupole(l=0.2, k1=9.293, eid='q2')

# SBends
bend = SBend(l=0.4, angle=0.03490658503988659, eid='bend')
half_bend = SBend(l=0.2, angle=0.017453292519943295, eid='half-bend')

# Sextupoles
sd = Sextupole(l=0.3, k2=-551.934, eid='sd')
sf = Sextupole(l=0.3, k2=321.474, eid='sf')

# Markers
start_bend1 = Marker(eid='start_bend1')
stop_bend1 = Marker(eid='stop_bend1')
start_bend2 = Marker(eid='start_bend2')
stop_bend2 = Marker(eid='stop_bend2')
start_bend3 = Marker(eid='start_bend3')
stop_bend3 = Marker(eid='stop_bend3')

# Lattice 
cell = (d1, start_bend1, bend, stop_bend1, d2, sd, d3, q3, d4, 
q1, d5, q2, d6, sf, d7, start_bend2, half_bend, half_bend, stop_bend2, 
d7, sf, d6, q2, d5, q1, d4, q3, d3, sd, 
d2, start_bend3, bend, stop_bend3, d1)
