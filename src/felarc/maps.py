from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import sympy as sp  # type: ignore
from ocelot.cpbd.beam import Twiss
from ocelot.cpbd.elements import Drift, Marker, Quadrupole, SBend
from scipy.sparse import spmatrix

from .constants import m_e_GeV


@dataclass
class OpticalParameters:
    beta_x: float
    alpha_x: float
    beta_y: float
    alpha_y: float
    dx: float = 0.0
    dxp: float = 0.0
    dy: float = 0.0
    dyp: float = 0.0

    @property
    def gamma_x(self):
        return (1 + self.alpha_x ** 2) / self.beta_x

    @property
    def gamma_y(self):
        return (1 + self.alpha_y ** 2) / self.beta_y

    def to_twiss(self) -> Twiss:
        return Twiss(beta_x=float(self.beta_x),
                     alpha_x=float(self.alpha_x),
                     beta_y=float(self.beta_y),
                     alpha_y=float(self.alpha_y),
                     Dx=float(self.dx),
                     Dy=float(self.dy),
                     Dxp=float(self.dxp),
                     Dyp=float(self.dyp))

    def _latex(self, val):
        """Helper: format numbers or sympy expressions as LaTeX."""
        if isinstance(val, (int, float)):
            return f"{val:.4g}"
        return sp.latex(val)

    def _repr_latex_(self):
        return r"""
$$
\begin{array}{l r}
\hline
\textbf{Parameter} & \textbf{Value} \\
\hline
\beta_x & """ + self._latex(self.beta_x) + r""" \\
\alpha_x & """ + self._latex(self.alpha_x) + r""" \\
\gamma_x & """ + self._latex(self.gamma_x) + r""" \\
\beta_y & """ + self._latex(self.beta_y) + r""" \\
\alpha_y & """ + self._latex(self.alpha_y) + r""" \\
\gamma_y & """ + self._latex(self.gamma_y) + r""" \\
\Delta x & """ + self._latex(self.dx) + r""" \\
\Delta x' & """ + self._latex(self.dxp) + r""" \\
\Delta y & """ + self._latex(self.dy) + r""" \\
\Delta y' & """ + self._latex(self.dyp) + r""" \\
\hline
\end{array}
$$
"""
#     def _repr_latex_(self):
#         return rf"""
# $$
# \begin{{array}}{{l r}}
# \hline
# \textbf{{Parameter}} & \textbf{{Value}} \\
# \hline
# \beta_x & {self.beta_x:.4g} \\
# \alpha_x & {self.alpha_x:.4g} \\
# \beta_y & {self.beta_y:.4g} \\
# \alpha_y & {self.alpha_y:.4g} \\
# D_x & {self.dx:.4g} \\
# D_x' & {self.dxp:.4g} \\
# \hline
# \end{{array}}
# $$
# """
# D_y & {self.dy:.4g} \\
# D_y' & {self.dyp:.4g} \\

#     def _repr_latex_(self):
#         # Use raw f-string; double braces to keep LaTeX braces literal
#         return rf"""
# $$
# \begin{{aligned}}
# \beta_x &= {self.beta_x:.4g} &\quad \alpha_x &= {self.alpha_x:.4g} &\quad \gamma_x &= {self.gamma_x:.4g} \\
# \beta_y &= {self.beta_y:.4g} &\quad \alpha_y &= {self.alpha_y:.4g} &\quad \gamma_y &= {self.gamma_y:.4g} \\
# \Delta x &= {self.dx:.4g} &\quad \Delta x' &= {self.dxp:.4g} &\quad \Delta y &= {self.dy:.4g} &\quad \Delta y' &= {self.dyp:.4g}
# \end{{aligned}}
# $$
# """
        

class BeamTransformation:
    def __init__(self, rmat: sp.Matrix) -> None:
        self.rmat = rmat

    def __mul__(self, other: BeamTransformation | OpticalParameters) -> BeamTransformation | OpticalParameters:
        rmat = self.rmat
        if isinstance(other, BeamTransformation):
            return BeamTransformation(rmat * other.rmat)
        elif isinstance(other, OpticalParameters):
            # Make twiss vector for x and y
            xtwiss0 = sp.Matrix([other.beta_x, other.alpha_x, other.gamma_x])
            ytwiss0 = sp.Matrix([other.beta_y, other.alpha_y, other.gamma_y])
            # Get the twiss transport matrix.
            xtwissmat = get_cs_transport_matrix(rmat, "x")
            ytwissmat = get_cs_transport_matrix(rmat, "y")

            xtwiss1 = xtwissmat * xtwiss0
            # from IPython import embed; embed()
            ytwiss1 = ytwissmat * ytwiss0

            out = OpticalParameters(alpha_x=xtwiss1[1],
                                    beta_x=xtwiss1[0],
                                    alpha_y=ytwiss1[1],
                                    beta_y=ytwiss1[0])

            dx0 = other.dx
            dxp0 = other.dxp
            dy0 = other.dy
            dyp0 = other.dyp

            # Dispersive trajectory is just like any other trajectory, but defined to have dp/p = 1.

            out.dx = rmat[0, 0] * dx0 + rmat[0, 1] * dxp0 + rmat[0, 5] # type: ignore
            out.dy = rmat[2, 2] * dy0 + rmat[2, 3] * dyp0 + rmat[2, 5] # type: ignore
            out.dxp = rmat[1, 0] * dx0 + rmat[1, 1] * dxp0 + rmat[1, 5] # type: ignore
            out.dyp = rmat[3, 2] * dy0 + rmat[3, 3] * dyp0 + rmat[3, 5] # type: ignore

            return out

    @classmethod
    def eye(cls) -> BeamTransformation:
        rmat = sp.eye(6)
        return BeamTransformation(rmat)

    def _repr_latex_(self):
        return self.rmat._repr_latex_()

    @property
    def rmatnp(self):
        return np.array(self.rmat).astype(np.float64)

def drift(energy: float, length = None) -> BeamTransformation:
    """Define a drift matrix with optionally symbolic length.  If kwarg `length` is a string,
    then it's used as a subscript symbolically.

    """
    relgamma = energy / m_e_GeV
    relbeta = (1 - relgamma**-2)**-0.5

    if length is None:
        length = sp.Symbol("l", real=True, positive=True)
    elif isinstance(length, str):
        length = sp.Symbol(f"l_{length}", real=True, positive=True)

    return BeamTransformation(sp.Matrix([
        [1, length, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, length, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, -length/(relbeta**2 * relgamma**2)],
        [0, 0, 0, 0, 0, 1],
    ]))


def thin_quadrupole(k1l=None) -> BeamTransformation:
    """Define a thin quadrupole matrix with optionally symbolic integrated strength.
    If kwarg `k1l` is a string, then it's used as a subscript for the symbol.

    """
    if k1l is None:
        k1l = sp.Symbol("K", real=True)
    elif isinstance(k1l, str):
        k1l = sp.Symbol(f"K_{k1l}", real=True)

    return BeamTransformation(sp.Matrix([
        [1, 0, 0, 0, 0, 0],
        [-k1l, 1, 0, 0, 0, 0],   # x' -> x' - k1l*x
        [0, 0, 1, 0, 0, 0],
        [0, 0,  k1l, 1, 0, 0],   # y' -> y' + k1l*y
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
    ]))


def sbend(length, angle, energy) -> BeamTransformation:
    """SBend is for now never symbolic as we assume it's fixed."""
    _, R, _ = SBend(l=length, angle=angle).R(energy=energy)
    return BeamTransformation(sp.Matrix(R))


def sequence_to_map(sequence, energy: float, variables=None) -> BeamTransformation:
    fullmap = BeamTransformation.eye()
    if variables is None:
        variables = set()
    for element in sequence:
        map = sp.eye(6)
        if isinstance(element, Marker):
            continue
        elif isinstance(element, SBend):
           map = sbend(element.l, element.angle, energy)
        elif isinstance(element, Quadrupole):
            if element in variables:
                map = thin_quadrupole(k1l=element.id)
            else:
                map = thin_quadrupole(k1l=element.k1l)
        elif isinstance(element, Drift):
            if element in variables:
                map = drift(energy=energy, length=element.id)
            else:
                map = drift(energy=energy, length=element.l)
        fullmap = map * fullmap
    return fullmap


def get_cs_transport_matrix(mat: sp.Matrix, dim="x") -> sp.Matrix:
    # cs vector: [beta, alpha, gamma]^T
    if dim == "x":
        c = mat[0, 0]
        s = mat[0, 1]
        cprime = mat[1, 0]
        sprime = mat[1, 1]
    elif dim == "y":
        c = mat[2, 2]
        s = mat[2, 3]
        cprime = mat[3, 2]
        sprime = mat[3, 3]
    else:
        raise ValueError(f"Unrecognised dimension: {dim}")
    
    beta_row = [c**2, - 2 * c * s, s**2] # type: ignore
    alpha_row = [-c * cprime, cprime * s + sprime * c, -s * sprime] # type: ignore
    gamma_row = [cprime**2, -2*sprime*cprime, sprime**2] # type: ignore
    csmat = sp.Matrix([beta_row, alpha_row, gamma_row]) # type: ignore

    return csmat
