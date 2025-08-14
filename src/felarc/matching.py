


from copy import deepcopy

from ocelot.cpbd.beam import Twiss
from ocelot.cpbd.beam_params import radiation_integrals
from ocelot.cpbd.elements import (Bend, Drift, Quadrupole, RBend,
                                  SBend, Solenoid)
from ocelot.cpbd.elements.optic_element import OpticElement
from scipy.optimize import fmin, fmin_bfgs, fmin_cg


def weights_default(val):
    if val == 'total_len':
        return 10000001.0
    if val in ['Dx', 'Dy']:
        return 10000002.0
    if val in ['Dxp', 'Dyp']:
        return 10000003.0
    if val == 'tau':
        return 10000004.0
    if val == 'negative_length':
        return 1.5e6
    if val in ['alpha_x', 'alpha_y']:
        return 100007.0
    if val in ['mux', 'muy']:
        return 10000006.0
    if val in ['beta_x', 'beta_y']:
        return 100007.0
    return 0.0001


def match_twiss(lat,
          constr,
          vars,
          tw,
          verbose=True,
          max_iter=1000,
          method='simplex',
          weights=weights_default,
          tol=1e-5):
    """
    Function to match twiss parameters. To find periodic solution for a lattice use MagneticLattice.periodic_twiss(tws)

    :param lat: MagneticLattice
    :param constr: dictionary, constrains. Example:
            'periodic':True - means the "match" function tries to find periodic solution at the ends of lattice:
                constr = {elem1:{'beta_x':15, 'beta_y':2}, 'periodic':True}

            "hard" constrains on the end of elements (elem1, elem2):
                constr = {elem1:{'alpha_x':5, 'beta_y':5}, elem2:{'Dx':0 'Dyp':0, 'alpha_x':5, 'beta_y':5}}

            or mixture of "soft" and "hard" constrains:
                constr = {elem1:{'alpha_x':[">", 5], 'beta_y':5}, elem2:{'Dx':0 'Dyp':0, 'alpha_x':5, 'beta_y':[">", 5]}}

                in case one wants to constrain absolute value of variable, the constrains can be:
                constr = {elem1:{'alpha_x':["a>", 5], "alpha_y": ["a<", 1]}}
                        - which means np.abs(alpha_x) > 5 and np.abs(alpha_y) < 1

            in case one needs global control on beta function, the constrains can be written following way.
                constr = {elem1:{'alpha_x':5, 'beta_y':5}, 'global': {'beta_x': ['>', 10]}}

            Experimental constrain (CAN BE DISABLED or CHANGED AT ANY MOMENT)
                constr = {"delta": {ELEM1: ["muy", 0],  ELEM2: ["muy", 0], "val":  3*np.pi/2, "weight": 100007}}
                        - try to satisfy: tws.muy at ELEM2 - tws.muy at ELEM1 == 'val'
                        - difference between ELEM1 and ELEM2 of twiss parameter "muy" (can be any) == "val"
                        - ELEM1: ["muy", 0] - 0 here means nothing it is just place to store value of muy
                        - pay attantion to order of elements. since val is sensitive to sign. SO element

    :param vars: list of elements e.g. vars = [QF, QD] or it can be initial twiss parameters:
                vars = [[tws0, 'beta_x'], [tws0, 'beta_y'], [tws0, 'alpha_x'], [tws0, 'alpha_y']].
                A tuple of quadrupoles can be passed as a variable to constrain their strengths
                to the same value, e.g., vars = [(QF, QD)].
                A dictionary with quadrupoles as keys and their relative strengths as values
                can be used for more flexible constraints, e.g., vars = [{QF: 1.0, QD: -1.0}],
                which constrains QD and QF to have equal strengths with opposite signs.
    :param tw: initial Twiss
    :param verbose: allow print output of minimization procedure
    :param max_iter:
    :param method: string, available 'simplex', 'cg', 'bfgs'
    :param weights: function returns weights, for example
                    def weights_default(val):
                        if val == 'periodic': return 10000001.0
                        if val == 'total_len': return 10000001.0
                        if val in ['Dx', 'Dy', 'Dxp', 'Dyp']: return 10000002.0
                        if val in ['alpha_x', 'alpha_y']: return 100007.0
                        if val in ['mux', 'muy']: return 10000006.0
                        if val in ['beta_x', 'beta_y']: return 100007.0
                        return 0.0001
    :param tol: tolerance default 1e-5
    :return: result
    """

    # tw = deepcopy(tw0)

    def errf(x):

        tw_loc = deepcopy(tw)
        tw0 = deepcopy(tw)


        # parameter to be varied is determined by variable class

        for i in range(len(vars)):
            if isinstance(vars[i], Drift):
                if x[i] < 0:
                    return weights('negative_length')

                vars[i].l = x[i]
            if isinstance(vars[i], Quadrupole):
                vars[i].k1 = x[i]
            if isinstance(vars[i], Solenoid):
                vars[i].k = x[i]
            if isinstance(vars[i], list):
                if isinstance(vars[i][0], Twiss) and isinstance(vars[i][1], str):
                    k = vars[i][1]
                    setattr(tw_loc, k, x[i])
            if isinstance(vars[i], tuple):
            # all quads strength in tuple varied simultaneously
                for v in vars[i]:
                    v.k1 = x[i]
            if isinstance(vars[i], dict):
            # all quads strength in dict keys varied simultaneously
            # with coupling parameters given as values.
                for q in vars[i].keys():
                    q.k1 = vars[i][q] * x[i]

        err = 0.0
        # save reference points where equality is asked

        ref_hsh = {}  # penalties on two-point inequalities
        for e in constr.keys():
            if e == 'total_len':
                continue
            for k in constr[e].keys():
                if isinstance(constr[e][k], list):
                    if constr[e][k][0] == '->':
                        # print 'creating reference to', constr[e][k][1].id
                        ref_hsh[constr[e][k][1]] = {k: 0.0}
        # evaluating global and point penalties

        tw_loc.s = 0

        for e in lat.sequence:
            for tm in e.first_order_tms:
                tw_loc = tm * tw_loc  # apply transfer map

                # --- Global constraints ---
                if 'global' in constr:
                    for c, rule in constr['global'].items():
                        if isinstance(rule, list):
                            op, v1 = rule[0], rule[1]
                            val = getattr(tw_loc, c)

                            if op == '<' and val > v1:
                                err += weights(c) * (val - v1) ** 2
                            elif op == '>' and val < v1:
                                err += weights(c) * (val - v1) ** 2

                # --- Delta constraint update ---
                if 'delta' in constr and e in constr['delta']:
                    tw_k = constr['delta'][e][0]
                    constr['delta'][e][1] = getattr(tw_loc, tw_k)

                # --- Update reference hash if needed ---
                if e in ref_hsh:
                    ref_hsh[e] = deepcopy(tw_loc)

                # --- Local constraints ---
                if e in constr:
                    for k, rule in constr[e].items():
                        val = getattr(tw_loc, k)

                        if isinstance(rule, list):
                            op = rule[0]
                            v1 = rule[1]

                            if op == '<' and val > v1:
                                err += weights(k) * (val - v1) ** 2
                            elif op == '>' and val < v1:
                                err += weights(k) * (val - v1) ** 2
                            elif op == 'a<' and abs(val) > v1:
                                err += weights(k) * (abs(val) - v1) ** 2
                            elif op == 'a>' and abs(val) < v1:
                                err += weights(k) * (abs(val) - v1) ** 2
                            elif op == '->':
                                try:
                                    dv1 = float(rule[2]) if len(rule) > 2 else 0.0
                                    ref_val = getattr(ref_hsh[v1], k)
                                    err += (val - (ref_val + dv1)) ** 2
                                    if val < v1:
                                        err += (val - v1) ** 2
                                except Exception as ex:
                                    print(f'Constraint error: rval should precede lval in lattice ({ex})')

                            if val < 0:
                                err += (val - v1) ** 2

                        elif isinstance(rule, str):
                            # handle symbolic constraints if any
                            pass
                        else:
                            # direct comparison
                            ref_val = rule
                            err += weights(k) * (ref_val - val) ** 2
        if "total_len" in constr.keys():
            total_len = constr["total_len"]
            err = err + weights('total_len') * (tw_loc.s - total_len) ** 2

        if 'delta' in constr.keys():
            delta_dict = constr['delta']
            elems = []
            for e in delta_dict.keys():
                if isinstance(e, OpticElement):
                    elems.append(e)
            delta_err = delta_dict["weight"] * (delta_dict[elems[1]][1] - delta_dict[elems[0]][1] - delta_dict["val"])**2
            err = err + delta_err


        if verbose:
            print('iteration error:', x, err)
        return err


    # list of arguments determined based on the variable class

    x = [0.0] * len(vars)
    for i in range(len(vars)):
        if isinstance(vars[i], list):
            if isinstance(vars[i][0], Twiss) and isinstance(vars[i][1], str):
                k = vars[i][1]
                x[i] = getattr(tw, k)
                # if k in ['beta_x', 'beta_y']:
                #     x[i] = 10
                # else:
                #     x[i] = 0.0
        if isinstance(vars[i], tuple):
            x[i] = vars[i][0].k1
        if isinstance(vars[i], dict):
            q = list(vars[i].keys())[0]
            x[i] = q.k1 / vars[i][q]
        if isinstance(vars[i], Quadrupole):
            x[i] = vars[i].k1
        if isinstance(vars[i], Drift):
            x[i] = vars[i].l

    if verbose:
        print("initial value: x = ", x)
    if method == 'simplex':
        res = fmin(errf, x, xtol=tol, maxiter=max_iter, maxfun=max_iter)
    if method == 'cg':
        res = fmin_cg(errf, x, gtol=tol, epsilon=1.e-5, maxiter=max_iter)
    if method == 'bfgs':
        res = fmin_bfgs(errf, x, gtol=tol, epsilon=1.e-5, maxiter=max_iter)


    # if initial twiss was varied set the twiss argument object to resulting value

    for i in range(len(vars)):
        if isinstance(vars[i], list):
            if isinstance(vars[i][0], Twiss) and isinstance(vars[i][1], str):
                k = vars[i][1]
                setattr(tw, k, res[i])

    # update MagneticLattice total length in case a Drift length was in list of variables

    return res
