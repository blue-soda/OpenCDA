# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 12:40:23 2020

@author: Anoop
"""

import numpy as np
import skfuzzy as fuzz


class FIS(object):

    def __init__(self, ni, nz, Ninmf, Noutmf, type_mf):
        self.ni = ni
        self.nz = nz
        self.Ninmf = Ninmf
        self.Noutmf = Noutmf
        self.type_mf = type_mf
        self.u_in = []
        self.u_out = []
        self.mf_in = []
        self.mf_out = []
        self.tri_in = []
        self.tri_out = []
        self.rules = []
        self.mins = []
        self.maxs = []
        for i in range(0, ni):
            self.u_in.append([])
            self.mf_in.append([])

        for i in range(0, nz):
            self.u_out.append([])
            self.mf_out.append([])

        for i in range(0, ni * Ninmf):
            self.tri_in.append([])

        for i in range(0, nz * Noutmf):
            self.tri_out.append([])

        self.rules = np.ones(9).astype('int')

    def eval_op(self, inputs):

        # Normalize using min and max
        mins = self.mins
        maxs = self.maxs
        inp = (2 / (maxs - mins)) * inputs + (1 - (2 * maxs / (maxs - mins)))

        # determining the degree of membership for the input values
        deg_mf = [[fuzz.interp_membership(self.u_in[i], self.tri_in[i][j], inp[i]) for j in range(self.Ninmf)] for i in
                  range(self.ni)]
        # pdb.set_trace()
        outs1 = []

        F = np.zeros((self.rules.shape[0], self.ni))
        for i in range(self.ni):
            for j in range(self.Ninmf):
                F[self.rules[:, i] == j, i] = deg_mf[i][j]
        prodF = np.prod(F, axis=1)

        unq_consq1 = set(self.rules[:, self.ni])

        for k1 in unq_consq1:
            RF = max(prodF[self.rules[:, self.ni] == k1])
            outs1.append(np.fmin(RF, self.tri_out[0][int(k1)]).tolist())

        arr1 = np.array(outs1)

        # Implication steps
        first1 = np.amax(arr1, axis=0)

        output1 = fuzz.defuzz(self.u_out[0], first1,
                              'centroid')

        return output1
