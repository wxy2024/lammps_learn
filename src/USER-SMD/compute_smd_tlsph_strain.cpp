/* ----------------------------------------------------------------------
 *
 *                    *** Smooth Mach Dynamics ***
 *
 * This file is part of the USER-SMD package for LAMMPS.
 * Copyright (2014) Georg C. Ganzenmueller, georg.ganzenmueller@emi.fhg.de
 * Fraunhofer Ernst-Mach Institute for High-Speed Dynamics, EMI,
 * Eckerstrasse 4, D-79104 Freiburg i.Br, Germany.
 *
 * ----------------------------------------------------------------------- */


/* ----------------------------------------------------------------------
 LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
 http://lammps.sandia.gov, Sandia National Laboratories
 Steve Plimpton, sjplimp@sandia.gov

 Copyright (2003) Sandia Corporation.  Under the terms of Contract
 DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
 certain rights in this software.  This software is distributed under
 the GNU General Public License.

 See the README file in the top-level LAMMPS directory.
 ------------------------------------------------------------------------- */
//更新应变
#include <string.h>
#include "compute_smd_tlsph_strain.h"
#include "atom.h"
#include "update.h"
#include "modify.h"
#include "comm.h"
#include "force.h"
#include "memory.h"
#include "error.h"
#include "pair.h"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <Eigen/Eigen>
using namespace Eigen;
using namespace std;
using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ComputeSMDTLSPHstrain::ComputeSMDTLSPHstrain(LAMMPS *lmp, int narg, char **arg) :
                Compute(lmp, narg, arg) {
        if (narg != 3)
                error->all(FLERR, "Illegal compute smd/tlsph_strain command");

        peratom_flag = 1;
        size_peratom_cols = 6;

        nmax = 0;
        strainVector = NULL;
}

/* ---------------------------------------------------------------------- */

ComputeSMDTLSPHstrain::~ComputeSMDTLSPHstrain() {
        memory->sfree(strainVector);
}

/* ---------------------------------------------------------------------- */

void ComputeSMDTLSPHstrain::init() {

        int count = 0;
        for (int i = 0; i < modify->ncompute; i++)
                if (strcmp(modify->compute[i]->style, "smd/tlsph_strain") == 0)
                        count++;
        if (count > 1 && comm->me == 0)
                error->warning(FLERR, "More than one compute smd/tlsph_strain");
}

/* ---------------------------------------------------------------------- */

void ComputeSMDTLSPHstrain::compute_peratom() {
        double **defgrad0 = atom->smd_data_9;

        invoked_peratom = update->ntimestep;

        // grow vector array if necessary

        if (atom->nmax > nmax) {
                memory->destroy(strainVector);
                nmax = atom->nmax;
                memory->create(strainVector, nmax, size_peratom_cols, "strainVector");
                array_atom = strainVector;
        }

        // copy data to output array
        int itmp = 0;
        //变形梯度的增量
        Matrix3d *Fincr = (Matrix3d *) force->pair->extract("smd/tlsph/Fincr_ptr", itmp);
        if (Fincr == NULL) {
                error->all(FLERR, "compute smd/tlsph_strain failed to access Fincr array");
        }

        int *mask = atom->mask;
        int nlocal = atom->nlocal;
        Matrix3d E, eye, Ftotal, F0;
        eye.setIdentity();

        for (int i = 0; i < nlocal; i++) {
                if (mask[i] & groupbit) {

                        // old deformation gradient 旧的形变梯度
                        F0(0, 0) = defgrad0[i][0];
                        F0(0, 1) = defgrad0[i][1];
                        F0(0, 2) = defgrad0[i][2];
                        F0(1, 0) = defgrad0[i][3];
                        F0(1, 1) = defgrad0[i][4];
                        F0(1, 2) = defgrad0[i][5];
                        F0(2, 0) = defgrad0[i][6];
                        F0(2, 1) = defgrad0[i][7];
                        F0(2, 2) = defgrad0[i][8];
                        //计算当前总的变形梯度
                        // compute current total deformation gradient
                        Ftotal = F0 * Fincr[i]; // this is the total deformation gradient: reference deformation times incremental deformation
                        
                        //计算应变E，把更新之后的应变存储到strainVector里，还是由二维变成一维
                        E = 0.5 * (Ftotal.transpose() * Ftotal - eye); // Green-Lagrange strain
                        strainVector[i][0] = E(0, 0);//全拉格朗日形变张量
                        strainVector[i][1] = E(1, 1);
                        strainVector[i][2] = E(2, 2);
                        strainVector[i][3] = E(0, 1);
                        strainVector[i][4] = E(0, 2);
                        strainVector[i][5] = E(1, 2);
                } else {
                        for (int j = 0; j < size_peratom_cols; j++) {
                                strainVector[i][j] = 0.0;
                        }
                }
        }
}

/* ----------------------------------------------------------------------
 memory usage of local atom-based array
 ------------------------------------------------------------------------- */

double ComputeSMDTLSPHstrain::memory_usage() {
        double bytes = size_peratom_cols * nmax * sizeof(double);
        return bytes;
}
