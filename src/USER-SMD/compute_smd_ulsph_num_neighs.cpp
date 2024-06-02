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
/*定义一个计算，返回通过更新的Lagrangian SPH对样式进行相互作用的粒子在平滑核半径内的邻居粒子数量。在SPH模拟中，粒子的相互作用是通过对其周围的粒子施加平滑核来模拟的。平滑核的半径决定了粒子与其邻居粒子之间的相互作用范围。因此，通过计算每个粒子在其平滑核范围内的邻居粒子数量，可以了解系统中的粒子分布情况和密度变化。*/
//传入邻居粒子的数量？
#include <string.h>
#include "compute_smd_ulsph_num_neighs.h"
#include "atom.h"
#include "update.h"
#include "modify.h"
#include "comm.h"
#include "force.h"
#include "memory.h"
#include "error.h"
#include "pair.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ComputeSMDULSPHNumNeighs::ComputeSMDULSPHNumNeighs(LAMMPS *lmp, int narg, char **arg) :
        Compute(lmp, narg, arg) {
    if (narg != 3)
        error->all(FLERR, "Illegal compute smd/ulsph_num_neighs command");

    peratom_flag = 1;
    size_peratom_cols = 0;

    nmax = 0;
    numNeighsOutput = NULL;
}

/* ---------------------------------------------------------------------- */

ComputeSMDULSPHNumNeighs::~ComputeSMDULSPHNumNeighs() {
    memory->destroy(numNeighsOutput);
}

/* ---------------------------------------------------------------------- */

void ComputeSMDULSPHNumNeighs::init() {
    int count = 0;
    for (int i = 0; i < modify->ncompute; i++)
        if (strcmp(modify->compute[i]->style, "smd/ulsph_num_neighs") == 0)
            count++;
    if (count > 1 && comm->me == 0)
        error->warning(FLERR, "More than one compute smd/ulsph_num_neighs");
}

/* ---------------------------------------------------------------------- */

void ComputeSMDULSPHNumNeighs::compute_peratom() {
    invoked_peratom = update->ntimestep;

    if (atom->nmax > nmax) {
        memory->destroy(numNeighsOutput);
        nmax = atom->nmax;
        memory->create(numNeighsOutput, nmax, "ulsph/num_neighs:numNeighsRefConfigOutput");
        vector_atom = numNeighsOutput;
    }

    int *mask = atom->mask;
    int nlocal = atom->nlocal;

    int itmp = 0;
    int *numNeighs = (int *) force->pair->extract("smd/ulsph/numNeighs_ptr", itmp);
    if (numNeighs == NULL) {
        error->all(FLERR, "compute smd/ulsph_num_neighs failed to access numNeighs array");
    }

    for (int i = 0; i < nlocal; i++) {
        if (mask[i] & groupbit) {
            numNeighsOutput[i] = numNeighs[i];
        } else {
            numNeighsOutput[i] = 0.0;
        }
    }
}

/* ----------------------------------------------------------------------
 memory usage of local atom-based array
 ------------------------------------------------------------------------- */

double ComputeSMDULSPHNumNeighs::memory_usage() {
    double bytes = nmax * sizeof(double);
    return bytes;
}
