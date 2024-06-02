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
/*定义一个计算，该计算输出与粒子i和j的实际相对分离相比，近似相对分离的误差。
理想情况下，如果变形梯度是精确的，并且在中心节点附近所有粒子的位置与变形梯度之间存在唯一映射，
那么在变形配置中，粒子i和j的近似相对分离将与实际相对分离完全一致。
这个计算实际上只对调试Total - Lagrangian SPH粒子样式中的小时玻璃控制机制有用。*/
//https://docs.lammps.org/compute_smd_hourglass_error.html
// 从原子中传入hourglass_error

#include <string.h>
#include "compute_smd_hourglass_error.h"
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
// 构造函数
ComputeSMDHourglassError::ComputeSMDHourglassError(LAMMPS *lmp, int narg, char **arg) : Compute(lmp, narg, arg)
{
        if (narg != 3)
                error->all(FLERR, "Illegal compute smd/hourglass_error command");
        if (atom->smd_flag != 1)
                error->all(FLERR, "compute smd/hourglass_error command requires atom_style with hourglass_error (e.g. smd)");

        peratom_flag = 1;
        size_peratom_cols = 0;

        nmax = 0;
        hourglass_error_vector = NULL;
}

/* ---------------------------------------------------------------------- */

ComputeSMDHourglassError::~ComputeSMDHourglassError()
{
        memory->sfree(hourglass_error_vector);
}

/* ---------------------------------------------------------------------- */

void ComputeSMDHourglassError::init()
{

        int count = 0;
        for (int i = 0; i < modify->ncompute; i++)
                if (strcmp(modify->compute[i]->style, "smd/hourglass_error") == 0)
                        count++;
        if (count > 1 && comm->me == 0)
                error->warning(FLERR, "More than one compute smd/hourglass_error");
}

/* ---------------------------------------------------------------------- */

void ComputeSMDHourglassError::compute_peratom()
{
        invoked_peratom = update->ntimestep;

        // grow output Vector array if necessary

        if (atom->nmax > nmax)
        {
                memory->sfree(hourglass_error_vector);
                nmax = atom->nmax;
                hourglass_error_vector = (double *)memory->smalloc(nmax * sizeof(double), "atom:hourglass_error_vector");
                vector_atom = hourglass_error_vector;
        }

        int itmp = 0;
        double *hourglass_error = (double *)force->pair->extract("smd/tlsph/hourglass_error_ptr", itmp);
        if (hourglass_error == NULL)
        {
                error->all(FLERR, "compute smd/hourglass_error failed to access hourglass_error array");
        }

        int *mask = atom->mask;
        int nlocal = atom->nlocal;

        for (int i = 0; i < nlocal; i++)
        {
                if (mask[i] & groupbit)
                {
                        hourglass_error_vector[i] = hourglass_error[i]; // （11）沙漏控制误差
                }
                else
                {
                        hourglass_error_vector[i] = 0.0;
                }
        }
}

/* ----------------------------------------------------------------------
 memory usage of local atom-based array
 ------------------------------------------------------------------------- */

double ComputeSMDHourglassError::memory_usage()
{
        double bytes = nmax * sizeof(double);
        return bytes;
}
