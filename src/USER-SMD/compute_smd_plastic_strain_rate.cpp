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
//定义一个计算，该计算输出等效塑性应变的时间率。此命令仅在定义了具有塑性的材料模型时才有意义。
//https://docs.lammps.org/compute_smd_plastic_strain_rate.html
//从原子中传入塑性应变率
#include <string.h>
#include "compute_smd_plastic_strain_rate.h"
#include "atom.h"
#include "update.h"
#include "modify.h"
#include "comm.h"
#include "force.h"
#include "memory.h"
#include "error.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ComputeSMDPlasticStrainRate::ComputeSMDPlasticStrainRate(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg)
{
  if (narg != 3) error->all(FLERR,"Illegal compute smd/plastic_strain command");
  if (atom->eff_plastic_strain_rate_flag != 1) error->all(FLERR,"compute smd/plastic_strain_rate command requires atom_style with plastic_strain_rate (e.g. smd)");

  peratom_flag = 1;
  size_peratom_cols = 0;

  nmax = 0;
  plastic_strain_rate_vector = NULL;
}

/* ---------------------------------------------------------------------- */

ComputeSMDPlasticStrainRate::~ComputeSMDPlasticStrainRate()
{
  memory->sfree(plastic_strain_rate_vector);
}

/* ---------------------------------------------------------------------- */

void ComputeSMDPlasticStrainRate::init()
{

  int count = 0;
  for (int i = 0; i < modify->ncompute; i++)
    if (strcmp(modify->compute[i]->style,"smd/plastic_strain_rate") == 0) count++;
  if (count > 1 && comm->me == 0)
    error->warning(FLERR,"More than one compute smd/plastic_strain_rate");
}

/* ---------------------------------------------------------------------- */

void ComputeSMDPlasticStrainRate::compute_peratom()
{
  invoked_peratom = update->ntimestep;

  // grow rhoVector array if necessary

  if (atom->nmax > nmax) {
    memory->sfree(plastic_strain_rate_vector);
    nmax = atom->nmax;
    plastic_strain_rate_vector = (double *) memory->smalloc(nmax*sizeof(double),"atom:plastic_strain_rate_vector");
    vector_atom = plastic_strain_rate_vector;
  }

  double *plastic_strain_rate = atom->eff_plastic_strain_rate;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
              plastic_strain_rate_vector[i] = plastic_strain_rate[i];//塑性应变率
      }
      else {
              plastic_strain_rate_vector[i] = 0.0;
      }
    }
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based array
------------------------------------------------------------------------- */

double ComputeSMDPlasticStrainRate::memory_usage()
{
  double bytes = nmax * sizeof(double);
  return bytes;
}
