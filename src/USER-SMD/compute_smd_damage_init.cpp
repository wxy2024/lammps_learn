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

/* ----------------------------------------------------------------------
   Contributing author: A. de Vaucorbeil, alban.devaucorbeil@monash.edu
                        Copyright (C) 2018
------------------------------------------------------------------------- */
//从原子中传入初始损伤值
#include <string.h>
#include "compute_smd_damage_init.h"
#include "atom.h"
#include "update.h"
#include "modify.h"
#include "comm.h"
#include "force.h"
#include "memory.h"
#include "error.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */
//构造函数
ComputeSMDDamageInit::ComputeSMDDamageInit(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg)
{
  if (narg != 3) error->all(FLERR,"Illegal compute smd/damage/init command");
  if (atom->damage_init_flag != 1) error->all(FLERR,"compute smd/damage/init command requires atom_style with damage_init (e.g. smd)");

  peratom_flag = 1;
  size_peratom_cols = 0;

  nmax = 0;
  damage_init_vector = NULL;
}

/* ---------------------------------------------------------------------- */

ComputeSMDDamageInit::~ComputeSMDDamageInit()
{
  memory->sfree(damage_init_vector);
}

/* ---------------------------------------------------------------------- */

void ComputeSMDDamageInit::init()
{

  int count = 0;
  for (int i = 0; i < modify->ncompute; i++)
    if (strcmp(modify->compute[i]->style,"smd/damage_init") == 0) count++;
  if (count > 1 && comm->me == 0)
    error->warning(FLERR,"More than one compute smd/damage/init");
}

/* ---------------------------------------------------------------------- */

void ComputeSMDDamageInit::compute_peratom()
{
  invoked_peratom = update->ntimestep;

  // grow rhoVector array if necessary

  if (atom->nmax > nmax) {
    memory->sfree(damage_init_vector);
    nmax = atom->nmax;
    damage_init_vector = (double *) memory->smalloc(nmax*sizeof(double),"atom:damage_init_vector");
    vector_atom = damage_init_vector;
  }

  double *damage_init = atom->damage_init;//原子中的
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
              damage_init_vector[i] = damage_init[i];//初始损伤值
      }
      else {
              damage_init_vector[i] = 0.0;
      }
    }
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based array
------------------------------------------------------------------------- */

double ComputeSMDDamageInit::memory_usage()
{
  double bytes = nmax * sizeof(double);
  return bytes;
}
