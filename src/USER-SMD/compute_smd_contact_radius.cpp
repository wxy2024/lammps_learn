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
/*定义一个计算，该计算输出接触半径，即用于防止粒子相互穿透的半径。
接触半径仅用于防止属于不同物理体的粒子相互穿透。
它被接触对样式（例如，smd/hertz和smd/tri_surface）使用。
对于不在指定计算组中的粒子，接触半径的值将为0.0。*/
//https://docs.lammps.org/compute_smd_contact_radius.html

//传入/更新接触半径
#include <string.h>
#include "compute_smd_contact_radius.h"
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
ComputeSMDContactRadius::ComputeSMDContactRadius(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg)
{
  if (narg != 3) error->all(FLERR,"Illegal compute smd/contact_radius command");
  if (atom->contact_radius_flag != 1) error->all(FLERR,"compute smd/contact_radius command requires atom_style with contact_radius (e.g. smd)");

  peratom_flag = 1;
  size_peratom_cols = 0;

  nmax = 0;
  contact_radius_vector = NULL;
}

/* ---------------------------------------------------------------------- */
//析构函数
ComputeSMDContactRadius::~ComputeSMDContactRadius()
{
  memory->sfree(contact_radius_vector);
}

/* ---------------------------------------------------------------------- */
//初始化
void ComputeSMDContactRadius::init()
{

  int count = 0;
  for (int i = 0; i < modify->ncompute; i++)
    if (strcmp(modify->compute[i]->style,"smd/contact_radius") == 0) count++;
  if (count > 1 && comm->me == 0)
    error->warning(FLERR,"More than one compute smd/contact_radius");
}

/* ---------------------------------------------------------------------- */

void ComputeSMDContactRadius::compute_peratom()//以原子为单位计算，而不是以整个系统
{
  invoked_peratom = update->ntimestep;
  //被调用的原子
  // grow rhoVector array if necessary
  //看有没有需要扩充空间
  if (atom->nmax > nmax) {
    memory->sfree(contact_radius_vector);
    nmax = atom->nmax;
    contact_radius_vector = (double *) memory->smalloc(nmax*sizeof(double),"atom:contact_radius_vector");
    vector_atom = contact_radius_vector;
  }

  double *contact_radius = atom->contact_radius;//接触半径，原子之间作用力的范围
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
              contact_radius_vector[i] = contact_radius[i];
      }
      else {
              contact_radius_vector[i] = 0.0;
      }
    }
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based array
------------------------------------------------------------------------- */
//计算内存使用量
double ComputeSMDContactRadius::memory_usage()
{
  double bytes = nmax * sizeof(double);
  return bytes;
}
