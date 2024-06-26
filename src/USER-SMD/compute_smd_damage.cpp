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
//https://docs.lammps.org/compute_smd_damage.html
/*定义一个计算，根据通过SMD SPH对样式定义的损伤模型，例如最大塑性应变失效准则，来计算SPH粒子的损伤状态。*/
//从原子中传入损伤数组
#include <string.h>
#include "compute_smd_damage.h"
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
ComputeSMDDamage::ComputeSMDDamage(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg)
{
  if (narg != 3) error->all(FLERR,"Illegal compute smd/damage command");
  if (atom->damage_flag != 1) error->all(FLERR,"compute smd/damage command requires atom_style with damage (e.g. smd)");
  //检查原子是否损伤，damage_flag==1说明损伤，否则没有损伤
  peratom_flag = 1;  //对每个原子进行计算
  size_peratom_cols = 0;  //每个原子的额外列数为0

  nmax = 0;  //
  damage_vector = NULL;  //初始化存储损伤值的数组
}

/* ---------------------------------------------------------------------- */
//析构函数
ComputeSMDDamage::~ComputeSMDDamage()
{
  memory->sfree(damage_vector);
}

/* ---------------------------------------------------------------------- */
//初始化
void ComputeSMDDamage::init()
{

  int count = 0;
  for (int i = 0; i < modify->ncompute; i++)//限制计算次数
    if (strcmp(modify->compute[i]->style,"smd/damage") == 0) count++;//检查每个计算类型是否为“smd/damage”是则加一
  if (count > 1 && comm->me == 0)
    error->warning(FLERR,"More than one compute smd/damage");//提示有多个“smd/damage”计算器存在
}

/* ---------------------------------------------------------------------- */

void ComputeSMDDamage::compute_peratom()
{
  invoked_peratom = update->ntimestep; //更新当前时间步数

  // grow rhoVector array if necessary 如有必要，增加rhoVector数组

  if (atom->nmax > nmax) {//如果当前原子数大于之前设置的原子数，就增大空间
    memory->sfree(damage_vector);
    nmax = atom->nmax;
    damage_vector = (double *) memory->smalloc(nmax*sizeof(double),"atom:damage_vector");
    vector_atom = damage_vector;
  }

  double *damage = atom->damage;//获取原子损伤数组
  int *mask = atom->mask;//获取掩码数组
  int nlocal = atom->nlocal;//获取本地原子数

    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {//把符合条件的原子的损伤值存储到damage_vector里，不符合的原子的损伤都置为0
              damage_vector[i] = damage[i];
      }
      else {
              damage_vector[i] = 0.0;
      }
    }
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based array
------------------------------------------------------------------------- */

double ComputeSMDDamage::memory_usage()
{
  double bytes = nmax * sizeof(double);
  return bytes;
}
