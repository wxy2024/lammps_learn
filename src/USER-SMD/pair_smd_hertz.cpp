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

/* ----------------------------------------------------------------------
   Contributing author: Mike Parks (SNL)
------------------------------------------------------------------------- */

#include <cmath>
#include <cfloat>
#include <cstdlib>
#include <cstring>
#include "pair_smd_hertz.h"
#include "atom.h"
#include "domain.h"
#include "force.h"
#include "update.h"
#include "modify.h"
#include "fix.h"
#include "comm.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "memory.h"
#include "error.h"

using namespace LAMMPS_NS;

#define SQRT2 1.414213562e0
/*pair_smd_hertz是一种用于描述基于Hertz接触力模型的分子间力的势函数，通常用于描述颗粒之间的力学相互作用。
Hertz接触力模型基于弹性理论，用于描述在颗粒之间的碰撞和接触时的力学行为。*/
/* ---------------------------------------------------------------------- */
//构造函数
PairHertz::PairHertz(LAMMPS *lmp) :
                Pair(lmp) {

        onerad_dynamic = onerad_frozen = maxrad_dynamic = maxrad_frozen = NULL;
        bulkmodulus = NULL;
        kn = NULL;
        scale = 1.0;
}

/* ---------------------------------------------------------------------- */
//析构函数
PairHertz::~PairHertz() {

        if (allocated) {
                memory->destroy(setflag);
                memory->destroy(cutsq);
                memory->destroy(bulkmodulus);
                memory->destroy(kn);

                delete[] onerad_dynamic;
                delete[] onerad_frozen;
                delete[] maxrad_dynamic;
                delete[] maxrad_frozen;
        }
}

/* ---------------------------------------------------------------------- */

void PairHertz::compute(int eflag, int vflag) {
        int i, j, ii, jj, inum, jnum, itype, jtype;
        double xtmp, ytmp, ztmp, delx, dely, delz;
        double rsq, r, evdwl, fpair;
        int *ilist, *jlist, *numneigh, **firstneigh;
        double rcut, r_geom, delta, ri, rj, dt_crit;
        double *rmass = atom->rmass;

        evdwl = 0.0;
        if (eflag || vflag)
                ev_setup(eflag, vflag);
        else
                evflag = vflag_fdotr = 0;

        double **f = atom->f;
        double **x = atom->x;
        double **x0 = atom->x0;
        double *damage = atom->damage;
        int *type = atom->type;
        int nlocal = atom->nlocal;
        double *radius = atom->contact_radius;
        double *sph_radius = atom->radius;
        double rcutSq;
        double delx0, dely0, delz0, rSq0, sphCut;

        int newton_pair = force->newton_pair;
        int periodic = (domain->xperiodic || domain->yperiodic || domain->zperiodic);
           //周期性边界条件

        inum = list->inum;
        ilist = list->ilist;
        numneigh = list->numneigh;
        firstneigh = list->firstneigh;

        stable_time_increment = 1.0e22;//时间步长

        // loop over neighbors of my atoms
        for (ii = 0; ii < inum; ii++) {
                i = ilist[ii];
                xtmp = x[i][0];
                ytmp = x[i][1];
                ztmp = x[i][2];
                itype = type[i];//原子的类型用一个int型变量表示，类型代表in文件中原子类型号吗？
                ri = scale * radius[i]; //这是啥？scale是啥？
                jlist = firstneigh[i];
                jnum = numneigh[i];

                for (jj = 0; jj < jnum; jj++) {
                        j = jlist[jj];
                        j &= NEIGHMASK;

                        jtype = type[j];//原子的类型用一个int型变量表示，类型代表什么？

                        delx = xtmp - x[j][0];//粒子对之间的距离
                        dely = ytmp - x[j][1];
                        delz = ztmp - x[j][2];

                        rsq = delx * delx + dely * dely + delz * delz;//距离的平方

                        rj = scale * radius[j];//这也有一个scale,是什么东西？
                        rcut = ri + rj;//两个粒子的半径的和
                        rcutSq = rcut * rcut;//半径和的平方

                        if (rsq < rcutSq) {//粒子对之间在距离上互相接触

                                /*
                                 * self contact option:
                                 * if pair of particles was initially close enough to interact via a bulk continuum mechanism (e.g. SPH), exclude pair from contact forces.
                                 * 如果粒子对最初足够接近，可以通过体连续机制（如 SPH）相互作用，则将粒子对排除在接触力之外。
                                 * this approach should work well if no updates of the reference configuration are performed.
                                 * 如果不对参考配置进行更新，这种方法应该会很有效。
                                 */

                                if (itype == jtype) {//如果属于同一类型的原子   所以说，如果是同一种类型的原子互相挤压，可能会融合是这意思不？
                                        delx0 = x0[j][0] - x0[i][0];
                                        dely0 = x0[j][1] - x0[i][1];
                                        delz0 = x0[j][2] - x0[i][2];//计算初始坐标下的原子对之间的距离
                                        if (periodic) {// 如果启用周期性边界条件
                                                domain->minimum_image(delx0, dely0, delz0);// 调用函数进行最小图像处理 ？？
                                        }
                                        rSq0 = delx0 * delx0 + dely0 * dely0 + delz0 * delz0; // initial distance 初始坐标原子对之间距离的平方
                                        sphCut = sph_radius[i] + sph_radius[j]; // 计算两个粒子（或原子）的球形截断半径之和
                                        if (rSq0 < sphCut * sphCut) { // 如果初始距离的平方小于球形截断半径的平方
                                                rcut = 0.5 * rcut;//将截断半径 rcut 减半 ？？？啥意思，就是让两个粒子各自缩小一半？还是两个粒子融合成一个？
                                                rcutSq = rcut * rcut;// 计算新的截断半径的平方
                                                if (rsq > rcutSq) {// 如果实际距离的平方大于新的截断半径的平方
                                                        continue;// 继续下一个循环（跳过后续的代码，进行下一次迭代）
                                                }
                                        }
                                }

                                r = sqrt(rsq);//r是原子对之间的距离
                                //printf("hertz interaction, r=%f, cut=%f, h=%f\n", r, rcut, sqrt(rSq0));
                                //赫兹相互作用
                                // Hertzian short-range forces
                                delta = rcut - r; // overlap distance重叠距离  =两个粒子的半径的和-原子对之间的距离
                                r_geom = ri * rj / rcut;//半径的积除以半径的和，这是啥
                                //assuming poisson ratio = 1/4 for 3d
                                //fpair表示粒子间的力，根据公式计算得到。
                                //其中，bulkmodulus[itype][jtype]是材料的体积模量，
                                //damage[i]和damage[j]表示两个粒子的损伤参数，
                                //delta是粒子间的重叠距离，r_geom是几何平均半径。
                                fpair = 1.066666667e0 * bulkmodulus[itype][jtype] * (1-damage[i]) * (1-damage[j]) * delta * sqrt(delta * r_geom); //  units: N
                                evdwl = fpair * 0.4e0 * delta; // GCG 25 April: this expression conserves total energy  GCG 4 月 25 日：这一表达式保存了总能量
                                dt_crit = 3.14 * sqrt(0.5 * (rmass[i] + rmass[j]) / (fpair / delta));
                                //dt_crit表示稳定时间步长，根据一定的公式计算得到。
                                //这个时间步长用于稳定性分析，确保模拟过程中的时间步不至于过大导致数值不稳定。
                                stable_time_increment = MIN(stable_time_increment, dt_crit);
                                if (r > 2.0e-16) {
                                        fpair /= r; // divide by r and multiply with non-normalized distance vector
                                                    //除以 r，再乘以非标准化距离向量
                                } else {
                                        fpair = 0.0;
                                }

                                /*
                                 * contact viscosity -- needs to be done, see GRANULAR package for normal & shear damping
                                 * for now: no damping and thus no viscous energy deltaE
                                 * 接触粘性--需要完成，参见 GRANULAR 软件包中的法向和剪切阻尼  
                                 * 目前：无阻尼，因此无粘性能量 deltaE 
                                 */

                                if (evflag) {
                                        ev_tally(i, j, nlocal, newton_pair, evdwl, 0.0, fpair, delx, dely, delz);
                                }

                                f[i][0] += delx * fpair;
                                f[i][1] += dely * fpair;
                                f[i][2] += delz * fpair;//节点力积分？

                                if (newton_pair || j < nlocal) {
                                        f[j][0] -= delx * fpair;
                                        f[j][1] -= dely * fpair;
                                        f[j][2] -= delz * fpair;
                                }

                        }
                }
        }

//      double stable_time_increment_all = 0.0;
//      MPI_Allreduce(&stable_time_increment, &stable_time_increment_all, 1, MPI_DOUBLE, MPI_MIN, world);
//      if (comm->me == 0) {
//              printf("stable time step for pair smd/hertz is %f\n", stable_time_increment_all);
//      }
}

/* ----------------------------------------------------------------------
 allocate all arrays 分配所有数组
 ------------------------------------------------------------------------- */

void PairHertz::allocate() {
        allocated = 1;
        int n = atom->ntypes;

        memory->create(setflag, n + 1, n + 1, "pair:setflag");
        for (int i = 1; i <= n; i++)
                for (int j = i; j <= n; j++)
                        setflag[i][j] = 0;

        memory->create(bulkmodulus, n + 1, n + 1, "pair:kspring");
        memory->create(kn, n + 1, n + 1, "pair:kn");

        memory->create(cutsq, n + 1, n + 1, "pair:cutsq"); // always needs to be allocated, even with granular neighborlist

        onerad_dynamic = new double[n + 1];
        onerad_frozen = new double[n + 1];
        maxrad_dynamic = new double[n + 1];
        maxrad_frozen = new double[n + 1];
}

/* ----------------------------------------------------------------------
 global settings
 ------------------------------------------------------------------------- */

void PairHertz::settings(int narg, char **arg) {
        if (narg != 1)
                error->all(FLERR, "Illegal number of args for pair_style hertz");

        scale = force->numeric(FLERR, arg[0]);
        if (comm->me == 0) {
                printf("\n>>========>>========>>========>>========>>========>>========>>========>>========\n");
                printf("SMD/HERTZ CONTACT SETTINGS:\n");
                printf("... effective contact radius is scaled by %f\n", scale);
                printf(">>========>>========>>========>>========>>========>>========>>========>>========\n");
        }

}

/* ----------------------------------------------------------------------
 set coeffs for one or more type pairs 为一个或多个类型对设置系数
 ------------------------------------------------------------------------- */

void PairHertz::coeff(int narg, char **arg) {
        if (narg != 3)
                error->all(FLERR, "Incorrect args for pair coefficients");
        if (!allocated)
                allocate();

        int ilo, ihi, jlo, jhi;
        force->bounds(FLERR,arg[0], atom->ntypes, ilo, ihi);
        force->bounds(FLERR,arg[1], atom->ntypes, jlo, jhi);

        double bulkmodulus_one = atof(arg[2]);

        // set short-range force constant
        double kn_one = 0.0;
        if (domain->dimension == 3) {
                kn_one = (16. / 15.) * bulkmodulus_one; //assuming poisson ratio = 1/4 for 3d
        } else {
                kn_one = 0.251856195 * (2. / 3.) * bulkmodulus_one; //assuming poisson ratio = 1/3 for 2d
        }

        int count = 0;
        for (int i = ilo; i <= ihi; i++) {
                for (int j = MAX(jlo, i); j <= jhi; j++) {
                        bulkmodulus[i][j] = bulkmodulus_one;
                        kn[i][j] = kn_one;
                        setflag[i][j] = 1;
                        count++;
                }
        }

        if (count == 0)
                error->all(FLERR, "Incorrect args for pair coefficients");
}

/* ----------------------------------------------------------------------
 init for one type pair i,j and corresponding j,i
 ------------------------------------------------------------------------- */

double PairHertz::init_one(int i, int j) {

        if (!allocated)
                allocate();

        if (setflag[i][j] == 0)
                error->all(FLERR, "All pair coeffs are not set");

        bulkmodulus[j][i] = bulkmodulus[i][j];
        kn[j][i] = kn[i][j];

        // cutoff = sum of max I,J radii for
        // dynamic/dynamic & dynamic/frozen interactions, but not frozen/frozen

        double cutoff = maxrad_dynamic[i] + maxrad_dynamic[j];
        cutoff = MAX(cutoff, maxrad_frozen[i] + maxrad_dynamic[j]);
        cutoff = MAX(cutoff, maxrad_dynamic[i] + maxrad_frozen[j]);

        if (comm->me == 0) {
                printf("cutoff for pair smd/hertz = %f\n", cutoff);
        }
        return cutoff;
}

/* ----------------------------------------------------------------------
 init specific to this pair style
 ------------------------------------------------------------------------- */

void PairHertz::init_style() {
        int i;

        // error checks

        if (!atom->contact_radius_flag)
                error->all(FLERR, "Pair style smd/hertz requires atom style with contact_radius");

        int irequest = neighbor->request(this);
        neighbor->requests[irequest]->size = 1;

        // set maxrad_dynamic and maxrad_frozen for each type
        // include future Fix pour particles as dynamic

        for (i = 1; i <= atom->ntypes; i++)
                onerad_dynamic[i] = onerad_frozen[i] = 0.0;

        double *radius = atom->radius;
        int *type = atom->type;
        int nlocal = atom->nlocal;

        for (i = 0; i < nlocal; i++) {
                onerad_dynamic[type[i]] = MAX(onerad_dynamic[type[i]], radius[i]);
        }

        MPI_Allreduce(&onerad_dynamic[1], &maxrad_dynamic[1], atom->ntypes, MPI_DOUBLE, MPI_MAX, world);
        MPI_Allreduce(&onerad_frozen[1], &maxrad_frozen[1], atom->ntypes, MPI_DOUBLE, MPI_MAX, world);
}

/* ----------------------------------------------------------------------
 neighbor callback to inform pair style of neighbor list to use
 optional granular history list
 ------------------------------------------------------------------------- */

void PairHertz::init_list(int id, NeighList *ptr) {
        if (id == 0)
                list = ptr;
}

/* ----------------------------------------------------------------------
 memory usage of local atom-based arrays
 ------------------------------------------------------------------------- */

double PairHertz::memory_usage() {

        return 0.0;
}

void *PairHertz::extract(const char *str, int &/*i*/) {
        //printf("in PairTriSurf::extract\n");
        if (strcmp(str, "smd/hertz/stable_time_increment_ptr") == 0) {
                return (void *) &stable_time_increment;
        }

        return NULL;

}
