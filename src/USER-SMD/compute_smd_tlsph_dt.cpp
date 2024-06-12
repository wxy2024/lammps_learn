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
/*定义一个计算，用于输出每个粒子的CFL稳定时间增量。这个时间增量基本上由声速除以SPH平滑长度得到。
因为声速和平滑长度通常在模拟过程中变化，所以稳定时间增量需要在每个时间步重新计算。
这个计算在相关的SPH对样式中会自动执行，而这个计算只是为了使稳定时间增量可用于输出目的。*/
//https://docs.lammps.org/compute_smd_tlsph_dt.html
//从原子中传入dt，时间步长
#include <string.h>
#include "compute_smd_tlsph_dt.h"
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

ComputeSMDTlsphDt::ComputeSMDTlsphDt(LAMMPS *lmp, int narg, char **arg) :
                Compute(lmp, narg, arg) {
        if (narg != 3)
                error->all(FLERR, "Illegal compute smd/tlsph_dt command");
        if (atom->contact_radius_flag != 1)
                error->all(FLERR,
                                "compute smd/tlsph_dt command requires atom_style with contact_radius (e.g. smd)");

        peratom_flag = 1;
        size_peratom_cols = 0;

        nmax = 0;
        dt_vector = NULL;
}

/* ---------------------------------------------------------------------- */

ComputeSMDTlsphDt::~ComputeSMDTlsphDt() {
        memory->sfree(dt_vector);
}

/* ---------------------------------------------------------------------- */

void ComputeSMDTlsphDt::init() {
    int count = 0;  // 初始化计数器，用于统计 "smd/tlsph_dt" 类型的计算实例数量

    // 遍历所有定义的 compute 实例
    for (int i = 0; i < modify->ncompute; i++)
        // 比较每个 compute 实例的 style 字符串，如果是 "smd/tlsph_dt" 则计数增加
        if (strcmp(modify->compute[i]->style, "smd/tlsph_dt") == 0)
            count++;
    
    // 如果计数大于1并且当前进程的 rank 为 0（通常表示主进程）
    if (count > 1 && comm->me == 0)
        // 发出警告，表示存在多个 "smd/tlsph_dt" 类型的 compute 实例
        error->warning(FLERR, "More than one compute smd/tlsph_dt");
}

/* ---------------------------------------------------------------------- */
//计算每个原子的时间步长（dt_vector），并将这些值存储在 dt_vector 数组中
void ComputeSMDTlsphDt::compute_peratom() {
    invoked_peratom = update->ntimestep;  // 记录当前时间步长

    // 如果需要，扩展 dt_vector 数组的大小
    if (atom->nmax > nmax) {
        memory->sfree(dt_vector);  // 释放旧的内存空间
        nmax = atom->nmax;
        dt_vector = (double *) memory->smalloc(nmax * sizeof(double), "atom:tlsph_dt_vector");  // 分配新的内存空间
        vector_atom = dt_vector;
    }

    int itmp = 0;
    double *particle_dt = (double *) force->pair->extract("smd/tlsph/particle_dt_ptr", itmp);  // 从外部提取 particle_dt 数组的指针
    if (particle_dt == NULL) {
        error->all(FLERR, "compute smd/tlsph_dt failed to access particle_dt array");  // 如果获取失败，则报错
    }

    int *mask = atom->mask;  // 获取原子的掩码数组
    int nlocal = atom->nlocal;  // 获取本地原子数量

    for (int i = 0; i < nlocal; i++) {
        if (mask[i] & groupbit) {  // 如果原子的掩码与 groupbit 相与为真
            dt_vector[i] = particle_dt[i];  // 将 particle_dt 中对应位置的值赋给 dt_vector
        } else {
            dt_vector[i] = 0.0;  // 否则将 dt_vector 中对应位置设为 0.0
        }
    }
}


/* ----------------------------------------------------------------------
 memory usage of local atom-based array  local atom-based array的内存使用情况"
 ------------------------------------------------------------------------- */
double ComputeSMDTlsphDt::memory_usage() {
        double bytes = nmax * sizeof(double);
        return bytes;
}
