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
/*该修正为根据Total-Lagrangian SPH pair style进行相互作用的粒子执行显式时间积分。
limit_velocity关键字将控制速度，如果速度超过此速度限制，则将速度向量的范数缩放到max_vel。*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "fix_smd_integrate_tlsph.h"
#include "atom.h"
#include "force.h"
#include "update.h"
#include "error.h"
#include "pair.h"
#include "neigh_list.h"
#include <Eigen/Eigen>
#include "domain.h"
#include "neighbor.h"
#include "comm.h"
#include "modify.h"
#include <stdio.h>
#include <iostream>

using namespace Eigen;
using namespace LAMMPS_NS;
using namespace FixConst;
using namespace std;

/* ---------------------------------------------------------------------- */

FixSMDIntegrateTlsph::FixSMDIntegrateTlsph(LAMMPS *lmp, int narg, char **arg) :
                Fix(lmp, narg, arg) {
    // 检查输入参数数量是否小于3，如果是则输出错误信息
    if (narg < 3) {
        printf("narg=%d\n", narg);
        error->all(FLERR, "Illegal fix smd/integrate_tlsph command");
    }

    xsphFlag = false;  // 初始化xsphFlag为false
    vlimit = -1.0;      // 初始化vlimit为-1.0
    int iarg = 3;       // 初始化参数索引iarg为3

    if (comm->me == 0) {
        // 如果当前进程是0号进程，则输出相关信息
        printf("\n>>========>>========>>========>>========>>========>>========>>========>>========\n");
        printf("fix smd/integrate_tlsph is active for group: %s \n", arg[1]);
    }

    while (true) {
        if (iarg >= narg) {  // 如果参数索引超出范围，则跳出循环
            break;
        }

        if (strcmp(arg[iarg], "xsph") == 0) {
            // 如果参数为"xsph"，将xsphFlag设置为true，并输出相关信息（暂时不可用）
            xsphFlag = true;
            if (comm->me == 0) {
                error->one(FLERR, "XSPH is currently not available");
                printf("... will use XSPH time integration\n");
            }
        } else if (strcmp(arg[iarg], "limit_velocity") == 0) {
            // 如果参数为"limit_velocity"，获取下一个参数作为限制速度的值
            iarg++;
            if (iarg == narg) {
                error->all(FLERR, "expected number following limit_velocity");
            }
            // 将下一个参数解析为数值，作为速度限制值，并输出相关信息
            vlimit = force->numeric(FLERR, arg[iarg]);
            if (comm->me == 0) {
                printf("... will limit velocities to <= %g\n", vlimit);
            }
        } else {
            // 如果遇到其他未知选项，则输出错误信息
            char msg[128];
            sprintf(msg, "Illegal keyword for smd/integrate_tlsph: %s\n", arg[iarg]);
            error->all(FLERR, msg);
        }

        iarg++;  // 处理下一个参数
    }

    if (comm->me == 0) {
        // 如果当前进程是0号进程，则输出相关信息
        printf(">>========>>========>>========>>========>>========>>========>>========>>========\n");
    }

    time_integrate = 1;  // 设置time_integrate为1

    // 设置comm sizes needed by this fix
    atom->add_callback(0);  // 调用atom对象的add_callback函数，参数为0
}


/* ---------------------------------------------------------------------- */

int FixSMDIntegrateTlsph::setmask() {//确定要执行的操作
        int mask = 0;
        mask |= INITIAL_INTEGRATE;
        mask |= FINAL_INTEGRATE;
        return mask;
}

/* ---------------------------------------------------------------------- */

void FixSMDIntegrateTlsph::init() {
    // 获取时间步长 dt 的数值并存储在变量 dtv 中
    dtv = update->dt;

    // 计算 dtf，它是时间步长 dt 和力转换因子 ftm2v 的乘积的一半
    // force->ftm2v 是一个将力转换为速度变更的因子 转换因子 = F/m。
    dtf = 0.5 * update->dt * force->ftm2v;

    // 计算速度限制 vlimit 的平方，并存储在变量 vlimitsq 中
    // 这个值用于在后续的速度更新中进行速度限制检查
    vlimitsq = vlimit * vlimit;
}


/* ----------------------------------------------------------------------
 ------------------------------------------------------------------------- */

void FixSMDIntegrateTlsph::initial_integrate(int vflag) {
        double dtfm, vsq, scale;

        // update v and x of atoms in group

        double **x = atom->x;
        double **v = atom->v;
        double **vest = atom->vest;
        double **f = atom->f;
        double *rmass = atom->rmass;
        int *mask = atom->mask;
        int nlocal = atom->nlocal;
        int itmp;
        double vxsph_x, vxsph_y, vxsph_z;
        if (igroup == atom->firstgroup)
                nlocal = atom->nfirst;

        Vector3d *smoothVelDifference = (Vector3d *) force->pair->extract("smd/tlsph/smoothVel_ptr", itmp);

        if (xsphFlag) {
                if (smoothVelDifference == NULL) {
                        error->one(FLERR,
                                        "fix smd/integrate_tlsph failed to access smoothVel array. Check if a pair style exist which calculates this quantity.");
                }
        }

        for (int i = 0; i < nlocal; i++) {// Velocity Verlet 积分算法
                if (mask[i] & groupbit) {
                        dtfm = dtf / rmass[i];//计算中间变量

                        // 1st part of Velocity_Verlet: push velocties 1/2 time increment ahead
                        //速度Verlet算法的第一部分：将速度推进半个时间步长
                        v[i][0] += dtfm * f[i][0];//更新粒子速度
                        v[i][1] += dtfm * f[i][1];
                        v[i][2] += dtfm * f[i][2];

			// if (atom->tag[i] == 18268)
			//   printf("Step %d INTEGRATION, Particle %d: f = [%.10e %.10e %.10e]\n",update->ntimestep, atom->tag[i], f[i][0], f[i][1], f[i][2]);

			if (vlimit > 0.0) {
                                vsq = v[i][0] * v[i][0] + v[i][1] * v[i][1] + v[i][2] * v[i][2];//速度的平方
                                if (vsq > vlimitsq) {//如果速度超出限制，就进行缩放
                                        scale = sqrt(vlimitsq / vsq);
                                        v[i][0] *= scale;
                                        v[i][1] *= scale;
                                        v[i][2] *= scale;
                                }
                        }

                        if (xsphFlag) {//如果使用了XSPH方法，会根据这种方法更新速度

                                // construct XSPH velocity
                                vxsph_x = v[i][0] + 0.5 * smoothVelDifference[i](0);//平滑速度差
                                vxsph_y = v[i][1] + 0.5 * smoothVelDifference[i](1);
                                vxsph_z = v[i][2] + 0.5 * smoothVelDifference[i](2);

                                vest[i][0] = vxsph_x + dtfm * f[i][0];
                                vest[i][1] = vxsph_y + dtfm * f[i][1];
                                vest[i][2] = vxsph_z + dtfm * f[i][2];

                                x[i][0] += dtv * vxsph_x;//更新粒子位置
                                x[i][1] += dtv * vxsph_y;
                                x[i][2] += dtv * vxsph_z;
                        } else {

                                // 从半步外推速度到全步
                                vest[i][0] = v[i][0] + dtfm * f[i][0];//用节点力更新速度
                                vest[i][1] = v[i][1] + dtfm * f[i][1];
                                vest[i][2] = v[i][2] + dtfm * f[i][2];

                                x[i][0] += dtv * v[i][0]; // 2nd part of Velocity-Verlet: push positions one full time increment ahead
                                x[i][1] += dtv * v[i][1];//更新粒子位置
                                x[i][2] += dtv * v[i][2];
                        }
                }
        }

}

/* ---------------------------------------------------------------------- */

void FixSMDIntegrateTlsph::final_integrate() {
        double dtfm, vsq, scale;

// update v of atoms in group

        double **v = atom->v;
        double **f = atom->f;
        double *e = atom->e;
        double *de = atom->de;
        double *rmass = atom->rmass;
        int *mask = atom->mask;
        int nlocal = atom->nlocal;
        if (igroup == atom->firstgroup)
                nlocal = atom->nfirst;
        int i;

        for (i = 0; i < nlocal; i++) {
                if (mask[i] & groupbit) {
                        dtfm = dtf / rmass[i];

                        v[i][0] += dtfm * f[i][0]; // 3rd part of Velocity-Verlet: push velocities another half time increment ahead
                        v[i][1] += dtfm * f[i][1]; // both positions and velocities are now defined at full time-steps.
                        v[i][2] += dtfm * f[i][2];

                        // limit velocity
                        if (vlimit > 0.0) {
                                vsq = v[i][0] * v[i][0] + v[i][1] * v[i][1] + v[i][2] * v[i][2];
                                if (vsq > vlimitsq) {
                                        scale = sqrt(vlimitsq / vsq);
                                        v[i][0] *= scale;
                                        v[i][1] *= scale;
                                        v[i][2] *= scale;
                                }
                        }

                        e[i] += dtv * de[i];
                }
        }
}

/* ---------------------------------------------------------------------- */

void FixSMDIntegrateTlsph::reset_dt() {
        dtv = update->dt;
        dtf = 0.5 * update->dt * force->ftm2v;
        vlimitsq = vlimit * vlimit;
}

/* ---------------------------------------------------------------------- */
