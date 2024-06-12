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
//定义一个计算，用于计算变形梯度。这仅对按照全拉格朗日SPH对样式进行相互作用的粒子具有意义。
//https://docs.lammps.org/compute_smd_tlsph_defgrad.html
//从原子中传入变形梯度，满足条件的更新
#include <string.h>
#include "compute_smd_tlsph_defgrad.h"
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

ComputeSMDTLSPHDefgrad::ComputeSMDTLSPHDefgrad(LAMMPS *lmp, int narg, char **arg) :
		Compute(lmp, narg, arg) {
	if (narg != 3)
		error->all(FLERR, "Illegal compute smd/tlsph_defgrad command");

	peratom_flag = 1;
	size_peratom_cols = 10;

	nmax = 0;
	defgradVector = NULL;
}

/* ---------------------------------------------------------------------- */

ComputeSMDTLSPHDefgrad::~ComputeSMDTLSPHDefgrad() {
	memory->sfree(defgradVector);
}

/* ---------------------------------------------------------------------- */

void ComputeSMDTLSPHDefgrad::init() {//检查合法性

	int count = 0;
	for (int i = 0; i < modify->ncompute; i++)
		if (strcmp(modify->compute[i]->style, "smd/tlsph_defgrad") == 0)
			count++;
	if (count > 1 && comm->me == 0)
		error->warning(FLERR, "More than one compute smd/tlsph_defgrad");
}

/* ---------------------------------------------------------------------- */

void ComputeSMDTLSPHDefgrad::compute_peratom() {
	// 获取存储变形梯度的二维数组
	double **defgrad = atom->smd_data_9;
	// 定义一个3x3矩阵F用于存储变形梯度
	Matrix3d F;
	// 记录当前时间步
	invoked_peratom = update->ntimestep;

	// 如果原子的最大数目超过了当前存储的最大数目，则需要扩展数组
	if (atom->nmax > nmax) {
		// 销毁旧的变形梯度向量数组
		memory->destroy(defgradVector);
		// 更新最大数目
		nmax = atom->nmax;
		// 创建新的变形梯度向量数组
		memory->create(defgradVector, nmax, size_peratom_cols, "defgradVector");
		// 将新的数组赋值给array_atom
		array_atom = defgradVector;
	}

	// 临时变量，用于提取Fincr指针
	int itmp = 0;
	// 提取变形梯度增量矩阵数组的指针
	Matrix3d *Fincr = (Matrix3d *) force->pair->extract("smd/tlsph/Fincr_ptr", itmp);
	// 如果提取失败，抛出错误
	if (Fincr == NULL) {
		error->all(FLERR, "compute smd/tlsph_strain failed to access Fincr array");
	}

	// 获取原子的掩码数组
	int *mask = atom->mask;
	// 获取本地原子数
	int nlocal = atom->nlocal;

	// 遍历所有本地原子
	for (int i = 0; i < nlocal; i++) {
		// 如果原子的掩码符合组掩码条件
		if (mask[i] & groupbit) {
			// 将defgrad数组中的数据赋值给F矩阵
			F(0, 0) = defgrad[i][0];
			F(0, 1) = defgrad[i][1];
			F(0, 2) = defgrad[i][2];
			F(1, 0) = defgrad[i][3];
			F(1, 1) = defgrad[i][4];
			F(1, 2) = defgrad[i][5];
			F(2, 0) = defgrad[i][6];
			F(2, 1) = defgrad[i][7];
			F(2, 2) = defgrad[i][8];

			// 用当前的变形梯度乘以增量变形梯度，更新变形梯度
			F = F * Fincr[i];

			// 将更新后的变形梯度存储到defgradVector数组中
			defgradVector[i][0] = F(0, 0);
			defgradVector[i][1] = F(0, 1);
			defgradVector[i][2] = F(0, 2);
			defgradVector[i][3] = F(1, 0);
			defgradVector[i][4] = F(1, 1);
			defgradVector[i][5] = F(1, 2);
			defgradVector[i][6] = F(2, 0);
			defgradVector[i][7] = F(2, 1);
			defgradVector[i][8] = F(2, 2);
			// 计算变形梯度的行列式，并存储到defgradVector数组中
			defgradVector[i][9] = F.determinant();
		} else {
			// 如果原子不符合组掩码条件，则将变形梯度设为单位矩阵，表示原子没有变形
			defgradVector[i][0] = 1.0;
			defgradVector[i][1] = 0.0;
			defgradVector[i][2] = 0.0;
			defgradVector[i][3] = 0.0;
			defgradVector[i][4] = 1.0;
			defgradVector[i][5] = 0.0;
			defgradVector[i][6] = 0.0;
			defgradVector[i][7] = 0.0;
			defgradVector[i][8] = 1.0;
			defgradVector[i][9] = 1.0;
		}
	}
}


/* ----------------------------------------------------------------------
 memory usage of local atom-based array
 ------------------------------------------------------------------------- */

double ComputeSMDTLSPHDefgrad::memory_usage() {
	double bytes = size_peratom_cols * nmax * sizeof(double);
	return bytes;
}
