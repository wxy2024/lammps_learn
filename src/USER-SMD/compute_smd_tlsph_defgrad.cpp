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

void ComputeSMDTLSPHDefgrad::compute_peratom() {//计算每个原子的变形梯度
	double **defgrad = atom->smd_data_9;//defgrad存储梯度
	Matrix3d F;//变形梯度
	invoked_peratom = update->ntimestep;

	// grow vector array if necessary
	if (atom->nmax > nmax) {
		memory->destroy(defgradVector);
		nmax = atom->nmax;
		memory->create(defgradVector, nmax, size_peratom_cols, "defgradVector");
		array_atom = defgradVector;
	}

	// copy data to output array
	int itmp = 0;
	Matrix3d *Fincr = (Matrix3d *) force->pair->extract("smd/tlsph/Fincr_ptr", itmp);
	if (Fincr == NULL) {
		error->all(FLERR, "compute smd/tlsph_strain failed to access Fincr array");
	}

	int *mask = atom->mask;
	int nlocal = atom->nlocal;

	//下面这段代码就是把所有满足条件原子的变形梯度传入，然后乘以增量（更新），然后又把所有原子更新后的变形梯度放到defgradVector里
	for (int i = 0; i < nlocal; i++) {
		if (mask[i] & groupbit) {
			F(0, 0) = defgrad[i][0];
			F(0, 1) = defgrad[i][1];
			F(0, 2) = defgrad[i][2];
			F(1, 0) = defgrad[i][3];
			F(1, 1) = defgrad[i][4];
			F(1, 2) = defgrad[i][5];
			F(2, 0) = defgrad[i][6];
			F(2, 1) = defgrad[i][7];
			F(2, 2) = defgrad[i][8];//传入变形梯度

			F = F * Fincr[i];//乘以变形梯度增量，以便于更新变形梯度（7）
			defgradVector[i][0] = F(0, 0);
			defgradVector[i][1] = F(0, 1);
			defgradVector[i][2] = F(0, 2);
			defgradVector[i][3] = F(1, 0);
			defgradVector[i][4] = F(1, 1);
			defgradVector[i][5] = F(1, 2);
			defgradVector[i][6] = F(2, 0);
			defgradVector[i][7] = F(2, 1);
			defgradVector[i][8] = F(2, 2);
			defgradVector[i][9] = F.determinant();//determinant() 方法用于计算张量或矩阵的行列式值
		} else {
			defgradVector[i][0] = 1.0;//将变形梯度设为单位矩阵，保证原子没有变形
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
