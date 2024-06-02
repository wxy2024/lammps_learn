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
 
//定义一个计算，用于计算每个粒子的质量密度。
//质量密度是粒子的质量（在模拟过程中保持恒定）除以其体积（由于机械变形而可能改变）得到的。
//https://docs.lammps.org/compute_smd_rho.html
//从原子中传入密度
#include <string.h>
#include "compute_smd_rho.h"
#include "atom.h"
#include "update.h"
#include "modify.h"
#include "comm.h"
#include "force.h"
#include "memory.h"
#include "error.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ComputeSMDRho::ComputeSMDRho(LAMMPS *lmp, int narg, char **arg) :
		Compute(lmp, narg, arg) {
	if (narg != 3)
		error->all(FLERR, "Illegal compute smd/rho command");
	if (atom->vfrac_flag != 1)
		error->all(FLERR, "compute smd/rho command requires atom_style with volume (e.g. smd)");

	peratom_flag = 1;
	size_peratom_cols = 0;

	nmax = 0;
	rhoVector = NULL;
}

/* ---------------------------------------------------------------------- */

ComputeSMDRho::~ComputeSMDRho() {
	memory->sfree(rhoVector);
}

/* ---------------------------------------------------------------------- */

void ComputeSMDRho::init() {

	int count = 0;
	for (int i = 0; i < modify->ncompute; i++)
		if (strcmp(modify->compute[i]->style, "smd/rho") == 0)
			count++;
	if (count > 1 && comm->me == 0)
		error->warning(FLERR, "More than one compute smd/rho");
}

/* ---------------------------------------------------------------------- */

void ComputeSMDRho::compute_peratom() {//密度，将需要计算密度的原子密度存储到rhoVector，其余置为0
	invoked_peratom = update->ntimestep;

	// grow rhoVector array if necessary

	if (atom->nmax > nmax) {
		memory->sfree(rhoVector);
		nmax = atom->nmax;
		rhoVector = (double *) memory->smalloc(nmax * sizeof(double), "atom:rhoVector");
		vector_atom = rhoVector;
	}

	double *rho = atom->rho;
	int *mask = atom->mask;
	int nlocal = atom->nlocal;

	for (int i = 0; i < nlocal; i++) {
		if (mask[i] & groupbit) {
			rhoVector[i] = rho[i];
		} else {
			rhoVector[i] = 0.0;
		}
	}

}

/* ----------------------------------------------------------------------
 memory usage of local atom-based array
 ------------------------------------------------------------------------- */

double ComputeSMDRho::memory_usage() {
	double bytes = nmax * sizeof(double);
	return bytes;
}
