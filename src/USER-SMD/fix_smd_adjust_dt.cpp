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
/*该修正计算一个新的稳定时间步长，用于在SMD时间积分器中使用。

稳定时间步长基于多个条件。对于SPH pair styles，会评估一个CFL准则（Courant, Friedrichs & Lewy, 1928），该准则确定声速在一个时间步长内不能传播得比粒子之间的典型间距更远，以确保不会丢失任何信息。对于接触对样式，会对成对势能进行线性分析，确定一个稳定的最大时间步长。

该修正查询定义了此修正的组中所有粒子中的最小稳定时间增量。额外的安全系数 s_fact 应用于时间增量。*/
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "fix_smd_adjust_dt.h"
#include "fix_smd_indent.h"
#include "fix_smd_indent_linear.h"
#include "atom.h"
#include "update.h"
#include "integrate.h"
#include "domain.h"
#include "lattice.h"
#include "force.h"
#include "pair.h"
#include "modify.h"
#include "fix.h"
#include "output.h"
#include "dump.h"
#include "comm.h"
#include "error.h"

using namespace LAMMPS_NS;
using namespace FixConst;

#define BIG 1.0e20

/* ---------------------------------------------------------------------- */

FixSMDTlsphDtReset::FixSMDTlsphDtReset(LAMMPS *lmp, int narg, char **arg) :
		Fix(lmp, narg, arg) {
	if (narg != 4)
		error->all(FLERR, "Illegal fix smd/adjust_dt command");

	// set time_depend, else elapsed time accumulation can be messed up

	time_depend = 1;
	scalar_flag = 1;
	vector_flag = 1;
	size_vector = 2;
	global_freq = 1;
	extscalar = 0;
	extvector = 0;
	restart_global = 1; // this fix stores global (i.e., not per-atom) info: elaspsed time

	safety_factor = atof(arg[3]);

	// initializations
	t_elapsed = 0.0;
}

/* ---------------------------------------------------------------------- */

int FixSMDTlsphDtReset::setmask() {
	int mask = 0;
	mask |= INITIAL_INTEGRATE;
	mask |= END_OF_STEP;
	return mask;
}

/* ---------------------------------------------------------------------- */

void FixSMDTlsphDtReset::init() {
	dt = update->dt;
}

/* ---------------------------------------------------------------------- */

void FixSMDTlsphDtReset::setup(int vflag) {
	end_of_step();
}

/* ---------------------------------------------------------------------- */

void FixSMDTlsphDtReset::initial_integrate(int vflag) {

	t_elapsed += update->dt;//累计时间
	update->update_time();
	// printf("in adjust_dt: dt = %.10e, t_elapsed = %.10e\n", update->dt, t_elapsed);
}

/* ---------------------------------------------------------------------- */

void FixSMDTlsphDtReset::end_of_step() {
	double dtmin = BIG;
	int itmp = 0;

	/*
	 * extract minimum CFL timestep from TLSPH and ULSPH pair styles从 TLSPH 和 ULSPH 对样式中提取最小 CFL 时间步长
	 */

	double *dtCFL_TLSPH = (double *) force->pair->extract("smd/tlsph/dtCFL_ptr", itmp);
	double *dtCFL_ULSPH = (double *) force->pair->extract("smd/ulsph/dtCFL_ptr", itmp);
	double *dt_TRI = (double *) force->pair->extract("smd/tri_surface/stable_time_increment_ptr", itmp);
	double *dt_HERTZ = (double *) force->pair->extract("smd/hertz/stable_time_increment_ptr", itmp);
	double *dt_PERI_IPMB = (double *) force->pair->extract("smd/peri_ipmb/stable_time_increment_ptr", itmp);
	double *dt_FIX_INDENT = NULL;
	double *dt_FIX_INDENT_LINEAR = NULL;
	// find associated smd/indent fix that must exist
	// could have changed locations in fix list since created

	int ifix_indent = -1;
	for (int i = 0; i < modify->nfix; i++)
		if (strcmp(modify->fix[i]->style, "smd/indent") == 0)
			ifix_indent = i;
	if (ifix_indent != -1) {
	  dt_FIX_INDENT = &((FixSMDIndent *) modify->fix[ifix_indent])->dtCFL;
	}

	int ifix_indent_linear = -1;
	for (int i = 0; i < modify->nfix; i++)
		if (strcmp(modify->fix[i]->style, "smd/indent/linear") == 0)
			ifix_indent_linear = i;
	if (ifix_indent_linear != -1) {
	  dt_FIX_INDENT_LINEAR = &((FixSMDIndentLinear *) modify->fix[ifix_indent_linear])->dtCFL;
	}

	if ((dtCFL_TLSPH == NULL) && (dtCFL_ULSPH == NULL) && (dt_TRI == NULL) && (dt_HERTZ == NULL)
			&& (dt_PERI_IPMB == NULL)) {
		error->all(FLERR, "fix smd/adjust_dt failed to access a valid dtCFL");
	}

	if (dtCFL_TLSPH != NULL) {
		dtmin = MIN(dtmin, *dtCFL_TLSPH);
	}

	if (dtCFL_ULSPH != NULL) {
		dtmin = MIN(dtmin, *dtCFL_ULSPH);
	}

	if (dt_TRI != NULL) {
		dtmin = MIN(dtmin, *dt_TRI);
	}

	if (dt_HERTZ != NULL) {
		dtmin = MIN(dtmin, *dt_HERTZ);
	}

	if (dt_PERI_IPMB != NULL) {
		dtmin = MIN(dtmin, *dt_PERI_IPMB);
	}

	if (dt_FIX_INDENT != NULL) {
	  dtmin = MIN(dtmin, *dt_FIX_INDENT);
	}

	if (dt_FIX_INDENT_LINEAR != NULL) {
	  dtmin = MIN(dtmin, *dt_FIX_INDENT_LINEAR);
	}

//	double **v = atom->v;
//	double **f = atom->f;
//	double *rmass = atom->rmass;
//	double *radius = atom->radius;
//	int *mask = atom->mask;
//	int nlocal = atom->nlocal;
//	double dtv, dtf, dtsq;
//	double vsq, fsq, massinv, xmax;
//	double delx, dely, delz, delr;

//	for (int i = 0; i < nlocal; i++) {
//		if (mask[i] & groupbit) {
//			xmax = 0.005 * radius[i];
//			massinv = 1.0 / rmass[i];
//			vsq = v[i][0] * v[i][0] + v[i][1] * v[i][1] + v[i][2] * v[i][2];
//			fsq = f[i][0] * f[i][0] + f[i][1] * f[i][1] + f[i][2] * f[i][2];
//			dtv = dtf = BIG;
//			if (vsq > 0.0)
//				dtv = xmax / sqrt(vsq);
//			if (fsq > 0.0)
//				dtf = sqrt(2.0 * xmax / (sqrt(fsq) * massinv));
//			dt = MIN(dtv, dtf);
//			dtmin = MIN(dtmin, dt);
//			dtsq = dt * dt;
//			delx = dt * v[i][0] + 0.5 * dtsq * massinv * f[i][0];
//			dely = dt * v[i][1] + 0.5 * dtsq * massinv * f[i][1];
//			delz = dt * v[i][2] + 0.5 * dtsq * massinv * f[i][2];
//			delr = sqrt(delx * delx + dely * dely + delz * delz);
//			if (delr > xmax)
//				dt *= xmax / delr;
//			dtmin = MIN(dtmin, dt);
//
////			xmax = 0.05 * radius[i];
////			massinv = 1.0 / rmass[i];
////			fsq = f[i][0] * f[i][0] + f[i][1] * f[i][1] + f[i][2] * f[i][2];
////			dtf = BIG;
////			if (fsq > 0.0)
////				dtf = sqrt(2.0 * xmax / (sqrt(fsq) * massinv));
////			dtmin = MIN(dtmin, dtf);
//		}
//	}

	dtmin *= safety_factor; // apply safety factor应用安全系数
	// Limit the increase to 0.5% of previous time step:将增幅限制在上一时间步长的 0.5%：

	dtmin = MIN(dtmin, 1.005 * update->dt);
	MPI_Allreduce(&dtmin, &dt, 1, MPI_DOUBLE, MPI_MIN, world);

	if (update->ntimestep == 0 || update->dt < 1.0e-16) {
		dt = 1.0e-16;
	}

	//printf("dtmin is now: %f, dt is now%f\n", dtmin, dt);


	if (dt != 0) update->dt = dt; // At restart dt can be null
	if (force->pair)
		force->pair->reset_dt();
	for (int i = 0; i < modify->nfix; i++)
		modify->fix[i]->reset_dt();
}

/* ---------------------------------------------------------------------- */

double FixSMDTlsphDtReset::compute_scalar() {
	return t_elapsed;
}

/* ----------------------------------------------------------------------
 pack entire state of Fix into one write将整个固定状态打包成一个文件
 ------------------------------------------------------------------------- */

void FixSMDTlsphDtReset::write_restart(FILE *fp) {
	int n = 0;
	double list[2];
	list[n++] = t_elapsed;
	list[n++] = update->dt;

	if (comm->me == 0) {
		int size = n * sizeof(double);
		fwrite(&size, sizeof(int), 1, fp);
		fwrite(list, sizeof(double), n, fp);
	}
}

/* ----------------------------------------------------------------------
 use state info from restart file to restart the Fix
 ------------------------------------------------------------------------- */

void FixSMDTlsphDtReset::restart(char *buf) {
	int n = 0;
	double *list = (double *) buf;
	t_elapsed = list[n++];
	update->atime = t_elapsed;
	update->atimestep = update->ntimestep;
	update->dt = list[n++];
}
