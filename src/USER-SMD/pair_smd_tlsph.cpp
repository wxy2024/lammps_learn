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
#include "group.h"
#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <string.h>
#include "pair_smd_tlsph.h"
#include "fix_smd_tlsph_reference_configuration.h"
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
#include <stdio.h>
#include <iostream>
#include "math_special.h"
#include <map>
#include "update.h"
#include <Eigen/Eigen>
#include "smd_material_models.h"
#include "smd_kernels.h"
#include "smd_math.h"
#include "output.h"
using namespace SMD_Kernels;
using namespace Eigen;
using namespace std;
using namespace LAMMPS_NS;
using namespace SMD_Math;

#define JAUMANN false
#define DETF_MIN 0.002 // maximum compression deformation allow最大压缩变形量
#define DETF_MAX 200.0 // maximum tension deformation allowed允许的最大拉伸变形
#define TLSPH_DEBUG 0
#define PLASTIC_STRAIN_AVERAGE_WINDOW 100.0
#define BUFEXTRA 1000
/*"smd/tlsph" 样式根据连续介质力学本构定律和全拉格朗日平滑粒子流体动力学算法计算粒子之间的相互作用。*/
/* ---------------------------------------------------------------------- */

PairTlsph::PairTlsph(LAMMPS *lmp) :
		Pair(lmp) {

	onerad_dynamic = onerad_frozen = maxrad_dynamic = maxrad_frozen = NULL;

	failureModel = NULL;
	strengthModel = eos = NULL;

	nmax = 0; // make sure no atom on this proc such that initial memory allocation is correct
	Fdot = Fincr = PK1 = NULL;
	R = FincrInv = W = D = NULL;
	detF = NULL;
	smoothVelDifference = NULL;
	numNeighsRefConfig = NULL;
	CauchyStress = NULL;
	hourglass_error = NULL;
	Lookup = NULL;
	particle_dt = NULL;
	vijSq_max = NULL;
	max_mu_ij = NULL;
	damage_increment = NULL;
	eff_plastic_strain_at_failure_init = NULL;

	updateFlag = 0;
	first = true;
	dtCFL = 0.0; // initialize dtCFL so it is set to safe value if extracted on zero-th timestep
	rSqMin = NULL;
	dvMax = NULL;
	flowstress_slope = NULL;
	shepardWeightInv = NULL;

	comm_forward = 26; // this pair style communicates 20 doubles to ghost atoms : PK1 tensor + F tensor + damage_increment + npartner + flowstress_slope
	fix_tlsph_reference_configuration = NULL;

	cut_comm = MAX(neighbor->cutneighmax, comm->cutghostuser); // cutoff radius within which ghost atoms are communicated.

	// Create MPI_failure_types:
	int blockcounts = 10;
	MPI_Aint offsets = 0;
	MPI_Datatype oldtypes = MPI_C_BOOL;
	MPI_Type_struct(1, &blockcounts, &offsets, &oldtypes, &MPI_failure_types);
	MPI_Type_commit(&MPI_failure_types);

	maxsend = 0;
	maxrecv = 0;
}

/* ---------------------------------------------------------------------- */

PairTlsph::~PairTlsph() {
	//printf("in PairTlsph::~PairTlsph()\n");

	if (allocated) {
		memory->destroy(setflag);
		memory->destroy(cutsq);
		memory->destroy(strengthModel);
		memory->destroy(eos);
		memory->destroy(Lookup);

		delete[] onerad_dynamic;
		delete[] onerad_frozen;
		delete[] maxrad_dynamic;
		delete[] maxrad_frozen;

		delete[] Fdot;
		delete[] Fincr;
		delete[] detF;
		delete[] PK1;
		delete[] smoothVelDifference;
		delete[] R;
		delete[] FincrInv;
		delete[] W;
		delete[] D;
		delete[] numNeighsRefConfig;
		delete[] CauchyStress;
		delete[] hourglass_error;
		delete[] particle_dt;
		delete[] vijSq_max;
		delete[] max_mu_ij;
		delete[] damage_increment;
		delete[] eff_plastic_strain_at_failure_init;
		delete[] rSqMin;
		delete[] dvMax;
		delete[] flowstress_slope;
		delete[] shepardWeightInv;

		delete[] failureModel;
		memory->destroy(nrecv);
		memory->destroy(buf_send);
		memory->destroy(buf_recv);
	}
	MPI_Type_free(&MPI_failure_types);
}

/* ----------------------------------------------------------------------
 *
 * use half neighbor list to re-compute shape matrix
 *
 ---------------------------------------------------------------------- */

void PairTlsph::PreCompute() {
	tagint *mol = atom->molecule;
	double *vfrac = atom->vfrac;
	double *radius = atom->radius;
	double **x0 = atom->x0;
	double **x = atom->x;
	double **v = atom->vest; // extrapolated velocities corresponding to current positions
	double **vint = atom->v; // Velocity-Verlet algorithm velocities
	double *damage = atom->damage;
	tagint *tag = atom->tag;
	int *type = atom->type;
	int nlocal = atom->nlocal;
	double dt = update->dt;
	int jnum, jj, i, j, itype, idim;

	tagint **partner = ((FixSMD_TLSPH_ReferenceConfiguration *) modify->fix[ifix_tlsph])->partner;
	int *npartner = ((FixSMD_TLSPH_ReferenceConfiguration *) modify->fix[ifix_tlsph])->npartner;
	float **wfd_list = ((FixSMD_TLSPH_ReferenceConfiguration *) modify->fix[ifix_tlsph])->wfd_list;
	float **wf_list = ((FixSMD_TLSPH_ReferenceConfiguration *) modify->fix[ifix_tlsph])->wf_list;
	float **degradation_ij = ((FixSMD_TLSPH_ReferenceConfiguration *) modify->fix[ifix_tlsph])->degradation_ij;
	Vector3d **partnerdx = ((FixSMD_TLSPH_ReferenceConfiguration *) modify->fix[ifix_tlsph])->partnerdx;
	Vector3d **g_list = ((FixSMD_TLSPH_ReferenceConfiguration *) modify->fix[ifix_tlsph])->g_list;
	Matrix3d *K = ((FixSMD_TLSPH_ReferenceConfiguration *) modify->fix[ifix_tlsph])->K;
	double rSq, wf, vwf, wfd, h, irad, voli, volj, shepardWeight;
	Vector3d dx, dx0, dx0mirror, dv, g;
	Matrix3d L, U, eye;
	Vector3d vi, vj, vinti, vintj, xi, xj, x0i, x0j, dvint;
	int periodic = (domain->xperiodic || domain->yperiodic || domain->zperiodic);
	bool status;
	Matrix3d F0;

	dtCFL = 1.0e22;

	eye.setIdentity();
	for (i = 0; i < nlocal; i++) {
		vijSq_max[i] = 0.0;//原子i和所有j的速度平方的最大值

		itype = type[i];
		
		if (setflag[itype][itype] == 1) {
			Fincr[i].setZero();
			Fdot[i].setZero();
			numNeighsRefConfig[i] = 0;
			smoothVelDifference[i].setZero();
			hourglass_error[i] = 0.0;

			if (mol[i] < 0) { // valid SPH particle have mol > 0
				continue;
			}

			// initialize aveage mass density
			h = 2.0 * radius[i];//作用范围的直径
			spiky_kernel_and_derivative(h, 0.0, domain->dimension, wf, wfd);//传入核函数及其导数

			jnum = npartner[i];//作用范围内的原子数量
			irad = radius[i];//作用半径
			voli = vfrac[i];//体积

			shepardWeight = wf * voli;//核函数乘体积

			// initialize Eigen data structures from LAMMPS data structures
			for (idim = 0; idim < 3; idim++) {
				xi(idim) = x[i][idim];
				x0i(idim) = x0[i][idim];
				vi(idim) = v[i][idim];
				vinti(idim) = vint[i][idim];
			}
			
			//Matrix3d gradAbsX;
			//gradAbsX.setZero();

			rSqMin[i] = 1.0e22;
			dvMax[i] = 1.0e-15;

			for (jj = 0; jj < jnum; jj++) {

				// if (degradation_ij[i][jj] >= 1.0) continue;
				
				j = atom->map(partner[i][jj]);
				if (j < 0) { //			// check if lost a partner without first breaking bond
				  printf("Link between %d and %d destroyed without first breaking bond! Damage level in the link was: %f\n", tag[i], partner[i][jj], degradation_ij[i][jj]);
				  error->all(FLERR, "Bond broken");
				  degradation_ij[i][jj] = 1.0;
				  continue;
				}

				if (abs(mol[i]) != abs(mol[j])) {//绝对值相等
					continue;
				}


				// distance vectors in current and reference configuration, velocity difference
				// initialize Eigen data structures from LAMMPS data structures without for loop to maximize memory performance
				x0j(0) = x0[j][0];
				x0j(1) = x0[j][1];
				x0j(2) = x0[j][2];
				dx0 = x0j - x0i;//Xij，参考配置下两个原子之间的距离？

				// initialize Eigen data structures from LAMMPS data structures
				xj(0) = x[j][0];
				xj(1) = x[j][1];
				xj(2) = x[j][2];
				dx = xj - xi;   //变化后的两个原子之间的距离

				// initialize Eigen data structures from LAMMPS data structures
				vj(0) = v[j][0];
				vj(1) = v[j][1];
				vj(2) = v[j][2];
				dv = vj - vi;

				// initialize Eigen data structures from LAMMPS data structures
				vintj(0) = vint[j][0];
				vintj(1) = vint[j][1];
				vintj(2) = vint[j][2];
				dvint = vintj - vinti;

				volj = vfrac[j];

				if (failureModel[itype].integration_point_wise == true) {
				  if (damage[j] > 0.0) {//如果原子j有损伤，i和j之间的距离要增加？
				    dv *= (1-damage[j]);
				    dvint *= (1-damage[j]);
				    partnerdx[i][jj] += dt * dv;
				    dx = partnerdx[i][jj];
				    // volj *= scale;
				    }
				}

				if (isnan(dx[0]) || isnan(dx[1]) || isnan(dx[2])) {
				  printf("x[%d] - x[%d] = [%f %f %f] - [%f %f %f], di = %f, dj = %f\n", tag[j], tag[i], xj[0], xj[1], xj[2], xi[0], xi[1], xi[2], damage[i], damage[j]);
				}
				rSq = dx.squaredNorm(); // current distance
				rSqMin[i] = MIN(rSq,rSqMin[i]);
				dvMax[i] = MAX(dv.norm(),dvMax[i]);

				if (periodic)
					domain->minimum_image(dx0(0), dx0(1), dx0(2));


				vijSq_max[i] = MAX(dv.squaredNorm(), vijSq_max[i]);
				vijSq_max[i] = MAX(dvint.squaredNorm(), vijSq_max[i]);//更新最大 速度差的平方

				vwf = volj * wf_list[i][jj];
				g = volj * g_list[i][jj];    //虚拟原子 这两个权重分别代表什么？？？

				/* build matrices */;
				Fdot[i].noalias() -= dv * g.transpose();//（8）更新变形矩阵的导数
				Fincr[i].noalias() -= (dx - dx0) * g.transpose();//（7）更新变形矩阵
				shepardWeight += vwf;//这是累计变形程度的意思？
				smoothVelDifference[i].noalias() += vwf * dvint;

				if (damage[j]<1.0) numNeighsRefConfig[i]++;
			} // end loop over j

			// normalize average velocity field around an integration point
			if (shepardWeight > 0.0) {//归一化i原子周围的速度场，为什么？
				shepardWeightInv[i] = 1.0/shepardWeight;
				smoothVelDifference[i] *= shepardWeightInv[i];
			} else {
				shepardWeightInv[i] = 0;
				smoothVelDifference[i].setZero();
			}

			Fdot[i] *= K[i];//用形状矩阵修正变形梯度的导数(8)
			Fincr[i] *= K[i];//用形状矩阵修正变形梯度(7)
			Fincr[i].noalias() += eye;//为什么？

			if (JAUMANN) {
				R[i].setIdentity(); // for Jaumann stress rate, we do not need a subsequent rotation back into the reference configuration
			    //对于琼曼应力速率，我们不需要随后旋转回参考构型
			} else {
				//分解为拉伸和旋转
				status = PolDec(Fincr[i], R[i], U, false); // polar decomposition of the deformation gradient, F = R * U
				if (!status) {
				  cout << "Here is Fincr[" << tag[i] << "]:" << endl << Fincr[i] << endl;
				  cout << "Here is K[" << tag[i] << "]:" << endl << K[i] << endl;
					error->message(FLERR, "Polar decomposition of deformation gradient failed.\n");
					mol[i] = -mol[i];
				} else {
					Fincr[i] = R[i] * U;
				}
			}

			detF[i] = Fincr[i].determinant();
			FincrInv[i] = Fincr[i].inverse();

			// velocity gradient
			L = Fdot[i] * FincrInv[i];//求速度梯度

			// symmetric (D) and asymmetric (W) parts of L
			D[i] = 0.5 * (L + L.transpose());//求应变率 论文伪代码22行
			W[i] = 0.5 * (L - L.transpose()); //自旋张量 spin tensor:: need this for Jaumann rate

			// unrotated rate-of-deformation tensor d, see right side of Pronto2d, eqn.(2.1.7)
			// convention: unrotated frame is that one, where the true rotation of an integration point has been subtracted.
			// stress in the unrotated frame of reference is denoted sigma (stress seen by an observer doing rigid body rotations along with the material)
			// stress in the true frame of reference (a stationary observer) is denoted by T, "true stress"
			D[i] = (R[i].transpose() * D[i] * R[i]).eval();//消除刚体旋转分量的应变率

			// limit strain rate
			//double limit = 1.0e-3 * Lookup[SIGNAL_VELOCITY][itype] / radius[i];
			//D[i] = LimitEigenvalues(D[i], limit);

			/*
			 * make sure F stays within some limits
			 */

			if (numNeighsRefConfig[i] == 0) {
			  printf("deleting particle [%d] because nn = %d\n", tag[i], numNeighsRefConfig[i]);
			  dtCFL = MIN(dtCFL, dt); //Keep the same (small) time step when a particule breaks.		       
			  mol[i] = -mol[i];
			}

			if (mol[i] < 0) {//极性分解失败则重置原子参数
				D[i].setZero();
				Fdot[i].setZero();
				Fincr[i].setIdentity();
				smoothVelDifference[i].setZero();
				detF[i] = 1.0;
				K[i].setIdentity();

				vint[i][0] = 0.0;
				vint[i][1] = 0.0;
				vint[i][2] = 0.0;
			}
		} // end check setflag 
	} // end loop over i
}

/* ---------------------------------------------------------------------- */

void PairTlsph::compute(int eflag, int vflag) {

	if (atom->nmax > nmax) {
		nmax = atom->nmax;
		delete[] Fdot;
		Fdot = new Matrix3d[nmax]; // memory usage: 9 doubles
		delete[] Fincr;
		Fincr = new Matrix3d[nmax]; // memory usage: 9 doubles
		delete[] PK1;
		PK1 = new Matrix3d[nmax]; // memory usage: 9 doubles; total 5*9=45 doubles
		delete[] detF;
		detF = new double[nmax]; // memory usage: 1 double; total 46 doubles
		delete[] smoothVelDifference;
		smoothVelDifference = new Vector3d[nmax]; // memory usage: 3 doubles; total 49 doubles
		delete[] R;
		R = new Matrix3d[nmax]; // memory usage: 9 doubles; total 67 doubles
		delete[] FincrInv;
		FincrInv = new Matrix3d[nmax]; // memory usage: 9 doubles; total 85 doubles
		delete[] W;
		W = new Matrix3d[nmax]; // memory usage: 9 doubles; total 94 doubles
		delete[] D;
		D = new Matrix3d[nmax]; // memory usage: 9 doubles; total 103 doubles
		delete[] numNeighsRefConfig;
		numNeighsRefConfig = new int[nmax]; // memory usage: 1 int; total 108 doubles
		delete[] CauchyStress;
		CauchyStress = new Matrix3d[nmax]; // memory usage: 9 doubles; total 118 doubles
		delete[] hourglass_error;
		hourglass_error = new double[nmax];
		delete[] particle_dt;
		particle_dt = new double[nmax];
		delete[] vijSq_max;
		vijSq_max = new double[nmax];
		delete[] max_mu_ij;
		max_mu_ij = new double[nmax];
		delete[] damage_increment;
		damage_increment = new double[nmax];
		delete[] eff_plastic_strain_at_failure_init;
		eff_plastic_strain_at_failure_init = new double[nmax];
		delete[] rSqMin;
		rSqMin = new double[nmax];
		delete[] dvMax;
		dvMax = new double[nmax];
		delete[] flowstress_slope;
		flowstress_slope = new double[nmax];
		delete[] shepardWeightInv;
	        shepardWeightInv = new double[nmax];
	}

	if (first) { // return on first call, because reference connectivity lists still needs to be built. Also zero quantities which are otherwise undefined.
		first = false;

		for (int i = 0; i < atom->nlocal; i++) {
			Fincr[i].setZero();
			detF[i] = 0.0;
			smoothVelDifference[i].setZero();
			D[i].setZero();
			numNeighsRefConfig[i] = 0;
			CauchyStress[i].setZero();
			hourglass_error[i] = 0.0;
			particle_dt[i] = 0.0;
			vijSq_max[i] = 0.0;
			max_mu_ij[i] = 0.0;
			damage_increment[i] = 0.0;
			eff_plastic_strain_at_failure_init[i] = 0.0;
		}

		return;
	}

	fix_tlsph_reference_configuration->forward_comm_tl();

	/*
	 * calculate deformations and rate-of-deformations
	 */
	PairTlsph::PreCompute();

	/*
	 * calculate stresses from constitutive models
	 */
	PairTlsph::AssembleStress();

	/*
	 * QUANTITIES ABOVE HAVE ONLY BEEN CALCULATED FOR NLOCAL PARTICLES.
	 * NEED TO DO A FORWARD COMMUNICATION TO GHOST ATOMS NOW
	 */
	comm->forward_comm_pair(this);
	forward_comm_pair_tl();

	/*
	 * compute forces between particles
	 */
	updateFlag = 0;
	ComputeForces(eflag, vflag);

	UpdateDegradation();
        fix_tlsph_reference_configuration->need_forward_comm = true;
}

void PairTlsph::ComputeForces(int eflag, int vflag) {
	tagint *mol = atom->molecule;
	tagint *tag = atom->tag;
	double **x = atom->x;
	double **v = atom->vest;
	double **x0 = atom->x0;
	double **f = atom->f;
	double *vfrac = atom->vfrac;
	double *rho = atom->rho;
	double *de = atom->de;
	double *rmass = atom->rmass;
	double *radius = atom->radius;
	double *damage = atom->damage;
	double *plastic_strain = atom->eff_plastic_strain;
	int *type = atom->type;
	int nlocal = atom->nlocal;
	int i, j, jj, jnum, itype, idim;
	double r, hg_mag, vwf, wf, wfd, h, r0_, r0inv_, irad, voli, volj, r_plus_h_inv;
	double delVdotDelR, visc_magnitude, delta, deltaE, mu_ij, hg_err, rmassij, gamma_dot_dx;
	double softening_strain;
	char str[128];
	Vector3d fi, fj, dx0, dx, dx_normalized, dv, f_stress, f_hg, dxp_i, dxp_j, gamma, g, x0i, x0j, f_stressbc, fbc;
	Vector3d xi, xj, vi, vj, f_visc, sumForces, f_spring;
	int periodic = (domain->xperiodic || domain->yperiodic || domain->zperiodic);

	tagint **partner = ((FixSMD_TLSPH_ReferenceConfiguration *) modify->fix[ifix_tlsph])->partner;
	int *npartner = ((FixSMD_TLSPH_ReferenceConfiguration *) modify->fix[ifix_tlsph])->npartner;
	float **wfd_list = ((FixSMD_TLSPH_ReferenceConfiguration *) modify->fix[ifix_tlsph])->wfd_list;
	float **wf_list = ((FixSMD_TLSPH_ReferenceConfiguration *) modify->fix[ifix_tlsph])->wf_list;
	float **degradation_ij = ((FixSMD_TLSPH_ReferenceConfiguration *) modify->fix[ifix_tlsph])->degradation_ij;
	float **energy_per_bond = ((FixSMD_TLSPH_ReferenceConfiguration *) modify->fix[ifix_tlsph])->energy_per_bond;
	Vector3d **partnerdx = ((FixSMD_TLSPH_ReferenceConfiguration *) modify->fix[ifix_tlsph])->partnerdx;
	Vector3d **g_list = ((FixSMD_TLSPH_ReferenceConfiguration *) modify->fix[ifix_tlsph])->g_list;
	double **r0 = ((FixSMD_TLSPH_ReferenceConfiguration *) modify->fix[ifix_tlsph])->r0;
	Matrix3d *K = ((FixSMD_TLSPH_ReferenceConfiguration *) modify->fix[ifix_tlsph])->K;
	double **K_g_dot_dx0_normalized = ((FixSMD_TLSPH_ReferenceConfiguration *) modify->fix[ifix_tlsph])->K_g_dot_dx0_normalized;
	Matrix3d eye, sigmaBC_i;

	double deltat_1, deltat_2, deltat_3;
	eye.setIdentity();

	if (eflag || vflag)
		ev_setup(eflag, vflag);
	else
		evflag = vflag_fdotr = 0;

	/*
	 * iterate over pairs of particles i, j and assign forces using PK1 stress tensor
	 */

	//updateFlag = 0;
	hMin = 1.0e22;
	dtRelative = 1.0e22;

	for (i = 0; i < nlocal; i++) {
		
		if (mol[i] < 0) {
			continue; // Particle i is not a valid SPH particle (anymore). Skip all interactions with this particle.
		}

		itype = type[i];
		jnum = npartner[i];
		irad = radius[i];
		voli = vfrac[i];
		max_mu_ij[i] = 0.0;

		for (idim = 0; idim < 3; idim++) {
			x0i(idim) = x0[i][idim];
			xi(idim) = x[i][idim];
			vi(idim) = v[i][idim];
		}
		
		for (jj = 0; jj < jnum; jj++) {
			j = atom->map(partner[i][jj]);
			if (j < 0) { //			// check if lost a partner without first breaking bond
			  error->all(FLERR, "Bond broken not detected during PreCompute - 2!");
			  continue;
			}

			if ((abs(mol[i]) != abs(mol[j]))) {
				continue;
			}

			if (type[j] != itype) {
				sprintf(str, "particle pair is not of same type!");
				error->all(FLERR, str);
			}

			x0j(0) = x0[j][0];
			x0j(1) = x0[j][1];
			x0j(2) = x0[j][2];

			// check that distance between i and j (in the reference config) is less than cutoff
			//检查 i 和 j 之间的距离（在参考配置中）是否小于截止值
			dx0 = x0j - x0i;

			if (periodic)
				domain->minimum_image(dx0(0), dx0(1), dx0(2));

			h = irad + radius[j];//大于等于二倍的irad才能相互作用
			hMin = MIN(hMin, h);//更新最小值
			volj = vfrac[j];

			// distance vectors in current and reference configuration, velocity difference
			xj(0) = x[j][0];
			xj(1) = x[j][1];
			xj(2) = x[j][2];
			dx = xj - xi;

			vj(0) = v[j][0];
			vj(1) = v[j][1];
			vj(2) = v[j][2];
			dv = vj - vi;

			if (failureModel[itype].integration_point_wise == true) {
			  if (damage[j] > 0.0) {//这是在干嘛
			    dv *= (1-damage[j]);
			    dx = partnerdx[i][jj];
			  }
			}

			r = dx.norm(); // current distance

			vwf = volj * wf_list[i][jj];
			wfd = wfd_list[i][jj];
			g = g_list[i][jj]; // uncorrected kernel gradient 未校正的核梯度

			/*
			 * force contribution -- note that the kernel gradient correction has been absorbed into PK1
			 */
			if (damage[i] < 1.0 && damage[j] < 1.0) {
			  if (damage[i] > 0.0 || damage[j] > 0.0) {
			    f_stress = -(voli * volj) * (PK1[j]*(1-damage[i]) + PK1[i]*(1-damage[j])) * g;
			  } else {
			    f_stress = -(voli * volj) * (PK1[j] + PK1[i]) * g;
			  }
			} else {
			  f_stress.setZero();
			}

			// if (tag[i] == 122951) {
			//   printf("Step %d FORCE,  %d-%d: f_stress = [%.10e %.10e %.10e] damage_i=%.10e damage_j=%.10e\n",update->ntimestep, tag[i], tag[j], f_stress(0), f_stress(1), f_stress(2), damage[i], damage[j]);
			// }

			/*
			 * artificial viscosity人工粘度
			 */
                        //这段代码涉及粒子间的粘滞力计算
						/*计算相对速度在单位粒子距离向量上的投影
						delVdotDelR = dx.dot(dv) / (r + 0.1 * h);
						这里dx是粒子 i 和 j 之间的距离向量，dv是粒子 i 和 j 之间的相对速度，r是粒子 i 和 j 之间的距离，h是一个常数。这个步骤将相对速度投影到了单位粒子距离向量上，单位是m/s。

						计算粘滞率mu_ij
						mu_ij = h * delVdotDelR / (r + 0.1 * h);
						这一步根据相对速度的投影和距离计算了粘滞率，单位是m/s。

						计算粘滞力的大小
						visc_magnitude = (-Lookup[VISCOSITY_Q1][itype] * Lookup[SIGNAL_VELOCITY][itype] * mu_ij + Lookup[VISCOSITY_Q2][itype] * mu_ij * mu_ij) / Lookup[REFERENCE_DENSITY][itype];
						这一步根据粘滞率计算了粘滞力的大小，单位是N。

						计算粘滞力
						f_visc = rmass[i] * rmass[j] * visc_magnitude * wfd * dx / (r + 1.0e-2 * h);
						最后，根据粘滞力的大小、粒子质量和距离等参数计算了粘滞力的矢量值，单位是N。*/
                        delVdotDelR = dx.dot(dv) / (r + 0.1 * h); // project relative velocity onto unit particle distance vector [m/s]
																  //将相对速度投影到单位粒子距离矢量上 [m/s]
                        //LimitDoubleMagnitude(delVdotDelR, 0.01 * Lookup[SIGNAL_VELOCITY][itype]);
                        mu_ij = h * delVdotDelR / (r + 0.1 * h); // units: [m * m/s / m = m/s]
                        visc_magnitude = (-Lookup[VISCOSITY_Q1][itype] * Lookup[SIGNAL_VELOCITY][itype] * mu_ij
					  + Lookup[VISCOSITY_Q2][itype] * mu_ij * mu_ij) / Lookup[REFERENCE_DENSITY][itype]; // units: m^5/(s^2 kg))
			f_visc = rmass[i] * rmass[j] * visc_magnitude * wfd * dx / (r + 1.0e-2 * h); // units: kg^2 * m^5/(s^2 kg) * m^-4 = kg m / s^2 = N

			/*
			 * hourglass deviation of particles i and j粒子 i 和粒子 j 的沙漏偏差
			 */
			/*1.计算畸变误差：首先，从预定义的数组r0中获取r0_的值。
							接着，计算gamma，其中包括了Fincr、dx0、dx的组合计算。
							然后，计算畸变误差hg_err，它是gamma的范数除以r0_。
			2.更新每个粒子的小时玻璃误差：将vwf乘以hg_err，然后加到hourglass_error[i]上。
			3.更新最大粘滞率：计算最大粘滞率max_mu_ij[i]，
			其值为max_mu_ij[i]和(Lookup[VISCOSITY_Q1][itype] + hg_err * Lookup[HOURGLASS_CONTROL_AMPLITUDE][itype]) * fabs(mu_ij)中较大的一个。*/
			r0_ = r0[i][jj];
                        gamma = 0.5 * (Fincr[i] + Fincr[j]) * dx0 - dx;
                        hg_err = gamma.norm() / r0_;
                        hourglass_error[i] += vwf * hg_err;
			max_mu_ij[i] = MAX(max_mu_ij[i], (Lookup[VISCOSITY_Q1][itype] + hg_err * Lookup[HOURGLASS_CONTROL_AMPLITUDE][itype])*fabs(mu_ij));

                        /* SPH-like hourglass formulation */

                        if (MAX(plastic_strain[i], plastic_strain[j]) > 1.0e-3) {
			  /*
			   * viscous hourglass formulation for particles with plastic deformation
			   */
			  delta = gamma.dot(dx);
			  if (delVdotDelR * delta < 0.0) {
			    hg_err = MAX(hg_err, 0.05); // limit hg_err to avoid numerical instabilities
                                        hg_mag = -hg_err * Lookup[HOURGLASS_CONTROL_AMPLITUDE][itype] * Lookup[SIGNAL_VELOCITY][itype] * mu_ij
					  / Lookup[REFERENCE_DENSITY][itype]; // this has units of pressure
			  } else {
			    hg_mag = 0.0;
			  }
			  f_hg = rmass[i] * rmass[j] * hg_mag * wfd * dx / (r + 1.0e-2 * h);

                        } else {
			  f_hg.setZero();
			}
			  /*
			   * stiffness hourglass formulation for particle in the elastic regime
			   */

			  gamma_dot_dx = gamma.dot(dx); // project hourglass error vector onto pair distance vector
			  //LimitDoubleMagnitude(gamma_dot_dx, 0.1 * r); // limit projected vector to avoid numerical instabilities
			  delta = 0.5 * gamma_dot_dx / (r + 0.1 * h); // delta has dimensions of [m]
			  hg_mag = Lookup[HOURGLASS_CONTROL_AMPLITUDE][itype] * delta / (r0_*r0_ + 0.01 * h * h); // hg_mag has dimensions [m^(-1)]
			  if (MAX(plastic_strain[i], plastic_strain[j]) > 1.0e-3) {
			    hg_mag *= -voli * volj * wf * 0.5 * (flowstress_slope[i] + flowstress_slope[j]); // hg_mag has dimensions [J*m^(-1)] = [N]
			    
			  } else {
			    hg_mag *= -voli * volj * wf * Lookup[YOUNGS_MODULUS][itype]; // hg_mag has dimensions [J*m^(-1)] = [N]
			  }
			  f_hg += (hg_mag / (r + 0.01 * h)) * dx;

                        // scale hourglass force with damage
			f_hg *= (1.0 - damage[i]) * (1.0 - damage[j]);

			// sum stress, viscous, and hourglass forces
			sumForces = f_stress + f_visc + f_hg; // + f_spring;

			// if ((tag[i] == 762 && tag[j] == 570)||(tag[i] == 570 && tag[j] == 762)) {
			//   printf("Step %d - sumForces[%d][%d] = [%.10e %.10e %.10e] f_stress = [%.10e %.10e %.10e] f_visc = [%.10e %.10e %.10e] f_hg = [%.10e %.10e %.10e], damagei = %f, damagej = %f\n",update->ntimestep, tag[i], tag[j], sumForces(0), sumForces(1), sumForces(2), f_stress(0), f_stress(1), f_stress(2), f_visc(0), f_visc(1), f_visc(2), f_hg(0), f_hg(1), f_hg(2), damage[i], damage[j]);
			// }

			// energy rate -- project velocity onto force vector
			deltaE = sumForces.dot(dv);

			// apply forces to pair of particles
			f[i][0] += sumForces(0);
			f[i][1] += sumForces(1);
			f[i][2] += sumForces(2);
			de[i] += deltaE;

			// tally atomistic stress tensor
			if (evflag) {
				ev_tally_xyz(i, j, nlocal, 0, 0.0, 0.0, sumForces(0), sumForces(1), sumForces(2), dx(0), dx(1), dx(2));
			}

			// check if a particle has moved too much w.r.t another particle
			if (r > r0_) {
				if (update_method == UPDATE_CONSTANT_THRESHOLD) {
					if (r - r0_ > update_threshold) {
						updateFlag = 1;
					}
				} else if (update_method == UPDATE_PAIRWISE_RATIO) {
					if ((r - r0_) / h > update_threshold) {
						updateFlag = 1;
					}
				}
			}

			if (failureModel[itype].failure_energy_release_rate) {
			  energy_per_bond[i][jj] += update->dt * f_stress.dot(dv) / (voli * volj);
			}

		} // end loop over jj neighbors of i
		
		// if (tag[i] == 18268)
		//   printf("Step %d, COMPUTE_FORCES Particle %d: f = [%.10e %.10e %.10e]\n",update->ntimestep, tag[i], f[i][0], f[i][1], f[i][2]);
		if (output->next_dump_any == update->ntimestep) {
		  if (shepardWeightInv[i] != 0.0) {
		    hourglass_error[i] *= shepardWeightInv[i];
		  } else {
		    hourglass_error[i] = 0;
		  }
		}

		deltat_1 = sqrt(sqrt(rSqMin[i]) * rmass[i]/ sqrt( f[i][0]*f[i][0] + f[i][1]*f[i][1] + f[i][2]*f[i][2] ));
		// if (particle_dt[i] > deltat_1) {
		//   printf("particle_dt[%d] > deltat_1 with f = [%f %f %f]\n", tag[i], f[i][0], f[i][1], f[i][2]);
		// }
		particle_dt[i] = MIN(particle_dt[i], deltat_1); // Monaghan deltat_1 

		deltat_2 = sqrt(rmass[i]/ (Lookup[YOUNGS_MODULUS][itype] * 2 * irad)); // Make sure that oscillations due to elasticity are well captured.
		particle_dt[i] = MIN(particle_dt[i], deltat_2);

		deltat_3 = sqrt(rSqMin[i])/dvMax[i];
		// if (particle_dt[i] > deltat_1) {
		//   printf("particle_dt[%d] > deltat_1 with f = [%f %f %f]\n", tag[i], f[i][0], f[i][1], f[i][2]);
		// }
		particle_dt[i] = MIN(particle_dt[i], deltat_3);

		dtCFL = MIN(dtCFL, particle_dt[i]);

		if (particle_dt[i] < 1e-9) {
		  printf("dtCFL = %.10e, i = %d, particle_dt = %.10e, rSqMin = %.10e, rmass = %.10e, deltat_1 = %.10e, deltat_2 = %.10e, deltat_3 = %.10e, f = [%f %f %f]\n", dtCFL, tag[i], particle_dt[i], rSqMin[i], rmass[i], deltat_1, deltat_2, deltat_3, f[i][0], f[i][1], f[i][2]);
		}

	} // end loop over i

	//cout << "Here is sumf_stress.norm(): " << sumf_stress.norm() << endl;
	if (vflag_fdotr)
		virial_fdotr_compute();
}

/* ----------------------------------------------------------------------
 assemble unrotated stress tensor using deviatoric and pressure components.
 Convert to corotational Cauchy stress, then to PK1 stress and apply
 shape matrix correction
 ------------------------------------------------------------------------- */
void PairTlsph::AssembleStress() {
	tagint *mol = atom->molecule;
	double **v = atom->vest;
	double *eff_plastic_strain = atom->eff_plastic_strain;
	double *eff_plastic_strain_rate = atom->eff_plastic_strain_rate;
	double **tlsph_stress = atom->smd_stress;
	int *type = atom->type;
	double *radius = atom->radius;
	double *damage = atom->damage;
	double *rmass = atom->rmass;
	double *vfrac = atom->vfrac;
	double *rho = atom->rho;
	double *e = atom->e;
	double pInitial, d_iso, pFinal, p_rate, plastic_strain_increment;
	int i, itype, idim;
	int nlocal = atom->nlocal;
	double dt = update->dt;
	double M_eff, K_eff, mu_eff, p_wave_speed, mass_specific_energy, vol_specific_energy, yieldstress, f;
	Matrix3d sigma_rate, eye, sigmaInitial, sigmaFinal, T, T_damaged, Jaumann_rate, sigma_rate_check;
	Matrix3d d_dev, sigmaInitial_dev, sigmaFinal_dev, sigma_dev_rate, strain, deltaSigma;
	Vector3d vi;
	Matrix3d *K = ((FixSMD_TLSPH_ReferenceConfiguration *) modify->fix[ifix_tlsph])->K;

	eye.setIdentity();
	dtCFL = 1.0e22;
	pFinal = 0.0;

	for (i = 0; i < nlocal; i++) {
		particle_dt[i] = 0.0;

		itype = type[i];
		if (setflag[itype][itype] == 1) {
			if (mol[i] > 0) { // only do the following if particle has not failed -- mol < 0 means particle has failed

				/*
				 * initial stress state: given by the unrotateted Cauchy stress.
				 * Assemble Eigen 3d matrix from stored stress state
				 */
				sigmaInitial(0, 0) = tlsph_stress[i][0];
				sigmaInitial(0, 1) = tlsph_stress[i][1];
				sigmaInitial(0, 2) = tlsph_stress[i][2];
				sigmaInitial(1, 1) = tlsph_stress[i][3];
				sigmaInitial(1, 2) = tlsph_stress[i][4];
				sigmaInitial(2, 2) = tlsph_stress[i][5];
				sigmaInitial(1, 0) = sigmaInitial(0, 1);
				sigmaInitial(2, 0) = sigmaInitial(0, 2);
				sigmaInitial(2, 1) = sigmaInitial(1, 2);

				//cout << "this is sigma initial" << endl << sigmaInitial << endl;

				pInitial = sigmaInitial.trace() / 3.0; // isotropic part of initial stress
				sigmaInitial_dev = Deviator(sigmaInitial);
				d_iso = D[i].trace(); // volumetric part of stretch rate拉伸率的体积部分 论文伪代码中24行
				d_dev = Deviator(D[i]); // deviatoric part of stretch rate拉伸率的偏差部分
				Matrix3d FtF = Fincr[i].transpose() * Fincr[i];
				strain = 0.5 * (FtF - eye);
				mass_specific_energy = e[i] / rmass[i]; // energy per unit mass单位质量的能量
				rho[i] = rmass[i] / (detF[i] * vfrac[i]);
				//rho[i] = rmass[i] / (sqrt(FtF.determinant()) * vfrac[i]);
				vol_specific_energy = mass_specific_energy * rho[i]; // energy per current volume

				/*
				 * pressure: compute pressure rate p_rate and final pressure pFinal
				 */

				ComputePressure(i, rho[i], mass_specific_energy, vol_specific_energy, pInitial, d_iso, pFinal, p_rate);

				/*
				 * material strength
				 */

				//cout << "this is the strain deviator rate" << endl << d_dev << endl;
				ComputeStressDeviator(i, mass_specific_energy, sigmaInitial_dev, d_dev, sigmaFinal_dev, sigma_dev_rate, plastic_strain_increment, pInitial, pFinal);
				//cout << "this is the stress deviator rate" << endl << sigma_dev_rate << endl;

				// keep a rolling average of the plastic strain rate over the last 100 or so timesteps
				eff_plastic_strain[i] += plastic_strain_increment;
				if (eff_plastic_strain[i] < 0.0) {
				  printf("eff_plastic_strain[%d] = %f, plastic_strain_increment = %f\n", atom->tag[i], eff_plastic_strain[i], plastic_strain_increment);
				}

				// compute a characteristic time over which to average the plastic strain计算平均塑性应变的特征时间
				double tav = 1000 * radius[i] / (Lookup[SIGNAL_VELOCITY][itype]);
				eff_plastic_strain_rate[i] -= eff_plastic_strain_rate[i] * dt / tav;
				eff_plastic_strain_rate[i] += plastic_strain_increment / tav;
				eff_plastic_strain_rate[i] = MAX(0.0, eff_plastic_strain_rate[i]);

				/*
				 *  assemble total stress from pressure and deviatoric stress
				 */
				sigmaFinal = pFinal * eye + sigmaFinal_dev; // this is the stress that is kept

				if (JAUMANN) {
					/*
					 * sigma is already the co-rotated Cauchy stress.
					 * The stress rate, however, needs to be made objective.
					 */

					if (dt > 1.0e-16) {
						sigma_rate = (1.0 / dt) * (sigmaFinal - sigmaInitial);
					} else {
						sigma_rate.setZero();
					}

					//这段代码涉及到材料中的Jaumann应力更新算法
					Jaumann_rate = sigma_rate + W[i] * sigmaInitial + sigmaInitial * W[i].transpose();//J
					sigmaFinal = sigmaInitial + dt * Jaumann_rate;
					T = sigmaFinal;
				} else {
					/*
					 * sigma is the unrotated stress.
					 * need to do forward rotation of the unrotated stress sigma to the current configuration
					 */
					T = R[i] * sigmaFinal * R[i].transpose();
				}

				/*
				 * store unrotated stress in atom vector在原子矢量中存储未旋转的应力
				 * symmetry is exploited利用了对称性
				 */
				tlsph_stress[i][0] = sigmaFinal(0, 0);
				tlsph_stress[i][1] = sigmaFinal(0, 1);
				tlsph_stress[i][2] = sigmaFinal(0, 2);
				tlsph_stress[i][3] = sigmaFinal(1, 1);
				tlsph_stress[i][4] = sigmaFinal(1, 2);
				tlsph_stress[i][5] = sigmaFinal(2, 2);

				/*
				 *  Damage due to failure criteria.故障标准造成的损坏。
				 */

				if (failureModel[itype].integration_point_wise) {
				  ComputeDamage(i, strain, T, T_damaged, plastic_strain_increment);
				  T = T_damaged;
				}

				// store rotated, "true" Cauchy stress存储旋转的、"真正的 "柯西应力  伪代码27行
				CauchyStress[i] = T;

				/*
				 * We have the corotational Cauchy stress.我们有柯西应力。
				 * Convert to PK1. Note that reference configuration used for computing the forces is linked via
				 * the incremental deformation gradient, not the full deformation gradient.
				 * 转换为 PK1。请注意，用于计算力的参考构型是通过增量变形梯度而不是完全变形梯度连接的。
				 */
				PK1[i] = detF[i] * T * FincrInv[i].transpose();

				/*
				 * pre-multiply stress tensor with shape matrix to save computation in force loop
				 将应力张量与形状矩阵预乘，以节省力循环中的计算量
				 */
				PK1[i] = PK1[i] * K[i].transpose();

				/*
				 * compute stable time step according to Pronto 2d
				 */

				deltaSigma = sigmaFinal - sigmaInitial;
				p_rate = deltaSigma.trace() / (3.0 * dt + 1.0e-16);
				sigma_dev_rate = Deviator(deltaSigma) / (dt + 1.0e-16);
				if (failureModel[itype].failure_gtn) {
				  double fcrQ1 = Lookup[GTN_Q1][itype] * Lookup[GTN_fcr][itype];
				  if (damage[i] <= fcrQ1) f = damage[i] / Lookup[GTN_Q1][itype];
				  else f = Lookup[GTN_fcr][itype] + (damage[i] - fcrQ1) / (1.0 - fcrQ1) * (Lookup[GTN_fF][itype] - Lookup[GTN_fcr][itype]);
				} else {
				  f = damage[i];
				}
				  effective_longitudinal_modulus(itype, dt, d_iso, p_rate, d_dev, sigma_dev_rate, f, K_eff, mu_eff, M_eff);

				// if ((damage[i] > 0.0) && (pFinal > 0.0)) {
				//   // If the particle is under tension, voids are open:
				//   p_wave_speed = sqrt(M_eff * (1.0 - damage[i]) / (rho[i]*(1.0 - f) + Lookup[REFERENCE_DENSITY][itype] * f));
				// } else {
				  p_wave_speed = sqrt(M_eff / MIN(rho[i], rmass[i] / vfrac[i]));
				// }

				if (mol[i] < 0) {
					error->one(FLERR, "this should not happen");
				}
				
				for (idim = 0; idim < 3; idim++) {
				  vi(idim) = v[i][idim];
				}
				//double max_damage = max(0.0001, 1 - f);
				particle_dt[i] = sqrt(rSqMin[i]) / (p_wave_speed + sqrt(vijSq_max[i])); // Monaghan deltat_2
				if (particle_dt[i] < 1e-9) {
				  printf("dtCFL = %.10e, i = %d, particle_dt = %.10e, rSqMin = %.10e, p_wave_speed = %.10e, max_mu_ij = %.10e, vijSq_max = %.10e\n", dtCFL, atom->tag[i], particle_dt[i], rSqMin[i], p_wave_speed, max_mu_ij[i], vijSq_max[i]);
		}
				dtCFL = MIN(dtCFL, particle_dt[i]);

				//Determine the derivative of the flowstress with respect to the strain:确定流应力相对于应变的导数：
				if (eff_plastic_strain[i] > 0.0) {
				  flowstress_slope[i] = MIN(Lookup[YOUNGS_MODULUS][itype],flowstress.evaluate_derivative(eff_plastic_strain[i]));
				} else {
				  flowstress_slope[i] = Lookup[YOUNGS_MODULUS][itype];
				}

				if (damage[i] > 0.0) {
				  flowstress_slope[i] *= (1.0 - damage[i]);
				}

			} else { // end if mol > 0
				PK1[i].setZero();
				K[i].setIdentity();
				CauchyStress[i].setZero();
				sigma_rate.setZero();
				tlsph_stress[i][0] = 0.0;
				tlsph_stress[i][1] = 0.0;
				tlsph_stress[i][2] = 0.0;
				tlsph_stress[i][3] = 0.0;
				tlsph_stress[i][4] = 0.0;
				tlsph_stress[i][5] = 0.0;
			} // end  if mol > 0
		} // end setflag
	} // end for
}

/* ----------------------------------------------------------------------
 allocate all arrays
 ------------------------------------------------------------------------- */

void PairTlsph::allocate() {
	allocated = 1;
	int n = atom->ntypes;

	memory->create(setflag, n + 1, n + 1, "pair:setflag");
	for (int i = 1; i <= n; i++)
		for (int j = i; j <= n; j++)
			setflag[i][j] = 0;

	memory->create(strengthModel, n + 1, "pair:strengthmodel");
	memory->create(eos, n + 1, "pair:eosmodel");
	failureModel = new failure_types[n + 1];
	memory->create(Lookup, MAX_KEY_VALUE, n + 1, "pair:LookupTable");

	memory->create(cutsq, n + 1, n + 1, "pair:cutsq"); // always needs to be allocated, even with granular neighborlist

	onerad_dynamic = new double[n + 1];
	onerad_frozen = new double[n + 1];
	maxrad_dynamic = new double[n + 1];
	maxrad_frozen = new double[n + 1];

	memory->create(nrecv,comm->nprocs,"pair_tlsph:nrecv");
	memory->create(buf_send,maxsend + BUFEXTRA,"pair_tlsph:buf_send");
	memory->create(buf_recv,maxrecv + BUFEXTRA,"pair_tlsph:buf_recv");

}

/* ----------------------------------------------------------------------
 global settings
 ------------------------------------------------------------------------- */

void PairTlsph::settings(int narg, char **arg) {

	if (comm->me == 0) {
		printf(
				"\n>>========>>========>>========>>========>>========>>========>>========>>========>>========>>========>>========>>========\n");
		printf("TLSPH settings\n");
	}

	/*
	 * default value for update_threshold for updates of reference configuration:
	 * The maximum relative displacement which is tracked by the construction of LAMMPS' neighborlists
	 * is the folowing.
	 */

	cut_comm = MAX(neighbor->cutneighmax, comm->cutghostuser); // cutoff radius within which ghost atoms are communicated.
	update_threshold = cut_comm;
	update_method = UPDATE_NONE;

	int iarg = 0;

	while (true) {

		if (iarg >= narg) {
			break;
		}

		if (strcmp(arg[iarg], "*UPDATE_CONSTANT") == 0) {
			iarg++;
			if (iarg == narg) {
				error->all(FLERR, "expected number following *UPDATE_CONSTANT keyword");
			}

			update_method = UPDATE_CONSTANT_THRESHOLD;
			update_threshold = force->numeric(FLERR, arg[iarg]);

		} else if (strcmp(arg[iarg], "*UPDATE_PAIRWISE") == 0) {
			iarg++;
			if (iarg == narg) {
				error->all(FLERR, "expected number following *UPDATE_PAIRWISE keyword");
			}

			update_method = UPDATE_PAIRWISE_RATIO;
			update_threshold = force->numeric(FLERR, arg[iarg]);

		} else {
			char msg[128];
			sprintf(msg, "Illegal keyword for smd/integrate_tlsph: %s\n", arg[iarg]);
			error->all(FLERR, msg);
		}

		iarg++;
	}

	if ((update_threshold > cut_comm) && (update_method == UPDATE_CONSTANT_THRESHOLD)) {
		if (comm->me == 0) {
			printf("\n                ***** WARNING ***\n");
			printf("requested reference configuration update threshold is %g length units\n", update_threshold);
			printf("This value exceeds the maximum value %g beyond which TLSPH displacements can be tracked at current settings.\n",
					cut_comm);
			printf("Expect loss of neighbors!\n");
		}
	}

	if (comm->me == 0) {

		if (update_method == UPDATE_CONSTANT_THRESHOLD) {
			printf("... will update reference configuration if magnitude of relative displacement exceeds %g length units\n",
					update_threshold);
		} else if (update_method == UPDATE_PAIRWISE_RATIO) {
			printf("... will update reference configuration if ratio pairwise distance / smoothing length  exceeds %g\n",
					update_threshold);
		} else if (update_method == UPDATE_NONE) {
			printf("... will never update reference configuration");
		}
		printf(
				">>========>>========>>========>>========>>========>>========>>========>>========>>========>>========>>========>>========\n");

	}

}

/* ----------------------------------------------------------------------
 set coeffs for one or more type pairs
 ------------------------------------------------------------------------- */

void PairTlsph::coeff(int narg, char **arg) {
	int ioffset, iarg, iNextKwd, itype;
	char str[128];
	std::string s, t;

	if (narg < 3) {
		sprintf(str, "number of arguments for pair tlsph is too small!");
		error->all(FLERR, str);
	}
	if (!allocated)
		allocate();

	/*
	 * check that TLSPH parameters are given only in i,i form
	 */
	if (force->inumeric(FLERR, arg[0]) != force->inumeric(FLERR, arg[1])) {
		sprintf(str, "TLSPH coefficients can only be specified between particles of same type!");
		error->all(FLERR, str);
	}
	itype = force->inumeric(FLERR, arg[0]);

	// Initialize all Lookup variables to 0:
	for (int ikey = 0; ikey<MAX_KEY_VALUE; ikey++)
	  Lookup[ikey][itype] = 0;

// set all eos, strength and failure models to inactive by default
	eos[itype] = EOS_NONE;
	strengthModel[itype] = STRENGTH_NONE;

	// if (comm->me == 0) {
	// 	printf(
	// 			"\n>>========>>========>>========>>========>>========>>========>>========>>========>>========>>========>>========>>========\n");
	// 	printf("SMD / TLSPH PROPERTIES OF PARTICLE TYPE %d:\n", itype);
	// }

	/*
	 * read parameters which are common -- regardless of material / eos model
	 */

	ioffset = 2;
	if (strcmp(arg[ioffset], "*COMMON") != 0) {
		sprintf(str, "common keyword missing!");
		error->all(FLERR, str);
	}

	t = string("*");
	iNextKwd = -1;
	for (iarg = ioffset + 1; iarg < narg; iarg++) {
		s = string(arg[iarg]);
		if (s.compare(0, t.length(), t) == 0) {
			iNextKwd = iarg;
			break;
		}
	}

//printf("keyword following *COMMON is %s\n", arg[iNextKwd]);

	if (iNextKwd < 0) {
		sprintf(str, "no *KEYWORD terminates *COMMON");
		error->all(FLERR, str);
	}

	if (iNextKwd - ioffset != 7 + 1) {
		sprintf(str, "expected 7 arguments following *COMMON but got %d\n", iNextKwd - ioffset - 1);
		error->all(FLERR, str);
	}

	Lookup[REFERENCE_DENSITY][itype] = force->numeric(FLERR, arg[ioffset + 1]);
	Lookup[YOUNGS_MODULUS][itype] = force->numeric(FLERR, arg[ioffset + 2]);
	Lookup[POISSON_RATIO][itype] = force->numeric(FLERR, arg[ioffset + 3]);
	Lookup[VISCOSITY_Q1][itype] = force->numeric(FLERR, arg[ioffset + 4]);
	Lookup[VISCOSITY_Q2][itype] = force->numeric(FLERR, arg[ioffset + 5]);
	Lookup[HOURGLASS_CONTROL_AMPLITUDE][itype] = force->numeric(FLERR, arg[ioffset + 6]);
	Lookup[HEAT_CAPACITY][itype] = force->numeric(FLERR, arg[ioffset + 7]);

	Lookup[LAME_LAMBDA][itype] = Lookup[YOUNGS_MODULUS][itype] * Lookup[POISSON_RATIO][itype]
	  / ((1.0 + Lookup[POISSON_RATIO][itype]) * (1.0 - 2.0 * Lookup[POISSON_RATIO][itype]));
	Lookup[SHEAR_MODULUS][itype] = Lookup[YOUNGS_MODULUS][itype] / (2.0 * (1.0 + Lookup[POISSON_RATIO][itype]));
	Lookup[M_MODULUS][itype] = Lookup[LAME_LAMBDA][itype] + 2.0 * Lookup[SHEAR_MODULUS][itype];
	Lookup[SIGNAL_VELOCITY][itype] = sqrt(
			(Lookup[LAME_LAMBDA][itype] + 2.0 * Lookup[SHEAR_MODULUS][itype]) / Lookup[REFERENCE_DENSITY][itype]);
	Lookup[BULK_MODULUS][itype] = Lookup[LAME_LAMBDA][itype] + 2.0 * Lookup[SHEAR_MODULUS][itype] / 3.0;
	Lookup[VISCOSITY_Q1_times_SIGNAL_VELOCITY][itype] = Lookup[VISCOSITY_Q1][itype] * Lookup[SIGNAL_VELOCITY][itype];
	Lookup[HOURGLASS_CONTROL_AMPLITUDE_times_YOUNGS_MODULUS][itype] = Lookup[HOURGLASS_CONTROL_AMPLITUDE][itype] * Lookup[YOUNGS_MODULUS][itype];
	Lookup[LONGITUDINAL_MODULUS][itype] = Lookup[LAME_LAMBDA][itype] + 2.0 * Lookup[SHEAR_MODULUS][itype];
	Lookup[SIGNAL_VELOCITY_OVER_REFERENCE_DENSITY][itype] = Lookup[SIGNAL_VELOCITY][itype] / Lookup[REFERENCE_DENSITY][itype];


	// if (comm->me == 0) {
	// 	printf("\n material unspecific properties for SMD/TLSPH definition of particle type %d:\n", itype);
	// 	printf("%60s : %g\n", "reference density", Lookup[REFERENCE_DENSITY][itype]);
	// 	printf("%60s : %g\n", "Young's modulus", Lookup[YOUNGS_MODULUS][itype]);
	// 	printf("%60s : %g\n", "Poisson ratio", Lookup[POISSON_RATIO][itype]);
	// 	printf("%60s : %g\n", "linear viscosity coefficient", Lookup[VISCOSITY_Q1][itype]);
	// 	printf("%60s : %g\n", "quadratic viscosity coefficient", Lookup[VISCOSITY_Q2][itype]);
	// 	printf("%60s : %g\n", "hourglass control coefficient", Lookup[HOURGLASS_CONTROL_AMPLITUDE][itype]);
	// 	printf("%60s : %g\n", "heat capacity [energy / (mass * temperature)]", Lookup[HEAT_CAPACITY][itype]);
	// 	printf("%60s : %g\n", "Lame constant lambda", Lookup[LAME_LAMBDA][itype]);
	// 	printf("%60s : %g\n", "shear modulus", Lookup[SHEAR_MODULUS][itype]);
	// 	printf("%60s : %g\n", "bulk modulus", Lookup[BULK_MODULUS][itype]);
	// 	printf("%60s : %g\n", "signal velocity", Lookup[SIGNAL_VELOCITY][itype]);
	// }

	/*
	 * read following material cards
	 */

//printf("next kwd is %s\n", arg[iNextKwd]);
	eos[itype] = EOS_NONE;
	strengthModel[itype] = STRENGTH_NONE;

	while (true) {
		if (strcmp(arg[iNextKwd], "*END") == 0) {
			// if (comm->me == 0) {
			// 	printf("found *END keyword");
			// 	printf(
			// 			"\n>>========>>========>>========>>========>>========>>========>>========>>========>>========>>========>>========>>========\n\n");
			// }
			break;
		}

		/*
		 * Linear Elasticity model based on deformation gradient
		 */
		ioffset = iNextKwd;
		if (strcmp(arg[ioffset], "*LINEAR_DEFGRAD") == 0) {
			strengthModel[itype] = LINEAR_DEFGRAD;

			// if (comm->me == 0) {
			// 	printf("reading *LINEAR_DEFGRAD\n");
			// }

			t = string("*");
			iNextKwd = -1;
			for (iarg = ioffset + 1; iarg < narg; iarg++) {
				s = string(arg[iarg]);
				if (s.compare(0, t.length(), t) == 0) {
					iNextKwd = iarg;
					break;
				}
			}

			if (iNextKwd < 0) {
				sprintf(str, "no *KEYWORD terminates *LINEAR_DEFGRAD");
				error->all(FLERR, str);
			}

			if (iNextKwd - ioffset != 1) {
				sprintf(str, "expected 0 arguments following *LINEAR_DEFGRAD but got %d\n", iNextKwd - ioffset - 1);
				error->all(FLERR, str);
			}

			// if (comm->me == 0) {
			// 	printf("\n%60s\n", "Linear Elasticity model based on deformation gradient");
			// }
		} else if (strcmp(arg[ioffset], "*STRENGTH_LINEAR") == 0) {

			/*
			 * Linear Elasticity strength only model based on strain rate
			 */

			strengthModel[itype] = STRENGTH_LINEAR;
			// if (comm->me == 0) {
			// 	printf("reading *STRENGTH_LINEAR\n");
			// }

			t = string("*");
			iNextKwd = -1;
			for (iarg = ioffset + 1; iarg < narg; iarg++) {
				s = string(arg[iarg]);
				if (s.compare(0, t.length(), t) == 0) {
					iNextKwd = iarg;
					break;
				}
			}

			if (iNextKwd < 0) {
				sprintf(str, "no *KEYWORD terminates *STRENGTH_LINEAR");
				error->all(FLERR, str);
			}

			if (iNextKwd - ioffset != 1) {
				sprintf(str, "expected 0 arguments following *STRENGTH_LINEAR but got %d\n", iNextKwd - ioffset - 1);
				error->all(FLERR, str);
			}

			// if (comm->me == 0) {
			// 	printf("%60s\n", "Linear Elasticity strength based on strain rate");
			// }
		} // end Linear Elasticity strength only model based on strain rate

		else if (strcmp(arg[ioffset], "*STRENGTH_LINEAR_PLASTIC") == 0) {

			/*
			 * Linear Elastic / perfectly plastic strength only model based on strain rate
			 */

			strengthModel[itype] = STRENGTH_LINEAR_PLASTIC;
			// if (comm->me == 0) {
			// 	printf("reading *STRENGTH_LINEAR_PLASTIC\n");
			// }

			t = string("*");
			iNextKwd = -1;
			for (iarg = ioffset + 1; iarg < narg; iarg++) {
				s = string(arg[iarg]);
				if (s.compare(0, t.length(), t) == 0) {
					iNextKwd = iarg;
					break;
				}
			}

			if (iNextKwd < 0) {
				sprintf(str, "no *KEYWORD terminates *STRENGTH_LINEAR_PLASTIC");
				error->all(FLERR, str);
			}

			if (iNextKwd - ioffset != 2 + 1) {
				sprintf(str, "expected 2 arguments following *STRENGTH_LINEAR_PLASTIC but got %d\n", iNextKwd - ioffset - 1);
				error->all(FLERR, str);
			}

			Lookup[YIELD_STRESS][itype] = force->numeric(FLERR, arg[ioffset + 1]);
			Lookup[HARDENING_PARAMETER][itype] = force->numeric(FLERR, arg[ioffset + 2]);

			flowstress.linear_plastic(Lookup[YIELD_STRESS][itype], Lookup[HARDENING_PARAMETER][itype]);
			// if (comm->me == 0) {
			// 	printf("%60s\n", "Linear elastic / perfectly plastic strength based on strain rate");
			// 	printf("%60s : %g\n", "Young's modulus", Lookup[YOUNGS_MODULUS][itype]);
			// 	printf("%60s : %g\n", "Poisson ratio", Lookup[POISSON_RATIO][itype]);
			// 	printf("%60s : %g\n", "shear modulus", Lookup[SHEAR_MODULUS][itype]);
			// 	printf("%60s : %g\n", "constant yield stress", Lookup[YIELD_STRESS][itype]);
			// 	printf("%60s : %g\n", "constant hardening parameter", Lookup[HARDENING_PARAMETER][itype]);
			// }
		} // end Linear Elastic / perfectly plastic strength only model based on strain rate

		else if (strcmp(arg[ioffset], "*JOHNSON_COOK") == 0) {

			/*
			 * JOHNSON - COOK
			 */

			strengthModel[itype] = STRENGTH_JOHNSON_COOK;
			// if (comm->me == 0) {
			// 	printf("reading *JOHNSON_COOK\n");
			// }

			t = string("*");
			iNextKwd = -1;
			for (iarg = ioffset + 1; iarg < narg; iarg++) {
				s = string(arg[iarg]);
				if (s.compare(0, t.length(), t) == 0) {
					iNextKwd = iarg;
					break;
				}
			}

			if (iNextKwd < 0) {
				sprintf(str, "no *KEYWORD terminates *JOHNSON_COOK");
				error->all(FLERR, str);
			}

			if (iNextKwd - ioffset != 8 + 1) {
				sprintf(str, "expected 8 arguments following *JOHNSON_COOK but got %d\n", iNextKwd - ioffset - 1);
				error->all(FLERR, str);
			}

			Lookup[JC_A][itype] = force->numeric(FLERR, arg[ioffset + 1]);
			Lookup[JC_B][itype] = force->numeric(FLERR, arg[ioffset + 2]);
			Lookup[JC_a][itype] = force->numeric(FLERR, arg[ioffset + 3]);
			Lookup[JC_C][itype] = force->numeric(FLERR, arg[ioffset + 4]);
			Lookup[JC_epdot0][itype] = force->numeric(FLERR, arg[ioffset + 5]);
			Lookup[JC_T0][itype] = force->numeric(FLERR, arg[ioffset + 6]);
			Lookup[JC_Tmelt][itype] = force->numeric(FLERR, arg[ioffset + 7]);
			Lookup[JC_M][itype] = force->numeric(FLERR, arg[ioffset + 8]);

			flowstress.JC(Lookup[JC_A][itype], Lookup[JC_B][itype], Lookup[JC_a][itype],
				      Lookup[JC_C][itype], Lookup[JC_epdot0][itype], Lookup[JC_T0][itype],
				      Lookup[JC_Tmelt][itype], Lookup[JC_M][itype]);

			// if (comm->me == 0) {
			// 	printf("%60s\n", "Johnson Cook material strength model");
			// 	printf("%60s : %g\n", "A: initial yield stress", Lookup[JC_A][itype]);
			// 	printf("%60s : %g\n", "B : proportionality factor for plastic strain dependency", Lookup[JC_B][itype]);
			// 	printf("%60s : %g\n", "a : exponent for plastic strain dependency", Lookup[JC_a][itype]);
			// 	printf("%60s : %g\n", "C : proportionality factor for logarithmic plastic strain rate dependency",
			// 			Lookup[JC_C][itype]);
			// 	printf("%60s : %g\n", "epdot0 : dimensionality factor for plastic strain rate dependency",
			// 			Lookup[JC_epdot0][itype]);
			// 	printf("%60s : %g\n", "T0 : reference (room) temperature", Lookup[JC_T0][itype]);
			// 	printf("%60s : %g\n", "Tmelt : melting temperature", Lookup[JC_Tmelt][itype]);
			// 	printf("%60s : %g\n", "M : exponent for temperature dependency", Lookup[JC_M][itype]);
			// }

		} else if (strcmp(arg[ioffset], "*LUDWICK_HOLLOMON") == 0) {

			/*
			 * LUDWICK - HOLLOMON
			 */

			strengthModel[itype] = STRENGTH_LUDWICK_HOLLOMON;
			// if (comm->me == 0) {
			// 	printf("reading *LUDWICK_HOLLOMON\n");
			// }

			t = string("*");
			iNextKwd = -1;
			for (iarg = ioffset + 1; iarg < narg; iarg++) {
				s = string(arg[iarg]);
				if (s.compare(0, t.length(), t) == 0) {
					iNextKwd = iarg;
					break;
				}
			}

			if (iNextKwd < 0) {
				sprintf(str, "no *KEYWORD terminates *LUDWICK_HOLLOMON");
				error->all(FLERR, str);
			}

			if (iNextKwd - ioffset != 3 + 1) {
				sprintf(str, "expected 3 arguments following *LUDWICK_HOLLOMON but got %d\n", iNextKwd - ioffset - 1);
				error->all(FLERR, str);
			}

			Lookup[LH_A][itype] = force->numeric(FLERR, arg[ioffset + 1]);
			Lookup[LH_B][itype] = force->numeric(FLERR, arg[ioffset + 2]);
			Lookup[LH_n][itype] = force->numeric(FLERR, arg[ioffset + 3]);

			flowstress.LH(Lookup[LH_A][itype], Lookup[LH_B][itype], Lookup[LH_n][itype]);

			// if (comm->me == 0) {
			// 	printf("%60s : %s\n", "Ludwick-Hollomon material strength model","A + B * pow(ep, n)");
			// 	printf("%60s : %g\n", "A: initial yield stress", Lookup[LH_A][itype]);
			// 	printf("%60s : %g\n", "B : proportionality factor for plastic strain dependency", Lookup[LH_B][itype]);
			// 	printf("%60s : %g\n", "n : exponent for plastic strain dependency", Lookup[LH_n][itype]);
			// }

		} else if (strcmp(arg[ioffset], "*SWIFT") == 0) {

			/*
			 * SWIFT
			 */

			strengthModel[itype] = STRENGTH_SWIFT;
			// if (comm->me == 0) {
			// 	printf("reading *SWIFT\n");
			// }

			t = string("*");
			iNextKwd = -1;
			for (iarg = ioffset + 1; iarg < narg; iarg++) {
				s = string(arg[iarg]);
				if (s.compare(0, t.length(), t) == 0) {
					iNextKwd = iarg;
					break;
				}
			}

			if (iNextKwd < 0) {
				sprintf(str, "no *KEYWORD terminates *SWIFT");
				error->all(FLERR, str);
			}

			if (iNextKwd - ioffset != 4 + 1) {
				sprintf(str, "expected 4 arguments following *SWIFT but got %d\n", iNextKwd - ioffset - 1);
				error->all(FLERR, str);
			}

			Lookup[SWIFT_A][itype] = force->numeric(FLERR, arg[ioffset + 1]);
			Lookup[SWIFT_B][itype] = force->numeric(FLERR, arg[ioffset + 2]);
			Lookup[SWIFT_n][itype] = force->numeric(FLERR, arg[ioffset + 3]);
			Lookup[SWIFT_eps0][itype] = force->numeric(FLERR, arg[ioffset + 4]);

			flowstress.SWIFT(Lookup[SWIFT_A][itype], Lookup[SWIFT_B][itype], Lookup[SWIFT_n][itype], Lookup[SWIFT_eps0][itype]);

			if (Lookup[SWIFT_eps0][itype] < 0.0){
			  sprintf(str, "the 4th argument following *SWIFT should be positive\n");
			  error->all(FLERR, str);
			}

			// if (comm->me == 0) {
			// 	printf("%60s : %s\n", "Swift strength model", "A + B * pow(ep - eps0, n)");
			// 	printf("%60s : %g\n", "A: initial yield stress", Lookup[SWIFT_A][itype]);
			// 	printf("%60s : %g\n", "B : proportionality factor for plastic strain dependency", Lookup[SWIFT_B][itype]);
			// 	printf("%60s : %g\n", "n : exponent for plastic strain dependency", Lookup[SWIFT_n][itype]);
			// 	printf("%60s : %g\n", "eps0 : initial plastic strain", Lookup[SWIFT_eps0][itype]);
			// }

		}  else if (strcmp(arg[ioffset], "*VOCE") == 0) {

			/*
			 * VOCE
			 */

			strengthModel[itype] = STRENGTH_VOCE;
			// if (comm->me == 0) {
			// 	printf("reading *VOCE\n");
			// }

			t = string("*");
			iNextKwd = -1;
			for (iarg = ioffset + 1; iarg < narg; iarg++) {
				s = string(arg[iarg]);
				if (s.compare(0, t.length(), t) == 0) {
					iNextKwd = iarg;
					break;
				}
			}

			if (iNextKwd < 0) {
				sprintf(str, "no *KEYWORD terminates *VOCE");
				error->all(FLERR, str);
			}

			if (iNextKwd - ioffset != 7 + 1) {
				sprintf(str, "expected 7 arguments following *VOCE but got %d\n", iNextKwd - ioffset - 1);
				error->all(FLERR, str);
			}

			Lookup[VOCE_A][itype] = force->numeric(FLERR, arg[ioffset + 1]);
			Lookup[VOCE_Q1][itype] = force->numeric(FLERR, arg[ioffset + 2]);
			Lookup[VOCE_n1][itype] = force->numeric(FLERR, arg[ioffset + 3]);
			Lookup[VOCE_Q2][itype] = force->numeric(FLERR, arg[ioffset + 4]);
			Lookup[VOCE_n2][itype] = force->numeric(FLERR, arg[ioffset + 5]);
			Lookup[VOCE_C][itype] = force->numeric(FLERR, arg[ioffset + 6]);
			Lookup[VOCE_epsdot0][itype] = force->numeric(FLERR, arg[ioffset + 7]);

			flowstress.VOCE(Lookup[VOCE_A][itype], Lookup[VOCE_Q1][itype], Lookup[VOCE_n1][itype], Lookup[VOCE_Q2][itype], Lookup[VOCE_n2][itype], Lookup[VOCE_C][itype], Lookup[VOCE_epsdot0][itype]);

			// if (comm->me == 0) {
			// 	printf("%60s : %s\n", "Voce strength model", "A - Q1 * exp(-n1 * ep) - Q2 * exp(-n2 * ep)");
			// 	printf("%60s : %g\n", "A", Lookup[VOCE_A][itype]);
			// 	printf("%60s : %g\n", "Q1", Lookup[VOCE_Q1][itype]);
			// 	printf("%60s : %g\n", "n1", Lookup[VOCE_n1][itype]);
			// 	printf("%60s : %g\n", "Q2", Lookup[VOCE_Q2][itype]);
			// 	printf("%60s : %g\n", "n2", Lookup[VOCE_n2][itype]);
			// 	printf("%60s : %g\n", "initial yield stress sigma0", Lookup[VOCE_A][itype] - Lookup[VOCE_Q1][itype] - Lookup[VOCE_Q2][itype]);
			// 	printf("%60s : %g\n", "proportionality factor for logarithmic plastic strain rate dependency C", Lookup[VOCE_C][itype]);
			// 	printf("%60s : %g\n", "dimensionality factor for plastic strain rate dependency epsdot0", Lookup[VOCE_epsdot0][itype]);
			// }
			
			if (Lookup[VOCE_A][itype] - Lookup[VOCE_Q1][itype] - Lookup[VOCE_Q2][itype]  < 0.0 ) {
			  sprintf(str, "(A - Q1 - Q2) cannot be negative but got %f.\n", Lookup[VOCE_A][itype] - Lookup[VOCE_Q1][itype] - Lookup[VOCE_Q2][itype]);
				error->all(FLERR, str);			  
			}

		} else if (strcmp(arg[ioffset], "*EOS_NONE") == 0) {

			/*
			 * no eos
			 */

			eos[itype] = EOS_NONE;
			// if (comm->me == 0) {
			// 	printf("reading *EOS_NONE\n");
			// }

			t = string("*");
			iNextKwd = -1;
			for (iarg = ioffset + 1; iarg < narg; iarg++) {
				s = string(arg[iarg]);
				if (s.compare(0, t.length(), t) == 0) {
					iNextKwd = iarg;
					break;
				}
			}

			if (iNextKwd < 0) {
				sprintf(str, "no *KEYWORD terminates *EOS_NONE");
				error->all(FLERR, str);
			}

			if (iNextKwd - ioffset != 1) {
				sprintf(str, "expected 0 arguments following *EOS_NONE but got %d\n", iNextKwd - ioffset - 1);
				error->all(FLERR, str);
			}

			if (comm->me == 0) {
				printf("\n%60s\n", "no EOS selected");
			}

		} else if (strcmp(arg[ioffset], "*EOS_LINEAR") == 0) {

			/*
			 * linear eos
			 */

			eos[itype] = EOS_LINEAR;
			// if (comm->me == 0) {
			// 	printf("reading *EOS_LINEAR\n");
			// }

			t = string("*");
			iNextKwd = -1;
			for (iarg = ioffset + 1; iarg < narg; iarg++) {
				s = string(arg[iarg]);
				if (s.compare(0, t.length(), t) == 0) {
					iNextKwd = iarg;
					break;
				}
			}

			if (iNextKwd < 0) {
				sprintf(str, "no *KEYWORD terminates *EOS_LINEAR");
				error->all(FLERR, str);
			}

			if (iNextKwd - ioffset != 1) {
				sprintf(str, "expected 0 arguments following *EOS_LINEAR but got %d\n", iNextKwd - ioffset - 1);
				error->all(FLERR, str);
			}

			// if (comm->me == 0) {
			// 	printf("\n%60s\n", "linear EOS based on strain rate");
			// 	printf("%60s : %g\n", "bulk modulus", Lookup[BULK_MODULUS][itype]);
			// }
		} // end linear eos
		else if (strcmp(arg[ioffset], "*EOS_SHOCK") == 0) {

			/*
			 * shock eos
			 */

			eos[itype] = EOS_SHOCK;
			// if (comm->me == 0) {
			// 	printf("reading *EOS_SHOCK\n");
			// }

			t = string("*");
			iNextKwd = -1;
			for (iarg = ioffset + 1; iarg < narg; iarg++) {
				s = string(arg[iarg]);
				if (s.compare(0, t.length(), t) == 0) {
					iNextKwd = iarg;
					break;
				}
			}

			if (iNextKwd < 0) {
				sprintf(str, "no *KEYWORD terminates *EOS_SHOCK");
				error->all(FLERR, str);
			}

			if (iNextKwd - ioffset != 3 + 1) {
				sprintf(str, "expected 3 arguments (c0, S, Gamma) following *EOS_SHOCK but got %d\n", iNextKwd - ioffset - 1);
				error->all(FLERR, str);
			}

			Lookup[EOS_SHOCK_C0][itype] = force->numeric(FLERR, arg[ioffset + 1]);
			Lookup[EOS_SHOCK_S][itype] = force->numeric(FLERR, arg[ioffset + 2]);
			Lookup[EOS_SHOCK_GAMMA][itype] = force->numeric(FLERR, arg[ioffset + 3]);
			// if (comm->me == 0) {
			// 	printf("\n%60s\n", "shock EOS based on strain rate");
			// 	printf("%60s : %g\n", "reference speed of sound", Lookup[EOS_SHOCK_C0][itype]);
			// 	printf("%60s : %g\n", "Hugoniot parameter S", Lookup[EOS_SHOCK_S][itype]);
			// 	printf("%60s : %g\n", "Grueneisen Gamma", Lookup[EOS_SHOCK_GAMMA][itype]);
			// }
		} // end shock eos

		else if (strcmp(arg[ioffset], "*EOS_POLYNOMIAL") == 0) {
			/*
			 * polynomial eos
			 */

			eos[itype] = EOS_POLYNOMIAL;
			// if (comm->me == 0) {
			// 	printf("reading *EOS_POLYNOMIAL\n");
			// }

			t = string("*");
			iNextKwd = -1;
			for (iarg = ioffset + 1; iarg < narg; iarg++) {
				s = string(arg[iarg]);
				if (s.compare(0, t.length(), t) == 0) {
					iNextKwd = iarg;
					break;
				}
			}

			if (iNextKwd < 0) {
				sprintf(str, "no *KEYWORD terminates *EOS_POLYNOMIAL");
				error->all(FLERR, str);
			}

			if (iNextKwd - ioffset != 7 + 1) {
				sprintf(str, "expected 7 arguments following *EOS_POLYNOMIAL but got %d\n", iNextKwd - ioffset - 1);
				error->all(FLERR, str);
			}

			Lookup[EOS_POLYNOMIAL_C0][itype] = force->numeric(FLERR, arg[ioffset + 1]);
			Lookup[EOS_POLYNOMIAL_C1][itype] = force->numeric(FLERR, arg[ioffset + 2]);
			Lookup[EOS_POLYNOMIAL_C2][itype] = force->numeric(FLERR, arg[ioffset + 3]);
			Lookup[EOS_POLYNOMIAL_C3][itype] = force->numeric(FLERR, arg[ioffset + 4]);
			Lookup[EOS_POLYNOMIAL_C4][itype] = force->numeric(FLERR, arg[ioffset + 5]);
			Lookup[EOS_POLYNOMIAL_C5][itype] = force->numeric(FLERR, arg[ioffset + 6]);
			Lookup[EOS_POLYNOMIAL_C6][itype] = force->numeric(FLERR, arg[ioffset + 7]);
			// if (comm->me == 0) {
			// 	printf("\n%60s\n", "polynomial EOS based on strain rate");
			// 	printf("%60s : %g\n", "parameter c0", Lookup[EOS_POLYNOMIAL_C0][itype]);
			// 	printf("%60s : %g\n", "parameter c1", Lookup[EOS_POLYNOMIAL_C1][itype]);
			// 	printf("%60s : %g\n", "parameter c2", Lookup[EOS_POLYNOMIAL_C2][itype]);
			// 	printf("%60s : %g\n", "parameter c3", Lookup[EOS_POLYNOMIAL_C3][itype]);
			// 	printf("%60s : %g\n", "parameter c4", Lookup[EOS_POLYNOMIAL_C4][itype]);
			// 	printf("%60s : %g\n", "parameter c5", Lookup[EOS_POLYNOMIAL_C5][itype]);
			// 	printf("%60s : %g\n", "parameter c6", Lookup[EOS_POLYNOMIAL_C6][itype]);
			// }
		} // end polynomial eos

		else if (strcmp(arg[ioffset], "*FAILURE_MAX_PLASTIC_STRAIN") == 0) {

			/*
			 * maximum plastic strain failure criterion
			 */

			// if (comm->me == 0) {
			// 	printf("reading *FAILURE_MAX_PLASTIC_SRTRAIN\n");
			// }

			t = string("*");
			iNextKwd = -1;
			for (iarg = ioffset + 1; iarg < narg; iarg++) {
				s = string(arg[iarg]);
				if (s.compare(0, t.length(), t) == 0) {
					iNextKwd = iarg;
					break;
				}
			}

			if (iNextKwd < 0) {
				sprintf(str, "no *KEYWORD terminates *FAILURE_MAX_PLASTIC_STRAIN");
				error->all(FLERR, str);
			}

			if (iNextKwd - ioffset != 1 + 1) {
				sprintf(str, "expected 1 arguments following *FAILURE_MAX_PLASTIC_STRAIN but got %d\n", iNextKwd - ioffset - 1);
				error->all(FLERR, str);
			}

			failureModel[itype].failure_max_plastic_strain = true;
			failureModel[itype].integration_point_wise = true;
			failureModel[itype].failure_none = false;
			Lookup[FAILURE_MAX_PLASTIC_STRAIN_THRESHOLD][itype] = force->numeric(FLERR, arg[ioffset + 1]);

			// if (comm->me == 0) {
			// 	printf("\n%60s\n", "maximum plastic strain failure criterion");
			// 	printf("%60s : %g\n", "failure occurs when plastic strain reaches limit",
			// 			Lookup[FAILURE_MAX_PLASTIC_STRAIN_THRESHOLD][itype]);
			// }
		} // end maximum plastic strain failure criterion
		else if (strcmp(arg[ioffset], "*FAILURE_MAX_PAIRWISE_STRAIN") == 0) {

			/*
			 * failure criterion based on maximum strain between a pair of TLSPH particles.
			 */

			// if (comm->me == 0) {
			// 	printf("reading *FAILURE_MAX_PAIRWISE_STRAIN\n");
			// }

			if (update_method != UPDATE_NONE) {
				error->all(FLERR, "cannot use *FAILURE_MAX_PAIRWISE_STRAIN with updated Total-Lagrangian formalism");
			}

			t = string("*");
			iNextKwd = -1;
			for (iarg = ioffset + 1; iarg < narg; iarg++) {
				s = string(arg[iarg]);
				if (s.compare(0, t.length(), t) == 0) {
					iNextKwd = iarg;
					break;
				}
			}

			if (iNextKwd < 0) {
				sprintf(str, "no *KEYWORD terminates *FAILURE_MAX_PAIRWISE_STRAIN");
				error->all(FLERR, str);
			}

			if (iNextKwd - ioffset != 1 + 1) {
				sprintf(str, "expected 1 arguments following *FAILURE_MAX_PAIRWISE_STRAIN but got %d\n", iNextKwd - ioffset - 1);
				error->all(FLERR, str);
			}

			failureModel[itype].failure_max_pairwise_strain = true;
			failureModel[itype].integration_point_wise = false;
			failureModel[itype].failure_none = false;
			Lookup[FAILURE_MAX_PAIRWISE_STRAIN_THRESHOLD][itype] = force->numeric(FLERR, arg[ioffset + 1]);

			// if (comm->me == 0) {
			// 	printf("\n%60s\n", "maximum pairwise strain failure criterion");
			// 	printf("%60s : %g\n", "failure occurs when pairwise strain reaches limit",
			// 			Lookup[FAILURE_MAX_PAIRWISE_STRAIN_THRESHOLD][itype]);
			// }
		} // end pair based maximum strain failure criterion
		else if (strcmp(arg[ioffset], "*FAILURE_MAX_PRINCIPAL_STRAIN") == 0) {
			error->all(FLERR, "this failure model is currently unsupported");

			/*
			 * maximum principal strain failure criterion
			 */
			// if (comm->me == 0) {
			// 	printf("reading *FAILURE_MAX_PRINCIPAL_STRAIN\n");
			// }

			t = string("*");
			iNextKwd = -1;
			for (iarg = ioffset + 1; iarg < narg; iarg++) {
				s = string(arg[iarg]);
				if (s.compare(0, t.length(), t) == 0) {
					iNextKwd = iarg;
					break;
				}
			}

			if (iNextKwd < 0) {
				sprintf(str, "no *KEYWORD terminates *FAILURE_MAX_PRINCIPAL_STRAIN");
				error->all(FLERR, str);
			}

			if (iNextKwd - ioffset != 1 + 1) {
				sprintf(str, "expected 1 arguments following *FAILURE_MAX_PRINCIPAL_STRAIN but got %d\n", iNextKwd - ioffset - 1);
				error->all(FLERR, str);
			}

			failureModel[itype].failure_max_principal_strain = true;
			failureModel[itype].integration_point_wise = true;
			failureModel[itype].failure_none = false;
			Lookup[FAILURE_MAX_PRINCIPAL_STRAIN_THRESHOLD][itype] = force->numeric(FLERR, arg[ioffset + 1]);

			// if (comm->me == 0) {
			// 	printf("\n%60s\n", "maximum principal strain failure criterion");
			// 	printf("%60s : %g\n", "failure occurs when principal strain reaches limit",
			// 			Lookup[FAILURE_MAX_PRINCIPAL_STRAIN_THRESHOLD][itype]);
			// }
		} // end maximum principal strain failure criterion
		else if (strcmp(arg[ioffset], "*FAILURE_JOHNSON_COOK") == 0) {
		  //error->all(FLERR, "this failure model is currently unsupported");
			// if (comm->me == 0) {
			// 	printf("reading *FAILURE_JOHNSON_COOK\n");
			// }

			t = string("*");
			iNextKwd = -1;
			for (iarg = ioffset + 1; iarg < narg; iarg++) {
				s = string(arg[iarg]);
				if (s.compare(0, t.length(), t) == 0) {
					iNextKwd = iarg;
					break;
				}
			}

			if (iNextKwd < 0) {
				sprintf(str, "no *KEYWORD terminates *FAILURE_JOHNSON_COOK");
				error->all(FLERR, str);
			}

			if (iNextKwd - ioffset != 5 + 1) {
			  sprintf(str, "expected 5 arguments following *FAILURE_JOHNSON_COOK but got %d\n", iNextKwd - ioffset - 1);
			  error->all(FLERR, str);
			}

			failureModel[itype].failure_johnson_cook = true;
			failureModel[itype].integration_point_wise = true;
			failureModel[itype].failure_none = false;

			Lookup[FAILURE_JC_D1][itype] = force->numeric(FLERR, arg[ioffset + 1]);
			Lookup[FAILURE_JC_D2][itype] = force->numeric(FLERR, arg[ioffset + 2]);
			Lookup[FAILURE_JC_D3][itype] = force->numeric(FLERR, arg[ioffset + 3]);
			Lookup[FAILURE_JC_D4][itype] = force->numeric(FLERR, arg[ioffset + 4]);
			Lookup[FAILURE_JC_EPDOT0][itype] = force->numeric(FLERR, arg[ioffset + 5]);

			// if (comm->me == 0) {
			// 	printf("\n%60s\n", "Johnson-Cook failure criterion");
			// 	printf("%60s : %g\n", "parameter d1", Lookup[FAILURE_JC_D1][itype]);
			// 	printf("%60s : %g\n", "parameter d2", Lookup[FAILURE_JC_D2][itype]);
			// 	printf("%60s : %g\n", "parameter d3", Lookup[FAILURE_JC_D3][itype]);
			// 	printf("%60s : %g\n", "parameter d4", Lookup[FAILURE_JC_D4][itype]);
			// 	printf("%60s : %g\n", "reference plastic strain rate", Lookup[FAILURE_JC_EPDOT0][itype]);
			// }

		} else if (strcmp(arg[ioffset], "*FAILURE_MAX_PRINCIPAL_STRESS") == 0) {
			error->all(FLERR, "this failure model is currently unsupported");

			/*
			 * maximum principal stress failure criterion
			 */

			// if (comm->me == 0) {
			// 	printf("reading *FAILURE_MAX_PRINCIPAL_STRESS\n");
			// }

			t = string("*");
			iNextKwd = -1;
			for (iarg = ioffset + 1; iarg < narg; iarg++) {
				s = string(arg[iarg]);
				if (s.compare(0, t.length(), t) == 0) {
					iNextKwd = iarg;
					break;
				}
			}

			if (iNextKwd < 0) {
				sprintf(str, "no *KEYWORD terminates *FAILURE_MAX_PRINCIPAL_STRESS");
				error->all(FLERR, str);
			}

			if (iNextKwd - ioffset != 1 + 1) {
				sprintf(str, "expected 1 arguments following *FAILURE_MAX_PRINCIPAL_STRESS but got %d\n", iNextKwd - ioffset - 1);
				error->all(FLERR, str);
			}

			failureModel[itype].failure_max_principal_stress = true;
			failureModel[itype].integration_point_wise = true;
			failureModel[itype].failure_none = false;
			Lookup[FAILURE_MAX_PRINCIPAL_STRESS_THRESHOLD][itype] = force->numeric(FLERR, arg[ioffset + 1]);

			// if (comm->me == 0) {
			// 	printf("\n%60s\n", "maximum principal stress failure criterion");
			// 	printf("%60s : %g\n", "failure occurs when principal stress reaches limit",
			// 			Lookup[FAILURE_MAX_PRINCIPAL_STRESS_THRESHOLD][itype]);
			// }
		} // end maximum principal stress failure criterion

		else if (strcmp(arg[ioffset], "*FAILURE_ENERGY_RELEASE_RATE") == 0) {
			// if (comm->me == 0) {
			// 	printf("reading *FAILURE_ENERGY_RELEASE_RATE\n");
			// }

			t = string("*");
			iNextKwd = -1;
			for (iarg = ioffset + 1; iarg < narg; iarg++) {
				s = string(arg[iarg]);
				if (s.compare(0, t.length(), t) == 0) {
					iNextKwd = iarg;
					break;
				}
			}

			if (iNextKwd < 0) {
				sprintf(str, "no *KEYWORD terminates *FAILURE_ENERGY_RELEASE_RATE");
				error->all(FLERR, str);
			}

			if (iNextKwd - ioffset != 1 + 1) {
				sprintf(str, "expected 1 arguments following *FAILURE_ENERGY_RELEASE_RATE but got %d\n", iNextKwd - ioffset - 1);
				error->all(FLERR, str);
			}

			failureModel[itype].failure_energy_release_rate = true;
			failureModel[itype].failure_none = false;
			Lookup[CRITICAL_ENERGY_RELEASE_RATE][itype] = force->numeric(FLERR, arg[ioffset + 1]);

			// if (comm->me == 0) {
			// 	printf("\n%60s\n", "critical energy release rate failure criterion");
			// 	printf("%60s : %g\n", "failure occurs when energy release rate reaches limit",
			// 			Lookup[CRITICAL_ENERGY_RELEASE_RATE][itype]);
			// }
		} // end energy release rate failure criterion

		else if (strcmp(arg[ioffset], "*GURSON_TVERGAARD_NEEDLEMAN") == 0) {

			/*
			 * GURSON - TVERGAARD - NEEDLEMAN Model
			 */
		  
			// if (comm->me == 0) {
			// 	printf("reading *GURSON_TVERGAARD_NEEDLEMAN\n");
			// }

			t = string("*");
			iNextKwd = -1;
			for (iarg = ioffset + 1; iarg < narg; iarg++) {
				s = string(arg[iarg]);
				if (s.compare(0, t.length(), t) == 0) {
					iNextKwd = iarg;
					break;
				}
			}

			if (iNextKwd < 0) {
				sprintf(str, "no *KEYWORD terminates *GURSON_TVERGAARD_NEEDLEMAN");
				error->all(FLERR, str);
			}

			if (iNextKwd - ioffset != 9 + 1) {
			  sprintf(str, "expected 9 arguments following *GURSON_TVERGAARD_NEEDLEMAN but got %d\n", iNextKwd - ioffset - 2);
			  error->all(FLERR, str);
			}

			failureModel[itype].failure_gtn = true;
			failureModel[itype].integration_point_wise = true;
			failureModel[itype].failure_none = false;
			
			Lookup[GTN_Q1][itype] = force->numeric(FLERR, arg[ioffset + 1]);
			Lookup[GTN_Q2][itype] = force->numeric(FLERR, arg[ioffset + 2]);
			Lookup[GTN_FN][itype] = force->numeric(FLERR, arg[ioffset + 3]);
			//Lookup[GTN_sN][itype] = force->numeric(FLERR, arg[ioffset + 4]);
			if (force->numeric(FLERR, arg[ioffset + 4]) == 0.0)
			  Lookup[GTN_inverse_sN][itype] = 0.0;
			else
			  Lookup[GTN_inverse_sN][itype] = 1.0/force->numeric(FLERR, arg[ioffset + 4]);
			Lookup[GTN_epsN][itype] = force->numeric(FLERR, arg[ioffset + 5]);
			Lookup[GTN_f0][itype] = force->numeric(FLERR, arg[ioffset + 6]);
			Lookup[GTN_fcr][itype] = force->numeric(FLERR, arg[ioffset + 7]);
			Lookup[GTN_fF][itype] = force->numeric(FLERR, arg[ioffset + 8]);
			Lookup[GTN_Komega][itype] = force->numeric(FLERR, arg[ioffset + 9]);

			// if (comm->me == 0) {
			// 	printf("%60s\n", " Gurson - Tvergaard - Needleman failure model");
			// 	printf("%60s : %g\n", "Q1", Lookup[GTN_Q1][itype]);
			// 	printf("%60s : %g\n", "Q2", Lookup[GTN_Q2][itype]);
			// 	if (Lookup[GTN_inverse_sN][itype] == 0.0) {
			// 	  printf("%60s : %g\n", "AN", Lookup[GTN_FN][itype]);
			// 	} else {
			// 	  printf("%60s : %g\n", "FN", Lookup[GTN_FN][itype]);
			// 	  printf("%60s : %g\n", "sN", 1.0/Lookup[GTN_inverse_sN][itype]);
			// 	  printf("%60s : %g\n", "epsN", Lookup[GTN_epsN][itype]);
			// 	}
			// 	printf("%60s : %g\n", "Initial void fraction f0", Lookup[GTN_f0][itype]);
			// 	printf("%60s : %g\n", "Critical void fraction", Lookup[GTN_fcr][itype]);
			// 	printf("%60s : %g\n", "Void fraction at failure", Lookup[GTN_fF][itype]);
			// 	printf("%60s : %g\n", "Komega: magnitude of the damage growth rate in pure shear states (from Nahshon and Hutchinson, 2008)", Lookup[GTN_Komega][itype]);
			// }

		} // End of GURSON_TVERGAARD_NEEDLEMAN

		else if (strcmp(arg[ioffset], "*COCKCROFT_LATHAM") == 0) {

			/*
			 * COCKCROFT - LATHAMGURSON Model
			 */
		  
			// if (comm->me == 0) {
			// 	printf("reading *COCKCROFT_LATHAM\n");
			// }

			t = string("*");
			iNextKwd = -1;
			for (iarg = ioffset + 1; iarg < narg; iarg++) {
				s = string(arg[iarg]);
				if (s.compare(0, t.length(), t) == 0) {
					iNextKwd = iarg;
					break;
				}
			}

			if (iNextKwd < 0) {
				sprintf(str, "no *KEYWORD terminates *COCKCROFT_LATHAM");
				error->all(FLERR, str);
			}

			if (iNextKwd - ioffset != 1 + 1) {
			    sprintf(str, "expected 1 arguments following *COCKCROFT_LATHAM but got %d\n", iNextKwd - ioffset - 1);
			    error->all(FLERR, str);
			}

			failureModel[itype].failure_cockcroft_latham = true;
			failureModel[itype].integration_point_wise = true;
			failureModel[itype].failure_none = false;
			
			Lookup[CL_W][itype] = force->numeric(FLERR, arg[ioffset + 1]);

			// if (comm->me == 0) {
			// 	printf("%60s\n", " Cockcroft - Latham failure model");
			// 	printf("%60s : %g\n", "Total plastic work per unit volume: W", Lookup[CL_W][itype]);
			// }

		} 

		else {
			sprintf(str, "unknown *KEYWORD: %s", arg[ioffset]);
			error->all(FLERR, str);
		}

	}

	// setflag[itype][itype] = 1;
	coeff_init(itype);
}

/* ----------------------------------------------------------------------
 init for one type pair i,j and corresponding j,i
 ------------------------------------------------------------------------- */

double PairTlsph::init_one(int i, int j) {

	if (!allocated)
		allocate();

	if (setflag[i][j] == 0)
		error->all(FLERR, "All pair coeffs are not set");

	if (force->newton == 1)
		error->all(FLERR, "Pair style tlsph requires newton off");

// cutoff = sum of max I,J radii for
// dynamic/dynamic & dynamic/frozen interactions, but not frozen/frozen

	double cutoff = maxrad_dynamic[i] + maxrad_dynamic[j];
	cutoff = MAX(cutoff, maxrad_frozen[i] + maxrad_dynamic[j]);
	cutoff = MAX(cutoff, maxrad_dynamic[i] + maxrad_frozen[j]);
//printf("cutoff for pair pair tlsph = %f\n", cutoff);
	return cutoff;
}

/* ----------------------------------------------------------------------
 init specific to this pair style
 ------------------------------------------------------------------------- */

void PairTlsph::init_style() {
	int i;

	if (force->newton_pair == 1) {
		error->all(FLERR, "Pair style tlsph requires newton pair off");
	}

// request a granular neighbor list
	int irequest = neighbor->request(this);
	neighbor->requests[irequest]->size = 1;
	//neighbor->requests[irequest]->total_lagran = 1;

// set maxrad_dynamic and maxrad_frozen for each type
// include future Fix pour particles as dynamic

	for (i = 1; i <= atom->ntypes; i++)
		onerad_dynamic[i] = onerad_frozen[i] = 0.0;

	double *radius = atom->radius;
	int *type = atom->type;
	int nlocal = atom->nlocal;

	for (i = 0; i < nlocal; i++)
		onerad_dynamic[type[i]] = MAX(onerad_dynamic[type[i]], radius[i]);

	MPI_Allreduce(&onerad_dynamic[1], &maxrad_dynamic[1], atom->ntypes, MPI_DOUBLE, MPI_MAX, world);
	MPI_Allreduce(&onerad_frozen[1], &maxrad_frozen[1], atom->ntypes, MPI_DOUBLE, MPI_MAX, world);

// if first init, create Fix needed for storing reference configuration neighbors

	int igroup = group->find("tlsph");
	if (igroup == -1)
		error->all(FLERR, "Pair style tlsph requires its particles to be part of a group named tlsph. This group does not exist.");

	if (fix_tlsph_reference_configuration == NULL) {
		char **fixarg = new char*[3];
		fixarg[0] = (char *) "SMD_TLSPH_NEIGHBORS";
		fixarg[1] = (char *) "tlsph";
		fixarg[2] = (char *) "SMD_TLSPH_NEIGHBORS";
		modify->add_fix(3, fixarg);
		delete[] fixarg;
		fix_tlsph_reference_configuration = (FixSMD_TLSPH_ReferenceConfiguration *) modify->fix[modify->nfix - 1];
		fix_tlsph_reference_configuration->pair = this;
	}

// find associated SMD_TLSPH_NEIGHBORS fix that must exist
// could have changed locations in fix list since created

	ifix_tlsph = -1;
	for (int i = 0; i < modify->nfix; i++)
		if (strcmp(modify->fix[i]->style, "SMD_TLSPH_NEIGHBORS") == 0)
			ifix_tlsph = i;
	if (ifix_tlsph == -1)
		error->all(FLERR, "Fix SMD_TLSPH_NEIGHBORS does not exist");

}

/* ----------------------------------------------------------------------
 neighbor callback to inform pair style of neighbor list to use
 optional granular history list
 ------------------------------------------------------------------------- */

void PairTlsph::init_list(int id, NeighList *ptr) {
	if (id == 0)
		list = ptr;
}

/* ----------------------------------------------------------------------
 memory usage of local atom-based arrays
 ------------------------------------------------------------------------- */

double PairTlsph::memory_usage() {

	return 118 * nmax * sizeof(double);
}

/* ----------------------------------------------------------------------
 extract method to provide access to this class' data structures
 ------------------------------------------------------------------------- */

void *PairTlsph::extract(const char *str, int &i) {
  //  printf("in PairTlsph::extract\n");
	if (strcmp(str, "smd/tlsph/Fincr_ptr") == 0) {
		return (void *) Fincr;
	} else if (strcmp(str, "smd/tlsph/detF_ptr") == 0) {
		return (void *) detF;
	} else if (strcmp(str, "smd/tlsph/PK1_ptr") == 0) {
		return (void *) PK1;
	} else if (strcmp(str, "smd/tlsph/smoothVel_ptr") == 0) {
		return (void *) smoothVelDifference;
	} else if (strcmp(str, "smd/tlsph/numNeighsRefConfig_ptr") == 0) {
		return (void *) numNeighsRefConfig;
	} else if (strcmp(str, "smd/tlsph/stressTensor_ptr") == 0) {
		return (void *) CauchyStress;
	} else if (strcmp(str, "smd/tlsph/updateFlag_ptr") == 0) {
		return (void *) &updateFlag;
	} else if (strcmp(str, "smd/tlsph/strain_rate_ptr") == 0) {
		return (void *) D;
	} else if (strcmp(str, "smd/tlsph/hMin_ptr") == 0) {
		return (void *) &hMin;
	} else if (strcmp(str, "smd/tlsph/dtCFL_ptr") == 0) {
		return (void *) &dtCFL;
	} else if (strcmp(str, "smd/tlsph/dtRelative_ptr") == 0) {
		return (void *) &dtRelative;
	} else if (strcmp(str, "smd/tlsph/hourglass_error_ptr") == 0) {
		return (void *) hourglass_error;
	} else if (strcmp(str, "smd/tlsph/particle_dt_ptr") == 0) {
		return (void *) particle_dt;
	} else if (strcmp(str, "smd/tlsph/rotation_ptr") == 0) {
		return (void *) R;
	} else if (strcmp(str, "smd/tlsph/damage_increment") == 0) {
		return (void *) damage_increment;
	} else if (strcmp(str, "smd/tlsph/eff_plastic_strain_at_failure_init") == 0) {
		return (void *) eff_plastic_strain_at_failure_init;
	} else if (strcmp(str, "smd/tlsph/rSqMin") == 0) {
	  return (void *) rSqMin;
	} else if (strcmp(str, "smd/tlsph/dvMax") == 0) {
	  return (void *) dvMax;
	} else if (strcmp(str, "smd/tlsph/flowstress_slope") == 0) {
	  return (void *) flowstress_slope;
	} else if (strcmp(str, "smd/tlsph/shepardWeightInv") == 0) {
	  return (void *) shepardWeightInv;
	}

	return NULL;
}

/* ---------------------------------------------------------------------- */

int PairTlsph::pack_forward_comm(int n, int *list, double *buf, int pbc_flag, int *pbc) {
	int i, j, m;
	tagint *mol = atom->molecule;
	double *damage = atom->damage;
	double *eff_plastic_strain = atom->eff_plastic_strain;
	double *eff_plastic_strain_rate = atom->eff_plastic_strain_rate;
	int *npartner = ((FixSMD_TLSPH_ReferenceConfiguration *) modify->fix[ifix_tlsph])->npartner;

	m = 0;
	for (i = 0; i < n; i++) {
		j = list[i];
		buf[m++] = PK1[j](0, 0); // PK1 is not symmetric
		buf[m++] = PK1[j](0, 1);
		buf[m++] = PK1[j](0, 2);
		buf[m++] = PK1[j](1, 0);
		buf[m++] = PK1[j](1, 1);
		buf[m++] = PK1[j](1, 2);
		buf[m++] = PK1[j](2, 0);
		buf[m++] = PK1[j](2, 1);
		buf[m++] = PK1[j](2, 2); // 9

		buf[m++] = Fincr[j](0, 0); // Fincr is not symmetric
		buf[m++] = Fincr[j](0, 1);
		buf[m++] = Fincr[j](0, 2);
		buf[m++] = Fincr[j](1, 0);
		buf[m++] = Fincr[j](1, 1);
		buf[m++] = Fincr[j](1, 2);
		buf[m++] = Fincr[j](2, 0);
		buf[m++] = Fincr[j](2, 1);
		buf[m++] = Fincr[j](2, 2); // 9 + 9 = 18

		buf[m++] = mol[j]; //19
		buf[m++] = damage[j]; //20
		buf[m++] = eff_plastic_strain[j]; //21
		buf[m++] = eff_plastic_strain_rate[j]; //22
		buf[m++] = damage_increment[j]; //23

		buf[m++] = npartner[j]; //24
		buf[m++] = flowstress_slope[j]; //25
		buf[m++] = eff_plastic_strain_at_failure_init[j]; //26

	}
	return m;
}

/* ---------------------------------------------------------------------- */

void PairTlsph::unpack_forward_comm(int n, int first, double *buf) {
	int i, m, last;
	tagint *mol = atom->molecule;
	double *damage = atom->damage;
	double *eff_plastic_strain = atom->eff_plastic_strain;
	double *eff_plastic_strain_rate = atom->eff_plastic_strain_rate;
	int *npartner = ((FixSMD_TLSPH_ReferenceConfiguration *) modify->fix[ifix_tlsph])->npartner;
	//printf("in PairTlsph::unpack_forward_comm\n");

	m = 0;
	last = first + n;
	for (i = first; i < last; i++) {

		PK1[i](0, 0) = buf[m++]; // PK1 is not symmetric
		PK1[i](0, 1) = buf[m++];
		PK1[i](0, 2) = buf[m++];
		PK1[i](1, 0) = buf[m++];
		PK1[i](1, 1) = buf[m++];
		PK1[i](1, 2) = buf[m++];
		PK1[i](2, 0) = buf[m++];
		PK1[i](2, 1) = buf[m++];
		PK1[i](2, 2) = buf[m++];

		Fincr[i](0, 0) = buf[m++];
		Fincr[i](0, 1) = buf[m++];
		Fincr[i](0, 2) = buf[m++];
		Fincr[i](1, 0) = buf[m++];
		Fincr[i](1, 1) = buf[m++];
		Fincr[i](1, 2) = buf[m++];
		Fincr[i](2, 0) = buf[m++];
		Fincr[i](2, 1) = buf[m++];
		Fincr[i](2, 2) = buf[m++];

		mol[i] = static_cast<int>(buf[m++]);
		damage[i] = buf[m++];
		eff_plastic_strain[i] = buf[m++]; //21
		eff_plastic_strain_rate[i] = buf[m++]; //22
		damage_increment[i] = buf[m++]; //23

		npartner[i] = static_cast<int>(buf[m++]); //24
		flowstress_slope[i] = buf[m++]; //25
		eff_plastic_strain_at_failure_init[i] = buf[m++]; //26
	}
}

/* ----------------------------------------------------------------------
 compute effective P-wave speed
 determined by longitudinal modulus
 ------------------------------------------------------------------------- */

void PairTlsph::effective_longitudinal_modulus(const int itype, const double dt, const double d_iso, const double p_rate,
		const Matrix3d d_dev, const Matrix3d sigma_dev_rate, const double damage, double &K_eff, double &mu_eff, double &M_eff) {
	double M0; // initial longitudinal modulus
	double shear_rate_sq;

//	if (damage >= 0.5) {
//		M_eff = Lookup[M_MODULUS][itype];
//		K_eff = Lookup[BULK_MODULUS][itype];
//		mu_eff = Lookup[SHEAR_MODULUS][itype];
//		return;
//	}

	M0 = Lookup[M_MODULUS][itype];

	if (dt * d_iso > 1.0e-10) {
		K_eff = p_rate / d_iso;
		if (K_eff < 0.0) { // it is possible for K_eff to become negative due to strain softening
//			if (damage == 0.0) {
//				error->one(FLERR, "computed a negative effective bulk modulus but particle is not damaged.");
//			}
			K_eff = Lookup[BULK_MODULUS][itype];
		}
	} else {
		K_eff = Lookup[BULK_MODULUS][itype];
	}

	//if (K_eff < Lookup[BULK_MODULUS][itype]) printf("K_eff = %f\n", K_eff);

	if (domain->dimension == 3) {
// Calculate 2 mu by looking at ratio shear stress / shear strain. Use numerical softening to avoid divide-by-zero.
		mu_eff = 0.5 * (sigma_dev_rate(0, 1) + sigma_dev_rate(0, 2) + sigma_dev_rate(1, 2) ) / (d_dev(0, 1) + d_dev(0, 2) + d_dev(1, 2) + 1.0e-16);
		// mu_eff = 0.5
		//		* (sigma_dev_rate(0, 1) / (d_dev(0, 1) + 1.0e-16) + sigma_dev_rate(0, 2) / (d_dev(0, 2) + 1.0e-16)
		//				+ sigma_dev_rate(1, 2) / (d_dev(1, 2) + 1.0e-16)); //This gives a mu_eff up to three times higher than what it should be.
		//double mut = 0.5*max(max((sigma_dev_rate(0, 1) / (d_dev(0, 1) + 1.0e-16)), sigma_dev_rate(0, 2) / (d_dev(0, 2) + 1.0e-16)), sigma_dev_rate(1, 2) / (d_dev(1, 2) + 1.0e-16));
		//if (mu_eff > 1.1*mut) {
		//  printf("mu_eff = %f, mut = %f\n", mu_eff, mut);
		//  printf("sigma_dev_rate(0, 1) / d_dev(0, 1) = %f\n", (sigma_dev_rate(0, 1) / (d_dev(0, 1) + 1.0e-16)));
		//  printf("sigma_dev_rate(0, 2) / d_dev(0, 2) = %f\n", (sigma_dev_rate(0, 2) / (d_dev(0, 2) + 1.0e-16)));
		//  printf("sigma_dev_rate(1, 2) / d_dev(1, 2) = %f\n", (sigma_dev_rate(1, 2) / (d_dev(1, 2) + 1.0e-16)));
		//}
// Calculate magnitude of deviatoric strain rate. This is used for deciding if shear modulus should be computed from current rate or be taken as the initial value.
		shear_rate_sq = d_dev(0, 1) * d_dev(0, 1) + d_dev(0, 2) * d_dev(0, 2) + d_dev(1, 2) * d_dev(1, 2);
	} else {
		mu_eff = 0.5 * (sigma_dev_rate(0, 1) / (d_dev(0, 1) + 1.0e-16));
		shear_rate_sq = d_dev(0, 1) * d_dev(0, 1);
	}

	if (dt * dt * shear_rate_sq < 1.0e-10) {
		mu_eff = Lookup[SHEAR_MODULUS][itype];
	}

	if (mu_eff < Lookup[SHEAR_MODULUS][itype]) { // it is possible for mu_eff to become negative due to strain softening
//		if (damage == 0.0) {
//			printf("mu_eff = %f, tau=%f, gamma=%f\n", mu_eff, sigma_dev_rate(0, 1), d_dev(0, 1));
//			error->message(FLERR, "computed a negative effective shear modulus but particle is not damaged.");
//		}
		mu_eff = Lookup[SHEAR_MODULUS][itype];
	}

//mu_eff = Lookup[SHEAR_MODULUS][itype];

	if (K_eff < 0.0) {
		printf("K_eff = %f, p_rate=%f, vol_rate=%f\n", K_eff, p_rate, d_iso);
	}

	if (mu_eff < 0.0) {
		printf("mu_eff = %f, tau=%f, gamma=%f\n", mu_eff, sigma_dev_rate(0, 1), d_dev(0, 1));
		error->one(FLERR, "");
	}

	M_eff = (K_eff + 4.0 * mu_eff / 3.0); // effective dilational modulus, see Pronto 2d eqn 3.4.8

	if (M_eff < M0) { // do not allow effective dilatational modulus to decrease beyond its initial value
		M_eff = M0;
	}
}

/* ----------------------------------------------------------------------
 compute pressure. Called from AssembleStress().
 ------------------------------------------------------------------------- */
void PairTlsph::ComputePressure(const int i, const double rho, const double mass_specific_energy, const double vol_specific_energy,
	const double pInitial, const double d_iso, double &pFinal, double &p_rate) {
	int *type = atom->type;
	double dt = update->dt;
	double *damage = atom->damage;
	int itype = type[i];

	switch (eos[itype]) {
	case EOS_LINEAR:
	  if ((failureModel[itype].integration_point_wise == true) && (damage[i] > 0.0)){
	    LinearEOSwithDamage(rho, Lookup[REFERENCE_DENSITY][itype], Lookup[BULK_MODULUS][itype], pInitial, dt, pFinal, p_rate, damage[i]);  
	  } else {
	    LinearEOS(Lookup[BULK_MODULUS][itype], pInitial, d_iso, dt, pFinal, p_rate);
	  }
	  break;
	case EOS_NONE:
		pFinal = 0.0;
		p_rate = 0.0;
		break;
	case EOS_SHOCK:
//  rho,  rho0,  e,  e0,  c0,  S,  Gamma,  pInitial,  dt,  &pFinal,  &p_rate);
	  ShockEOS(rho, Lookup[REFERENCE_DENSITY][itype], mass_specific_energy, 0.0, Lookup[EOS_SHOCK_C0][itype],
		   Lookup[EOS_SHOCK_S][itype], Lookup[EOS_SHOCK_GAMMA][itype], pInitial, dt, pFinal, p_rate, damage[i]);
	    break;
	case EOS_POLYNOMIAL:
	  polynomialEOS(rho, Lookup[REFERENCE_DENSITY][itype], vol_specific_energy, Lookup[EOS_POLYNOMIAL_C0][itype],
			Lookup[EOS_POLYNOMIAL_C1][itype], Lookup[EOS_POLYNOMIAL_C2][itype], Lookup[EOS_POLYNOMIAL_C3][itype],
			Lookup[EOS_POLYNOMIAL_C4][itype], Lookup[EOS_POLYNOMIAL_C5][itype], Lookup[EOS_POLYNOMIAL_C6][itype], pInitial, dt,
			pFinal, p_rate, damage[i]);
	    break;
	default:
		error->one(FLERR, "unknown EOS.");
		break;
	}
}

/* ----------------------------------------------------------------------
 Compute stress deviator. Called from AssembleStress().
 ------------------------------------------------------------------------- */
void PairTlsph::ComputeStressDeviator(const int i, const double mass_specific_energy, const Matrix3d sigmaInitial_dev, const Matrix3d d_dev, Matrix3d &sigmaFinal_dev,
				      Matrix3d &sigma_dev_rate, double &plastic_strain_increment, const double pInitial, double &pFinal) {
	double *eff_plastic_strain = atom->eff_plastic_strain;
	double *eff_plastic_strain_rate = atom->eff_plastic_strain_rate;
	int *type = atom->type;
	double *rmass = atom->rmass;
//double *vfrac = atom->vfrac;
	double *e = atom->e;
	double dt = update->dt;
	int itype;
	double *damage = atom->damage;
	double *damage_init = atom->damage_init;
	plastic_strain_increment = 0.0;
	itype = type[i];
	double d;
	double yieldStress;


	switch (strengthModel[itype]) {
	case STRENGTH_LINEAR:

		sigma_dev_rate = 2.0 * Lookup[SHEAR_MODULUS][itype] * d_dev;
		sigmaFinal_dev = sigmaInitial_dev + dt * sigma_dev_rate;

		break;
	case LINEAR_DEFGRAD:
//LinearStrengthDefgrad(Lookup[LAME_LAMBDA][itype], Lookup[SHEAR_MODULUS][itype], Fincr[i], &sigmaFinal_dev);
//eff_plastic_strain[i] = 0.0;
//p_rate = pInitial - sigmaFinal_dev.trace() / 3.0;
//sigma_dev_rate = sigmaInitial_dev - Deviator(sigmaFinal_dev);
		error->one(FLERR, "LINEAR_DEFGRAD is only for debugging purposes and currently deactivated.");
		R[i].setIdentity();
		break;
	case STRENGTH_LINEAR_PLASTIC:
		if (failureModel[itype].failure_gtn) {
		  damage_increment[i] =  GTNStrength(Lookup[SHEAR_MODULUS][itype], flowstress, Lookup[GTN_Q1][itype],
					   Lookup[GTN_Q2][itype], Lookup[GTN_fcr][itype], Lookup[GTN_fF][itype], Lookup[GTN_FN][itype], Lookup[GTN_inverse_sN][itype],
					   Lookup[GTN_epsN][itype], Lookup[GTN_Komega][itype], dt, damage[i], eff_plastic_strain[i], sigmaInitial_dev, d_dev,
					   pInitial, pFinal, sigmaFinal_dev, sigma_dev_rate, plastic_strain_increment, atom->tag[i]);
		  damage[i] += damage_increment[i];

		  double deltat_damage;

		  if (damage_increment[i] > 0.0) deltat_damage = dt / (100 * damage_increment[i]);
		  particle_dt[i] = MIN(particle_dt[i], deltat_damage);

		} else {
		  yieldStress = flowstress.evaluate(eff_plastic_strain[i]);
		  LinearPlasticStrength(Lookup[SHEAR_MODULUS][itype], yieldStress, sigmaInitial_dev, d_dev, dt, sigmaFinal_dev,
					sigma_dev_rate, plastic_strain_increment, damage[i]);
		}
		break;
	case STRENGTH_LUDWICK_HOLLOMON:
		if (failureModel[itype].failure_gtn == true) {
		  damage_increment[i] =  GTNStrength(Lookup[SHEAR_MODULUS][itype], flowstress, Lookup[GTN_Q1][itype],
					   Lookup[GTN_Q2][itype], Lookup[GTN_fcr][itype], Lookup[GTN_fF][itype], Lookup[GTN_FN][itype], Lookup[GTN_inverse_sN][itype],
					   Lookup[GTN_epsN][itype], Lookup[GTN_Komega][itype], dt, damage[i], eff_plastic_strain[i], sigmaInitial_dev, d_dev,
					   pInitial, pFinal, sigmaFinal_dev, sigma_dev_rate, plastic_strain_increment, atom->tag[i]);
		  damage[i] += damage_increment[i];
		} else {
		  
		  yieldStress = flowstress.evaluate(eff_plastic_strain[i]);
		  
		  if (yieldStress != Lookup[LH_A][itype] + Lookup[LH_B][itype] * pow(eff_plastic_strain[i], Lookup[LH_n][itype]))
		    printf("ERROR: yieldStress = %.10e != %.10e", yieldStress, Lookup[LH_A][itype] + Lookup[LH_B][itype] * pow(eff_plastic_strain[i], Lookup[LH_n][itype]));
		  LinearPlasticStrength(Lookup[SHEAR_MODULUS][itype], yieldStress, sigmaInitial_dev, d_dev, dt, sigmaFinal_dev,
					sigma_dev_rate, plastic_strain_increment, damage[i]);
		}
		break;
	case STRENGTH_SWIFT:
		if (failureModel[itype].failure_gtn) {
		  damage_increment[i] =  GTNStrength(Lookup[SHEAR_MODULUS][itype], flowstress, Lookup[GTN_Q1][itype],
					   Lookup[GTN_Q2][itype], Lookup[GTN_fcr][itype], Lookup[GTN_fF][itype], Lookup[GTN_FN][itype], Lookup[GTN_inverse_sN][itype],
					   Lookup[GTN_epsN][itype], Lookup[GTN_Komega][itype], dt, damage[i], eff_plastic_strain[i], sigmaInitial_dev, d_dev,
					   pInitial, pFinal, sigmaFinal_dev, sigma_dev_rate, plastic_strain_increment, atom->tag[i]);
		  damage[i] += damage_increment[i];
		} else {
		  
		  yieldStress = flowstress.evaluate(eff_plastic_strain[i]);
		  
		  LinearPlasticStrength(Lookup[SHEAR_MODULUS][itype], yieldStress, sigmaInitial_dev, d_dev, dt, sigmaFinal_dev,
					sigma_dev_rate, plastic_strain_increment, damage[i]);
		}
		break;
	case STRENGTH_VOCE:
		if (failureModel[itype].failure_gtn) {
		  damage_increment[i] =  GTNStrength(Lookup[SHEAR_MODULUS][itype], flowstress, Lookup[GTN_Q1][itype],
					   Lookup[GTN_Q2][itype], Lookup[GTN_fcr][itype], Lookup[GTN_fF][itype], Lookup[GTN_FN][itype], Lookup[GTN_inverse_sN][itype],
					   Lookup[GTN_epsN][itype], Lookup[GTN_Komega][itype], dt, damage[i], eff_plastic_strain[i], sigmaInitial_dev, d_dev,
					   pInitial, pFinal, sigmaFinal_dev, sigma_dev_rate, plastic_strain_increment, atom->tag[i]);
		  damage[i] += damage_increment[i];
		} else {

		  yieldStress = flowstress.evaluate(eff_plastic_strain[i]);

		/*yieldStress = Lookup[VOCE_A][itype] - Lookup[VOCE_Q1][itype] * exp(-Lookup[VOCE_n1][itype] * eff_plastic_strain[i])
		  - Lookup[VOCE_Q2][itype] * exp(-Lookup[VOCE_n2][itype] * eff_plastic_strain[i]);

		  if ( Lookup[VOCE_C][itype] != 0.0 ) {
		  double epdot_ratio = eff_plastic_strain_rate[i] / Lookup[VOCE_epsdot0][itype];
		  epdot_ratio = MAX(epdot_ratio, 1.0);
		  yieldStress *= 1.0 + Lookup[VOCE_C][itype] * log(epdot_ratio);
		  }*/
		  LinearPlasticStrength(Lookup[SHEAR_MODULUS][itype], yieldStress, sigmaInitial_dev, d_dev, dt, sigmaFinal_dev,
					sigma_dev_rate, plastic_strain_increment, damage[i]);
		}
		break;
	case STRENGTH_JOHNSON_COOK:
		JohnsonCookStrength(Lookup[SHEAR_MODULUS][itype], Lookup[HEAT_CAPACITY][itype], mass_specific_energy, Lookup[JC_A][itype],
				    Lookup[JC_B][itype], Lookup[JC_a][itype], Lookup[JC_C][itype], Lookup[JC_epdot0][itype], Lookup[JC_T0][itype],
				    Lookup[JC_Tmelt][itype], Lookup[JC_M][itype], dt, eff_plastic_strain[i], eff_plastic_strain_rate[i],
				    sigmaInitial_dev, d_dev, sigmaFinal_dev, sigma_dev_rate, plastic_strain_increment, damage[i]);
		break;
	case STRENGTH_NONE:
		sigmaFinal_dev.setZero();
		sigma_dev_rate.setZero();
		break;
	default:
		error->one(FLERR, "unknown strength model.");
		break;
	}
}

/* ----------------------------------------------------------------------
 Compute damage. Called from AssembleStress().
 ------------------------------------------------------------------------- */
void PairTlsph::ComputeDamage(const int i, const Matrix3d strain, const Matrix3d stress, Matrix3d &stress_damaged, double plastic_strain_increment) {
	double *eff_plastic_strain = atom->eff_plastic_strain;
	double *eff_plastic_strain_rate = atom->eff_plastic_strain_rate;
	double *radius = atom->radius;
	double *damage = atom->damage;
	double *damage_init = atom->damage_init;
	double dt = update->dt;
	int *type = atom->type;
	int itype = type[i];
	double jc_failure_strain;
//double damage_gap, damage_rate;
	Matrix3d eye, stress_deviator;

	eye.setIdentity();
	stress_deviator = Deviator(stress);
	double pressure = -stress.trace() / 3.0;

	//// First apply damage to integration point (to stay consistent throughout the loop):
	//if (pressure > 0.0) { // compression: particle can carry compressive load but reduced shear
	//  stress_damaged = -pressure * eye + (1.0 - damage[i]) * Deviator(stress);
        //} else { // tension: particle has reduced tensile and shear load bearing capability
	//  stress_damaged = (1.0 - damage[i]) * (-pressure * eye + Deviator(stress));
        //}

	stress_damaged = stress;

	// Then calculate updated damage onset value:

	if (failureModel[itype].failure_max_principal_stress) {
		error->one(FLERR, "not yet implemented");
		/*
		 * maximum stress failure criterion:
		 */
		IsotropicMaxStressDamage(stress, Lookup[FAILURE_MAX_PRINCIPAL_STRESS_THRESHOLD][itype]);
	} else if (failureModel[itype].failure_max_principal_strain) {
		error->one(FLERR, "not yet implemented");
		/*
		 * maximum strain failure criterion:
		 */
		IsotropicMaxStrainDamage(strain, Lookup[FAILURE_MAX_PRINCIPAL_STRAIN_THRESHOLD][itype]);
	} else if (failureModel[itype].failure_max_plastic_strain) {
		if (eff_plastic_strain[i] >= Lookup[FAILURE_MAX_PLASTIC_STRAIN_THRESHOLD][itype]) {
			damage[i] = 1.0;
			//double damage_gap = 0.5 * Lookup[FAILURE_MAX_PLASTIC_STRAIN_THRESHOLD][itype];
			//damage[i] = (eff_plastic_strain[i] - Lookup[FAILURE_MAX_PLASTIC_STRAIN_THRESHOLD][itype]) / damage_gap;
		}
	} else if (failureModel[itype].failure_johnson_cook) {

	  double plastic_strain_at_failure_increment = 0.1;
	  if ((eff_plastic_strain_at_failure_init[i] == 0) && (damage[i] >= 0.0) && (damage_init[i] >= 1.0)) {
	    // This can happen at restart since eff_plastic_strain_at_failure_init is not stored in the restart file.
	    eff_plastic_strain_at_failure_init[i] = eff_plastic_strain[i] - plastic_strain_at_failure_increment * damage[i];
	  }

	  damage_increment[i] = JohnsonCookDamageIncrement(pressure, stress_deviator, Lookup[FAILURE_JC_D1][itype],
							Lookup[FAILURE_JC_D2][itype], Lookup[FAILURE_JC_D3][itype], Lookup[FAILURE_JC_D4][itype],
							Lookup[FAILURE_JC_EPDOT0][itype], eff_plastic_strain_rate[i], plastic_strain_increment);

	  damage_init[i] += damage_increment[i];


	  if ((eff_plastic_strain_at_failure_init[i] == 0) && (damage_init[i] >= 1.0)) {
	    // Save damage initiation stresses and strains
	    eff_plastic_strain_at_failure_init[i] = MIN(eff_plastic_strain[i], MAX(0.0, eff_plastic_strain[i] + plastic_strain_increment/damage_increment[i] * (1.0 - damage_init[i])));
	  }

	  if ((eff_plastic_strain_at_failure_init[i] > 0) && (damage_init[i] >= 1.0) && (damage[i] < 1.0)) {
	    // Damage has started:
	    double damage_old = damage[i];

	    damage[i] = MAX(0.0, (eff_plastic_strain[i] - eff_plastic_strain_at_failure_init[i])/plastic_strain_at_failure_increment);
	    damage[i] = MIN(1.0, damage[i]);

	    particle_dt[i] = MIN(particle_dt[i], dt/(100*(damage[i] - damage_old)));
	  }
	  // if (atom->tag[i]==78119) {
	  //   printf("Particle %d, eff_plastic_strain_at_failure_init = %.10e, eff_plastic_strain = %.10e, plastic_strain_increment = %.10e, damage_increment = %.10e, damage_init = %.10e, damage = %.10e, \n",atom->tag[i], eff_plastic_strain_at_failure_init[i], eff_plastic_strain[i], plastic_strain_increment, damage_increment[i], damage_init[i], damage[i]);
	  // }

	} else if (failureModel[itype].failure_gtn) {

	  /*
	   * Gurson - Tvergaard - Needleman damage :
	   */
	  if (damage[i] == 0) damage[i] = Lookup[GTN_f0][itype] * Lookup[GTN_Q1][itype];
	  // Damage increment is determined in ComputeStressDeviator
	  if (damage[i] >= 1.0) damage[i] = 1.0;

	} else if (failureModel[itype].failure_cockcroft_latham) {
	  /*
	   * Cockcroft - Latham damage :
	   */
	  damage_increment[i] = CockcroftLathamDamageIncrement(stress, Lookup[CL_W][itype], plastic_strain_increment);
	  damage_init[i] += damage_increment[i];

	  // double deltat_damage;

	  if (damage_init[i] >= 1.0) {
	    // if (damage_increment[i] > 0.0) deltat_damage = dt / (100 * damage_increment[i]);
	    double damage_old = damage[i];
	    damage[i] = MIN(10*(damage_init[i]-1.0), 1.0);
	    damage_increment[i] = damage[i] - damage_old;
	  }//  else {
	  //   if (damage_increment[i] > 0.0) deltat_damage = dt / (10 * damage_increment[i]);
	  // }

	  // particle_dt[i] = MIN(particle_dt[i], deltat_damage);
	}

	damage[i] = MIN(damage[i], 1.0);
}

void PairTlsph::UpdateDegradation() {
	tagint *mol = atom->molecule;
	tagint *tag = atom->tag;
	double **x = atom->x;
	double **v = atom->vest;
	double **x0 = atom->x0;
	double **f = atom->f;
	double *vfrac = atom->vfrac;
	double *radius = atom->radius;
	double *damage = atom->damage;
	double **vint = atom->v;
	double *plastic_strain = atom->eff_plastic_strain;
	double *eff_plastic_strain_rate = atom->eff_plastic_strain_rate;
	int *type = atom->type;
	int nlocal = atom->nlocal;
	int i, j, jj, jnum, itype, idim;
	double r, h, r0_;
	double strain1d, strain1d_max, softening_strain;
	Vector3d dx0, dx, dv, x0i, x0j;
	Vector3d xi, xj;
	int periodic = (domain->xperiodic || domain->yperiodic || domain->zperiodic);

	char str[128];
	tagint **partner = ((FixSMD_TLSPH_ReferenceConfiguration *) modify->fix[ifix_tlsph])->partner;
	int *npartner = ((FixSMD_TLSPH_ReferenceConfiguration *) modify->fix[ifix_tlsph])->npartner;
	float **degradation_ij = ((FixSMD_TLSPH_ReferenceConfiguration *) modify->fix[ifix_tlsph])->degradation_ij;
	float **energy_per_bond = ((FixSMD_TLSPH_ReferenceConfiguration *) modify->fix[ifix_tlsph])->energy_per_bond;
        Vector3d **partnerdx = ((FixSMD_TLSPH_ReferenceConfiguration *) modify->fix[ifix_tlsph])->partnerdx;
        double **r0 = ((FixSMD_TLSPH_ReferenceConfiguration *) modify->fix[ifix_tlsph])->r0;

	for (i = 0; i < nlocal; i++) {

		int numNeighbors = 0;
		
		if (mol[i] < 0) {
                        vint[i][0] = 0.0;
                        vint[i][1] = 0.0;
			vint[i][2] = 0.0;
			f[i][0] = 0.0;
			f[i][1] = 0.0;
			f[i][2] = 0.0;
			smoothVelDifference[i].setZero();
			damage_increment[i] = 0.0;
			continue; // Particle i is not a valid SPH particle (anymore). Skip all interactions with this particle.
		}

		itype = type[i];
		
		if (failureModel[itype].failure_none) { // Do not update degradation if no failure mode is activated for the mol.
		  continue;
		}

		itype = type[i];
		jnum = npartner[i];

		// initialize aveage mass density
		h = 2.0 * radius[i];
		r = 0.0;
		if (failureModel[itype].failure_max_pairwise_strain || failureModel[itype].integration_point_wise) {
		  for (idim = 0; idim < 3; idim++) {
		    x0i(idim) = x0[i][idim];
		    xi(idim) = x[i][idim];
		  }
		}
		
		for (jj = 0; jj < jnum; jj++) {
			// if (degradation_ij[i][jj] >= 1.0)
			// 	continue;
			j = atom->map(partner[i][jj]);
			if (j < 0) { //			// check if lost a partner without first breaking bond
			  error->all(FLERR, "Bond broken not detected during PreCompute -3!");
			  continue;
			}

			//if (mol[j] < 0) {
			//	continue; // Particle j is not a valid SPH particle (anymore). Skip all interactions with this particle.
			//}

			if (abs(mol[i]) != abs(mol[j])) {
				continue;
			}

			if (type[j] != itype) {
				sprintf(str, "particle pair is not of same type!");
				error->all(FLERR, str);
			}

			if (failureModel[itype].failure_max_pairwise_strain) {
			  for (idim = 0; idim < 3; idim++) {
			    x0j(idim) = x0[j][idim];
			    xj(idim) = x[j][idim];
			  }

			  if (periodic)
			    domain->minimum_image(dx0(0), dx0(1), dx0(2));

			  // check that distance between i and j (in the reference config) is less than cutoff
			  dx0 = x0j - x0i;
			  h = radius[i] + radius[j];
			  r0_ = r0[i][jj];

			  // distance vectors in current and reference configuration, velocity difference
			  dx = xj - xi;
			  r = dx.norm(); // current distance

			  strain1d = (r - r0_) / r0_;
			  strain1d_max = Lookup[FAILURE_MAX_PAIRWISE_STRAIN_THRESHOLD][itype];
			  softening_strain = 2.0 * strain1d_max;

			  if (strain1d > strain1d_max) {
			    degradation_ij[i][jj] = max(degradation_ij[i][jj], float((strain1d - strain1d_max) / softening_strain));
			    if (degradation_ij[i][jj] >= 0.99) {
			      printf("Link between %d and %d destroyed.\n", tag[i], partner[i][jj]);
			      cout << "Here is dx0:" << endl << dx0 << endl;
			      degradation_ij[i][jj] = 0.99;
			    }
			    //degradation_ij[i][jj] = (strain1d - strain1d_max) / softening_strain;
			  } else {
			    //degradation_ij[i][jj] = 0.0;
			  }
			}

			if (failureModel[itype].failure_energy_release_rate) {
			  
				h = radius[i] + radius[j];
				double Vic = (2.0 / 3.0) * h * h * h * h; // interaction volume for 2d plane strain
				double critical_energy_per_bond = Lookup[CRITICAL_ENERGY_RELEASE_RATE][itype] / (2.0 * Vic);

				if (energy_per_bond[i][jj] > critical_energy_per_bond) {
					degradation_ij[i][jj] = 1.0;
				}
			}

			if (failureModel[itype].integration_point_wise) {
			  for (idim = 0; idim < 3; idim++) {
			    xj(idim) = x[j][idim];
			  }

			  // distance vectors in current and reference configuration, velocity difference
			  dx = xj - xi;

			  if (damage[j] == 0.0) {
			    partnerdx[i][jj] = dx;
			  }

			  if (damage[j] > 0.0) {
			    degradation_ij[i][jj] = damage[j]/ npartner[j];
			  } else {
			    degradation_ij[i][jj] = 0;
			  }

			  //if (damage[i] >= 1.0) degradation_ij[i][jj] = 1.0;

			  if (degradation_ij[i][jj] >= 1.0) { // delete interaction if fully damaged
			    //printf("Link between %d and %d destroyed due to complete degradation with damage[i] = %f and damage[j] = %f.\n", tag[i], partner[i][jj], damage[i], damage[j]);
			    degradation_ij[i][jj] = 1.0;
			  }
			}
			
			if (degradation_ij[i][jj] < 1.0 && damage[i] < 1.0) {
			  numNeighbors += 1;
			}
		} // end loop over jj neighbors of i
		
		if (numNeighbors == 0) {
		  printf("Deleting particle [%d] because damage = %f\n", tag[i], damage[i]);
		  dtCFL = MIN(dtCFL, update->dt);
		  mol[i] = -mol[i];
		  vint[i][0] = 0.0;
		  vint[i][1] = 0.0;
		  vint[i][2] = 0.0;
		  f[i][0] = 0.0;
		  f[i][1] = 0.0;
		  f[i][2] = 0.0;
		  smoothVelDifference[i].setZero();
		  }

	} // end loop over i
}

double PairTlsph::CalculateScale(const float degradation, const int itype) {
  if (failureModel[itype].integration_point_wise == true) {
    double start = 0; // Arbitrary value that seems to work
    if (degradation <= start) {
      return 1.0;
    }
    if (degradation >= 1.0) {
      return 0.0;
    }

    return 0.5 + 0.5 * cos( M_PI * (degradation - start) / (1.0 - start) );
  } else {
    return 1.0 - degradation;
  }
}

Matrix3d PairTlsph::CreateOrthonormalBasisFromOneVector(Vector3d sU) {
  Matrix3d P;
  Vector3d sV, sW;
  double sU_Norm;
  
  // Make sure that sU has a norm of one:
  sU_Norm = sU.norm();
  if (sU_Norm != 1.0) {
    sU /= sU_Norm;
  }
  
  if (abs(float(sU[1])) > 1.0e-15) {
    sV[0] = 0.0;
    sV[1] = - sU[2];
    sV[2] = sU[1];
  } else if (abs(float(sU[2])) > 1.0e-15) {
    sV[0] = sU[2];
    sV[1] = 0.0;
    sV[2] = -sU[0];
  } else {
    sV[0] = 0.0;
    sV[1] = 1.0;
    sV[2] = 0.0;
  }
  
  sV /= sV.norm();
  sW = sU.cross(sV);
  //sW /= sW.norm(); This can be skipped since sU and sV are orthogonal and both unitary.
  
  P.col(0) = sU;
  P.col(1) = sV;
  P.col(2) = sW;
  
  return P;
}

void PairTlsph::AdjustStressForZeroForceBC(const Matrix3d sigma, const Vector3d sU, Matrix3d &sigmaBC) {
  Vector3d sV, sW, sigman;
  Matrix3d P;
  //cout << "Creating mirror particle i=" << tag[i] << " and j=" << tag[j] << endl;

  P = CreateOrthonormalBasisFromOneVector(sU);

  sigmaBC = P.transpose() * sigma * P; // We transform sigmaBC to the surface basis

  sigmaBC.col(0).setZero();
  sigmaBC.row(0).setZero();

  sigmaBC = P * sigmaBC * P.transpose();

  // Check if sigmaBC * surfaceNormalNormi = 0:
  sigman = sigmaBC * sU;
  if (sigman.norm() > 1.0e-5){
    cout << "Here is sigman :" << endl << sigman << endl;
    cout << "Here is P.transpose() * sigmaBC * P :" << endl << P.transpose() * sigmaBC * P << endl;
    cout << "Here is P.transpose() * sU :" << endl << P.transpose() * sU << endl;
    cout << "Here is P :" << endl << P << endl;
  }
}


void PairTlsph::write_restart(FILE *fp){

  // Save all the settings, eos, strengthModel, Lookup and failureModel vectors in the restart file.
  write_restart_settings(fp);
  fwrite(&eos[1],sizeof(int),atom->ntypes,fp); // Do not save eos[0] as it is not used
  fwrite(&strengthModel[1],sizeof(int),atom->ntypes,fp);
  for (int ikey = 0; ikey<MAX_KEY_VALUE; ikey++)
    fwrite(&Lookup[ikey][1],sizeof(double),atom->ntypes,fp);
  fwrite(&failureModel[1],sizeof(struct failure_types),atom->ntypes,fp);
}

void PairTlsph::read_restart(FILE *fp) {
  read_restart_settings(fp);
  allocate();

  int me = comm->me;
  if (me == 0) {
    fread(&eos[1],sizeof(int),atom->ntypes,fp);
    fread(&strengthModel[1],sizeof(int),atom->ntypes,fp);
    for (int ikey = 0; ikey<MAX_KEY_VALUE; ikey++)
      fread(&Lookup[ikey][1],sizeof(double),atom->ntypes,fp);
    fread(&failureModel[1],sizeof(struct failure_types),atom->ntypes,fp);
  }
  
  MPI_Bcast(&eos[1],atom->ntypes,MPI_INT,0,world);
  MPI_Bcast(&strengthModel[1],atom->ntypes,MPI_INT,0,world);
  for (int ikey = 0; ikey<MAX_KEY_VALUE; ikey++)
    MPI_Bcast(&Lookup[ikey][1],atom->ntypes,MPI_DOUBLE,0,world);
  MPI_Bcast(&failureModel[1],atom->ntypes,MPI_failure_types,0,world);

  for (int itype=1; itype<=atom->ntypes; itype++)
    coeff_init(itype);
}

void PairTlsph::write_restart_settings(FILE *fp){
  fwrite(&cut_comm,sizeof(double),1,fp);
  fwrite(&update_threshold,sizeof(double),1,fp);
  fwrite(&update_method,sizeof(int),1,fp);
}

void PairTlsph::read_restart_settings(FILE *fp) {
  int me = comm->me;
  if (me == 0) {
    fread(&cut_comm,sizeof(double),1,fp);
    fread(&update_threshold,sizeof(double),1,fp);
    fread(&update_method,sizeof(int),1,fp);
  }
  MPI_Bcast(&cut_comm,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&update_threshold,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&update_method,1,MPI_INT,0,world);
  
}

void PairTlsph::coeff_init(int itype){

  setflag[itype][itype] = 1;
  if (comm->me == 0) {
    printf(
	   "\n>>========>>========>>========>>========>>========>>========>>========>>========>>========>>========>>========>>========\n");
    printf("SMD / TLSPH PROPERTIES OF PARTICLE TYPE %d:\n", itype);

    printf("\n material unspecific properties for SMD/TLSPH definition of particle type %d:\n", itype);
    printf("%60s : %g\n", "reference density", Lookup[REFERENCE_DENSITY][itype]);
    printf("%60s : %g\n", "Young's modulus", Lookup[YOUNGS_MODULUS][itype]);
    printf("%60s : %g\n", "Poisson ratio", Lookup[POISSON_RATIO][itype]);
    printf("%60s : %g\n", "linear viscosity coefficient", Lookup[VISCOSITY_Q1][itype]);
    printf("%60s : %g\n", "quadratic viscosity coefficient", Lookup[VISCOSITY_Q2][itype]);
    printf("%60s : %g\n", "hourglass control coefficient", Lookup[HOURGLASS_CONTROL_AMPLITUDE][itype]);
    printf("%60s : %g\n", "heat capacity [energy / (mass * temperature)]", Lookup[HEAT_CAPACITY][itype]);
    printf("%60s : %g\n", "Lame constant lambda", Lookup[LAME_LAMBDA][itype]);
    printf("%60s : %g\n", "shear modulus", Lookup[SHEAR_MODULUS][itype]);
    printf("%60s : %g\n", "bulk modulus", Lookup[BULK_MODULUS][itype]);
    printf("%60s : %g\n", "signal velocity", Lookup[SIGNAL_VELOCITY][itype]);  
  }

  if (strengthModel[itype] == STRENGTH_LINEAR_PLASTIC) {

    flowstress.linear_plastic(Lookup[YIELD_STRESS][itype], Lookup[HARDENING_PARAMETER][itype]);

    if (comm->me == 0) {
      printf("\n%60s\n", "Linear elastic / perfectly plastic strength based on strain rate");
      printf("%60s : %g\n", "Young's modulus", Lookup[YOUNGS_MODULUS][itype]);
      printf("%60s : %g\n", "Poisson ratio", Lookup[POISSON_RATIO][itype]);
      printf("%60s : %g\n", "shear modulus", Lookup[SHEAR_MODULUS][itype]);
      printf("%60s : %g\n", "constant yield stress", Lookup[YIELD_STRESS][itype]);
      printf("%60s : %g\n", "constant hardening parameter", Lookup[HARDENING_PARAMETER][itype]);
    }
  } else if (strengthModel[itype] == STRENGTH_JOHNSON_COOK) {
    
    flowstress.JC(Lookup[JC_A][itype], Lookup[JC_B][itype], Lookup[JC_a][itype],
		  Lookup[JC_C][itype], Lookup[JC_epdot0][itype], Lookup[JC_T0][itype],
		  Lookup[JC_Tmelt][itype], Lookup[JC_M][itype]);

    if (comm->me == 0) {
      printf("\n%60s\n", "Johnson Cook material strength model");
      printf("%60s : %g\n", "A: initial yield stress", Lookup[JC_A][itype]);
      printf("%60s : %g\n", "B : proportionality factor for plastic strain dependency", Lookup[JC_B][itype]);
      printf("%60s : %g\n", "a : exponent for plastic strain dependency", Lookup[JC_a][itype]);
      printf("%60s : %g\n", "C : proportionality factor for logarithmic plastic strain rate dependency",
	     Lookup[JC_C][itype]);
      printf("%60s : %g\n", "epdot0 : dimensionality factor for plastic strain rate dependency",
	     Lookup[JC_epdot0][itype]);
      printf("%60s : %g\n", "T0 : reference (room) temperature", Lookup[JC_T0][itype]);
      printf("%60s : %g\n", "Tmelt : melting temperature", Lookup[JC_Tmelt][itype]);
      printf("%60s : %g\n", "M : exponent for temperature dependency", Lookup[JC_M][itype]);
    }
  } else if (strengthModel[itype] == STRENGTH_LUDWICK_HOLLOMON) {
    
    flowstress.LH(Lookup[LH_A][itype], Lookup[LH_B][itype], Lookup[LH_n][itype]);

    if (comm->me == 0) {
      printf("\n%60s : %s\n", "Ludwick-Hollomon material strength model","A + B * pow(ep, n)");
      printf("%60s : %g\n", "A: initial yield stress", Lookup[LH_A][itype]);
      printf("%60s : %g\n", "B : proportionality factor for plastic strain dependency", Lookup[LH_B][itype]);
      printf("%60s : %g\n", "n : exponent for plastic strain dependency", Lookup[LH_n][itype]);
    }
  } else if (strengthModel[itype] == STRENGTH_SWIFT) {

    flowstress.SWIFT(Lookup[SWIFT_A][itype], Lookup[SWIFT_B][itype], Lookup[SWIFT_n][itype], Lookup[SWIFT_eps0][itype]);

    if (comm->me == 0) {
      printf("\n%60s : %s\n", "Swift strength model", "A + B * pow(ep - eps0, n)");
      printf("%60s : %g\n", "A: initial yield stress", Lookup[SWIFT_A][itype]);
      printf("%60s : %g\n", "B : proportionality factor for plastic strain dependency", Lookup[SWIFT_B][itype]);
      printf("%60s : %g\n", "n : exponent for plastic strain dependency", Lookup[SWIFT_n][itype]);
      printf("%60s : %g\n", "eps0 : initial plastic strain", Lookup[SWIFT_eps0][itype]);
    }
  } else if (strengthModel[itype] == STRENGTH_VOCE) {
    flowstress.VOCE(Lookup[VOCE_A][itype], Lookup[VOCE_Q1][itype], Lookup[VOCE_n1][itype], Lookup[VOCE_Q2][itype], Lookup[VOCE_n2][itype], Lookup[VOCE_C][itype], Lookup[VOCE_epsdot0][itype]);

    if (comm->me == 0) {
      printf("\n%60s : %s\n", "Voce strength model", "A - Q1 * exp(-n1 * ep) - Q2 * exp(-n2 * ep)");
      printf("%60s : %g\n", "A", Lookup[VOCE_A][itype]);
      printf("%60s : %g\n", "Q1", Lookup[VOCE_Q1][itype]);
      printf("%60s : %g\n", "n1", Lookup[VOCE_n1][itype]);
      printf("%60s : %g\n", "Q2", Lookup[VOCE_Q2][itype]);
      printf("%60s : %g\n", "n2", Lookup[VOCE_n2][itype]);
      printf("%60s : %g\n", "initial yield stress sigma0", Lookup[VOCE_A][itype] - Lookup[VOCE_Q1][itype] - Lookup[VOCE_Q2][itype]);
      printf("%60s : %g\n", "proportionality factor for logarithmic plastic strain rate dependency C", Lookup[VOCE_C][itype]);
      printf("%60s : %g\n", "dimensionality factor for plastic strain rate dependency epsdot0", Lookup[VOCE_epsdot0][itype]);
    }
  }

  if (eos[itype] == EOS_LINEAR) {
    if (comm->me == 0) {
      printf("\n%60s\n", "linear EOS based on strain rate");
      printf("%60s : %g\n", "bulk modulus", Lookup[BULK_MODULUS][itype]);
    }
  } else if (eos[itype] == EOS_SHOCK) {
    if (comm->me == 0) {
      printf("\n%60s\n", "shock EOS based on strain rate");
      printf("%60s : %g\n", "reference speed of sound", Lookup[EOS_SHOCK_C0][itype]);
      printf("%60s : %g\n", "Hugoniot parameter S", Lookup[EOS_SHOCK_S][itype]);
      printf("%60s : %g\n", "Grueneisen Gamma", Lookup[EOS_SHOCK_GAMMA][itype]);
    }
  } else if (eos[itype] == EOS_POLYNOMIAL) {
    if (comm->me == 0) {
      printf("\n%60s\n", "polynomial EOS based on strain rate");
      printf("%60s : %g\n", "parameter c0", Lookup[EOS_POLYNOMIAL_C0][itype]);
      printf("%60s : %g\n", "parameter c1", Lookup[EOS_POLYNOMIAL_C1][itype]);
      printf("%60s : %g\n", "parameter c2", Lookup[EOS_POLYNOMIAL_C2][itype]);
      printf("%60s : %g\n", "parameter c3", Lookup[EOS_POLYNOMIAL_C3][itype]);
      printf("%60s : %g\n", "parameter c4", Lookup[EOS_POLYNOMIAL_C4][itype]);
      printf("%60s : %g\n", "parameter c5", Lookup[EOS_POLYNOMIAL_C5][itype]);
      printf("%60s : %g\n", "parameter c6", Lookup[EOS_POLYNOMIAL_C6][itype]);
    }
  }

  if (failureModel[itype].failure_max_plastic_strain) {
    if (comm->me == 0) {
      printf("\n%60s\n", "maximum plastic strain failure criterion");
      printf("%60s : %g\n", "failure occurs when plastic strain reaches limit",
	     Lookup[FAILURE_MAX_PLASTIC_STRAIN_THRESHOLD][itype]);
    }
  }

  if (failureModel[itype].failure_max_pairwise_strain) {
    if (comm->me == 0) {
      printf("\n%60s\n", "maximum pairwise strain failure criterion");
      printf("%60s : %g\n", "failure occurs when pairwise strain reaches limit",
	     Lookup[FAILURE_MAX_PAIRWISE_STRAIN_THRESHOLD][itype]);
    }
  }

  if (failureModel[itype].failure_max_principal_strain) {
    if (comm->me == 0) {
      printf("\n%60s\n", "maximum principal strain failure criterion");
      printf("%60s : %g\n", "failure occurs when principal strain reaches limit",
	     Lookup[FAILURE_MAX_PRINCIPAL_STRAIN_THRESHOLD][itype]);
    }
  }

  if (failureModel[itype].failure_johnson_cook) {
    if (comm->me == 0) {
      printf("\n%60s\n", "Johnson-Cook failure criterion");
      printf("%60s : %g\n", "parameter d1", Lookup[FAILURE_JC_D1][itype]);
      printf("%60s : %g\n", "parameter d2", Lookup[FAILURE_JC_D2][itype]);
      printf("%60s : %g\n", "parameter d3", Lookup[FAILURE_JC_D3][itype]);
      printf("%60s : %g\n", "parameter d4", Lookup[FAILURE_JC_D4][itype]);
      printf("%60s : %g\n", "reference plastic strain rate", Lookup[FAILURE_JC_EPDOT0][itype]);
    }
  }

  if (failureModel[itype].failure_max_principal_stress) {
    if (comm->me == 0) {
      printf("\n%60s\n", "maximum principal stress failure criterion");
      printf("%60s : %g\n", "failure occurs when principal stress reaches limit",
	     Lookup[FAILURE_MAX_PRINCIPAL_STRESS_THRESHOLD][itype]);
    }
  }

  if (failureModel[itype].failure_energy_release_rate) {
    if (comm->me == 0) {
      printf("\n%60s\n", "critical energy release rate failure criterion");
      printf("%60s : %g\n", "failure occurs when energy release rate reaches limit",
	     Lookup[CRITICAL_ENERGY_RELEASE_RATE][itype]);
    }
  }

  if (failureModel[itype].failure_gtn) {
    if (comm->me == 0) {
      printf("\n%60s\n", " Gurson - Tvergaard - Needleman failure model");
      printf("%60s : %g\n", "Q1", Lookup[GTN_Q1][itype]);
      printf("%60s : %g\n", "Q2", Lookup[GTN_Q2][itype]);
      if (Lookup[GTN_inverse_sN][itype] == 0.0) {
	printf("%60s : %g\n", "AN", Lookup[GTN_FN][itype]);
      } else {
	printf("%60s : %g\n", "FN", Lookup[GTN_FN][itype]);
	printf("%60s : %g\n", "sN", 1.0/Lookup[GTN_inverse_sN][itype]);
	printf("%60s : %g\n", "epsN", Lookup[GTN_epsN][itype]);
      }
      printf("%60s : %g\n", "Initial void fraction f0", Lookup[GTN_f0][itype]);
      printf("%60s : %g\n", "Critical void fraction", Lookup[GTN_fcr][itype]);
      printf("%60s : %g\n", "Void fraction at failure", Lookup[GTN_fF][itype]);
      printf("%60s : %g\n", "Komega: magnitude of the damage growth rate in pure shear states (from Nahshon and Hutchinson, 2008)", Lookup[GTN_Komega][itype]);
    }
  }

  if (failureModel[itype].failure_cockcroft_latham) {
    if (comm->me == 0) {
      printf("\n%60s\n", " Cockcroft - Latham failure model");
      printf("%60s : %g\n", "Total plastic work per unit volume: W", Lookup[CL_W][itype]);
    }
  }

  if (comm->me == 0) {
    printf(
	   "\n>>========>>========>>========>>========>>========>>========>>========>>========>>========>>========>>========>>========\n\n");
  }
}

void PairTlsph::setup() {
  //printf("In PairTlsph::setup()\n");
}


void PairTlsph::forward_comm_pair_tl() {
  int n_missing_all = ((FixSMD_TLSPH_ReferenceConfiguration *) modify->fix[ifix_tlsph])->n_missing_all;
  int *nsendlist = ((FixSMD_TLSPH_ReferenceConfiguration *) modify->fix[ifix_tlsph])->nsendlist;
  int **sendlist = ((FixSMD_TLSPH_ReferenceConfiguration *) modify->fix[ifix_tlsph])->sendlist;
  int firstrecv = ((FixSMD_TLSPH_ReferenceConfiguration *) modify->fix[ifix_tlsph])->firstrecv;
  int nprocs = comm->nprocs;

  if (n_missing_all == 0) return;

  // printf("In PairTlsph::forward_comm_tl()\n");

  int n, nrecv_tot, nrecv_pos;

  nrecv_tot = 0;

  for (int proc_i = 0; proc_i < nprocs; proc_i++) {
    
    if (proc_i != comm->me) {
      if (nsendlist[proc_i]) {
	// Now that the sendlist is created, pack attributes:
	if (nsendlist[proc_i]*comm_forward > maxsend) {
	  maxsend = nsendlist[proc_i]*comm_forward;
	  memory->grow(buf_send,maxsend + BUFEXTRA,"pair_smd_tlsph:buf_send");
	}

	n = pack_forward_comm(nsendlist[proc_i],sendlist[proc_i],buf_send,0,NULL);
      }

      MPI_Send(&nsendlist[proc_i], 1, MPI_INT, proc_i, 0, world);

      if (nsendlist[proc_i]) {
	MPI_Send(buf_send, n, MPI_DOUBLE, proc_i, 0, world);
      }
    } else {

      // If I am proc_i:

      nrecv_pos = 0;

      for (int proc_j = 0; proc_j < nprocs; proc_j++) {
	if (proc_j != comm->me) {
	  MPI_Recv(&nrecv[proc_j], 1, MPI_INT, proc_j, 0, world, MPI_STATUS_IGNORE);

	  if (nrecv[proc_j] == 0) continue;

	  nrecv_tot += nrecv[proc_j];


	  if (nrecv_tot*comm_forward > maxrecv) {
	    maxrecv = nrecv_tot*comm_forward;
	    memory->grow(buf_recv,maxrecv + BUFEXTRA,"pair_smd_tlsph:buf_recv");
	  }

	  if (nrecv[proc_j]) {
	    MPI_Recv(&buf_recv[nrecv_pos], nrecv[proc_j]*comm_forward,MPI_DOUBLE, proc_j, 0, world, MPI_STATUS_IGNORE);
	    nrecv_pos += nrecv[proc_j]*comm_forward;
	  }
	}
      }
    }
    MPI_Barrier(world);
  }

  // unpack buffer
  if (nrecv_tot) unpack_forward_comm(nrecv_tot,firstrecv,buf_recv);
}
