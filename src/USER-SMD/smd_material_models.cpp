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
#include <iostream>
#include "math_special.h"
#include "smd_math.h"
#include <stdio.h>

#include <Eigen/Eigen>
#include "smd_material_models.h"

using namespace LAMMPS_NS::MathSpecial;
using namespace SMD_Math;
using namespace std;
using namespace Eigen;

#define MIN(A,B) ((A) < (B) ? (A) : (B))
#define MAX(A,B) ((A) > (B) ? (A) : (B))

/* ----------------------------------------------------------------------
 linear EOS for use with linear elasticity
 input: initial pressure pInitial, isotropic part of the strain rate d, time-step dt
 output: final pressure pFinal, pressure rate p_rate
 ------------------------------------------------------------------------- */
void LinearEOS(const double lambda, const double pInitial, const double d, const double dt, double &pFinal, double &p_rate) {

	/*
	 * pressure rate
	 */
	p_rate = lambda * d;

	pFinal = pInitial + dt * p_rate; // increment pressure using pressure rate
	//cout << "hurz" << endl;

}

/* ----------------------------------------------------------------------
 Linear EOS when there is damage integration point wise
 input:
 current density rho
 reference density rho0
 reference bulk modulus K
 initial pressure pInitial
 time step dt
 current damage

 output:
 pressure rate p_rate
 final pressure pFinal

 ------------------------------------------------------------------------- */
void LinearEOSwithDamage(const double rho, const double rho0, const double K, const double pInitial, const double dt, double &pFinal, double &p_rate, const double damage) {

  double mu = rho / rho0 - 1.0;

  if ((damage > 0.0) && (mu < 0.0)) {
    if (damage >= 1.0) {
      pFinal = 0.0;
    } else {
      mu = (1 - damage) * mu;//破坏材料变软
      pFinal = -(1 - damage) * K * mu;
    }
  } else {
    pFinal = -K * mu;
  }

  p_rate = (pFinal - pInitial) / dt;
}

/* ----------------------------------------------------------------------
 shock EOS
 input:
 current density rho
 reference density rho0
 current energy density e
 reference energy density e0
 reference speed of sound c0
 shock Hugoniot parameter S
 Grueneisen parameter Gamma
 initial pressure pInitial
 time step dt

 output:
 pressure rate p_rate
 final pressure pFinal

 ------------------------------------------------------------------------- */
void ShockEOS(const double rho, const double rho0, const double e, const double e0, const double c0, const double S, const double Gamma, const double pInitial, const double dt,
	      double &pFinal, double &p_rate, const double damage) {

	double mu = rho / rho0 - 1.0;
	double pH = rho0 * square(c0) * mu * (1.0 + mu) / square(1.0 - (S - 1.0) * mu);

	pFinal = -(pH + rho * Gamma * (e - e0));

	if ( damage > 0.0 ) {
	  if ( pFinal > 0.0 ) {
	    if ( damage >= 1.0) {
	      pFinal = -rho0 * Gamma * (e - e0);
	    } else {
	      double mu_damaged = (1.0 - damage) * mu;
	      double pH_damaged = rho0 * (1.0 - damage) * square(c0) * mu_damaged * (1.0 + mu_damaged) / square(1.0 - (S - 1.0) * mu_damaged);
	      pFinal = (-pH_damaged + rho0 * (1 + mu_damaged) * Gamma * (e - e0));;
	    }
	  }
	}

	//printf("shock EOS: rho = %g, rho0 = %g, Gamma=%f, c0=%f, S=%f, e=%f, e0=%f\n", rho, rho0, Gamma, c0, S, e, e0);
	//printf("pFinal = %f\n", pFinal);
	p_rate = (pFinal - pInitial) / dt;

}

/* ----------------------------------------------------------------------
 polynomial EOS
 input:
 current density rho
 reference density rho0
 coefficients 0 .. 6
 initial pressure pInitial
 time step dt

 output:
 pressure rate p_rate
 final pressure pFinal

 ------------------------------------------------------------------------- */
void polynomialEOS(const double rho, const double rho0, const double e, const double C0, const double C1, const double C2, const double C3, const double C4, const double C5, const double C6,
		   const double pInitial, const double dt, double &pFinal, double &p_rate, const double damage) {

	double mu = rho / rho0 - 1.0;

	if (mu > 0.0) {
		pFinal = C0 + C1 * mu + C2 * mu * mu + C3 * mu * mu * mu; // + (C4 + C5 * mu + C6 * mu * mu) * e;
	} else {
		pFinal = C0 + C1 * mu + C3 * mu * mu * mu; //  + (C4 + C5 * mu) * e;
	}
	pFinal = -pFinal; // we want the mean stress, not the pressure.

	if ( damage > 0.0 ) {
	  double mu_damaged = (1.0 - damage) * mu;
	  double pFinal_damaged;
	  if (mu_damaged > 0.0) {
	    pFinal_damaged = C0 + C1 * mu_damaged + C2 * mu_damaged * mu_damaged + C3 * mu_damaged * mu_damaged * mu_damaged; // + (C4 + C5 * mu_damaged + C6 * mu_damaged * mu_damaged) * e;
	  } else {
	    pFinal_damaged = C0 + C1 * mu_damaged + C3 * mu_damaged * mu_damaged * mu_damaged; //  + (C4 + C5 * mu_damaged) * e;
	  }
	  pFinal_damaged = -pFinal_damaged;
	  pFinal = MIN(pFinal, pFinal_damaged);
	}

	pFinal = -pFinal; // we want the mean stress, not the pressure.

	//printf("pFinal = %f\n", pFinal);
	p_rate = (pFinal - pInitial) / dt;

}

/* ----------------------------------------------------------------------
 Tait EOS based on current density vs. reference density.

 input: (1) reference sound speed
 (2) equilibrium mass density
 (3) current mass density

 output:(1) pressure
 (2) current speed of sound
 ------------------------------------------------------------------------- */
void TaitEOS_density(const double exponent, const double c0_reference, const double rho_reference, const double rho_current,
		double &pressure, double &sound_speed) {

	double B = rho_reference * c0_reference * c0_reference / exponent;
	double tmp = pow(rho_current / rho_reference, exponent);
	pressure = B * (tmp - 1.0);
	double bulk_modulus = B * tmp * exponent; // computed as rho * d(pressure)/d(rho)
	sound_speed = sqrt(bulk_modulus / rho_current);

//	if (fabs(pressure) > 0.01) {
//		printf("tmp = %f, press=%f, K=%f\n", tmp, pressure, bulk_modulus);
//	}

}

/* ----------------------------------------------------------------------
 perfect gas EOS
 input: gamma -- adiabatic index (ratio of specific heats)
 J -- determinant of deformation gradient
 volume0 -- reference configuration volume of particle
 energy -- energy of particle
 pInitial -- initial pressure of the particle
 d -- isotropic part of the strain rate tensor,
 dt -- time-step size

 output: final pressure pFinal, pressure rate p_rate
 ------------------------------------------------------------------------- */
void PerfectGasEOS(const double gamma, const double vol, const double mass, const double energy, double &pFinal, double &c0) {

	/*
	 * perfect gas EOS is p = (gamma - 1) rho e
	 */

	if (energy > 0.0) {

		pFinal = (1.0 - gamma) * energy / vol;
//printf("gamma = %f, vol%f, e=%g ==> p=%g\n", gamma, vol, energy, *pFinal__/1.0e-9);

		c0 = sqrt((gamma - 1.0) * energy / mass);

	} else {
		pFinal = c0 = 0.0;
	}

}

/* ----------------------------------------------------------------------
 linear strength model for use with linear elasticity
 input: lambda, mu : Lame parameters
 input: sigmaInitial_dev, d_dev: initial stress deviator, deviatoric part of the strain rate tensor
 input: dt: time-step
 output:  sigmaFinal_dev, sigmaFinal_dev_rate__: final stress deviator and its rate.
 ------------------------------------------------------------------------- */
void LinearStrength(const double mu, const Matrix3d sigmaInitial_dev, const Matrix3d d_dev, const double dt,
		Matrix3d &sigmaFinal_dev__, Matrix3d &sigma_dev_rate__) {

	/*
	 * deviatoric rate of unrotated stress
	 */
	sigma_dev_rate__ = 2.0 * mu * d_dev;

	/*
	 * elastic update to the deviatoric stress
	 */
	sigmaFinal_dev__ = sigmaInitial_dev + dt * sigma_dev_rate__;
}

/* ----------------------------------------------------------------------
 linear strength model for use with linear elasticity
 input: lambda, mu : Lame parameters
 input: F: deformation gradient
 output:  total stress tensor, deviator + pressure
 ------------------------------------------------------------------------- */
//void PairTlsph::LinearStrengthDefgrad(double lambda, double mu, Matrix3d F, Matrix3d *T) {
//	Matrix3d E, PK2, eye, sigma, S, tau;
//
//	eye.setIdentity();
//
//	E = 0.5 * (F * F.transpose() - eye); // strain measure E = 0.5 * (B - I) = 0.5 * (F * F^T - I)
//	tau = lambda * E.trace() * eye + 2.0 * mu * E; // Kirchhoff stress, work conjugate to above strain
//	sigma = tau / F.determinant(); // convert Kirchhoff stress to Cauchy stress
//
////printf("l=%f, mu=%f, sigma xy = %f\n", lambda, mu, sigma(0,1));
//
////    E = 0.5 * (F.transpose() * F - eye); // Green-Lagrange Strain E = 0.5 * (C - I)
////    S = lambda * E.trace() * eye + 2.0 * mu * Deviator(E); // PK2 stress
////    tau = F * S * F.transpose(); // convert PK2 to Kirchhoff stress
////    sigma = tau / F.determinant();
//
//	//*T = sigma;
//
//	/*
//	 * neo-hookean model due to Bonet
//	 */
////    lambda = mu = 100.0;
////    // left Cauchy-Green Tensor, b = F.F^T
//	double J = F.determinant();
//	double logJ = log(J);
//	Matrix3d b;
//	b = F * F.transpose();
//
//	sigma = (mu / J) * (b - eye) + (lambda / J) * logJ * eye;
//	*T = sigma;
//}
/* ----------------------------------------------------------------------
 linear strength model for use with linear elasticity
 input: lambda, mu : Lame parameters
 input: sigmaInitial_dev, d_dev: initial stress deviator, deviatoric part of the strain rate tensor
 input: dt: time-step
 output:  sigmaFinal_dev, sigmaFinal_dev_rate__: final stress deviator and its rate.
 ------------------------------------------------------------------------- */
void LinearPlasticStrength(const double G, double yieldStress, const Matrix3d sigmaInitial_dev, const Matrix3d d_dev,
			   const double dt, Matrix3d &sigmaFinal_dev__, Matrix3d &sigma_dev_rate__, double &plastic_strain_increment, const double damage) {

	Matrix3d sigmaTrial_dev, dev_rate;
	double J2;
	double Gd = G * (1.0 - damage);
	yieldStress *= (1.0 - damage);
	/*
	 * deviatoric rate of unrotated stress
	 */
	dev_rate = 2.0 * Gd * d_dev;

	/*
	 * perform a trial elastic update to the deviatoric stress
	 */
	sigmaTrial_dev = sigmaInitial_dev + dt * dev_rate; // increment stress deviator using deviatoric rate

	/*
	 * check yield condition
	 */
	J2 = sqrt(3. / 2.) * sigmaTrial_dev.norm();

	if (J2 < yieldStress) {
		/*
		 * no yielding has occured.
		 * final deviatoric stress is trial deviatoric stress
		 */
		sigma_dev_rate__ = dev_rate;
		sigmaFinal_dev__ = sigmaTrial_dev;
		plastic_strain_increment = 0.0;
		//printf("no yield\n");

	} else {
		//printf("yiedl\n");
		/*
		 * yielding has occured
		 */

		plastic_strain_increment = (J2 - yieldStress) / (3.0 * Gd);
		/*
		 * new deviatoric stress:
		 * obtain by scaling the trial stress deviator
		 */
		sigmaFinal_dev__ = (yieldStress / J2) * sigmaTrial_dev;

		/*
		 * new deviatoric stress rate
		 */
		sigma_dev_rate__ = sigmaFinal_dev__ - sigmaInitial_dev;
		//printf("yielding has occured.\n");
	}
	
}

/* ----------------------------------------------------------------------
 Johnson Cook Material Strength model
 input:
 G : shear modulus
 cp : heat capacity
 espec : energy / mass
 A : initial yield stress under quasi-static / room temperature conditions
 B : proportionality factor for plastic strain dependency
 a : exponent for plastic strain dpendency
 C : proportionality factor for logarithmic plastic strain rate dependency
 epdot0 : dimensionality factor for plastic strain rate dependency
 T : current temperature
 T0 : reference (room) temperature
 Tmelt : melting temperature
 input: sigmaInitial_dev, d_dev: initial stress deviator, deviatoric part of the strain rate tensor
 input: dt: time-step
 output:  sigmaFinal_dev, sigmaFinal_dev_rate__: final stress deviator and its rate.
 ------------------------------------------------------------------------- */
void JohnsonCookStrength(const double G, const double cp, const double espec, const double A, const double B, const double a,
			 const double C, const double epdot0, const double T0, const double Tmelt, const double M, const double dt, const double ep,
			 const double epdot, const Matrix3d sigmaInitial_dev, const Matrix3d d_dev, Matrix3d &sigmaFinal_dev__,
			 Matrix3d &sigma_dev_rate__, double &plastic_strain_increment, const double damage) {

	double yieldStress;

	double TH = espec / (cp * (Tmelt - T0));
	TH = MAX(TH, 0.0);
	double epdot_ratio = epdot / epdot0;
	epdot_ratio = MAX(epdot_ratio, 1.0);
	//printf("current temperature delta is %f, TH=%f\n", deltaT, TH);
	
	yieldStress = (A + B * pow(ep, a)) * pow(1.0 + epdot_ratio, C); // * (1.0 - pow(TH, M));
	if (isnan(yieldStress)){
	  printf("yieldStress = %f, ep = %f, epdot_ratio = %f, epdot = %f, epdot0 = %f\n", yieldStress,ep,epdot_ratio, epdot, epdot0);
	}
	
	LinearPlasticStrength(G, yieldStress, sigmaInitial_dev, d_dev, dt, sigmaFinal_dev__, sigma_dev_rate__, plastic_strain_increment, damage);
}

/* ----------------------------------------------------------------------
 Gurson - Tvergaard - Needleman (GTN) Material Strength model
 input:
 G : shear modulus
 Q1, Q2: two model parameters
 damage
 
 input: sigmaInitial_dev, d_dev: initial stress deviator, deviatoric part of the strain rate tensor
 input: dt: time-step
 output:  sigmaFinal_dev, sigmaFinal_dev_rate__: final stress deviator and its rate.
 ------------------------------------------------------------------------- */
double GTNStrength(const double G, FlowStress flowstress, const double Q1, const double Q2,
		   const double fcr, const double fF, const double FN, const double inverse_sN, const double epsN, const double Komega,
		   const double dt, const double damage, const double ep, const Matrix3d sigmaInitial_dev, const Matrix3d d_dev,
		   const double pInitial, double &pFinal, Matrix3d &sigmaFinal_dev__, Matrix3d &sigma_dev_rate__,
		   double &plastic_strain_increment, const int tag) {

  Matrix3d sigmaTrial_dev, dev_rate, plastic_strain_increment_array;
  double J2, yieldStress_undamaged, damage_increment;
  double Gd = G * (1.0 - damage);
  double f;
  double fcrQ1 = fcr*Q1;
  double x;
  double F, Q2triaxx, Q1f, Q1fSq, Q2triax, cosh_Q2triaxx;

  if (damage <= fcrQ1) f = damage/Q1;
  else {
    f = fcr + (damage - fcrQ1)/(1 - fcrQ1) * (fF - fcr);
  }

  damage_increment = 0.0;

  /*
   * deviatoric rate of unrotated stress
   */
  dev_rate = 2.0 * Gd * d_dev;

  /*
   * perform a trial elastic update to the deviatoric stress
   */
  sigma_dev_rate__ = dt * dev_rate; // I am using this variable to store dt * dev_rate
  sigmaTrial_dev = sigmaInitial_dev + sigma_dev_rate__; // increment stress deviator using deviatoric rate

  /*
   * check yield condition
   */
  J2 = sqrt(3. / 2.) * sigmaTrial_dev.norm();
  yieldStress_undamaged = flowstress.evaluate(ep);

  // determine stress triaxiality
  double triax = 0.0;
  if (pFinal != 0.0 && J2 != 0.0) {
    triax = pFinal / (J2 + 0.01 * fabs(pFinal)); // have softening in denominator to avoid divison by zero
  }

  Q1f = damage; // In reality it is Q1 * void function (f*) which is equal to damage
  Q1fSq = damage * damage;
  x = J2/yieldStress_undamaged;
  Q2triax = 1.5 * Q2 * triax;
  F = x*x + 2 * damage * cosh(Q2triax * x) - (1 + Q1fSq);

  if (F <= 0.0) {
    /*
     * no yielding has occured.
     * final deviatoric stress is trial deviatoric stress
     */
    sigma_dev_rate__ = dev_rate;
    sigmaFinal_dev__ = sigmaTrial_dev;
    plastic_strain_increment = 0.0;
    //printf("no yield F = %.10e\n", F);

  } else {
    //printf("yiedl\n");
    /*
     * yielding has occured
     */
    /*
     * NEWTON - RAPHSON METHOD TO DETERMINE THE YIELD STRESS:
     */

    double Q1fQ2triax = damage * Q2triax;

    x = MIN(1.0, x);
    double dx = 1.0; // dx = x_{n+1} - x_{n} initiated at a value higher than the accepted error margin.
    double error = 0.1;
    double Fprime, Q2triaxx;

    int i = 0;
    while ((dx > error) || (dx < -error) || (J2 - x*yieldStress_undamaged) < 0.0) {
      Q2triaxx = Q2triax * x;
      F = x*x + 2 * Q1f * cosh(Q2triaxx) - (1 + Q1fSq);
      Fprime = 2 * (x + Q1fQ2triax * sinh(Q2triaxx));

      dx = -F/Fprime;
      x += dx;
      if (i>5) printf("Loop1: %d - %d - F = %.10e, Fprime = %.10e, x = %f, J2 = %.10e, yieldStress_undamaged = %.10e, ep = %.10e, Q1f = %.10e, triax = %.10e, dx = %.10e, damage = %10.e\n", tag, i, F, Fprime, x, J2, yieldStress_undamaged, ep, Q1f, triax, dx, damage);
      i++;
    }

    /*
     * NEWTON - RAPHSON METHOD TO DETERMINE THE MATRIX PLASTIC STRAIN INCREMENT:
     */
    int j = 0;
    double yieldStress, plastic_strain_increment_old, delta_plastic_strain_increment, alpha, beta, sinh_Q2triaxx, inverse_x, J3, omega;

    dx = 1.0;

    yieldStress = x * yieldStress_undamaged;
    sinh_Q2triaxx = sinh(Q2triaxx);
    beta = (x + 1.5 * Q1fQ2triax * sinh_Q2triaxx) / (3.0 * Gd * (1 - f));
    plastic_strain_increment = (J2 - yieldStress) * beta; // This assumes all strain increase is plastic!
    int output = 0;
    if (plastic_strain_increment < 0.0) {
      output = 1;
      printf("%d - %d - plastic_strain_increment = %.10e, dx = %f, J2 = %.10e, yieldStress = %.10e, ep = %.10e, F = %.10e, x = %.10e, f = %.10e, triax = %.10e\n", tag, j, plastic_strain_increment, dx, J2, yieldStress, ep, F, x, f, triax);
      sigma_dev_rate__ = dev_rate;
      sigmaFinal_dev__ = sigmaTrial_dev;
      plastic_strain_increment = 0.0;
    } else {

      plastic_strain_increment_old = plastic_strain_increment + 1.0; // arbitrary value
      while(abs(dx) > 1e-12 && abs(dx/plastic_strain_increment) > error && j<5) {
	j++;
	yieldStress = x * flowstress.evaluate(ep + plastic_strain_increment);
	F = plastic_strain_increment - (J2 - yieldStress) * beta;
	Fprime = 1.0 + beta * x * flowstress.evaluate_derivative(ep + plastic_strain_increment);
	plastic_strain_increment_old = plastic_strain_increment;
	plastic_strain_increment -= F/Fprime;
	if (plastic_strain_increment < 0.0) {
	  plastic_strain_increment = 0.5*plastic_strain_increment_old;
	}
	dx = plastic_strain_increment - plastic_strain_increment_old;
	if (j>5) output = 1;
	if (output == 1) printf("Loop2: %d - %d - plastic_strain_increment = %.10e, dx = %.10e, J2 = %f, yieldStress = %f, ep = %.10e, F = %.10e, Fprime = %.10e\n", tag, j, plastic_strain_increment, dx, J2, yieldStress, ep, F, Fprime);
      }
      yieldStress_undamaged = flowstress.evaluate(ep + plastic_strain_increment);
      yieldStress = x * yieldStress_undamaged;
      sigmaFinal_dev__ = (yieldStress/J2) * sigmaTrial_dev;

      /*
       * new deviatoric stress rate
       */
      sigma_dev_rate__ = sigmaFinal_dev__ - sigmaInitial_dev;
      //printf("yielding has occured.\n");


      /*
       * Update f
       */
      inverse_x = 1.0/x;
      alpha = 1.5 * Q1f * Q2 * sinh_Q2triaxx * inverse_x;

      if (Komega != 0) {
	J3 = sigmaTrial_dev.determinant();
	omega = 1 - 182.25 * J3 * J3/(J2 * J2 * J2 * J2 * J2 * J2);
	if (omega < 0.0) omega = 0;
	else if (omega > 1.0) omega = 1.0;
      } else omega = 0;

      pFinal -= plastic_strain_increment * triax/beta;

      // Damage growth:
      f += (1-f) * inverse_x * (max(0.0, alpha) * (1-f) + f * Komega * omega) * plastic_strain_increment / (alpha*triax + 1);

      //Damage nucleation:
      if (FN != 0) {
	if (inverse_sN == 0)
	  f += FN * plastic_strain_increment ; // 0.39894228 = 1/sqrt(2*PI)
	else
	  if (f < FN)
	    f += FN * 0.39894228 * inverse_sN * exp(-0.5 * inverse_sN * inverse_sN * (ep - epsN) * (ep - epsN)) * plastic_strain_increment; // 0.39894228 = 1/sqrt(2*PI)
      }
      if (f <= fcr) damage_increment = Q1*f - damage;
      else {
	damage_increment = fcrQ1 + (1.0 - fcrQ1)/(fF - fcr)*(f - fcr) - damage;
      }
      //if (tag == 34937) printf("plastic_strain_increment = %.10e, alpha = %.10e, x = %.10e, triax = %.10e, Komega = %.10e, omega = %.10e, f = %.10e, f_increment = %.10e, fN_increment = %.10e, damage_increment = %.10e, damage = %.10e\n", plastic_strain_increment, alpha, x, triax, Komega, omega, f, (1-f) * inverse_x * (max(0.0, alpha) * (1-f)/(alpha*triax + 1) + f * Komega * omega) * plastic_strain_increment, FN * 0.39894228 * 1/sN * exp(-0.5 * 1/sN * (ep - epsN)) * plastic_strain_increment, damage_increment, damage);
    }
  }
  return damage_increment;
}

/* ----------------------------------------------------------------------
 isotropic maximum strain damage model
 input:
 current strain
 maximum value of allowed principal strain

 output:
 return value is true if any eigenvalue of the current strain exceeds the allowed principal strain

 ------------------------------------------------------------------------- */

bool IsotropicMaxStrainDamage(const Matrix3d E, const double maxStrain) {

	/*
	 * compute Eigenvalues of strain matrix
	 */
	SelfAdjointEigenSolver < Matrix3d > es;
	es.compute(E); // compute eigenvalue and eigenvectors of strain

	double max_eigenvalue = es.eigenvalues().maxCoeff();

	if (max_eigenvalue > maxStrain) {
		return true;
	} else {
		return false;
	}
}

/* ----------------------------------------------------------------------
 isotropic maximum stress damage model
 input:
 current stress
 maximum value of allowed principal stress

 output:
 return value is true if any eigenvalue of the current stress exceeds the allowed principal stress

 ------------------------------------------------------------------------- */

bool IsotropicMaxStressDamage(const Matrix3d S, const double maxStress) {

	/*
	 * compute Eigenvalues of strain matrix
	 */
	SelfAdjointEigenSolver < Matrix3d > es;
	es.compute(S); // compute eigenvalue and eigenvectors of strain

	double max_eigenvalue = es.eigenvalues().maxCoeff();

	if (max_eigenvalue > maxStress) {
		return true;
	} else {
		return false;
	}
}

/* ----------------------------------------------------------------------
 Johnson-Cook failure model
 input:


 output:


 ------------------------------------------------------------------------- */

double JohnsonCookDamageIncrement(const double p, const Matrix3d Sdev, const double d1, const double d2, const double d3,
                  const double d4, const double epdot0, const double epdot, const double plastic_strain_increment) {
    // 计算 von-Mises 等效应力
    double vm = sqrt(3. / 2.) * Sdev.norm(); // von-Mises equivalent stress
    if (vm < 0.0) {
        cout << "this is sdev " << endl << Sdev << endl;
        printf("vm=%f < 0.0, surely must be an error\n", vm);
        exit(1);
    }

    // 计算应力三轴性
    double triax = 0.0;
    if (p != 0.0 && vm != 0.0) {
      triax = -p / (vm + 0.01 * fabs(p)); // 为避免除以零，分母加上软化因子0.01
    }
    if (triax > 3.0) {
      triax = 3.0;
    }

    // Johnson-Cook损伤模型中的失效应变，与应力三轴性有关
    double jc_failure_strain = d1 + d2 * exp(d3 * triax);

    // 如果定义了参数 d4，并且当前塑性应变率超过参考应变率，则考虑应变速率依赖性
    if (d4 != 0.0) {
        if (epdot > epdot0) {
          double epdot_ratio = epdot / epdot0;
          jc_failure_strain *= pow(1.0 + epdot_ratio, d4);
        }
    }
    
    // 返回塑性应变增量与Johnson-Cook失效应变的比值作为损伤增量
    return plastic_strain_increment / jc_failure_strain;
}


/* ----------------------------------------------------------------------
 Gurson-Tvergaard-Needleman damage evolution model
 input:
 An and Komega: parameters
 equivalent plastic strain increment
 Finc
 dt
 damage

 output:
 damage increment

 ------------------------------------------------------------------------- */

double GTNDamageIncrement(const double Q1, const double Q2, const double An, const double Komega, const double pressure, const Matrix3d Sdev, const Matrix3d stress, const double plastic_strain_increment, const double damage, const double fcr, const double yieldstress_undamaged) { // Following K. Nahshon, J.W. Hutchinson / European Journal of Mechanics A/Solids 27 (2008) 1–17
  // 如果损伤达到或超过1.0，返回0.0（没有损伤增量）
  if (damage >= 1.0) return 0.0;
  // 如果塑性应变增量为0，返回0.0（没有损伤增量）
  if (plastic_strain_increment == 0.0) return 0.0;
  // 初始化空洞成核增长率增量
  double fn_increment = 0;
  // 如果An不为0，计算空洞成核增长率增量
  if (An != 0) fn_increment = An * plastic_strain_increment; // rate of void nucleation
  // 如果当前损伤值为0，仅返回成核增长率增量
  if (damage == 0.0) return fn_increment;
  // 初始化空洞增长率增量
  double fs_increment = 0;
  // 计算当前空洞体积分数f
  double f = damage * fcr;
  // 定义 von-Mises 等效应力 vm，逆向屈服应力 inverse_sM，第三不变量 J3 和 omega
  double vm, inverse_sM, J3, omega;
  double lambda_increment, tmp1, sinh_tmp1;
  // 计算 von-Mises 等效应力
  vm = sqrt(3. / 2.) * Sdev.norm(); // von-Mises equivalent stress

  if (vm < 0.0) {
    cout << "this is sdev " << endl << Sdev << endl;
    printf("vm=%f < 0.0, surely must be an error\n", vm);
    exit(1);
  }
  // 如果 von-Mises 等效应力为0，返回0.0（没有损伤增量）
  if ( vm == 0.0 ) return 0.0;
  // 计算逆向屈服应力
  inverse_sM = 1.0/yieldstress_undamaged;
  // 计算 Sdev 的第三不变量
  J3 = Sdev.determinant();
  //printf("vm = %f, yieldstress_undamaged = %f, J3 = %f\n", vm, yieldstress_undamaged, J3);
  // 计算 omega 参数（用于描述材料各向异性影响）
  omega = 1 - 182.25 * J3 * J3/(vm * vm * vm * vm * vm * vm);
  // 对 omega 值进行约束，如果小于0则设为0，大于1则设为1
  if (omega < 0.0) {
    // printf("omega=%.10e < 0.0, surely must be an error\n", omega);
    // cout << "vm = " << vm << "\t";
    // cout << "J3 = " << J3 << "\t";
    // cout << "J3 * J3/(vm * vm * vm * vm * vm * vm) = " << J3 * J3/(vm * vm * vm * vm * vm * vm) << endl;
    // cout << "Here is S:" << endl << Sdev << endl;
    omega = 0;
  }
  else if (omega > 1.0) {
    // printf("omega=%.10e > 1.0, surely must be an error\n", omega);
    // cout << "vm = " << vm << "\t";
    // cout << "J3 = " << J3 << "\t";
    // cout << "J3 * J3/(vm * vm * vm * vm * vm * vm) = " << J3 * J3/(vm * vm * vm * vm * vm * vm) << endl;
    // cout << "Here is S:" << endl << Sdev << endl;
    omega = 1.0;
  }
  // 计算 tmp1 和 sinh_tmp1，用于后续的计算
  tmp1 = -1.5 * Q2 * pressure * inverse_sM;
  sinh_tmp1 = sinh(tmp1);
  // 计算增量塑性乘子 lambda_increment
  lambda_increment = 0.5 * yieldstress_undamaged * plastic_strain_increment * (1 - f) 
  / (vm * vm * inverse_sM * inverse_sM + Q1 * f * tmp1 * sinh_tmp1);
  // 计算空洞增长率增量
  fs_increment = lambda_increment * f * inverse_sM * ((1 - f) * 3 * Q1 * Q2 * sinh_tmp1 
  + Komega * omega * 2 * vm * inverse_sM);
  // 如果计算结果出现 NaN，则输出调试信息
  if (isnan(fs_increment) || isnan(-fs_increment)) {
    printf("GTN f increment: %.10e\n", fs_increment);
    cout << "vm = " << vm << "\t";
    cout << "yieldstress_undamaged = " << yieldstress_undamaged << "\t";
    cout << "tmp1 = " << tmp1 << endl;
    cout << "f = " << f << endl;
    cout << "omega = " << omega << endl;
    cout << "F = " << vm * vm * inverse_sM * inverse_sM + 2 * Q1 * f * cosh(tmp1)
    - (1 + Q1 * Q1 * f * f) << endl;
    cout << "plastic_strain_increment = " << plastic_strain_increment << endl;
  }
  // 如果空洞增长率增量小于0，则设为0
  if (fs_increment < 0.0) fs_increment = 0.0;
  // 总的空洞体积分数增量为成核增长率增量和空洞增长率增量之和
  double f_increment = fn_increment + fs_increment;
  // 如果计算结果出现 NaN，则输出调试信息
  if (isnan(f_increment) || isnan(-f_increment)){
    cout << "fs_increment = " << fs_increment << "\t" << "fn_increment = " << fn_increment << endl;
  }
  // 返回归一化的空洞体积分数增量
  return f_increment / fcr;
}

/* ----------------------------------------------------------------------
 Cockcroft-Latham failure model
 input:


 output:


 ------------------------------------------------------------------------- */
// 定义函数来计算Cockcroft-Latham损伤增量
double CockcroftLathamDamageIncrement(const Matrix3d S, const double W, const double plastic_strain_increment) {
// 检查塑性应变增量是否大于0，如果不大于0，直接返回0
  if (plastic_strain_increment > 0.0) {
    // Principal stress:
    // 创建EigenSolver对象用于计算应力矩阵的特征值和特征向量
    EigenSolver<Matrix3d> ES;
    //计算特征值和特征向量
    ES.compute(S); // Compute eigenvalues and eigenvectors
    //获取实部特征值向量
    Vector3d Lambda = ES.eigenvalues().real(); // Vector of eigenvalues (real).
    //获取特征值绝对值的向量
    Vector3d Lambda_abs = ES.eigenvalues().real().cwiseAbs(); // Vector of the absolute value of eigenvalues (real).
    //获取特征值中的最大值
    double Lambda_max = Lambda.maxCoeff();
    //检查最大特征值是否为正
    if (Lambda_max == Lambda_abs.maxCoeff()) {
      // The principal stress is positive如果最大特征值为正，则计算并返回损伤增量
      //损伤增量 = 最大主应力 * 塑性应变增量 / 材料强度
      //printf("Lambda = [%.10e %.10e %.10e], Lambda_max = %.10e, plastic_strain_increment / W = %.10e, damage_increment = %.10e\n", Lambda[0], Lambda[1], Lambda[2], Lambda_max, plastic_strain_increment / W,Lambda_max * plastic_strain_increment / W);
      return Lambda_max * plastic_strain_increment / W;
    } else {
      // 如果最大特征值为负，则返回0，因为负的主应力不产生损伤
      //printf("Lambda = [%.10e %.10e %.10e], damage_increment = 0\n", Lambda[0], Lambda[1], Lambda[2]);
      // The principal stress is negative
      return 0;
    }
  } else {
    // 如果塑性应变增量不大于0，则直接返回0
    return 0;
  }
}
