// copy this macro to every file you use this in
#ifdef GTEST
#define CUDA_DEV
#define __device__
#include <algorithm>
#include <assert.h>
#include <cmath>
#include <cstdio>
#include <cstring>
using namespace std;
#elif PYTHON_TEST
#define CUDA_DEV __global__
#else
#define CUDA_DEV __device__
#endif

#ifndef MATH_HELPERS
#define MATH_HELPERS

__device__ int binomial_coefficient(int n, int k) {
    if (n < k) {
        return 0;
    }
    int result = 1;
    if (k > n - k) {
        k = n - k;
    }
    for (int i = 0; i < k; ++i) {
        result *= (n - i);
        result /= (i + 1);
    }
    return result;
}

// poz is a helper function for the afterwards defined pchisq
// It calculates the probability of a normal z score
// Implementation of the following function is adapted from Gary Perlman,
// source retrievable at
// https://www.netlib.org/a/perlman
//    Module:       z.c
//    Purpose:      compute approximations to normal z distribution probabilities
//    Programmer:   Gary Perlman
//    Organization: Wang Institute, Tyngsboro, MA 01879
//    Tester:       compile with -DZTEST to include main program
//    Copyright:    none
//    Tabstops:     4
//
// z: z-score
#define Z_MAX 6.0                               /* maximum meaningful z value */
#define LOG_SQRT_PI 0.5723649429247000870717135 /* log (sqrt (pi)) */
#define I_SQRT_PI 0.5641895835477562869480795   /* 1 / sqrt (pi) */
#define BIGX 20.0                               /* max value to represent exp (x) */
#define ex(x) (((x) < -BIGX) ? 0.0 : exp(x))
__device__ double poz(double z) {
    double y;
    double x;
    double w;

    if (z == 0.0) {
        x = 0.0;
    } else {
        y = 0.5 * fabs(z);
        if (y >= (Z_MAX * 0.5)) {
            x = 1.0;
        } else if (y < 1.0) {
            w = y * y;
            x = ((((((((0.000124818987 * w - 0.001075204047) * w + 0.005198775019) * w - 0.019198292004) * w +
                     0.059054035642) *
                        w -
                    0.151968751364) *
                       w +
                   0.319152932694) *
                      w -
                  0.531923007300) *
                     w +
                 0.797884560593) *
                y * 2.0;
        } else {
            y -= 2.0;
            x = (((((((((((((-0.000045255659 * y + 0.000152529290) * y - 0.000019538132) * y - 0.000676904986) * y +
                          0.001390604284) *
                             y -
                         0.000794620820) *
                            y -
                        0.002034254874) *
                           y +
                       0.006549791214) *
                          y -
                      0.010557625006) *
                         y +
                     0.011630447319) *
                        y -
                    0.009279453341) *
                       y +
                   0.005353579108) *
                      y -
                  0.002141268741) *
                     y +
                 0.000535310849) *
                    y +
                0.999936657524;
        }
    }
    return (z > 0.0 ? ((x + 1.0) * 0.5) : ((1.0 - x) * 0.5));
}

// Calculates the probability of a given chi-squared value
// x: obtained chi-squared value
// df: degrees of freedom

// Implementation of the following function
// is adapted from Gary Perlman, source retrievable at
// https://www.netlib.org/a/perlman
//    Module:       chisq.c
//    Purpose:      compute approximations to chisquare distribution probabilities
//    Contents:     pochisq(), critchi()
//    Uses:         poz() in z.c (Algorithm 209)
//    Programmer:   Gary Perlman
//    Organization: Wang Institute, Tyngsboro, MA 01879
//    Tester:       compile with -DCHISQTEST to include main program
//    Copyright:    none
//    Tabstops:     4
// which itself adapted from:
//            Hill, I. D. and Pike, M. C.  Algorithm 299
//            Collected Algorithms for the CACM 1967 p. 243
//    Updated for rounding errors based on remark in
//            ACM TOMS June 1985, page 185
__device__ double pchisq(double x, int df) {
    double a;
    double y = 0.0;
    double s;

    double e;
    double c;
    double z;

    bool even; /* true if df is an even number */

    if (x <= 0.0 || df < 1) {
        return (1.0);
    }

    a = 0.5 * x;
    even = (2 * (df / 2)) == df;
    if (df > 1) {
        y = ex(-a);
    }
    s = (even ? y : (2.0 * poz(-sqrt(x))));

    if (df <= 2) {
        return (s);
    }

    x = 0.5 * (df - 1.0);
    z = (even ? 1.0 : 0.5);
    if (a > BIGX) {
        e = (even ? 0.0 : LOG_SQRT_PI);
        c = log(a);
        while (z <= x) {
            e = log(z) + e;
            s += ex(c * z - a - e);
            z += 1.0;
        }
        return (s);
    }

    e = (even ? 1.0 : (I_SQRT_PI / sqrt(a)));
    c = 0.0;
    while (z <= x) {
        e = e * (a / z);
        c = c + e;
        z += 1.0;
    }
    return (c * y + s);
}

#endif
