
#pragma once

#include "grid_2d.hpp"
#include "tridiagonal_matrix.hpp"
#include <span>

namespace aks::math {

// Solve the equation of the form
// du/dt = D(x,t) * d2u/dx2 + S(x,t) * du/dx + R(x,t) * u + F(x,t)
// with boundary conditions and initial conditions
// u(0,t) = f(t)
// u(L,t) = g(t)
// u(x,0) = h(x)
// using the Crank-Nicholson method
template <typename Policy_> struct crank_nicholson_problem {
  using policy = Policy_;
  policy policy_;
  explicit crank_nicholson_problem(const policy &p) : policy_(p) {}
  std::span<double const> xgrid() const { return policy_.xgrid(); }
  std::span<double const> tgrid() const { return policy_.tgrid(); }
  double D(double x, double t) const { return policy_.D(x, t); }
  double S(double x, double t) const { return policy_.S(x, t); }
  double R(double x, double t) const { return policy_.R(x, t); }
  double F(double x, double t) const { return policy_.F(x, t); }
  double h(double x) const { return policy_.h(x); }
  double f(double t) const { return policy_.f(t); }
  double g(double t) const { return policy_.g(t); }
  void process(std::span<double> data) const { return policy_.process(data); }
};

template <typename Problem>
void solve_crank_nicholson(const Problem &problem, grid_2d<double> &u) {

  auto const xgrid = problem.xgrid();
  auto const tgrid = problem.tgrid();
  auto const x_sz = xgrid.size();
  auto const t_sz = tgrid.size();

  if (u.size<0>() != t_sz || u.size<1>() != x_sz) {
    throw std::invalid_argument("u grid size does not match problem grid size");
  }

  // create coefficient grids
  grid_2d<double> D_coeff(t_sz, x_sz, u.get_allocator());
  grid_2d<double> S_coeff(t_sz, x_sz, u.get_allocator());
  grid_2d<double> R_coeff(t_sz, x_sz, u.get_allocator());
  grid_2d<double> F_coeff(t_sz, x_sz, u.get_allocator());

  // create matrices A and B and temporary vector Bu
  aks::tridiagonalmatrix<double> A(x_sz, u.get_allocator());
  aks::tridiagonalmatrix<double> B(x_sz, u.get_allocator());
  std::pmr::vector<double> Bu(x_sz, u.get_allocator());

  // Single persistent parallel region for entire computation
#ifdef AKS_ENABLE_OPENMP
#pragma omp parallel
  {
#pragma omp for collapse(2) schedule(static) nowait
#endif
    for (size_t t = 0; t < t_sz; ++t) {
      for (size_t x = 0; x < x_sz; ++x) {
        auto const x_val = xgrid[x];
        auto const t_val = tgrid[t];
        D_coeff(t, x) = problem.D(x_val, t_val);
        S_coeff(t, x) = problem.S(x_val, t_val);
        R_coeff(t, x) = problem.R(x_val, t_val);
        F_coeff(t, x) = problem.F(x_val, t_val);
      }
    }

#ifdef AKS_ENABLE_OPENMP
#pragma omp for schedule(static) nowait
#endif
    for (size_t x = 0; x < x_sz; ++x) {
      u(0, x) = problem.h(xgrid[x]);
    }

#ifdef AKS_ENABLE_OPENMP
#pragma omp for schedule(static)
#endif
    for (size_t t = 0; t < t_sz; ++t) {
      auto const t_val = tgrid[t];
      u(t, 0) = problem.f(t_val);
      u(t, x_sz - 1) = problem.g(t_val);
    }

    // Time stepping loop
    for (size_t t_idx = 0; t_idx < t_sz - 1; ++t_idx) {
      double const dt = tgrid[t_idx + 1] - tgrid[t_idx];

      std::span<double const> u_row = u.row_span(t_idx);
      std::span<double> u_row_next = u.row_span(t_idx + 1);

      std::span<double const> D_row = D_coeff.row_span(t_idx);
      std::span<double const> S_row = S_coeff.row_span(t_idx);
      std::span<double const> R_row = R_coeff.row_span(t_idx);
      std::span<double const> F_row = F_coeff.row_span(t_idx);
      std::span<double const> D_row_next = D_coeff.row_span(t_idx + 1);
      std::span<double const> S_row_next = S_coeff.row_span(t_idx + 1);
      std::span<double const> R_row_next = R_coeff.row_span(t_idx + 1);
      std::span<double const> F_row_next = F_coeff.row_span(t_idx + 1);

      // Implementation of Crank-Nicholson
      // Build matrices A and B for the scheme: A * u(t+1) = B * u(t) + rhs
      // Use u_row_next as rhs storage (it already has boundary conditions)

#ifdef AKS_ENABLE_OPENMP
#pragma omp for schedule(static) nowait
#endif
      for (size_t x_idx = 1; x_idx < x_sz - 1; ++x_idx) {
        double dx_left = xgrid[x_idx] - xgrid[x_idx - 1];
        double dx_right = xgrid[x_idx + 1] - xgrid[x_idx];
        double dx_avg = 0.5 * (dx_left + dx_right);

        // Average coefficients for Crank-Nicholson (theta = 0.5)
        double D_avg = 0.5 * (D_row[x_idx] + D_row_next[x_idx]);
        double S_avg = 0.5 * (S_row[x_idx] + S_row_next[x_idx]);
        double R_avg = 0.5 * (R_row[x_idx] + R_row_next[x_idx]);
        double F_avg = 0.5 * (F_row[x_idx] + F_row_next[x_idx]);

        // Second derivative coefficients (d2u/dx2)
        double alpha = D_avg / (dx_left * dx_avg);
        double gamma = D_avg / (dx_right * dx_avg);
        double beta = -(alpha + gamma);

        // First derivative coefficients (du/dx, central difference)
        double s_left = -S_avg / (2.0 * dx_avg);
        double s_right = S_avg / (2.0 * dx_avg);

        // Matrix A (implicit part, left-hand side)
        A(x_idx, x_idx - 1) = -0.5 * dt * (alpha + s_left);
        A(x_idx, x_idx) = 1.0 - 0.5 * dt * (beta + R_avg);
        if (x_idx < x_sz - 1) {
          A(x_idx, x_idx + 1) = -0.5 * dt * (gamma + s_right);
        }

        // Matrix B (explicit part, right-hand side)
        B(x_idx, x_idx - 1) = 0.5 * dt * (alpha + s_left);
        B(x_idx, x_idx) = 1.0 + 0.5 * dt * (beta + R_avg);
        if (x_idx < x_sz - 1) {
          B(x_idx, x_idx + 1) = 0.5 * dt * (gamma + s_right);
        }

        // Add source term to rhs (which is u_row_next)
        u_row_next[x_idx] += dt * F_avg;
      }

      // Single thread computes B * u(t)
#ifdef AKS_ENABLE_OPENMP
#pragma omp single
#endif
      aks::matmul(B, u_row, std::span<double>{Bu});

      // Parallel add Bu to interior points
#ifdef AKS_ENABLE_OPENMP
#pragma omp for schedule(static) nowait
#endif
      for (size_t x_idx = 1; x_idx < x_sz - 1; ++x_idx) {
        u_row_next[x_idx] += Bu[x_idx];
      }

      // Single thread applies boundary conditions and solves
#ifdef AKS_ENABLE_OPENMP
#pragma omp single
#endif
      {
        // Left boundary: u(0, t) = f(t)
        A(0, 0) = 1.0;
        A(0, 1) = 0.0;
        B(0, 0) = 0.0;
        B(0, 1) = 0.0;
        // u_row_next[0] already has the boundary value

        // Right boundary: u(L, t) = g(t)
        A(x_sz - 1, x_sz - 2) = 0.0;
        A(x_sz - 1, x_sz - 1) = 1.0;
        B(x_sz - 1, x_sz - 2) = 0.0;
        B(x_sz - 1, x_sz - 1) = 0.0;
        // u_row_next[x_sz - 1] already has the boundary value

        // Solve A * u(t+1) = rhs using Thomas algorithm (in-place)
        A.solve(u_row_next, u_row_next);

        // process the solution at the new time step
        problem.process(u_row_next);
      }
      // Implicit barrier here before next time step iteration
    }
#ifdef AKS_ENABLE_OPENMP
  }
#endif
}

} // namespace aks::math