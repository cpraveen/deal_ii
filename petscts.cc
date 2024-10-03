#include <deal.II/lac/petsc_ts.h>

// https://petsc.org/release/manualpages/TS/TSType
// https://petsc.org/release/manualpages/TS/TSSSPType
// https://petsc.org/release/manualpages/TS/TSRKType/
template <int dim>
void
DGSystem<dim>::run_with_petsc()
{
   PETScWrappers::TimeStepperData ts_data;
   ts_data.ts_type = "ssp";
   ts_data.ts_adapt_type = "none";
   ts_data.initial_time = 0.0;
   ts_data.initial_step_size = dt;
   ts_data.final_time = param->final_time;
   ts_data.match_step = true;

   PETScWrappers::TimeStepper<PVector> ts(ts_data, mpi_comm);
   TSSSPSetType(ts.petsc_ts(), "rks3"); // order
   TSSSPSetNumStages(ts.petsc_ts(), 3); // number of stafes

   // RHS function
   ts.explicit_function = [&](const double time,
                              const PVector &y,
                              PVector &res)
   {
      this->stage_time = time;
      this->solution = y;
      this->solution.update_ghost_values();
      this->assemble_rhs();
      res = this->rhs;
   };

   // Monitor function
   ts.monitor = [&](const double time,
                    const PVector &y,
                    const unsigned int iter)
   {
      pcout << "Time step " << iter << " at t=" << time << std::endl;
      // Compute time step

      // Save solution
      if(iter % this->param->output_step == 0)
      {
         this->solution = y;
         this->solution.update_ghost_values();
         this->output_results(time);
      }
   };

   solution_old = solution;
   ts.solve(solution_old);
}
