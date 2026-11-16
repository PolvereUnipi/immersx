// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by Luca Heltai
//
// This file is part of the ImmersX application, based on the deal.II
// library.
//
// ---------------------------------------------------------------------

#include <immersx/core/ida_lagging_policy.templates.h>

namespace ImmersX
{
  template class IDALaggingPolicy<ImmersXLA::MPI::Vector>;
  template class IDALaggingPolicy<ImmersXLA::MPI::BlockVector>;
} // namespace ImmersX
