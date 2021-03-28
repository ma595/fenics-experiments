import numpy as np
import dolfinx
import dolfinx.geometry
from mpi4py import MPI
import ufl
import pytest
import getopt, sys
from scipy.sparse import coo_matrix, csr_matrix
sys.path.append(r'.')
from dolfinx import fem, Function, DirichletBC
from dolfinx.fem import locate_dofs_topological
from dolfinx.mesh import locate_entities_boundary
from dolfinx.cpp.mesh import entities_to_geometry, exterior_facet_indices
import time 
np.set_printoptions(threshold=sys.maxsize)


# references
# https://fenicsproject.org/olddocs/dolfin/1.3.0/python/programmers-reference/fem/solving/solve.html
# https://jorgensd.github.io/dolfinx-tutorial/chapter2/nonlinpoisson_code.html
# https://docs.fenicsproject.org/dolfinx/main/python/generated/dolfinx.fem.html#dolfinx.fem.LinearProblem.solve
# https://github.com/FEniCS/dolfinx/blob/main/python/demo/poisson/demo_poisson.py

fenicsx_comm = MPI.COMM_WORLD
world_rank = fenicsx_comm.Get_rank() 
world_size = fenicsx_comm.Get_size()


def solve_system(N):
    fenics_mesh = dolfinx.UnitCubeMesh(fenicsx_comm, N, N, N)
    fenics_space = dolfinx.FunctionSpace(fenics_mesh, ("CG", 1))
    u = ufl.TrialFunction(fenics_space)
    v = ufl.TestFunction(fenics_space)
    k = 2
    # print(u*v*ufl.ds)
    form = (ufl.inner(ufl.grad(u), ufl.grad(v)) - k ** 2 * ufl.inner(u, v)) * ufl.dx
    
    # locate facets on the cube boundary
    facets = locate_entities_boundary(fenics_mesh, 2, lambda x: np.logical_or( np.logical_or(np.logical_or(np.isclose(x[2], 0.0), np.isclose(x[2], 1.0)),np.logical_or(np.isclose(x[1], 0.0), np.isclose(x[1], 1.0))), np.logical_or(np.isclose(x[0], 0.0), np.isclose(x[0], 1.0))))

    facets.sort()
   
    # alternative - more general approach
    boundary = entities_to_geometry( fenics_mesh,
            fenics_mesh.topology.dim - 1,
            exterior_facet_indices(fenics_mesh),
            True,
            )
    # print(len(facets)
    assert len(facets) == len(exterior_facet_indices(fenics_mesh))
    
    u0 = fem.Function(fenics_space)

    with u0.vector.localForm() as u0_loc:
        u0_loc.set(0)
    # solution vector
    bc = DirichletBC(u0, locate_dofs_topological(fenics_space, 2, facets))

    A = 1 + 1j
    f = Function(fenics_space)
    f.interpolate(lambda x: A * k**2 * np.cos(k * x[0]) * np.cos(k * x[1]))

    L = ufl.inner(f, v) * ufl.dx
    u0.name = "u"
    problem = fem.LinearProblem(form, L, u=u0, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    # problem = fem.LinearProblem(form, L, bcs=[bc], u=u0, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    
    start_time = time.time()
    soln = problem.solve()
    if world_rank == 0:
        print("--- fenics solve done in %s seconds ---" % (time.time() - start_time))
    # solution
    # print(u0.vector[:])

solve_system(10)

# print(dir(out))
# A = dolfinx.fem.assemble_matrix(form)
# A.assemble()
# E0 = dolfinx.Constant(0.0)
# def E0_boundary(x, on_boundary):
#     return on_boundary
# bc = dolfinx.DirichletBC(fenics_space, 0, 'on_boundary')
# A, rhs = dolfinx.fem.assemble_system(form, L)
