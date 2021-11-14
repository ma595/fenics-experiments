import numpy as np
import dolfinx
import dolfinx.geometry
from mpi4py import MPI
import ufl
from scipy.sparse import csr_matrix
import pytest
import sys, os
# sys.path.append("./fem-bem/")
from dolfinx.cpp.mesh import entities_to_geometry, exterior_facet_indices
from scipy.sparse import coo_matrix
from dolfinx.io import XDMFFile
from collections import OrderedDict, Counter

comm = MPI.COMM_WORLD
# create mesh
N = 2
fenics_mesh = dolfinx.UnitCubeMesh(comm, N, N, N)
fenics_space = dolfinx.FunctionSpace(fenics_mesh, ("CG", 1))


tets = fenics_mesh.topology.connectivity(3, 0)
fenics_mesh.topology.create_connectivity(2, 0)
tris = fenics_mesh.topology.connectivity(2, 0)
fenics_mesh.topology.create_connectivity(2, 3)
tri_to_tet = fenics_mesh.topology.connectivity(2, 3)


dofmap = fenics_space.dofmap.index_map.global_indices()
ghosts = fenics_space.dofmap.index_map.ghosts
ghosts_list = list(ghosts)
ghost_owner = fenics_space.dofmap.index_map.ghost_owner_rank()

print("rank ", comm.rank, dofmap, 
        "\nghosts", ghosts, ghost_owner, '\n')

