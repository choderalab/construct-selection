#!/usr/bin/env python

import os
import numpy
import copy
import time

from simtk import openmm, unit
from simtk.openmm import app

def _findForces(system, force_name, first_only=False):
    # Build a dictionary of forces, assuming we only have one of each type.
    # force_dict = { system.getForce(index).__class__.__name__ : system.getForce(index) for index in range(system.getNumForces()) }
    forces = list()
    for index in range(system.getNumForces()):
        force = system.getForce(index)
        if (force.__class__.__name__ == force_name):
            forces.append(force)

    if first_only and (len(forces) > 0):
        forces = forces[0]

    return forces

if __name__ == "__main__":
    from openmmtools import testsystems
    testsystem = testsystems.AlanineDipeptideImplicit()
    [system, positions, topology] = [testsystem.system, testsystem.positions, testsystem.topology]

    # Get NonbondedForce objects for system copy.
    force = _findForces(system, 'NonbondedForce', first_only=True)

    # Turn off standard particle interactions.
    zero_charge = 0.0 * unit.elementary_charge
    unit_sigma = 1.0 * unit.angstroms
    zero_epsilon = 0.0 * unit.kilocalories_per_mole
    for atom in topology.atoms():
        [charge, sigma, epsilon] = force.getParticleParameters(atom.index)
        force.setParticleParameters(atom.index, zero_charge, unit_sigma, zero_epsilon)

    timestep = 2.0 * unit.femtoseconds
    collision_rate = 5.0 / unit.picoseconds
    temperature = 298.0 * unit.kelvin
    integrator = openmm.LangevinIntegrator(temperature, collision_rate, timestep)

    niterations = 1000
    nsteps = 500

    output_filename = 'output.pdb'
    simulation = app.Simulation(topology, system, integrator)
    simulation.context.setPositions(positions)
    simulation.minimizeEnergy(maxIterations=500)
    simulation.reporters.append( app.PDBReporter(output_filename, nsteps) )
    simulation.step(niterations * nsteps)


