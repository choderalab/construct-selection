#!/usr/bin/env python

#=============================================================================================
# MODULE DOCSTRING
#=============================================================================================

"""
Utility routines for modifying OpenMM System objects to mimic the unfolded state or truncated constructs.

DESCRIPTION


EXAMPLES

COPYRIGHT

@author John D. Chodera <jchodera@gmail.com>

All code in this repository is released under the GNU General Public License.

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program.  If not, see <http://www.gnu.org/licenses/>.

TODO

"""

#=============================================================================================
# GLOBAL IMPORTS
#=============================================================================================

import os
import numpy
import copy
import time

from simtk import openmm, unit
from simtk.openmm import app

#=============================================================================================
# Unfolded state model
#=============================================================================================

def _findForces(system, force_name, first_only=False):
    """
    Extract the Force objects matching the requested name.


    """

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

def replaceGBSAOBCForce(system):
    """
    If the System contains a GBSAOBCForce, replace it with a CustomGBForce.

    """
    from simtk.openmm.app.internal.customgbforces import GBSAOBC2Force

    # Go through forces in reverse so we can delete GB forces as we go along.
    for force_index in range(system.getNumForces()-1, 0, -1):
        force = system.getForce(force_index)
        if (force.__class__.__name__ != 'GBSAOBCForce'):
            continue

        # Convert GBSAOBCForce to CustomGBForce
        cutoff = None
        if force.getNonbondedMethod() == 'CutoffNonPeriodic':
            cutoff = force.getCutoffDistance().value_in_unit_system(unit.md_unit_system)

        custom_force = GBSAOBC2Force(
            solventDielectric=force.getSolventDielectric(),
            soluteDielectric=force.getSoluteDielectric(),
            cutoff=cutoff,
            SA='ACE')

        custom_force.setNonbondedMethod(force.getNonbondedMethod()) # TODO: Go through enum types rather than integers
        custom_force.setCutoffDistance(force.getCutoffDistance())

        for particle_index in range(force.getNumParticles()):
            [charge, radius, scalingFactor] = force.getParticleParameters(particle_index)
            custom_force.addParticle([charge, radius, scalingFactor])

        # Add new force and delete old force.
        system.addForce(custom_force)
        system.removeForce(force_index)

    return

def createConstruct(topology, reference_system, construct_residues):
    """
    Create a construct where interactions outside of specified construct residues are set to noninteracting 'ghost' particles.

    Parameters
    ----------
    topology : simtk.openmm.app.Topology
        The Topology object for the reference system.
    reference_system : simtk.openmm.System
        The System object for the reference system to be modified.
    construct_residues: array_like
        The set of integer residue numbers defining the construct.

    Returns
    -------
    system : simtk.openmm.System
        A System object corresponding to the desired construct.

    """


    # Make a copy of the reference system if a working copy is not specified.
    system = copy.deepcopy(reference_system)

    # Get NonbondedForce objects for system copy.
    nonbonded_force = _findForces(system, 'NonbondedForce', first_only=True)
    gb_force = _findForces(system, 'GBSAOBCForce', first_only=True)

    # Turn off charges and Lennard-Jones parameters for atoms outside of the construct.
    zero_charge = 0.0 * unit.elementary_charge
    unit_sigma = 1.0 * unit.angstroms
    zero_radius = 0.0 * unit.angstroms
    zero_epsilon = 0.0 * unit.kilocalories_per_mole
    for atom in topology.atoms():
        if atom.residue.index not in construct_residues:
            nonbonded_force.setParticleParameters(atom.index, zero_charge, unit_sigma, zero_epsilon)
            gb_force.setParticleParameters(atom.index, zero_charge, unit_sigma, 0.0)

    # Make a list of "ghost" atoms.
    ghost_atoms = [ atom.index for atom in topology.atoms() if (atom.residue.index not in construct_residues) ]

    # Modify exclusions to exclude interactions with "ghost" termini.
    zero_chargeprod = zero_charge * zero_charge
    for index in range(nonbonded_force.getNumExceptions()):
        [atom1_index, atom2_index, chargeprod, sigma, epsilon] = nonbonded_force.getExceptionParameters(index)
        if (atom1_index in ghost_atoms) or (atom2_index in ghost_atoms):
            nonbonded_force.setExceptionParameters(index, atom1_index, atom2_index, zero_chargeprod, unit_sigma, zero_epsilon)

    # Return modified system.
    return system

def createUnfoldedSurrogate(topology, reference_system, locality=5):
    """
    Create a surrogate for the unfolded state in which non-local residue interactions are excluded.

    Parameters
    ----------
    topology : simtk.openmm.app.Topology
        The Topology object for the reference system.
    reference_system : simtk.openmm.System
        The System object for the reference system to be modified.
    locality : int, optional
        The number of residues beyond which nonbonded interactions will be excluded.


    Returns
    -------
    system : simtk.openmm.System
        A System object corresponding to the desired construct.

    TODO
    ----
    * Replace NonbondedForce with CustomBondForce for improved efficiency.
    * Replace CustomGBForce with CustomBondForce for improved efficiency.

    """

    # Make a copy of the reference system if a working copy is not specified.
    system = copy.deepcopy(reference_system)

    # Get NonbondedForce objects for system copy.
    nonbonded_force = _findForces(system, 'NonbondedForce', first_only=True)
    gb_force = _findForces(system, 'CustomGBForce', first_only=True)

    # Add exclusions for non-local interactions.
    for atom1 in topology.atoms():
        [charge1, sigma1, epsilon1] = nonbonded_force.getParticleParameters(atom1.index)
        for atom2 in topology.atoms():
            if (atom1.index < atom2.index):
                if (abs(atom1.residue.index - atom2.residue.index) <= locality):
                    # Create exclusion.
                    try:
                        [charge2, sigma2, epsilon2] = nonbonded_force.getParticleParameters(atom1.index)
                        chargeprod = charge1 * charge2
                        sigma = 0.5 * (sigma1 + sigma2)
                        epsilon = unit.sqrt(epsilon1 * epsilon2)
                        nonbonded_force.addException(atom1.index, atom2.index, chargeprod, sigma, epsilon, False)
                    except:
                        # Exception already exists; don't modify it.
                        pass
                else:
                    # Exclude GB interactions between nonlocal particles.
                    gb_force.addExclusion(atom1.index, atom2.index)

    # Turn off standard particle interactions.
    zero_charge = 0.0 * unit.elementary_charge
    unit_sigma = 1.0 * unit.angstroms
    zero_epsilon = 0.0 * unit.kilocalories_per_mole
    for atom in topology.atoms():
        nonbonded_force.setParticleParameters(atom.index, zero_charge, unit_sigma, zero_epsilon)

    # Return modified system.
    return system

def createUnfoldedSurrogate2(topology, reference_system, locality=5):

    # Create new deep copy reference system to modify.
    system = openmm.System()

    # Set periodic box vectors.
    [a,b,c] = reference_system.getDefaultPeriodicBoxVectors()
    system.setDefaultPeriodicBoxVectors(a,b,c)

    # Add atoms.
    for atom_index in range(reference_system.getNumParticles()):
        mass = reference_system.getParticleMass(atom_index)
        system.addParticle(mass)

    # Add constraints
    for constraint_index in range(reference_system.getNumConstraints()):
        [iatom, jatom, r0] = reference_system.getConstraintParameters(constraint_index)
        system.addConstraint(iatom, jatom, r0)

    # Modify forces as appropriate, copying other forces without modification.
    nforces = reference_system.getNumForces()
    for force_index in range(nforces):
        reference_force = reference_system.getForce(force_index)

        if isinstance(reference_force, openmm.NonbondedForce):
            # Create CustomBondForce instead.
            energy_expression = "4*epsilon*((sigma/r)^12 - (sigma/r)^6) + 138.935456*chargeprod/r;" # TODO: Check coulomb constant
            custom_bond_force = openmm.CustomBondForce(energy_expression)
            custom_bond_force.addPerBondParameter("chargeprod") # charge product
            custom_bond_force.addPerBondParameter("sigma") # Lennard-Jones sigma
            custom_bond_force.addPerBondParameter("epsilon") # Lennard-Jones epsilon
            #system.addForce(custom_bond_force)

            # Add exclusions.
            from sets import Set
            exceptions = Set()
            for index in range(reference_force.getNumExceptions()):
                [atom1_index, atom2_index, chargeprod, sigma, epsilon] = reference_force.getExceptionParameters(index)
                custom_bond_force.addBond(atom1_index, atom2_index, [chargeprod, sigma, epsilon])

                if atom2_index < atom1_index:
                    exceptions.add( (atom2_index, atom1_index) )
                else:
                    exceptions.add( (atom1_index, atom2_index) )

            # Add local interactions.
            for atom1 in topology.atoms():
                [charge1, sigma1, epsilon1] = reference_force.getParticleParameters(atom1.index)
                for atom2 in topology.atoms():
                    if (atom1.index < atom2.index) and (abs(atom1.residue.index - atom2.residue.index) <= locality) and ((atom1.index, atom2.index) not in exceptions):
                        [charge2, sigma2, epsilon2] = reference_force.getParticleParameters(atom1.index)
                        chargeprod = charge1 * charge2
                        sigma = 0.5 * (sigma1 + sigma2)
                        epsilon = unit.sqrt(epsilon1 * epsilon2)
                        custom_bond_force.addBond(atom1.index, atom2.index, [chargeprod, sigma, epsilon])

        elif isinstance(reference_force, openmm.GBSAOBCForce):
            # Copy force without modification.
            force = copy.deepcopy(reference_force)
            #system.addForce(force)

            zero_charge = 0.0 * unit.elementary_charge
            unit_sigma = 1.0 * unit.angstroms
            for atom in topology.atoms():
                force.setParticleParameters(atom.index, zero_charge, unit_sigma, 0.0)
        else:

            # Copy force without modification.
            force = copy.deepcopy(reference_force)
            system.addForce(force)

    return system

def createUnfoldedSurrogateCutoff(topology, reference_system, cutoff=5.0*unit.angstrom, switch_width=1.0*unit.angstrom):
    """
    Create a surrogate for the unfolded state in which non-local residue interactions are eliminated
    by using a short cutoff.

    Parameters
    ----------
    topology : simtk.openmm.app.Topology
        The Topology object for the reference system.
    reference_system : simtk.openmm.System
        The System object for the reference system to be modified.
    cutoff : simtk.unit.Quantity with dimension distance, optional, default=5*angstrom
        The cutoff to use for nonbonded interactions.
    switch_width : simtk.unit.Quantity with dimension distance, optional, default=1*angstrom
        The switch width to use.

    Returns
    -------
    system : simtk.openmm.System
        A System object corresponding to the desired construct.

    NOTE: This is only appropriate for GBSA systems.

    """

    # Make a copy of the reference system if a working copy is not specified.
    system = copy.deepcopy(reference_system)

    # Set cutoff for all force objects.
    for force_index in range(system.getNumForces()):
        force = system.getForce(force_index)

        if hasattr(force, 'setNonbondedMethod'):
            force.setNonbondedMethod(getattr(force, 'CutoffNonPeriodic'))
        if hasattr(force, 'setCutoffDistance'):
            force.setCutoffDistance(cutoff)
        if hasattr(force, 'setUseSwitchingFunction'):
            force.setUseSwitchingFunction(True)
        if hasattr(force, 'setSwitchingDistance'):
            force.setSwitchingDistance(cutoff - switch_width)

    # Return modified system.
    return system

def benchmark(system, positions, nsteps=500):
    timestep = 2.0 * unit.femtoseconds
    collision_rate = 90.0 / unit.picoseconds
    temperature = 298.0 * unit.kelvin
    integrator = openmm.LangevinIntegrator(temperature, collision_rate, timestep)
    context = openmm.Context(system, integrator)
    context.setPositions(positions)
    state = context.getState(getEnergy=True)
    print state.getPotentialEnergy()
    print("    Minimizing...")
    tolerance = 1.0 * unit.kilocalories_per_mole / unit.angstrom
    maxIterations = 500
    openmm.LocalEnergyMinimizer.minimize(context, tolerance, maxIterations)
    print("    Simulating...")
    initial_time = time.time()
    integrator.step(nsteps)
    state = context.getState(getEnergy=True)
    final_time = time.time()
    elapsed_time = final_time - initial_time
    print state.getPotentialEnergy()
    print("%d steps took %.3f s (%.3f ns/day)" % (nsteps, elapsed_time, nsteps * (timestep/unit.nanoseconds) / elapsed_time * 24*60*60))
    del context, integrator, state
    return

def simulate(topology, system, positions, output_filename):
    import simtk.openmm.app as app

    timestep = 2.0 * unit.femtoseconds
    collision_rate = 20.0 / unit.picoseconds
    temperature = 298.0 * unit.kelvin
    integrator = openmm.LangevinIntegrator(temperature, collision_rate, timestep)

    niterations = 1000
    nsteps = 500

    simulation = app.Simulation(topology, system, integrator)
    simulation.context.setPositions(positions)

    reporter = app.PDBReporter(output_filename, nsteps)
    simulation.reporters.append( reporter )

    # Write first frame.
    state = simulation.context.getState(getPositions=True)
    reporter.report(simulation, state)

    print("    Minimizing...")
    simulation.minimizeEnergy(maxIterations=50)

    # Write first frame.
    state = simulation.context.getState(getPositions=True)
    reporter.report(simulation, state)

    print("    Simulating...")
    for iteration in range(niterations):
        print("        Iteration %d / %d" % (iteration, niterations))
        simulation.step(nsteps)

    return

#=============================================================================================
# MAIN
#=============================================================================================

if __name__ == "__main__":
    # Run doctests.
    import doctest
    doctest.testmod(verbose=False)

    # Test with a real protein.
    import simtk.openmm.app as app
    print "Loading protein..."
    pdb = app.PDBFile('T4-lysozyme.pdb')
    print "Loading forcefield..."
    forcefield = app.ForceField('amber96.xml', 'amber96_obc.xml')
    print "Parameterizing protein..."
    reference_system = forcefield.createSystem(pdb.topology, constraints=True)
    print "Replacing GBSAOBCForce with CustomGBForce to allow exclusions to be added..."
    replaceGBSAOBCForce(reference_system)

    print "Creating unfolded surrogate..."
    system = createUnfoldedSurrogate(pdb.topology, reference_system, locality=5)
    #system = createUnfoldedSurrogateCutoff(pdb.topology, system)

    #print "Benchmarking unfolded surrogate..."
    #benchmark(system, pdb.positions)

    print "Simulating unfolded surrogate..."
    simulate(pdb.topology, system, pdb.positions, 'unfolded.pdb')

    print "Testing construct creation..."
    atoms = [ atom for atom in pdb.topology.atoms() ]
    first_residue = atoms[0].residue.index
    last_residue = atoms[-1].residue.index
    construct_residues = range(first_residue + 5, last_residue - 5 + 1)
    print construct_residues
    system = createConstruct(pdb.topology, reference_system, construct_residues)

    print "Simulating construct..."
    benchmark(system, pdb.positions)
