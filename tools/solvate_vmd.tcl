# VMD/TopoTools helper (batch mode)
# Usage (headless): vmd -dispdev text -e tools/solvate_vmd.tcl -args input_pdb output_data

# This script attempts to write a LAMMPS data file using TopoTools. For TIP4P,
# you need the M-sites present in the PDB and appropriate molecule typing.

package require topotools

if {[llength $argv] < 2} {
    puts stderr "Usage: vmd -dispdev text -e solvate_vmd.tcl -args input.pdb output.data"
    exit 1
}
set input [lindex $argv 0]
set output [lindex $argv 1]

mol new $input type pdb waitfor all
# You may need to create a PSF or set types manually depending on your system
# topo autogenerate or use topo tools to set masses/charges
# topo writetypes
# topo writedata supports full LAMMPS datafile (check TopoTools version for options)

# Attempt write (user should verify atomtypes / masses / charges)
topo writedata $output full
puts "Wrote LAMMPS data placeholder to $output"
