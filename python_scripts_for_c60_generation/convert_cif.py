import numpy as np

def parse_cif_and_write_lammps(cif_file, output_file):
    atoms = []
    bonds = []
    
    with open(cif_file, 'r') as f:
        lines = f.readlines()
    
    # Parse Atoms
    reading_atoms = False
    reading_bonds = False
    
    for line in lines:
        if line.startswith('_chem_comp_atom.pdbx_ordinal'):
            reading_atoms = True
            continue
        if line.startswith('#') and reading_atoms:
            reading_atoms = False
            continue
            
        if line.startswith('_chem_comp_bond.pdbx_ordinal'):
            reading_bonds = True
            continue
        if line.startswith('#') and reading_bonds:
            reading_bonds = False
            continue
            
        if reading_atoms:
            parts = line.split()
            if len(parts) > 12:
                # 60C  C1   C1   C  0  1  Y  N  N  12.686  29.115  18.875 ...
                # Indices: 9, 10, 11 are x, y, z
                atom_name = parts[1]
                elem = parts[3]
                x = float(parts[9])
                y = float(parts[10])
                z = float(parts[11])
                atoms.append({'name': atom_name, 'elem': elem, 'x': x, 'y': y, 'z': z})

    # Center atoms
    coords = np.array([[a['x'], a['y'], a['z']] for a in atoms])
    center = np.mean(coords, axis=0)
    coords -= center
    
    # Write LAMMPS Data File
    with open(output_file, 'w') as f:
        f.write("C60 molecule from CIF\n\n")
        f.write(f"{len(atoms)} atoms\n")
        f.write("3 atom types\n") # Carbon, Oxygen, Hydrogen
        f.write("\n")
        
        # Box bounds - 60x60x60 Angstroms
        f.write("-30.0 30.0 xlo xhi\n")
        f.write("-30.0 30.0 ylo yhi\n")
        f.write("-30.0 30.0 zlo zhi\n")
        f.write("\n")
        
        f.write("Masses\n\n")
        f.write("1 12.011\n")
        f.write("2 15.999\n")
        f.write("3 1.008\n")
        f.write("\n")
        
        f.write("Atoms # full\n\n")
        # ID mol type q x y z
        for i, (atom, coord) in enumerate(zip(atoms, coords)):
            f.write(f"{i+1} 1 1 0.0 {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}\n")

if __name__ == "__main__":
    parse_cif_and_write_lammps('data_files/60C.cif', 'data_files/C60.data')
    print("Converted 60C.cif to data_files/C60.data")
