# VMD Visualization Script for C60 Nanoparticle Solvation Study
# =================================================================
# 
# This script creates publication-quality visualizations of:
# - C60 nanoparticles (CPK representation)
# - Water molecules (Lines/CPK)  
# - Hydration shells around nanoparticles
# - Color-coded by epsilon value
#
# Usage:
#   vmd -e visualize_system.tcl
#
# Author: Scientific Analysis Suite
# Date: November 2025

# Function to load and visualize one epsilon system
proc load_epsilon_system {epsilon_val} {
    global BASE_DIR
    
    if {$epsilon_val == 0.0} {
        set eps_dir "$BASE_DIR/epsilon_0.0"
    } else {
        set eps_dir [format "$BASE_DIR/epsilon_%.2f" $epsilon_val]
    }
    
    set traj_file "$eps_dir/production.lammpstrj"
    
    if {![file exists $traj_file]} {
        puts "WARNING: Trajectory file not found: $traj_file"
        return -1
    }
    
    # Load trajectory
    puts "Loading $traj_file..."
    mol new $traj_file type lammpstrj first 0 last -1 step 10 filebonds 1 autobonds 1 waitfor all
    
    set molid [molinfo top]
    
    # Set atom names based on types
    set sel_carbon [atomselect $molid "type 1"]
    $sel_carbon set name C
    $sel_carbon set element C
    
    set sel_oxygen [atomselect $molid "type 2"]
    $sel_oxygen set name O
    $sel_oxygen set element O
    
    set sel_hydrogen [atomselect $molid "type 3"]
    $sel_hydrogen set name H
    $sel_hydrogen set element H
    
    # Delete existing representations
    mol delrep 0 $molid
    
    # Representation 1: C60 nanoparticles (CPK, large spheres)
    mol representation CPK 1.5 0.3 12 12
    mol color ColorID 0
    mol selection "name C"
    mol material Opaque
    mol addrep $molid
    
    # Representation 2: Water oxygens (VDW, smaller)
    mol representation VDW 0.4 12
    mol color ColorID 1
    mol selection "name O"
    mol material Opaque
    mol addrep $molid
    
    # Representation 3: Water hydrogens (Points, very small)
    mol representation Points 1.0
    mol color ColorID 7
    mol selection "name H"
    mol material Opaque
    mol addrep $molid
    
    # Representation 4: First hydration shell (transparent surface)
    mol representation QuickSurf 1.0 0.5 1.0 1.0
    mol color ColorID 3
    mol selection "name O and within 5.0 of name C"
    mol material Transparent
    mol addrep $molid
    
    puts "Loaded molecule $molid for epsilon = $epsilon_val"
    
    return $molid
}

# Main script
set BASE_DIR "/store/shuvam/solvent_effects/6ns_sim/6ns_sim_v2"

# Display settings
display projection Orthographic
display depthcue off
axes location Off
color Display Background white

# Load all epsilon values (one at a time)
set epsilon_values {0.05 0.10 0.15 0.20 0.25}

puts "================================================================"
puts "C60 NANOPARTICLE SOLVATION - VMD VISUALIZATION"
puts "================================================================"

# Load first epsilon value
set first_eps [lindex $epsilon_values 0]
set molid [load_epsilon_system $first_eps]

if {$molid >= 0} {
    # Rotate for good viewing angle
    rotate x by 45
    rotate y by 30
    
    # Scale to fit
    scale by 1.2
    
    puts ""
    puts "Visualization ready!"
    puts "Use 'Graphics > Representations' to modify display"
    puts "To load different epsilon values:"
    puts "  load_epsilon_system 0.10"
    puts "  load_epsilon_system 0.15"
    puts "etc."
    puts ""
    puts "Color scheme:"
    puts "  Black  = C60 nanoparticles"
    puts "  Red    = Water oxygens"
    puts "  White  = Water hydrogens"
    puts "  Orange = First hydration shell"
}

# Function to render high-quality image
proc render_snapshot {filename {width 2400} {height 2400}} {
    render Tachyon $filename \
        "/usr/local/lib/vmd/tachyon_LINUXAMD64" \
        -fullshade -aasamples 12 \
        %s -format TARGA -res $width $height -o %s.tga
    puts "Rendered: $filename.tga"
}

# Function to create rotation movie
proc create_rotation_movie {output_prefix {frames 120}} {
    global BASE_DIR
    
    set output_dir "$BASE_DIR/analysis/plots/vmd_snapshots"
    file mkdir $output_dir
    
    for {set i 0} {$i < $frames} {incr i} {
        set angle [expr {360.0 * $i / $frames}]
        rotate y by [expr {360.0 / $frames}]
        
        set filename [format "$output_dir/${output_prefix}_%04d.tga" $i]
        render snapshot $filename
        
        if {$i % 10 == 0} {
            puts "Rendered frame $i / $frames"
        }
    }
    
    puts "Movie frames saved to: $output_dir"
    puts "Create movie with: ffmpeg -i ${output_prefix}_%04d.tga -c:v libx264 -pix_fmt yuv420p movie.mp4"
}

puts "Additional commands:"
puts "  render_snapshot output_file - Create high-quality image"
puts "  create_rotation_movie output_prefix - Create 360° rotation movie"
