#!/bin/bash

##############################################################################
# Setup Data Files for Solvent Structure Around Nanoparticles
# 
# This script automatically downloads nanoparticle structures and water
# templates from atomify and lammps-input-files repositories.
#
# Usage: bash setup_data_files.sh [OPTIONS]
##############################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT_FILES_DIR="$SCRIPT_DIR/input_files"
TEMP_DIR="/tmp/solvent_data_download_$$"

##############################################################################
# Helper Functions
##############################################################################

print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ $1${NC}"
}

cleanup() {
    print_info "Cleaning up temporary files..."
    rm -rf "$TEMP_DIR"
}

# Trap EXIT to cleanup
trap cleanup EXIT

##############################################################################
# Main Setup
##############################################################################

print_header "Solvent Data Files Setup"

# Create directories
print_info "Creating directories..."
mkdir -p "$INPUT_FILES_DIR"
mkdir -p "$TEMP_DIR"
print_success "Directories ready"

# Check dependencies
print_info "Checking dependencies..."
if ! command -v git &> /dev/null; then
    print_error "Git not found. Please install git."
    exit 1
fi
if ! command -v wget &> /dev/null && ! command -v curl &> /dev/null; then
    print_error "wget or curl not found. Please install one of them."
    exit 1
fi
print_success "Dependencies OK"

echo ""

##############################################################################
# Download Water Template (TIP4P)
##############################################################################

print_header "Downloading Water Template"

if [ -f "$INPUT_FILES_DIR/H2O_TIP4P.txt" ]; then
    print_success "H2O_TIP4P.txt already exists"
else
    print_info "Downloading TIP4P water template..."
    
    cd "$TEMP_DIR"
    
    # Try downloading directly from GitHub raw
    if command -v wget &> /dev/null; then
        wget -q "https://raw.githubusercontent.com/simongravelle/lammps-input-files/main/LAMMPS-molecules/H2O_TIP4P.txt" \
            -O H2O_TIP4P.txt 2>/dev/null || {
            print_error "Failed to download via wget, trying git clone..."
            git clone -q --depth 1 https://github.com/simongravelle/lammps-input-files.git lammps-files 2>/dev/null || {
                print_error "Failed to clone repository"
                exit 1
            }
            cp lammps-files/LAMMPS-molecules/H2O_TIP4P.txt H2O_TIP4P.txt
            rm -rf lammps-files
        }
    else
        curl -s "https://raw.githubusercontent.com/simongravelle/lammps-input-files/main/LAMMPS-molecules/H2O_TIP4P.txt" \
            -o H2O_TIP4P.txt || {
            print_error "Failed to download via curl, trying git clone..."
            git clone -q --depth 1 https://github.com/simongravelle/lammps-input-files.git lammps-files 2>/dev/null || {
                print_error "Failed to clone repository"
                exit 1
            }
            cp lammps-files/LAMMPS-molecules/H2O_TIP4P.txt H2O_TIP4P.txt
            rm -rf lammps-files
        }
    fi
    
    if [ -f "H2O_TIP4P.txt" ]; then
        cp H2O_TIP4P.txt "$INPUT_FILES_DIR/"
        print_success "Downloaded H2O_TIP4P.txt"
    else
        print_error "Failed to download H2O_TIP4P.txt"
        exit 1
    fi
    
    cd "$SCRIPT_DIR"
fi

echo ""

##############################################################################
# Download SiC Nanoparticle
##############################################################################

print_header "Downloading SiC Nanoparticle"

if [ -f "$INPUT_FILES_DIR/sic_nanoparticle.data" ]; then
    print_success "sic_nanoparticle.data already exists"
else
    print_info "Downloading SiC nanoparticle from atomify..."
    
    cd "$TEMP_DIR"
    
    if command -v wget &> /dev/null; then
        wget -q "https://raw.githubusercontent.com/andeplane/atomify/main/public/examples/sic/nanoparticle/sic_nanoparticle.data" \
            -O sic_nanoparticle.data 2>/dev/null || {
            print_error "Failed to download, trying git clone..."
            git clone -q --depth 1 --sparse https://github.com/andeplane/atomify.git atomify 2>/dev/null || {
                print_error "Failed to clone atomify"
                exit 1
            }
            cd atomify
            git sparse-checkout set "public/examples/sic/nanoparticle" 2>/dev/null || true
            git checkout 2>/dev/null || true
            cd "$TEMP_DIR"
            if [ -f "atomify/public/examples/sic/nanoparticle/sic_nanoparticle.data" ]; then
                cp atomify/public/examples/sic/nanoparticle/sic_nanoparticle.data .
            fi
        }
    else
        curl -s "https://raw.githubusercontent.com/andeplane/atomify/main/public/examples/sic/nanoparticle/sic_nanoparticle.data" \
            -o sic_nanoparticle.data || {
            print_error "Failed to download, trying git clone..."
            git clone -q --depth 1 --sparse https://github.com/andeplane/atomify.git atomify 2>/dev/null || {
                print_error "Failed to clone atomify"
                exit 1
            }
            cd atomify
            git sparse-checkout set "public/examples/sic/nanoparticle" 2>/dev/null || true
            git checkout 2>/dev/null || true
            cd "$TEMP_DIR"
            if [ -f "atomify/public/examples/sic/nanoparticle/sic_nanoparticle.data" ]; then
                cp atomify/public/examples/sic/nanoparticle/sic_nanoparticle.data .
            fi
        }
    fi
    
    if [ -f "sic_nanoparticle.data" ]; then
        cp sic_nanoparticle.data "$INPUT_FILES_DIR/"
        print_success "Downloaded sic_nanoparticle.data"
    else
        print_error "Failed to download sic_nanoparticle.data"
        exit 1
    fi
    
    cd "$SCRIPT_DIR"
fi

echo ""

##############################################################################
# Verify Downloaded Files
##############################################################################

print_header "Verification"

MISSING=0

# Check TIP4P
if [ -f "$INPUT_FILES_DIR/H2O_TIP4P.txt" ]; then
    SIZE=$(wc -l < "$INPUT_FILES_DIR/H2O_TIP4P.txt")
    print_success "H2O_TIP4P.txt ($SIZE lines)"
else
    print_error "H2O_TIP4P.txt NOT found"
    MISSING=$((MISSING + 1))
fi

# Check SiC
if [ -f "$INPUT_FILES_DIR/sic_nanoparticle.data" ]; then
    SIZE=$(wc -l < "$INPUT_FILES_DIR/sic_nanoparticle.data")
    print_success "sic_nanoparticle.data ($SIZE lines)"
else
    print_error "sic_nanoparticle.data NOT found"
    MISSING=$((MISSING + 1))
fi

echo ""

if [ $MISSING -eq 0 ]; then
    print_header "Setup Complete!"
    echo -e "${GREEN}All required files downloaded successfully!${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. cd shell_scripts"
    echo "  2. ./run_solvation_study.sh SiC 500 100 10"
    echo ""
    exit 0
else
    print_error "Setup incomplete ($MISSING files missing)"
    echo ""
    echo "You can try again or manually download from:"
    echo "  - TIP4P: https://github.com/simongravelle/lammps-input-files"
    echo "  - SiC:   https://github.com/andeplane/atomify"
    echo ""
    exit 1
fi
