#!/bin/bash
# debug_mpi.sh

source setup_env.sh

echo "--- MPI Environment ---"
module list
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "--- Libfabric Info ---"
fi_info -p cxi
echo "--- Python MPI Check ---"
python3 -c "import mpi4py.MPI; print('Rank:', mpi4py.MPI.COMM_WORLD.Get_rank(), 'of', mpi4py.MPI.COMM_WORLD.Get_size())"
