#pragma once

#include "binary_io_base.h"
#include "binary_io_mpi.h"

////////////////////////////////////////////////////////////
//                  Format of Binary I/O                  //
//                    General Purpose                     //
////////////////////////////////////////////////////////////
// : Sec. 0: sizeof( size_t ), endian, start of bulk data  :
// :_______________________________________________________:
// | { is_little | sizeof( size_t ) } |   Starting point   |
// |        [ bootstrapping! ]        |    of bulk data    |
// \------------- byte ---------------|------ size_t ------|
// :                                                       :
// : Sec. 1: Serialized map from tag to size, unit size    :
// :         ( "u_size" ) , and offset.                    :
// : Note: The offset is relative to the start of Sec. 2   :
// :_______________________________________________________:
// | # of entry |/-tag    0   |   ....   |/-tag    N - 1   |
// |            |\[size   0   |   ....   |\[size   N - 1   |
// |            | [u_size 0   |   ....   | [u_size N - 1   |
// |            | [offset 0   |   ....   | [offset N - 1   |
// |-- size_t --|- entry  0 --|   ....   |- entry  N - 1 --|
// \-- std::map< std::string, b_info_t > ------------------|
// :                                                       :
// : Sec. 2: Bulk data                                     :
// :_______________________________________________________:
// |   data 0   |   data  1   |   ....   |   data  N - 1   |
// \-- size 0 --\-- size  1 --|   ....   \-- size  N - 1 --|
//  ^offset 0    ^offset  1       ....    ^offset  N - 1   |
////////////////////////////////////////////////////////////

namespace binary_io
{
////////////////////////////////////////////////////////////
// Default type: Use the MPI version if possible
#ifndef __MPI__
typedef base_t default_t;
#else
typedef  mpi_t default_t;
#endif  // MPI
////////////////////////////////////////////////////////////
};                              // namespace binary_io
