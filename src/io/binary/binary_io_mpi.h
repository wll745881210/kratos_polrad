#pragma once
#ifdef __MPI__

#include "binary_io_base.h"
#include <functional>
#include <mpi.h>

namespace binary_io
{
////////////////////////////////////////////////////////////
// The MPI-parallel version of binary I/O

class mpi_t : public base_t
{
    ////////// Con-/destructor //////////
public:                         // Functions
     mpi_t( const int color = 1  ,
            const int key   = 2 );
    ~mpi_t(  );
    virtual void finalize(  );

    ////////// File //////////
protected:                      // Types
    using f_io_t = std::function
        < int( void *, const int &, const size_t & ) >;
protected:                      // Data
    MPI_File   fh;
    MPI_Comm comm;
    int      rank;
    int    n_rank;
    int file_mode;
protected:                      // Functions
    virtual void map_io( const f_io_t &, void *,
                         const size_t &, const size_t & );
public:                         // Functions
    virtual void open
    ( const std::string   & name, const std::string & mode,
      const buffer_read_t & buffer_read = unbuffered );
    virtual void close(  );
    virtual void   f_read_at
    (       void   * data, const size_t & size,
      const size_t & offset );
    virtual void   f_read_at_all
    (       void   * data, const size_t & size,
      const size_t & offset );
    virtual size_t f_write_at
    ( const void   * data, const size_t & size,
      const size_t & offset );    

    ////////// Save and load (overall) //////////
protected:                      // Function
    virtual void save_header(  );
    virtual void check_open_mpi
    ( const int & desired_mode );
public:                         // Functions
    virtual void        save(  );
    virtual void        load(  );
};

};                              // namespace binary_io

#endif // __MPI__
