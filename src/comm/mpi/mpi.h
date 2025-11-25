#pragma once
#ifdef __MPI__

#include "../comm.h"
#include "req_list.h"
#include "buffer.h"

namespace comm
{
////////////////////////////////////////////////////////////
// MPI-based communication

class mpi_t : public base_t
{
    ////////// Con-/destructor //////////
protected:                      // Data
    static  int         count;  // Count of instances
public:                         // Function
     mpi_t(  );
    ~mpi_t(  );
    virtual void init( const input & args );
    virtual std::shared_ptr < base_t > split(  ) const;

    ////////// Node-local optimization //////////
protected:                      // Data
    MPI_Comm                   l_comm;
    int                       nl_rank;
    int               rank_l_lim[ 2 ];
    bool               local_optimize;
    std::vector< local_t > local_send;
    std::vector< local_t > local_recv;    
protected:                      // Functions
    template< class fun_T >
    void    map_local ( const fun_T & f );
    virtual bool is_local( const int &  );
    virtual void local_init          (  );
    virtual void local_finalize      (  );
    virtual void local_synchronize   (  );

    ////////// Ordinary remote communication //////////
protected:                      // Data
    std::map < int, remote_t > buf_send;
    std::map < int, remote_t > buf_recv;
protected:                      // Functions
    virtual void remote_finalize   (  );
    virtual void remote_synchronize(  );

    ////////// Communication handlers //////////
protected:                      // Data
    MPI_Comm           comm;
    req_ls_t         reqs_d;
    req_ls_t         reqs_h;
    bool    wait_background;
public:                         // Function
    virtual void wait_all_d(  );
    virtual void wait_all_h(  );

    ////////// Device-side async send and recv //////////
public:                         // Functions
    virtual void  isend_d
    ( void       * data, const size_t & size ,
      const int  & rank, const int    &  tag ,
      const device::stream_t & stream = strm );
    virtual void  irecv_d
    ( void       * data, const size_t & size ,
      const int  & rank, const int    &  tag ,
      const device::stream_t & stream = strm );
    virtual void  isend_h
    ( void       * buff, const size_t & size ,
      const int  & rank, const int    &  tag );
    virtual void  irecv_h
    ( void       * buff, const size_t & size ,
      const int  & rank, const int    &  tag );

    ////////// Host-side broadcast //////////
public:                         // Function
    virtual void bcast  ( void * p, const size_t & size );
    virtual void barrier(  );    

    ////////// Host-side reduction //////////
protected:                      // Data
    MPI_Request reduce_req;
protected:                      // Function
    virtual void reduce_all_ker
    ( void * p,    const std:: type_info & tp,
      const  operation_t & op, const int &  n,
      const bool & async = false );
public:                         // Function
    void reduce_all_finish(  );
};

};                              // namespace communicate
#endif // __MPI__
