#pragma once

#include "../device/device.h"
#include "../utilities/data_transfer/serialize.h"
#include <memory>
#include <typeinfo>
#include <type_traits>

namespace comm
{
////////////////////////////////////////////////////////////
// Operation for reductions

enum  operation_t          {    min,    max,       sum };
const device::stream_t strm( ( device::stream_t )( 0 ) );

////////////////////////////////////////////////////////////
// Base class for communication

class base_t
{
    ////////// Con-/destructor //////////
public:                         // Function
    base_t(  ) :   root ( 0 ),   rank_( 0 ) ,
                 l_rank_( 0 ), n_rank_( 1 ) {  };
    virtual void init( const input & args ) {  };
    virtual void finalize(  )                        {  };
    virtual std::shared_ptr < base_t > split(  )     const
    {     return std::make_shared< base_t > (  );       };

    ////////// Communication handlers //////////
protected:                      // Data
    int     root ;
    int     rank_;
    int   l_rank_;
    int   n_rank_;
public:                         // Function
    virtual void    wait_all_d(  ){  };
    virtual void    wait_all_h(  ){  };
    virtual const int &   rank(  ){ return         rank_; };
    virtual const int & l_rank(  ){ return       l_rank_; };
    virtual const int & n_rank(  ){ return       n_rank_; };
    virtual bool       is_root(  ){ return root == rank_; };

    ////////// Device-side async send and recv //////////
public:                         // Module
    std::shared_ptr< device::base_t > p_dev;
public:                         // Functions
    virtual void  isend_d
    ( void       * buff, const size_t & size ,
      const int  & rank, const int    &  tag ,
      const device::stream_t & stream = strm ) {  };
    virtual void  irecv_d
    ( void       * buff, const size_t & size ,
      const int  & rank, const int    &  tag ,
      const device::stream_t & stream = strm ) {  };
    virtual void  isend_h
    ( void       * buff, const size_t & size ,
      const int  & rank, const int    &  tag ) {  };
    virtual void  irecv_h
    ( void       * buff, const size_t & size ,
      const int  & rank, const int    &  tag ) {  };

    ////////// Host-side broadcast //////////
public:                         // Functions
    virtual void bcast  ( void * p, const size_t & s ) {  };
    virtual void barrier(                            ) {  };
    template< class T, serialize::if_serialized_t< T > = 0 >
    void bcast( T & t );

    ////////// Host-side Reduction //////////
protected:                      // Function
    virtual void reduce_all_ker
    ( void  * p,  const std::type_info & t,
      const operation_t & o, const int & n,
      const bool & async = false ) {  };
public:                         // Interface
    template< class T, std::enable_if_t
            < ! std::is_pointer< T >::value, int > = 0 >
    void reduce_all( T & t, const operation_t & op,
                     const bool & async = false );
    void reduce_all_finish(  ) {  };
    template< class T, std::enable_if_t
            <   std::is_pointer< T >::value, int > = 0 >
    void reduce_all( T   t, const operation_t & op ,
                     const int  & n = 1,
                     const bool & async = false );
};

////////////////////////////////////////////////////////////
// Generic serialized broadcast

template< class T, serialize::if_serialized_t< T > >
void base_t::bcast( T & t )
{
    size_t     size( 0 );
    std::stringstream ss;
    std::string      buf;

    if(  is_root(  ) )
    {
        serialize::write  (  ss,  t  );
        size = ss.str(  ) .  size (  );
        ss.     clear(  ) ;
        buf.resize            ( size );
        ss.read   ( & buf[ 0 ], size );
    }
    bcast( & size, sizeof( size_t ) );
    buf.resize           ( size     );
    bcast( &  buf[  0 ],   size     );
    if( ! is_root(  ) )
    {
        ss.write  ( & buf[ 0 ], size );
        ss.clear  (  );
        serialize::read    ( ss,   t );
    }
    return;
}

////////////////////////////////////////////////////////////
// Interface for all-reduce

template< class T, std::enable_if_t
        < ! std::is_pointer< T >::value, int > >
void base_t::reduce_all( T & t, const operation_t & op,
                         const bool & async )
{
    reduce_all_ker( & t, typeid( T ), op, 1, async );
}

template< class T, std::enable_if_t
        <   std::is_pointer< T >::value, int > >
void base_t::reduce_all
( T t, const operation_t & op, const int & n,
  const bool & async )
{
    reduce_all_ker( t, typeid( std::remove_pointer_t< T > ),
                    op, n, async );
}

};                              // namespace comm
