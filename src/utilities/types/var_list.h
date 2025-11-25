#pragma once

#include "../../device/device.h"
#include "../functions.h"

namespace utils
{
////////////////////////////////////////////////////////////
// List on const memory that naturally handels host-device.
// Required member methods for val_T:
// __host__   size_t set_shift ( const size_t & );
// __host__   void   relocate_shifted( void   * );
// __host__   void   setup           ( void   * );

template < class val_T, class size_T = int >
struct const_list_t
{
    ////////// Data //////////
    size_T     size_byte;
    size_T          size;
    val_T         * data;

    ////////// Host-side functions //////////
    __host__ const_list_t(  )
        : data( nullptr ), size( 0 ), size_byte( 0 ) {  };
    __host__ void init( const size_T & size )
    {
        this-> size     = size  ;
        data = new val_T[ size ];
        return;
    };
    __host__ void regularize( device::base_t & f_dev )
    {
        const size_t shift_0 = size * sizeof ( val_T );
        size_byte = utils::align( shift_0, 8 );
        for( size_t  i = 0 ; i  < size; ++ i )
             size_byte = utils::align
                ( data[ i ].set_shift( size_byte ), 8 );
        void *  p_h =          std::malloc( size_byte );
        void *  p_c = f_dev.f_malloc_const( size_byte );

        for( size_T i = 0 ; i < size;  ++ i )
            data[ i ].relocate_shifted( p_h );
        for( size_T i = 0 ; i < size;  ++ i )
            data[ i ].setup           ( p_c );
        memcpy( p_h, ( void * ) data, shift_0 );
        f_dev.f_cc   ( p_c,  p_h ,  size_byte );

        std::free ( p_h );
        delete [  ]  data;
        data = ( val_T * ) p_c;
        return ;
    };
    template < class idx_T >
    __host__ val_T & operator [  ] ( const idx_T & i )
    {
        return data [ i ] ;
    };

    ////////// Device-side functions //////////
    __device__ __forceinline__ operator bool (  ) const
    {
        return ( data != nullptr );
    };
    template < class idx_T >  __device__ __forceinline__
    const val_T & operator [  ] ( const idx_T & i ) const
    {
        return data[ i ];
    };
};
};                              // namespace utils
