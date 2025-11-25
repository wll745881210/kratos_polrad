#pragma once

#include "../../utilities/mapping/mapping.h"
#include "../../types.h"
#include "functors.h"

namespace mesh
{
////////////////////////////////////////////////////////////
// Types

using type::idx_t;

////////////////////////////////////////////////////////////
// Unified 3D data access

template< class f_T >
struct dat_3d_t
{
    ////////// Type //////////
    using f_t = f_T;
    
    ////////// Data //////////
    f_T *       dat;            // Data at current "step"
    bool      proxy;            // Proxy of another dat_3d
    int       n_int;            // Internal dimension
    int      n_ctot;            // Total # of cells
    int      n_step;            // Total # of steps
    idx_t    n_cell;            // # of cells along axes
    idx_t      n_gh;            // Ghost zones
    size_t   n_size;            // Total  size

    ////////// Initializer and finalizer //////////
    __host__ dat_3d_t(  )
    {
        dat    =         nullptr;
        proxy  =            true;
        n_int  =               0;
        n_ctot =               0;
        n_step =               0;
        n_size =               0;
        n_gh   = idx_t::null(  );
        n_cell = idx_t::null(  );
    };
    __host__ void operator = ( const dat_3d_t< f_T > & s )
    {
        memcpy( this,  & s,  sizeof( dat_3d_t< f_T > )   );
        proxy = true;
    };
    __host__ void init
    ( const f_new_t & f_new, const idx_t n_ceff,
      const   idx_t & n_gh   = idx_t::null (  ),
      const   int   & n_int  = 1 ,
      const   int   & n_step = 1 )
    {
        this->n_step = n_step;
        this->n_int  =  n_int;
        n_ctot       =  n_int;
        for( int   i = 0; i < 3; ++ i )
        {
            this->n_gh[ i ] = n_gh[ i ];
            n_cell[ i ]   = n_ceff[ i ]  + 2 * n_gh [ i ];
            n_ctot       *= n_cell[ i ];
        }
        this->n_size = n_step  * n_ctot * sizeof( f_T );
        this->   dat = ( f_T * ) f_new(  this->n_size );
        proxy        =  false  ;
        return;
    };
    __host__ void cp_to
    ( dat_3d_t< f_T > & tgt , const  f_cp_t & f_cp,
      const bool & cp_steps = false ) const
    {
        f_cp( tgt.dat, dat, n_ctot * sizeof( f_T )
                     * ( cp_steps ? n_step : 1 ) );
        tgt.n_int  = this-> n_int;
        tgt.n_ctot = this->n_ctot;
        tgt.n_size = this->n_size;
        tgt.n_step = this->n_step;
        for( int a = 0; a < 3; ++ a )
        {
            tgt.n_cell[ a ] = this->n_cell[ a ];
            tgt.n_gh  [ a ] = this->n_gh  [ a ];
        }
        return;
    };
    template < class fun_T > __host__
    void free( const fun_T & f_free )
    {
        if( ! proxy &&  ( * this ) )
            f_free( dat ) ;
        dat = nullptr;
        return;
    };
    __host__ void  read( const    f_read_t & f_r ,
                         const std::string & pre )
    {
        if( n_ctot != f_r( dat, pre + "field" ) )
            throw std::runtime_error( "Invalid read size" );
        return;
    };
    __host__ size_t  read( const    f_read_t & f_r ,
      const f_new_t & f_n, const std::string & pre )
    {
        auto res = f_r( & n_int,  pre + "n_int"  );
        res     += f_r( & n_step, pre + "n_step" );
        res     += f_r( n_cell.x, pre + "n_cell" );
        res     += f_r( n_gh.x,   pre + "n_gh"   );

        auto n_ceff = n_cell;
        for( int a = 0; a < 3; ++ a )
            n_ceff [ a ] -= n_gh[ a ];
        init( f_n, n_ceff, n_gh, n_int, n_step );
        read( f_r, pre  );
        return res * sizeof( int ) + n_ctot * sizeof( f_T );
    };
    __host__   void write( const   f_write_t & f_w ,
                           const std::string & pre )
    {
        f_w( & n_int,  1, sizeof( int ), pre + "n_int"  );
        f_w( & n_step, 1, sizeof( int ), pre + "n_step" );
        f_w( n_cell.x, 3, sizeof( int ), pre + "n_cell" );
        f_w( n_gh.x,   3, sizeof( int ), pre + "n_gh"   );
        f_w( dat, n_ctot, sizeof( f_T ), pre + "field"  );
        return;
    };

    ////////// Indexing and data access //////////
    template< class idx_T > __device__ __host__
    __forceinline__ int idx( const idx_T & i ) const
    {
#if defined(__CPU_DEBUG__) || defined( __GPU_DEBUG__ )
        for( int a = 0; a < 3; ++ a )
            if( i[ a ] <  -n_gh[ a ] ||
                i[ a ] >= -n_gh[ a ] + n_cell[ a ] )
            {
                PRINT_IAR3( i );      printf( "3d" );
#ifdef     __CPU_DEBUG__
                throw std::out_of_range ( "dat_3d" );
#elif defined( __CUDA_ARCH__ )
                atomicAdd ( device::dbg_flag_c,  1 );
#endif  // __CPU_DEBUG__
            }
#endif  // __CPU_DEBUG__ || __GPU_DEBUG__
        return ( i[ 0 ] + n_gh[ 0 ] + n_cell[ 0 ] *
               ( i[ 1 ] + n_gh[ 1 ] + n_cell[ 1 ] *
               ( i[ 2 ] + n_gh[ 2 ] ) ) ) * n_int ;
    };
    template< class idx_T > __host__ __device__
    __forceinline__ f_T * at( const idx_T & i ) const
    {
        return const_cast< f_T * > ( dat ) + idx( i );
    };
    template< class idx_T > __host__ __device__
    __forceinline__ f_T * at
    ( const idx_T & i,  const int & step ) const
    {
#if defined(__CPU_DEBUG__) || defined( __GPU_DEBUG__ )
        if( step >= n_step )
        {
#ifdef     __CPU_DEBUG__
                throw std::out_of_range( "step" );
#else
                printf( "step" );
#endif  // __CPU_DEBUG__
        }
#endif  // __CPU_DEBUG__ || __GPU_DEBUG__        
        return at( i ) + step * n_ctot;
    };
    template< class idx_T > __host__ __device__
    __forceinline__ f_T & operator(  )
    ( const idx_T & i ) const
    {
        return const_cast< f_T * > ( dat ) [ idx( i ) ];
    };
    __host__ __device__ __forceinline__
    explicit operator bool (  ) const
    {
        return this->dat != nullptr;
    };
};
};                              // namespace mesh
