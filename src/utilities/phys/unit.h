#pragma  once
#include "../../io/args/input.h"
#include "../../types.h"
#include "constants.h"
#include <string>

namespace phys
{
////////////////////////////////////////////////////////////
// Conversion between code and physical units

template< class f_T = type::float2_t >
struct unit_t
{
    ////////// Types //////////
    using  f_t = f_T;

    ////////// Data //////////
    f_t   m0;    f_t   l0;    f_t   t0;
    f_t vel0;    f_t rho0;    f_t ene0;

    ////////// Host-side functions //////////
    __host__ unit_t(  ) : m0  ( 0 ), l0  ( 0 ), t0( 0 ),
                          rho0( 0 ), ene0( 0 )    {   };
    __host__ virtual void init
    ( const input & args, const std::string sec = "unit"  )
    {
        l0 = args.get< f_t > ( sec, "length", 1 );
        t0 = args.get< f_t > ( sec, "time"  , 1 );

        if( args.found( sec, "mass" ) )
        {
            m0 = args.get< f_t > ( sec, "mass" );
            rho0 = m0 / pow( l0, 3 );
        }
        else
        {
            const auto rhos = args.get< std::string >
                ( sec, "density", "" );
            rho0 = ( ( rhos == "mp" || rhos == "mh" )
                     ? cgs::mp : args.get< f_t >
                     ( sec, "density"  , 1 ) );
            m0   = rho0 * pow( l0, 3 );
        }
        ene0 = derive( 1, -1, -2 );
        vel0 = derive( 0,  1, -1 );
        const auto tst = l0 + m0 + t0 + ene0 + vel0;
        if( std::isnan( tst ) || std::isinf( tst ) )
            throw std::runtime_error( "Unit sys overflow" );
        return;
    };
    
    __host__  f_t derive( const f_t & i_m , const f_t & i_l,
                          const f_t & i_t ) const
    {
        using f2_t = double;
        return pow( f2_t( m0 ), i_m ) * pow
            ( f2_t( l0 ), i_l ) * pow( f2_t( t0 ), i_t );
    };
};
};                              // namespace phys::unit
