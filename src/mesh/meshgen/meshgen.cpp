#include "../../utilities/mapping/loop.h"
#include "../../mesh/geometry/geometry.h"
#include "../../mesh/mesh.h"
#include "meshgen.h"

#include <cmath>

namespace mesh::meshgen
{
////////////////////////////////////////////////////////////
// Initialization

void base_t::read( const input & args )
{
    x_lim_global[ 0 ]
        = args.get< coord2_t > ( "mesh", "x_min" );
    x_lim_global[ 1 ]
        = args.get< coord2_t > ( "mesh", "x_max" );
    n_ceff_global= args.get<    idx_t >
        ( "mesh", "n_cell_global" );
    n_ceff_block = args.get<    idx_t >
        ( "mesh", "n_cell_block", n_ceff_global );

    n_dim  = 0 ;
    for( int a = 0; a < 3 ;    ++ a )
        if ( n_ceff_global[ a ] > 1 )
            n_dim = a + 1 ;
    for( int a = 0; a < n_dim; ++ a )
    {
        if( n_ceff_global[ a ] == 0 )
            throw std::runtime_error( "Zero mesh size"  );
        if( n_ceff_global[ a ] % n_ceff_block[ a ] != 0 )
            throw std::runtime_error
                ( "Mesh size indivisible by sub-mesh."  );
        i_logic_lim[ a ] = n_ceff_global[ a ]
                         / n_ceff_block [ a ];
    }
    geo_regenerate = args.get< bool >
        ( "mesh",  "geo_regenerate",   false );

    return;
}

void base_t::init( const input & args, mesh_t & mesh )
{
    this->read( args );
    mesh. tree(  ).  n_dim  =       n_dim;
    mesh. tree(  ).reg_max  = i_logic_lim;
    mesh. tree(  ).periodic = { false, false, false };

    std::vector< std::string > bnd_kinds  ;
    args( "boundary", "kinds", bnd_kinds );
    for( int side = 0; side < bnd_kinds.size(  ); ++ side )
        mesh.tree(  ).periodic[ side / 2 ] = bnd_kinds
             [ side ].find( "per"  ) != std::string::npos;

    dist_block = [ & ]     ( mesh_t  & mesh )
    {
        std::vector <  float2_t  >            bal_coef  ;
        args( "mesh", "balance_coef",         bal_coef );
        const auto &   size = mesh.tree(  ).size_eff(  );
        const auto & n_rank = mesh.p_com  ->  n_rank(  );
        if( n_rank > size )
            throw std::runtime_error
                ( "There are more ranks than blocks." );
        if( bal_coef . size(  ) < n_rank )
        {
            bal_coef .clear(  );
            for( int  r = 0;  r < n_rank; ++ r )
                bal_coef.push_back( 1 );
        }
        float2_t norm( 0 );
        for( auto & b : bal_coef )
            norm += b ;
        for( auto & b : bal_coef )
            b /= norm ;

        auto res( size ) ;
        for( int r = 0; r < n_rank; ++ r )
        {
            per_rank.push_back( bal_coef.at( r ) * size );
            res -= per_rank.back(  );
        }
        for( int r = 0; r < res; ++ r )
            ++ per_rank[ r ];
        if( args.get< int >( "mesh", "dist_mode", 1 ) == 0 )
            this->dist_default      ( mesh );
        else
            this->dist_fractal      ( mesh );
    };
    if( ! mesh.is_restart(  ) )
    {
        load_refinement( args );
        generate_tree  ( mesh );
    }
    else
    {
        if( mesh.p_com->is_root(  ) )
        {
            mesh.p_bio->read ( mesh.tree(  ), "mesh_tree" );
            dist_block       ( mesh );
        }
        mesh.p_com->bcast    ( mesh.tree(  ) );
    }
    return create( mesh );
}

////////////////////////////////////////////////////////////
// Default block distribution (No load balancing)

void base_t::load_refinement( const input & args )
{
    for( const auto & p :  args.get_prefixes(  ) )
    {
        if( p.find( "refine_region" ) == std::string::npos )
            continue;
        ref_regions     .emplace_back(  );
        auto & reg = ref_regions.back(  );

        reg. level = args.get< int > ( p, "level", 0 );
        reg.x[ 0 ] = coord_t::null(  );
        args( p,   "loc", reg.x[ 0 ] );
        args( p, "x_min", reg.x[ 0 ] );
        reg.x[ 1 ] =      reg.x[ 0 ]  ;
        args( p, "x_max", reg.x[ 1 ] );

        for( int l = 0; l < 2; ++ l )
        {
            for( int n = 0; n < n_dim; ++ n )
            {
                reg.i[ l ][ n ]  =    search_blk_idx
                   ( reg.x[ l ][ n ], reg.level, n );
                if ( reg.i[ l ][ n ] < 0 )
                    throw std::runtime_error
                          ( "Incorrect SMR region" );
            }
            for( int n = n_dim; n < 3; ++ n )
                reg.i[ l ][ n ] = 0;
        }
    }
    return;
}

void base_t::generate_tree( mesh_t & mesh )
{
    if( mesh.p_com->is_root(  ) )
    {
        region_logic_t       reg ;
        for(  const  auto  & ref : ref_regions )
            if( ref. level > 0 )
                for( const auto & i : utils::loop
                         ( ref.i[ 0 ], ref.i[ 1 ], true ) )
                {
                    reg          =         i  ;
                    reg.level    = ref.level  ;
                    mesh.tree(  ).insert( reg );
                }
        mesh.tree(  ).fill_base_level(  );
        dist_block               ( mesh );
    }
    mesh.p_com->bcast( mesh.tree (    ) );
    return;
}

////////////////////////////////////////////////////////////
// Load balancing

void base_t::dist_default( mesh_t & mesh )
{
    int  id_g ( 0 ) , id_l( 0 ), rank( 0 );
    for( auto & p : mesh.tree (  ) )
    {
        if( p.second.mask )
            continue;
        p.second.rank = rank;
        p.second.id_g = id_g;
        p.second.id_l = id_l;
        ++  id_g;
        ++  id_l;
        if( id_l >= per_rank[ rank ] &&
            rank != per_rank. size (  ) - 1 )
        {
            id_l  = 0;
            rank += 1;
        }
    }
    return;
}

void base_t::dist_fractal( mesh_t & mesh )
{
    int  n_dim ( 0 ),   n_base( 0 );
    for( int a = 0; a < n_ceff_global.size(  ); ++ a )
    {
        if( n_ceff_global[ a ] > 1 )
            n_dim = a + 1;
        n_base = std::max( n_base,  i_logic_lim[ a ] );
    }
    if( n_dim  < 1 )            // Zero-dimensional dot
        return this->dist_default( mesh );

    --     n_base;
    int    l_base( 0 );
    while( n_base >>= 1 )
        ++ l_base;
    ++     l_base;

    if     ( n_dim == 3 )       // 3D Hilbert curve
        l.rule[ 'X' ] = "^<XF^<XFX-F^>>XFX&F+>>XFX-F>X->";
    else if( n_dim == 2 )       // 2D Hilbert curve
    {
        l.rule[ 'A' ] = "+BF-AFA-FB+";
        l.rule[ 'B' ] = "-AF+BFB+FA-";
    }
    else if( n_dim == 1 )       // 1D is trivial
        l.rule[ 'X' ] = "F";
    else
        throw std::runtime_error( "Incorrect dimension" );

    int id_g ( 0 ), id_l( 0 ), rank( 0 );
    l.rule_ref = [ & ] ( auto & n )
    {
        mesh::region_logic_t  r;
        for( int a = 0; a < n.x.size(  ); ++ a )
            r[ a ] = n. x [ a ];
        r . level  = n. l -   l_base;
        if( mesh.tree(  ).contains( r ) )
        {
            auto & tn = mesh.tree(  ).at( r );
            if( tn.mask )
                return true;
            tn.rank  = rank;
            tn.id_g  = id_g;
            tn.id_l  = id_l;
            ++ id_g;
            ++ id_l;
            if( id_l >= per_rank[ rank ] &&
                rank != per_rank. size (  ) - 1 )
            {
                id_l = 0;
                ++  rank;
            }
        }
        return false;
    };
    return l.generate( l_base );
}

////////////////////////////////////////////////////////////
// Default geometry functions

float2_t base_t::location
( const float2_t & x_logic, const  int & axis ) const
{
    const  auto & x_lim = x_lim_global;
    return x_lim[ 0 ][ axis ] + x_logic *
         ( x_lim[ 1 ][ axis ] - x_lim [ 0 ][ axis ] );
}

float2_t base_t::surface( const idx_t &  idx ,
       const geo_t & geo, const   int & axis ) const
{
    float2_t res( 1. );
    for( int n = 1; n < 3; ++ n )
    {
        const int & l = ( axis  + n )  % 3;
        res          *= geo.dx_f( l, idx );
    }
    return res;
}

float2_t base_t::volume
( const idx_t & idx, const geo_t & geo ) const
{
    float2_t res ( 1. );
    for( int n = 0; n < 3; ++ n )
        res *= geo.dx_f( n, idx );
    return res;
}

void base_t::set_uniformity( block_t & b )
{
    return;  // Do nothing --> Uniform by default
}

void base_t::set_coord_axes( block_t & b )
{
    auto & geo( b.geo );
    auto & reg( b.reg );

    for( int a = 0; a < 3; ++ a )
    {
        auto loc  = [ & ] ( float2_t i ) -> float2_t
        {
            if( geo.n_ceff[ a ] == 1 )
                return i; // [0, 1) Even at higher level
            i  = ( reg[ a ] + i / geo.n_ceff[ a ] )
               / ( i_logic_lim[ a ]  << reg.level );
            return location( i, a );
        };
        geo.xf0[ a ] = loc( 0 ) ;
        if( geo.is_uniform[ a ] )
        {
            geo.dx0[ a ] = loc( 1 ) - geo.xf0[ a ];
            continue;
        }
        else
            geo.dx0[ a ] = -1;
        auto *  xf =  geo.xf[ a ] + geo.n_gh;
        auto *  xv =  geo.xv[ a ] + geo.n_gh;
        for( int i = -geo.n_gh;
                 i <= geo.n_ceff[ a ] + geo.n_gh; ++ i )
        {
            xf[ i ]  = loc ( ( float2_t )  i       );
            xv[ i ]  = loc ( ( float2_t )  i + 0.5 );
        }
    }
    return;
}

void base_t::set_geo_fulldim( block_t & b )
{
    auto & geo( b.geo ) ;
    auto & reg( b.reg ) ;
    for( int a = 0; a < geo.n_dim; ++ a )
    {
        const auto n_ceff = geo.n_ceff[ a ];
        const auto n_mid  = n_ceff / 2;

        reg.x_lim[ 0 ][ a ] =   geo.x_fc( a,      0 );
        reg.x_lim[ 1 ][ a ] =   geo.x_fc( a, n_ceff );
        reg.x_mid     [ a ] = ( geo.x_fc( a, n_mid   - 1 )
                              + geo.x_fc( a, n_mid ) ) / 2;
    }
    for( int a = 0; a < geo.n_dim; ++ a )
    {
        if( geo.sfc0[ a ] > 0 )
        {
            geo.sfc0[ a ] = surface( idx_t( 0 ), geo, a );
            continue;
        }
        for( auto i : utils::loop( geo.sfc[ a ].n_cell ) )
            geo.sfc[ a ]( i ) = surface( i, geo, a );
    }
    if( geo.vol0 > 0 )
        geo.vol0 = volume( idx_t::null(  ), geo );
    else
        for( auto i : utils::loop( geo.vol.n_cell ) )
            geo.vol( i ) = volume( i, geo );
    return;
}

int base_t::search_blk_idx
( const float2_t & loc, const int & lvl, const int & axis )
{
    if( loc > x_lim_global[ 1 ][ axis ] ||
        loc < x_lim_global[ 0 ][ axis ]  )
        return -1;

    const  int   i_tot = i_logic_lim[ axis ] << lvl;
    int    i_r = i_tot ;
    int    i_l =     0 ;
    const  float2_t  i_ftot( i_tot );
    while( i_r - i_l > 1 )
    {
        const int     i_m = ( i_l +  i_r ) / 2;
        if( location( i_m / i_ftot, axis ) < loc )
            i_l = i_m;
        else
            i_r = i_m;
   }
    return  i_l;
}

////////////////////////////////////////////////////////////
// Interface

void base_t::associate_mesh( block_t & b, mesh_t & mesh )
{
    b.p_mesh = ( & mesh );
    auto & geo ( b. geo );
    for( const auto & [ nb, reg ] : b.neighbors )
        if( nb.mode == 0 && nb.d_lvl !=   0 )
            for( int a = 0; a < 3;   ++   a )
                if( nb[ a ] < 0 ||   nb [ a ] > 1 )
                  ( nb[ a ] > 1 )  [ nb.d_lvl > 0 ?
                    geo.refine_flag[ a ] :
                    geo.coarse_flag[ a ] ] = true;

    const auto & reg_lim
        = mesh.tree(  ).region_max( b.reg.level );
    for( int a = 0; a < 3; ++ a )
        if( ! mesh.tree(  ).periodic[ a ] )
        {
            geo.phybnd_flag[ a ][ 0 ] = ( b.reg[ a ] == 0 );
            geo.phybnd_flag[ a ][ 1 ]
                = ( reg_lim[ a ]  - 1 ==  b.reg[ a ] );
        }
    return;
}

void base_t::create( mesh_t & mesh )
{
    for( auto &  p : mesh.tree(  ) )
    {
        if( p.second.mask ||
            p.second.rank != mesh.p_com->rank(  ) )
            continue;

        auto &    b = mesh.           emplace( p.first );
        b.neighbors = mesh.tree(  ).neighbors( p.first );
        b.reg     = p.      first;
        b.id_l    = p.second.id_l;
        b.id_g    = p.second.id_g;
        b.rank    = p.second.rank;
        associate_mesh( b, mesh );
    }
    for( auto & b : mesh )
    {
        if( ! mesh.is_restart(   )  || geo_regenerate )
        {
            set_uniformity( b );
            b.geo.setup( n_ceff_block, mesh.n_order(  ),
                         mesh.p_dev->  f_malloc_host  );
            set_coord_axes( b );
        }
        else
            b.read( * mesh.p_bio, * mesh.p_dev );
        set_geo_fulldim  ( b );
    }
    return;
}

void base_t::save( mesh_t & mesh, binary_io::base_t & bio )
{
    if( mesh.p_com->is_root(  ) )
        bio.write ( mesh.tree(  ), "mesh_tree" );
    for( auto & b : mesh )
        b.write( bio );
    return;
}

void base_t::update( mesh_t & mesh )
{
    return;
}   // Interface for AMR

};                              // namespace mesh::meshgen
