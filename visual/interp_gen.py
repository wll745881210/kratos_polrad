from binary_io import *
from numpy import zeros, ones, linspace, meshgrid, transpose

############################################################
# Generate the 

def interp_reg_save( file_name, x0, dx, dat, prefix = '', \
                     int_type   =   'int32' ,\
                     float_type = 'float32' ):
    if len( dat.shape ) != len( x0 ) or\
       len( dat.shape ) != len( dx ) :
        raise ValueError( "Inconsistent dimensions" );

    bio = binary_io( file_name )
    n   = list( dat.shape )[ : : -1 ];
    bio.cache( prefix + 'n_pts',  n, dtype =   int_type );
    bio.cache( prefix + 'x0',    x0, dtype = float_type );
    bio.cache( prefix + 'dx',    dx, dtype = float_type );
    bio.cache( prefix + 'data', dat, dtype = float_type );
    bio.save (  );
    return;
#

def interp_reg_gen\
    ( file_name,  x0, x1, n, func, prefix = [ '' ],
      int_type = 'int32', float_type = 'float32' ):
    if  len( n ) != len( x0 ) or len( n ) != len( x1 ):
        raise ValueError ( "Inconsistent sizes" );
    if  not hasattr( func, '__len__' ):
        func = [ func ];
    if  len( prefix ) != len( func ):
        raise ValueError ( "Inconsistent function prefix" );
    #
    x_arr =  [   ];
    dx    =  zeros( len( n ) );
    for a in range( len( n ) ):
        xs  = linspace( x0[ a ], x1[ a ], int( n[ a ] ) );
        x_arr.append( xs );
        dx[ a ] = xs[ 1 ] - xs[ 0 ];
    #
    z, y, x = meshgrid( * x_arr, indexing = 'ij' );

    bio = binary_io( file_name )
    for f, p, in zip( func, prefix ):
        dat = f( x, y, z ).T
        n   = list( dat.shape )[ : : -1 ];
        bio.cache( p + 'n_pts',  n, dtype =   int_type );
        bio.cache( p + 'x0',    x0, dtype = float_type );
        bio.cache( p + 'dx',    dx, dtype = float_type );
        bio.cache( p + 'data', dat, dtype = float_type );
    bio.save (  );
    bio.close(  );
    return;
#
