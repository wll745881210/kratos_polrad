import re

from numpy import copy, average, argmin, pi, meshgrid, \
                  sin, cos, sqrt, log, log10, array, \
                  minimum, maximum, floor

from matplotlib.colors       import Normalize, LogNorm, \
                                    SymLogNorm

from mpl_toolkits.axisartist import floating_axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.projections  import PolarAxes

import matplotlib.pyplot as plt

############################################################
# General

def recognize_vec( field ):
    vflag = re.search( r'\[', field );
    if vflag is None:
        return  None;
    #
    key = re.search( r'.*(?=\[)', field );
    key = key.group(  );
    i   = re.search( r'(?<=%s\[).*\d' % key, field );
    i   = int( i.group(  ) );
    return ( key, i );
#

def slice_generate( x, d_v, axis, loc, coord ):
    x_int = x[ axis ];
    if len( x_int ) > 2 and \
       ( loc > x_int[ -1 ] or loc < x_int[ 0 ] ):
        return None;
    #
    x_mid = ( x_int[ : -1 ] + x_int[ 1 : ] ) / 2.;
    if( len( x_mid ) > 0 ):
        x_int = x_mid;
    idx = argmin( abs( x_int - loc ) );
    if   axis == 0:
        x_s   = x[ 1 ];
        y_s   = x[ 2 ];
        slice = d_v[ :, :, idx ];
    elif axis == 1:
        x_s   = x[ 0 ];
        y_s   = x[ 2 ];
        slice = d_v[ :, idx, : ].T;
        if 'car' in coord:
            slice = slice.T;
        #
    elif axis == 2:
        slice = d_v[ idx, :, : ];
        if 'car' in coord :
            x_s   =  x[ 0 ];
            y_s   =  x[ 1 ];
        elif 'cyl' in coord:
            x_s   = -x[ 1 ] + pi / 2;
            y_s   =  x[ 0 ];
        #
        elif 'sph' in coord:
            x_s   =  x[ 1 ];
            y_s   =  x[ 0 ];
        #
    #
    return ( x_s, y_s, slice );
#

def slice_prepare( dobj, args, fig ):
    if not 'zlim' in args:
        zlim_  = array( [ 1e99, -1e99, 1e99 ] );
        field = args[ 'field' ];
        for b, bd in dobj.data.items(  ):
            if not 'block_' in b:
                continue;
            vflag = recognize_vec( field );
            if vflag is not None:
                dat  = copy( bd[ vflag[ 0 ] ]
                               [ vflag[ 1 ] ] );
            else:
                dat  = copy( bd[ field ] );
            if 'norm' in args:
                dat /= bd[ args[ 'norm' ] ];
            #
            zlim_[ 0 ] = minimum( zlim_[ 0 ], dat.min(  ) );
            zlim_[ 1 ] = maximum( zlim_[ 1 ], dat.max(  ) );
            zlim_[ 2 ] = minimum( zlim_[ 2 ],
                                  abs  ( dat )   .min(  ) );
        #
        if 'z0'  in args:
            zlim_ *= args[ 'z0' ];
        if 'log' in args and \
           isinstance( args[ 'log' ], str ) and\
           'sym' in args[ 'log' ].lower(  ):
            zabs_max = abs( zlim_ ).max(  );
            z_linth  = maximum( zlim_[ 2 ], zabs_max / 1e3 )
            zlim_[ 0 ] = 10**floor( log10( z_linth ) ) ;
            zlim_[ 1 ] = zabs_max;
        #
        zlim = { 'vmin' : zlim_[ 0 ], 'vmax' : zlim_[ 1 ] };
    #
    else:
        zlim = { 'vmin' : min( args[ 'zlim' ] ) ,\
                 'vmax' : max( args[ 'zlim' ] ) };
    #
    if not 'log' in args and zlim[ 'vmin' ] > 0:
        args[ 'log' ] = True;
    #

    axis = args[ 'axis' ];
    if not 'xlim' in args:
        args[ 'xlim' ] = dobj.x_lim[ ( axis + 1 ) % 3 ];
    if not 'ylim' in args:
        args[ 'ylim' ] = dobj.x_lim[ ( axis + 2 ) % 3 ];
    #
    if 'sph' in args[ 'coord' ]:
        args[ 'ylim' ] = pi - array( args[ 'ylim' ] );
    #
    if 'log' in args and args[ 'log' ] != False:
        if   args[ 'log' ] == True:
            norm = LogNorm   ( ** zlim );
        elif  'sym' in args[ 'log' ].lower(  ):
            zlim[ 'linthresh' ] =  zlim[ 'vmin' ];
            zlim[ 'vmin'      ] = -zlim[ 'vmax' ];
            try:
                norm = SymLogNorm( ** zlim, base = 10 );
            except:
                norm = SymLogNorm( ** zlim );
        #
    else:
        norm = Normalize     ( ** zlim );
    #

    if not 'rect' in args:
        args[ 'rect' ] = 111;
    #

    if   'car' in args[ 'coord' ]:
        if 'ax' in args:
            ax = args[ 'ax' ];
        else:
            ax = fig.add_subplot( args[ 'rect' ] );
        aux_ax = ax;
    elif 'sph' in args[ 'coord' ] or \
         'cyl' in args[ 'coord' ] :
        ax, aux_ax = slice_sph_prepare( fig, args );
    #
    return ax, aux_ax, norm;
#

def slice_plot( d_s, args, aux_ax, norm ):
    xs0 = d_s[ 0 ].copy(  );
    xs1 = d_s[ 1 ].copy(  );
    ds  = d_s[ 2 ].copy(  );
    if 'x0' in args:
        xs0 *= args[ 'x0' ];
        xs1 *= args[ 'x0' ];
    if 'z0' in args:
        ds  *= args[ 'z0' ];
    #
    if not 'zlim' in args:
        args[ 'zlim' ] = [ ds.min(  ), ds.max(  ) ];
    #
    if 'abs' in args and args[ 'abs' ]:
        ds = abs( ds );
    if 'log' in args and ( 'sym' != args[ 'log' ]  \
                                and args[ 'log' ] ):
        ds[ ds <= 0 ] = 1e-99;
    #
    if   'car' in args[ 'coord' ]:
        pcm   = slice_basic \
            ( xs0, xs1, ds, args, aux_ax, norm );
    elif 'sph' in args[ 'coord' ] and args[ 'axis' ] == 1:
        pcm   = slice_sph_phi   \
            ( xs0, xs1, ds.T, args, aux_ax, norm );
    elif 'cyl' in args[ 'coord' ]  or \
       ( 'sph' in args[ 'coord' ] and args[ 'axis' ] == 2 ):
        # Note the sequence!
        theta = pi / 2 - xs0;
        pcm   = slice_sph_phi   \
            ( xs1, theta, ds, args, aux_ax, norm );
    #
    return pcm;
#

def annotate_plot( fig, ax, pcm, args ):
    if   'car' in args[ 'coord' ]:
        annotate_car( fig, ax, pcm, args );
    else:
        annotate_sph( fig, ax, pcm, args );
    #
    return;
#

############################################################
# Cartesian coordinates

def slice_basic( x, y, d, args, ax, norm ):
    if not 'cmap' in args:
        # if 'log' in args and args[ 'log' ] == 'sym':
        #     args[ 'cmap' ] = 'nipy_spectral';
        # else:
        args[ 'cmap' ] = 'turbo';
        #
    #

    rasterized = True;
    if 'rasterize' in args and args[ 'rasterize' ]:
        rasterized =  args[ 'rasterize' ];
    #
    return ax.pcolormesh\
        ( x, y, d, norm = norm, cmap = args[ 'cmap' ], \
          antialiased = False, rasterized = rasterized );
#

def annotate_car( fig, ax, pcm, args ):
    if 'labelsize' in args:
        lsize = args[ 'labelsize' ];
    else:
        lsize = 14;
    #
    tsize = lsize - 2;

    if 'xlabel' in args:
        ax.set_xlabel( args[ 'xlabel' ], fontsize = lsize );
    if 'ylabel' in args:
        ax.set_ylabel( args[ 'ylabel' ], fontsize = lsize );
    #

    ax.tick_params( labelsize = tsize );
    ax.set_aspect( 'equal' );

    divider = make_axes_locatable( plt.gca(  ) )
    if  not 'cbar_geo' in args:
        cbar_geo = [ '3%', '2%' ];
    else:
        cbar_geo = args[ 'cbar_geo' ]
    #
    if 'no_cbar' in args and args[ 'no_cbar' ]:
        return;
    #
    cax  = divider.append_axes\
         ( "right", cbar_geo[ 0 ], pad = cbar_geo[ 1 ] );
    cbar = fig.colorbar( pcm, orientation = 'vertical', \
                         cax = cax );
    if 'zlabel' in args:
        cbar.set_label( args[ 'zlabel' ], \
                        size = lsize );
    cbar.ax.tick_params( labelsize = tsize );
    # plt.tight_layout(  );
    return;
#

############################################################
# Spherical-polar coordinates

def slice_sph_prepare( fig, args ):
    fig.subplots_adjust( wspace = 0.5, left = 0.15, \
                         right  = 0.9, top  = 0.9 );

    # extremes = ( max( args[ 'ylim' ] ), \
    #              min( args[ 'ylim' ] ), \
    #              max( args[ 'xlim' ] ), \
    #              min( args[ 'xlim' ] )  );

    # tr = PolarAxes.PolarTransform(  );
    # grid_helper = floating_axes.GridHelperCurveLinear\
    #     ( tr, extremes = extremes );
    # ax = floating_axes.FloatingSubplot\
    #     ( fig, args[ 'rect' ], grid_helper = grid_helper )

    # fig.add_subplot( ax );
    # aux_ax = ax.get_aux_axes( tr );
    # aux_ax.patch = ax.patch;
    # ax.patch.zorder = 0.9;

    ax = fig.add_subplot( 111, projection = 'polar' );

    return ax, ax;
#

def slice_sph_phi( r, theta, d, args, aux_ax, norm ):
    rr, tt = meshgrid( r, theta );
    if ( 'rlog' in args ) and \
       ( not ( args[ 'rlog' ] is None ) ):
        rr = log10( rr / args[ 'rlog' ] );
    return slice_basic( tt, rr, d, args, aux_ax, norm );
#

def annotate_sph( fig, ax, pcm, args ):
    # ax.axis[ 'right' ].set_axis_direction( 'bottom' );
    # ax.axis[ 'right' ].toggle( ticklabels = True, \
    #                            label      = True );
    if 'labelsize' in args:
        lsize = args[ 'labelsize' ];
    else:
        lsize = 14;
    #
    tsize = lsize;

    plt.setp( ax.get_xticklabels(  ), visible = False );
    plt.setp( ax.get_yticklabels(  ), visible = False );

    if 'xlabel' in args:
        ax.set_xlabel( args[ 'xlabel' ] );
    if 'ylabel' in args:
        ax.set_ylabel( args[ 'ylabel' ] );
    # For the new scheme "xlim" is for theta
    # and "ylim" is for radius.

    if args[ 'axis' ] == 2:
        if 'xlim' in args:
            ax.set_xlim  ( args[ 'ylim' ] - pi / 2 );
        if 'ylim' in args:
            ax.set_ylim  ( args[ 'xlim' ] );
    #

    # ax.axis[ 'right'  ].label.set_size           ( lsize )
    # ax.axis[ 'right'  ].major_ticklabels.set_size( tsize )
    # ax.axis[ 'left'   ].label.set_size           ( lsize )
    # ax.axis[ 'left'   ].major_ticklabels.set_size( tsize )

    cbaxes = fig.add_axes( [ 0.26, 0.99, 0.55, 0.02 ] );
    cbar = fig.colorbar( pcm, orientation = 'horizontal', \
                         cax = cbaxes );
    if 'zlabel' in args:
        cbar.set_label( args[ 'zlabel' ], size = lsize );
        cbar.ax.tick_params( labelsize = tsize );
    #
    return;
#

############################################################
# Interface

def slice( dobj, args, fig ):
    field = args[ 'field' ];
    if not 'axis' in args:
        args[ 'axis' ] = 2;
    if not 'loc'  in args:
        args[ 'loc'  ] \
            = average( dobj.x_lim[ args[ 'axis' ] ] );
    #
    if not 'coord' in args:
        args[ 'coord' ] = 'cartesian';
    #

    ax, aux_ax, norm = slice_prepare( dobj, args, fig );

    for block, b_data in dobj.data.items(  ):
        if not 'block_' in block:
            continue;
        vflag = recognize_vec( field );
        if  vflag is not None:
            key, i = vflag;
            d_v = copy( b_data[ key ][ i ] );
        elif field in b_data:
            d_v = copy( b_data[ field    ] );
        else:
            raise ValueError( "%s is undefined" % field );
        #
        if 'norm' in args and args[ 'norm' ] is not None:
            d_v /= b_data [   args[ 'norm' ] ]
        #
        d_s = slice_generate\
              ( b_data[ 'x_f'   ], d_v, \
                args  [ 'axis'  ], args[ 'loc' ], \
                args  [ 'coord' ] );
        if d_s is None:
            continue;
        #
        pcm = slice_plot( d_s, args, aux_ax, norm );
    #

    annotate_plot( fig, ax, pcm, args );
    return ax, aux_ax;
#
