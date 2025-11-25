from binary_io  import binary_io
from hydro_data import get_dd, get_last_dd, enroll_interp
from slice_plot import slice
from numpy      import linspace, meshgrid, zeros, array, \
                       pi, copy, cos, sin, sqrt,  floor, \
                       minimum, maximum, log10, log, exp,\
                       histogram, cumsum

############################################################
# Preamble

import matplotlib
import matplotlib.pyplot as plt

matplotlib.rc( 'font', family = 'serif' );
matplotlib.rc( 'text', usetex =  False  );

matplotlib.rcParams[ 'font.size'        ] = 14;
matplotlib.rcParams[ 'axes.labelsize'   ] = 16;
matplotlib.rcParams[ 'legend.fontsize'  ] = 14;
matplotlib.rcParams[ 'legend.edgecolor' ] = 'k';
matplotlib.rcParams[ 'figure.facecolor' ] = 'w';

import numpy as np
np.set_printoptions( edgeitems = 3 );
np.core.arrayprint._line_width = 30;

############################################################
# Constants

h      = 6.62607e-27;   # CGS Planctk constant
kb     = 1.38065e-16;   # CGS Boltzmann constant
eV     = 1.60218e-12;   # CGS eV
c      = 2.99792458e10; # CGS speed of light
q_e    = 4.80321e-10;   # CGS electron charge
me     = 9.1094e-28;    # CGS electron mass;
mp     = 1.67262e-24;   # CGS proton mass;
AU     = 1.49598e13     # Astronomical Unit in cm
yr     = 3.1536e7       # 1 yr in sec
G      = 6.6742831e-8   # CGS graviational constant
sig_sb = 5.6704e-5      # Stefan-Boltzmann
yr     = 365. * 86400.; # Year in secons

modot  = 1.9891e33;     # Solar mass
rodot  = 6.96e10;       # Solar radius
lodot  = 3.828e33;      # Solar luminosity
mearth = 5.9742e27      # Earth mass
rearth = 6.37814e8;     # Earth Radius

############################################################
# Basic dimension recoverage
t0     = 1 * yr;
l0     = 1 * AU;
m0     = modot;
rho0   = m0 / l0**3;
v0     = l0 / t0;
G_code = G / ( l0**3 / t0**2 / m0 );

############################################################
# Basics extra fields

def f_dm( db ):
    res = copy( db[ 'rho' ] );
    for i in range( 3 ):
        res *=  db[  'dx' ] [ i ];
    return res;
#
def f_grav( db, x_c ):
    x   = [ 0, 0, 0 ];
    r_2 = 0;
    for i in range( 3 ):
        x[ i ] = db[  'x' ][ i ] - x_c[ i ];
        r_2   += x [ i ]**2;
    #
    res = [ 0,  0,  0 ];
    for i in range( 3 ):
        res[ i ] = db[ 'dm' ] * x[ i ] / sqrt( r_2 )**3;
    #
    return array( res );
#
def f_vr( db ):
    x   = db[   'x' ];
    v   = db[ 'vel' ];
    res = 0;
    r_2 = 0;
    for i in range( 3 ):
        res += v[ i ] * x[ i ];
        r_2 += x[ i ]**2;
    #
    db[ 'r' ] = sqrt( r_2 );
    return res / db [ 'r' ];
#
def f_vinert( d, db ):
    gm_tot = d.args( "prob", "gm_pri"   )\
           + d.args( "prob", "gm_sec"   );
    dist   = d.args( "prob", "distance" );
    omega  = sqrt  ( gm_tot  / dist**3  );
    
    v  =       db[ 'vel' ]  ;
    vi = copy( db[ 'vel' ] );
    x, y, z  = db[  'x'  ]  ;
    R        = sqrt( x**2 + y**2 );
    vi[ 0 ] -= y * omega;
    vi[ 1 ] += x * omega;
    db[ 'vi_mag' ] = sqrt\
    ( vi[ 0 ]**2 + vi[ 1 ]**2 + vi[ 2 ]**2 );
    db[ 'machi_m1' ] = db[ 'vi_mag' ] / db[ 'c_s' ] - 1;
    return vi;
#

def f_torque( d, db ):
    gm_p = d.args( "prob", "gm_pri_nominal" );
    if gm_p is None:
        gm_p = d.args( "prob", "gm_pri" );
	#
    gm_s = d.args( "prob", "gm_sec"   );
    s    = d.args( "prob", "distance" );
    dx_p = copy( db[ 'x' ] );
    dx_s = copy( db[ 'x' ] );
    x_p  = array ( [ 0, -gm_s / ( gm_p + gm_s ), 0 ] ) * s;
    x_s  = array ( [ 0,  gm_p / ( gm_p + gm_s ), 0 ] ) * s;
    r_p  = 0;
    r_s  = 0;
    mu_s = gm_p * gm_s / ( gm_p + gm_s ) * s;
    for  i    in   range( 3 ):
        dx_p[ i ] -= x_p[ i ];
        dx_s[ i ] -= x_s[ i ];
        r_p += dx_p[ i ] ** 2;
        r_s += dx_s[ i ] ** 2;
    #
    r_p = sqrt( r_p );
    r_s = sqrt( r_s );
    return db[ 'rho' ] * mu_s * ( dx_p[ 0 ] / r_p**3 - \
                                  dx_s[ 0 ] / r_s**3 ) ;
#

def f_parker( r ):
    f = lambda mach : mach**2 - 2 * log( mach ) \
        - 4 * log( r ) - 4 / r + 3;
    print( fsolve( f, 1 ) );
#

############################################################
# Slice plots

def figgen( figsize = None, count = None ):
    if count is None:
        count = figgen.count % 1000;
    figgen.count += 1;
    if figsize is None:
        figsize = ( 10, 7 );
    return plt.figure\
           ( count, figsize = figsize, facecolor = 'w' );
#
figgen.count = 0;

def plot_field( fig, d, f, ** kwargs ):
    args = { 'field' : f, 'xlabel' : r'$x/l_c$', \
             'ylabel' : r'$y/l_c$'};
    for k, v in kwargs.items(  ):
        args[ k ] = v;
    ax = slice( d, args, fig );
    return ax;
#
def f_plot( d, f, ** args):
    if not 'loc' in args:
        args[ 'loc' ] = 0;
    return plot_field( figgen( figsize = ( 6, 5 ) ),\
                       d, f, ** args );
#
def regularize_bb( ax, d, axis = 2 ):
    ax.set_xlim( d.x_lim[ ( axis + 1 ) % 3 ] );
    ax.set_ylim( d.x_lim[ ( axis + 2 ) % 3 ] );
    return;
#

############################################################
# Streamlines

def enroll_interp_safe( d, fields ):
    done = True;
    for f in fields:
        done &= ( 'interp_%s' % f in d.data[ 'block_0' ] );
    if not done:
        enroll_interp ( d, fields );
    return;
#

def get_interp_field \
    ( d, fields = [ 'rho' ], axis = 2, resol = 128 ) :
    if isinstance( fields, str ):
        fields = [ fields ];
    enroll_interp( d, fields, 'nearest' );
    ax_1, ax_2 = ( axis  + 1 ) % 3, ( axis + 2 ) %   3  ;
    xx         = linspace( * d.x_lim[ ax_1 ],    resol );
    yy         = linspace( * d.x_lim[ ax_2 ],    resol );
    xx,  yy    = meshgrid( xx,   yy );
    res        = [ zeros ( xx.shape ) for f in fields ] ;
    for i in range( len( xx ) ):
        x_loc  =  [ 0,  0,  0 ];
        x_loc[ ax_1 ] = xx[ i ];
        x_loc[ ax_2 ] = yy[ i ];
        x_loc[ axis ] = zeros( xx[ i ].shape );
        x_loc = array( x_loc ).T;
        itp = d.interp( x_loc, fields ).T;
        if  len( res ) == 1:
            itp = [ itp ];
        for i_s, r_s in zip( itp, res ):
            r_s[ i ] = i_s;
        #
    #
    if len( res ) == 1:
        res = res[ 0 ];
    return ( xx, yy, res );
#

def enroll_stream_field( d, field = 'vel' ):
    if 'interp_%s|0' % field in d.data[ 'block_0' ]:
        return;
    for i in range( 3 ):
        d.enroll_field( '%s|%d' % ( field,  i ) , \
                        lambda db : db[ field ] [ i ] );
    enroll_interp  \
    ( d, [ '%s|%d' % ( field, i ) for i in range( 3 ) ] );
    return;
#

def get_stream_field \
    ( d, field = 'vel', axis = 2, resol = 128 ) :
    enroll_stream_field( d, field );
    ax_1, ax_2 = ( axis  + 1 ) % 3, ( axis + 2 ) %   3  ;
    v_label    = [ '%s|%d' % ( field,   i ) \
                   for  i in [ ax_1, ax_2 ] ];
    xx, yy, rr = get_interp_field( d, v_label, axis, resol )
    return ( xx, yy, rr[ 0 ], rr[ 1 ] );
#

############################################################
# More specific streamlines

def get_binary_streamlines\
    ( d, field = 'vel', n_start =    8 , \
      h_factor =   0.5, step_m  = 2048 ) :
    dist   = d.args( "prob", "distance"       );
    gm_pri = d.args( "prob", "gm_pri_nominal" );
    gm_sec = d.args( "prob", "gm_sec"         );
    rs_pri = d.args( "prob", "r_sink_pri"     );
    rs_sec = d.args( "prob", "r_sink_sec"     );
    if  gm_pri is  None:
        gm_pri = d.args( "prob", "gm_pri"     );
    gm_tot = gm_pri + gm_sec;

    enroll_stream_field( d, field );
    v_label = [ '%s|%d' % ( field, i ) for i in range( 3 ) ]

    def get_dx( d, x ):
        b = d.data[ d.tree[ tuple( d.find_blk( x ) ) ] ];
        return b[ 'x_f' ][ 0 ][ 1 ] - b[ 'x_f' ][ 0 ][ 0 ]
    #
    x0_pri      = zeros( 3 );
    x0_pri[ 1 ] = - gm_sec / gm_tot * dist;
    x0_sec      = zeros( 3 );
    x0_sec[ 1 ] =   gm_pri / gm_tot * dist;
    xs_all      = [ [  ], [  ] ] ;
    n_start    += 1 ;
    for x0, rs, xs in zip( [ x0_pri, x0_sec ], \
                           [ rs_pri, rs_sec ], xs_all ):
        for phi in linspace( 0, 2 * pi, n_start )[ : -1 ]:
            x_stream = [  ];
            x  = copy( x0 );
            x[ 0 ]  += rs * cos( phi );
            x[ 1 ]  += rs * sin( phi );
            for i in range( step_m ):
                try:
                    h      = get_dx  ( d, x ) * h_factor;
                    v      = d.interp( x, v_label );
                    v[ 2 ] = 0;
                    v     /= sqrt( sum( v**2 ) );
                    x_stream.append( copy( x ) );
                except:
                    break;
                #
                x += v * h;
            #
            xs.append( array( x_stream ).T );
        #
    #
    return xs_all;
#

def plot_binary_streamline\
    ( d, ax, xs_all, lw = 0.5, colors = 'wk' ):
    for xs_one, color in zip( xs_all, colors ):
        for xs in xs_one:
            ax.plot( xs[ 0 ], xs[ 1 ], color + '-', \
                     linewidth = lw );
    return;
#

############################################################
# Torque contours

def get_torq_contours( d, resol = 128 ):
    d.enroll_field( 'torque', lambda db : \
                    abs( f_torque( d, db ) ) );
    torq_min = 1e99;
    torq_max = 0;
    for b, db in d.data.items(  ):
        torq = db[ 'torque' ];
        torq_min = minimum( torq_min, torq.min(  ) );
        torq_max = maximum( torq_max, torq.max(  ) );
    #
    torq_lvl = 10**floor( log10( torq_min ) );
    def f_tp( d, db ):
        torque = f_torque( d, db );
        torque[ torque < 0 ] =  torq_min;
        return log10(  torque ) - log10( torq_lvl )
    #
    def f_tm( d, db ):
        torque = f_torque( d, db );
        torque[ torque > 0 ] = -torq_min;
        return log10( -torque ) - log10( torq_lvl );
    #
    d.enroll_field( 'lg_tp', lambda db : f_tp( d, db ) );
    d.enroll_field( 'lg_tm', lambda db : f_tm( d, db ) );

    tf   = get_interp_field\
        ( d, [ 'lg_tp', 'lg_tm' ], resol = resol );
    t_p  = ( tf[ 0 ], tf[ 1 ], tf[ 2 ][ 0 ] );
    t_m  = ( tf[ 0 ], tf[ 1 ], tf[ 2 ][ 1 ] );
    return ( torq_lvl, t_p, t_m );
#

############################################################
# Spatial convergence of gravitation

def get_spatial_a_converge( d, n_rl = 60 ):
    dist   = d.args( "prob", "distance" );
    gm_pri = d.args( "prob", "gm_pri"   );
    gm_sec = d.args( "prob", "gm_sec"   );    
    y_pri = dist * -gm_sec / ( gm_pri + gm_sec );
    y_sec = dist *  gm_pri / ( gm_pri + gm_sec );
    d.enroll_field( 'dm', f_dm );
    d.enroll_field( 'vr', f_vr );
    d.enroll_field( 'grav_pri', lambda db : f_grav\
                    ( db, [ 0, y_pri, 0 ] ) );
    d.enroll_field( 'grav_sec', lambda db : f_grav\
                    ( db, [ 0, y_sec, 0 ] ) );
    r_min   = maximum( y_pri, y_sec );
    x_max   = array( d.args( "mesh", "x_max" ) );
    r_max   = sqrt( sum( x_max**2 ) );
    rl_arr  = exp( log( linspace( r_min, r_max, n_rl ) ) );
    rl_arr[ 0 ] = 0;
    acc_pri = zeros( len( rl_arr ) - 1 );
    acc_sec = zeros( len( rl_arr ) - 1 );
    for b, db in d.data.items(  ):
        a_p      = db[ 'grav_pri' ][ 0 ];
        a_s      = db[ 'grav_sec' ][ 0 ];
        r        = db[ 'r' ];
        h_p, _   = histogram( r, rl_arr, weights = a_p );
        h_s, _   = histogram( r, rl_arr, weights = a_s );
        acc_pri += h_p;
        acc_sec += h_s;
    #
    return ( ( rl_arr[ 1 : ] + rl_arr[ : -1 ] ) / 2,
             cumsum( acc_pri ), cumsum( acc_sec ) );
#
