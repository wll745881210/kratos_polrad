from binary_io   import binary_io
from numpy       import frombuffer, zeros, array, meshgrid,\
                        sqrt, log2, sum, minimum, maximum, \
                        copy, log10, arange, cbrt, unique,\
                        concatenate
from glob        import glob
from bisect      import bisect
from collections import OrderedDict
from scipy. interpolate import RegularGridInterpolator

import types

try:
    from configparser import ConfigParser
except ImportError:
    from ConfigParser import ConfigParser  # Python ver < 3
#

############################################################
# Data class for hydrodynamics

class hydro_data:
    ########################################################
    # Initialization and finalization
    def __init__( self, file_name, read_now = True, \
                  cache_used = True ):
        cache_used    = cache_used or read_now;
        self.bin_data = binary_io( file_name, cache_used );
        self.bin_data . open(  );

        xl_0          = [ 1e99, -1e99 ];
        self.x_lim    = array( [ xl_0, xl_0, xl_0 ] );
        self.data     = dict(  );
        self.globals  = dict(  );        
        if read_now:
            self.read(  );
        #
        return;
    #
    def close( self ):
        self.bin_data.close(  );
    #
    def __enter__( self ):
        return self;
    #
    def __exit__ ( self, etype, evalue, traceback ):
        self.close(  );
        if etype is not None:
            print( etype, evalue, traceback );
        #
    #
    ########################################################
    # Get/parse data
    def get_entry( self, key, dtype ):
        u_size = self.bin_data.hmap[ key ][ 1 ];
        return frombuffer( self.bin_data[ key ], dtype = \
                           '<%s%d' % ( dtype, u_size ) ) ;
    #
    def get_field( self, block, field, datfld_name = None,
                   load_raw = False, dtype = 'f' ):
        f_name = '%s|%s'  % ( block, field );
        n_i = self.get_entry( f_name + '_n_int' , 'i'   );
        n_g = self.get_entry( f_name + '_n_gh'  , 'i'   )\
              [ : : -1 ];
        n_c = self.get_entry( f_name + '_n_cell', 'i'   )\
              [ : : -1 ];
        fld = self.get_entry( f_name + '_field' , dtype )\
                  .  reshape( ( * n_c, -1 ) );
        if  not load_raw:
            fld = fld[ n_g[ 0 ] : n_c[ 0 ] - n_g[ 0 ] ,
                       n_g[ 1 ] : n_c[ 1 ] - n_g[ 1 ] ,
                       n_g[ 2 ] : n_c[ 2 ] - n_g[ 2 ] ];
        #
        if fld.shape[ -1 ] != n_i:
            raise ValueError( "Inconsistent n_internal" );
        #
        res = array( [ fld[ :, :, :, n ] for n in \
                       range( fld.shape[ -1 ] ) ] ) ;
        if  datfld_name is None:
            return res;
        self.data[ block ][ datfld_name ] = res;
        return;
    #
    def prep_block( self ):
        for key in self.bin_data.hmap:
            if not 'block_' in key:
                continue;
            #
            blk = key.split( '|' )[ 0 ];
            if not blk in self.data:
                self.data[ blk ] = dict(  );
            #
        #
        return;
    #
    def read_block_geometry( self, blk ):
        dat          = self. data[ blk ];
        dat[ 'x_f' ] = [ 0, 0, 0 ];
        dat[ 'x_c' ] = [ 0, 0, 0 ];
        n_c      = self.get_entry( '%s|n_ceff' % blk, 'i' );
        uniform = self.get_entry( '%s|uniform' % blk, 'b' );
        n_gh  = self.get_entry( '%s|n_gh' % blk, 'i' )[ 0 ];
        try:
            dat[ 'id_g'    ] = self.get_entry\
                ( '%s|id_g'    % blk, 'i' )[ 0 ];        
            dat[ 'i_logic' ] = self.get_entry\
                ( '%s|i_logic' % blk, 'i' );
            dat[ 'level'   ] = self.get_entry\
                ( '%s|level'   % blk, 'i' )[ 0 ];
        except:
            pass;
        #

        def f_set_uniform( i ):
            xf0 = self.get_entry( '%s|xf0' % blk, 'f' );
            dx0 = self.get_entry( '%s|dx0' % blk, 'f' );
            dat[ 'dx0' ] = dx0;
            xv  = xf0[ i ] + dx0[ i ] \
                * ( arange ( n_c[ i ] ) + 0.5 );
            xf  = xf0[ i ] + dx0[ i ] \
                * ( arange ( n_c[ i ] + 1 )   );
            return xv, xf;
        #
            
        for i in range( 3 ):        
            if not uniform[ i ]:
                try:
                    xv = self.get_entry\
                        ( '%s|xv_%d' % ( blk, i ), 'f' );
                    xf = self.get_entry\
                        ( '%s|xf_%d' % ( blk, i ), 'f' );
                    if n_gh > 0:
                        xv = xv[  n_gh : -( n_gh + 1 ) ];
                        xf = xf[  n_gh : -  n_gh       ];
                    else:
                        xv = xv[ : -1 ];                    
                except:
                    xv, xf = f_set_uniform( i );
                #
            else:
                xv, xf = f_set_uniform( i );
            #
            
            dat[ 'x_c' ][ i ] = xv;
            dat[ 'x_f' ][ i ] = xf;
            try:
                self.x_lim[ i ][ 0 ] = minimum\
              ( self.x_lim[ i ][ 0 ] , xf. min(  ) );
                self.x_lim[ i ][ 1 ] = maximum\
              ( self.x_lim[ i ][ 1 ] , xf. max(  ) );
            except:
                print( blk );
                raise ValueError;
        #
        self.n_dim  = sum( n_c > 1 );
        dat[ 'n_cell' ] =        n_c;
        dat[ 'n_face' ] = 1   +  n_c;
        dat[ 'n_gh'   ] =       n_gh;
        dat[ 'n_ceff' ] = n_c - n_gh;
        return;
    #
    def read_block_data( self, blk ):
        dat = self.data[ blk ];
        if  not 'n_cell' in dat:
            self.read_block_geometry( blk );
        #
        bd = self.get_field( blk, 'hydro_cons' );
        dat[ 'rho' ] = bd[ 0 ];
        dat[ 'ene' ] = bd[ 1 ];
        dat[ 'mom' ] = bd[ 2 : 5 ];
        # dat[ 'rho' ] = bd[ :, :, :, 0 ];
        # dat[ 'ene' ] = bd[ :, :, :, 1 ];
        # dat[ 'mom' ] = array( [ bd[ :, :, :, 2 ], \
        #                         bd[ :, :, :, 3 ], \
        #                         bd[ :, :, :, 4 ]  ] );
        # if   bd.shape[ 3 ] > 7:
        #     dat[ 'bcc' ] = array( [ bd[ :, :, :, 5 ], \
        #                             bd[ :, :, :, 6 ], \
        #                             bd[ :, :, :, 7 ]  ] );
        # elif bd.shape[ 3 ] > 5:
        #     dat[ 'ent' ] = bd[ :, :, :, 5 ];
        #
        return;
    #
    def read_geometry( self ):
        if not self.data:
            self.prep_block(  );
        #        
        for blk in self.data:
            self.read_block_geometry( blk );
        #
        return;
    #    
    def read_data( self ):
        if not self.data:
            self.prep_block(  );
        #
        for blk in self.data:
            self.read_block_data    ( blk );
        #
        return;
    #
    def read_args( self ):
        self.args_data = dict(  );
        bin_args    = self.bin_data[ 'input_args' ];
        if len( bin_args ) == 0:
            return;
        #
        offset    = 0;
        def __get_bin( length ):
            nonlocal offset;
            offset += length;
            return bin_args[ offset - length : offset ];
        #
        def __get_size_t(  ):
            return int.from_bytes\
            ( __get_bin( self.bin_data.s_size_t ), \
              self.bin_data.endian, signed = False );
        #
        def __get_string(  ):
            len_str = __get_size_t(  );
            bin_str = __get_bin( len_str );
            return bin_str.decode( 'ascii' );
        #
        len_tot   = __get_size_t(  );
        for i in range( len_tot ):
            key  = __get_string(  ).split( '|' );
            item = __get_string(  ).split(     );
            if  len( key ) == 1:
                self.args_data[ key[ 0 ] ] = item;
                continue;
            elif not key[ 0 ] in self.args_data:
                self.args_data[ key[ 0 ] ] = dict(  );
            #
            self.args_data[ key[ 0 ] ][ key[ 1 ] ] = item;
        #
        return;
    #
    def args( self, sec, key = None ):
        if key is None:
            key = sec;
            sec = "";
        #
        try:
            item = self.args_data[ sec ][ key ];
        except:
            return None;
        #
        res  = [  ];
        for it in item:
            try:
                it = int( it );
            except ValueError:
                try:
                    it = float( it );
                except ValueError:
                    pass;
                #
            #
            res.append( it );
        #
        if len( res ) == 1:
            res = res[ 0 ];
        return res;
    #
    def save_args( self, file_name ):
        config = ConfigParser(  );
        config.optionxform =  str;
        for sec, sec_item in self.args_data.items(  ):
            config.add_section( sec );
            for key, item in sec_item.items(  ):
                config.set( sec, key, ' '.join( item ) );
        with open( file_name, 'w' ) as f:
            config.write( f );
        #
        return;
    #
    def read( self ):
        self.globals[ 'time' ] =  0.0;
        if 'time' in self.bin_data.hmap:
            self.globals[ 'time' ] \
                = self.get_entry( 'time', 'f' )[ 0 ];
        #
        self.read_geometry(  );
        self.read_args    (  );
        self.read_data    (  );
        self.close        (  );
        return;
    #
    ########################################################
    # Data access
    def __getitem__( self, key ):
        if key == 'time':
            return self.globals[ 'time' ];
        if key == 'gam1':
            return self.globals[ 'gam1' ];
        #
        if not key in self.data:
            if not self.data:
                self.prep_block(  );
            self.read_block_geometry( key );
        #
        return self.data[ key ];
    #
    def keys ( self ):
        return self.data[ 'block_0' ].keys(  );
    #
    ########################################################
    # Enroll data
    def enroll_field( self, name, func ):
        for blk, dat in self.data.items(  ):
            if  not 'block' in blk:
                continue;
            self.data[ blk ][ name ] = func( dat );
        #
        return;
    #
#

############################################################
# Wrapper and useful tools

def enroll_prim_single( bd, d ):
    xc = bd[ 'x_c' ];
    dx =   [       ];
    for i, dx_ in enumerate( bd[ 'x_f' ] ):
        dx.append( dx_[ 1 : ] - dx_[ : -1 ] );
    #
    y,  z,  x  = meshgrid( xc[ 1 ], xc[ 2 ], xc[ 0 ] );
    dy, dz, dx = meshgrid( dx[ 1 ], dx[ 2 ], dx[ 0 ] );
    bd[ 'x'  ] = array( (  x,  y,  z ) );
    bd[ 'dx' ] = array( ( dx, dy, dz ) );

    if 'gam1' in bd:  # Local gam1 has highest priority
        gam1 = bd[ 'gam1' ];
    else:
        gam1 = d.args( 'dynamics', 'gamma' );
        if gam1 is not None:
            gam1 -= 1.;
        else:
            print( "Unable to find [ gamma - 1 ]; " \
                   "using gamma = 5/3" );
            gam1 = 5. / 3. - 1;
        #
        bd[ 'gam1' ] = gam1;            
    #
    bd[ 'vel' ] = bd[ 'mom' ] / bd[ 'rho' ];
    bd[ 'kin' ] = sum ( bd[ 'mom' ]**2, axis = 0 )\
                / ( 2 * bd[ 'rho' ] );
    # if 'ent' in bd:
    #     bd[ 'pre' ] = bd[ 'ent' ] * bd[ 'rho' ]**gam1;
    # else:
    bd[ 'pre' ] = gam1 * \
                    ( bd[ 'ene' ] - bd[ 'kin' ] );
    t_floor = d.args( 'dynamics', 'tmp_floor' );
    if  t_floor is None:
        t_floor = 0;
    p_floor = bd[ 'rho' ] * t_floor;
    p_flag  = bd[ 'pre' ] < p_floor;
    bd[ 'pre' ][ p_flag ] = p_floor[ p_flag ];

    bd[ 'c_s' ] = sqrt( ( bd[ 'gam1' ] + 1 ) \
                      * ( bd[ 'pre' ] / bd[ 'rho' ] ) );
    bd[ 'v_mag' ] = sqrt\
                  ( sum( bd[ 'vel' ]**2, axis = 0 ) );
    bd[ 'mach_m1' ] = bd[ 'v_mag' ] / bd[ 'c_s' ] - 1;
    return;
#

def enroll_prim_quantities( d ):
    for b, bd in d.data.items(  ):
        enroll_prim_single ( bd, d );
    return;
#

def enroll_mesh_tree( d ):
    d.n_base = array( d.args( "mesh", "n_cell_global" ) )
    for b, db in d.data.items(  ):
        try:
            lvl = d.get_entry(   '%s|level' % b, 'i' )[ 0 ];
            i_l = d.get_entry( '%s|i_logic' % b, 'i' );
        except:
            dz  = db[ 'x_f' ][ 2 ][ -1 ] \
                - db[ 'x_f' ][ 2 ][  0 ] ;
            i_l = [ 0,  0,  0 ];
            for i in range( 3 ):
                dx_b = d.x_lim[ i ][ 1 ] - d.x_lim[ i ][ 0 ]
                i_l[ i ]  = ( db[ 'x_f' ][ i ][ 0 ] + 1e-4 \
                            - d.x_lim[ i ][ 0 ]  ) / dx_b;
            #
            n_b = d.n_base / db[ 'n_cell' ];
            lvl = log2 ( dx_b / dz / n_b[ 2 ] );
            i_l = array( i_l ) * 2**( lvl ) * n_b;
        #
        db[   'level' ] = int  ( lvl );
        db[ 'i_logic' ] = array( i_l, dtype = 'int' );
    #
    d. xf_idx = [ {  }, {  }, {  } ];
    d.lvl_max = [ 0, 0, 0 ];
    d.tree    = {         };
    for b, db in d.data.items(  ):
        x_min = array( [ xf[ 0 ] for xf in db[ 'x_f' ] ] );
        for i in range( 3 ):
            lvl = db[ 'level' ];
            if not x_min[ i ] in d.xf_idx[ i ]:
                d.xf_idx[ i ] [ x_min[ i ] ] = lvl;
            d.xf_idx[ i ] [ x_min[ i ] ] = maximum\
                ( lvl,  d. xf_idx[ i ] [  x_min[ i ]  ]   );
            d.lvl_max[ i ] = maximum( d.lvl_max[ i ], lvl );
        #
        d.tree[ ( * db[ 'i_logic' ], db[ 'level' ] ) ] = b;
    #
    d.lvl_max = max( d.lvl_max );
    for i, xf_d in enumerate( d.xf_idx ):
        x_tmp = OrderedDict( sorted( xf_d.items(  ) ) );
        count =    0;
        xf_i  = [  ];
        il_i  = [  ];
        for x, l  in x_tmp.items(  ):
            xf_i. append(     x );
            il_i. append( count );
            count += 2**( d.lvl_max - l );
        #
        d.xf_idx[ i ] = [ array( xf_i ), \
                          array( il_i  , dtype = 'int' ) ];
    #
    def find_blk( self, x ):
        i_l   = zeros( 4, dtype = 'int' );
        i_l[ 3 ] = self.lvl_max;
        outside = False
        for i, x_s in enumerate( x ):
            if x_s > self.x_lim[ i ][ 1 ] or \
               x_s < self.x_lim[ i ][ 0 ] :
                outside = True;
                break;
            #
            i_l[ i ]  =  self.xf_idx[ i ][ 1 ]\
               [ bisect( self.xf_idx[ i ][ 0 ], x_s ) - 1 ];
        #
        while not tuple( i_l ) in self.tree \
              and i_l[ 3 ] >= 0:
            i_l[ : 3 ] //= 2;
            i_l[   3 ]  -= 1;
        if  i_l[   3 ]   < 0:
            raise KeyError ( i_l );
        return i_l, outside;
    #        
    d.find_blk = types.MethodType( find_blk, d );
    # lambda x : __find_blk__( d, x );
    return;
#

def enroll_interp ( d, fields, method = 'linear',
                    generic = True ):
    if not hasattr( d, 'tree'   ):
        enroll_mesh_tree    ( d );
    if not hasattr( d, 'interp' ):
        enroll_multi_interp ( d );
    if isinstance ( fields, str ):
        fields  = [ fields ];            
    for b, db in d.data.items(  ):
        for f in fields:
            bt = db[ f ].transpose( 2, 1, 0 );
            db [ 'interp_'  + f ] = RegularGridInterpolator\
               ( tuple( db[ 'x_c' ] ), bt, method = method,\
                bounds_error = generic, fill_value = None );
            db [ 'interpn_' + f ] = RegularGridInterpolator\
               ( tuple( db[ 'x_c' ] ), bt,
                 method = 'nearest',\
                 bounds_error =  False, fill_value = None );
        #
    #
    return;
#

def enroll_multi_interp ( d ):
    def __interp_multi__( d, x, fields ):
        if isinstance( x, list ) or len( x.shape ) == 1:
            x = array( [ x ] );
        if isinstance ( fields, str ):
            fields  = [ fields ];            
        #
        res = zeros( ( x.shape[ 0 ], len( fields ) ) );
        for i, x_s in enumerate( x ):
            i_b, outside = d.find_blk( x_s );
            if outside:
                for j, field in enumerate( fields ):
                    res[ i ][ j ] = 0;
                #
                continue;
            #
            b = d.tree[ tuple( i_b ) ]; 
            for j, field in enumerate( fields ):
                try:
                    res[ i ][ j ] = d.data[ b ]\
                       [ 'interp_'  + field ]( x_s ) ;
                except:
                    res[ i ][ j ] = d.data[ b ]\
                       [ 'interpn_' + field ]( x_s ) ;
                #
            #
        #
        if len( x )     == 1 :
            res = res   [ 0 ];
        if len( fields ) == 1:
            res = res.T [ 0 ];
        return res;
    #
    
    d.interp = lambda x, f : __interp_multi__( d, x, f );
    return;
#

def get_dd( file_name,           read_now  = True ,\
            enroll_prim = False, elaborate = True ,\
            data_type   = hydro_data ):
    res = data_type( file_name, read_now, read_now );
    if  enroll_prim:
        enroll_prim_quantities( res );
    if elaborate:
        print( "file: %s; time = %g" \
           % ( file_name, res.globals[ 'time' ] ) );
    #
    return res;
#

def get_last_dd( pattern,  idx =  -1, read_now  = True ,\
                 enroll_prim = False, elaborate = True ,\
                 data_type   = hydro_data ):
    files = glob( pattern );
    if files is None:
        raise FileNotFoundError( pattern );
    #
    files.sort(  );
    return get_dd( files[ idx ], read_now, \
                   enroll_prim,  elaborate, data_type );
#

def stitch_fields( d, fields ):
    db0   = d.data[ 'block_0' ];
    dx0   = array( [ x[ -1 ] - x[ 0 ] for \
                     x in db0[ 'x_f' ] ] );
    lb    = array( db0[ 'rho' ].shape );
    ltot  = d.args( "mesh", "n_cell_global" )[ : : -1 ];
    lblk  = d.args( "mesh", "n_cell_block"  );
    if  lblk is not None:
        l = lb * ltot // lblk[ : : -1 ];
    else:
        l = lb;
    #
    x_min = d.x_lim.T[ 0 ];
    # res   = { f : zeros( l ) for f in fields };
    res   = {  };
    for f in fields:
        shape = d.data[ 'block_0' ][ f ].shape;
        if  len( shape ) <= 3:
            shape = l;
        else:
            shape = ( shape[ 0 ], * l );
        res [ f ] = zeros( shape );
    #
    x_f   = [ [  ], [  ], [  ] ];
    for b, db in d.data.items(  ):
        if 'particle' in b:
            continue;
        x_min_s = [ x[ 0 ] for x in db[ 'x_f' ] ];
        for a in range( 3 ):
            x_f[ a ].append( db[ 'x_f' ][ a ] );
        dl = [ int( ( x_min_s[ a ] - x_min[ a ] \
                      + dx0[ a ] * 0.1 ) / dx0[ a ] ) \
               for a in range( 3 ) ][ : : -1 ];
        for f in fields:
            shape = db[ f ].shape;
            if len( shape ) > 3:
                for a in range( 3 ):
                    res[ f ]\
          [ a,
            dl[ 0 ] * lb[ 0 ] : lb[ 0 ] * ( dl[ 0 ] + 1 ),
            dl[ 1 ] * lb[ 1 ] : lb[ 1 ] * ( dl[ 1 ] + 1 ),
            dl[ 2 ] * lb[ 2 ] : lb[ 2 ] * ( dl[ 2 ] + 1 ) ]\
                    += db[ f ][ a ];
            else:
                res[ f ]\
        [ dl[ 0 ] * lb[ 0 ] : lb[ 0 ] * ( dl[ 0 ] + 1 ),
          dl[ 1 ] * lb[ 1 ] : lb[ 1 ] * ( dl[ 1 ] + 1 ),
          dl[ 2 ] * lb[ 2 ] : lb[ 2 ] * ( dl[ 2 ] + 1 ) ]\
          += db[ f ];
            # except:
            #     print( f, res[ f ].shape, db[ f ].shape );
        #
    #
    x_f = [ unique( concatenate( x_f [ a ] ). round
                  ( decimals = 5 ) ) for a in range( 3 ) ];
    return res, x_f, l;
#
