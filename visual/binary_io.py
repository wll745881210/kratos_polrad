from numpy import array, frombuffer, ndarray

############################################################
# Basic binary output reader

class binary_io:
    ########################################################
    # Initialization and finalization
    def __init__( self,    file_name, cache_used = True ):
        self. file_name =  file_name;
        self.      dmap =  dict(   );
        self.      hmap =  dict(   );
        self.cache_used = cache_used;
        self.    endian =   'little';
        self.    offset =          0;
        self.    stream =       None;
    #
    def set_stream( self, stream ):
        self.stream     = stream  ;
    #
    def get_size_t( self, bin_data = None ):
        if  bin_data is None:
            bin_data =  self.stream.read( self.s_size_t );
        return int.from_bytes ( bin_data, self.  endian );
    #
    def get_char_t( self, bin_data = None ):
        if  bin_data is None:
            bin_data =  self.stream.read            ( 1 );
        return int.from_bytes ( bin_data, self.  endian );
    #
    def open( self ):
        if self.stream is not None:
            return;
        #
        self.stream   = open( self.file_name, 'rb' );
        self.stream   . seek( 0, 0 );
        # First byte stores: ( is_le | sizeof( size_t ) )
        # Note: one byte does not care about endian
        sst           = int.from_bytes\
                      ( self.stream.read( 1 ), 'little' );
        self.endian   = "little" if sst & 1  else "big";
        self.s_size_t = sst & ( ~ 1 );
        self.s_hdr    = self.get_size_t(  );
        self.s_hmap   = self.get_size_t(  );
        for i_hmap in range( self.s_hmap  ):
            s_kstr = self.get_size_t(  );
            key    = self.stream.read( s_kstr );
            size   = self.get_size_t(  );
            u_size = self.get_char_t(  );
            offset = self.get_size_t(  ) + self.s_hdr;
            self.hmap[ key.decode( 'ascii' ) ]\
                   = ( size, u_size,  offset );
        #
        return;
    #
    def close( self ):
        if  self.stream is not None:
            self.stream.close(  );
            self.stream =    None;
        return;
    #
    def __enter__( self ):
        self.open(  );
        return self;
    #
    
    def __exit__ ( self, etype, evalue, traceback ):
        self.close(  );
        if etype is not None:
            print( etype, evalue, traceback );
        #
    #
    ########################################################
    # Data access
    def __getitem__  ( self,    key ):
        if  key in self.dmap:
            return self.dmap[ key ];
        #
        size, u_size, offset = self.hmap[ key ];
        self.stream.seek( offset, 0 );
        bin_data     = self.stream.read( size );
        if  self.cache_used:
            self.dmap[ key ] = bin_data;
        #
        return bin_data;
    #
    def read( self ):
        for key in self.hmap:
            size, u_size, offset = self.hmap   [  key ];
            self.stream.seek ( offset, 0 );
            self.dmap[ key ] = self.stream.read( size );
        #
    #
    def as_array( self, key, dtype  = 'f' ):
        u_size  = self.hmap[ key ][ 1 ];
        return  frombuffer( self[ key ], dtype = \
                            '<%s%d' %  ( dtype, u_size ) );
    #    
    ########################################################
    # Write data to file
    def cache( self, key, data, dtype = None,
               write_now = False  ):
        if not isinstance( data, ndarray ):
            dat = array( data, dtype = dtype ).ravel(  );
        elif dtype is not None and data.dtype != dtype:
            dat = data.ravel(  ).astype( dtype )
        else:
            dat = data.ravel(  );
        #
        usize = len( dat[ 0 ].tobytes(  ) );
        size  = dat.size * usize;
        self.hmap[ key ] = [ size, usize, self.offset ];
        if write_now:
            self.stream.write( dat.tobytes(  ) );
        else:
            self.dmap[ key ] = dat.tobytes(  );
        self.offset  += size;
        return;
    #
    def hold( self, key ):
        self. hmap[ key ] = [ 0, 0, 0 ];
        return;
    #
    def merge( self, src, skip_keys = [  ] ):
        if  isinstance( src, str ):
            src = binary_io( src );
        if  src.stream is None:
            src.open(  );
            src.read(  );
        #
        offset = self.offset;
        for key, d in src .hmap.items(  ):
            if key in self.hmap or key in skip_keys:
                print( "Skipping key " + key );
                continue;
            #
            self.hmap[ key ] = ( d[ 0 ] , d[ 1 ], offset )
            self.dmap[ key ] = src.dmap[ key ];
            offset += d[ 0 ] ;
        #
        self.offset = offset;
        return;
    #
    def write_header( self ):
        if self.stream is None:
            self.stream = open( self.file_name, 'wb' );
        self.stream.seek( 0, 0 );

        self.endian   = 'little';
        self.s_size_t = 8;
        def i2b( i, length = self.s_size_t ):
            return int.to_bytes( i, byteorder = self.endian,
                                 length = length );
        #        
        hdr = i2b( self.s_size_t + \
                   ( self.endian == 'little' ), 1 );
        # "+ 1" for the little endian
        skip_map = i2b( len( self.hmap ) );
        for key in self.hmap:
            size, usize, offset = self.hmap[ key ];
            k_bin     = key.encode( 'ascii' );
            skip_map += i2b( len( k_bin ) ) + k_bin \
            + i2b( size ) + i2b( usize, 1 ) + i2b( offset );
        #
        hdr += i2b( len( skip_map ) + 1 + self.s_size_t );
        hdr += skip_map;
        self.stream.write( hdr );        
        return;
    #
    def save( self ):
        self.write_header(  );
        for key in self.hmap:
            self.stream.write( self.dmap[ key ] );
        return self.stream.close(  );
    #
#
