#include "binary_io_base.h"
#include <iostream>
#ifdef    __MPI__
#include <mpi.h>
#endif // __MPI__

namespace binary_io
{
////////////////////////////////////////////////////////////
// Types

static_assert( sizeof( char ) == 1, "char != byte" );
static_assert( sizeof( int  ) != 1, "char ==  int" );

////////////////////////////////////////////////////////////
// Con-/destructor

base_t::base_t(  ) : proc_offset( 0 ),  header_size( 0 ),
                     buffer_read( unbuffered )
{
    const static int n( 1 );
    is_le  = ( * ( char * ) & n == 1 );
    return;
}

base_t::~base_t(  )
{
    return;
}

////////////////////////////////////////////////////////////
// File operations

void base_t::open
( const std::string   & name, const std::string & mode,
  const buffer_read_t & buffer_read )
{
    clear( true );
    file_mode  = std::fstream::binary;
    auto found = [ & ] ( const char & c ) ->  bool
    {   return mode.find( c ) != std::string::npos;  };

    if( found( 'l' ) || found( 'r' ) || found( '*' ) )
        file_mode |= std::fstream:: in;
    if( found( 's' ) || found( 'w' ) || found( '*' ) )
        file_mode |= std::fstream::out;
    if( found( 'a' ) || found( '+' ) )
        file_mode |= std::fstream::app;

    this->  file_name =        name;
    this->buffer_read = buffer_read;
    return;
}

void base_t::close(  )
{
    fio.close(  );
    return;
}

////////////////////////////////////////////////////////////
// Save and load

void base_t::mpi_warning(  )
{
#ifdef   __MPI__
    int rank( 0 );
    int flag( 0 ),   mt ( 0 );
    MPI_Initialized( & flag );
    if( flag == 0 )
    {
        MPI_Init_thread( 0, 0, MPI_THREAD_MULTIPLE, & mt );
        if( mt < MPI_THREAD_MULTIPLE )
            throw std::runtime_error( "Not multi-th MPI" );
    }

    MPI_Comm_rank( MPI_COMM_WORLD, & rank );
    MPI_Comm_size( MPI_COMM_WORLD, & flag );
    if( rank == 0 && flag > 1 )
        std::cerr << "Non-MPI version of the binary-IO "
            "module invoked in the parallel mode; rank "
            "numbers will be appended to file name(s).\n";
    if( flag > 1 )
        file_name += "_" + std::to_string( rank );
#endif // __MPI__
    return;                    // Do nothing w/o MPI
}

void base_t::prep_header(  )
{
    proc_offset = 0;
    size_t offset_count( 0 );
    for( auto & p : b_info )
    {
        p.second.offset =  offset_count;
        offset_count   += p.second.size;
    }
    return;
}

void base_t::save_header(  )
{
    char size_size_t = ( char )( is_le | sizeof( size_t ) );
    const auto & bis = serialize::get( b_info );
    header_size  = 1 + sizeof( header_size ) + bis.size(  );

    size_t offset  ( 0 );
    offset += f_write_at(      & size_size_t, 1,  offset  );
    offset += f_write_at
         ( & header_size, sizeof( header_size ),  offset  );
    offset += f_write_at
         ( bis.data(   ), bis.size(           ),  offset  );
    if( header_size != offset )
        throw std::runtime_error  ( "Inconsistent offset" );

    proc_offset = header_size;
    return;
}

void base_t::save_bulk(  )
{
    for( auto & it : buffer )
    {
         auto & b  = it.second;
         auto & bi = b_info.at( it.first ); // Don't create
         if( b.size != it.second.size )
             throw std::runtime_error( "Incosistent size" );
         f_write_at( b.data, b.size,
                     proc_offset + bi.offset );
    }
    return;
}

void base_t::load_header(  )
{
    char size_size_t_c(   0 );
    f_read_at_all( & size_size_t_c,     1 , 0 );
    const bool is_le_input( size_size_t_c & 1 );
    size_size_t_c  &= ( ~ 1 );

    if( ( size_t ) ( size_size_t_c ) != sizeof( size_t ) )
        throw std::runtime_error( "Inconsistent size_t " );
    if( is_le_input ^ is_le )
        throw std::runtime_error( "Inconsistent endian " );

    const size_t size_size_t( ( size_t )  size_size_t_c  );
    f_read_at_all       ( & header_size,  size_size_t, 1 );
    const size_t bis_size = header_size - size_size_t - 1 ;

    char * bis_data = new char[ bis_size ];
    f_read_at_all ( bis_data,   bis_size, size_size_t + 1 );
    serialize::set(   b_info,   bis_data,        bis_size );
    delete [  ] bis_data;

    proc_offset = header_size;
    return;
}

void base_t::load_bulk(  )
{
    if( buffer_read == buffered )
        for( auto & it : b_info )
        {
             auto & bi = it.second;
             auto & b  = buffer[ it.first ]; // May create
             b.resize  ( bi.size );
             f_read_at ( b .data,      bi.  size  ,
                         proc_offset + bi.offset );
        }
    return;
}

void base_t::check_open( const base_t::o_t & desired_mode )
{
    if( ! ( file_mode & desired_mode ) )
        throw std::runtime_error( "Incorrect mode" );
    if( file_mode & ( std::fstream::out |
                      std::fstream::app ) )
        mpi_warning (  );

    fio.open( file_name.c_str(  ), file_mode );
    if( ! fio )
        throw std::runtime_error
            ( "Cannot open file " + file_name );
    return;
}

void base_t::save(  )
{
    check_open ( std::fstream::out | std::fstream::app );
    prep_header(  );
    save_header(  );
    save_bulk  (  );
    return;
}

void base_t::load(  )
{
    check_open ( std::fstream::in  );
    load_header(  );
    load_bulk  (  );
    return;
}

void base_t::clear( const bool & deep )
{
    buffer.clear(  );
    if( deep )
        b_info.clear(  );
    return;
}

////////////////////////////////////////////////////////////
// Information

const b_info_t & base_t::info( const std::string & tag )
{
    return b_info.at( tag );
}

////////////////////////////////////////////////////////////
// Generic R/W interfaces--you can do almost anything

void base_t:: read_callback
( const std::string & tag, const f_callback_t & f )
{
    f( b_info.at( tag ) , buffer_read == buffered ?
       buffer.at( tag ) : buffer[ tag ] );
    return;                     // May also create
}

void base_t::write_callback
( const std::string & tag, const f_callback_t & f )
{
    f( b_info   [ tag ], buffer[ tag ] );
    return;                     // Create if necessary
}

void base_t::f_read_at( void  * data , const size_t & size,
                        const size_t & offset )
{
    fio.seekp( offset, fio.beg );
    fio.read ( ( char * ) data, size );
    return;
}

void base_t::f_read_at_all
( void  * data , const size_t & size,
  const size_t & offset )
{
    return f_read_at( data, size, offset );
}

size_t base_t::f_write_at
( const void   * data, const size_t & size,
  const size_t & offset )
{
    fio.seekp( offset,       std::ios::beg );
    fio.write( ( const char * ) data, size );
    return size;
}

////////////////////////////////////////////////////////////
// Read and write, generic interfaces

bool base_t::has_tag( const std::string & tag )
{
    return b_info.find( tag ) != b_info.end(  );
}

size_t base_t::read( void * p, const std::string & tag )
{
    if( p == nullptr )
        throw std::runtime_error( "nullptr in binary r" );
    if( ! has_tag( tag ) )
        throw std::runtime_error
            ( "Read cannot find tag: " + tag );

    auto & bi    = b_info.at( tag );
    if( bi.size  % bi.u_size != 0 )
        throw std::runtime_error( "Indivisible   read" );
    if( buffer_read == unbuffered )
        f_read_at( p, bi.size, header_size + bi.offset );
    else                        // Don't create
        memcpy   ( p, buffer.at( tag ).data, bi  .size );
    return bi.size / bi.u_size;
}

size_t base_t::write( const void        *      p ,
                      const size_t      &  count ,
                      const size_t      & u_size ,
                      const std::string &    tag ,
                      const bool        &   copy )
{   // Not using "std::map::at": Create if necessary
    if( p == nullptr )
        throw std::runtime_error( "nullptr in binary w" );
    auto & b  = buffer [ tag ];
    auto & bi = b_info [ tag ];
    bi.size   = count * u_size;
    bi.u_size =         u_size;
    if( copy )
    {
        b.resize         ( bi.size );
        b.copy           =    true  ;
        memcpy( b.data, p, bi.size );
    }
    else
    {
        b.data = ( char * )        p;
        b.size =             bi.size;
        b.copy =               false;
    }
    return count;
}

};                              // namespace binary_io
