#ifdef    __MPI__
#include "binary_io_mpi.h"

#include <cstdint>
#include <climits>

////////////////////////////////////////////////////////////
// Types

#if     SIZE_MAX == UCHAR_MAX
#define MPI_SIZE_T  MPI_UNSIGNED_CHAR
#elif   SIZE_MAX == USHRT_MAX
#define MPI_SIZE_T  MPI_UNSIGNED_SHORT
#elif   SIZE_MAX == UINT_MAX
#define MPI_SIZE_T  MPI_UNSIGNED
#elif   SIZE_MAX == ULONG_MAX
#define MPI_SIZE_T  MPI_UNSIGNED_LONG
#elif   SIZE_MAX == ULLONG_MAX
#define MPI_SIZE_T  MPI_UNSIGNED_LONG_LONG
#else
#error "Cannot specify size_t for MPI."
#endif

namespace binary_io
{
////////////////////////////////////////////////////////////
// Constructor and destructor

mpi_t:: mpi_t( const int color, const int key )
{
    int flag( 0 ),   mt ( 0 );
    MPI_Initialized( & flag );
    if( flag == 0 )
    {
        MPI_Init_thread( 0, 0, MPI_THREAD_MULTIPLE, & mt );
        if( mt < MPI_THREAD_MULTIPLE )
            throw std::runtime_error( "Not multi-th MPI" );
    }
    MPI_Comm_dup ( MPI_COMM_WORLD, & comm );
    MPI_Comm_rank( comm,         &   rank );
    MPI_Comm_size( comm,         & n_rank );
    return;
}

mpi_t::~mpi_t(  )
{
    return;
}

void mpi_t::finalize(  )
{
    if( comm != MPI_COMM_WORLD )    
        MPI_Comm_free( &  comm );
    return;
}

////////////////////////////////////////////////////////////
// File operations

static void mpi_api( const int & ierr )
{
    if( ierr == MPI_SUCCESS )
        return;
    char str[ 256 ];
    int  dum(  0  );
    MPI_Error_string( ierr, str, & dum );
    std::cerr << str << '\n';
    throw std::runtime_error( "MPI binary IO" );
    return;
}

void mpi_t::map_io
( const f_io_t & f, void * data ,
  const size_t & size, const size_t & offset )
{
    size_t offset_ext( offset ), byte_remain( size );
    while( byte_remain > 0 )
    {
        const auto count
          = std::min( size_t( INT_MAX ) / 2, byte_remain );
        mpi_api( f  ( data, count, offset_ext ) );
        byte_remain -=               count;
        offset_ext  +=               count;
        data = ( ( char * ) data ) + count;
    }
    return;
}

void mpi_t::open
( const std::string   & name, const std::string & mode,
  const buffer_read_t & buffer_read )
{
    MPI_Barrier( comm );
    clear( true );
    file_mode = 0;
    auto found = [ & ] ( const char & c ) -> bool
    {  return mode.find( c ) != std::string::npos;  };

    if( found( 'l' ) || found( 'r' ) )
        file_mode |= MPI_MODE_RDONLY;
    if( found( 's' ) || found( 'w' ) )
        file_mode |= MPI_MODE_WRONLY | MPI_MODE_CREATE;
    if( found( 'a' ) || found( '+' ) )
        file_mode |= MPI_MODE_APPEND | MPI_MODE_CREATE;
    if( found( '*' ) )
        file_mode |= MPI_MODE_RDWR   | MPI_MODE_CREATE;

    this->  file_name =        name;
    this->buffer_read = buffer_read;
    return;
}

void mpi_t::close(  )
{
    MPI_File_close( & fh );
    MPI_Barrier( comm );    
    return;
}

void mpi_t::f_read_at( void  * data , const size_t & size,
                       const size_t & offset )
{
    return map_io( [ & ] 
    ( void * p, const int & cnt, const size_t & off )
    {
        return MPI_File_read_at( fh, off, p, cnt, MPI_BYTE,
                                 MPI_STATUS_IGNORE );
    },  data, size, offset );    
}

void mpi_t::f_read_at_all
( void         * data , const size_t & size,
  const size_t & offset )
{
    return map_io( [ & ]
    ( void * p, const int & cnt, const size_t & off )
    {
        return MPI_File_read_at_all( fh, off, p, cnt ,
                         MPI_BYTE, MPI_STATUS_IGNORE );
    },  data, size, offset );    
}

size_t mpi_t::f_write_at
( const void   * data, const size_t & size,
  const size_t & offset )
{
    map_io( [ & ]
    ( void * p, const int & cnt, const size_t & off )
    {
        return MPI_File_write_at( fh, off, p, cnt, MPI_BYTE,
                                  MPI_STATUS_IGNORE );
    },  ( void * ) data, size, offset );        
    return size;
}

void mpi_t::save_header(  )
{
    static const int root( 0 );

    //////////////////////////////////////////////////
    // Stat the local buffer information
    std::vector< size_t > size_global;
    std::vector< size_t > cumu_global;
    if( rank  == root )
    {
        size_global.resize( n_rank     );
        cumu_global.resize( n_rank + 1 );
    }

    //////////////////////////////////////////////////
    // Root collects the size of all serialized b_info
    std::string bis = serialize::get( b_info );
    const auto  b_s = bis.size(  );
    MPI_Gather ( &  b_s,               1, MPI_SIZE_T,
                 size_global.data(  ), 1, MPI_SIZE_T,
                 root, comm );
    std::string bi_global_str;
    if( rank == root )
    {
        cumu_global[ 0 ] = 0;
        for( int i = 0;  i <  size_global.size(   ); ++ i )
            cumu_global[ i + 1 ] = size_global[ i ]
                                 + cumu_global[ i ];
        bi_global_str.resize( cumu_global.back(   ) );
    }
    
    //////////////////////////////////////////////////
    // Then collects and converts all serialized b_info
    std::vector< int > ci_global, si_global;
    if( rank == root )
    {
        for( auto & size_s : size_global )
            si_global.push_back ( size_s );
        for( auto & cumu_s : cumu_global )
            ci_global.push_back ( cumu_s );
    }
    MPI_Gatherv( bis.data(  ),  bis.size(  ),    MPI_CHAR,
                 & bi_global_str[ 0 ], si_global.data(  ),
                 & ci_global[ 0 ], MPI_CHAR, root, comm );
    if( rank == root )
    {
        size_global[ 0 ] = b_info.rbegin(  )->second.offset
                         + b_info.rbegin(  )->second. size;
        for( int i = 1;  i < n_rank; ++ i  )
        {
            decltype( b_info ) bi_;
            auto bis_ = bi_global_str.substr
                ( cumu_global[ i ], size_global[ i ] );
            serialize::set
                ( bi_, bis_.c_str(  ), bis_.size(  ) );
            
            size_global[ i ] = bi_.rbegin(  )->second.offset
                             + bi_.rbegin(  )->second. size;
            cumu_global[ i ] = size_global[ i - 1 ]
                             + cumu_global[ i - 1 ];

            for( auto & p : bi_ )
                p.second.offset    += cumu_global[  i ];
            b_info.insert( bi_.begin(  ), bi_.end(  ) );
        }
        base_t::save_header(  );
    }
    MPI_Bcast  ( & header_size, 1, MPI_SIZE_T, root, comm );
    MPI_Scatter( cumu_global.data(  ),      1, MPI_SIZE_T  ,
                 & proc_offset, 1, MPI_SIZE_T, root, comm );
    proc_offset += header_size;
    return;
}

void mpi_t::check_open_mpi( const int & desired_mode  )
{
    if( ! ( file_mode & desired_mode ) )
        throw std::runtime_error(   "Not in write mode" );
    const auto ierr
        = MPI_File_open( comm, file_name.c_str(  ),
                         file_mode, MPI_INFO_NULL, & fh );
    if( ierr != MPI_SUCCESS )
        throw std::runtime_error
            ( "MPI Cannot open file " +  file_name );
    return;
}

void mpi_t::save  (  )
{
    check_open_mpi( MPI_MODE_WRONLY | MPI_MODE_RDWR   |
                    MPI_MODE_APPEND | MPI_MODE_CREATE );
    prep_header   (  );
    save_header   (  );
    save_bulk     (  );
    return;
}

void mpi_t::load  (  )
{
    check_open_mpi( MPI_MODE_RDONLY | MPI_MODE_RDWR );
    load_header   (  );
    load_bulk     (  );
    return; 
}

};                              // namespace binary_io

#endif  // __MPI__
