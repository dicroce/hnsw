
#include "framework.h"
#include <time.h>
#include <stdarg.h>
#include <stdlib.h>
#ifdef IS_WINDOWS
#include <Windows.h>
#include <tchar.h> 
#include <stdio.h>
#include <strsafe.h>
#endif
#ifdef IS_LINUX
#include <sys/time.h>
#include <unistd.h>
#include <dirent.h>
#endif

using namespace std;

vector<shared_ptr<test_fixture>> _test_fixtures;

#ifdef IS_WINDOWS
int64_t GetSystemTimeAsUnixTime()
{
   //Get the number of seconds since January 1, 1970 12:00am UTC
   //Code released into public domain; no attribution required.

   const int64_t UNIX_TIME_START = 0x019DB1DED53E8000; //January 1, 1970 (start of Unix epoch) in "ticks"
   const int64_t TICKS_PER_SECOND = 10000000; //a tick is 100ns

   FILETIME ft;
   GetSystemTimeAsFileTime(&ft); //returns ticks in UTC

   //Copy the low and high parts of FILETIME into a LARGE_INTEGER
   //This is so we can access the full 64-bits as an int64_t without causing an alignment fault
   LARGE_INTEGER li;
   li.LowPart  = ft.dwLowDateTime;
   li.HighPart = ft.dwHighDateTime;
 
   //Convert ticks since 1/1/1970 into seconds
   return (li.QuadPart - UNIX_TIME_START) / TICKS_PER_SECOND;
}
#endif

void rtf_usleep(unsigned int usec)
{
#ifdef IS_LINUX
    usleep(usec);
#endif
#ifdef IS_WINDOWS
    Sleep(usec / 1000);
#endif
}

string rtf_format(const char* fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    const string result = rtf_format(fmt, args);
    va_end(args);
    return result;
}

string rtf_format(const char* fmt, va_list& args)
{
    va_list newargs;
    va_copy(newargs, args);
    const int chars_written = vsnprintf(nullptr, 0, fmt, newargs);
    const int len = chars_written + 1;

    vector<char> str(len);

    va_end(newargs);

    va_copy(newargs, args);

    vsnprintf(&str[0], len, fmt, newargs);

    va_end(newargs);

    string formatted(&str[0]);

    return formatted;
}

// This is a globally (across test) incrementing counter so that tests can avoid having hardcoded port
// numbers but can avoid stepping on eachothers ports.
int _next_port = 5000;

int rtf_next_port()
{
    int ret = _next_port;
    _next_port++;
    return ret;
}

void handle_terminate()
{
    printf( "\nuncaught exception terminate handler called!\n" );
    fflush(stdout);

    std::exception_ptr p = std::current_exception();

    if( p )
    {
        try
        {
            std::rethrow_exception( p );
        }
        catch(std::exception& ex)
        {
            printf("caught an exception in custom terminate handler: %s, %s:%d\n", ex.what(), __FILE__, __LINE__);
        }
        catch(...)
        {
            printf("caught an unknown exception in custom terminate handler.\n");
        }
    }
}

bool rtf_ends_with(const string& a, const string& b)
{
    if (b.size() > a.size()) return false;
    return std::equal(a.begin() + a.size() - b.size(), a.end(), b.begin());
}

vector<string> rtf_regular_files_in_dir(const string& dir)
{
    vector<string> names;

#ifdef IS_LINUX
    DIR* d = opendir(dir.c_str());
    if(!d)
        throw std::runtime_error("Unable to open directory");
    
    struct dirent* e = readdir(d);

    if(e)
    {
        do
        {
            string name(e->d_name);
            if(e->d_type == DT_REG && name != "." && name != "..")
                names.push_back(name);
            e = readdir(d);
        } while(e);
    }

    closedir(d);
#endif

#ifdef IS_WINDOWS
   WIN32_FIND_DATA ffd;
   TCHAR szDir[1024];
   HANDLE hFind;
   
   StringCchCopyA(szDir, 1024, dir.c_str());
   StringCchCatA(szDir, 1024, "\\*");

   // Find the first file in the directory.

   hFind = FindFirstFileA(szDir, &ffd);

   if (INVALID_HANDLE_VALUE == hFind) 
       throw std::runtime_error("Unable to open directory");

   do
   {
      if(!(ffd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY))
         names.push_back(string(ffd.cFileName)); 
   }
   while (FindNextFileA(hFind, &ffd) != 0);

   FindClose(hFind);
#endif

    return names;
}

int main( int argc, char* argv[] )
{
    set_terminate( handle_terminate );

    std::string fixture_name = "";

    if( argc > 1 )
        fixture_name = argv[1];

#ifdef IS_WINDOWS
    srand( (unsigned int)GetSystemTimeAsUnixTime() );
#endif
#ifdef IS_LINUX
    srand( time(0) );
#endif

    bool something_failed = false;

    for(auto& tf : _test_fixtures)
    {
        if( !fixture_name.empty() )
            if(tf->get_name() != fixture_name )
                continue;

        tf->run_tests();

        if( tf->something_failed() )
        {
            something_failed = true;
            tf->print_failures();
        }
    }

    if( !something_failed )
        printf("\nSuccess.\n");
    else printf("\nFailure.\n");

    if(something_failed)
        if(system("/bin/bash -c 'read -p \"Press Any Key\"'") < 0)
        {
            printf("system() failure.\n");
        }

    return 0;
}
