
#ifndef rtf_framework_h
#define rtf_framework_h

#include <vector>
#include <stdexcept>
#include <exception>
#include <string>
#include <memory>
#include <stdio.h>

void rtf_usleep(unsigned int usec);

/// Normally, you will use TEST_FIXTURE like this:
///
/// TEST_FIXTURE(MyTesck_tFixture);
///     TEST(MyTestFixture::TestFoo);
///     TEST(MyTestFixture::TestBar);
/// TEST_FIXTURE_END();
///
/// But if your fixture has its own member variables that you really need to initialize in its constructor
/// you can do so like this (note the slightly different starting macro, and the presence of TEST_FIXTURE_BEGIN()).
///
/// TEST_FIXTURE_INIT(MyTestFixture)
///     _lok(),
///     _cond(_lok)
/// TEST_FIXTURE_BEGIN()
///     TEST(MyTestFixture::TestFoo);
///     TEST(MyTestFixture::TestBar);
/// TEST_FIXTURE_END()

#define RTF_FIXTURE(a) \
        a() : test_fixture(#a) {

#define RTF_FIXTURE_INIT(a) a() : test_fixture(#a),
#define RTF_FIXTURE_BEGIN() {

#define TEST(a) \
        add_test((void (test_fixture::*)())&a,#a)

#define RTF_FIXTURE_END() }

std::string rtf_format(const char* fmt, ...);
std::string rtf_format(const char* fmt, va_list& args);

class test_fixture;

struct test_host
{
    test_host() :
        fixture(),
        test(nullptr),
        test_name(),
        exception_msg(),
        passed(false)
    {
    }

    test_fixture* fixture;
    void (test_fixture::*test)();
    std::string test_name;
    std::string exception_msg;
    bool passed;
};

#define RTF_ASSERT(a) do{ if( !(a) ) { throw std::runtime_error(rtf_format("%s at Line:%d File:%s", #a, __LINE__, __FILE__)); } } while(false)

#define RTF_ASSERT_EQUAL(a,b) do{ if( !(a==b) ) { throw std::runtime_error(rtf_format("%s != %s at Line:%d File:%s", #a, #b, __LINE__, __FILE__)); } } while(false)

#define RTF_ASSERT_THROWS(thing_that_throws,what_is_thrown) do { try { bool threw = false; try { thing_that_throws; } catch( what_is_thrown& ) { threw=true; } if(!threw) throw false; } catch(...) { throw std::runtime_error(rtf_format("Expected exception not thrown at Line:%d File:%s", __LINE__, __FILE__)); } } while(false)

#define RTF_ASSERT_NO_THROW(thing_that_doesnt_throw) do { bool threw = false; try { thing_that_doesnt_throw; } catch( ... ) { threw=true; } if(threw) { throw std::runtime_error(rtf_format("Unexpected exception at Line:%d File:%s", __LINE__, __FILE__)); } } while(false)

class test_fixture
{
public:
    test_fixture(std::string fixture_name)
        : _tests(),
          _something_failed( false ),
          _fixture_name( fixture_name )
    {
    }

    virtual ~test_fixture() throw()
    {
    }

    void run_tests()
    {
        std::vector<struct test_host>::iterator i = _tests.begin();
        for( ; i != _tests.end(); i++ )
        {
            setup();

            try
            {
                (i->fixture->*(*i).test)();
                i->passed = true;
            }
            catch(const std::exception& ex)
            {
                _something_failed = true;
                i->passed = false;
                i->exception_msg = ex.what();
            }
            catch(...)
            {
                _something_failed = true;
                (*i).passed = false;
            }

            printf("[%s] %-50s\n",(_something_failed)?"F":"P",(*i).test_name.c_str());

            teardown();
        }
    }

    bool something_failed()
    {
        return _something_failed;
    }

    void print_failures()
    {
        std::vector<struct test_host>::iterator i = _tests.begin();
        for( ; i != _tests.end(); i++ )
        {
            if(!(*i).passed)
            {
                printf("\nRTF_FAIL: %s failed with exception: %s\n",
                       (*i).test_name.c_str(),
                       (*i).exception_msg.c_str());
            }
        }
    }

    std::string get_name() const { return _fixture_name; }

protected:
    virtual void setup() {}
    virtual void teardown() {}
    void add_test( void (test_fixture::*test)(), std::string name )
    {
        struct test_host tc;
        tc.test = test;
        tc.test_name = name;
        tc.fixture = this;
        _tests.push_back( tc );
    }

    std::vector<struct test_host> _tests;
    bool _something_failed;
    std::string _fixture_name;
};

extern std::vector<std::shared_ptr<test_fixture>> _test_fixtures;

#define REGISTER_TEST_FIXTURE(a) \
class a##_static_init \
{ \
public: \
    a##_static_init() { \
        _test_fixtures.push_back(std::make_shared<a>());   \
    } \
}; \
a##_static_init a##_static_init_instance;

// This is a globally (across test) incrementing counter so that tests can avoid having hardcoded port
// numbers but can avoid stepping on eachothers ports.
int& rtf_get_next_port();

int rtf_next_port();

#define RTF_NEXT_PORT() rtf_next_port()

bool rtf_ends_with(const std::string& a, const std::string& b);
std::vector<std::string> rtf_regular_files_in_dir(const std::string& dir);

#endif
