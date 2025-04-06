
#include "framework.h"

class test_hnsw : public test_fixture
{
public:
    RTF_FIXTURE(test_hnsw);
      TEST(test_hnsw::test_basic);
    RTF_FIXTURE_END();

    virtual ~test_hnsw() throw() {}

    virtual void setup();
    virtual void teardown();

    void test_basic();
};
