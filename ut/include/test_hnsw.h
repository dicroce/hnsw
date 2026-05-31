
#include "framework.h"

class test_hnsw : public test_fixture
{
public:
    RTF_FIXTURE(test_hnsw);
      TEST(test_hnsw::test_basic);
      TEST(test_hnsw::test_recall);
      TEST(test_hnsw::test_cosine);
      TEST(test_hnsw::test_edge_cases);
      TEST(test_hnsw::test_external_ids);
      TEST(test_hnsw::test_save_load);
    RTF_FIXTURE_END();

    virtual ~test_hnsw() throw() {}

    virtual void setup();
    virtual void teardown();

    void test_basic();
    void test_recall();
    void test_cosine();
    void test_edge_cases();
    void test_external_ids();
    void test_save_load();
};
