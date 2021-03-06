#include "kdtree.h"
#include <scoped_timer/scoped_timer.hpp>

#include <Corrade/Containers/GrowableArray.h>

#include <nanoflann.hpp>
#include <random>

using namespace Corrade;
using namespace Magnum;

constexpr int n = 1'000'000;
constexpr int m = 10000;

struct PointCloud
{
    using coord_t = float;
    using Point = Vector3;

    Containers::Array<Point> pts;
};


template <typename Derived>
struct Adapter
{
    using coord_t = typename Derived::coord_t;

    const Derived &obj;

    Adapter(const Derived &obj_) : obj(obj_) { }

    inline const Derived& derived() const { return obj; }

    // Must return the number of data points
    inline size_t kdtree_get_point_count() const { return derived().pts.size(); }

    // Returns the dim'th component of the idx'th point in the class:
    // Since this is inlined and the "dim" argument is typically an immediate value, the
    //  "if/else's" are actually solved at compile time.
    inline coord_t kdtree_get_pt(const size_t idx, const size_t dim) const
    {
        if (dim == 0) return derived().pts[idx].x();
        else if (dim == 1) return derived().pts[idx].y();
        else return derived().pts[idx].z();
    }

    // Optional bounding-box computation: return false to default to a standard bbox computation loop.
    //   Return true if the BBOX was already computed by the class and returned in "bb" so it can be avoided to redo it again.
    //   Look at bb.size() to find out the expected dimensionality (e.g. 2 or 3 for point clouds)
    template <class BBOX>
    bool kdtree_get_bbox(BBOX& /*bb*/) const { return false; }

};


// construct a kd-tree index:
using adapter_t = Adapter<PointCloud>;
using kd_tree_t = nanoflann::KDTreeSingleIndexAdaptor<
                        nanoflann::L2_Simple_Adaptor<float, adapter_t>,
                        adapter_t,
                        3 /* dim */
                    >;

template<class T>
CORRADE_ALWAYS_INLINE void doNotOptimize(T const& value) {
    asm volatile("" : : "r,m"(value) : "memory");
}

int main() {

    Containers::Array<Vector3> queries(Containers::NoInit, m);
    PointCloud cloud;
    Containers::arrayResize(cloud.pts, Containers::NoInit, n);

    std::default_random_engine engine(0);
    std::uniform_real_distribution<float> dist(-1,1);

    for (auto& p : cloud.pts){
        p = Vector3(dist(engine), dist(engine), dist(engine));
    }

    for (auto& q : queries){
        q = Vector3(dist(engine), dist(engine), dist(engine));
    }


    {
        for (uint32_t i = 0; i < 10; ++i) {
            ScopedTimer t{"KDTree Construction", true};
            KDTree tree(cloud.pts);
            doNotOptimize(tree);
        }

    }

    {
        for (uint32_t i = 0; i < 10; ++i) {
            ScopedTimer t{"nano flann Construction", true};
            adapter_t adapter{cloud};
            kd_tree_t index(3 /*dim*/, adapter, nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */) );
            index.buildIndex();
            doNotOptimize(index);
        }
    }

    adapter_t adapter{cloud};
    kd_tree_t index(3 /*dim*/, adapter, nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */) );
    index.buildIndex();

    KDTree tree(cloud.pts);

    for (auto const& q : queries) {
        ScopedTimer t{"KDTree Query"};
        auto result = tree.nearestNeighbor(q);

        doNotOptimize(result);
    }

    for(auto const& q : queries){
        ScopedTimer t{"nano flann Query"};
        const size_t num_results = 1;
        size_t ret_index;
        float out_dist_sqr;
        nanoflann::KNNResultSet<float> resultSet(num_results);
        resultSet.init(&ret_index, &out_dist_sqr);
        index.findNeighbors(resultSet, q.data(), nanoflann::SearchParams{});
        doNotOptimize(resultSet);
    }

    ScopedTimer::printStatistics();
    return 0;
}
