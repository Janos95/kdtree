//
// Created by janos on 16.06.20.
//

#ifndef KDTREE_KDTREE_H
#define KDTREE_KDTREE_H

#include "kdtree_shader.h"

#include <Magnum/Magnum.h>
#include <Magnum/Math/Vector3.h>
#include <Magnum/Math/Distance.h>
#include <Magnum/Math/Range.h>
#include <Magnum/Math/Functions.h>
#include <Magnum/Math/FunctionsBatch.h>
#include <Magnum/Math/Algorithms/Svd.h>
#include <Corrade/Containers/GrowableArray.h>
#include <Corrade/Containers/StridedArrayView.h>
#include <Corrade/Utility/Algorithms.h>

#include <Magnum/GL/Texture.h>
#include <Magnum/GL/PixelFormat.h>
#include <Magnum/GL/TextureFormat.h>
#include <Magnum/GL/Renderer.h>
#include <Magnum/PixelFormat.h>
#include <Magnum/ImageView.h>

#include <scoped_timer/scoped_timer.hpp>

#include <algorithm>
#include <type_traits>
#include <string>

namespace Mg = Magnum;
namespace Cr = Corrade;

//template<std::size_t size, class T>
//auto orientedBB(Containers::StridedArrayView1D<const Math::Vector<size, T>> const& points){
//    Math::Vector<size, T> mean{Math::ZeroInit};
//    for(auto const& point : points)
//        mean += point;
//    mean /= T{points.size()};
//    Math::Matrix<size, T> covariance{Math::ZeroInit};
//    for (auto const& point : points) {
//        covariance += point * point.transposed();
//    }
//    /* not strictly necessary, but anyway this should help numerical precision in the svd */
//    covariance /= T{points.size()};
//    Math::Vector<size, T> eigenValuesSorted;
//    Math::Matrix<size, T> eigenVectorsSorted;
//    Eigen::Map<Eigen::Matrix<float, size, size>> mappedCovariance(covariance.data());
//    Eigen::SelfAdjointEigenSolver<Eigen::Matrix4f> es(mappedCovariance);
//    Eigen::Matrix<T, size, 1> eigenValues = es.eigenvalues();
//    Eigen::Matrix<T, size, size> eigenVectors = es.eigenvalues();
//    /* sort eigenvectors in descending order */
//    Containers::StaticArray<size, std::size_t> indices;
//    for (std::size_t i = 0; i < size; ++i)
//        indices[i] = i;
//    std::sort(indices.begin(), indices.end(),
//            [&](std::size_t a, std::size_t b){ return eigenValues[a] > eigenValues[b]; });
//    for (std::size_t i = 0; i < size; ++i){
//        eigenVectorsSorted[i] = eigenVectors[indices[i]];
//        eigenValuesSorted[i] = eigenValues[indices[i]];
//    }
//    return std::make_tuple(eigenVectorsSorted, mean);
//}

template<class Vector, bool dynamic = false>
class KDTree {
public:

    struct Node {
        constexpr static int32_t Invalid = -1;
        int32_t leftChild;
        int32_t rightChild;
        int32_t pointIndex;
    };

    enum {
        Size = Vector::Size
    };
    using Type = typename Vector::Type;
    using Range = Mg::Math::Range<Size, Type>;
    using NodeHandle = int32_t;
    using Container = std::conditional_t<dynamic, Cr::Containers::Array<Vector>, Cr::Containers::StridedArrayView1D<const Vector>>;

    KDTree() = default;

    explicit KDTree(Cr::Containers::StridedArrayView1D<const Vector> const& points) :
            m_nodes(Cr::Containers::NoInit, points.size()),
            m_bb(Mg::Math::minmax(points))
    {
        if constexpr(dynamic){
           Cr::Containers::arrayResize(m_points, Cr::Containers::NoInit, points.size());
           Cr::Utility::copy(points, m_points);
        } else {
            m_points = points;
        }
        Cr::Containers::Array<Mg::UnsignedInt> nodeIds(Cr::Containers::NoInit, points.size());
        for (Mg::UnsignedInt i = 0; i < nodeIds.size(); ++i)
            nodeIds[i] = i;

        uint32_t size = 0;
        ScopedTimer t("Kd tree recursion");
        m_root = constructTreeMedian(nodeIds.begin(), nodeIds.end(), 0, size);
    }

    auto& point(NodeHandle handle) const { return m_points[m_nodes[handle].pointIndex]; }
    auto leftChild(NodeHandle handle) const { return m_nodes[handle].leftChild; }
    auto rightChild(NodeHandle handle) const { return m_nodes[handle].rightChild; }
    auto root() const { return m_root; }
    auto bb() const { return m_bb; }

    struct NNResult{
        int pointIndex;
        Type distanceSquared = Mg::Math::Constants<Type>::inf();
    };

    /**
     * @param queryPoint the point for which to perform
     * the nearest neighbor query.
     * @return struct containing index corresponding to nearst neighbor and
     * squared distance
     */
    NNResult nearestNeighbor(Vector const& queryPoint){
        NNResult result;
        recurse(queryPoint, m_root, 0, m_bb, result);
        return result;
    }

    /* if you call this make sure you are attached to a valid gl context */
    void upload(){
        m_nodesTex = Magnum::GL::Texture1D{};
        m_pointsTex = Magnum::GL::Texture1D{};

        m_nodes.setStorage(1, Mg::GL::TextureFormat::RGB32I, m_nodes.size())
               .setSubImage(0, {}, Mg::ImageView1D{Mg::PixelFormat::RGB32I, m_nodes.size(), m_nodes});
        m_points.setStorage(1, Mg::GL::TextureFormat::RGB32F, m_points.size());
        if(dynamic || m_points.isContigous()){
            m_pointsTex.setSubImage(0, {}, Mg::ImageView1D{Mg::PixelFormat::RGB32F, m_points.size(), m_points});
        } else {
            Cr::Containers::Array<char> pcData(Cr::Containers::NoInit, m_points.size() * sizeof(Vector));
            Cr::Utility::copy(m_points, pcData); /* @todo only works with Vector3 */
            m_pointsTex.setSubImage(0, {}, Mg::ImageView1D{Mg::PixelFormat::RGB32F, m_points.size(), m_points});
        }
        m_shader = KdTreeShader{Mg::Math::log2(m_nodes) + 1};

        Mg::Debug{} << m_shader.maxComputeWorkGroupCount();
        Mg::Debug{} << m_shader.maxComputeWorkGroupSize();
        Mg::Debug{} << m_shader.maxComputeWorkGroupInvocations();
    }

    void acceleratedNearestNeighborInto(
            Cr::Containers::ArrayView<const Vector> queryPoints,
            Cr::Containers::ArrayView<Type> distances,
            Cr::Containers::ArrayView<Mg::UnsignedInt> indices)
    {
        m_queriesTex = Magnum::GL::Texture1D{};
        m_indicesTex = Magnum::GL::Texture1D{};
        m_distTex = Magnum::GL::Texture1D{};

        m_shader.bindDistances(m_distTex)
                .bindIndices(m_indicesTex)
                .bindNodes(m_nodesTex)
                .bindPoints(m_pointsTex)
                .bindQueries(m_queriesTex)
                .dispatchCompute();
        Mg::GL::Renderer::setMemoryBarrier(Mg::GL::Renderer::MemoryBarrier::TextureFetch);

        CORRADE_ASSERT(queryPoints.size() == distances.size(), "queryPoints.size() != distances.size()", );
        CORRADE_ASSERT(queryPoints.size() == indices.size(), "queryPoints.size() != indices.size()", );
        Mg::MutableImageView1D distImage{Mg::PixelFormat::R32F, distances.size(), distances};
        Mg::MutableImageView1D indImage{Mg::PixelFormat::R32UI, indices.size(), indices};
        m_distTex.image(0, distImage);
        m_indicesTex.image(0, indImage);
    }

private:

    uint32_t constructTreeMedian(uint32_t* begin, uint32_t* end, uint32_t depth, uint32_t& size){
        if(end <= begin) return Node::Invalid;

        auto cd = depth % Size;
        auto n = begin + (end - begin)/2;

        std::nth_element(begin, n, end,
                [&](uint32_t id1, uint32_t id2){ return m_points[id1][cd] < m_points[id2][cd]; });

        auto handle = size++;
        auto& node = m_nodes[handle];
        node.pointIndex = *n;
        node.leftChild = constructTreeMedian(begin, n, depth + 1, size);
        node.rightChild = constructTreeMedian(n + 1, end, depth + 1, size);
        return handle;
    }

    void recurse(Vector const& q, int nodeId, int cd, Range const& bb, NNResult& result){
        if(nodeId == Node::Invalid) return;
        auto const& node = m_nodes[nodeId];
        auto const& p = m_points[node.pointIndex];
        auto distSq = (p - q).dot();
        if(distSq < result.distanceSquared){
            result.distanceSquared = distSq;
            result.pointIndex = node.pointIndex;
        }
        auto nextCd = (cd+1) % Size;
        if(q[cd] < p[cd]){ /* q is closer to left child */
            recurse(q, node.leftChild, nextCd, result);
            /* prune by computing distance to splitting plane */
            if(p[cd] - q[cd] < result.distanceSquared)
                recurse(q, node.rightChild, nextCd, result);
        } else { /* q is closer to right child */
            recurse(q, node.rightChild, nextCd, result);
            if(q[cd] - p[cd] < result.distanceSquared)
                recurse(q, node.leftChild, nextCd, result);
        }
    }

    int m_root = -1;
    Range m_bb;
    Cr::Containers::Array<Node> m_nodes;
    Container m_points;

    Mg::GL::Texture1D m_nodesTex{Mg::NoCreate}, m_pointsTex{Mg::NoCreate};
    Mg::GL::Texture1D m_queriesTex{Mg::NoCreate}, m_distTex{Mg::NoCreate}, m_indicesTex{Mg::NoCreate};
    KdTreeShader m_shader{Mg::NoCreate};
};

/* deduction guide */
template<class Vector>
KDTree(Cr::Containers::Array<Vector> const& points) -> KDTree<Vector, false>;

template<class T, bool dynamic>
void formatTree(
        Magnum::Debug& debug,
        std::string const& prefix,
        KDTree<T, dynamic> const& tree,
        typename KDTree<T, dynamic>::NodeHandle handle,
        bool isLeft,
        const char* arrow)
{
    if(handle != KDTree<T, dynamic>::Node::Invalid){
        debug << prefix.c_str() << arrow << tree.point(handle) << "\n";
        // enter the next tree level - left and right branch

        const char* leftChildArrow = "├──";
        if(tree.rightChild(handle) == KDTree<T, dynamic>::Node::Invalid)
            leftChildArrow = "└──";
        formatTree(debug, prefix + (isLeft ? " │   " : "     "), tree, tree.leftChild(handle), true, leftChildArrow);
        formatTree(debug, prefix + (isLeft ? " │   " : "     "), tree, tree.rightChild(handle), false, "└──");
    }
}

template<class T, bool dynamic>
Mg::Debug& operator<<(Mg::Debug& debug, KDTree<T, dynamic> const& tree){
    formatTree(debug, "", tree, tree.root(), false, "└──");
    return debug;
}

#endif //KDTREE_KDTREE_H
