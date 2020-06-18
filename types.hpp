//
// Created by janos on 02.05.20.
//

#pragma once

#include <Magnum/Magnum.h>
#include <Magnum/SceneGraph/SceneGraph.h>
#include <Corrade/Containers/EnumSet.h>
#include <Corrade/Containers/Containers.h>

namespace Cr = Corrade;
namespace Mg = Magnum;

using Object3D = Mg::SceneGraph::Object<Mg::SceneGraph::MatrixTransformation3D>;
using Scene3D = Mg::SceneGraph::Scene<Mg::SceneGraph::MatrixTransformation3D>;
using Drawable = Mg::SceneGraph::Drawable3D;
using DrawableGroup = Mg::SceneGraph::DrawableGroup3D;

template<class T>
using View1D = Cr::Containers::StridedArrayView1D<T>;
template<class T>
using View2D = Cr::Containers::StridedArrayView2D<T>;
template<class T>
using View3D = Cr::Containers::StridedArrayView3D<T>;



