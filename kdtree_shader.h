//
// Created by janos on 11.06.20.
//


#pragma once

#include <Magnum/GL/AbstractShaderProgram.h>
#include <Magnum/GL/Texture.h>
#include <Magnum/Math/Vector2.h>
#include <Magnum/Math/Vector3.h>
#include <Magnum/Math/Matrix4.h>
#include <Magnum/Math/Matrix3.h>
#include <Magnum/Math/Color.h>
#include <Magnum/Magnum.h>

namespace Mg = Magnum;
namespace Cr = Corrade;

struct KdTreeShader : public Mg::GL::AbstractShaderProgram {

    explicit KdTreeShader(Mg::UnsignedInt depth);
    explicit KdTreeShader(Mg::NoCreateT) : Mg::GL::AbstractShaderProgram{Mg::NoCreate} {}

    KdTreeShader& bindNodes(Magnum::GL::Texture1D& texture){
        texture.bind(nodesUnit);
        return *this;
    }

    KdTreeShader& bindPoints(Magnum::GL::Texture1D& texture){
        texture.bind(pointsUnit);
        return *this;
    }

    KdTreeShader& bindQueries(Magnum::GL::Texture1D& texture){
        texture.bind(queriesUnit);
        return *this;
    }

    KdTreeShader& bindDistances(Magnum::GL::Texture1D& texture){
        texture.bind(distancesUnit);
        return *this;
    }

    KdTreeShader& bindIndices(Magnum::GL::Texture1D& texture){
        texture.bind(indicesUnit);
        return *this;
    }

    Mg::Int nodesUnit = 0;
    Mg::Int pointsUnit = 1;
    Mg::Int queriesUnit = 2;
    Mg::Int distancesUnit = 3;
    Mg::Int indicesUnit = 4;
};
