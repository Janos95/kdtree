#include "kdtree_shader.h"

#include <Corrade/Containers/Reference.h>
#include <Corrade/Utility/Resource.h>
#include <Corrade/Utility/FormatStl.h>

#include <Magnum/GL/Context.h>
#include <Magnum/GL/Shader.h>
#include <Magnum/GL/Version.h>

using namespace Magnum;

KdTreeShader::KdTreeShader(UnsignedInt depth){
    MAGNUM_ASSERT_GL_VERSION_SUPPORTED(GL::Version::GL450);

    const Utility::Resource rs{"kdtree-data"};

    GL::Shader comp{GL::Version::GL450, GL::Shader::Type::Compute};
    comp.addSource(Utility::formatString("#define STACK_SIZE {}\n", depth))
        .addSource(rs.get("kdtree.vert"));

    CORRADE_INTERNAL_ASSERT_OUTPUT(Mg::GL::Shader::compile({comp}));

    attachShaders({comp});

    CORRADE_INTERNAL_ASSERT_OUTPUT(link());

}
