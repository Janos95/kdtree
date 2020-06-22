//
// Created by janos on 17.06.20.
//

#include "kdtree.h"
#include "arc_ball_camera.hpp"
#include "types.hpp"

#include <scoped_timer/scoped_timer.hpp>

#include <Corrade/Containers/GrowableArray.h>
#include <Corrade/Containers/StaticArray.h>

#include <Magnum/GL/DefaultFramebuffer.h>
#include <Magnum/GL/Renderer.h>
#include <Magnum/Platform/Sdl2Application.h>
#include <Magnum/SceneGraph/Scene.h>
#include <Magnum/SceneGraph/MatrixTransformation3D.h>
#include <Magnum/SceneGraph/Drawable.h>
#include <Magnum/Trade/MeshData.h>
#include <Magnum/Trade/ImageData.h>
#include <Magnum/ImageView.h>
#include <Magnum/GL/Texture.h>
#include <Magnum/GL/TextureFormat.h>
#include <Magnum/DebugTools/ColorMap.h>
#include <Magnum/Shaders/Phong.h>
#include <Magnum/Shaders/Flat.h>
#include <Magnum/Primitives/Plane.h>
#include <Magnum/Primitives/UVSphere.h>
#include <Magnum/Primitives/Cube.h>
#include <Magnum/GL/Mesh.h>
#include <Magnum/PixelFormat.h>
#include <Magnum/MeshTools/Compile.h>
#include <Magnum/MeshTools/Transform.h>
#include <Magnum/Math/Color.h>
#include <Magnum/Math/Quaternion.h>

#include <MagnumPlugins/AssimpImporter/AssimpImporter.h>

#include <random>

using namespace Magnum;
using namespace Corrade;

using namespace Magnum::Math::Literals;

constexpr Vector3 lightPositions[6] = {
        {10,0,0},
        {-10,0,0},
        {0,10,0},
        {0,-10,0},
        {0,0,10},
        {0,0,-10}
};

constexpr Color4 lightColors[6] = {
        Color4{1.f/2.f,1.f/2.f,1.f/2.f,1},
        Color4{1.f/2.f,1.f/2.f,1.f/2.f,1},
        Color4{1.f/2.f,1.f/2.f,1.f/2.f,1},
        Color4{1.f/2.f,1.f/2.f,1.f/2.f,1},
        Color4{1.f/2.f,1.f/2.f,1.f/2.f,1},
        Color4{1.f/2.f,1.f/2.f,1.f/2.f,1}
};

template <class F>
struct YCombinator {
    F&& f;
    template <class... Args>
    decltype(auto) operator()(Args&&... args) const {
        return f(*this, std::forward<Args>(args)...);
    }
};

/* deduction guide */
template<class F>
YCombinator(F&& f) -> YCombinator<std::decay_t<F>>;

struct InstanceData {
    Matrix4 tf;
    Matrix3 normalMatrix;
    Color4 color;
};

struct KdTreeDrawable : Object3D, Drawable {
    explicit KdTreeDrawable(KDTree<Vector3> const& tree, Object3D* parent, SceneGraph::DrawableGroup3D& drawables, int depth);
    void draw(const Matrix4& transformation, SceneGraph::Camera3D& camera) override;

    Shaders::Flat3D flat;
    Shaders::Phong phong;
    GL::Mesh cubeWire;
    GL::Mesh sphere;
    GL::Mesh plane;
    GL::Buffer instanceBuffer;
    Containers::Array<Matrix4> boxInstanceData;
    Containers::Array<InstanceData> planeInstanceData;
    Containers::Array<InstanceData> sphereInstanceData;
    bool drawKdTreeDrawable = true;
};

constexpr Matrix3 flipPlane(uint32_t cd, Range3D const& bb){
    switch (cd) {
        case 0 : return Matrix3{Vector3::zAxis(), Vector3::yAxis(), -Vector3::xAxis()};
        case 1 : return Matrix3{Vector3::xAxis(), -Vector3::zAxis(), Vector3::yAxis()};
        case 2 : return Matrix3{Math::IdentityInit};
        default: return {};
    }
}

constexpr Matrix3 scalePlane(uint32_t cd, Range3D const& bb){
    switch (cd) {
        case 0 : return Matrix3::fromDiagonal({bb.sizeZ(), bb.sizeY(), 1.f}) * 0.5;
        case 1 : return Matrix3::fromDiagonal({bb.sizeX(), bb.sizeZ(), 1.f}) * 0.5;
        case 2 : return Matrix3::fromDiagonal({bb.sizeX(), bb.sizeY(), 1.f}) * 0.5;
        default: return {};
    }
}

KdTreeDrawable::KdTreeDrawable(KDTree<Vector3> const& tree, Object3D* parent, SceneGraph::DrawableGroup3D& drawables, int depth):
        Object(parent),
        Drawable{*this, &drawables},
        flat{Shaders::Flat3D::Flag::InstancedTransformation},
        phong{Shaders::Phong::Flag::InstancedTransformation|Shaders::Phong::Flag::VertexColor, 6},
        cubeWire{MeshTools::compile(Primitives::cubeWireframe())},
        sphere{MeshTools::compile(Primitives::uvSphereSolid(20,20))},
        plane(MeshTools::compile(Primitives::planeSolid()))
{
    Deg hue = 42.0_degf;
    YCombinator generateInstanceData{
        [&](auto&& generateInstanceData, KDTree<Vector3>::NodeHandle node, Range3D const &bb, uint32_t d) -> void {
            if (node == KDTree<Vector3>::Node::Invalid || d == depth) return;

            auto cd = d % 3;
            auto p = tree.point(node);
            auto tfSphere = Matrix4::from(Matrix3{Math::IdentityInit} * 0.03, p);
            auto flip1 = flipPlane(cd, bb);
            auto flip2 = flip1 * Matrix3::fromDiagonal({1,-1,-1});
            auto q = bb.center();
            q[cd] = p[cd];
            auto tfPlane1 = Matrix4::from(flip1*scalePlane(cd, bb), q);
            auto tfPlane2 = Matrix4::from(flip2*scalePlane(cd, bb), q);
            auto color = Color3::fromHsv({hue += 137.5_degf, 0.75f, 0.9f});

            Containers::arrayAppend(boxInstanceData, Matrix4::from(.5f * Matrix3::fromDiagonal({bb.sizeX(), bb.sizeY(), bb.sizeZ()}), bb.center()));
            Containers::arrayAppend(sphereInstanceData, Containers::InPlaceInit, tfSphere, Matrix3{Math::IdentityInit}, color);
            Containers::arrayAppend(planeInstanceData, Containers::InPlaceInit, tfPlane1, flip1, color);
            Containers::arrayAppend(planeInstanceData, Containers::InPlaceInit, tfPlane2, flip2, color);

            generateInstanceData(tree.leftChild(node), trim<Side::Left>(bb, cd, p), d + 1);
            generateInstanceData(tree.rightChild(node), trim<Side::Right>(bb, cd, p), d + 1);
        }
    };

    generateInstanceData(tree.root(), tree.bb(), 0);

    cubeWire.addVertexBufferInstanced(instanceBuffer, 1, 0,
                                       Shaders::Flat3D::TransformationMatrix{});
    sphere.addVertexBufferInstanced(instanceBuffer, 1, 0,
                                 Shaders::Phong::TransformationMatrix{},
                                 Shaders::Phong::NormalMatrix{},
                                 Shaders::Phong::Color4{});
    plane.addVertexBufferInstanced(instanceBuffer, 1, 0,
                                    Shaders::Phong::TransformationMatrix{},
                                    Shaders::Phong::NormalMatrix{},
                                    Shaders::Phong::Color4{});
}

void KdTreeDrawable::draw(const Matrix4& tf, SceneGraph::Camera3D& camera){

    if(!drawKdTreeDrawable) return;

    Containers::StaticArray<6, Vector3> transformedLights;
    for (uint32_t i = 0; i < 6; ++i)
        transformedLights[i] = tf.transformPoint(lightPositions[i]);

    /* draw spheres at splitting points */
    instanceBuffer.setData(sphereInstanceData, GL::BufferUsage::DynamicDraw);
    sphere.setInstanceCount(sphereInstanceData.size());
    phong.setAmbientColor(0x111111_rgbf)
         .setSpecularColor(0x330000_rgbf)
         .setSpecularColor(0x330000_rgbf)
         .setShininess(20.)
         .setLightPositions(transformedLights)
         .setLightColors(lightColors)
         .setProjectionMatrix(camera.projectionMatrix())
         .setTransformationMatrix(tf)
         .setNormalMatrix(tf.normalMatrix())
         .draw(sphere);


    /* draw splitting planes*/
    instanceBuffer.setData(planeInstanceData, GL::BufferUsage::DynamicDraw);
    plane.setInstanceCount(planeInstanceData.size());
    phong.setAmbientColor(0x111111_rgbf)
         .setSpecularColor(0x330000_rgbf)
         .setSpecularColor(0x330000_rgbf)
         .setShininess(20.)
         .setLightPositions(transformedLights)
         .setLightColors(lightColors)
         .setProjectionMatrix(camera.projectionMatrix())
         .setTransformationMatrix(tf)
         .setNormalMatrix(tf.normalMatrix())
         .draw(plane);

    /* draw cube wireframes */
    instanceBuffer.setData(boxInstanceData, GL::BufferUsage::DynamicDraw);
    cubeWire.setInstanceCount(boxInstanceData.size());
    flat.setTransformationProjectionMatrix(camera.projectionMatrix() * tf)
            .setColor(Color4{0})
            .draw(cubeWire);
}

template <class T = Float>
Math::Quaternion<T> randomRotation(){
    //http://planning.cs.uiuc.edu/node198.html
    static std::default_random_engine engine(std::random_device{}());
    static std::uniform_real_distribution<T> dist(0,1);
    auto u = randomScalar(dist(engine));
    auto v = 2 * Math::Constants<T>::pi() * randomScalar(dist(engine));
    auto w = 2 * Math::Constants<T>::pi() * randomScalar(dist(engine));
    return {{sqrt<T>(1 - u) * std::sin(v), sqrt<T>(1 - u) * std::cos(v), sqrt<T>(u) * std::sin(w)}, sqrt<T>(u) * std::cos(w)};
}

struct PhongDrawable : Drawable
{
public:

    explicit PhongDrawable(Object3D& obj, GL::Mesh& m, GL::Texture2D& t, Shaders::Phong& s, DrawableGroup* group) :
            Drawable(obj, group),
            shader(s),
            mesh(m),
            texture(t)
    {
    }

    void draw(const Matrix4& tf, SceneGraph::Camera3D& camera) override {
        Containers::StaticArray<6, Vector3> transformedLights;
        for (uint32_t i = 0; i < 6; ++i)
            transformedLights[i] = tf.transformPoint(lightPositions[i]);

        shader.bindDiffuseTexture(texture)
              .setShininess(200.0f)
              .setLightPositions(transformedLights)
              .setLightColors(lightColors)
              .setTransformationMatrix(tf)
              .setNormalMatrix(tf.normalMatrix())
              .setProjectionMatrix(camera.projectionMatrix())
              .draw(mesh);
    }

    Shaders::Phong& shader;
    GL::Mesh& mesh;
    GL::Texture2D& texture;
    Color4 color{0x2f83cc_rgbf};
};


class KdTreeApplication: public Platform::Application {
public:
    explicit KdTreeApplication(const Arguments& arguments);

private:
    void drawEvent() override;
    void viewportEvent(ViewportEvent& event) override;
    void keyPressEvent(KeyEvent& event) override;
    void mousePressEvent(MouseEvent& event) override;
    void mouseReleaseEvent(MouseEvent& event) override;
    void mouseMoveEvent(MouseMoveEvent& event) override;
    void mouseScrollEvent(MouseScrollEvent& event) override;

    Corrade::Containers::Optional<ArcBallCamera> camera;
    Scene3D scene;
    DrawableGroup drawables;
    Trade::MeshData meshData{MeshPrimitive::Points, 0};
    GL::Mesh mesh{Magnum::NoCreate};
    GL::Texture2D texture{Magnum::NoCreate};
    KDTree<Vector3> tree;
    KdTreeDrawable* treeDrawable;
    PhongDrawable* test;
    Shaders::Phong phong{Magnum::NoCreate};

    uint32_t depth = 4;
};

KdTreeApplication::KdTreeApplication(const Arguments& arguments):
    Platform::Application{arguments, Magnum::NoCreate}
{
/* Setup window */
    {
        const Vector2 dpiScaling = this->dpiScaling({});
        Configuration conf;
        conf.setTitle("Viewer")
                .setSize(conf.size(), dpiScaling)
                .setWindowFlags(Configuration::WindowFlag::Resizable);
        GLConfiguration glConf;
        glConf.setSampleCount(dpiScaling.max() < 2.0f ? 8 : 2);
        if(!tryCreate(conf, glConf)) {
            create(conf, glConf.setSampleCount(0));
        }
    }

    /* Set up the camera */
    {
        /* Setup the arcball after the camera objects */
        const Vector3 eye = Vector3::zAxis(-10.0f);
        const Vector3 center{};
        const Vector3 up = Vector3::yAxis();
        camera.emplace(scene, eye, center, up, 45.0_degf,
                               windowSize(), framebufferSize());
    }

    /* load assets and setup scene */
    {
        PluginManager::Manager<Trade::AbstractImporter> manager;

        auto sceneImporter = manager.loadAndInstantiate("AssimpImporter");
        if (!sceneImporter) std::exit(1);
        if (!sceneImporter->openFile("/home/janos/kdtree/assets/spot_triangulated.obj")) std::exit(2);

        if (sceneImporter->meshCount() && sceneImporter->mesh(0)) {
            meshData = *(sceneImporter->mesh(0));
            mesh = MeshTools::compile(meshData);
            tree = KDTree{meshData.attribute<Vector3>(Trade::MeshAttribute::Position)};
        } else std::exit(3);

        auto imageImporter = manager.loadAndInstantiate("PngImporter");
        if (!imageImporter) std::exit(1);
        if (!imageImporter->openFile("/home/janos/kdtree/assets/spot_texture.png")) std::exit(2);

        auto image = imageImporter->image2D(0);

        texture = GL::Texture2D{};
        texture.setWrapping(GL::SamplerWrapping::ClampToEdge)
                .setMagnificationFilter(GL::SamplerFilter::Linear)
                .setMinificationFilter(GL::SamplerFilter::Linear)
                .setStorage(1, GL::textureFormat(image->format()), image->size())
                .setSubImage(0, {}, *image);

        phong =  Shaders::Phong{Shaders::Phong::Flag::DiffuseTexture, 6};
        treeDrawable = new KdTreeDrawable(tree, &scene, drawables, depth);

        auto object = new Object3D{&scene};
        new PhongDrawable{*object, mesh, texture, phong, &drawables};
        test = new PhongDrawable{*object, mesh, texture, phong, &drawables};
    }

    GL::Renderer::enable(GL::Renderer::Feature::DepthTest);
    GL::Renderer::enable(GL::Renderer::Feature::FaceCulling);

    /* Start the timer, loop at 60 Hz max */
    setSwapInterval(1);
    setMinimalLoopPeriod(16);
}

void KdTreeApplication::drawEvent() {
    GL::defaultFramebuffer.clear(
            GL::FramebufferClear::Color|GL::FramebufferClear::Depth);
    bool camChanged = camera->update();
    camera->draw(drawables);

    swapBuffers();

    if(camChanged) redraw();
}

void KdTreeApplication::viewportEvent(ViewportEvent& event) {
    GL::defaultFramebuffer.setViewport({{}, event.framebufferSize()});
    camera->reshape(event.windowSize(), event.framebufferSize());
}

void KdTreeApplication::keyPressEvent(KeyEvent& event) {
    switch(event.key()) {
        case KeyEvent::Key::L:
            if(camera->lagging() > 0.0f) {
                Debug{} << "Lagging disabled";
                camera->setLagging(0.0f);
            } else {
                Debug{} << "Lagging enabled";
                camera->setLagging(0.85f);
            }
            break;
        case KeyEvent::Key::R:
            camera->reset();
            break;
        case KeyEvent::Key::T:
            treeDrawable->drawKdTreeDrawable = !(treeDrawable->drawKdTreeDrawable);
        case KeyEvent::Key::A:
            if(depth)
                --depth;
            scene.erase(treeDrawable);
            treeDrawable = new KdTreeDrawable(tree, &scene, drawables, depth);
            break;
        case KeyEvent::Key::D:
            ++depth;
            scene.erase(treeDrawable);
            treeDrawable = new KdTreeDrawable(tree, &scene, drawables, depth);
            break;
        default: return;
    }

    event.setAccepted();
    redraw(); /* camera has changed, redraw! */
}

void KdTreeApplication::mousePressEvent(MouseEvent& event) {
    /* Enable mouse capture so the mouse can drag outside of the window */
    /** @todo replace once https://github.com/mosra/magnum/pull/419 is in */
    SDL_CaptureMouse(SDL_TRUE);

    camera->initTransformation(event.position());

    event.setAccepted();
    redraw(); /* camera has changed, redraw! */
}

void KdTreeApplication::mouseReleaseEvent(MouseEvent&) {
    /* Disable mouse capture again */
    /** @todo replace once https://github.com/mosra/magnum/pull/419 is in */
    SDL_CaptureMouse(SDL_FALSE);
}

void KdTreeApplication::mouseMoveEvent(MouseMoveEvent& event) {
    if(!event.buttons()) return;

    if(event.modifiers() & MouseMoveEvent::Modifier::Shift)
        camera->translate(event.position());
    else camera->rotate(event.position());

    event.setAccepted();
    redraw(); /* camera has changed, redraw! */
}

void KdTreeApplication::mouseScrollEvent(MouseScrollEvent& event) {
    const Float delta = event.offset().y();
    if(Math::abs(delta) < 1.0e-2f) return;

    camera->zoom(delta);

    event.setAccepted();
    redraw(); /* camera has changed, redraw! */
}

MAGNUM_APPLICATION_MAIN(KdTreeApplication)
