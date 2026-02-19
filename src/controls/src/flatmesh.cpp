/*
 * Copyright (C) 2016 Florent Revest <revestflo@gmail.com>
 * All rights reserved.
 *
 * You may use this file under the terms of BSD license as follows:
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the author nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <QOpenGLShaderProgram>
#include <QOpenGLContext>
#include <QVector>
#include <QSettings>

#include "flatmesh.h"
#include "flatmeshgeometry.h"

// Convert the triangle-strip index buffer (which uses 0xFFFF as a primitive-restart marker)
// into a plain GL_TRIANGLES index list.  This removes the dependency on
// GL_PRIMITIVE_RESTART_FIXED_INDEX, which is only available in OpenGL ES 3.0+ and OpenGL 4.3+.
static QVector<unsigned short> buildTriangleIndices()
{
    QVector<unsigned short> result;
    QVector<unsigned short> strip;
    strip.reserve(32);

    // Flush one accumulated triangle strip into the output triangle list.
    // In a triangle strip the winding alternates: even triangles keep the natural order
    // (v[j-2], v[j-1], v[j]) while odd triangles swap the first two vertices
    // (v[j-1], v[j-2], v[j]) to maintain a consistent front-face orientation.
    // This also preserves the provoking vertex (last vertex) for flat shading.
    auto flushStrip = [&]() {
        for (int j = 2; j < strip.size(); j++) {
            if ((j % 2) == 0)
                result << strip[j-2] << strip[j-1] << strip[j];
            else
                result << strip[j-1] << strip[j-2] << strip[j];
        }
        strip.clear();
    };

    for (int i = 0, n = flatmesh_indices_sz; i < n; i++) {
        if (flatmesh_indices[i] == 0xFFFFu)
            flushStrip();
        else
            strip << flatmesh_indices[i];
    }
    flushStrip(); // handle the last strip (no trailing restart marker)

    return result;
}

// Return the cached explicit-triangle index buffer (computed once on first call).
static const QVector<unsigned short> &triangleIndices()
{
    static const QVector<unsigned short> s_indices = buildTriangleIndices();
    return s_indices;
}

// Our Adreno drivers fail to load shaders that are too long so we have to be concise and skip
// every unnecessary character such as spaces, \n, etc... This is effectively one long line!

// High-version shaders: GLSL ES 3.00 (OpenGL ES 3.x) / GLSL 3.30 (OpenGL 3.3+).
// Use in/out qualifiers, flat shading, and bitwise AND for safe power-of-2 modulo.
static const char *vertexShaderSourceHigh =
    // Attributes are per-vertex information, they give base coordinates and colors
    "in vec4 coord;"
    "in vec4 color;"

    // Uniforms are FlatMesh-wide, they give scaling information, the animation state or shifts
    "uniform mat4 matrix;"
    "uniform float shiftMix;"
    "uniform int loopNb;"
    "uniform vec2 shifts[" FLATMESH_SHIFTS_NB_STR "];"

    // The flat keyword enables flat shading (no interpolation between the vertices of a triangle)
    "flat out vec4 fragColor;"

    "void main()"
    "{"
         // Two vertices can have the same coordinate (if they give different colors to 2 triangles)
         // However, they need to move in sync, so we hash their coordinates as an index for shifts
        "int xHash=int(coord.x*100.0);"
        "int yHash=int(coord.y*100.0);"
        "int si=loopNb+xHash+yHash;"

         // Interpolate between (coord + shiftA) and (coord + shiftB) in the [-0.5, 0.5] domain.
         // Use bitwise AND instead of % so that negative si values still produce a valid index
         // (FLATMESH_SHIFTS_NB is a power of 2, so (si & (N-1)) == ((si % N + N) % N)).
        "vec2 pos=coord.xy+mix(shifts[si&(" FLATMESH_SHIFTS_NB_STR "-1)],"
                              "shifts[(si+1)&(" FLATMESH_SHIFTS_NB_STR "-1)],"
                              "shiftMix);"

        // Apply scene graph transformations (FlatMesh position and size) to get the final coords
        "gl_Position=matrix*vec4(pos,0,1);"

        // Forward the color in the vertex attribute to the fragment shaders
        "fragColor=color;"
    "}";

static const char *fragmentShaderSourceHigh =
    "#ifdef GL_ES\n"
    "precision mediump float;\n"
    "#endif\n"

    // The flat keyword disables interpolation in triangles
    // Each pixel gets the color of the last vertex of the triangle it belongs to
    "flat in vec4 fragColor;"
    "out vec4 color;"

    // Just keep the provided color
    "void main()"
    "{"
        "color=fragColor;"
    "}";

// Low-version shaders: GLSL ES 1.00 (OpenGL ES 2.x) / GLSL 1.20 (OpenGL 2.x).
// Use attribute/varying qualifiers, gl_FragColor, and float mod() instead of integer %.
// No flat qualifier is available so colors are smoothly interpolated across triangles.
static const char *vertexShaderSourceLow =
    "attribute vec4 coord;"
    "attribute vec4 color;"
    "uniform mat4 matrix;"
    "uniform float shiftMix;"
    "uniform int loopNb;"
    "uniform vec2 shifts[" FLATMESH_SHIFTS_NB_STR "];"
    "varying vec4 fragColor;"
    "void main()"
    "{"
        "int xHash=int(coord.x*100.0);"
        "int yHash=int(coord.y*100.0);"
        "int si=loopNb+xHash+yHash;"
        // mod() on floats always returns a non-negative result for a positive divisor,
        // so this handles negative si values correctly without risking an out-of-bounds index.
        "float N=float(" FLATMESH_SHIFTS_NB_STR ");"
        "vec2 s1=shifts[int(mod(float(si),N))];"
        "vec2 s2=shifts[int(mod(float(si+1),N))];"
        "vec2 pos=coord.xy+mix(s1,s2,shiftMix);"
        "gl_Position=matrix*vec4(pos,0.0,1.0);"
        "fragColor=color;"
    "}";

static const char *fragmentShaderSourceLow =
    "#ifdef GL_ES\n"
    "precision mediump float;\n"
    "#endif\n"
    "varying vec4 fragColor;"
    "void main()"
    "{"
        "gl_FragColor=fragColor;"
    "}";

// Return true when the current context supports GLSL ES 3.00 / GLSL 3.30 or higher.
static bool useHighVersionShaders()
{
    QOpenGLContext *ctx = QOpenGLContext::currentContext();
    if (!ctx)
        return false;
    const QSurfaceFormat fmt = ctx->format();
    if (ctx->isOpenGLES())
        return fmt.majorVersion() >= 3;
    // Desktop OpenGL: require at least 3.3 for the 'flat' qualifier and '#version 330'
    return (fmt.majorVersion() == 3 && fmt.minorVersion() >= 3)
        || fmt.majorVersion() > 3;
}

static QByteArray versionedShaderCode(const char *highSrc, const char *lowSrc)
{
    if (useHighVersionShaders()) {
        return (QOpenGLContext::currentContext()->isOpenGLES()
                ? QByteArrayLiteral("#version 300 es\n")
                : QByteArrayLiteral("#version 330\n"))
               + highSrc;
    }
    return (QOpenGLContext::currentContext()->isOpenGLES()
            ? QByteArrayLiteral("#version 100\n")
            : QByteArrayLiteral("#version 120\n"))
           + lowSrc;
}

// This class wraps the FlatMesh vertex and fragment shaders
class SGFlatMeshMaterialShader : public QSGMaterialShader
{
public:
    SGFlatMeshMaterialShader() {}
    const char *vertexShader() const override {
        static QByteArray source = versionedShaderCode(vertexShaderSourceHigh, vertexShaderSourceLow);
        return source.constData();
    }
    const char *fragmentShader() const override {
        static QByteArray source = versionedShaderCode(fragmentShaderSourceHigh, fragmentShaderSourceLow);
        return source.constData();
    }
    void updateState(const RenderState &state, QSGMaterial *newEffect, QSGMaterial *oldEffect) override {
        // On every run, update the animation state uniforms
        SGFlatMeshMaterial *material = static_cast<SGFlatMeshMaterial *>(newEffect);
        program()->setUniformValue(m_shiftMix_id, material->shiftMix());
        program()->setUniformValue(m_loopNb_id, material->loopNb());

        if (state.isMatrixDirty()) {
            // Vertices coordinates are always in the [-0.5, 0.5] range, modify QtQuick's projection matrix to do the scaling for us
            QMatrix4x4 combinedMatrix = state.combinedMatrix();
            combinedMatrix.scale(material->width(), material->height());
            combinedMatrix.translate(0.5, 0.5);
            combinedMatrix.scale(material->screenScaleFactor());
            program()->setUniformValue(m_matrix_id, combinedMatrix);
        }
    }
    char const *const *attributeNames() const override {
        // Map attribute numbers to attribute names in the vertex shader
        static const char *const attr[] = { "coord", "color", nullptr };
        return attr;
    }
private:
    void initialize() override {
        // Seed the array of shifts with pre-randomized shifts
        program()->setUniformValueArray("shifts", flatmesh_shifts, flatmesh_shifts_nb, 2);
        // Get the ids of the uniforms we regularly update
        m_matrix_id = program()->uniformLocation("matrix");
        m_shiftMix_id = program()->uniformLocation("shiftMix");
        m_loopNb_id = program()->uniformLocation("loopNb");
    }
    int m_matrix_id;
    int m_shiftMix_id;
    int m_loopNb_id;
};

QSGMaterialShader *SGFlatMeshMaterial::createShader() const
{
    return new SGFlatMeshMaterialShader;
}

FlatMesh::FlatMesh(QQuickItem *parent) : QQuickItem(parent), m_geometry(QSGGeometry::defaultAttributes_ColoredPoint2D(), flatmesh_vertices_sz, triangleIndices().size())
{
    // Don't overflow the item dimensions
    setClip(true);

    // Draw explicit triangles; no primitive-restart extension required
    m_geometry.setDrawingMode(QSGGeometry::DrawTriangles);

    // Dilate the FlatMesh more or less on squared or round screens
    QSettings machineConf("/etc/asteroid/machine.conf", QSettings::IniFormat);
    m_material.setScreenScaleFactor(machineConf.value("Display/ROUND", false).toBool() ? 1.2 : 1.7);

    // Iterate over all vertices and assign them the coordinates of their base point from flatmesh_vertices
    QSGGeometry::ColoredPoint2D *vertices = m_geometry.vertexDataAsColoredPoint2D();
    for (int i = 0; i < flatmesh_vertices_sz; i++) {
        vertices[i].x = flatmesh_vertices[i].x();
        vertices[i].y = flatmesh_vertices[i].y();
    }
    // Copy the pre-expanded triangle index buffer
    const QVector<unsigned short> &ti = triangleIndices();
    memcpy(m_geometry.indexData(), ti.constData(), ti.size() * sizeof(unsigned short));

    // Give initial colors to the vertices
    setColors(QColor("#ffaa39"), QColor("#df4829"));


    // m_animation interpolates the shiftMix, a float between 0.0 and 1.0
    // This is used by the vertex shader as the mix ratio between two shifts
    m_animation.setStartValue(0.0);
    m_animation.setEndValue(1.0);
    m_animation.setDuration(4000);
    m_animation.setLoopCount(-1);
    m_animation.setEasingCurve(QEasingCurve::InOutQuad);
    QObject::connect(&m_animation, &QVariantAnimation::currentLoopChanged, [this]() {
        m_material.incrementLoopNb();
    });
    QObject::connect(&m_animation, &QVariantAnimation::valueChanged, [this](const QVariant& value) {
        m_material.setShiftMix(value.toFloat());
        update();
    });

    // Run m_animation depending on the item's visibility
    connect(this, SIGNAL(visibleChanged()), this, SLOT(maybeEnableAnimation()));
    setAnimated(true);

    // Tell QtQuick we have graphic content and that updatePaintNode() needs to run
    setFlag(ItemHasContents);
}

void FlatMesh::updateColors()
{
    // Iterate over all vertices and give them the rgb values of the triangle they represent
    // In the flat shading model we use, each triangle is colored by its last vertex
    QSGGeometry::ColoredPoint2D *vertices = m_geometry.vertexDataAsColoredPoint2D();
    for (int i = 0; i < flatmesh_vertices_sz; i++) {
        // Ratios are pre-calculated to save some computation, we just need to do the mix
        // We do the color blending on the CPU because center and outer colors change rarely
        // and it would be a waste of GPU time to re-calculate that in every vertex shader
        float ratio = flatmesh_vertices[i].z();
        float inverse_ratio = 1-ratio;
        vertices[i].r = m_centerColor.red()*inverse_ratio + m_outerColor.red()*ratio;
        vertices[i].g = m_centerColor.green()*inverse_ratio + m_outerColor.green()*ratio;
        vertices[i].b = m_centerColor.blue()*inverse_ratio + m_outerColor.blue()*ratio;
    }
    m_geometryDirty = true;
}

void FlatMesh::setColors(QColor center, QColor outer)
{
    if (center == m_centerColor && outer == m_outerColor)
        return;
    m_centerColor = center;
    m_outerColor = outer;
    updateColors();
    update();
}

void FlatMesh::setCenterColor(QColor c)
{
    setColors(c, m_outerColor);
}

void FlatMesh::setOuterColor(QColor c)
{
    setColors(m_centerColor, c);
}

void FlatMesh::maybeEnableAnimation()
{
    // Only run the animation if the item is visible. No point running the shaders if this is hidden
    if (isVisible() && m_animated)
        m_animation.start();
    else
        m_animation.pause();
}

void FlatMesh::setAnimated(bool animated)
{
    if (animated == m_animated)
        return;
    m_animated = animated;
    emit animatedChanged();
    maybeEnableAnimation();
}

void FlatMesh::geometryChanged(const QRectF &newGeometry, const QRectF &oldGeometry)
{
    // On resizes, tell the vertex shader about the new size so the transformation matrix compensates it
    m_material.setSize(newGeometry.width(), newGeometry.height());

    QQuickItem::geometryChanged(newGeometry, oldGeometry);
}

// Called by the SceneGraph on every update()
QSGNode *FlatMesh::updatePaintNode(QSGNode *old, UpdatePaintNodeData *)
{
    // On the first update(), create a scene graph node for the mesh
    QSGGeometryNode *n = static_cast<QSGGeometryNode *>(old);
    if (!n) {
        n = new QSGGeometryNode;
        n->setOpaqueMaterial(&m_material);
        n->setGeometry(&m_geometry);
    }

    // On every update(), mark the material dirty so the shaders run again
    n->markDirty(QSGNode::DirtyMaterial);
    // And if colors changed, mark the geometry dirty so the new vertex attributes are sent to the GPU
    if (m_geometryDirty) {
        n->markDirty(QSGNode::DirtyGeometry);
        m_geometryDirty = false;
    }

    return n;
}
