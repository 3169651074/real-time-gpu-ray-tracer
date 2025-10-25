#ifndef RENDERERINTERACTIVE_TRIANGLE_CUH
#define RENDERERINTERACTIVE_TRIANGLE_CUH

#include <AS/BoundingBox.cuh>

namespace project {
    /*
     * 三角形类，普通类型
     */
    class Triangle {
    public:
        //顶点
        Point3 vertexes[3];

        //顶点法向量
        Vec3 vertexNormals[3];

        //两个边向量（预计算）
        Vec3 e1, e2;

        //材质
        MaterialType materialType;
        size_t materialIndex;

        //使用三个顶点构造三角形，所有顶点法向量设置为垂直于三角形平面且相等
        Triangle(MaterialType materialType, size_t materialIndex, const std::array<Point3, 3> & vertexes)
        : materialType(materialType), materialIndex(materialIndex), vertexes{}, vertexNormals{}, e1{}, e2{} {
            e1 = Point3::constructVector(vertexes[0], vertexes[1]); //p1, p2
            e2 = Point3::constructVector(vertexes[0], vertexes[2]); //p2, p3
            for (size_t i = 0; i < 3; i++) {
                this->vertexes[i] = vertexes[i];
                this->vertexNormals[i] = Vec3::cross(e1, e2).unitVector();
            }
        }

        //使用三个顶点和每个顶点的法向量构造三角形
        Triangle(MaterialType materialType, size_t materialIndex, const std::array<Point3, 3> & vertexes, const std::array<Vec3, 3> & vertexNormals)
        : materialType(materialType), materialIndex(materialIndex), vertexes{}, vertexNormals{}, e1{}, e2{}
        {
            for (size_t i = 0; i < 3; i++) {
                this->vertexes[i] = vertexes[i];
                this->vertexNormals[i] = vertexNormals[i];
            }
            e1 = Point3::constructVector(vertexes[0], vertexes[1]);
            e2 = Point3::constructVector(vertexes[0], vertexes[2]);
        }

        //使用Möller–Trumbore算法进行光线-三角形相交测试
        __device__ bool hit(const Ray & ray, const Range & range, HitRecord & record) const;

        BoundingBox constructBoundingBox() const;

        Point3 centroid() const {
            //返回三角形的重心
            Point3 ret;
            for (size_t i = 0; i < 3; i++) {
                ret[i] = vertexes[0][i] + vertexes[1][i] + vertexes[2][i];
                ret[i] /= 3.0f;
            }
            return ret;
        }

        size_t objectPrimitiveCount() const {
            return 1;
        }
    };
}

#endif //RENDERERINTERACTIVE_TRIANGLE_CUH
