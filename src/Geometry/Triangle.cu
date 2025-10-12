#include <Geometry/Triangle.cuh>

namespace renderer {
    __device__ bool Triangle::hit(const Ray & ray, const Range & range, HitRecord & record) const {
        const Vec3 h = ray.direction.cross(e2); //h = d x e2
        //系数行列式
        const float detA = e1.dot(h); //detA = e1 * (d x e2)

        //行列式为0，说明方程组无解或有无穷解（光线和三角形平行或有无数个交点）
        if (MathHelper::floatValueNearZero(detA)) {
            return false;
        }
        const Vec3 s = Point3::constructVector(vertexes[0], ray.origin); //s = O - v0

        //计算未知数U并检查
        static constexpr Range coefficientRange{0.0f, 1.0f};
        const float u = s.dot(h) / detA; // u = (s · h) / det
        if (!coefficientRange.inRange(u)) {
            return false;
        }
        const Vec3 q = s.cross(e1);  // q = s × e1

        //计算未知数V并检查
        const float v = ray.direction.dot(q) / detA; // v = (D · q) / det
        if (!coefficientRange.inRange(v) || u + v > 1.0f) {
            return false;
        }

        //满足相交条件，计算碰撞参数
        record.t = e2.dot(q) / detA; // t = (e2 · q) / det
        if (!range.inRange(record.t)) {
            return false;
        }
        record.hitPoint = ray.at(record.t);
        record.materialType = materialType;
        record.materialIndex = materialIndex;
        record.uvPair = {u, v};

        //交点法向量为三个顶点法向量的插值平滑
        const Vec3 n = ((1.0f - u - v) * vertexNormals[0] + u * vertexNormals[1] + v * vertexNormals[2]).unitVector();
        record.hitFrontFace = Vec3::dot(ray.direction, n) < 0.0f;
        record.normalVector = record.hitFrontFace ? n : -n;
        return true;
    }

    BoundingBox Triangle::constructBoundingBox() const {
        //找出三个顶点在每个轴分量的最值
        const Point3 & p1 = vertexes[0];
        const Point3 & p2 = vertexes[1];
        const Point3 & p3 = vertexes[2];
        const Point3 minPoint{
                std::min({p1[0], p2[0], p3[0]}),
                std::min({p1[1], p2[1], p3[1]}),
                std::min({p1[2], p2[2], p3[2]})
        };
        const Point3 maxPoint{
                std::max({p1[0], p2[0], p3[0]}),
                std::max({p1[1], p2[1], p3[1]}),
                std::max({p1[2], p2[2], p3[2]})
        };
        return {minPoint, maxPoint};
    }
}