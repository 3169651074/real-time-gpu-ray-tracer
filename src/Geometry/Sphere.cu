#include <Geometry/Sphere.cuh>

namespace project {
    __device__ bool Sphere::hit(const Ray & ray, const Range & range, HitRecord & record) const {
        //解一元二次方程，判断光线和球体的交点个数
        const Vec3 cq = Point3::constructVector(ray.origin, center);
        const Vec3 dir = ray.direction;
        const float a = Vec3::dot(dir, dir);
        const float b = -2.0f * Vec3::dot(cq, dir);
        const float c = Vec3::dot(cq, cq) - radius * radius;
        float delta = b * b - 4.0f * a * c;

        if (delta < 0.0f) return false;
        delta = sqrt(delta);

        //root1对应较小的t值，为距离摄像机较近的交点
        const float root1 = (-b - delta) / (a * 2.0f);
        const float root2 = (-b + delta) / (a * 2.0f);

        float root;
        if (range.inRange(root1)) { //先判断root1
            root = root1;
        } else if (range.inRange(root2)) {
            root = root2;
        } else {
            return false; //两个根均不在允许范围内
        }

        //设置碰撞信息
        record.t = root;
        record.hitPoint = ray.at(root);
        record.materialType = materialType;
        record.materialIndex = materialIndex;

        //outwardNormal为球面向外的单位法向量，通过此向量和光线方向向量的点积符号判断光线撞击了球的内表面还是外表面
        //若点积小于0，则两向量夹角大于90度，两向量不同方向
        const Vec3 outwardNormal = Point3::constructVector(center, record.hitPoint).unitVector();
        record.hitFrontFace = Vec3::dot(ray.direction, outwardNormal) < 0.0f;
        record.normalVector = record.hitFrontFace ? outwardNormal : -outwardNormal;

        //将碰撞点从世界坐标系变换到以球心为原点的局部坐标系：直接在世界坐标系中构造向量
        const Vec3 localVector = Point3::constructVector(center, record.hitPoint).unitVector();

        //计算纹理坐标
        const float theta = acos(-localVector[1]);
        const float phi = atan2(-localVector[2], localVector[0]) + PI;
        record.uvPair = {phi / (2.0f * PI), theta / PI};
        return true;
    }

    BoundingBox Sphere::constructBoundingBox() const {
        //以图元的几何中心为局部空间包围盒的原点
        const Vec3 edge = {radius, radius, radius};
        return {center - edge, center + edge};
    }
}