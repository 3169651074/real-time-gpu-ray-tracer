#ifndef RENDERERINTERACTIVE_BOUNDINGBOX_CUH
#define RENDERERINTERACTIVE_BOUNDINGBOX_CUH

#include <Basic/Ray.cuh>

namespace project {
    /*
     * 轴对齐包围盒，普通类型
     * 包围盒的构造由CPU完成，相交测试由GPU线程完成
     *
     * 构造方法：
     *   指定每个轴向范围（默认为空）
     *   使用两个对角点
     *   使用存储6个浮点数的bounds数组
     *   合并两个包围盒
     *
     * 对象操作：
     *   使用矩阵变换包围盒
     *   包围盒相交测试
     */
    class BoundingBox {
    private:
        //确保包围盒体积有效
        void ensureVolume() {
            for (auto & i : range) {
                if (i.length() < FLOAT_ZERO_VALUE) { i.expand(FLOAT_ZERO_VALUE); }
            }
        }

    public:
        Range range[3] {};

        // ====== 构造方法 ======
        //默认构造空包围盒
        explicit BoundingBox(const Range & x = Range(), const Range & y = Range(), const Range & z = Range()) {
            range[0] = x; range[1] = y; range[2] = z;
            ensureVolume();
        }

        //使用两个对角点构造包围盒
        BoundingBox(const Point3 & p1, const Point3 & p2) {
            //取两个点每个分量的有效值
            for (size_t i = 0; i < 3; i++) {
                range[i] = p1[i] < p2[i] ? Range{p1[i], p2[i]} : Range{p2[i], p1[i]};
            }
            ensureVolume();
        }

        //使用bounds数组构造包围盒
        //[x1, x2], [y1, y2], [z1, z2]
        explicit BoundingBox(const float bounds[6]) {
            for (size_t i = 0; i < 6; i += 2) {
                range[i / 2] = Range{bounds[i], bounds[i + 1]};
            }
        }

        //构造两个包围盒的合并
        BoundingBox(const BoundingBox & b1, const BoundingBox & b2) {
            //合并不会减小包围盒的体积
            for (size_t i = 0; i < 3; i++) {
                range[i] = Range::merge(b1.range[i], b2.range[i]);
            }
        }

        // ====== 对象操作 ======
        //使用矩阵变换包围盒的每个顶点
        BoundingBox transformBoundingBox(const Matrix & matrix) const;

        //相交测试
        __device__ bool hit(const Ray & ray, const Range & checkRange, float & t) const;

        std::string toString() const {
            char buf[200];
            snprintf(buf, 200, "BBox: x=[%.4lf,%.4lf],y=[%.4lf,%.4lf],z=[%.4lf,%.4lf]",
                     range[0].min,range[0].max,range[1].min,range[1].max,range[2].min,range[2].max);
            return {buf};
        }
    };
}

#endif //RENDERERINTERACTIVE_BOUNDINGBOX_CUH
