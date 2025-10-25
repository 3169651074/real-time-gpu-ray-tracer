#include <Basic/Vec3.cuh>

namespace project {
    //生成每个分量都在指定范围内的随机向量
    Vec3 Vec3::randomVector(float componentMin, float componentMax) {
        Vec3 ret{};
        for (size_t i = 0; i < 3; i++) {
            ret[i] = RandomGenerator::randomDouble(componentMin, componentMax);
        }
        return ret;
    }
    __device__ Vec3 Vec3::randomVector(curandState * state, float componentMin, float componentMax) {
        Vec3 ret{};
        for (size_t i = 0; i < 3; i++) {
            ret[i] = RandomGenerator::randomDouble(state, componentMin, componentMax);
        }
        return ret;
    }

    //生成平面（x，y，0）上模长不大于指定长度的向量
    Vec3 Vec3::randomPlaneVector(float maxLength) {
        float x, y;
        do {
            x = RandomGenerator::randomDouble(-1.0f, 1.0f);
            y = RandomGenerator::randomDouble(-1.0f, 1.0f);
        } while (x * x + y * y > maxLength * maxLength);
        return {x, y, 0.0f};
    }
    __device__ Vec3 Vec3::randomPlaneVector(curandState * state, float maxLength) {
        float x, y;
        do {
            x = RandomGenerator::randomDouble(state, -1.0f, 1.0f);
            y = RandomGenerator::randomDouble(state, -1.0f, 1.0f);
        } while (x * x + y * y > maxLength * maxLength);
        return {x, y, 0.0f};
    }

    //生成模长为length的空间向量
    Vec3 Vec3::randomSpaceVector(float length) {
        Vec3 ret{};
        float lengthSquare;
        //先生成单位向量，再缩放到指定模长
        do {
            for (size_t i = 0; i < 3; i++) {
                ret[i] = RandomGenerator::randomDouble(-1.0f, 1.0f);
            }
            lengthSquare = ret.lengthSquared();
        } while (lengthSquare < FLOAT_ZERO_VALUE * FLOAT_ZERO_VALUE);

        //单位化ret，确定模长
        ret.unitize();
        return ret * length;
    }
    __device__ Vec3 Vec3::randomSpaceVector(curandState * state, float length) {
        Vec3 ret{};
        float lengthSquare;
        do {
            for (size_t i = 0; i < 3; i++) {
                ret[i] = RandomGenerator::randomDouble(state, -1.0f, 1.0f);
            }
            lengthSquare = ret.lengthSquared();
        } while (lengthSquare < FLOAT_ZERO_VALUE * FLOAT_ZERO_VALUE);
        ret.unitize();
        return ret * length;
    }

    //生成遵守按指定轴余弦分布的随机向量，非单位向量
    Vec3 Vec3::randomCosineVector(int axis, bool toPositive) {
        float coord[3];
        const auto r1 = RandomGenerator::randomDouble();
        const auto r2 = RandomGenerator::randomDouble();
        coord[0] = cos(2.0f * PI * r1) * 2.0f * sqrt(r2);
        coord[1] = sin(2.0f * PI * r1) * 2.0f * sqrt(r2);
        coord[2] = sqrt(1.0f - r2);
        switch (axis) {
            case 0:
                std::swap(coord[0], coord[2]);
                break;
            case 1:
                std::swap(coord[1], coord[2]);
                break;
            case 2:
            default:
                break;
        }
        if (!toPositive) {
            coord[axis] = -coord[axis];
        }
        return {coord[0], coord[1], coord[2]};
    }
    __device__ Vec3 Vec3::randomCosineVector(curandState * state, int axis, bool toPositive) {
        float coord[3];
        const auto r1 = RandomGenerator::randomDouble(state);
        const auto r2 = RandomGenerator::randomDouble(state);
        coord[0] = cos(2.0f * PI * r1) * 2.0f * sqrt(r2);
        coord[1] = sin(2.0f * PI * r1) * 2.0f * sqrt(r2);
        coord[2] = sqrt(1.0f - r2);
        switch (axis) {
            case 0: {
                const float tmp = coord[0];
                coord[0] = coord[2];
                coord[2] = tmp;
                break;
            }
            case 1: {
                const float tmp = coord[1];
                coord[1] = coord[2];
                coord[2] = tmp;
                break;
            }
            case 2:
            default:
                break;
        }
        if (!toPositive) {
            coord[axis] = -coord[axis];
        }
        return {coord[0], coord[1], coord[2]};
    }
}