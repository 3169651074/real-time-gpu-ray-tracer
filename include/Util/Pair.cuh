#ifndef RENDERERINTERACTIVE_PAIR_CUH
#define RENDERERINTERACTIVE_PAIR_CUH

namespace project {
    /*
     * Pair工具类，聚合类型
     * 由于std::pair不是聚合类型，则定义更加简单的Pair用于代替std::pair
     */
    template<typename T1, typename T2>
    struct Pair {
        T1 first;
        T2 second;
    };
}

#endif //RENDERERINTERACTIVE_PAIR_CUH
