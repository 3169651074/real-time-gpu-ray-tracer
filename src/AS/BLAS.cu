#include <AS/BLAS.cuh>

namespace renderer {
    BLASBuildResult BLAS::constructBLAS(
                const Sphere * spheres, size_t sphereStartIndex, size_t sphereCount,
                const Parallelogram * parallelograms, size_t parallelogramStartIndex, size_t parallelogramCount,
                const Triangle * triangles, size_t triangleStartIndex, size_t triangleCount)
    {
        //合并所有图元信息，构造统一信息列表
        //对于多图元物体，需要动态计算总图元数
        const size_t primitiveCount = sphereCount + parallelogramCount + triangleCount;
        std::vector<PrimitiveInfo> primitiveArray;
        primitiveArray.reserve(primitiveCount);

#define _constructPrimitiveArray(typeName, arrayName, startIndexName, countName)\
        do {\
            for (size_t i = 0; i < countName; i++) {\
                const PrimitiveInfo info = {\
                        .boundingBox = arrayName[startIndexName + i].constructBoundingBox(),\
                        .centroid = arrayName[startIndexName + i].centroid(),\
                        .type = PrimitiveType::typeName,\
                        .index = startIndexName + i\
                };\
                primitiveArray.push_back(info);\
            }\
        } while (false)

        _constructPrimitiveArray(SPHERE, spheres, sphereStartIndex, sphereCount);
        _constructPrimitiveArray(PARALLELOGRAM, parallelograms, parallelogramStartIndex, parallelogramCount);
        _constructPrimitiveArray(TRIANGLE, triangles, triangleStartIndex, triangleCount);
#undef _constructPrimitiveArray

        //二叉树数组，有 N 个叶子节点的二叉树共有 2N - 1 个节点（N - 1 个中间节点）
        std::vector<BLASNode> retTree;
        retTree.resize(2 * primitiveCount - 1); //预留空间同时创建元素

        //图元索引数组，长度为图元总数
        std::vector<Pair<PrimitiveType, size_t>> retIndexArray;
        retIndexArray.reserve(primitiveCount);

        //当前已构建的BLASNode数量，在构建完成后，就是实际的节点数量
        size_t nodeCount = 0;
        //任务栈
        std::stack<BuildingTask> stack;

        //创建根节点，将整个物体数组加入根任务
        stack.push({0, primitiveCount, 0});
        nodeCount++;

        while (!stack.empty()) {
            //弹出一个任务，根据类型选择不同的处理方式
            const BuildingTask & task = stack.top();
            stack.pop();

            //当前操作的节点，方便访问
            BLASNode & node = retTree[task.nodeIndex];

            if (task.primitiveCount <= PRIMITIVE_COUNT_PER_LEAF_NODE) {
                //叶子节点，将当前task的所有图元添加到叶子节点中
                node.primitiveCount = task.primitiveCount;

                //retIndexArray.size()：已经被分配到叶子节点中，并被添加到retIndexArray这个图元索引数组里的图元引用总数
                //只在处理叶子节点时增加
                node.index = retIndexArray.size();

                //构造包含当前节点所有图元的包围盒
                node.boundingBox = constructBoundingBoxForPrimitiveList(
                        primitiveArray, task.primitiveStartIndex,
                        task.primitiveStartIndex + task.primitiveCount);

                //添加到当前节点的多个图元信息到索引数组
                for (size_t i = 0; i < task.primitiveCount; i++) {
                    //从统一数据列表中取出图元索引信息
                    retIndexArray.push_back({
                            primitiveArray[task.primitiveStartIndex + i].type,
                            primitiveArray[task.primitiveStartIndex + i].index});
                }
            } else {
                //中间节点，为左右子节点分配节点索引
                const size_t leftChildIndex = nodeCount++;
                const size_t rightChildIndex = nodeCount++;

                //随机选择轴，将统一信息列表按照当前情况排序
                const int axis = RandomGenerator::randomInteger(0, 2);
                std::sort(
                        primitiveArray.begin() + (int)task.primitiveStartIndex,
                        primitiveArray.begin() + (int)(task.primitiveStartIndex + task.primitiveCount),
                        [axis](const PrimitiveInfo & a, const PrimitiveInfo & b)
                        { return a.centroid[axis] < b.centroid[axis]; });

                /*
                 * 创建当前节点
                 * 此处创建的包围盒并没有直接合并两个子节点的包围盒，因为构建顺序为从树的根节点向下
                 *   在构建中间节点时，两个叶子节点还不存在
                 * 当前任务包含的所有图元，包括了此节点的叶子节点所包含的所有图元，则包围盒符合要求
                 */
                node.boundingBox = constructBoundingBoxForPrimitiveList(
                        primitiveArray, task.primitiveStartIndex,
                        task.primitiveStartIndex + task.primitiveCount);
                node.primitiveCount = 0;
                node.index = leftChildIndex;

                //分割图元列表，根据空间排序结果确定左右子树的所有图元
                const size_t middleIndex = task.primitiveCount / 2;

                //创建左右节点的子任务并推到栈中，先推右节点，下一次循环先处理左子节点
                stack.push({task.primitiveStartIndex + middleIndex, task.primitiveCount - middleIndex, rightChildIndex});
                stack.push({task.primitiveStartIndex, middleIndex, leftChildIndex});
            }
        }

        //根据实际的节点数量释放未使用的数组空间
        retTree.resize(nodeCount);
        retTree.shrink_to_fit();

        return {retTree, retIndexArray};
    }

    __device__ bool BLAS::hit(
            const BLASNode * const __restrict__ treeArray, const Pair<PrimitiveType, size_t> * const __restrict__ indexArray,
            const Ray * ray, const Range * range, HitRecord * record,
            const Sphere * const __restrict__ spheres, const Parallelogram * const __restrict__ parallelograms,
            const Triangle * const __restrict__ triangles)
    {
        /*
         * 使用栈进行递归，栈存储待访问的BLASNode索引
         * 用固定大小的数组代替动态的stack容器以允许在GPU上运行
         */
        size_t stack[64];
        size_t stackSize = 0;
        //stack.push(0)，将根节点加入待访问列表
        stack[stackSize++] = 0;

        HitRecord tempRecord;
        bool isHit = false;
        Range currentRange = *range;

        while (stackSize > 0) {
            //stack.top + stack.pop：弹出栈顶元素。前置--对应后置++
            const size_t index = stack[--stackSize];

            //检查光线是否和当前节点的包围盒相交
            float t;
            if (!treeArray[index].boundingBox.hit(*ray, currentRange, t)) {
                //不相交，继续弹元素
                continue;
            }

            //相交，分为叶子节点和中间节点两种情况
            const BLASNode & node = treeArray[index];

            if (node.primitiveCount > 0) {
                //叶子节点，遍历叶子中的所有图元，依次进行相交测试
                for (size_t i = 0; i < node.primitiveCount; i++) {
                    const auto & pair = indexArray[node.index + i];
                    const PrimitiveType primitiveType = pair.first;
                    const size_t primitiveIndex = pair.second;

                    //根据相交的图元类型访问对应数组中的图元
                    switch (primitiveType) {
#define _primitiveHitTest(arrayName, typeName)\
                        case PrimitiveType::typeName:\
                            if (arrayName[primitiveIndex].hit(*ray, currentRange, tempRecord)) {\
                                isHit = true;\
                                currentRange.max = tempRecord.t;\
                                *record = tempRecord;\
                            }\
                            break

                        _primitiveHitTest(spheres, SPHERE);
                        _primitiveHitTest(parallelograms, PARALLELOGRAM);
                        _primitiveHitTest(triangles, TRIANGLE);
#undef _primitiveHitTest
                    }
                }
            } else {
                //中间节点，则将左右子节点的索引入栈
                //优先处理更近的节点，将更远的子节点先入栈，后处理
                const size_t leftID = node.index;
                const size_t rightID = leftID + 1;
                float tLeft, tRight;
                treeArray[leftID].boundingBox.hit(*ray, currentRange, tLeft);
                treeArray[rightID].boundingBox.hit(*ray, currentRange, tRight);

                const bool hitLeft = treeArray[leftID].boundingBox.hit(*ray, currentRange, tLeft);
                const bool hitRight = treeArray[rightID].boundingBox.hit(*ray, currentRange, tRight);
                if (hitLeft && hitRight) {
                    //先推入t值大的（远的）节点，后推入t值小的（近的）节点
                    if (tLeft > tRight) {
                        stack[stackSize++] = leftID;
                        stack[stackSize++] = rightID;
                    } else {
                        stack[stackSize++] = rightID;
                        stack[stackSize++] = leftID;
                    }
                } else if (hitLeft) {
                    //只有相交的节点才入栈，避免二次包围盒相交测试
                    stack[stackSize++] = leftID;
                } else if (hitRight) {
                    stack[stackSize++] = rightID;
                }
            }
        }
        return isHit;
    }

    BoundingBox BLAS::constructBoundingBoxForPrimitiveList(
            const std::vector<PrimitiveInfo> & primitives, size_t startIndex,size_t endIndex)
    {
        BoundingBox ret = primitives[startIndex].boundingBox;
        for (size_t i = startIndex + 1; i < (endIndex > primitives.size() ? primitives.size() : endIndex); i++) {
            ret = BoundingBox(ret, primitives[i].boundingBox);
        }
        return ret;
    }
}