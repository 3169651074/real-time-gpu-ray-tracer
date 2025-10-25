#include <AS/TLAS.cuh>

namespace project {
    TLASBuildResult TLAS::constructTLAS(const Instance * instanceArray, size_t instanceCount) {
        std::vector<InstanceAndIndex> instanceAndIndexArray(instanceCount);
        for (size_t i = 0; i < instanceCount; i++) {
            instanceAndIndexArray[i].instance = &instanceArray[i];
            instanceAndIndexArray[i].originalIndex = i;
        }

        //二叉树数组
        std::vector<TLASNode> retTree;
        retTree.resize(2 * instanceCount - 1);

        //索引数组
        std::vector<size_t> retIndexArray;
        retIndexArray.reserve(instanceCount);

        //当前已构建的TLASNode数量
        size_t nodeCount = 0;
        //任务栈
        std::stack<BuildingTask> stack;

        //创建根节点，将整个实例数组加入根任务
        stack.push({0, instanceCount, 0});
        nodeCount++;

        while (!stack.empty()) {
            //弹出一个任务，根据类型选择不同的处理方式
            /*
             * 此处不能按引用传递：在读取top之后，pop会释放掉原来节点的空间，导致下一次push时，task是悬垂引用，对其访问为未定义行为
             * 下一次push节点所在内存很可能是上一次pop出来的空间。当第一次push后，新的节点覆盖了task的空间，导致
             *   task的属性值被修改，从而第二次push中对task属性的读取出现错误
             * 修改方法：改为按值取元素而不是按引用
             *
             * 当总实例数量能够被一个叶子节点容纳时，没有触发节点分割逻辑中的push，悬垂引用所在内存没有被修改，可以正常工作
             */
            //const BuildingTask & task = stack.top();
            const BuildingTask task = stack.top();
            stack.pop();
            TLASNode & node = retTree[task.nodeIndex];

            //只有叶子节点才会向索引数组添加数据，索引数组中没有中间节点的信息
            if (task.instanceCount <= INSTANCE_COUNT_PER_LEAF_NODE) {
                //叶子节点
                node.instanceCount = task.instanceCount;
                //向后添加起始索引
                node.index = retIndexArray.size();
                node.boundingBox = constructBoundingBoxForInstanceList(
                        instanceAndIndexArray, task.instanceStartIndex,
                        task.instanceStartIndex + task.instanceCount);
                //添加此节点的每个实例对BLAS的引用信息到索引数组
                for (size_t i = 0; i < task.instanceCount; i++) {
                    retIndexArray.push_back(instanceAndIndexArray[task.instanceStartIndex + i].originalIndex);
//                    SDL_Log("Push oriIdx %zd to idxArray",
//                            instanceAndIndexArray[task.instanceStartIndex + i].originalIndex);
                }
            } else {
                //中间节点
                const size_t leftChildIndex = nodeCount++;
                const size_t rightChildIndex = nodeCount++;

                //随机选择轴，将实例列表按照当前情况排序
                const int axis = RandomGenerator::randomInteger(0, 2);
                std::sort(
                        instanceAndIndexArray.begin() + (int)task.instanceStartIndex,
                        instanceAndIndexArray.begin() + (int)(task.instanceStartIndex + task.instanceCount),
                        [axis](const InstanceAndIndex & a, const InstanceAndIndex & b)
                        { return a.instance->transformedCentroid[axis] < b.instance->transformedCentroid[axis]; });
                for (size_t i = 0; i < instanceCount; i++) {
//                    SDL_Log("Sort:InstanceArray[%zd]:oriIdx=%zd,asIdx=%zd",i,
//                            instanceAndIndexArray[i].originalIndex,instanceAndIndexArray[i].instance->asIndex);
                }

                //创建当前节点
                node.boundingBox = constructBoundingBoxForInstanceList(
                        instanceAndIndexArray, task.instanceStartIndex,
                        task.instanceStartIndex + task.instanceCount);
                node.instanceCount = 0;
                node.index = leftChildIndex;

                const size_t middleIndex = task.instanceCount / 2;
                //SDL_Log("Mid idx=%zd",middleIndex);

                //如果不修改悬垂引用，此版本恰好能工作，因为在push修改内存前读取了内存
//                const BuildingTask rightTask = {
//                        .instanceStartIndex = task.instanceStartIndex + middleIndex,
//                        .instanceCount = task.instanceCount - middleIndex,
//                        .nodeIndex = rightChildIndex
//                };
//                const BuildingTask leftTask = {
//                        .instanceStartIndex = task.instanceStartIndex,
//                        .instanceCount = middleIndex,
//                        .nodeIndex = leftChildIndex
//                };
//                stack.push(rightTask);
//                stack.push(leftTask);

                stack.push({
                                   task.instanceStartIndex + middleIndex,
                                   task.instanceCount - middleIndex,
                                   rightChildIndex
                });
                stack.push({
                                   task.instanceStartIndex,
                                   middleIndex,
                                   leftChildIndex
                });

//                SDL_Log("Split task: first push: stIdx=%zd,insCnt=%zd,nodeIdx=%zd",
//                        rightTask.instanceStartIndex,
//                        rightTask.instanceCount,
//                        rightTask.nodeIndex);
//                SDL_Log("Split task: second push: stIdx=%zd,insCnt=%zd,nodeIdx=%zd",
//                        leftTask.instanceStartIndex,
//                        leftTask.instanceCount,
//                        leftTask.nodeIndex);
            }
        }
        retTree.resize(nodeCount);
        retTree.shrink_to_fit();

        return {retTree, retIndexArray};
    }

    __device__ bool TLAS::hit(
            const TLASNode * const __restrict__ treeArray, const size_t * const __restrict__ indexArray,
            const Instance * const __restrict__ instances, const BLASArray * const __restrict__ blasArray,
            const Ray * ray, const Range * range, HitRecord * record,
            const Sphere * const __restrict__ spheres, const Parallelogram * const __restrict__ parallelograms,
            const Triangle * const __restrict__ triangles)
    {
        size_t stack[64];
        size_t stackSize = 0;
        stack[stackSize++] = 0;

        HitRecord tempRecord;
        bool isHit = false;
        Range currentRange = *range;

        while (stackSize > 0) {
            const size_t index = stack[--stackSize];

            float t;
            if (!treeArray[index].boundingBox.hit(*ray, currentRange, t)) {
                continue;
            }
            const TLASNode & node = treeArray[index];

            if (node.instanceCount > 0) {
                //叶子节点，遍历此节点所有实例
                for (size_t i = 0; i < node.instanceCount; i++) {
                    //取出此叶子节点包含的所有实例
                    //indexArray包含了实例在原始实例数组（instances）中的索引，因此instances[instanceIndex]取到原始实例
                    const size_t instanceIndex = indexArray[node.index + i];
                    const Instance & instance = instances[instanceIndex];

                    if (instance.hit(blasArray, ray, &currentRange, &tempRecord,
                                     spheres, parallelograms, triangles))
                    {
                        isHit = true;
                        currentRange.max = tempRecord.t;
                        *record = tempRecord;
                    }
                }
            } else {
                //中间节点
                const size_t leftID = node.index;
                const size_t rightID = leftID + 1;
                float tLeft, tRight;
                treeArray[leftID].boundingBox.hit(*ray, currentRange, tLeft);
                treeArray[rightID].boundingBox.hit(*ray, currentRange, tRight);

                const bool hitLeft = treeArray[leftID].boundingBox.hit(*ray, currentRange, tLeft);
                const bool hitRight = treeArray[rightID].boundingBox.hit(*ray, currentRange, tRight);

                if (hitLeft && hitRight) {
                    if (tLeft > tRight) {
                        stack[stackSize++] = leftID;
                        stack[stackSize++] = rightID;
                    } else {
                        stack[stackSize++] = rightID;
                        stack[stackSize++] = leftID;
                    }
                } else if (hitLeft) {
                    stack[stackSize++] = leftID;
                } else if (hitRight) {
                    stack[stackSize++] = rightID;
                }
            }
        }
        return isHit;
    }

    BoundingBox TLAS::constructBoundingBoxForInstanceList(const std::vector<InstanceAndIndex> & instanceAndIndexArray, size_t startIndex, size_t endIndex) {
        BoundingBox ret = instanceAndIndexArray[startIndex].instance->transformedBoundingBox;
        for (size_t i = startIndex + 1; i < endIndex; i++) {
            ret = BoundingBox(ret, instanceAndIndexArray[i].instance->transformedBoundingBox);
        }
        return ret;
    }
}