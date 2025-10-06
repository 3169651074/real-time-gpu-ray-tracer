#include <AS/TLAS.cuh>

namespace renderer {
    TLASBuildResult TLAS::constructTLAS(Instance * instanceArray, size_t instanceCount) {
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
            const BuildingTask & task = stack.top();
            stack.pop();
            TLASNode & node = retTree[task.nodeIndex];

            if (task.instanceCount <= INSTANCE_COUNT_PER_LEAF_NODE) {
                //叶子节点
                node.instanceCount = task.instanceCount;
                node.index = retIndexArray.size();
                node.boundingBox = constructBoundingBoxForInstanceList(
                        instanceArray, task.instanceStartIndex,
                        task.instanceStartIndex + task.instanceCount);
                //添加此节点的每个实例对BLAS的引用信息到索引数组
                for (size_t i = 0; i < task.instanceCount; i++) {
                    retIndexArray.push_back(instanceArray[task.instanceStartIndex + i].asIndex);
                }
            } else {
                //中间节点
                const size_t leftChildIndex = nodeCount++;
                const size_t rightChildIndex = nodeCount++;

                //随机选择轴，将实例列表按照当前情况排序
                //直接操作参数数组
                const int axis = RandomGenerator::randomInteger(0, 2);
                std::sort(
                        instanceArray + (int)task.instanceStartIndex,
                        instanceArray + (int)(task.instanceStartIndex + task.instanceCount),
                        [axis](const Instance & a, const Instance & b)
                        { return a.transformedCentroid[axis] < b.transformedCentroid[axis]; });

                //创建当前节点
                node.boundingBox = constructBoundingBoxForInstanceList(
                        instanceArray, task.instanceStartIndex,
                        task.instanceStartIndex + task.instanceCount);
                node.instanceCount = 0;
                node.index = leftChildIndex;

                const size_t middleIndex = task.instanceCount / 2;
                stack.push({task.instanceStartIndex + middleIndex, task.instanceCount - middleIndex, rightChildIndex});
                stack.push({task.instanceStartIndex, middleIndex, leftChildIndex});
            }
        }
        return {retTree, retIndexArray};
    }

    __device__ bool TLAS::hit(
            const TLASNode * treeArray, const size_t * indexArray,
            const Ray * ray, const Range * range, HitRecord * record,
            const Instance * instances, const BLASArray * blasArray,
            const Sphere * spheres, const Parallelogram * parallelograms)
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
                    const size_t instanceIndex = indexArray[node.index + i];
                    const Instance & instance = instances[instanceIndex];

                    if (instance.hit(blasArray, ray, &currentRange, &tempRecord,
                                     spheres, parallelograms))
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

    BoundingBox TLAS::constructBoundingBoxForInstanceList(const Instance * instances, size_t startIndex, size_t endIndex) {
        BoundingBox ret = instances[startIndex].transformedBoundingBox;
        for (size_t i = startIndex + 1; i < endIndex; i++) {
            ret = BoundingBox(ret, instances[i].transformedBoundingBox);
        }
        return ret;
    }
}