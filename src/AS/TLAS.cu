#include <AS/TLAS.cuh>

namespace renderer {
    TLASBuildResult TLAS::constructTLAS(const Instance * instanceArray, size_t instanceCount) {
        //SDL_Log("\n=== TLAS Construction Debug ===");
        //SDL_Log("Instance count: %zu", instanceCount);
        //SDL_Log("INSTANCE_COUNT_PER_LEAF_NODE: %d", INSTANCE_COUNT_PER_LEAF_NODE);

        //此处若对参数数组发生有效排序，则会导致main中对实例对象的按索引的更新出现不匹配，则排序拷贝数组
        //将实例和其在原始数组中的索引打包在一起，用于排序时保持对原始数组的引用
        typedef struct InstanceInfo {
            const Instance * instance;
            size_t originalIndex;
        } InstanceInfo;

        std::vector<InstanceInfo> copyInstanceArray;
        copyInstanceArray.reserve(instanceCount);
        for (size_t i = 0; i < instanceCount; i++) {
            copyInstanceArray.push_back({&instanceArray[i], i});
            //SDL_Log("Instance[%zu] originalIndex=%zu asIndex=%d centroid=(%f, %f, %f)",
            //        i, i, instanceArray[i].asIndex,
            //        instanceArray[i].transformedCentroid.x,
            //        instanceArray[i].transformedCentroid.y,
            //        instanceArray[i].transformedCentroid.z);
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

        //SDL_Log("\n--- Building BVH Tree ---");
        //int taskCounter = 0;

        while (!stack.empty()) {
            //弹出一个任务，根据类型选择不同的处理方式
            const BuildingTask task = stack.top();  // 复制值而不是引用
            stack.pop();
            TLASNode & node = retTree[task.nodeIndex];

            //SDL_Log("\nTask #%d: nodeIndex=%zu startIndex=%zu count=%zu",
            //        taskCounter++, task.nodeIndex, task.instanceStartIndex, task.instanceCount);

            if (task.instanceCount <= INSTANCE_COUNT_PER_LEAF_NODE) {
                //叶子节点
                node.instanceCount = task.instanceCount;
                node.index = retIndexArray.size();
                //SDL_Log("  -> LEAF NODE: indexArrayPos=%zu", node.index);

                //计算包围盒
                BoundingBox bbox = copyInstanceArray[task.instanceStartIndex].instance->transformedBoundingBox;
                for (size_t i = 1; i < task.instanceCount; i++) {
                    bbox = BoundingBox(bbox, copyInstanceArray[task.instanceStartIndex + i].instance->transformedBoundingBox);
                }
                node.boundingBox = bbox;
                //添加此节点的每个实例在原始实例数组中的索引到索引数组
                for (size_t i = 0; i < task.instanceCount; i++) {
                    size_t originalIdx = copyInstanceArray[task.instanceStartIndex + i].originalIndex;
                    retIndexArray.push_back(originalIdx);
                    //SDL_Log("     Instance at copyArray[%zu] -> originalIndex=%zu",
                    //        task.instanceStartIndex + i, originalIdx);
                }
            } else {
                //中间节点
                const size_t leftChildIndex = nodeCount++;
                const size_t rightChildIndex = nodeCount++;

                //SDL_Log("  -> INTERNAL NODE: leftChild=%zu rightChild=%zu",
                //        leftChildIndex, rightChildIndex);

                //随机选择轴，将实例列表按照当前情况排序
                //直接操作参数数组
                const int axis = RandomGenerator::randomInteger(0, 2);
                //SDL_Log("     Sorting on axis: %d", axis);

                //SDL_Log("     Before sort:");
                //for (size_t i = 0; i < task.instanceCount; i++) {
                //    size_t idx = task.instanceStartIndex + i;
                //    SDL_Log("       copyArray[%zu] originalIndex=%zu",
                //            idx, copyInstanceArray[idx].originalIndex);
                //}

                std::sort(
                        copyInstanceArray.begin() + (int)task.instanceStartIndex,
                        copyInstanceArray.begin() + (int)(task.instanceStartIndex + task.instanceCount),
                        [axis](const InstanceInfo & a, const InstanceInfo & b)
                        { return a.instance->transformedCentroid[axis] < b.instance->transformedCentroid[axis]; });

                //SDL_Log("     After sort:");
                //for (size_t i = 0; i < task.instanceCount; i++) {
                //    size_t idx = task.instanceStartIndex + i;
                //    SDL_Log("       copyArray[%zu] originalIndex=%zu",
                //            idx, copyInstanceArray[idx].originalIndex);
                //}

                //创建当前节点，计算包围盒
                BoundingBox bbox = copyInstanceArray[task.instanceStartIndex].instance->transformedBoundingBox;
                for (size_t i = 1; i < task.instanceCount; i++) {
                    bbox = BoundingBox(bbox, copyInstanceArray[task.instanceStartIndex + i].instance->transformedBoundingBox);
                }
                node.boundingBox = bbox;
                node.instanceCount = 0;
                node.index = leftChildIndex;

                const size_t middleIndex = task.instanceCount / 2;
                //SDL_Log("     Split at middle: %zu", middleIndex);

                //右子树
                const size_t rightStart = task.instanceStartIndex + middleIndex;
                const size_t rightCount = task.instanceCount - middleIndex;
                //SDL_Log("     Pushing RIGHT task: start=%zu count=%zu nodeIndex=%zu",
                //        rightStart, rightCount, rightChildIndex);
                stack.push({rightStart, rightCount, rightChildIndex});

                // 左子树
                size_t leftStart = task.instanceStartIndex;
                size_t leftCount = middleIndex;
                //SDL_Log("     Pushing LEFT task: start=%zu count=%zu nodeIndex=%zu",
                //        leftStart, leftCount, leftChildIndex);
                stack.push({leftStart, leftCount, leftChildIndex});
            }
        }
        retTree.resize(nodeCount);
        retTree.shrink_to_fit();

        //SDL_Log("\n--- Final Results ---");
        //SDL_Log("Total nodes created: %zu", nodeCount);
        //SDL_Log("Index array size: %zu", retIndexArray.size());
        //SDL_Log("Index array contents:");
        //for (size_t i = 0; i < retIndexArray.size(); i++) {
        //    SDL_Log("  [%zu] = %zu", i, retIndexArray[i]);
        //}
        //SDL_Log("=== TLAS Construction Complete ===\n");

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

}