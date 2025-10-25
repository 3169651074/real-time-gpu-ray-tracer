#include <Global/RendererImpl.cuh>

namespace project {
#define _mallocGlobal(type, _arrayName, _countName, _className) \
        do {                                                 \
            if (type##DataWithPinPtr._countName != 0) {       \
                cudaCheckError(cudaMallocAsync(&type##DataWithDevPtr._arrayName, type##DataWithPinPtr._countName * sizeof(_className), stream));\
            }                                                \
        } while (false)
#define _mallocGeoGlobal(arrayName, countName, className) _mallocGlobal(geometry, arrayName, countName, className)
#define _mallocMatGlobal(arrayName, countName, className) _mallocGlobal(material, arrayName, countName, className)

#define _copyToGlobal(type, _arrayName, _countName, _className) \
        do {                                                 \
            if (type##DataWithPinPtr._countName != 0) {       \
                cudaCheckError(cudaMemcpyAsync(type##DataWithDevPtr._arrayName, type##DataWithPinPtr._arrayName, type##DataWithPinPtr._countName * sizeof(_className), cudaMemcpyHostToDevice, stream));\
        }                                                    \
        } while (false)
#define _copyGeoToGlobal(arrayName, countName, className) _copyToGlobal(geometry, arrayName, countName, className)
#define _copyMatToGlobal(arrayName, countName, className) _copyToGlobal(material, arrayName, countName, className)

#define _freeGlobal(type, _arrayName)                         \
        do {                                                  \
            cudaCheckError(cudaFreeAsync(type##DataWithDevPtr._arrayName, stream));\
            type##DataWithDevPtr._arrayName = nullptr;        \
        } while (false)
#define _freeGeoGlobal(arrayName) _freeGlobal(geometry, arrayName)
#define _freeMatGlobal(arrayName) _freeGlobal(material, arrayName)

    // ====== 几何体 ======

    SceneGeometryData RendererImpl::allocGeoGlobMem(cudaStream_t stream, const SceneGeometryData & geometryDataWithPinPtr) {
        //拷贝长度信息
        SceneGeometryData geometryDataWithDevPtr = geometryDataWithPinPtr;

        _mallocGeoGlobal(spheres, sphereCount, Sphere);
        _mallocGeoGlobal(parallelograms, parallelogramCount, Parallelogram);
        _mallocGeoGlobal(triangles, triangleCount, Triangle);

        return geometryDataWithDevPtr;
    }
    void RendererImpl::copyGeoToGlobMem(cudaStream_t stream, SceneGeometryData & geometryDataWithDevPtr, const SceneGeometryData & geometryDataWithPinPtr) {
        _copyGeoToGlobal(spheres, sphereCount, Sphere);
        _copyGeoToGlobal(parallelograms, parallelogramCount, Parallelogram);
        _copyGeoToGlobal(triangles, triangleCount, Triangle);
    }
    void RendererImpl::freeGeoGlobMem(cudaStream_t stream, SceneGeometryData & geometryDataWithDevPtr) {
        _freeGeoGlobal(spheres);
        _freeGeoGlobal(parallelograms);
        _freeGeoGlobal(triangles);

        geometryDataWithDevPtr = {};
    }

    // ====== 材质 ======

    SceneMaterialData RendererImpl::allocMatGlobMem(cudaStream_t stream, const SceneMaterialData & materialDataWithPinPtr) {
        SceneMaterialData materialDataWithDevPtr = materialDataWithPinPtr;

        _mallocMatGlobal(roughs, roughCount, Rough);
        _mallocMatGlobal(metals, metalCount, Metal);

        return materialDataWithDevPtr;
    }
    void RendererImpl::copyMatToGlobMem(cudaStream_t stream, SceneMaterialData & materialDataWithDevPtr, const SceneMaterialData & materialDataWithPinPtr) {
        _copyMatToGlobal(roughs, roughCount, Rough);
        _copyMatToGlobal(metals, metalCount, Metal);
    }
    void RendererImpl::freeMatGlobMem(cudaStream_t stream, SceneMaterialData & materialDataWithDevPtr) {
        _freeMatGlobal(roughs);
        _freeMatGlobal(metals);

        materialDataWithDevPtr = {};
    }

    // ====== 实例 ======

    Instance * RendererImpl::allocInstGlobMem(cudaStream_t stream, size_t instanceCount) {
        Instance * dev_instances;
        cudaCheckError(cudaMallocAsync(&dev_instances, instanceCount * sizeof(Instance), stream));
        return dev_instances;
    }
    void RendererImpl::copyInstToGlobMem(cudaStream_t stream, Instance * dev_instances, const Instance * pin_instances, size_t instanceCount) {
        cudaCheckError(cudaMemcpyAsync(dev_instances, pin_instances, instanceCount * sizeof(Instance), cudaMemcpyHostToDevice, stream));
    }
    void RendererImpl::freeInstGlobMem(cudaStream_t stream, Instance * & dev_instances) {
        cudaCheckError(cudaFreeAsync(dev_instances, stream));
        dev_instances = nullptr;
    }

    // ====== BLAS ======

    Pair<BLASArray *, size_t> RendererImpl::copyBLASToGlobMem(cudaStream_t stream, const Pair<BLASArray *, size_t> & pin_blas) {
        const size_t blasCount = pin_blas.second;

        //分配临时指针数组，BLASArray本身作为指针
        auto tempBlasArray = new BLASArray [blasCount];

        //逐个分配并拷贝BLAS
        for (size_t i = 0; i < blasCount; i++) {
            const auto & blas = pin_blas.first[i];
            const auto & blasNodeArray = blas.first.first;
            const size_t blasNodeArrayLength = blas.first.second;
            const auto & blasIndexArray = blas.second.first;
            const size_t blasIndexArrayLength = blas.second.second;

            //Node
            cudaCheckError(cudaMallocAsync(&tempBlasArray[i].first.first, blasNodeArrayLength * sizeof(BLASNode), stream));
            cudaCheckError(cudaMemcpyAsync(tempBlasArray[i].first.first, blasNodeArray, blasNodeArrayLength * sizeof(BLASNode), cudaMemcpyHostToDevice, stream));
            tempBlasArray[i].first.second = blasNodeArrayLength;

            //Index
            cudaCheckError(cudaMallocAsync(&tempBlasArray[i].second.first, blasIndexArrayLength * sizeof(BLASIndex), stream));
            cudaCheckError(cudaMemcpyAsync(tempBlasArray[i].second.first, blasIndexArray, blasIndexArrayLength * sizeof(BLASIndex), cudaMemcpyHostToDevice, stream));
            tempBlasArray[i].second.second = blasIndexArrayLength;
        }

        //将临时指针数组的指针和长度拷贝到设备
        Pair<BLASArray *, size_t> ret{};

        cudaCheckError(cudaMallocAsync(&ret.first, blasCount * sizeof(BLASArray), stream));
        cudaCheckError(cudaMemcpyAsync(ret.first, tempBlasArray, blasCount * sizeof(BLASArray), cudaMemcpyHostToDevice, stream));
        ret.second = blasCount;
        delete[] tempBlasArray;

        return ret;
    }
    void RendererImpl::freeBLASGlobMem(cudaStream_t stream, Pair<BLASArray *, size_t> & blasWithDevPtr) {
        const size_t blasCount = blasWithDevPtr.second;

        //1. 在主机端创建一个临时数组来接收从设备传回的BLAS指针数组
        auto tempBlasArray = new BLASArray [blasCount];
        //2. 将设备端的BLAS指针数组拷贝回主机端的临时数组
        cudaCheckError(cudaMemcpyAsync(tempBlasArray, blasWithDevPtr.first, blasCount * sizeof(BLASArray), cudaMemcpyDeviceToHost, stream));

        //3. 遍历主机端的临时数组，释放其中存储的每一个设备指针
        for (size_t i = 0; i < blasCount; i++) {
            //Node
            cudaCheckError(cudaFreeAsync(tempBlasArray[i].first.first, stream));
            //Index
            cudaCheckError(cudaFreeAsync(tempBlasArray[i].second.first, stream));
        }
        //4. 释放设备端的BLAS指针数组本身
        cudaCheckError(cudaFreeAsync(blasWithDevPtr.first, stream));
        //5. 释放主机端的临时数组
        delete[] tempBlasArray;

        blasWithDevPtr = {};
    }

    // ====== TLAS ======

    TLASArray RendererImpl::copyTLASGlobMem(cudaStream_t stream, const TLASArray & pin_tlas) {
        TLASArray ret{};
        const auto & tlasNodeArray = pin_tlas.first.first;
        const size_t tlasNodeArrayLength = pin_tlas.first.second;
        const auto & tlasIndexArray = pin_tlas.second.first;
        const size_t tlasIndexArrayLength = pin_tlas.second.second;

        //Node
        cudaCheckError(cudaMallocAsync(&ret.first.first, tlasNodeArrayLength * sizeof(TLASNode), stream));
        cudaCheckError(cudaMemcpyAsync(ret.first.first, tlasNodeArray, tlasNodeArrayLength * sizeof(TLASNode), cudaMemcpyHostToDevice, stream));
        ret.first.second = tlasNodeArrayLength;

        //Index
        cudaCheckError(cudaMallocAsync(&ret.second.first, tlasIndexArrayLength * sizeof(TLASIndex), stream));
        cudaCheckError(cudaMemcpyAsync(ret.second.first, tlasIndexArray, tlasIndexArrayLength * sizeof(TLASIndex), cudaMemcpyHostToDevice, stream));
        ret.second.second = tlasIndexArrayLength;

        return ret;
    }
    void RendererImpl::freeTLASGlobMem(cudaStream_t stream, TLASArray & tlasWithDevPtr) {
        cudaCheckError(cudaFreeAsync(tlasWithDevPtr.first.first, stream));
        cudaCheckError(cudaFreeAsync(tlasWithDevPtr.second.first, stream));

        tlasWithDevPtr = {};
    }

    // ====== 相机 ======

    void RendererImpl::copyCamToConstMem(cudaStream_t stream, const Camera * pin_camera) {
        cudaCheckError(cudaMemcpyToSymbolAsync(dev_camera, pin_camera, sizeof(Camera), 0, cudaMemcpyHostToDevice, stream));
    }
}