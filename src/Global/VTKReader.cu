#include <Global/VTKReader.cuh>

#include <vtkSmartPointer.h>
#include <vtkPolyData.h>
#include <vtkPolyDataReader.h>
#include <vtkCellData.h>
#include <vtkDataArray.h>
#include <vtkFieldData.h>
#include <vtkCell.h>
#include <vtkCellTypes.h>
#include <vtkVersion.h>
#include <vtkPolyDataNormals.h>
#include <vtkPointData.h>

namespace project {
    std::vector<VTKParticle> VTKReader::readVTKFile(const std::string & filePath) {
        SDL_Log("VTK version: %s", vtkVersion::GetVTKVersion());
        SDL_Log("Reading vtk file: %s...", filePath.c_str());

        //检查VTK文件头
        std::ifstream file(filePath);
        if (!file.is_open()) {
            SDL_Log("Failed to open vtk file: %s!", filePath.c_str());
            exit(-1);
        }
        std::string line;
        getline(file, line);
        if (line.find("# vtk DataFile Version") == std::string::npos) {
            SDL_Log("Illegal vtk file header: %s. In file %s!", line.c_str(), filePath.c_str());
            exit(-1);
        }
        file.close();

        //读取VTK文件并获取vtkPolyData指针
        vtkSmartPointer<vtkPolyDataReader> reader = vtkSmartPointer<vtkPolyDataReader>::New();
        reader->SetFileName(filePath.c_str());
        reader->Update();
        vtkPolyData * polyData = reader->GetOutput();
        if (polyData == nullptr || polyData->GetNumberOfPoints() == 0) {
            SDL_Log("Failed to get poly data pointer or there is no points in this file!");
            exit(-1);
        }

        //全局点数量和粒子数量
        const vtkIdType numCells = polyData->GetNumberOfCells();
        SDL_Log("Total point count: %zd, cell count: %zd.", polyData->GetNumberOfPoints(), numCells);
        std::vector<VTKParticle> ret;
        ret.reserve(numCells);

        //读取几何数据
        vtkCellData * cellData = polyData->GetCellData();
        vtkDataArray * idArray = cellData ? cellData->GetArray("id") : nullptr;
        vtkDataArray * velArray = cellData ? cellData->GetArray("vel") : nullptr;
        if (cellData == nullptr || idArray == nullptr || velArray == nullptr) {
            SDL_Log("Failed to read cell data!");
            exit(-1);
        }

        //计算全局顶点法向量
        vtkNew<vtkPolyDataNormals> normalsFilter;
        normalsFilter->SetInputData(polyData);
        normalsFilter->SetComputePointNormals(true); // 计算顶点法向量
        normalsFilter->SetComputeCellNormals(false); // 不计算面法向量
        normalsFilter->SetSplitting(false);          // 不要因为法线差异而分裂顶点
        normalsFilter->SetConsistency(true);         // 尝试使所有法线方向一致
        normalsFilter->SetAutoOrientNormals(true);   // 将法向量定向到外侧
        normalsFilter->Update();

        vtkPolyData * resultWithNormals = normalsFilter->GetOutput();
        vtkDataArray * normals = resultWithNormals->GetPointData()->GetNormals();
        vtkPoints * meshPoints = polyData->GetPoints();

        //逐个Cell读取
        for (vtkIdType i = 0; i < numCells; i++) {
            VTKParticle particle{};

            //获取Cell作为独立几何对象
            vtkCell * cell = polyData->GetCell(i);

            //检查几何类型是否为vtkTriangleStrip
            if (strcmp(vtkCellTypes::GetClassNameFromTypeId(cell->GetCellType()), "vtkTriangleStrip") != 0) {
                SDL_Log("Found illegal cell type: %s, aborting!", vtkCellTypes::GetClassNameFromTypeId(cell->GetCellType()));
                exit(-1);
            }

            //ID
            particle.id = static_cast<size_t>(idArray->GetTuple1(i));

            //速度
            const double * vel = velArray->GetTuple3(i);
            particle.velocity = Vec3{
                static_cast<float>(vel[0]),
                static_cast<float>(vel[1]),
                static_cast<float>(vel[2])};

            //包围盒
            double bounds[6];
            cell->GetBounds(bounds);
            particle.bounds = std::array<Range, 3>{
                    Range{static_cast<float>(bounds[0]), static_cast<float>(bounds[1])},
                    Range{static_cast<float>(bounds[2]), static_cast<float>(bounds[3])},
                    Range{static_cast<float>(bounds[4]), static_cast<float>(bounds[5])},
            };

            //质心
            const vtkIdType numCellPoints = cell->GetNumberOfPoints();
            if (numCellPoints > 0) {
                double centroid[3] = {0.0, 0.0, 0.0};
                vtkPoints * points = cell->GetPoints();

                //累加所有顶点坐标
                for (vtkIdType j = 0; j < numCellPoints; j++) {
                    double point[3];
                    points->GetPoint(j, point);
                    centroid[0] += point[0];
                    centroid[1] += point[1];
                    centroid[2] += point[2];
                }

                //取平均值
                centroid[0] /= static_cast<double>(numCellPoints);
                centroid[1] /= static_cast<double>(numCellPoints);
                centroid[2] /= static_cast<double>(numCellPoints);

                particle.centroid = Point3{
                    static_cast<float>(centroid[0]),
                    static_cast<float>(centroid[1]),
                    static_cast<float>(centroid[2])};
            } else {
                SDL_Log("There is no points in cell %zd!", i);
                exit(-1);
            }

            //读取该Cell的所有顶点坐标和法向量
            particle.vertices.reserve(numCellPoints);
            particle.verticeNormals.reserve(numCellPoints);

            vtkIdList * pointIds = cell->GetPointIds();
            for (vtkIdType j = 0; j < numCellPoints; j++) {
                const vtkIdType pointId = pointIds->GetId(j);

                //获取顶点坐标
                double coords[3];
                meshPoints->GetPoint(pointId, coords);
                particle.vertices.push_back(Point3{
                        static_cast<float>(coords[0]),
                        static_cast<float>(coords[1]),
                        static_cast<float>(coords[2]),
                });

                //获取顶点法向量
                double normal[3];
                normals->GetTuple(pointId, normal);
                particle.verticeNormals.push_back(Vec3{
                    static_cast<float>(normal[0]),
                    static_cast<float>(normal[1]),
                    static_cast<float>(normal[2]),
                });
            }
            ret.push_back(particle);
        }

        SDL_Log("VTK file %s read completed.", filePath.c_str());
        return ret;
    }

    Pair<std::vector<Triangle>, std::vector<Instance>> VTKReader::convertToRendererData(
            const std::vector<VTKParticle> & particles)
    {
        SDL_Log("Converting VTK particles data...");

        std::vector<Triangle> triangles;
        std::vector<Instance> instances;
        instances.resize(particles.size()); //一个粒子对应一个实例

        size_t triangleIndex = 0;
        for (size_t i = 0; i < particles.size(); i++) {
            //获取组成粒子的所有点，构造三角形数组
            //vtkTriangleStrip由N个点组成N - 2个三角形
            const size_t triangleCount = particles[i].vertices.size() - 2;
            //预留空间
            triangles.reserve(triangles.size() + triangleCount);

            for (size_t j = 0; j < triangleCount; j++) {
                std::array<Point3, 3> vertices;
                std::array<Vec3, 3> normals;
                if ((j & 1) == 0) {
                    //偶数三角形，顶点顺序保持不变
                    vertices = {particles[i].vertices[j], particles[i].vertices[j + 1], particles[i].vertices[j + 2],};
                    normals = {particles[i].verticeNormals[j], particles[i].verticeNormals[j + 1], particles[i].verticeNormals[j + 2],};
                } else {
                    //奇数三角形，第2，3个顶点需要取反以保持面法线方向一致
                    vertices = {particles[i].vertices[j], particles[i].vertices[j + 2], particles[i].vertices[j + 1],};
                    normals = {particles[i].verticeNormals[j], particles[i].verticeNormals[j + 2], particles[i].verticeNormals[j + 1],};
                }

                //构造三角形并存入数组，当前使用固定材质
                triangles.emplace_back(
                        MaterialType::METAL, 0,
                        vertices, normals
                );
            }

            //构造实例对象
            instances[i].primitiveType = PrimitiveType::TRIANGLE;
            instances[i].primitiveIndex = triangleIndex; //当前实例管理的一组三角形在全局数组中的起始下标
            instances[i].primitiveCount = triangleCount;
            instances[i].boundingBox = BoundingBox(particles[i].bounds[0], particles[i].bounds[1], particles[i].bounds[2]);
            instances[i].centroid = particles[i].centroid;
            instances[i].updateTransformArguments(
                    {0.0, 4.0, 0.0},
                    {90.0, 0.0, 0.0},
                    {3.0, 3.0, 3.0}
            );
            triangleIndex += triangleCount;
        }

        SDL_Log("VTK data conversion completed.");
        return {triangles, instances};
    }
}
