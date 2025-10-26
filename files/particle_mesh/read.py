import vtk

reader = vtk.vtkPolyDataReader()
reader.SetFileName("./particle_mesh/particle_000000000040000.vtk")
reader.Update()
output = reader.GetOutput()

cell_data = output.GetCellData()
print("Cell Data Arrays:")
for i in range(cell_data.GetNumberOfArrays()):
    array = cell_data.GetCellData(i)
    printf(f" {i}: Name='{array.GetName()}', Type={array.GetDataTypeAsString()}")

    
