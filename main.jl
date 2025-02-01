using BenchmarkTools
include("utils.jl")
# include("O_Quickhull.jl")
include("O_Quickhull_Parallel.jl")
include("O_Quickhull_Non-recursive_Parallel_version1.jl")
include("O_Quickhull_Non-recursive_Parallel_version2.jl")
include("O_Quickhull_Non-recursive_Parallel_version3.jl")

function main()
    sizes = [1_000_000, 10_000_000, 20_000_000, 30_000_000, 40_000_000, 50_000_000, 60_000_000, 70_000_000, 80_000_000, 90_000_000, 100_000_000]
    setNumbers = 1

    resultDirectory = "result/"

    # set benchmarking = true if want to benchmark
    benchmarking = true
    
    # dataType: 
        #1 for discs type, 
        #2 for hollow discs type
        #3 for square type
        #4 for hollow square type
        #5 for sun type
        #6 for hollow sun type
        #7 for circles type
    dataType = 7

    dataTypeName = ["discs", "hollowDiscs", "square", "hollowSquare", "sun","hollowSun","circles"]

    instanceNames = Vector{String}(undef, length(sizes)*setNumbers)

    # running times
    runningTimeOQhullParallel = Matrix{Float32}(undef, length(instanceNames), 7)
    runningTimeOQhullParallelNrV1 = Matrix{Float32}(undef, length(instanceNames), 7)
    runningTimeOQhullParallelNrV2 = Matrix{Float32}(undef, length(instanceNames), 7)
    runningTimeOQhullParallelNrV3 = Matrix{Float32}(undef, length(instanceNames), 7)
    Random.seed!(42)
    for i in 1:length(sizes)
        for j in 1:setNumbers
            k = (i-1)*setNumbers+j
            instanceNames[k] = string(dataTypeName[dataType], "_", sizes[i], "_", j)

            println()
            println("Consider instance ", instanceNames[k])

            # create random data
            points = Matrix{Float64}(undef, sizes[i], 2)
            if dataType == 1
                points = create_discs(sizes[i])
            elseif dataType == 2
                points = create_hollowDiscs(sizes[i])
            elseif dataType == 3
                points = create_square(sizes[i])
            elseif dataType == 4
                points = create_hollowSquare(sizes[i])
            elseif dataType == 5
                points = create_sun(sizes[i])
            elseif dataType == 6
                points = create_hollowSun(sizes[i])
            else
                points = create_circles(sizes[i])
            end

            if benchmarking
                println("Parallel_O_Quickhull")
                bmOQhullParallel = run(@benchmarkable find_o_quickhull_parallel($points) samples=5 seconds=10000)
                runningTimeOQhullParallel[k,:] = report(bmOQhullParallel, 2)
                println("Parallel_O_Quickhull_nr_v1")
                bmOQhullParallelNrV1 = run(@benchmarkable find_o_quickhull_parallel_nr_v1($points) samples=5 seconds=10000)
                runningTimeOQhullParallelNrV1[k,:] = report(bmOQhullParallelNrV1, 2)
                println("Parallel_O_Quickhull_nr_v2")
                bmOQhullParallelNrV2 = run(@benchmarkable find_o_quickhull_parallel_nr_v2($points) samples=5 seconds=10000)
                runningTimeOQhullParallelNrV2[k,:] = report(bmOQhullParallelNrV2, 2)
                println("Parallel_O_Quickhull_nr_v3")
                bmOQhullParallelNrV3 = run(@benchmarkable find_o_quickhull_parallel_nr_v3($points) samples=5 seconds=10000)
                runningTimeOQhullParallelNrV3[k,:] = report(bmOQhullParallelNrV3, 2)
             else
                
                 exportFileOQhullParallel = string(resultDirectory, instanceNames[k], "_OQHullParallel")
                 find_o_quickhull_parallel_nr_ex(points, exportFileOQhullParallel)

                 exportFileOQhull = string(resultDirectory, instanceNames[k], "_OQhull")
                 find_o_quickhull_ex(points, exportFileOQhull)
            end
        end
    end

    if benchmarking
        baseName = string(resultDirectory, dataTypeName[dataType], "_")

        exportReport(instanceNames, runningTimeOQhullParallel, string(baseName,"OQhullParallel_running_time.csv"))
        exportReport(instanceNames, runningTimeOQhullParallelNrV1, string(baseName,"OQhullParallelNrV1_running_time.csv"))
        exportReport(instanceNames, runningTimeOQhullParallelNrV2, string(baseName,"OQhullParallelNrV2_running_time.csv"))
        exportReport(instanceNames, runningTimeOQhullParallelNrV3, string(baseName,"OQhullParallelNrV3_running_time_6threads.csv"))
        
    end
end

main()
