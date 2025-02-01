using Random
using Printf
using CSV
using DataFrames
using StatsBase
mutable struct I 
    value :: Float64
    minValue :: Float64
    maxValue :: Float64
end
function create_discs(n)
    points = Matrix{Float64}(undef,n,2)
    alpha = rand(0:359,n)
    bankinh = rand(0:149,n)
    points[:,1] = bankinh .* sin.(alpha)
    points[:,2] = bankinh .* cos.(alpha)
    points = round.(points,digits=1)
    return points
end

function create_hollowDiscs(n)
    points = Matrix{Float64}(undef,n,2)
    alpha = rand(0:359,n)
    bankinh =130 .+ rand(0:149,n)
    points[:,1] = bankinh .* sin.(alpha)
    points[:,2] = bankinh .* cos.(alpha)
    points = round.(points,digits=1)
    return points
end

function create_square(n)
    points = Matrix{Float64}(undef,n,2)
    points[:,1] = 200 .* rand(n)
    points[:,2] = 200 .* rand(n)
    points = round.(points,digits=1)
    return points
end

function create_hollowSquare(n)
    points = Matrix{Float64}(undef,n,2)
    for i in 1:n 
        if i <= n÷4
            points[i,1] = 75 + 25 * rand()
            points[i,2] = 100 * rand()
        elseif i <= n÷2
            points[i,1] = 25 * rand()
            points[i,2] = 100 * rand()
        elseif i <= 3n÷4
            points[i,1] = 100 * rand()
            points[i,2] = 75 + 25 * rand()
        else
            points[i,1] = 100 * rand()
            points[i,2] = 25 * rand()
        end
    end
    points = round.(points,digits=1)
    return points
end

function create_sun(n)
    points = Matrix{Float64}(undef,n,2)
    alpha = rand(0:99,n)
    bankinh = rand(1:300,n)
    points[:,1] = bankinh .* sin.(alpha)
    points[:,2] = bankinh .* cos.(alpha)
    points = round.(points,digits=1)
    return points
end

function create_hollowSun(n)
    points = Matrix{Float64}(undef,n,2)
    alpha = rand(0:99,n)
    bankinh = 380 .+ rand(1:300,n)
    points[:,1] = bankinh .* sin.(alpha)
    points[:,2] = bankinh .* cos.(alpha)
    points = round.(points,digits=1)
    return points
end

function create_circles(n)
    points = Matrix{Float64}(undef,n,2)
    scale1 = rand(n÷2)
    scale2 = rand(n-n÷2)
    points[1:n÷2,1] = 200 .* scale1 .-100
    points[1:n÷2,2] = sqrt.(10000 .- points[1:n÷2,1].^2)
    points[n÷2+1:n,1] = 200 .* scale2 .-100
    points[n÷2+1:n,2] = -sqrt.(10000 .- points[n÷2+1:n,1].^2)
    points = round.(points,digits=1)
    return points
end

function report(bm, ignore::Int)
    # times in nano seconds hence multiplied by 1.e-9
    sorted_times = sort(bm.times)
    s = max(1, length(sorted_times)-ignore)
    min_time = minimum(sorted_times[1:s])*1.e-9
    max_time = maximum(sorted_times[1:s])*1.e-9
    mean_time = mean(sorted_times[1:s])*1.e-9
    geomean_time = geomean(sorted_times[1:s])*1.e-9
    median_time = median(sorted_times[1:s])*1.e-9
    runs = length(bm)
    @printf("running time (seconds):  min=%.5f", min_time)
    @printf(" max=%.5f", max_time)
    @printf(" mean=%.5f", mean_time)
    @printf(" geomean=%.5f", geomean_time)
    @printf(" median=%.5f", median_time)
    print(" runs=", runs)
    println(" ignore=", ignore)
    return [min_time, max_time, mean_time, geomean_time, median_time, runs, ignore]
end

function exportReport(names, runningTime, file_name)
    df = DataFrame(instance = names,
                    min = map(i -> runningTime[i,1], collect(1:length(names))),
                    max = map(i -> runningTime[i,2], collect(1:length(names))),
                    mean = map(i -> runningTime[i,3], collect(1:length(names))),
                    geomean = map(i -> runningTime[i,4], collect(1:length(names))),
                    median = map(i -> runningTime[i,5], collect(1:length(names))),
                    runs = map(i -> runningTime[i,6], collect(1:length(names))),
                    ignore = map(i -> runningTime[i,7], collect(1:length(names))))
    CSV.write(file_name, df)
end

function exportResult(points, exportFile)
    jldsave(string(exportFile, ".jld2"); points)
    df = DataFrame(points)
    CSV.write(string(exportFile, ".csv"), df)
end

@inline function mySplit(startIndex, endIndex, numberOfSet)
    step = Int(ceil((endIndex-startIndex+1)/numberOfSet))
    grid = [startIndex:step:endIndex;endIndex+1]
    return grid
end