using LinearAlgebra
using Base.Threads
using LoopVectorization
include("utils.jl")
mutable struct I 
    value :: Float64
    minValue :: Float64
    maxValue :: Float64
end

@inline function find_special_points_parallel(points,startIndex, endIndex)
    Is = [
        I(points[startIndex,1],points[startIndex,2],points[startIndex,2]),
        I(points[startIndex,2],points[startIndex,1],points[startIndex,1]),
        I(points[startIndex,1],points[startIndex,2],points[startIndex,2]),
        I(points[startIndex,2],points[startIndex,1],points[startIndex,1])
    ]
    # @inbounds @simd 
    for i in startIndex+1:endIndex 
        maxX_lg = points[i,1] > Is[1].value
        maxX_eq = points[i,1] == Is[1].value
        Is[1].value = maxX_lg ? points[i,1] : Is[1].value
        Is[1].minValue = maxX_lg ? points[i,2] : Is[1].minValue
        Is[1].maxValue = maxX_lg ? points[i,2] : Is[1].maxValue

        Is[1].minValue = (maxX_eq & (points[i,2] < Is[1].minValue)) ? points[i,2] : Is[1].minValue 
        Is[1].maxValue = (maxX_eq & (points[i,2] > Is[1].maxValue)) ? points[i,2] : Is[1].maxValue
        
        minX_gt = points[i,1] < Is[3].value
        minX_eq = points[i,1] == Is[3].value
        Is[3].value = minX_gt ? points[i,1] : Is[3].value
        Is[3].minValue = minX_gt ? points[i,2] : Is[3].minValue
        Is[3].maxValue = minX_gt ? points[i,2] : Is[3].maxValue

        Is[3].minValue = (minX_eq & (points[i,2] < Is[3].minValue)) ? points[i,2] : Is[3].minValue 
        Is[3].maxValue = (minX_eq & (points[i,2] > Is[3].maxValue)) ? points[i,2] : Is[3].maxValue

        maxY_lg = points[i,2] > Is[2].value
        maxY_eq = points[i,2] == Is[2].value
        Is[2].value = maxY_lg ? points[i,2] : Is[2].value
        Is[2].minValue = maxY_lg ? points[i,1] : Is[2].minValue
        Is[2].maxValue = maxY_lg ? points[i,1] : Is[2].maxValue

        Is[2].minValue = (maxY_eq & (points[i,1] < Is[2].minValue)) ? points[i,1] : Is[2].minValue 
        Is[2].maxValue = (maxY_eq & (points[i,1] > Is[2].maxValue)) ? points[i,1] : Is[2].maxValue 
        
        minY_gt = points[i,2] < Is[4].value
        minY_eq = points[i,2] == Is[4].value
        Is[4].value = minY_gt ? points[i,2] : Is[4].value
        Is[4].minValue = minY_gt ? points[i,1] : Is[4].minValue
        Is[4].maxValue = minY_gt ? points[i,1] : Is[4].maxValue

        Is[4].minValue = (minY_eq & (points[i,1] < Is[4].minValue)) ? points[i,1] : Is[4].minValue 
        Is[4].maxValue = (minY_eq & (points[i,1] > Is[4].maxValue)) ? points[i,1] : Is[4].maxValue 
    end
    return Is
end


@inline function find_special_points_parallel(points)
    len = size(points,1)
    grid = mySplit(1,len, 64*nthreads())
    result = Vector{Vector{I}}(undef, length(grid)-1)
    @threads for i in 1:(length(grid)-1)
        result[i] = find_special_points_parallel(points, grid[i],grid[i+1]-1)
    end
    Is = [result[1][1],
    result[1][2],
    result[1][3],
    result[1][4]
    ]
    # @inbounds @simd 
    for i in 2:(length(grid)-1)
        result1_gt = result[i][1].value > Is[1].value 
        result1_eq = result[i][1].value == Is[1].value
        Is[1] = result1_gt ? result[i][1] : Is[1]

        Is[1].minValue = (result1_eq & (result[i][1].minValue < Is[1].minValue)) ? result[i][1].minValue : Is[1].minValue
        Is[1].maxValue = (result1_eq & (result[i][1].maxValue > Is[1].maxValue)) ? result[i][1].maxValue : Is[1].maxValue
        
        result2_gt = result[i][2].value > Is[2].value 
        result2_eq = result[i][2].value == Is[2].value
        Is[2] = result2_gt ? result[i][2] : Is[2]

        Is[2].minValue = (result2_eq & (result[i][2].minValue < Is[2].minValue)) ? result[i][2].minValue : Is[2].minValue
        Is[2].maxValue = (result2_eq & (result[i][2].maxValue > Is[2].maxValue)) ? result[i][2].maxValue : Is[2].maxValue

        result3_lg = result[i][3].value < Is[3].value 
        result3_eq = result[i][3].value == Is[3].value
        Is[3] = result3_lg ? result[i][3] : Is[3]

        Is[3].minValue = (result3_eq & (result[i][3].minValue < Is[3].minValue)) ? result[i][3].minValue : Is[3].minValue
        Is[3].maxValue = (result3_eq & (result[i][3].maxValue > Is[3].maxValue)) ? result[i][3].maxValue : Is[3].maxValue

        result4_lg = result[i][4].value < Is[4].value
        result4_eq = result[i][4].value == Is[4].value
        Is[4] = result4_lg ? result[i][4] : Is[4]

        Is[4].minValue = (result4_eq & (result[i][4].minValue < Is[4].minValue)) ? result[i][4].minValue : Is[4].minValue
        Is[4].maxValue = (result4_eq & (result[i][4].maxValue > Is[4].maxValue)) ? result[i][4].maxValue : Is[4].maxValue
        
    end

    return [[Is[1].value,Is[1].maxValue], [Is[2].maxValue,Is[2].value]],
            [[Is[2].minValue,Is[2].value], [Is[3].value,Is[3].maxValue]],
            [[Is[3].value,Is[3].minValue], [Is[4].minValue,Is[4].value]],
            [[Is[4].maxValue,Is[4].value], [Is[1].value,Is[1].minValue]]
end

@inline function find_sets_parallel(points, p_start, p_end, i)
    if i == 1
        set_p = points[(points[:,2] .> p_start[2]) .& (points[:,1] .> p_end[1]),:]
    elseif i == 2 
        set_p = points[(points[:,1] .< p_start[1]) .& (points[:,2] .> p_end[2]),:]
    elseif i == 3
        set_p = points[(points[:,2] .< p_start[2]) .& (points[:,1] .< p_end[1]),:]
    else
        set_p = points[(points[:,1] .> p_start[1]) .& (points[:,2] .< p_end[2]),:]
    end
    return set_p
end

@inline function find_o_hull1_parallel(set1, p1_2,p2_1)
    len = size(set1,1)
    if len == 0
        return Vector{Vector{Float64}}(undef,0)
    end
    
    maxset1 = ((set1[1,1] - p2_1[1])^2) + ((set1[1,2] - p1_2[2])^2)
    new_point1 = set1[1,:]
    # @avx 
    for i in 2:len 
        key1 = ((set1[i,1] - p2_1[1])^2) + ((set1[i,2] - p1_2[2])^2)
        ok = key1 > maxset1
        maxset1 = ok ? key1 : maxset1
        new_point1 = ok ? set1[i,:] : new_point1
    end
    new_set11 = set1[set1[:,1] .> new_point1[1],:]
    new_set12 = set1[set1[:,2] .> new_point1[2],:]
    return append!(find_o_hull1_parallel(new_set11,p1_2, new_point1),[new_point1],find_o_hull1_parallel(new_set12,new_point1,p2_1))
end

@inline function find_o_hull2_parallel(set2, p2_2,p3_1)
    len = size(set2,1)
    if len == 0
        return Vector{Vector{Float64}}(undef,0)
    end
    
    maxset2 = ((set2[1,1] - p2_2[1])^2) + ((set2[1,2] - p3_1[2])^2)
    new_point2 = set2[1,:]
    # @avx 
    for i in 2:len 
        key2 = ((set2[i,1] - p2_2[1])^2) + ((set2[i,2] - p3_1[2])^2)
        ok2 = key2 > maxset2
        maxset2 = ok2 ? key2 : maxset2
        new_point2 = ok2 ? set2[i,:] : new_point2
    end
    new_set21 = set2[set2[:,2] .> new_point2[2],:]
    new_set22 = set2[set2[:,1] .< new_point2[1],:]

    return append!(find_o_hull2_parallel(new_set21,p2_2, new_point2),[new_point2],find_o_hull2_parallel(new_set22,new_point2,p3_1))
end

@inline function find_o_hull3_parallel(set3, p3_2,p4_1)
    len = size(set3,1)
    if len == 0
        return Vector{Vector{Float64}}(undef,0)
    end
    
    maxset3 = ((set3[1,1] - p4_1[1])^2) + ((set3[1,2] - p3_2[2])^2)
    new_point3 = set3[1,:]
    # @avx 
    for i in 2:len 
        key3 = ((set3[i,1] - p4_1[1])^2) + ((set3[i,2] - p3_2[2])^2)
        ok3 = key3 > maxset3
        maxset3 = ok3 ? key3 : maxset3
        new_point3 = ok3 ? set3[i,:] : new_point3
    end
    new_set31 = set3[set3[:,1] .< new_point3[1],:]
    new_set32 = set3[set3[:,2] .< new_point3[2],:]
    return append!(find_o_hull3_parallel(new_set31,p3_2, new_point3),[new_point3],find_o_hull3_parallel(new_set32,new_point3,p4_1))
end

@inline function find_o_hull4_parallel(set4, p4_2,p1_1)
    len = size(set4,1)
    if len == 0
        return Vector{Vector{Float64}}(undef,0)
    end
    
    maxset4 = ((set4[1,1] - p4_2[1])^2) + ((set4[1,2] - p1_1[2])^2)
    new_point4 = set4[1,:]
    # @avx 
    for i in 2:len 
        key4 = ((set4[i,1] - p4_2[1])^2) + ((set4[i,2] - p1_1[2])^2)
        ok4 = key4 > maxset4
        maxset4 = ok4 ? key4 : maxset4
        new_point4 = ok4 ? set4[i,:] : new_point4
    end
    new_set41 = set4[set4[:,2] .< new_point4[2],:]
    new_set42 = set4[set4[:,1] .> new_point4[1],:]
    return append!(find_o_hull4_parallel(new_set41,p4_2, new_point4),[new_point4],find_o_hull4_parallel(new_set42,new_point4,p1_1))
end

@inline function find_o_hull_parallel(set_p, p_start, p_end, i)
    if i == 1
        arranged_points = find_o_hull1_parallel(set_p,p_start,p_end)
    elseif i == 2
        arranged_points = find_o_hull2_parallel(set_p,p_start,p_end)
    elseif i == 3
        arranged_points = find_o_hull3_parallel(set_p,p_start,p_end)
    else 
        arranged_points = find_o_hull4_parallel(set_p,p_start,p_end)
    end
    return arranged_points
end
@inline function find_o_quickhull_parallel(points)
    p = find_special_points_parallel(points)
    arranged_points = Vector{Vector{Vector{Float64}}}(undef,4)
    @threads for i in 1:4 
        arranged_points[i] = find_o_hull_parallel(find_sets_parallel(points,p[i][1],p[i][2],i),p[i][1],p[i][2],i)
    end
    result = append!([p[1][1]], arranged_points[1],[p[1][2]])
    if p[1][2] != p[2][1]
        push!(result,p[2][1])
    end
    append!(result,arranged_points[2],[p[2][2]])

    if p[2][2]!= p[3][1]
        push!(result,p[3][1])
    end
    append!(result,arranged_points[3],[p[3][2]])
    
    if p[3][2]!=p[4][1]
        push!(result,p[4][1])
    end
    append!(result,arranged_points[4])
    if p[4][2] != p[1][1]
        push!(result,p[4][2])
    end
    push!(result,p[1][1])
    return result
end
@inline function find_o_quickhull_parallel_ex(points, exportFile)
    arranged_points = find_o_quickhull_parallel(points)
    exportResult(arranged_points, exportFile)
end

points = create_circles(3000)
arranged_points = @time find_o_quickhull_parallel(points)
println()

# # using Plots

# x = []
# y = []
# for i in 1:length(arranged_points)
#     push!(x,arranged_points[i][1])
#     push!(y, arranged_points[i][2])
# end

# s =Vector{Tuple{Int, Vector{Float64}}}(undef,0)
# for i in 1:(size(arranged_points,1)-1)
#     if arranged_points[i+1][1] > arranged_points[i][1] && arranged_points[i+1][2] > arranged_points[i][2]
#         push!(s,(i,[arranged_points[i][1],arranged_points[i+1][2]]))
#     elseif arranged_points[i+1][1] > arranged_points[i][1] && arranged_points[i+1][2] < arranged_points[i][2]
#         push!(s,(i,[arranged_points[i+1][1],arranged_points[i][2]]))
#     elseif arranged_points[i+1][1] < arranged_points[i][1] && arranged_points[i+1][2] < arranged_points[i][2]
#         push!(s,(i,[arranged_points[i][1],arranged_points[i+1][2]]))
#     elseif arranged_points[i+1][1] < arranged_points[i][1] && arranged_points[i+1][2] > arranged_points[i][2]
#         push!(s,(i,[arranged_points[i+1][1],arranged_points[i][2]]))        
#     end
# end

# for i in 1:length(s) 
#     insert!(x,s[i][1]+i,s[i][2][1])
#     insert!(y,s[i][1]+i,s[i][2][2])
# end
# push!(x,x[1])
# push!(y,y[1])

# scatter(points[:,1],points[:,2],color=:blue, markersize=1.5,legend=false,aspect_ratio = 1)
# scatter!(x,y, color=:red, markersize=2)
# plot!(x,y,color=:red, markersize=8)

