using LinearAlgebra
using Base.Threads
using LoopVectorization
include("utils.jl")

@inline function find_special_points_parallel_nr_v3(points,startIndex, endIndex)
    Is = [
        I(points[startIndex,1],points[startIndex,2],points[startIndex,2]),
        I(points[startIndex,2],points[startIndex,1],points[startIndex,1]),
        I(points[startIndex,1],points[startIndex,2],points[startIndex,2]),
        I(points[startIndex,2],points[startIndex,1],points[startIndex,1])
    ]
    @inbounds @simd for i in startIndex+1:endIndex 
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
#song song tìm tập I và tổng hợp ra p theo cách chia tập dữ liệu
@inline function find_special_points_parallel_nr_v3(points)
    len = size(points,1)
    grid = mySplit(1,len, 64*nthreads())
    result = Vector{Vector{I}}(undef, length(grid)-1)
    @threads for i in 1:(length(grid)-1)
        result[i] = find_special_points_parallel_nr_v3(points, grid[i],grid[i+1]-1)
    end
    Is = [result[1][1],
    result[1][2],
    result[1][3],
    result[1][4]
    ]
    @inbounds @simd for i in 2:(length(grid)-1)
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


@inline function find_o_hull1_parallel_nr_v3(set1, p1_2,p2_1)
    len = size(set1, 1)
    if len == 0
        return Vector{Vector{Float64}}(undef, 0)
    end
    hull = Vector{Vector{Float64}}()
    stack = [(set1, p1_2, p2_1)]

    while !isempty(stack)
        (current_set, current_p1_2, current_p2_1) = pop!(stack)
        len = size(current_set, 1)

        maxset1 = ((current_set[1, 1] - current_p2_1[1])^2) + ((current_set[1, 2] - current_p1_2[2])^2)
        new_point1 = current_set[1, :]
        @avx for i in 2:len
            key1 = ((current_set[i, 1] - current_p2_1[1])^2) + ((current_set[i, 2] - current_p1_2[2])^2)
            ok1 = key1 > maxset1
            maxset1 = ok1 ? key1 : maxset1
            new_point1 = ok1 ? current_set[i, :] : new_point1
        end
        push!(hull,new_point1)
        mask = [Vector{Bool}(undef, size(current_set,1)) for _ in 1:2]
        @avx for i in 1:len
            mask[1][i] = current_set[i,1] > new_point1[1]
            mask[2][i] = current_set[i,2] > new_point1[2]           
        end
        new_set11 = current_set[mask[1], :]
        new_set12 = current_set[mask[2], :]
        if size(new_set11,1)!=0
            push!(stack, (new_set11, current_p1_2, new_point1))
        end
        if size(new_set12,1)!=0
            push!(stack, (new_set12, new_point1, current_p2_1))
        end
    end
    sort!(hull, rev = true)
    return hull
end

@inline function find_o_hull2_parallel_nr_v3(set2, p2_2, p3_1)
    len = size(set2, 1)
    if len == 0
        return Vector{Vector{Float64}}(undef, 0)
    end
    hull = Vector{Vector{Float64}}()
    stack = [(set2, p2_2, p3_1)]
    while !isempty(stack)
        (current_set, current_p2_2, current_p3_1) = pop!(stack)
        len = size(current_set, 1)

        maxset2 = ((current_set[1, 1] - current_p2_2[1])^2) + ((current_set[1, 2] - current_p3_1[2])^2)
        new_point2 = current_set[1, :]
        @avx for i in 2:len
            key2 = ((current_set[i, 1] - current_p2_2[1])^2) + ((current_set[i, 2] - current_p3_1[2])^2)
            ok2 = key2 > maxset2
            maxset2 = ok2 ? key2 : maxset2
            new_point2 = ok2 ? current_set[i, :] : new_point2
        end
        push!(hull, new_point2)
        mask = [Vector{Bool}(undef, size(current_set,1)) for _ in 1:2]
        @avx for i in 1:len
            mask[1][i] = current_set[i,2] > new_point2[2]
            mask[2][i] = current_set[i,1] < new_point2[1]           
        end
        new_set21 = current_set[mask[1], :]
        new_set22 = current_set[mask[2], :]
        if size(new_set21,1) !=0
            push!(stack, (new_set21, current_p2_2, new_point2))
        end
        if size(new_set22,1)!=0
            push!(stack, (new_set22, new_point2, current_p3_1))
        end
    end
    sort!(hull, rev = true)
    return hull
end

@inline function find_o_hull3_parallel_nr_v3(set3, p3_2,p4_1)
    len = size(set3, 1)
    if len == 0
        return Vector{Vector{Float64}}(undef, 0)
    end
    hull = Vector{Vector{Float64}}()
    stack = [(set3, p3_2, p4_1)]

    while !isempty(stack)
        (current_set, current_p3_2, current_p4_1) = pop!(stack)
        len = size(current_set, 1)

        maxset3 = ((current_set[1, 1] - current_p4_1[1])^2) + ((current_set[1, 2] - current_p3_2[2])^2)
        new_point3 = current_set[1, :]
        @avx for i in 2:len
            key3 = ((current_set[i, 1] - current_p4_1[1])^2) + ((current_set[i, 2] - current_p3_2[2])^2)
            ok3 = key3 > maxset3
            maxset3 = ok3 ? key3 : maxset3
            new_point3 = ok3 ? current_set[i, :] : new_point3
        end
        push!(hull, new_point3)
        mask = [Vector{Bool}(undef, size(current_set,1)) for _ in 1:2]
        @avx for i in 1:len
            mask[1][i] = current_set[i,1] < new_point3[1]
            mask[2][i] = current_set[i,2] < new_point3[2]           
        end
        new_set31 = current_set[mask[1], :]
        new_set32 = current_set[mask[2], :]
        if size(new_set31,1)!=0
            push!(stack, (new_set31, current_p3_2, new_point3))
        end
        if size(new_set32,1)!=0
            push!(stack, (new_set32, new_point3, current_p4_1))
        end
    end
    sort!(hull)
    return hull
end

@inline function find_o_hull4_parallel_nr_v3(set4, p4_2,p1_1)
    len = size(set4, 1)
    if len == 0
        return Vector{Vector{Float64}}(undef, 0)
    end
    hull = Vector{Vector{Float64}}()
    stack = [(set4, p4_2, p1_1)]

    while !isempty(stack)
        (current_set, current_p4_2, current_p1_1) = pop!(stack)
        len = size(current_set, 1)

        maxset4 = ((current_set[1, 1] - current_p4_2[1])^2) + ((current_set[1, 2] - current_p1_1[2])^2)
        new_point4 = current_set[1, :]
        @avx for i in 2:len
            key4 = ((current_set[i, 1] - current_p4_2[1])^2) + ((current_set[i, 2] - current_p1_1[2])^2)
            ok4 = key4 > maxset4
            maxset4 = ok4 ? key4 : maxset4
            new_point4 = ok4 ? current_set[i, :] : new_point4
        end
        push!(hull, new_point4)
        mask = [Vector{Bool}(undef, size(current_set,1)) for _ in 1:2]
        @avx for i in 1:len
            mask[1][i] = current_set[i,2] < new_point4[2]
            mask[2][i] = current_set[i,1] > new_point4[1]           
        end
        new_set41 = current_set[mask[1], :]
        new_set42 = current_set[mask[2], :]
        if size(new_set41,1)!=0
            push!(stack, (new_set41, current_p4_2, new_point4))
        end
        if size(new_set42,1)!=0
            push!(stack, (new_set42, new_point4, current_p1_1))
        end
    end
    sort!(hull)
    return hull
end

@inline function find_sets_and_o_hull_parallel_nr_v3(points, p_start, p_end, i)
    if i == 1
        set_p = points[(points[:,2] .> p_start[2]) .& (points[:,1] .> p_end[1]),:]
        arranged_points = find_o_hull1_parallel_nr_v3(set_p,p_start,p_end)
    elseif i == 2
        set_p = points[(points[:,1] .< p_start[1]) .& (points[:,2] .> p_end[2]),:]
        arranged_points = find_o_hull2_parallel_nr_v3(set_p,p_start,p_end)
    elseif i == 3
        set_p = points[(points[:,2] .< p_start[2]) .& (points[:,1] .< p_end[1]),:]
        arranged_points = find_o_hull3_parallel_nr_v3(set_p,p_start,p_end)
    else 
        set_p = points[(points[:,1] .> p_start[1]) .& (points[:,2] .< p_end[2]),:]
        arranged_points = find_o_hull4_parallel_nr_v3(set_p,p_start,p_end)
    end
    return arranged_points
end

@inline function find_o_quickhull_parallel_nr_v3(points)
    #song song chia tập dữ liệu, mỗi tập tìm tập I con, tổng hợp thanh 4 tập I và tìm p
    p = find_special_points_parallel_nr_v3(points)
    arranged_points = Vector{Vector{Vector{Float64}}}(undef,4)
    #song song 4 luồng theo 4 hướng tìm tập góc và tập cực biên
    @threads for i in 1:4 
        arranged_points[i] = find_sets_and_o_hull_parallel_nr_v3(points,p[i][1],p[i][2],i)
    end
    #tổng hợp kết quả
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
@inline function find_o_quickhull_parallel_nr_ex_v3(points, exportFile)
    arranged_points = find_o_quickhull_parallel_nr_v3(points)
    exportResult(arranged_points, exportFile)
end

points = create_discs(1000)
arranged_points1 = @time find_o_quickhull_parallel_nr_v3(points)
println()



# using Plots
# p = plot()
# scatter!(p,points[:,1],points[:,2],color=:red, markersize=2,legend=false)
# x = []
# y = []
# for i in 1:length(arranged_points1)
#     append!(x,arranged_points1[i][1])
#     append!(y,arranged_points1[i][2])
# end
# s =Vector{Tuple{Int, Vector{Float64}}}(undef,0)
# for i in 1:(length(arranged_points1)-1)
#     if arranged_points1[i+1][1] > arranged_points1[i][1] && arranged_points1[i+1][2] > arranged_points1[i][2]
#         push!(s,(i,[arranged_points1[i][1],arranged_points1[i+1][2]]))
#     elseif arranged_points1[i+1][1] > arranged_points1[i][1] && arranged_points1[i+1][2] < arranged_points1[i][2]
#         push!(s,(i,[arranged_points1[i+1][1],arranged_points1[i][2]]))
#     elseif arranged_points1[i+1][1] < arranged_points1[i][1] && arranged_points1[i+1][2] < arranged_points1[i][2]
#         push!(s,(i,[arranged_points1[i][1],arranged_points1[i+1][2]]))
#     elseif arranged_points1[i+1][1] < arranged_points1[i][1] && arranged_points1[i+1][2] > arranged_points1[i][2]
#         push!(s,(i,[arranged_points1[i+1][1],arranged_points1[i][2]]))        
#     end
# end

# for i in 1:length(s) 
#     insert!(x,s[i][1]+i,s[i][2][1])
#     insert!(y,s[i][1]+i,s[i][2][2])
# end
# plot!(x,y,color=:green, markersize=8)
