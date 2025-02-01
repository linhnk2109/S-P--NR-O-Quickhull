using LinearAlgebra
using Base.Threads
using LoopVectorization
include("utils.jl")

#=
version 2: O-Quickhull khử đệ quy, song song theo chia bộ dữ liệu
=#

@inline function find_special_points_parallel_nr_v2_mask(points,startIndex, endIndex)
    Is = [
        I(points[startIndex,1],points[startIndex,2],points[startIndex,2]),
        I(points[startIndex,2],points[startIndex,1],points[startIndex,1]),
        I(points[startIndex,1],points[startIndex,2],points[startIndex,2]),
        I(points[startIndex,2],points[startIndex,1],points[startIndex,1])
    ]
    @inbounds @simd for i in startIndex+1:endIndex 
        maxX_lt = points[i,1] > Is[1].value
        maxX_eq = points[i,1] == Is[1].value
        Is[1].value = maxX_lt ? points[i,1] : Is[1].value
        Is[1].minValue = maxX_lt ? points[i,2] : Is[1].minValue
        Is[1].maxValue = maxX_lt ? points[i,2] : Is[1].maxValue

        Is[1].minValue = (maxX_eq & (points[i,2] < Is[1].minValue)) ? points[i,2] : Is[1].minValue 
        Is[1].maxValue = (maxX_eq & (points[i,2] > Is[1].maxValue)) ? points[i,2] : Is[1].maxValue
        
        minX_gt = points[i,1] < Is[3].value
        minX_eq = points[i,1] == Is[3].value
        Is[3].value = minX_gt ? points[i,1] : Is[3].value
        Is[3].minValue = minX_gt ? points[i,2] : Is[3].minValue
        Is[3].maxValue = minX_gt ? points[i,2] : Is[3].maxValue

        Is[3].minValue = (minX_eq & (points[i,2] < Is[3].minValue)) ? points[i,2] : Is[3].minValue 
        Is[3].maxValue = (minX_eq & (points[i,2] > Is[3].maxValue)) ? points[i,2] : Is[3].maxValue

        maxY_lt = points[i,2] > Is[2].value
        maxY_eq = points[i,2] == Is[2].value
        Is[2].value = maxY_lt ? points[i,2] : Is[2].value
        Is[2].minValue = maxY_lt ? points[i,1] : Is[2].minValue
        Is[2].maxValue = maxY_lt ? points[i,1] : Is[2].maxValue

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
#song song tìm tập I và tổng hợp ra p
@inline function find_special_points_parallel_nr_v2_mask(points,grid)
    result = Vector{Vector{I}}(undef, length(grid)-1)
    @threads for i in 1:(length(grid)-1)
        result[i] = find_special_points_parallel_nr_v2_mask(points, grid[i],grid[i+1]-1)
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

        result3_lt = result[i][3].value < Is[3].value 
        result3_eq = result[i][3].value == Is[3].value
        Is[3] = result3_lt ? result[i][3] : Is[3]

        Is[3].minValue = (result3_eq & (result[i][3].minValue < Is[3].minValue)) ? result[i][3].minValue : Is[3].minValue
        Is[3].maxValue = (result3_eq & (result[i][3].maxValue > Is[3].maxValue)) ? result[i][3].maxValue : Is[3].maxValue

        result4_lt = result[i][4].value < Is[4].value
        result4_eq = result[i][4].value == Is[4].value
        Is[4] = result4_lt ? result[i][4] : Is[4]

        Is[4].minValue = (result4_eq & (result[i][4].minValue < Is[4].minValue)) ? result[i][4].minValue : Is[4].minValue
        Is[4].maxValue = (result4_eq & (result[i][4].maxValue > Is[4].maxValue)) ? result[i][4].maxValue : Is[4].maxValue
        
    end

    return [[Is[1].value,Is[1].maxValue], [Is[2].maxValue,Is[2].value]],
            [[Is[2].minValue,Is[2].value], [Is[3].value,Is[3].maxValue]],
            [[Is[3].value,Is[3].minValue], [Is[4].minValue,Is[4].value]],
            [[Is[4].maxValue,Is[4].value], [Is[1].value,Is[1].minValue]]
end

@inline function find_sets_v2_mask(points, startIndex, endIndex, p)
    current_points = @view points[startIndex:endIndex,:]
    len = size(current_points,1)
    mask = [Vector{Bool}(undef,len) for _ in 1:4]
    @avx for i in 1:len
        x = current_points[i,1]
        y = current_points[i,2]
        mask[1][i] = (y > p[1][1][2]) & (x > p[1][2][1])
        mask[2][i] = (x < p[2][1][1]) & (y > p[2][2][2])
        mask[3][i] = (y < p[3][1][2]) & (x < p[3][2][1])
        mask[4][i] = (x > p[4][1][1]) & (y < p[4][2][2])
    end
    return [current_points[mask[1],:],current_points[mask[2],:],current_points[mask[3],:],current_points[mask[4],:]]
    
end
# song song tìm tập góc
@inline function find_sets_parallel_nr_v2_mask(points, p, grid)
    result = Vector{Vector{Matrix{Float64}}}(undef,length(grid)-1)
    
    @threads for i in 1:(length(grid)-1)
        result[i] = find_sets_v2_mask(points, grid[i],grid[i+1]-1,p)
    end
    sets = [zeros(0,2) for _ in 1:4]
    for i in 1:(length(grid)-1)
        sets[1] = vcat(sets[1],result[i][1])
        sets[2] = vcat(sets[2],result[i][2])
        sets[3] = vcat(sets[3],result[i][3])
        sets[4] = vcat(sets[4],result[i][4])
    end
    return sets
end

@inline function find_newpoint13_v2_mask(current_set, startIndex, endIndex, pStart, pEnd)
    maxset = ((current_set[startIndex,1] - pEnd[1])^2) + ((current_set[startIndex,2] - pStart[2])^2)
    new_point = current_set[startIndex,:]
    @avx for i in (startIndex+1):endIndex
        key = ((current_set[i,1] - pEnd[1])^2) + ((current_set[i,2] - pStart[2])^2)
        ok = key > maxset
        maxset = ok ? key : maxset
        new_point = ok ? current_set[i,:] : new_point
    end
    return new_point
end
#song song tìm điểm xa nhất (trường hợp 1,3)
@inline function find_newpoint13_v2_mask(current_set, pStart, pEnd,grid)
    result = Vector{Vector{Float64}}(undef,length(grid)-1)
    @threads for i in 1:(length(grid)-1)
        result[i] = find_newpoint13_v2_mask(current_set,grid[i],grid[i+1]-1,pStart,pEnd)
    end
    maxset = ((result[1][1] - pEnd[1])^2) + ((result[1][2] - pStart[2])^2)
    new_point = result[1]
    @inbounds @simd for i in 2:(length(grid)-1)
        key = ((result[i][1] - pEnd[1])^2) + ((result[i][2] - pStart[2])^2)
        ok = key > maxset
        maxset = ok ? key : maxset
        new_point = ok ? result[i] : new_point
    end
    return new_point
end

@inline function find_newpoint24_v2_mask(current_set, startIndex, endIndex, pStart, pEnd)

    maxset = ((current_set[startIndex,1] - pStart[1])^2) + ((current_set[startIndex,2] - pEnd[2])^2)
    new_point = current_set[startIndex,:]
    @avx for i in (startIndex+1):endIndex
        key = ((current_set[i,1] - pStart[1])^2) + ((current_set[i,2] - pEnd[2])^2)
        ok = key > maxset
        maxset = ok ? key : maxset
        new_point = ok ? current_set[i,:] : new_point
    end
    return new_point
end
#song song tìm điểm xa nhất (trường hợp 2,4)
@inline function find_newpoint24_v2_mask(current_set, pStart, pEnd,grid)
    result = Vector{Vector{Float64}}(undef,length(grid)-1)
    @threads for i in 1:(length(grid)-1)
        result[i] = find_newpoint24_v2_mask(current_set,grid[i],grid[i+1]-1,pStart,pEnd)
    end
    maxset = ((result[1][1] - pStart[1])^2) + ((result[1][2] - pEnd[2])^2)
    new_point = result[1]
    for i in 2:(length(grid)-1)
        key = ((result[i][1] - pStart[1])^2) + ((result[i][2] - pEnd[2])^2)
        ok = key > maxset
        maxset = ok ? key : maxset
        new_point = ok ? result[i] : new_point
    end
    return new_point
end

@inline function find_newset_v2_mask(current_set, new_point, startIndex, endIndex,j)

    current_set2 = @view current_set[startIndex:endIndex,:]
    len = size(current_set2,1)
    mask = [Vector{Bool}(undef,len) for _ in 1:2]
    if j==1
        
        @avx for i in 1:len 
            mask[1][i] = current_set2[i,1] > new_point[1]
            mask[2][i] = current_set2[i,2] > new_point[2]
        end
    elseif j==2
        @avx for i in 1:len 
            mask[1][i] = current_set2[i,2] > new_point[2]
            mask[2][i] = current_set2[i,1] < new_point[1]
        end
    elseif j==3
        @avx for i in 1:len 
            mask[1][i] = current_set2[i,1] < new_point[1]
            mask[2][i] = current_set2[i,2] < new_point[2]
        end
    else
        @avx for i in 1:len 
            mask[1][i] = current_set2[i,2] < new_point[2]
            mask[2][i] = current_set2[i,1] > new_point[1]
        end
    end
    return [current_set2[mask[1],:],current_set2[mask[2],:]]
end
# song song tìm 2 tập con
@inline function find_newset_v2_mask(current_set,new_point, grid,j)
    result = Vector{Vector{Matrix{Float64}}}(undef,length(grid)-1)
    @threads for i in 1:(length(grid)-1)
        result[i] = find_newset_v2_mask(current_set, new_point, grid[i],grid[i+1]-1,j)
    end

    new_sets =[zeros(0,2) for _ in 1:2]
    for i in 1:(length(grid)-1)
        new_sets[1] = vcat(new_sets[1],result[i][1])
        new_sets[2] = vcat(new_sets[2],result[i][2])
    end
    return new_sets[1],new_sets[2]
end
@inline function find_o_hull1_parallel_nr_v2_mask(set1, p1_2,p2_1)
    len = size(set1,1)
    if len == 0
        return Vector{Vector{Float64}}(undef, 0)
    end
    hull = Vector{Vector{Float64}}()
    stack = [(set1, p1_2, p2_1)]

    while !isempty(stack)
        (current_set, current_p1_2, current_p2_1) = pop!(stack)
        clen = size(current_set,1)
        grid = mySplit(1,clen,64*nthreads())
        new_point1 = find_newpoint13_v2_mask(current_set,current_p1_2,current_p2_1, grid)
        push!(hull,new_point1)
        new_set11,new_set12 = find_newset_v2_mask(current_set, new_point1,grid,1)
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

@inline function find_o_hull2_parallel_nr_v2_mask(set2, p2_2, p3_1)
    len = size(set2,1)
    if len == 0
        return Vector{Vector{Float64}}(undef, 0)
    end
    hull = Vector{Vector{Float64}}()
    stack = [(set2, p2_2, p3_1)]
    while !isempty(stack)
        (current_set, current_p2_2, current_p3_1) = pop!(stack)
        clen = size(current_set,1)
        grid = mySplit(1,clen,64*nthreads())
        new_point2 = find_newpoint24_v2_mask(current_set, current_p2_2,current_p3_1,grid)
        push!(hull, new_point2)
        new_set21,new_set22 = find_newset_v2_mask(current_set,new_point2,grid,2)
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

@inline function find_o_hull3_parallel_nr_v2_mask(set3, p3_2,p4_1)
    len = size(set3,1)
    if len == 0
        return Vector{Vector{Float64}}(undef, 0)
    end
    hull = Vector{Vector{Float64}}()
    stack = [(set3, p3_2, p4_1)]

    while !isempty(stack)
        (current_set, current_p3_2, current_p4_1) = pop!(stack)
        clen = size(current_set,1)
        grid = mySplit(1,clen,64*nthreads())
        new_point3 = find_newpoint13_v2_mask(current_set,current_p3_2,current_p4_1,grid)
        push!(hull, new_point3)
        new_set31,new_set32 = find_newset_v2_mask(current_set,new_point3,grid,3)
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

@inline function find_o_hull4_parallel_nr_v2_mask(set4, p4_2,p1_1)
    len = size(set4,1)
    if len == 0
        return Vector{Vector{Float64}}(undef, 0)
    end
    hull = Vector{Vector{Float64}}()
    stack = [(set4, p4_2, p1_1)]

    while !isempty(stack)
        (current_set, current_p4_2, current_p1_1) = pop!(stack)
        clen = size(current_set,1)
        grid = mySplit(1,clen,64*nthreads())
        new_point4 = find_newpoint24_v2_mask(current_set, current_p4_2,current_p1_1,grid)
        push!(hull, new_point4)
        new_set41,new_set42 = find_newset_v2_mask(current_set, new_point4,grid,4)
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

@inline function find_o_quickhull_parallel_nr_v2_mask(points)
    len = size(points,1)
    grid = mySplit(1,len, 64*nthreads())
    # song song tìm p bằng cách chia tập điểm theo grid
    p = find_special_points_parallel_nr_v2_mask(points, grid)
    #song song tìm tập góc bằng cách chia tập điểm theo grid:
    #mỗi tập tìm ra 4 tập góc con, sau đó tổng hợp ra 4 tập góc
    sets = find_sets_parallel_nr_v2_mask(points,p,grid)

    arranged_points = Vector{Vector{Vector{Float64}}}(undef,4)
    # song song tìm tập biên mỗi góc bằng cách chia nhỏ tập góc: song song tìm điểm xa nhất, 
    #từ điểm xa nhất song song tìm 2 tập con.
    #lần lượt với 4 tập góc
    arranged_points[1] = find_o_hull1_parallel_nr_v2_mask(sets[1],p[1][1],p[1][2])
    arranged_points[2] = find_o_hull2_parallel_nr_v2_mask(sets[2],p[2][1],p[2][2])
    arranged_points[3] = find_o_hull3_parallel_nr_v2_mask(sets[3],p[3][1],p[3][2])
    arranged_points[4] = find_o_hull4_parallel_nr_v2_mask(sets[4],p[4][1],p[4][2])
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
@inline function find_o_quickhull_parallel_nr_ex_v2_mask(points, exportFile)
    arranged_points = find_o_quickhull_parallel_nr_v2_mask(points)
    exportResult(arranged_points, exportFile)
end

points = create_discs(1000)
arranged_points1 = @time find_o_quickhull_parallel_nr_v2_mask(points)
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
