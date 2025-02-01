using LinearAlgebra
# using CSV
# using DataFrames
function find_special_points1(points)
    maxY = maximum(points[:,2])
    minY = minimum(points[:,2])
    maxX = maximum(points[:,1])
    minX = minimum(points[:,1])

    rightPoints = points[findall(row -> row[1]==maxX,eachrow(points)),:]
    leftPoints = points[findall(row -> row[1]==minX,eachrow(points)),:]
    topPoints = points[findall(row -> row[2] == maxY,eachrow(points)),:]
    bottomPoints = points[findall(row -> row[2] == minY,eachrow(points)),:]

    if size(topPoints, 1) == 1
        top = [topPoints[1, :]]
    else
        idx = sortperm(topPoints[:, 1])
        topPoints = topPoints[idx,:]
        top = [topPoints[1, :], topPoints[end, :]]
    end
    
    if size(bottomPoints, 1) == 1
        bottom = [bottomPoints[1, :]]
    else
        idx = sortperm(bottomPoints[:,1],rev = true)
        bottomPoints = bottomPoints[idx,:]
        bottom = [bottomPoints[1, :], bottomPoints[end, :]]
    end
    
    if size(rightPoints, 1) == 1
        right = [rightPoints[1, :]]
    else
        idx = sortperm(rightPoints[:,2],rev = true)
        rightPoints = rightPoints[idx,:]
        right = [rightPoints[1, :], rightPoints[end, :]]
    end
    
    if size(leftPoints, 1) == 1
        left = [leftPoints[1, :]]
    else
        idx = sortperm(leftPoints[:,2])
        leftPoints = leftPoints[idx,:]
        left = [leftPoints[1, :], leftPoints[end, :]]
    end
    
    if length(top) == 1
        p2_1 = p2_2 = top[1]
    else
        p2_2 = top[1]
        p2_1 = top[2]
    end
    p1_2 = right[1]
    if length(right) == 1
        p1_1 = right[1]
    else
        p1_1 = right[2]
    end
    p4_2 = bottom[1]
    if length(bottom) == 1
        p4_1 = bottom[1]
    else
        p4_1 = bottom[2]
    end
    p3_2 = left[1]
    if length(left) == 1
        p3_1 = left[1]
    else
        p3_1 = left[2]
    end
    return p1_1,p1_2, p2_1,p2_2, p3_1,p3_2, p4_1,p4_2
end

function find_sets1(points,p1_1,p1_2, p2_1,p2_2, p3_1,p3_2, p4_1,p4_2)
    set1 = points[(points[:,1] .>= p2_1[1]) .& (points[:,2] .>=p1_2[2]),:]
    set2 = points[(points[:,1] .<= p2_2[1]) .& (points[:,2] .>=p3_1[2]),:]
    set3 = points[(points[:,1] .<= p4_1[1]) .& (points[:,2] .<=p3_2[2]),:]
    set4 = points[(points[:,1] .>= p4_2[1]) .& (points[:,2] .<=p1_1[2]),:]
    return set1,set2,set3,set4
end

function find_o_hull11(set1, p1_2,p2_1)
    if size(set1,1) == 0
        return zeros(0,2)
    end
    key1 = ((set1[:,1] .- p2_1[1]).^2) .+ ((set1[:,2] .- p1_2[2]).^2)
    maxset1 = maximum(key1)
    new_point1 = set1[findfirst(key1 .== maxset1),:]
    new_set11 = set1[set1[:,1] .> new_point1[1],:]
    new_set12 = set1[set1[:,2] .> new_point1[2],:]
    return vcat(find_o_hull11(new_set11,p1_2, new_point1),new_point1',find_o_hull11(new_set12,new_point1,p2_1))
end

function find_o_hull21(set2, p2_2,p3_1)
    if size(set2,1) == 0
        return zeros(0,2)
    end
    key2 = ((set2[:,1] .- p2_2[1]).^2) .+ ((set2[:,2] .- p3_1[2]).^2)
    maxset2 = maximum(key2)
    new_point2 = set2[findfirst(key2 .== maxset2),:]
    new_set21 = set2[set2[:,2] .> new_point2[2],:]
    new_set22 = set2[set2[:,1] .< new_point2[1],:]
    return vcat(find_o_hull21(new_set21,p2_2, new_point2),new_point2',find_o_hull21(new_set22,new_point2,p3_1))
end

function find_o_hull31(set3, p3_2,p4_1)
    if size(set3,1) == 0
        return zeros(0,2)
    end
    key3 = ((set3[:,1] .- p4_1[1]).^2) .+ ((set3[:,2] .- p3_2[2]).^2)
    maxset3 = maximum(key3)
    new_point3 = set3[findfirst(key3 .== maxset3),:]
    new_set31 = set3[set3[:,1] .< new_point3[1],:]
    new_set32 = set3[set3[:,2] .< new_point3[2],:]
    return vcat(find_o_hull31(new_set31,p3_2, new_point3),new_point3',find_o_hull31(new_set32,new_point3,p4_1))
end

function find_o_hull41(set4, p4_2,p1_1)
    if size(set4,1) == 0
        return zeros(0,2)
    end
    key4 = ((set4[:,1] .- p4_2[1]).^2) .+ ((set4[:,2] .- p1_1[2]).^2)
    maxset4 = maximum(key4)
    new_point4 = set4[findfirst(key4 .== maxset4),:]
    new_set41 = set4[set4[:,2] .< new_point4[2],:]
    new_set42 = set4[set4[:,1] .> new_point4[1],:]
    return vcat(find_o_hull41(new_set41,p4_2, new_point4),new_point4',find_o_hull41(new_set42,new_point4,p1_1))
end

function find_o_quickhull(points)
    p1_1,p1_2, p2_1,p2_2, p3_1,p3_2, p4_1,p4_2 = find_special_points1(points)
    set1,set2,set3,set4 = find_sets1(points,p1_1,p1_2, p2_1,p2_2, p3_1,p3_2, p4_1,p4_2)
    arranged_points = zeros(0,2)
    arranged_points = vcat(arranged_points, p1_2', find_o_hull11(set1,p1_2,p2_1), p2_1')
    arranged_points = vcat(arranged_points, p2_2', find_o_hull21(set2,p2_2,p3_1), p3_1')
    arranged_points = vcat(arranged_points, p3_2', find_o_hull31(set3,p3_2,p4_1), p4_1')
    arranged_points = vcat(arranged_points, p4_2', find_o_hull41(set4,p4_2,p1_1), p1_1')
    arranged_points = vcat(arranged_points,arranged_points[1,:]')
    return arranged_points
end
function find_o_quickhull_ex(points, exportFile)
    arranged_points = find_o_quickhull(points)
    exportResult(arranged_points, exportFile)
end

include("utils.jl")
points = create_discs(1000)

arranged_points = @time find_o_quickhull(points)
println()

# using Plots

# scatter(points[:,1],points[:,2],color=:red, markersize=2,legend=false)
# x = arranged_points[:,1]
# y = arranged_points[:,2]
# s =Vector{Tuple{Int, Vector{Float64}}}(undef,0)
# for i in 1:(size(arranged_points,1)-1)
#     if arranged_points[i+1,1] > arranged_points[i,1] && arranged_points[i+1,2] > arranged_points[i,2]
#         push!(s,(i,[arranged_points[i,1],arranged_points[i+1,2]]))
#     elseif arranged_points[i+1,1] > arranged_points[i,1] && arranged_points[i+1,2] < arranged_points[i,2]
#         push!(s,(i,[arranged_points[i+1,1],arranged_points[i,2]]))
#     elseif arranged_points[i+1,1] < arranged_points[i,1] && arranged_points[i+1,2] < arranged_points[i,2]
#         push!(s,(i,[arranged_points[i,1],arranged_points[i+1,2]]))
#     elseif arranged_points[i+1,1] < arranged_points[i,1] && arranged_points[i+1,2] > arranged_points[i,2]
#         push!(s,(i,[arranged_points[i+1,1],arranged_points[i,2]]))        
#     end
# end

# for i in 1:length(s) 
#     insert!(x,s[i][1]+i,s[i][2][1])
#     insert!(y,s[i][1]+i,s[i][2][2])
# end

# plot!(x,y,color=:green, markersize=8)
