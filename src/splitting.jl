const F_NODE = 0
const C_NODE = 1
const U_NODE = 2

struct RS
end
#=function split_nodes(::RS, S)
	n = size(S, 1)
	for i = 1:n
		for j in nzrange(S, i)
			row = S.rowval[j]
			if row == i
				S.nzval[j] = 0
			end
		end
	end
	i, j, v = findnz(S)
	RS_CF_splitting(sparse(i,j,v,n,n), sparse(j,i,v,n,n))
end=#
function split_nodes(::RS, S)
	T = S'
	RS_CF_splitting(S - spdiagm(diag(S)), T - spdiagm(diag(T)))
end

function RS_CF_splitting(S::SparseMatrixCSC, T::SparseMatrixCSC)

	m,n = size(S)

	n_nodes = n
	lambda = zeros(Int, n)

	Tp = T.colptr
	Tj = T.rowval
	Sp = S.colptr
	Sj = S.rowval

    # compute lambdas - number of neighbors
    for i = 1:n
        lambda[i] = Sp[i+1] - Sp[i]
    end

	interval_ptr = zeros(Int, n+1)
	interval_count = zeros(Int, n+1)
	index_to_node = zeros(Int,n)
	node_to_index = zeros(Int,n)

	# Number of nodes with a certain neighbor count
    for i = 1:n
        interval_count[lambda[i] + 1] += 1
    end
	csum = 0
    for i = 1:n
        interval_ptr[i] = csum
        csum += interval_count[i]
        interval_count[i] = 0
    end
    for i = 1:n
        lambda_i = lambda[i] + 1
        index    = interval_ptr[lambda_i] + interval_count[lambda_i]
        index_to_node[index+1] = i
        node_to_index[i]     = index+1
        interval_count[lambda_i] += 1
    end
	splitting = fill(2, n)

    # all nodes with no neighbors become F nodes
    for i = 1:n
        # check if diagonal or check if no neighbors
        if lambda[i] == 0 || (lambda[i] == 1 && Tj[Tp[i]] == i)
            splitting[i] = F_NODE
		end
    end

	for top_index = n_nodes:-1:1
		i = index_to_node[top_index]
		lambda_i = lambda[i] + 1
		interval_count[lambda_i] -= 1
		if splitting[i] == F_NODE
			continue
		else
			@assert splitting[i] == U_NODE
			splitting[i] = C_NODE
			for j in nzrange(S, i)
				row = S.rowval[j]
				if splitting[row] == U_NODE
					splitting[row] = F_NODE

					for k in nzrange(T, row)
						rowk = T.rowval[k]

						if splitting[rowk] == U_NODE
							lambda[rowk] >= n_nodes - 1 && continue
							lambda_k = lambda[rowk] + 1
						  	old_pos  = node_to_index[rowk]
						  	new_pos  = interval_ptr[lambda_k] + interval_count[lambda_k]# - 1

						  	node_to_index[index_to_node[old_pos]] = new_pos
						  	node_to_index[index_to_node[new_pos]] = old_pos
						  	(index_to_node[old_pos], index_to_node[new_pos]) = (index_to_node[new_pos], index_to_node[old_pos])

						  	# update intervals
						  	interval_count[lambda_k]   -= 1
						  	interval_count[lambda_k + 1] += 1 # invalid write!
						  	interval_ptr[lambda_k + 1]    = new_pos - 1

						 	# increment lambda_k
						 	lambda[rowk] += 1
						end
					end
				end
			end

			for j in nzrange(T, i)
				row = T.rowval[j]
				if splitting[row] == U_NODE

					lambda[row] == 0 && continue

					# assert(lambda[j] > 0);//this would cause problems!

					# move j to the beginning of its current interval
					lambda_j = lambda[row] + 1
					old_pos  = node_to_index[row]
					new_pos  = interval_ptr[lambda_j] + 1

					node_to_index[index_to_node[old_pos]] = new_pos
					node_to_index[index_to_node[new_pos]] = old_pos
					(index_to_node[old_pos],index_to_node[new_pos]) = (index_to_node[new_pos],index_to_node[old_pos])

					# update intervals
					interval_count[lambda_j]   -= 1
					interval_count[lambda_j-1] += 1
					interval_ptr[lambda_j]     += 1
					interval_ptr[lambda_j-1]    = interval_ptr[lambda_j] - interval_count[lambda_j-1]

					# decrement lambda_j
					lambda[row] -= 1
				end
			end
		end
	end
	splitting
end
