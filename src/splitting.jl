const F_NODE = 0
const C_NODE = 1
const U_NODE = 2

struct RS
end

function remove_diag!(a)
	n = size(a, 1)
	for i = 1:n
		for j in nzrange(a, i)
			if a.rowval[j] == i
				a.nzval[j] = 0
			end
   		end
	end
	dropzeros!(a)
end

function (::RS)(S)
	remove_diag!(S)
	RS_CF_splitting(S, adjoint(S))
end

function RS_CF_splitting(S::SparseMatrixCSC, T::SparseMatrixCSC)

	n = size(S,1)

	lambda = zeros(Int, n)

	Tp = T.colptr
	Tj = T.rowval
	Sp = S.colptr
	Sj = S.rowval

	interval_ptr = zeros(Int, n+1)
	interval_count = zeros(Int, n+1)
	index_to_node = zeros(Int,n)
	node_to_index = zeros(Int,n)

    for i = 1:n
		# compute lambda[i] - the number of nodes strongly coupled to node i
		lambda[i] = Sp[i+1] - Sp[i]
        interval_count[lambda[i] + 1] += 1
    end
	
	# initial interval_ptr
	@views accumulate!(+, interval_ptr[2:end], interval_count[1:end-1])

	# sort the nodes by the number of nodes strongly coupled to them:
	#   `index_to_node[idx]` is the node with `idx`th least number of nodes coupled to it
	#   `node_to_index[idx]` is the position of the `idx`th node in `index_to_node`
	# linear time and allocation-free equivalent to:
	#   sortperm!(index_to_node, lambda)
	#   node_to_index[index_to_node] .= 1:n
	interval_count .= 0 # temporarily zeroed, goes back to its original at end of loop
    for i = 1:n
        lambda_i = lambda[i] + 1
        interval_count[lambda_i] += 1
        index = interval_ptr[lambda_i] + interval_count[lambda_i]
        index_to_node[index] = i
        node_to_index[i]     = index
    end
	splitting = fill(U_NODE, n)

    # all nodes which no other nodes are strongly coupled to become F nodes
    for i = 1:n
		if lambda[i] == 0
            splitting[i] = F_NODE
		end
    end

	# i = index_to_node[top_index] can either refer to an F node or to the U node with the
	#	highest lambda[i].

	# index_to_node[interval_ptr[i]+1 : interval_ptr[i+1]] includes the set of U nodes with 
	#	i-1 nodes strongly coupled to them, and other "inactive" F and C nodes.
	
	# C nodes are always in index_to_node[top_index:n]. So at the end of the last 
	#	non-empty interval_ptr[i]+1 : interval_ptr[i+1] will be all the C nodes together 
	#	with some other inactive F nodes.
	
	# when lambda_k+1 > lambda_i, i.e. lambda_k == lambda_i,  where lambda_k+1 = lambda_i+1 
	#	is the new highest lambda[i], the line: `interval_ptr[lambda_k+1] = new_pos - 1`
	#	pushes the all the inactive C and F points to the end of the next now-non-empty 
	#	interval.
	for top_index = n:-1:1
		i = index_to_node[top_index]
		lambda_i = lambda[i] + 1
		interval_count[lambda_i] -= 1

		splitting[i] == F_NODE && continue

		@assert splitting[i] == U_NODE
		splitting[i] = C_NODE
		for j in nzrange(S, i)
			row = S.rowval[j]
			if splitting[row] == U_NODE
				splitting[row] = F_NODE

				# increment lambda for all U nodes that node `row` is strongly coupled to
				for k in nzrange(T, row)
					rowk = T.rowval[k]

					if splitting[rowk] == U_NODE
						# to ensure `intervalcount` is inbounds
						lambda[rowk] >= n - 1 && continue

						# move rowk to the end of its current interval
						lambda_k = lambda[rowk] + 1
						old_pos  = node_to_index[rowk]
						new_pos  = interval_ptr[lambda_k] + interval_count[lambda_k]

						swap_node = index_to_node[new_pos]
						(index_to_node[old_pos], index_to_node[new_pos]) = (swap_node, rowk)
						node_to_index[rowk] = new_pos
						node_to_index[swap_node] = old_pos

						# increment lambda[rowk]
						lambda[rowk] += 1

						# update intervals
						interval_count[lambda_k]   -= 1
						interval_count[lambda_k+1] += 1
						interval_ptr[lambda_k+1]    = new_pos - 1
					end
				end
			end
		end

		# decrement lambda for all U nodes that node i is strongly coupled to
		for j in nzrange(T, i)
			row = T.rowval[j]
			if splitting[row] == U_NODE
				# to ensure `intervalcount` is inbounds
				lambda[row] == 0 && continue

				# move row to the beginning of its current interval
				lambda_j = lambda[row] + 1
				old_pos  = node_to_index[row]
				new_pos  = interval_ptr[lambda_j] + 1

				swap_node = index_to_node[new_pos]
				(index_to_node[old_pos], index_to_node[new_pos]) = (swap_node, row)
				node_to_index[row] = new_pos
				node_to_index[swap_node] = old_pos

				# decrement lambda[row]
				lambda[row] -= 1

				# update intervals
				interval_count[lambda_j]   -= 1
				interval_count[lambda_j-1] += 1
				interval_ptr[lambda_j]     += 1
			end
		end
	end
	splitting
end
