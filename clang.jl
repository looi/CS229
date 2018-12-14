import Clang, DelimitedFiles, GZip, JSON
using Clang.cindex: children, cu_type, cu_file, idx_create, name, spelling, tokenize, tu_cursor, tu_dispose, tu_parse, TranslationUnit_Flags, getCursorReferenced, CLCursor
using Printf: @printf, @sprintf

function rParse(node, depth, res, clangnodes, vnames, fnames)
	tot = 1
	#println(repeat(" ", depth), node, " ", [tk.text for tk in tokenize(node)])
	push!(clangnodes, repr(typeof(node)))
	for cursor in children(node)
		if cu_file(cursor) != tempcppname continue end
		tot += rParse(cursor, depth+1, res, clangnodes, vnames, fnames)
	end

	typ = repr(typeof(node))
	res[typ] = get(res, typ, 0) + 1
	res["depth_max"] = max(get(res, "depth_max", 0), depth)

	if isa(node, Clang.cindex.FunctionDecl)
		res["fn_sz_max"] = max(get(res, "fn_sz_max", 0), tot)
		push!(fnames, spelling(node))
	end
	if isa(node, Clang.cindex.VarDecl)
		push!(vnames, spelling(node))
	end
	if isa(node, Clang.cindex.FunctionDecl) || isa(node, Clang.cindex.VarDecl)
		nl = length(spelling(node))
		key = "len_" * typ * "_" * repr(nl)
		res[key] = get(res, key, 0) + 1
		key = "depth_" * typ * "_" * repr(depth)
		res[key] = get(res, key, 0) + 1
	end
	# Look for references to stuff outside the file.
	if isa(node, Clang.cindex.DeclRefExpr) || isa(node, Clang.cindex.TypeRef) || isa(node, Clang.cindex.TemplateRef)
		refnode = getCursorReferenced(node)
		if cu_file(refnode) != tempcppname
			dn = spelling(refnode)
			res["ref_"*dn] = get(res, "ref_"*dn, 0) + 1
		end
	end
	push!(clangnodes, "endblock")
	tot
end

function addRefsToVisited(node, visited)
	if node in visited
		return
	end
	push!(visited, node)
	if isa(node, Clang.cindex.DeclRefExpr) || isa(node, Clang.cindex.TypeRef)
		refnode = getCursorReferenced(node)
		if cu_file(refnode) == tempcppname
			addRefsToVisited(refnode, visited)
		end
	end
	for cursor in children(node)
		addRefsToVisited(cursor, visited)
	end
end

function getRes(source, index)
	open(tempcppname, "w") do f
		write(f, source)
	end
	# Avoid parse_header because it leaks memory.
	args = ["-x", "c++"]
	tu = tu_parse(index, tempcppname, args, length(args), C_NULL, 0, TranslationUnit_Flags.DetailedPreprocessingRecord)
	node = tu_cursor(tu)

	clangnodes = Array{String}([])
	vnames = Array{String}([])
	fnames = Array{String}([])
	tokens = [tk.text for tk in tokenize(node)]
	res = Dict{String, Int64}()
	total_nodes = rParse(node, 0, res, clangnodes, vnames, fnames)
	res["total_nodes"] = total_nodes
	res["total_length"] = length(source)
	tu_dispose(tu) # Prevent leaking memory.
	if total_nodes == 0
		println("FAIL")
		exit()
	end
	res, tokens, clangnodes, vnames, fnames
end

function readdata(f)
	GZip.open("cs229_project/data2.json.gz") do infile
		while true
			obj = try
				JSON.parse(infile)
			catch
				nothing
			end
			if obj == nothing
				break
			end
			f(obj)
		end
	end
end

#tempcppname = "cs229_project/temp.cpp"
#index = idx_create(0, 0)
#println(getRes("", index))
#exit()

if length(ARGS) != 2
	println("Invalid args")
	exit()
end

cur_file = parse(Int, ARGS[1])
tot_file = parse(Int, ARGS[2])
tempcppname = @sprintf("cs229_project/temp%d.cpp", cur_file)

tot_entries = 0
readdata(
function(obj)
	global tot_entries
	tot_entries += 1
end)

per_file = div(tot_entries, tot_file)
start_entry = 1 + (cur_file-1)*per_file
end_entry = cur_file == tot_file ? tot_entries : cur_file*per_file

@printf("Processing %d-%d of %d\n", start_entry, end_entry, tot_entries)

index = idx_create(0, 0)
total = 0
GZip.open(@sprintf("cs229_project/data2-clang-%d-of-%d-allnodes.json.gz", cur_file, tot_file), "w") do outfile
	readdata(
	function(obj)
		global total, start_entry, end_entry, tot_entries, index
		total += 1
		if total < start_entry || total > end_entry
			return
		end
		@printf("Processed %d/%d submissions\n", total, tot_entries)
		res, tokens, clangnodes, vnames, fnames = getRes(obj["source"], index)
		obj["clang"] = res
		obj["tokens"] = tokens
		obj["clangnodes"] = clangnodes
		obj["vnames"] = vnames
		obj["fnames"] = fnames
		JSON.print(outfile, obj)
		write(outfile, '\n')
	end)
end
