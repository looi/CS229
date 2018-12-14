include("learnlib.jl")

allX = Array{Array{String}}([]) # Words, heavily processed.
allXv1 = Array{Array{String}}([]) # Words, clang tokens only.
allY = Array{Float32}([]) # Ratings
allC = Array{Int32}([]) # Contests
allT = Array{Int32}([]) # Countries (for top 10, otherwise -1)

println("Reading data")
all_contests = Set{Int32}([])
readdata(
function(obj)
	x, y = tokensXYfromobj(obj)
	push!(allX, x)
	push!(allXv1, obj["tokens"])
	push!(allY, y)
	push!(allC, obj["contest"])
	country = findfirst(COUNTRIES .== obj["country"])
	if country == nothing
		country = -1
	end
	push!(allT, country)
	push!(all_contests, obj["contest"])
end)
all_contests = collect(all_contests)
sort!(all_contests)

function writeit(filename, trainIdx, testIdx)
	trainX = allX[trainIdx]
	testX = allX[testIdx]
	trainXv1 = allXv1[trainIdx]
	testXv1 = allXv1[testIdx]
	@printf("Read %d training %d test\n", size(trainX, 1), size(testX, 1))

	#, ngram_range=(1,2)
	tfidf = TfidfVectorizer(lowercase=false, min_df=0.01, ngram_range=(1,2), use_idf=false, tokenizer = (x) -> x)
	trainX = fit_transform!(tfidf, trainX)
	testX = transform(tfidf, testX)
	vocab = [(idx[1], word) for (word, idx) in tfidf[:vocabulary_]]
	sort!(vocab)
	vocab = [x[2] for x in vocab]
	println("tfidf vocab ", length(vocab))
	tfidf = nothing

	counter = CountVectorizer(lowercase=false, min_df=0.01, tokenizer = (x) -> x)
	trainXv1 = fit_transform!(counter, trainXv1)
	testXv1 = transform(counter, testXv1)
	println("counter vocab ", length(counter[:vocabulary_]))
	counter = nothing

	trainX = convert(Array{Float32}, trainX[:toarray]())
	testX = convert(Array{Float32}, testX[:toarray]())
	trainXv1 = convert(Array{Float32}, trainXv1[:toarray]())
	testXv1 = convert(Array{Float32}, testXv1[:toarray]())

	println("Scaling")
	normalizer = StandardScaler()
	trainX = fit_transform!(normalizer, trainX);
	testX = transform(normalizer, testX);
	normalizer = nothing;
	println("Scaled")

	JLD.save(filename,
			 "vocab", vocab,
			 "trainX", trainX,
			 "trainXv1", trainXv1,
			 "trainY", allY[trainIdx],
			 "trainT", allT[trainIdx],
			 "testX", testX,
			 "testXv1", testXv1,
			 "testY", allY[testIdx],
			 "testT", allT[testIdx])
end

for contest_id in all_contests
	@printf("Processing %d\n", contest_id)
	writeit(@sprintf("cs229_project/data2-tokens-big-2gram-cv-%d.jld", contest_id), allC .!= contest_id, allC .== contest_id)
end
