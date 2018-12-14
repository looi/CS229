import GZip, JLD, JSON
using Base.Iterators: repeated
using LinearAlgebra: Diagonal
using Printf: @printf, @sprintf
using ScikitLearn: @sk_import, fit!, fit_transform!, predict, predict_log_proba, predict_proba, transform
using Statistics: mean, std

@sk_import decomposition: PCA
@sk_import discriminant_analysis: LinearDiscriminantAnalysis
@sk_import ensemble: RandomForestClassifier
@sk_import feature_extraction.text: (CountVectorizer, TfidfVectorizer)
@sk_import linear_model: (LinearRegression, LogisticRegression, Ridge)
@sk_import naive_bayes: MultinomialNB
@sk_import preprocessing: StandardScaler
@sk_import svm: SVC
@sk_import tree: (DecisionTreeClassifier, DecisionTreeRegressor)

ALL_CONTESTS = [1023, 1025, 1028, 1033, 1037, 1043, 1054, 1055, 1056, 1060]

COUNTRIES = ["in", "cn", "ru", "bd", "vn", "ua", "pl", "eg", "us", "ir"]
# Top 10 countries: Tuple{Int64,String}[(12331, "in"), (8818, "cn"), (6761, "ru"), (4536, "bd"), (1753, "vn"), (1694, "ua"), (1664, "pl"), (1662, "eg"), (1450, "us"), (1406, "ir")] total: 59912 sumtop: 42075 (70%)

function readdata(f)
	files = [@sprintf("cs229_project/data2-clang-%d-of-4-allnodes.json.gz", i) for i in 1:4]
	for file in files
		GZip.open(file) do infile
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
end

function printdatastats()
	countries = Dict{String, Int64}()
	ranks = Dict{String, Int64}()
	tottokens = 0
	totclangnodes = 0
	total = 0
	readdata(
	function(obj)
		country = obj["country"]
		countries[country] = get(countries, country, 0) + 1
		rank = obj["old_rank"]
		ranks[rank] = get(ranks, rank, 0) + 1
		tottokens += length(obj["tokens"])
		totclangnodes += length(obj["clangnodes"])
		total += 1
	end)
	countries = sort([(count, country) for (country, count) in countries], rev=true)[1:10]
	sumtop = sum(count for (count, country) in countries)
	println("Top 10 countries: ", countries, " total: ", total, " sumtop: ", sumtop)
	println("Ranks: ", ranks)
	println("Avg tokens: ", tottokens/total)
	println("Avg clangnodes: ", totclangnodes/total)
end

function tokensXYfromobj(obj)
	vnames = Set{String}(obj["vnames"])
	fnames = Set{String}(obj["fnames"])
	tokens = Array{String}(undef, length(obj["tokens"]))
	for (i, tok) in enumerate(obj["tokens"])
		if startswith(tok, "\"")
			tokens[i] = "!!STR"
		elseif startswith(tok, "'")
			tokens[i] = "!!CHR"
		elseif tok != "main" && tok in fnames
			tokens[i] = "!!FUN"
		elseif tok in vnames
			tokens[i] = "!!VAR"
		else
			tokens[i] = tok
		end
	end
	append!(tokens, obj["clangnodes"])
	tokens, obj["old_rating"]
end

function trainlinear_scikit(x, y, weights)
	model = LinearRegression(fit_intercept=true)
	fit!(model, x, y, weights)
	accuracy = sum(ratingstoranks(predict(model, x)) .== ratingstoranks(y)) / length(y)
	println("train accuracy: $accuracy")
	model
end

function trainpca_scikit(x, n_components)
	model = PCA(svd_solver="full", n_components=n_components)
	fit!(model, x)
	model
end

function ratingtorank(rating)
	if rating <= 1199
		return 1 # Newbie
	elseif rating <= 1399
		return 2 # Pupil
	elseif rating <= 1599
		return 3 # Specialist
	elseif rating <= 1899
		return 4 # Expert
	elseif rating <= 2099
		return 5 # Candidate Master
	elseif rating <= 2299
		return 6 # Master
	elseif rating <= 2399
		return 7 # International Master
	elseif rating <= 2599
		return 8 # Grandmaster
	elseif rating <= 2999
		return 9 # International Grandmaster
	else
		return 10 # Legendary Grandmaster
	end
end

function ratingstoranks(ratings)
	return [ratingtorank(rating) for rating in ratings]
end

function getweights(trainY)
	tot_rank = zeros(Float32, 10)
	trainR = ratingstoranks(trainY)
	for r in trainR
		tot_rank[r] += 1
	end
	m = length(trainR)
	[1000/tot_rank[trainR[i]] for i in 1:m]
end

function getweights2(trainY)
	tot_rank = zeros(Float32, 10)
	trainR = ratingstoranks(trainY)
	for r in trainR
		tot_rank[r] += 1
	end
	tot_rank[1] *= 1.5
	tot_rank[10] *= 1.5
	m = length(trainR)
	[1000/tot_rank[trainR[i]] for i in 1:m]
end

function getweightscountry(trainTc)
	tot_country = zeros(Float32, 10)
	for c in trainTc
		tot_country[c] += 1
	end
	m = length(trainTc)
	[1000/tot_country[c] for c in trainTc]
end

function evaluateclassifier(name, r, rpred)
	accuracy_off_by_one_by_rank = zeros(10)
	tot_rank = zeros(10)
	for i in 1:length(r)
		if abs(r[i] - rpred[i]) <= 1
			accuracy_off_by_one_by_rank[r[i]] += 1
		end
		tot_rank[r[i]] += 1
	end
	accuracy_off_by_one_by_rank ./= tot_rank
	@printf("\n%s\n", name)
	@printf("Overall proportions: %s\n", tot_rank / length(r))
	@printf("Correct rank: %.5f\n", mean(r .== rpred))
	@printf("Correct rank within one: %.5f\n", mean(abs.(r - rpred) .<= 1))
	@printf("Correct rank within one by rank: %.5f %s\n", mean(accuracy_off_by_one_by_rank), accuracy_off_by_one_by_rank)
	#mean(accuracy_off_by_one_by_rank)
	accuracy_off_by_one_by_rank
end

function evaluatecountryclassifier(name, r, rpred)
	accuracy_by_class = zeros(10)
	tot_rank = zeros(10)
	for i in 1:length(r)
		if r[i] == rpred[i]
			accuracy_by_class[r[i]] += 1
		end
		tot_rank[r[i]] += 1
	end
	accuracy_by_class ./= tot_rank
	@printf("\n%s\n", name)
	@printf("Overall proportions: %s\n", tot_rank / length(r))
	@printf("Correct country: %.5f\n", mean(r .== rpred))
	@printf("Correct country by country: %.5f %s\n", mean(accuracy_by_class), accuracy_by_class)
	#mean(accuracy_by_class)
	accuracy_by_class
end

function evaluatelinear(name, y, ypred, weights)
	r = ratingstoranks(y)
	rpred = ratingstoranks(ypred)
	ans = evaluateclassifier(name, r, rpred)
	rms = sqrt(mean((y - ypred).^2))
	weighted_rms = sqrt(sum(((y-ypred).^2).*weights) / sum(weights))
	@printf("RMS: %.1f\n", rms)
	@printf("Weighted RMS: %.1f\n", weighted_rms)
	ans
end
