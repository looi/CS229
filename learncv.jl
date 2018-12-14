include("learnlib.jl")
include("learnlibtf.jl")

function run_country_gda(trainXc, trainTc)
	println("Training country gda")
	model = LinearDiscriminantAnalysis(priors=[0.1 for x in 1:10])
	fit!(model, trainXc, trainTc)
	model
end

function run_rank_gda(trainX, trainY)
	println("Training rank gda")
	model = LinearDiscriminantAnalysis(priors=[0.1 for x in 1:10])
	fit!(model, trainX, ratingstoranks(trainY))
	model
end

all_accuracies = Dict{String, Array{Array{Float64}}}(
	"country_tf_logit_train" => [],
	"country_tf_logit_test" => [],
	"country_tf_nnclass_train" => [],
	"country_tf_nnclass_test" => [],
	"country_gda_train" => [],
	"country_gda_test" => [],
	"rank_tf_logit_train" => [],
	"rank_tf_logit_test" => [],
	"rank_tf_nnclass_train" => [],
	"rank_tf_nnclass_test" => [],
	"rank_gda_train" => [],
	"rank_gda_test" => [],
	"rank_linear_train" => [],
	"rank_linear_test" => [],
)

for contestid in ALL_CONTESTS
	println("Loading data ", contestid)

	objs = JLD.load(@sprintf("cs229_project/data2-tokens-big-2gram-cv-%s.jld", contestid));
	trainX = objs["trainX"];
	trainXv1 = objs["trainXv1"];
	trainY = objs["trainY"];
	trainT = objs["trainT"];
	testX = objs["testX"];
	testXv1 = objs["testXv1"];
	testY = objs["testY"];
	testT = objs["testT"];
	weights = getweights(trainY);
	weights2 = getweights2(trainY);
	trainXc = trainX[trainT .!= -1, :];
	trainTc = trainT[trainT .!= -1];
	testXc = testX[testT .!= -1, :];
	testTc = testT[testT .!= -1];
	weightscountry = getweightscountry(trainTc)
	objs = nothing;
	@printf("Main set: %d train %d test\n", size(trainX, 1), size(testX, 1))
	@printf("Country set: %d train %d test\n", size(trainXc, 1), size(testXc, 1))

	country_tf = JLD.load(@sprintf("cs229_project/data2-tokens-big-2gram-cv-%s-tf-country.jld", contestid))
	rank_tf = JLD.load(@sprintf("cs229_project/data2-tokens-big-2gram-cv-%s-tf-rank.jld", contestid))

	push!(all_accuracies["country_tf_logit_train"],
		  evaluatecountryclassifier("country_tf_logit_train", trainTc,
		  tfeval_logistic(trainXc, country_tf["tf_logit"]["W"], country_tf["tf_logit"]["b"])))
	push!(all_accuracies["country_tf_logit_test"],
		  evaluatecountryclassifier("country_tf_logit_test", testTc,
		  tfeval_logistic(testXc, country_tf["tf_logit"]["W"], country_tf["tf_logit"]["b"])))

	push!(all_accuracies["country_tf_nnclass_train"], evaluatecountryclassifier("country_tf_nnclass_train", trainTc, tfeval_nnclass(trainXc, country_tf["tf_nnclass"]["W1"], country_tf["tf_nnclass"]["b1"], country_tf["tf_nnclass"]["W2"], country_tf["tf_nnclass"]["b2"])))
	push!(all_accuracies["country_tf_nnclass_test"], evaluatecountryclassifier("country_tf_nnclass_test", testTc, tfeval_nnclass(testXc, country_tf["tf_nnclass"]["W1"], country_tf["tf_nnclass"]["b1"], country_tf["tf_nnclass"]["W2"], country_tf["tf_nnclass"]["b2"])))

	country_gda = run_country_gda(trainXc, trainTc)
	push!(all_accuracies["country_gda_train"],
		  evaluatecountryclassifier("country_gda_train", trainTc,
									predict(country_gda, trainXc)))
	push!(all_accuracies["country_gda_test"],
		  evaluatecountryclassifier("country_gda_test", testTc,
									predict(country_gda, testXc)))

	push!(all_accuracies["rank_tf_logit_train"],
		  evaluateclassifier("rank_tf_logit_train", ratingstoranks(trainY),
		  tfeval_logistic(trainX, rank_tf["tf_logit"]["W"], rank_tf["tf_logit"]["b"])))
	push!(all_accuracies["rank_tf_logit_test"],
		  evaluateclassifier("rank_tf_logit_test", ratingstoranks(testY),
		  tfeval_logistic(testX, rank_tf["tf_logit"]["W"], rank_tf["tf_logit"]["b"])))

	push!(all_accuracies["rank_tf_nnclass_train"], evaluateclassifier("rank_tf_nnclass_train", ratingstoranks(trainY), tfeval_nnclass(trainX, rank_tf["tf_nnclass"]["W1"], rank_tf["tf_nnclass"]["b1"], rank_tf["tf_nnclass"]["W2"], rank_tf["tf_nnclass"]["b2"])))
	push!(all_accuracies["rank_tf_nnclass_test"], evaluateclassifier("rank_tf_nnclass_test", ratingstoranks(testY), tfeval_nnclass(testX, rank_tf["tf_nnclass"]["W1"], rank_tf["tf_nnclass"]["b1"], rank_tf["tf_nnclass"]["W2"], rank_tf["tf_nnclass"]["b2"])))

	rank_gda = run_rank_gda(trainX, trainY)
	push!(all_accuracies["rank_gda_train"], evaluateclassifier("rank_gda_train", ratingstoranks(trainY),
															   predict(rank_gda, trainX)))
	push!(all_accuracies["rank_gda_test"], evaluateclassifier("rank_gda_test", ratingstoranks(testY),
															   predict(rank_gda, testX)))

	println("Training rank linear")
	rank_linear = trainlinear_scikit(trainX, trainY, weights)
	push!(all_accuracies["rank_linear_train"],
		  evaluatelinear("rank_linear_train", trainY, predict(rank_linear, trainX), weights))
	weightstest = getweights(testY);
	push!(all_accuracies["rank_linear_test"],
		  evaluatelinear("rank_linear_test", testY, predict(rank_linear, testX), weightstest))
end

for key in sort(collect(keys(all_accuracies)))
	arr = all_accuracies[key]
	if length(arr) == 0
		continue
	end
	@printf("%s: %s %s\n", key, mean(arr), arr)
end
