include("learnlib.jl")
include("learnlibtf.jl")

function run_rank_tf_logit()
	println("Training rank tf logit")
	tf_W, tf_b = tftrain_logistic(trainX, Array{Int32}(ratingstoranks(trainY)), weights2)

	trainYpred = [a[2] for a in argmax(trainX*tf_W .+ tf_b, dims=2)]
	testYpred = [a[2] for a in argmax(testX*tf_W .+ tf_b, dims=2)]
	evaluateclassifier("rank_tf_logit_train", ratingstoranks(trainY), trainYpred)
	evaluateclassifier("rank_tf_logit_test", ratingstoranks(testY), testYpred)
	Dict(
		 "W"=> tf_W,
		 "b"=> tf_b,
		 )
end

function run_country_tf_logit()
	println("Training country tf logit")
	tf_W, tf_b = tftrain_logistic_country(trainXc, trainTc, weightscountry)

	trainTcpred = [a[2] for a in argmax(trainXc*tf_W .+ tf_b, dims=2)]
	testTcpred = [a[2] for a in argmax(testXc*tf_W .+ tf_b, dims=2)]
	evaluatecountryclassifier("country_tf_logit_train", trainTc, trainTcpred)
	evaluatecountryclassifier("country_tf_logit_test", testTc, testTcpred)
	Dict(
		 "W"=> tf_W,
		 "b"=> tf_b,
		 )
end

function run_rank_tf_nnclass()
	println("Training rank tf nnclass")
	tf_W1, tf_b1, tf_W2, tf_b2 = tftrain_nnclass(trainX, Array{Int32}(ratingstoranks(trainY)), weights2)

	trainYpred = tfeval_nnclass(trainX, tf_W1, tf_b1, tf_W2, tf_b2)
	testYpred = tfeval_nnclass(testX, tf_W1, tf_b1, tf_W2, tf_b2)
	evaluateclassifier("rank_tf_nnclass_train", ratingstoranks(trainY), trainYpred)
	evaluateclassifier("rank_tf_nnclass_test", ratingstoranks(testY), testYpred)
	Dict(
		 "W1"=> tf_W1,
		 "b1"=> tf_b1,
		 "W2"=> tf_W2,
		 "b2"=> tf_b2,
		 )
end

function run_country_tf_nnclass()
	println("Training country tf nnclass")
	tf_W1, tf_b1, tf_W2, tf_b2 = tftrain_nn_country(trainXc, trainTc, weightscountry)

	trainTcpred = tfeval_nnclass(trainXc, tf_W1, tf_b1, tf_W2, tf_b2)
	testTcpred = tfeval_nnclass(testXc, tf_W1, tf_b1, tf_W2, tf_b2)
	evaluatecountryclassifier("country_tf_nnlass_train", trainTc, trainTcpred)
	evaluatecountryclassifier("country_tf_nnclass_test", testTc, testTcpred)
	Dict(
		 "W1"=> tf_W1,
		 "b1"=> tf_b1,
		 "W2"=> tf_W2,
		 "b2"=> tf_b2,
		 )
end

if length(ARGS) != 1
	println("Invalid args")
	exit()
end

println("Loading data")
objs = JLD.load(@sprintf("cs229_project/data2-tokens-big-2gram-cv-%s.jld", ARGS[1]));
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

# Uncomment to train rank classifier instead of country.
#tf_nnclass = run_rank_tf_nnclass()
#tf_logit = run_rank_tf_logit()
tf_nnclass = run_country_tf_nnclass()
tf_logit = run_country_tf_logit()
JLD.save(@sprintf("cs229_project/data2-tokens-big-2gram-cv-%s-tf-country.jld", ARGS[1]),
		 "tf_nnclass", tf_nnclass,
		 "tf_logit", tf_logit)
