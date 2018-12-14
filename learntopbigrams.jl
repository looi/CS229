include("learnlib.jl")

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

contestid = 1023
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
vocab = objs["vocab"];
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

#rank_tf = JLD.load(@sprintf("cs229_project/data2-tokens-big-2gram-cv-%s-tf-rank.jld", contestid))
#
#diff = rank_tf["tf_logit"]["W"][:,2]# - rank_tf["tf_logit"]["W"][:,9]
#diff = [(diff, vocab[idx], idx) for (idx, diff) in enumerate(diff)]
#sort!(diff, rev=true)
#println(diff[1:20])
#
#diff = rank_tf["tf_logit"]["W"][:,9]# - rank_tf["tf_logit"]["W"][:,2]
#diff = [(diff, vocab[idx], idx) for (idx, diff) in enumerate(diff)]
#sort!(diff, rev=true)
#println(diff[1:20])

#rank_gda = run_rank_gda(trainX, trainY)
country_gda = run_country_gda(trainXc, trainTc)

n = size(trainX, 2)

for i in 1:10
	diff = country_gda[:means_][i,:]
	diff = [(diff, vocab[idx], idx) for (idx, diff) in enumerate(diff)]
	sort!(diff, rev=true)
	println(COUNTRIES[i])
	println(diff[1:20])
	println()
end

#diff = rank_gda[:means_][2,:] - rank_gda[:means_][2,:]
#diff = [(diff, vocab[idx], idx) for (idx, diff) in enumerate(diff)]
#sort!(diff, rev=true)
#println(diff[1:20])
#
#diff = rank_gda[:means_][9,:] - rank_gda[:means_][2,:]
#diff = [(diff, vocab[idx], idx) for (idx, diff) in enumerate(diff)]
#sort!(diff, rev=true)
#println(diff[1:20])
