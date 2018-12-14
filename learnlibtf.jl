import TensorFlow
using Printf: @printf

function tftrain_logistic(x, y, weights)
	sess = TensorFlow.Session(TensorFlow.Graph())

	num_classes = 10
	m = size(x, 1)
	n = size(x, 2)

	yonehot = zeros(Float32, m, num_classes)
	for i in 1:m
		yonehot[i, y[i]] = 1.0
		if y[i] > 1
			yonehot[i, y[i]-1] = 1.0
		end
		if y[i] < num_classes
			yonehot[i, y[i]+1] = 1.0
		end
	end

	X = TensorFlow.constant(x)
	Y = TensorFlow.constant(yonehot)
	WTS = TensorFlow.constant(weights[:,:])

	W = TensorFlow.get_variable("W", [n, num_classes], Float32)
	b = TensorFlow.get_variable("b", [1, num_classes], Float32)

	pred = X*W .+ b

	cost = TensorFlow.reduce_sum( TensorFlow.nn.sigmoid_cross_entropy_with_logits(logits=pred, targets=Y) .* WTS) / sum(weights)

	optimizer = TensorFlow.train.GradientDescentOptimizer(0.1)
	minimize_op = TensorFlow.train.minimize(optimizer, cost)

	# Run training
	TensorFlow.run(sess, TensorFlow.global_variables_initializer())
	for epoch in 1:20000
		cur_loss, _ = TensorFlow.run(sess, [cost, minimize_op])
		if epoch % 2000 == 0
			@printf("Current loss is %.2f\n", cur_loss)
		end
	end
	cur_loss, W, b = TensorFlow.run(sess, [cost, W, b])
	@printf("Final loss is %.2f\n", cur_loss)
	W, b
end

function tftrain_logistic_country(x, y, weights)
	sess = TensorFlow.Session(TensorFlow.Graph())

	num_classes = 10
	m = size(x, 1)
	n = size(x, 2)

	X = TensorFlow.constant(x)
	Y = TensorFlow.constant(y)
	WTS = TensorFlow.constant(weights)

	W = TensorFlow.get_variable("W", [n, num_classes], Float32)
	b = TensorFlow.get_variable("b", [1, num_classes], Float32)

	pred = X*W .+ b

	cost = TensorFlow.reduce_sum( TensorFlow.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=Y) .* WTS) / sum(weights)

	optimizer = TensorFlow.train.GradientDescentOptimizer(0.1)
	minimize_op = TensorFlow.train.minimize(optimizer, cost)

	# Run training
	TensorFlow.run(sess, TensorFlow.global_variables_initializer())
	for epoch in 1:20000
		cur_loss, _ = TensorFlow.run(sess, [cost, minimize_op])
		if epoch % 2000 == 0
			@printf("Current loss is %.2f\n", cur_loss)
		end
	end
	cur_loss, W, b = TensorFlow.run(sess, [cost, W, b])
	@printf("Final loss is %.2f\n", cur_loss)
	W, b
end

function tftrain_nn(x, y, weights)
	sess = TensorFlow.Session(TensorFlow.Graph())

	m = size(x, 1)
	n = size(x, 2)

	X = TensorFlow.constant(x)
	Y = TensorFlow.constant(y[:,:])
	WTS = TensorFlow.constant(weights[:,:])

	num_hidden = 300
	W1 = TensorFlow.get_variable("W1", [n, num_hidden], Float32)
	b1 = TensorFlow.get_variable("b1", [1, num_hidden], Float32)
	W2 = TensorFlow.get_variable("W2", [num_hidden, 1], Float32)
	b2 = TensorFlow.get_variable("b2", [1, 1], Float32)

	layer_1 = TensorFlow.nn.dropout(TensorFlow.nn.relu(X*W1 .+ b1), 0.5f0)
	pred = layer_1*W2 .+ b2

	cost = TensorFlow.reduce_sum( ((pred - Y).^2) .* WTS) / sum(weights)

	optimizer = TensorFlow.train.AdamOptimizer(0.001)
	minimize_op = TensorFlow.train.minimize(optimizer, cost)

	# Run training
	TensorFlow.run(sess, TensorFlow.global_variables_initializer())
	for epoch in 1:10000
		cur_loss, _ = TensorFlow.run(sess, [cost, minimize_op])
		if epoch % 1000 == 0
			@printf("Current loss is %.2f\n", cur_loss)
		end
	end
	cur_loss, tf_W1, tf_b1, tf_W2, tf_b2 = TensorFlow.run(sess, [cost, W1, b1, W2, b2])
	@printf("Final loss is %.2f\n", cur_loss)
	tf_W1, tf_b1, tf_W2, tf_b2
end

function tfeval_nn(x, W1, b1, W2, b2)
	layer_1 = max.(0.0, x*W1 .+ b1)
	convert(Array{Float64}, layer_1*W2 .+ b2)
end

function tftrain_nnclass(x, y, weights)
	sess = TensorFlow.Session(TensorFlow.Graph())

	num_classes = 10
	m = size(x, 1)
	n = size(x, 2)

	yonehot = zeros(Float32, m, num_classes)
	for i in 1:m
		yonehot[i, y[i]] = 1.0
		if y[i] > 1
			yonehot[i, y[i]-1] = 1.0
		end
		if y[i] < num_classes
			yonehot[i, y[i]+1] = 1.0
		end
	end

	X = TensorFlow.constant(x)
	Y = TensorFlow.constant(yonehot)
	WTS = TensorFlow.constant(weights[:,:])

	num_hidden = 100
	W1 = TensorFlow.get_variable("W1", [n, num_hidden], Float32)
	b1 = TensorFlow.get_variable("b1", [1, num_hidden], Float32)
	W2 = TensorFlow.get_variable("W2", [num_hidden, num_classes], Float32)
	b2 = TensorFlow.get_variable("b2", [1, num_classes], Float32)

	layer_1 = TensorFlow.nn.dropout(TensorFlow.nn.relu(X*W1 .+ b1), 0.5f0)
	pred = layer_1*W2 .+ b2

	cost = TensorFlow.reduce_sum( TensorFlow.nn.sigmoid_cross_entropy_with_logits(logits=pred, targets=Y) .* WTS) / sum(weights)

	optimizer = TensorFlow.train.AdamOptimizer(0.0001)
	minimize_op = TensorFlow.train.minimize(optimizer, cost)

	# Run training
	TensorFlow.run(sess, TensorFlow.global_variables_initializer())
	for epoch in 1:3000
		cur_loss, _ = TensorFlow.run(sess, [cost, minimize_op])
		if epoch % 300 == 0
			@printf("Current loss is %.2f\n", cur_loss)
		end
	end
	cur_loss, tf_W1, tf_b1, tf_W2, tf_b2 = TensorFlow.run(sess, [cost, W1, b1, W2, b2])
	@printf("Final loss is %.2f\n", cur_loss)
	tf_W1, tf_b1, tf_W2, tf_b2
end

function tftrain_nn_country(x, y, weights)
	sess = TensorFlow.Session(TensorFlow.Graph())

	num_classes = 10
	m = size(x, 1)
	n = size(x, 2)

	X = TensorFlow.constant(x)
	Y = TensorFlow.constant(y)
	WTS = TensorFlow.constant(weights)

	num_hidden = 100
	W1 = TensorFlow.get_variable("W1", [n, num_hidden], Float32)
	b1 = TensorFlow.get_variable("b1", [1, num_hidden], Float32)
	W2 = TensorFlow.get_variable("W2", [num_hidden, num_classes], Float32)
	b2 = TensorFlow.get_variable("b2", [1, num_classes], Float32)

	layer_1 = TensorFlow.nn.dropout(TensorFlow.nn.relu(X*W1 .+ b1), 0.5f0)
	pred = layer_1*W2 .+ b2

	cost = TensorFlow.reduce_sum( TensorFlow.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=Y) .* WTS) / sum(weights)

	optimizer = TensorFlow.train.AdamOptimizer(0.0001)
	minimize_op = TensorFlow.train.minimize(optimizer, cost)

	# Run training
	TensorFlow.run(sess, TensorFlow.global_variables_initializer())
	for epoch in 1:3000
		cur_loss, _ = TensorFlow.run(sess, [cost, minimize_op])
		if epoch % 300 == 0
			@printf("Current loss is %.2f\n", cur_loss)
		end
	end
	cur_loss, tf_W1, tf_b1, tf_W2, tf_b2 = TensorFlow.run(sess, [cost, W1, b1, W2, b2])
	@printf("Final loss is %.2f\n", cur_loss)
	tf_W1, tf_b1, tf_W2, tf_b2
end

function tfeval_logistic(x, W, b)
	[a[2] for a in argmax(x*W .+ b, dims=2)]
end

function tfeval_nnclass(x, W1, b1, W2, b2)
	layer_1 = max.(0.0, x*W1 .+ b1)
	[a[2] for a in argmax(layer_1*W2 .+ b2, dims=2)]
end
