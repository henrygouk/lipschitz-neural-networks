#!/usr/bin/env dub
/+ dub.sdl:
dependency "dopt" version="==0.3.17"
dependency "progress-d" version="~>1.0.0"
+/
module mnist;

import dopt.core;
import dopt.nnet;
import dopt.online;
import progress;

Layer maybeDropout(Layer l, bool drop, float prob)
{
    if(drop)
    {
        return l.dropout(prob);
    }
    else
    {
        return l;
    }
}

Layer maybeBatchNorm(Layer l, bool bn, BatchNormOptions opts = new BatchNormOptions())
{
    if(bn)
    {
        return l.batchNorm(opts);
    }
    else
    {
        return l;
    }
}

void main(string[] args)
{
	import std.algorithm : joiner;
	import std.array : array;
	import std.format : format;
    import std.getopt : getopt, config;
	import std.range : zip, chunks;
	import std.stdio : File, stderr, stdout, write, writeln;

    float norm = float.nan;
    float lambda = float.infinity;
    float spectralDecay = 0.0f;
    bool dropout = false;
    bool batchnorm = false;
    string modelpath = "/dev/null";
    string logpath = "/dev/null";
    string datapath;
    bool validation;

    getopt(
        args,
        "norm", &norm,
        "lambda", &lambda,
        "spectral-decay", &spectralDecay,
        "dropout", &dropout,
        "batchnorm", &batchnorm,
        "modelpath", &modelpath,
        "logpath", &logpath,
        "valid", &validation,
        config.required, "datapath", &datapath
    );

	writeln("Loading data...");
    auto data = loadMNIST(datapath, validation);

	writeln("Constructing network graph...");
	size_t batchSize;
    Operation features;
    Operation labels;

    Layer preds;
    size_t epochs;
    float[size_t] learningRateSchedule;

    auto denseOpts = new DenseOptions().spectralDecay(spectralDecay);
    auto convOpts1 = new Conv2DOptions().spectralDecay(spectralDecay);
    auto convOpts2 = new Conv2DOptions().spectralDecay(spectralDecay);
    auto bnOpts = new BatchNormOptions();

    if(lambda != float.infinity)
    {
        denseOpts.weightProj = projMatrix(float32Constant(lambda), norm);
        convOpts1.filterProj = projConvParams(float32Constant(lambda), [28, 28], [1, 1], [0, 0], norm);
        convOpts2.filterProj = projConvParams(float32Constant(lambda), [12, 12], [1, 1], [0, 0], norm);
        bnOpts.lipschitz = lambda;
    }

    batchSize = 100;
    features = float32([batchSize, 1, 28, 28]);
    labels = float32([batchSize, 10]);

    preds  = dataSource(features)
            .conv2D(64, [5, 5], convOpts1)
            .maybeBatchNorm(batchnorm, bnOpts)
            .relu()
            .maybeDropout(dropout, 0.3)
            .maxPool([2, 2])
            .conv2D(128, [5, 5], convOpts2)
            .maybeBatchNorm(batchnorm, bnOpts)
            .relu()
            .maybeDropout(dropout, 0.3)
            .maxPool([2, 2])
            .dense(128, denseOpts)
            .maybeBatchNorm(batchnorm, bnOpts)
            .relu()
            .maybeDropout(dropout, 0.5)
            .dense(10, denseOpts)
            .softmax();
    
    epochs = 60;
    learningRateSchedule = [0: 0.0001f, 50: 0.00001f];
    
    auto network = new DAGNetwork([features], [preds]);

    auto lossSym = crossEntropy(preds.trainOutput, labels) + network.paramLoss;
	auto testLossSym = crossEntropy(preds.output, labels) + network.paramLoss;

	writeln("Creating optimiser...");
	auto learningRate = float32([], [0.0001f]);
	auto testPlan = compile([testLossSym, preds.output]);

    Updater updater = amsgrad([lossSym, preds.trainOutput], network.params, network.paramProj, learningRate);

	writeln("Training...");

	float[] fs = new float[features.volume];
	float[] ls = new float[labels.volume];

    auto logfile = File(logpath, "w");
    logfile.writeln("epoch,train_loss,train_acc,test_loss,test_acc");
	logfile.flush();

	foreach(e; 0 .. epochs)
	{
		float trainLoss = 0;
        float testLoss = 0;
        float trainAcc = 0;
        float testAcc = 0;
        float trainNum = 0;
        float testNum = 0;

		auto newlr = learningRateSchedule.get(e, float.nan);

        import std.math : isNaN;

        if(!isNaN(newlr))
        {
            learningRate.value.set([newlr]);
        }

		data.train.restart();
		data.test.restart();

		auto trainProgress = new Progress(data.train.length / batchSize);

		while(!data.train.finished())
		{
			data.train.getBatch([fs, ls]);

			auto res = updater([
				features: buffer(fs),
				labels: buffer(ls)
			]);

			trainLoss += res[0].get!float[0] * batchSize;
			trainAcc += computeAccuracy(ls, res[1].get!float);
			trainNum += batchSize;

			float loss = trainLoss / trainNum;
			float acc = trainAcc / trainNum;

			trainProgress.title = format("Epoch: %03d  Loss: %02.4f  Acc: %.4f", e + 1, loss, acc);
            trainProgress.next();
		}

		writeln();

		auto testProgress = new Progress(data.test.length / batchSize);

		while(!data.test.finished())
		{
			data.test.getBatch([fs, ls]);

			auto res = testPlan.execute([
				features: buffer(fs),
				labels: buffer(ls)
			]);

			testLoss += res[0].get!float[0] * batchSize;
			testAcc += computeAccuracy(ls, res[1].get!float);
			testNum += batchSize;

			float loss = testLoss / testNum;
			float acc = testAcc / testNum;

			testProgress.title = format("            Loss: %02.4f  Acc: %.4f", loss, acc);
            testProgress.next();
		}

        logfile.writeln(e + 1, ",", trainLoss / trainNum, ",", trainAcc / trainNum, ",", testLoss / testNum, ",",
			testAcc / testNum);
		logfile.flush();

		writeln();
		writeln();
	}

    logfile.close();

    network.save(modelpath);
}

float computeAccuracy(float[] ls, float[] preds)
{
	import std.algorithm : maxElement, maxIndex;
	import std.range : chunks, zip;

	float correct = 0;

	foreach(p, t; zip(preds.chunks(10), ls.chunks(10)))
	{
		if(p.maxIndex == t.maxIndex && t.maxElement() == 1.0f)
		{
			correct++;
		}
	}

	return correct;
}
