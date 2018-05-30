#!/usr/bin/env dub
/+ dub.sdl:
dependency "dopt" version="~>0.3.12"
dependency "progress-d" version="~>1.0.0"
+/
module cifar10;

import dopt.core;
import dopt.nnet;
import dopt.online;
import progress;

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
    string arch;
    string datapath;

    getopt(
        args,
        "norm", &norm,
        "lambda", &lambda,
        "spectral-decay", &spectralDecay,
        "dropout", &dropout,
        "batchnorm", &batchnorm,
        "modelpath", &modelpath,
        "logpath", &logpath,
        config.required, "arch", &arch,
        config.required, "datapath", &datapath
    );

	writeln("Loading data...");
    auto data = loadCIFAR10(datapath);
	data.train = new ImageTransformer(data.train, 4, 4, true, false);

	writeln("Constructing network graph...");
	size_t batchSize = 100;
    auto features = float32([batchSize, 3, 32, 32]);
    auto labels = float32([batchSize, 10]);

    Layer preds;
    size_t epochs;
    float[size_t] learningRateSchedule;

    if(arch == "vgg")
    {
        auto opts = new VGGOptions();
        opts.lipschitzNorm = norm;
        opts.maxNorm = lambda;
        opts.spectralDecay = spectralDecay;
        opts.dropout = dropout;
        opts.batchnorm = batchnorm;

        preds  = vgg19(features, [512, 512], opts)
                .dense(10)
                .softmax();
        
        epochs = 140;
        learningRateSchedule = [0: 0.0001f, 100: 0.00001f, 120: 0.000001f];
    }
    else if(arch == "wrn")
    {
        auto opts = new WRNOptions();
        opts.lipschitzNorm = norm;
        opts.maxNorm = lambda;
        opts.spectralDecay = spectralDecay;
        opts.dropout = dropout;

        preds = wideResNet(features, 16, 10, opts)
               .dense(10)
               .softmax();
        
        epochs = 200;
        learningRateSchedule = [0: 0.1f, 60: 0.02f, 120: 0.004f, 160: 0.0008f];
    }
    else
    {
        assert(0);
    }
    
    auto network = new DAGNetwork([features], [preds]);

    auto lossSym = crossEntropy(preds.trainOutput, labels) + network.paramLoss;
	auto testLossSym = crossEntropy(preds.output, labels) + network.paramLoss;

	writeln("Creating optimiser...");
	auto learningRate = float32([], [0.0001f]);
	auto testPlan = compile([testLossSym, preds.output]);

    Updater updater;
    
    if(arch == "vgg")
    {
        updater = amsgrad([lossSym, preds.trainOutput], network.params, network.paramProj, learningRate);
    }
    else if(arch == "wrn")
    {
        updater = sgd([lossSym, preds.trainOutput], network.params, network.paramProj, learningRate,
            float32Constant(0.9), true);
    }
    else
    {
        assert(0);
    }

	writeln("Training...");

	float[] fs = new float[features.volume];
	float[] ls = new float[labels.volume];

    auto logfile = File(logpath, "w");
    logfile.write("epoch,train_loss,train_acc,test_loss,test_acc");

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

        logfile.writeln(e + 1, ",", trainLoss, ",", trainAcc, ",", testLoss, ",", testAcc);

		writeln();
		writeln();
	}

    logfile.close();

    network.save(modelpath);
}

float computeAccuracy(float[] ls, float[] preds)
{
	import std.algorithm : maxIndex;
	import std.range : chunks, zip;

	float correct = 0;

	foreach(p, t; zip(preds.chunks(10), ls.chunks(10)))
	{
		if(p.maxIndex == t.maxIndex)
		{
			correct++;
		}
	}

	return correct;
}
