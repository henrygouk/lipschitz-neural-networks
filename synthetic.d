#!/usr/bin/env dub
/+ dub.sdl:
dependency "dopt" version="==0.3.16"
dependency "progress-d" version="~>1.0.0"
+/

module synthetic;

import dopt.core;
import dopt.nnet;
import dopt.online;

Layer maybeBatchNorm(Layer input, bool bn, float lambda)
{
    if(!bn)
    {
        return input;
    }
    else
    {
        return input.batchNorm(new BatchNormOptions().lipschitz(lambda));
    }
}

void main(string[] args)
{
    import std.algorithm : joiner, map;
	import std.array : array;
	import std.format : format;
    import std.getopt : getopt, config;
    import std.math : sin, cos;
    import std.random : randomShuffle, rndGen, uniform;
	import std.range : iota, zip, chunks;
	import std.stdio : File, stderr, stdout, write, writeln;
    
    float norm = float.nan;
    float lambda = float.infinity;
    float spectralDecay = 0.0f;
    bool batchnorm = false;
    size_t width = 1_000;
    
    getopt(
        args,
        "norm", &norm,
        "lambda", &lambda,
        "spectral-decay", &spectralDecay,
        "batchnorm", &batchnorm,
        "width", &width,
    );

    stderr.writeln("Generating data...");

    rndGen.seed(42);

    float[] x = iota(0, 1000)
               .map!(x => uniform(-5.0f, 5.0f))
               .array();
    
    float[] y = x
               .map!(x => sin(x) + cos(19.0f * x) / 5.0f)
               .array();

    stderr.writeln("Constructing operation graph");

    size_t batchSize = 100;
    auto features = float32([batchSize, 1]);
    auto labels = float32([batchSize, 1]);

    auto preds = dataSource(features)
                .dense(width, new DenseOptions().weightProj(projMatrix(float32Constant(lambda), norm)))
                .maybeBatchNorm(batchnorm, lambda)
                .relu()
                .dense(width, new DenseOptions().weightProj(projMatrix(float32Constant(lambda), norm)))
                .maybeBatchNorm(batchnorm, lambda)
                .relu()
                .dense(labels.shape[1], new DenseOptions().weightProj(projMatrix(float32Constant(lambda), norm)));

    auto network = new DAGNetwork([features], [preds]);
    auto lossSym = squaredError(preds.trainOutput, labels) + network.paramLoss;

    stderr.writeln("Creating optimiser...");
	auto learningRate = float32([], [0.001f]);
    auto updater = adam([lossSym, preds.trainOutput], network.params, network.paramProj, learningRate);
    auto testPlan = compile([preds.output]);

    stderr.writeln("Training...");

	foreach(e; 0 .. 5_000)
	{
        randomShuffle(zip(x, y));

        if(e == 3_000 && batchnorm)
        {
            learningRate.value.set([0.0001f]);
        }

        float epochLoss = 0.0f;
        float count = 0.0f;

        foreach(batch; zip(x.chunks(batchSize), y.chunks(batchSize)))
        {
            auto res = updater([
				features: buffer(batch[0].array()),
				labels: buffer(batch[1].array())
			]);

            epochLoss += res[0].get!float[0];
            count++;
        }

        if(e % 500 == 499)
        {
            stderr.writeln(epochLoss / count);
        }
    }

    for(size_t i = 0; i < x.length; i++)
    {
        writeln(x[i], ",", y[i]);
    }

    auto testX = iota(-5.0f, 5.0f, 0.01f).array();

    foreach(batch; testX.chunks(100))
    {
        auto res = testPlan.execute([
            features: buffer(batch.array())
        ]);

        auto testY = res[0].get!float;

        for(size_t i = 0; i < batch.length; i++)
        {
            writeln(batch[i], ",", testY[i]);
        }
    }
}